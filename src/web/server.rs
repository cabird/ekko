//! Web server lifecycle: standalone (`ekko web`) and co-hosted (inside
//! `ekko mcp` when `web_serve` is on — PORT_PLAN phase 4).
//!
//! Both paths funnel into [`serve_with_lease`]: bind the port, hold the
//! [`WEB_LEASE`] with a heartbeat, serve axum, release on the way out. The
//! election order is bind-FIRST, lease-second, so a failed bind can never
//! strand a lease pointing at nothing. Multiple `ekko mcp` instances (every
//! Claude Code session starts one) race for the port once a minute; exactly
//! one wins, the rest stay quiet, and when the winner dies another takes
//! over within [`ELECTION_INTERVAL`].

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

use tokio::net::TcpListener;
use tokio::sync::watch;

use crate::config::get_config;
use crate::db::{Database, leases::WEB_LEASE};
use crate::mcp::EkkoServer;

/// How long a lease lives without a heartbeat; a dead host is replaced
/// within this bound.
const LEASE_TTL_SECS: i64 = 180;
/// Heartbeat / election cadence.
const ELECTION_INTERVAL: Duration = Duration::from_secs(60);

/// Where the UI should listen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WebOptions {
    pub port: u16,
    /// Try to open the page in the default browser once listening.
    pub open: bool,
}

/// Serve until interrupted. Returns the exit code for `main`.
pub async fn run(opts: WebOptions) -> i32 {
    let _ = dotenvy::dotenv();
    let config = get_config();
    let server = match EkkoServer::from_config(config) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ekko web: {e}");
            return 1;
        }
    };

    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), opts.port);
    let listener = match TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
            eprintln!(
                "ekko web: port {} is already in use (another ekko web, or a co-hosting \
`ekko mcp` with web_serve on? pass --port to pick another)",
                opts.port
            );
            return 1;
        }
        Err(e) => {
            eprintln!("ekko web: cannot bind {addr}: {e}");
            return 1;
        }
    };
    let url = format!("http://{addr}/");
    eprintln!("ekko web: serving {url}  (Ctrl-C to stop)");
    if opts.open {
        open_browser(&url);
    }

    // Ctrl-C → flip the same watch channel the co-hosted path uses.
    let (stop_tx, stop_rx) = watch::channel(false);
    tokio::spawn(async move {
        shutdown_signal().await;
        let _ = stop_tx.send(true);
    });

    match serve_with_lease(server, listener, opts.port, stop_rx).await {
        ServeOutcome::Shutdown => 0,
        ServeOutcome::LeaseLost => {
            // Port ownership makes this near-impossible (the lease can only
            // be taken from a live holder by itself); losing it means the
            // clock or the database did something exotic.
            eprintln!("ekko web: hosting lease lost; exiting");
            1
        }
        ServeOutcome::Error(e) => {
            eprintln!("ekko web: server error: {e}");
            1
        }
    }
}

/// Co-hosted election loop, spawned by `ekko mcp` when `web_serve` is on.
/// Runs until `shutdown` flips; never returns an error (hosting is a
/// best-effort side job of the MCP server — failures log and retry).
pub async fn run_cohosted(server: EkkoServer, port: u16, mut shutdown: watch::Receiver<bool>) {
    tracing::info!(port, "web co-hosting election loop started");
    loop {
        if *shutdown.borrow() {
            break;
        }
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
        match TcpListener::bind(addr).await {
            Ok(listener) => {
                tracing::info!(url = %format!("http://{addr}/"), "web UI co-hosted by this instance");
                match serve_with_lease(server.clone(), listener, port, shutdown.clone()).await {
                    ServeOutcome::Shutdown => break,
                    ServeOutcome::LeaseLost => {
                        tracing::warn!("web hosting lease lost; standing down");
                    }
                    ServeOutcome::Error(e) => {
                        tracing::warn!(error = %e, "web server error; will retry");
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
                // Someone else (another instance, or a standalone `ekko
                // web`) is serving. Stay quiet and check again later.
                tracing::debug!(port, "web port in use; not this instance's turn");
            }
            Err(e) => {
                tracing::warn!(error = %e, "web bind failed; will retry");
            }
        }
        tokio::select! {
            _ = tokio::time::sleep(ELECTION_INTERVAL) => {}
            _ = shutdown.changed() => break,
        }
    }
    tracing::info!("web co-hosting election loop stopped");
}

enum ServeOutcome {
    Shutdown,
    LeaseLost,
    Error(String),
}

/// The shared hosting core: take the lease (port is already bound — the
/// caller's job), heartbeat it, serve, release. Returns why it stopped.
async fn serve_with_lease(
    server: EkkoServer,
    listener: TcpListener,
    port: u16,
    mut shutdown: watch::Receiver<bool>,
) -> ServeOutcome {
    let db = server.db().clone();
    let holder = format!("{}:{port}", crate::background::instance_id());

    // We own the port, so any live lease belongs to a holder that lost its
    // listener (crashed standalone on another port, clock skew). Acquire is
    // also renewal, so a retry loop is unnecessary: fail-fast and let the
    // caller's cadence retry.
    match acquire(&db, &holder).await {
        Ok(true) => {}
        Ok(false) => {
            // We own the port, so the lease's holder is not serving it —
            // typically a host that died without releasing. Owning the
            // socket is the authoritative fact; take the lease over.
            let stale = db.lease_info(WEB_LEASE).ok().flatten();
            tracing::warn!(?stale, "port bound but lease held elsewhere; taking it over");
            let db2 = db.clone();
            let _ = tokio::task::spawn_blocking(move || {
                if let Ok(Some((old_holder, _))) = db2.lease_info(WEB_LEASE) {
                    db2.lease_release(WEB_LEASE, &old_holder)
                } else {
                    Ok(())
                }
            })
            .await;
            match acquire(&db, &holder).await {
                Ok(true) => {}
                Ok(false) => return ServeOutcome::Error("lease takeover failed".to_string()),
                Err(e) => return ServeOutcome::Error(format!("lease acquire failed: {e}")),
            }
        }
        Err(e) => return ServeOutcome::Error(format!("lease acquire failed: {e}")),
    }

    // Heartbeat: renew ahead of expiry; a failed renewal stops the server.
    let (lost_tx, mut lost_rx) = tokio::sync::oneshot::channel::<()>();
    let hb_db = db.clone();
    let hb_holder = holder.clone();
    let mut hb_shutdown = shutdown.clone();
    let heartbeat = tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = tokio::time::sleep(ELECTION_INTERVAL) => {
                    match acquire(&hb_db, &hb_holder).await {
                        Ok(true) => {}
                        Ok(false) => {
                            let _ = lost_tx.send(());
                            break;
                        }
                        Err(e) => tracing::warn!(error = %e, "web lease heartbeat failed"),
                    }
                }
                _ = hb_shutdown.changed() => break,
            }
        }
    });

    let mut lease_lost = false;
    let result = {
        let graceful = async {
            tokio::select! {
                _ = shutdown.changed() => false,
                _ = &mut lost_rx => true,
            }
        };
        tokio::pin!(graceful);
        let app = super::router(server);
        let serve = axum::serve(listener, app);
        tokio::select! {
            r = serve => r.map(|()| false),
            lost = &mut graceful => {
                lease_lost = lost;
                Ok(lost)
            }
        }
    };
    heartbeat.abort();
    let _ = tokio::task::spawn_blocking(move || db.lease_release(WEB_LEASE, &holder)).await;

    match result {
        Ok(_) if lease_lost => ServeOutcome::LeaseLost,
        Ok(_) => ServeOutcome::Shutdown,
        Err(e) => ServeOutcome::Error(e.to_string()),
    }
}

async fn acquire(db: &Database, holder: &str) -> crate::db::DbResult<bool> {
    let db = db.clone();
    let holder = holder.to_string();
    tokio::task::spawn_blocking(move || db.lease_acquire(WEB_LEASE, &holder, LEASE_TTL_SECS))
        .await
        .unwrap_or_else(|e| {
            tracing::warn!(error = %e, "lease task join failed");
            Ok(false)
        })
}

/// Best-effort `open`/`xdg-open`/`start`; failures are logged, not fatal.
fn open_browser(url: &str) {
    #[cfg(target_os = "macos")]
    let result = std::process::Command::new("open").arg(url).spawn();
    #[cfg(target_os = "windows")]
    let result = std::process::Command::new("cmd")
        .args(["/C", "start", "", url])
        .spawn();
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let result = std::process::Command::new("xdg-open").arg(url).spawn();
    if let Err(e) = result {
        tracing::warn!(error = %e, "could not open a browser");
    }
}

use crate::signals::shutdown_signal;
