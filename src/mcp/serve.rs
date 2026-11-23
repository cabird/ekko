//! Process wiring: logging, the stdio transport, and the lifespan that owns
//! the background indexer.

use std::io::IsTerminal;

use rmcp::ServiceExt;
use rmcp::transport::stdio;
use tracing_subscriber::prelude::*;
use tracing_subscriber::{EnvFilter, fmt};

use crate::background;
use crate::config::get_config;
use crate::mcp::EkkoServer;

/// Environment variable naming an additional log file (same variable the
/// background indexer uses for `indexer.log` overrides).
pub const ENV_LOG_FILE: &str = background::ENV_LOG_FILE;

/// Environment variable holding the tracing filter (default `info`).
pub const ENV_LOG_LEVEL: &str = "EKKO_LOG";

/// Install the tracing subscriber.
///
/// **stdout carries JSON-RPC**, so every log line goes to stderr. When
/// `EKKO_LOG_FILE` is set, the same events are additionally appended to that
/// file.
///
/// Returns a guard that must be kept alive for the duration of the process
/// when file logging is active (`None` otherwise).
pub fn init_tracing() -> Option<tracing_appender::non_blocking::WorkerGuard> {
    let filter = EnvFilter::try_from_env(ENV_LOG_LEVEL).unwrap_or_else(|_| EnvFilter::new("info"));
    let stderr_layer = fmt::layer()
        .with_writer(std::io::stderr)
        .with_ansi(std::io::stderr().is_terminal());

    let (file_layer, guard) = match std::env::var(ENV_LOG_FILE) {
        Ok(path) if !path.is_empty() => match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            Ok(file) => {
                let (writer, guard) = tracing_appender::non_blocking(file);
                (
                    Some(fmt::layer().with_writer(writer).with_ansi(false)),
                    Some(guard),
                )
            }
            Err(e) => {
                eprintln!("ekko: cannot open {ENV_LOG_FILE}={path}: {e}");
                (None, None)
            }
        },
        _ => (None, None),
    };

    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(stderr_layer)
        .with(file_layer)
        .try_init();

    guard
}

/// Run the MCP server over stdio until the client disconnects (or the
/// process is interrupted), then stop the background indexer.
///
/// # Errors
///
/// Fails if the database cannot be opened or the transport dies unexpectedly.
pub async fn run_stdio() -> Result<(), String> {
    // Load .env before anything else, so API keys are visible to the config.
    let _ = dotenvy::dotenv();
    let config = get_config();

    let server = EkkoServer::from_config(config)?;

    // Lifespan, part 1: schedule background indexing (60s startup delay).
    let handle = background::start(server.db().clone());
    server.indexer_status().mark_running();
    tracing::info!("Background indexing scheduled");

    // Optionally co-host the web UI: a lease-elected
    // side job — exactly one MCP instance serves the port at a time.
    let (web_stop_tx, web_handle) = if config.web_serve {
        let (tx, rx) = tokio::sync::watch::channel(false);
        let web_server = server.clone();
        let port = config.web_port;
        (
            Some(tx),
            Some(tokio::spawn(crate::web::server::run_cohosted(
                web_server, port, rx,
            ))),
        )
    } else {
        (None, None)
    };

    // Serve, then wait. Any failure here still falls through to the
    // lifespan cleanup below: the web co-host and the indexer must be
    // stopped (and the hosting lease released) on every exit path, not
    // just the graceful one.
    let outcome: Result<String, String> = match server.clone().serve(stdio()).await {
        Err(e) => Err(format!("failed to start MCP server: {e}")),
        Ok(service) => Ok(tokio::select! {
            quit = service.waiting() => match quit {
                Ok(reason) => format!("{reason:?}"),
                Err(e) => format!("transport error: {e}"),
            },
            _ = shutdown_signal() => "interrupt".to_string(),
        }),
    };
    let reason = match &outcome {
        Ok(r) => r.clone(),
        Err(e) => format!("startup failed: {e}"),
    };

    // Lifespan, part 2.
    if let Some(tx) = web_stop_tx {
        let _ = tx.send(true);
    }
    if let Some(web) = web_handle {
        // Graceful web shutdown is quick (lease release + listener close);
        // don't let a wedged task hold the whole process hostage.
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), web).await;
    }
    handle.stop().await;
    server.indexer_status().mark_stopped();
    tracing::info!(%reason, "Background indexing stopped");
    outcome.map(|_| ())
}

use crate::signals::shutdown_signal;
