//! The `ekko` binary.
//!
//! ```text
//! ekko            # same as `ekko mcp`
//! ekko mcp        # run the MCP server over stdio
//! ekko mcp --check  # legacy form, same as `ekko check`
//! ekko check      # run health/configuration checks
//! ekko index      # index everything now, in parallel, then exit
//! ekko web        # serve the browser UI on localhost
//! ```

use clap::{Parser, Subcommand};

/// ekko — searchable archive of past AI coding sessions.
#[derive(Debug, Parser)]
#[command(name = "ekko", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run the MCP server over stdio (default).
    Mcp {
        /// Run health/configuration checks instead of the server.
        ///
        /// Legacy form; `ekko check` is the preferred spelling.
        #[arg(long)]
        check: bool,
    },
    /// Run health/configuration checks.
    Check,
    /// Index everything now, in parallel, then exit.
    ///
    /// Scans the session directories, indexes transcripts into FTS, extracts
    /// keywords, then summarizes and snapshots every eligible session with
    /// --workers LLM calls in flight. The "ekko mcp" server does the same
    /// work in the background, but deliberately slowly (10 sessions per
    /// 5-minute cycle, one LLM call at a time); use this to catch up a fresh
    /// or long-neglected database. Safe to run while a server is up —
    /// sessions are claimed one at a time, so the two never duplicate work.
    ///
    /// Sessions whose transcript changed in the last hour are skipped in
    /// either mode: they are probably still being written.
    ///
    /// The scan and keyword passes re-read every transcript each run, so even
    /// a no-op run takes a few minutes on a large archive.
    Index {
        /// Sessions summarized/snapshotted concurrently
        /// [default: the config's max_workers, normally 7].
        #[arg(long, value_name = "N")]
        workers: Option<usize>,

        /// Skip the LLM passes: scan, transcript FTS and keywords only.
        #[arg(long)]
        no_llm: bool,

        /// Re-summarize and re-snapshot sessions that are already up to date.
        #[arg(long)]
        force: bool,
    },
    /// Serve the browser UI on localhost until interrupted.
    ///
    /// Read-only browsing of the same archive the MCP server uses — sessions,
    /// summaries, context snapshots, transcripts, search — plus "Ask", which
    /// runs the ask_question tool's Q&A path. Run it alongside "ekko mcp";
    /// the two never conflict. This command runs no indexing passes (the
    /// one write it makes: viewing a not-yet-ingested transcript stores its
    /// capped events, so the archive keeps it after the file is cleaned up).
    Web {
        /// Port on 127.0.0.1 [default: the config's web_port, normally 3556].
        #[arg(long, value_name = "PORT")]
        port: Option<u16>,

        /// Open the UI in the default browser once listening.
        #[arg(long)]
        open: bool,
    },
}

/// `ekko index` — a foreground, parallel indexing run.
///
/// Progress goes to **stdout** (this is not an MCP process, so stdout is free;
/// keeping it there makes the run pipeable), diagnostics to stderr.
async fn run_index(workers: Option<usize>, no_llm: bool, force: bool) -> i32 {
    use ekko::background::FileLog;
    use ekko::fastindex::{IndexOptions, run};

    let config = ekko::config::get_config();
    let db_path = match &config.database_path {
        Some(p) => p.clone(),
        None => match ekko::paths::get_database_path() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("ekko: cannot resolve the database path: {e}");
                return 1;
            }
        },
    };
    if let Some(parent) = db_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let db = match ekko::db::Database::new(&db_path) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("ekko: cannot open {}: {e}", db_path.display());
            return 1;
        }
    };

    let mut opts = IndexOptions::from_config();
    if let Some(w) = workers {
        opts.workers = w;
    }
    opts.no_llm = no_llm;
    opts.force = force;

    // The same `indexer.log` the background indexer writes to.
    let log = FileLog::from_env();
    let mut progress = |line: &str| println!("{line}");

    match run(&db, &log, &opts, &mut progress).await {
        Ok(report) => {
            println!("{}", report.render_summary());
            0
        }
        Err(e) => {
            eprintln!("ekko: indexing failed: {e}");
            1
        }
    }
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("ekko: cannot start async runtime: {e}");
            return std::process::ExitCode::from(1);
        }
    };

    // Load .env before configuring logging, so EKKO_LOG / EKKO_LOG_FILE set
    // in a .env file reach the tracing subscriber too.
    let _ = dotenvy::dotenv();

    let code = runtime.block_on(async move {
        match cli.command.unwrap_or(Command::Mcp { check: false }) {
            Command::Check | Command::Mcp { check: true } => {
                // Keep the file-logging guard alive for the whole run.
                let _guard = ekko::mcp::init_tracing();
                ekko::check::run_checks().await
            }
            Command::Index {
                workers,
                no_llm,
                force,
            } => {
                let _guard = ekko::mcp::init_tracing();
                run_index(workers, no_llm, force).await
            }
            Command::Web { port, open } => {
                let _guard = ekko::mcp::init_tracing();
                let port = port.unwrap_or_else(|| ekko::config::get_config().web_port);
                ekko::web::server::run(ekko::web::server::WebOptions { port, open }).await
            }
            Command::Mcp { check: false } => {
                // Keep the file-logging guard alive for the whole run.
                let _guard = ekko::mcp::init_tracing();
                match ekko::mcp::run_stdio().await {
                    Ok(()) => 0,
                    Err(e) => {
                        tracing::error!("{e}");
                        eprintln!("ekko: {e}");
                        1
                    }
                }
            }
        }
    });

    // Spawned tasks (web co-host, indexer) must not outlive the command:
    // a lingering runtime would keep serving with no one to renew leases.
    runtime.shutdown_timeout(std::time::Duration::from_secs(5));
    std::process::ExitCode::from(code.clamp(0, 255) as u8)
}
