//! Background indexing and summarization.
//!
//! Design:
//!
//! - Starts [`STARTUP_DELAY_SECONDS`] after server launch — if the process is
//!   still alive by then, it is long-running and worth indexing for.
//! - Only summarizes/snapshots transcripts at least [`MIN_AGE_HOURS`] old
//!   (stability window; active sessions keep changing).
//! - Coordinates across instances through the `summary_claims` table with a
//!   [`CLAIM_TTL_SECONDS`] TTL. There is no leader election: whichever
//!   instance wins the atomic claim does the work.
//! - Idempotent: an existing summary drops the session out of the queue, and
//!   a lost claim simply skips.
//!
//! Logging: every cycle event goes to `tracing` (stderr — stdout is JSON-RPC)
//! **and** is appended to `indexer.log` in the data directory. Set
//! `EKKO_LOG_FILE` to relocate the file and `EKKO_DEBUG=1` for debug lines.
//!
//! # Concurrency
//!
//! The loop is a tokio task. File parsing and SQLite work go through
//! `spawn_blocking`; the LLM calls are natively async and are awaited
//! directly. Shutdown is a `tokio::sync::watch` flag, checked between
//! sessions so a cycle never has to be torn down mid-session.
//!
//! # Error handling
//!
//! [`BackgroundIndexer::run_cycle`] is selective about what it swallows:
//! scanning and transcript FTS absorb their own errors, but the staleness
//! queries, the claim/release calls and the claim cleanup propagate —
//! reaching the loop's handler, which logs and backs off for
//! [`ERROR_BACKOFF_SECONDS`].

use std::fmt::Write as _;
use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use tokio::sync::watch;
use tokio::task::{JoinError, JoinHandle};

use crate::db::{Database, DbError, SessionRow};
use crate::indexer;
use crate::llm::{Role, check_backend_available};
use crate::models::truncate_chars;
use crate::{config, paths, snapshot, summarize};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Wait this long before starting background work.
pub const STARTUP_DELAY_SECONDS: u64 = 60;
/// Only summarize/snapshot transcripts at least this old.
pub const MIN_AGE_HOURS: f64 = 1.0;
/// How often to re-scan for new sessions (5 minutes). Also the idle sleep.
pub const SCAN_INTERVAL_SECONDS: u64 = 300;
/// How long a claim lasts (10 minutes).
pub const CLAIM_TTL_SECONDS: i64 = 600;
/// How many sessions to process per cycle.
pub const BATCH_SIZE: i64 = 10;
/// Back-off after an unhandled error in the loop.
pub const ERROR_BACKOFF_SECONDS: u64 = 60;

/// Environment variable overriding the log-file location.
pub const ENV_LOG_FILE: &str = "EKKO_LOG_FILE";
/// Environment variable enabling debug-level background logging.
pub const ENV_DEBUG: &str = "EKKO_DEBUG";

/// Build a fresh instance id: the process id, a hyphen, and 8 random
/// lowercase hex digits.
pub fn new_instance_id() -> String {
    let mut hex = String::with_capacity(8);
    for _ in 0..4 {
        let byte: u8 = rand::random();
        let _ = write!(hex, "{byte:02x}");
    }
    format!("{}-{}", std::process::id(), hex)
}

/// The process-wide background instance id. Deliberately distinct from the
/// telemetry recorder's id: one identifies the indexing worker, the other
/// identifies the serving process.
pub fn instance_id() -> &'static str {
    static ID: OnceLock<String> = OnceLock::new();
    ID.get_or_init(new_instance_id)
}

/// Anything that escapes a cycle and triggers the 60s back-off.
#[derive(Debug, thiserror::Error)]
pub enum CycleError {
    /// A SQLite failure in one of the unguarded steps.
    #[error("{0}")]
    Db(#[from] DbError),
    /// A blocking task panicked or was cancelled.
    #[error("background task failed: {0}")]
    Join(#[from] JoinError),
}

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

/// Appends the same lines `tracing` emits to `indexer.log`.
///
/// Lines are formatted `<timestamp> [<LEVEL>] ekko.background: <message>`.
/// There is no rotation, and a failed open silently disables file logging
/// rather than taking the indexer down with it.
#[derive(Debug, Clone, Default)]
pub struct FileLog {
    path: Option<PathBuf>,
    debug: bool,
}

impl FileLog {
    /// Resolve the log file from `EKKO_LOG_FILE`, else `<data_dir>/indexer.log`.
    pub fn from_env() -> Self {
        let path = std::env::var(ENV_LOG_FILE)
            .ok()
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .or_else(|| paths::get_data_dir().ok().map(|d| d.join("indexer.log")));
        FileLog {
            path,
            debug: std::env::var(ENV_DEBUG).is_ok_and(|v| !v.is_empty()),
        }
    }

    /// A logger that writes to `path` only.
    pub fn to_path(path: impl Into<PathBuf>) -> Self {
        FileLog {
            path: Some(path.into()),
            debug: false,
        }
    }

    /// A logger that never writes to disk (tests, embedded use).
    pub fn none() -> Self {
        FileLog::default()
    }

    fn append(&self, level: &str, message: &str) {
        let Some(path) = &self.path else { return };
        let ts = chrono::Local::now().format("%Y-%m-%d %H:%M:%S,%3f");
        let line = format!("{ts} [{level}] ekko.background: {message}\n");
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
            let _ = f.write_all(line.as_bytes());
        }
    }

    /// Log at INFO.
    pub fn info(&self, message: impl AsRef<str>) {
        let message = message.as_ref();
        tracing::info!("{message}");
        self.append("INFO", message);
    }

    /// Log at WARNING.
    pub fn warn(&self, message: impl AsRef<str>) {
        let message = message.as_ref();
        tracing::warn!("{message}");
        self.append("WARNING", message);
    }

    /// Log at ERROR.
    pub fn error(&self, message: impl AsRef<str>) {
        let message = message.as_ref();
        tracing::error!("{message}");
        self.append("ERROR", message);
    }

    /// Log at DEBUG (suppressed in the file unless `EKKO_DEBUG` is set).
    pub fn debug(&self, message: impl AsRef<str>) {
        let message = message.as_ref();
        tracing::debug!("{message}");
        if self.debug {
            self.append("DEBUG", message);
        }
    }
}

// ---------------------------------------------------------------------------
// Claim keys (pure — exercised directly by tests)
// ---------------------------------------------------------------------------

/// The `content_hash` used to claim a session's **summary**: its
/// `source_hash` when present and non-empty, else the session id.
pub fn summary_claim_key(session: &SessionRow) -> String {
    match session.source_hash.as_deref() {
        Some(h) if !h.is_empty() => h.to_string(),
        _ => session.session_id.clone(),
    }
}

/// The `content_hash` used to claim a session's **snapshot**: the summary key
/// with a `snapshot:` prefix so the two never collide.
pub fn snapshot_claim_key(session: &SessionRow) -> String {
    format!("snapshot:{}", summary_claim_key(session))
}

// ---------------------------------------------------------------------------
// The indexer
// ---------------------------------------------------------------------------

/// Run a blocking closure on the blocking pool.
async fn blocking<T, F>(f: F) -> Result<T, JoinError>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(f).await
}

/// Background indexer state.
///
/// Drive it one cycle at a time with [`BackgroundIndexer::run_cycle`] (what
/// the tests do) or as a long-running task via [`start`].
pub struct BackgroundIndexer {
    db: Database,
    instance_id: String,
    log: FileLog,
    llm_warning_logged: bool,
    /// `None` until the first scan, which always counts as "long enough ago".
    last_scan: Option<Instant>,
    stop: watch::Receiver<bool>,
}

impl BackgroundIndexer {
    /// A never-cancelled indexer using the process-wide instance id and the
    /// environment-configured log file.
    pub fn new(db: Database) -> Self {
        let (tx, rx) = watch::channel(false);
        // Dropping the sender leaves `borrow()` returning `false` forever.
        drop(tx);
        BackgroundIndexer {
            db,
            instance_id: instance_id().to_string(),
            log: FileLog::from_env(),
            llm_warning_logged: false,
            last_scan: None,
            stop: rx,
        }
    }

    /// Override the log destination (tests, or an embedder that wants
    /// stderr-only logging).
    #[must_use]
    pub fn with_log(mut self, log: FileLog) -> Self {
        self.log = log;
        self
    }

    /// Override the instance id used for claims.
    #[must_use]
    pub fn with_instance_id(mut self, id: impl Into<String>) -> Self {
        self.instance_id = id.into();
        self
    }

    /// Wire up a shutdown signal.
    #[must_use]
    pub fn with_stop(mut self, stop: watch::Receiver<bool>) -> Self {
        self.stop = stop;
        self
    }

    /// The instance id this indexer claims work under.
    pub fn instance_id(&self) -> &str {
        &self.instance_id
    }

    fn stopped(&self) -> bool {
        *self.stop.borrow()
    }

    /// Run one indexing/summarization/snapshot cycle.
    ///
    /// `Ok(true)` means work was done. The session re-scan — the slow part on
    /// network mounts — only runs every [`SCAN_INTERVAL_SECONDS`].
    pub async fn run_cycle(&mut self) -> Result<bool, CycleError> {
        let mut indexed = 0usize;
        let due = match self.last_scan {
            None => true,
            Some(t) => t.elapsed() >= Duration::from_secs(SCAN_INTERVAL_SECONDS),
        };
        if due {
            indexed = self.scan_and_index().await;
            self.last_scan = Some(Instant::now());
        }

        // Transcript FTS needs no LLM — just parsing.
        let transcripts_done = self.index_transcripts_batch().await;

        // Summarization and snapshots require LLM access.
        let config = config::get_config();
        let (available, msg) = check_backend_available(config, Role::Index);
        if !available {
            if !self.llm_warning_logged {
                self.log
                    .warn(format!("Skipping summarization/snapshots: {msg}"));
                self.llm_warning_logged = true;
            }
            return Ok(indexed > 0 || transcripts_done > 0);
        }

        let summaries_done = self.summarize_batch().await?;
        let snapshots_done = self.snapshot_batch().await?;

        // Cleanup expired claims periodically.
        let db = self.db.clone();
        blocking(move || db.cleanup_expired_claims()).await??;

        Ok(indexed > 0 || transcripts_done > 0 || summaries_done > 0 || snapshots_done > 0)
    }

    /// Scan for new sessions and index them. Swallows every error: a bad
    /// transcript must not stop the cycle.
    async fn scan_and_index(&self) -> usize {
        let db = self.db.clone();
        match blocking(move || indexer::scan_all(&db)).await {
            Ok(stats) => {
                if stats.total_indexed > 0 {
                    self.log
                        .info(format!("Indexed {} sessions", stats.total_indexed));
                }
                stats.total_indexed
            }
            Err(e) => {
                self.log.error(format!("Error scanning sessions: {e}"));
                0
            }
        }
    }

    /// Index parsed transcript text into FTS. Also swallows every error.
    async fn index_transcripts_batch(&self) -> usize {
        let db = self.db.clone();
        let stats =
            match blocking(move || indexer::index_transcripts_fts(&db, Some(BATCH_SIZE))).await {
                Ok(s) => s,
                Err(e) => {
                    self.log.error(format!("Error indexing transcripts: {e}"));
                    return 0;
                }
            };
        let (count, skipped, errors) = (stats.indexed, stats.skipped, stats.errors);
        if count > 0 {
            let suffix = if skipped > 0 || errors > 0 {
                format!(" ({skipped} skipped, {errors} errors)")
            } else {
                String::new()
            };
            self.log
                .info(format!("Indexed {count} transcripts into FTS{suffix}"));
        } else if skipped > 0 || errors > 0 {
            self.log.info(format!(
                "Transcript FTS batch: 0 indexed, {skipped} skipped, {errors} errors"
            ));
        }
        count
    }

    /// Summarize a batch of stable sessions.
    async fn summarize_batch(&mut self) -> Result<usize, CycleError> {
        let db = self.db.clone();
        let sessions = blocking(move || {
            db.get_sessions_needing_summary_stable(MIN_AGE_HOURS, Some(BATCH_SIZE))
        })
        .await??;

        if sessions.is_empty() {
            return Ok(0);
        }
        self.log.info(format!(
            "Found {} sessions needing summaries",
            sessions.len()
        ));

        let mut count = 0usize;
        for session in &sessions {
            if self.stopped() {
                break;
            }
            let content_hash = summary_claim_key(session);
            if !self.claim(&session.session_id, &content_hash).await? {
                self.log.debug(format!(
                    "Session {} claimed by another instance, skipping",
                    session.session_id
                ));
                continue;
            }

            let outcome = self.summarize_one(session).await;
            if outcome {
                count += 1;
            }

            // Release the claim (optional — the TTL would handle it anyway).
            self.release(&session.session_id, &content_hash).await?;
        }
        Ok(count)
    }

    /// The per-session body of `_summarize_batch`'s `try` block. Returns
    /// whether the session counts as summarized.
    ///
    /// Thin wrapper over the free function [`summarize_one`], which
    /// `fastindex` drives in parallel.
    async fn summarize_one(&self, session: &SessionRow) -> bool {
        summarize_one(&self.db, &self.log, session).await
    }

    /// Generate snapshots for a batch of sessions. Only sessions that already
    /// have summaries and are stable are eligible.
    async fn snapshot_batch(&mut self) -> Result<usize, CycleError> {
        let db = self.db.clone();
        let sessions = blocking(move || {
            db.get_sessions_needing_snapshot(Some(BATCH_SIZE), true, Some(MIN_AGE_HOURS), true)
        })
        .await??;

        if sessions.is_empty() {
            return Ok(0);
        }
        self.log.info(format!(
            "Found {} sessions needing snapshots",
            sessions.len()
        ));

        let mut count = 0usize;
        for session in &sessions {
            if self.stopped() {
                break;
            }
            let content_hash = snapshot_claim_key(session);
            if !self.claim(&session.session_id, &content_hash).await? {
                self.log.debug(format!(
                    "Session {} snapshot claimed by another instance, skipping",
                    session.session_id
                ));
                continue;
            }

            if self.snapshot_one(session).await {
                count += 1;
            }

            self.release(&session.session_id, &content_hash).await?;
        }
        Ok(count)
    }

    /// Snapshot one session, absorbing its own failures.
    ///
    /// Thin wrapper over the free function [`snapshot_one`].
    async fn snapshot_one(&self, session: &SessionRow) -> bool {
        snapshot_one(&self.db, &self.log, session).await
    }

    /// Try to claim a session. Deliberately **unguarded**: a SQLite failure
    /// here escapes the cycle rather than being silently skipped.
    async fn claim(&self, session_id: &str, content_hash: &str) -> Result<bool, CycleError> {
        claim(&self.db, session_id, content_hash, &self.instance_id).await
    }

    /// Release a claim — also unguarded, for the same reason.
    async fn release(&self, session_id: &str, content_hash: &str) -> Result<(), CycleError> {
        release(&self.db, session_id, content_hash).await
    }

    /// The main loop: cycle, sleep
    /// [`SCAN_INTERVAL_SECONDS`] only when the cycle found nothing to do, and
    /// back off [`ERROR_BACKOFF_SECONDS`] on an escaped error.
    async fn run_loop(mut self) {
        while !self.stopped() {
            match self.run_cycle().await {
                Ok(true) => {}
                Ok(false) => {
                    self.sleep(Duration::from_secs(SCAN_INTERVAL_SECONDS)).await;
                }
                Err(e) => {
                    self.log.error(format!("Error in background indexer: {e}"));
                    self.sleep(Duration::from_secs(ERROR_BACKOFF_SECONDS)).await;
                }
            }
        }
        self.log.info("Background indexer stopped");
    }

    /// Sleep, waking early on shutdown.
    ///
    /// A dropped `watch::Sender` makes `changed()` return `Err` immediately
    /// and forever; treating that as a wake-up would turn every sleep into a
    /// hot spin hammering SQLite. A dropped sender means no one can signal
    /// stop anymore, so it is treated as "never signalled".
    async fn sleep(&mut self, duration: Duration) {
        let stop = &mut self.stop;
        tokio::select! {
            _ = tokio::time::sleep(duration) => {}
            _ = async {
                if stop.changed().await.is_err() {
                    std::future::pending::<()>().await;
                }
            } => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Per-item steps (shared with `crate::fastindex`)
// ---------------------------------------------------------------------------
//
// These are the exact bodies the background loop used to inline as `&self`
// methods; the methods above are now one-line delegations, so the loop's
// behavior is unchanged. They are free functions taking `(db, log, …)` so the
// foreground `ekko index` driver can run the *same* claim → parse → LLM →
// write → release path concurrently.

/// Summarize one session and write the result. Returns whether the session
/// counts as summarized. Never panics and never propagates: every failure is
/// logged and reported as `false`.
pub async fn summarize_one(db: &Database, log: &FileLog, session: &SessionRow) -> bool {
    let session_id = &session.session_id;
    let result = match summarize::summarize_session(session).await {
        Ok(r) => r,
        Err(e) => {
            log.warn(format!("Error summarizing {session_id}: {e}"));
            return false;
        }
    };

    // CRITICAL: `title`/`summary` stay `Option` all the way to SQL. A model
    // that emits `"title": null` must bind SQL NULL, which leaves the session
    // in the needs-summary queue. Substituting `""` here would instead mark
    // the session done with an empty summary.
    let db2 = db.clone();
    let sid = session_id.clone();
    let (title, summary) = (result.title.clone(), result.summary.clone());
    let write =
        blocking(move || db2.update_summary_opt(&sid, title.as_deref(), summary.as_deref())).await;
    match write {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            log.error(format!("Error summarizing session {session_id}: {e}"));
            return false;
        }
        Err(e) => {
            log.error(format!("Error summarizing session {session_id}: {e}"));
            return false;
        }
    }

    match result.title {
        Some(title) => {
            log.info(format!(
                "Summarized session: {}",
                truncate_chars(&title, 50)
            ));
            true
        }
        None => {
            // A null title is an error *after* the NULL write: the write
            // stands (keeping the session queued), the failure is logged,
            // and the session is not counted as summarized.
            log.error(format!(
                "Error summarizing session {session_id}: title is null"
            ));
            false
        }
    }
}

/// Generate and store one session's snapshot. Same never-propagates contract
/// as [`summarize_one`].
pub async fn snapshot_one(db: &Database, log: &FileLog, session: &SessionRow) -> bool {
    let session_id = &session.session_id;
    let markdown = match snapshot::generate_snapshot(session).await {
        Ok(m) => m,
        Err(e) => {
            log.warn(format!("Error generating snapshot {session_id}: {e}"));
            return false;
        }
    };

    // Stored verbatim — no fence stripping or other post-processing.
    let db2 = db.clone();
    let sid = session_id.clone();
    let write = blocking(move || db2.update_snapshot(&sid, &markdown)).await;
    match write {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            log.error(format!(
                "Error generating snapshot for session {session_id}: {e}"
            ));
            return false;
        }
        Err(e) => {
            log.error(format!(
                "Error generating snapshot for session {session_id}: {e}"
            ));
            return false;
        }
    }

    match &session.title {
        Some(title) => {
            let label = truncate_chars(title, 50);
            let label = if label.is_empty() {
                truncate_chars(session_id, 8)
            } else {
                label
            };
            log.info(format!("Generated snapshot: {label}"));
            true
        }
        None => {
            // A NULL title column is reported as an error even though the
            // snapshot itself was already written.
            log.error(format!(
                "Error generating snapshot for session {session_id}: title is null"
            ));
            false
        }
    }
}

/// Atomically claim a session's summary/snapshot slot for `instance_id`
/// (TTL [`CLAIM_TTL_SECONDS`]). `Ok(false)` means another instance holds it.
///
/// # Errors
///
/// Propagates SQLite failures: claiming is deliberately unguarded, so a
/// failure backs the whole loop off rather than silently skipping work.
pub async fn claim(
    db: &Database,
    session_id: &str,
    content_hash: &str,
    instance_id: &str,
) -> Result<bool, CycleError> {
    let db = db.clone();
    let (sid, hash, inst) = (
        session_id.to_string(),
        content_hash.to_string(),
        instance_id.to_string(),
    );
    Ok(blocking(move || db.try_claim_summary(&sid, &hash, &inst, CLAIM_TTL_SECONDS)).await??)
}

/// Release a claim — also unguarded, for the same reason.
///
/// # Errors
///
/// Propagates SQLite failures.
pub async fn release(
    db: &Database,
    session_id: &str,
    content_hash: &str,
) -> Result<(), CycleError> {
    let db = db.clone();
    let (sid, hash) = (session_id.to_string(), content_hash.to_string());
    blocking(move || db.release_claim(&sid, &hash)).await??;
    Ok(())
}

/// Handle to a running background indexer: the spawned task plus its stop
/// signal.
#[derive(Debug)]
pub struct BackgroundHandle {
    stop: watch::Sender<bool>,
    task: JoinHandle<()>,
}

impl BackgroundHandle {
    /// Signal shutdown and wait for the loop to finish.
    pub async fn stop(self) {
        let _ = self.stop.send(true);
        let _ = self.task.await;
    }

    /// Signal shutdown and abort the task without waiting.
    pub fn abort(&self) {
        let _ = self.stop.send(true);
        self.task.abort();
    }
}

/// Start background indexing after the startup delay. Cheap and
/// non-blocking: the returned handle owns the task.
///
/// There is no module-global singleton — the MCP lifespan owns the handle,
/// and dropping or stopping it is what shuts the indexer down.
pub fn start(db: Database) -> BackgroundHandle {
    let (tx, rx) = watch::channel(false);
    let indexer = BackgroundIndexer::new(db).with_stop(rx.clone());
    let log = indexer.log.clone();
    let id = indexer.instance_id.clone();
    log.info(format!(
        "Background indexer scheduled to start in {STARTUP_DELAY_SECONDS}s (instance: {id})"
    ));

    let task = tokio::spawn(async move {
        let mut rx = rx;
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_secs(STARTUP_DELAY_SECONDS)) => {}
            _ = rx.changed() => {}
        }
        if *rx.borrow() {
            log.info("Background indexer cancelled during startup delay");
            return;
        }
        log.info(format!("Background indexer starting (instance: {id})"));
        indexer.run_loop().await;
    });

    BackgroundHandle { stop: tx, task }
}
