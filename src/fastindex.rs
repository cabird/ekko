//! Foreground bulk indexing — the engine behind `ekko index`.
//!
//! # Why a separate module (and not a mode of `background.rs`)
//!
//! [`crate::background`] models a *daemon lifecycle*: a 60s startup delay, a
//! 10-session batch per cycle, a 300s re-scan interval, sequential LLM calls,
//! a stop signal and an error back-off — all deliberately throttled so an MCP
//! server stays light. This module is a *driver* with the opposite goal:
//! finish everything as fast as the API allows and exit. Folding it into
//! `BackgroundIndexer` would mean threading "am I the fast one?" through the
//! cycle logic and putting the daemon's byte-for-byte-ported behavior at risk.
//!
//! What it does **not** re-implement: every unit of work goes through the
//! existing pipeline —
//!
//! * [`indexer::scan_all`], [`indexer::index_transcripts_fts`],
//!   [`indexer::index_keywords`] (fast, local, already batch-shaped);
//! * [`background::claim`] → [`background::summarize_one`] /
//!   [`background::snapshot_one`] → [`background::release`], the *same* free
//!   functions the background loop calls per session (claim → parse → LLM →
//!   write → release, same `FileLog` lines, same `summary_claims` rows, same
//!   [`background::CLAIM_TTL_SECONDS`]).
//!
//! The drivers here are a `max_workers` fan-out with per-item error
//! isolation, using the bounded-fan-out pattern already established in
//! [`crate::memory::qa`]:
//! `stream::iter(boxed_futures).buffer_unordered(workers)`.
//!
//! # Concurrency and safety
//!
//! * `workers` is the only throttle: at most that many sessions are in flight,
//!   which caps LLM concurrency. No new rate limiting is introduced.
//! * Claims are taken per item exactly as the background loop takes them, so
//!   running `ekko index` while an `ekko mcp` server is up cannot duplicate
//!   work. A slow LLM call inside a claim is the same exposure the background
//!   loop already has (the 600s TTL can expire mid-call).
//! * SQLite writes are connection-per-call (`BEGIN IMMEDIATE` where needed),
//!   so parallel item completion needs no extra coordination.
//!
//! # Deliberate differences from the background loop
//!
//! * **Batch size**: no [`background::BATCH_SIZE`] cap — the passes take the
//!   whole queue.
//! * **Claim/release errors**: the background loop lets them escape the cycle
//!   (60s back-off); here they fail that one item and the run continues. A
//!   bulk run must not die on one bad row.
//! * **Stability gate kept, staleness gate optional**: see
//!   [`IndexOptions::force`].

use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use futures_util::StreamExt;
use futures_util::future::BoxFuture;
use serde_json::Value;

use crate::background::{
    self, CycleError, FileLog, MIN_AGE_HOURS, snapshot_claim_key, summary_claim_key,
};
use crate::config;
use crate::db::{Database, SessionRow};
use crate::indexer::{self, IndexStats};
use crate::llm::{Role, check_backend_available, get_usage_tracker};
use crate::utils::parse_datetime;

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Knobs for [`run`].
#[derive(Debug, Clone)]
pub struct IndexOptions {
    /// Maximum sessions summarized/snapshotted concurrently. Clamped to at
    /// least 1.
    pub workers: usize,
    /// Skip the LLM passes entirely: scan + transcript FTS + keywords only.
    pub no_llm: bool,
    /// Re-summarize/re-snapshot sessions that are already up to date: the
    /// needs-work queries are swapped for every session with a transcript.
    ///
    /// `force` bypasses the *staleness* filter only. It does **not** bypass
    /// the [`min_age_hours`](Self::min_age_hours) stability window (a bulk run
    /// should still not burn tokens on a session that is being written right
    /// now) and it does **not** bypass claims.
    pub force: bool,
    /// Stability window: only sessions whose transcript has been untouched
    /// this long are eligible. Defaults to [`MIN_AGE_HOURS`] (1h), the same
    /// gate the background loop applies.
    pub min_age_hours: f64,
}

impl IndexOptions {
    /// Defaults with `workers` taken from the configured `max_workers`
    /// (config.json / default 7).
    pub fn from_config() -> Self {
        IndexOptions {
            workers: crate::llm::effective_max_workers(config::get_config()),
            no_llm: false,
            force: false,
            min_age_hours: MIN_AGE_HOURS,
        }
    }

    /// `workers`, floored at 1.
    pub fn effective_workers(&self) -> usize {
        self.workers.max(1)
    }
}

impl Default for IndexOptions {
    fn default() -> Self {
        IndexOptions {
            workers: 7,
            no_llm: false,
            force: false,
            min_age_hours: MIN_AGE_HOURS,
        }
    }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// What one session's summarize/snapshot attempt produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemOutcome {
    /// Written successfully.
    Done,
    /// Attempted and failed (LLM error, parse error, write error, or a
    /// claim/release database error).
    Failed,
    /// Another instance held the claim; not attempted.
    Skipped,
}

/// Running totals for one parallel pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PassCounters {
    /// Items in the queue.
    pub total: usize,
    /// Items finished (done + failed + skipped).
    pub completed: usize,
    /// Items written successfully.
    pub done: usize,
    /// Items that errored.
    pub failed: usize,
    /// Items claimed by another instance.
    pub skipped: usize,
}

impl PassCounters {
    fn record(&mut self, outcome: ItemOutcome) {
        self.completed += 1;
        match outcome {
            ItemOutcome::Done => self.done += 1,
            ItemOutcome::Failed => self.failed += 1,
            ItemOutcome::Skipped => self.skipped += 1,
        }
    }

    /// `"42/436 (3 failed, 1 claimed elsewhere)"`.
    pub fn render(&self) -> String {
        let mut notes: Vec<String> = Vec::new();
        if self.failed > 0 {
            notes.push(format!("{} failed", self.failed));
        }
        if self.skipped > 0 {
            notes.push(format!("{} claimed elsewhere", self.skipped));
        }
        let suffix = if notes.is_empty() {
            String::new()
        } else {
            format!(" ({})", notes.join(", "))
        };
        format!("{}/{}{}", self.completed, self.total, suffix)
    }
}

/// Everything one `ekko index` run did.
#[derive(Debug, Clone)]
pub struct IndexReport {
    /// Sessions upserted by [`indexer::scan_all`].
    pub scanned: usize,
    /// Transcript files found by the scan.
    pub found: usize,
    /// Transcript-FTS pass.
    pub fts: IndexStats,
    /// Keyword-extraction pass.
    pub keywords: IndexStats,
    /// Why the LLM passes were skipped, if they were.
    pub llm_skipped: Option<String>,
    /// Summary pass counters.
    pub summaries: PassCounters,
    /// Snapshot pass counters.
    pub snapshots: PassCounters,
    /// Wall time.
    pub elapsed: Duration,
    /// Process-wide LLM usage after the run
    /// ([`crate::llm::get_usage_tracker`]).
    pub usage: Value,
}

impl IndexReport {
    /// The single closing line printed by the CLI.
    pub fn render_summary(&self) -> String {
        let usage = &self.usage;
        let calls = usage.get("calls").and_then(Value::as_u64).unwrap_or(0);
        let tokens = usage
            .get("total_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        // Keys per `UsageStats::to_json`.
        let cost = usage
            .get("cost_display")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| "$0.0000".to_string());
        format!(
            "done in {:.1}s — scanned {}, fts {}, keywords {}, summaries {}, snapshots {}; \
llm {calls} calls, {tokens} tokens, {cost}",
            self.elapsed.as_secs_f64(),
            self.scanned,
            self.fts.indexed,
            self.keywords.indexed,
            self.summaries.render(),
            self.snapshots.render(),
        )
    }
}

// ---------------------------------------------------------------------------
// Stability filter (pure)
// ---------------------------------------------------------------------------

/// Whether a session's transcript has been untouched for `min_age_hours`.
///
/// Used only on the `--force` paths, where the SQL queries that embed
/// `datetime('now', '-N hours')` are not available. A missing or unparseable
/// `modified_at` counts as **not** stable: the point of the gate is to avoid
/// spending tokens on a session that may still be growing, and an unreadable
/// timestamp is no evidence that it isn't.
///
/// Parsing goes through [`parse_datetime`], so timestamps carrying an offset
/// are compared correctly (the SQL gate compares them as strings against a UTC
/// `datetime('now')`, which is only correct for UTC/naive values).
pub fn is_stable(modified_at: Option<&str>, now: DateTime<Utc>, min_age_hours: f64) -> bool {
    let Some(raw) = modified_at.filter(|s| !s.is_empty()) else {
        return false;
    };
    let Some(dt) = parse_datetime(raw) else {
        return false;
    };
    let age = now.signed_duration_since(dt.with_timezone(&Utc));
    age.num_milliseconds() as f64 >= min_age_hours * 3_600_000.0
}

// ---------------------------------------------------------------------------
// The bounded parallel driver
// ---------------------------------------------------------------------------

/// Run `futures` with at most `workers` in flight, counting outcomes and
/// calling `on_progress` after each completion.
///
/// As in [`crate::memory::qa`], `buffered` yields in input order rather than
/// completion order; only the progress reporting is affected, and a stable
/// order is the friendlier one.
///
/// Every future is expected to be infallible — the per-item helpers below
/// swallow and log their own errors, which is what isolates one bad session
/// from the rest of the run.
pub async fn drive<'a>(
    futures: Vec<BoxFuture<'a, ItemOutcome>>,
    workers: usize,
    on_progress: &mut dyn FnMut(&PassCounters),
) -> PassCounters {
    let mut counters = PassCounters {
        total: futures.len(),
        ..PassCounters::default()
    };
    if counters.total == 0 {
        return counters;
    }
    // buffer_unordered, not buffered: ordered admission holds a worker slot
    // for every completed-but-not-yet-yielded item, so one slow map-reduce
    // session at the queue head would collapse effective LLM concurrency to
    // 1 until it finishes. Completion order is fine here — the counters are
    // order-independent (review find M1).
    let mut stream = futures_util::stream::iter(futures).buffer_unordered(workers.max(1));
    while let Some(outcome) = stream.next().await {
        counters.record(outcome);
        on_progress(&counters);
    }
    counters
}

// ---------------------------------------------------------------------------
// Per-item steps (claim → work → release)
// ---------------------------------------------------------------------------

async fn run_claimed<F>(
    db: &Database,
    log: &FileLog,
    instance_id: &str,
    session_id: &str,
    content_hash: &str,
    kind: &str,
    work: F,
) -> ItemOutcome
where
    F: AsyncFnOnce() -> bool,
{
    match background::claim(db, session_id, content_hash, instance_id).await {
        Ok(true) => {}
        Ok(false) => {
            log.debug(format!(
                "Session {session_id} {kind} claimed by another instance, skipping"
            ));
            return ItemOutcome::Skipped;
        }
        Err(e) => {
            log.error(format!("Error claiming {kind} for {session_id}: {e}"));
            return ItemOutcome::Failed;
        }
    }

    let ok = work().await;

    // Release even on failure — same as the background loop's `finally`.
    if let Err(e) = background::release(db, session_id, content_hash).await {
        log.error(format!(
            "Error releasing {kind} claim for {session_id}: {e}"
        ));
    }

    if ok {
        ItemOutcome::Done
    } else {
        ItemOutcome::Failed
    }
}

/// One session's summary, through the background loop's own code path.
pub async fn summarize_item(
    db: &Database,
    log: &FileLog,
    instance_id: &str,
    session: &SessionRow,
) -> ItemOutcome {
    let key = summary_claim_key(session);
    run_claimed(
        db,
        log,
        instance_id,
        &session.session_id,
        &key,
        "summary",
        async || background::summarize_one(db, log, session).await,
    )
    .await
}

/// One session's snapshot, through the background loop's own code path.
pub async fn snapshot_item(
    db: &Database,
    log: &FileLog,
    instance_id: &str,
    session: &SessionRow,
) -> ItemOutcome {
    let key = snapshot_claim_key(session);
    run_claimed(
        db,
        log,
        instance_id,
        &session.session_id,
        &key,
        "snapshot",
        async || background::snapshot_one(db, log, session).await,
    )
    .await
}

// ---------------------------------------------------------------------------
// Queue selection
// ---------------------------------------------------------------------------

/// Sessions for the summary pass.
///
/// * default: [`Database::get_sessions_needing_summary_stable`] — exactly the
///   background loop's query, minus the batch limit;
/// * `--force`: every session with a transcript, filtered back down to the
///   stable ones by [`is_stable`].
fn summary_queue(db: &Database, opts: &IndexOptions) -> Result<Vec<SessionRow>, CycleError> {
    if opts.force {
        let now = Utc::now();
        let all = db.get_all_sessions_with_transcripts(None)?;
        Ok(all
            .into_iter()
            .filter(|s| is_stable(s.modified_at.as_deref(), now, opts.min_age_hours))
            .collect())
    } else {
        Ok(db.get_sessions_needing_summary_stable(opts.min_age_hours, None)?)
    }
}

/// Sessions for the snapshot pass — same structure as [`summary_queue`]. The
/// `--force` branch keeps the background loop's "must already have a summary"
/// requirement; it is queried *after* the summary pass, so sessions summarized
/// moments ago are included.
fn snapshot_queue(db: &Database, opts: &IndexOptions) -> Result<Vec<SessionRow>, CycleError> {
    if opts.force {
        let now = Utc::now();
        let all = db.get_all_sessions_with_transcripts(None)?;
        Ok(all
            .into_iter()
            .filter(|s| {
                s.summary.as_deref().is_some_and(|v| !v.is_empty())
                    && is_stable(s.modified_at.as_deref(), now, opts.min_age_hours)
            })
            .collect())
    } else {
        Ok(db.get_sessions_needing_snapshot(None, true, Some(opts.min_age_hours), true)?)
    }
}

// ---------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------

/// Run a full foreground indexing pass and return once everything is done.
///
/// `progress` receives one preformatted line per stage transition and per
/// completed item (the CLI prints them to stdout); it is never called with a
/// trailing newline.
///
/// # Errors
///
/// Only the *queue* queries and the final claim cleanup propagate: if the
/// database itself is unusable the run should stop. Scanning, FTS, keywords
/// and every individual session isolate their own failures.
pub async fn run(
    db: &Database,
    log: &FileLog,
    opts: &IndexOptions,
    progress: &mut dyn FnMut(&str),
) -> Result<IndexReport, CycleError> {
    let started = Instant::now();
    let workers = opts.effective_workers();
    let instance_id = background::instance_id().to_string();

    // ---- 1. Scan (blocking, already batch-shaped) -------------------------
    progress("scanning session directories...");
    let scan = {
        let db = db.clone();
        tokio::task::spawn_blocking(move || indexer::scan_all(&db)).await?
    };
    progress(&format!(
        "scan: {} transcripts found, {} indexed, {} skipped (not interactive)",
        scan.total_found, scan.total_indexed, scan.total_skipped_non_interactive
    ));

    // ---- 2. Local text indexing (no LLM) ----------------------------------
    progress("indexing transcripts into FTS...");
    let fts = {
        let db = db.clone();
        // `None` = no limit: the background loop's BATCH_SIZE cap is what makes
        // it slow, and slow is what this command exists to avoid.
        tokio::task::spawn_blocking(move || indexer::index_transcripts_fts(&db, None)).await?
    };
    progress(&format!(
        "fts: {} indexed, {} skipped, {} errors",
        fts.indexed, fts.skipped, fts.errors
    ));

    progress("extracting keywords...");
    let keywords = {
        let db = db.clone();
        tokio::task::spawn_blocking(move || indexer::index_keywords(&db, None)).await?
    };
    progress(&format!(
        "keywords: {} indexed, {} skipped, {} errors",
        keywords.indexed, keywords.skipped, keywords.errors
    ));

    let mut report = IndexReport {
        scanned: scan.total_indexed,
        found: scan.total_found,
        fts,
        keywords,
        llm_skipped: None,
        summaries: PassCounters::default(),
        snapshots: PassCounters::default(),
        elapsed: Duration::ZERO,
        usage: get_usage_tracker().to_json(),
    };

    // ---- 3. LLM gate -------------------------------------------------------
    if opts.no_llm {
        report.llm_skipped = Some("--no-llm".to_string());
    } else {
        let config = config::get_config();
        let (available, msg) = check_backend_available(config, Role::Index);
        if !available {
            report.llm_skipped = Some(msg);
        }
    }
    if let Some(reason) = &report.llm_skipped {
        progress(&format!("skipping summaries and snapshots: {reason}"));
        log.info(format!("Fast index: skipping LLM passes: {reason}"));
        report.elapsed = started.elapsed();
        report.usage = get_usage_tracker().to_json();
        return Ok(report);
    }

    // ---- 4. Summaries ------------------------------------------------------
    let sessions = {
        let db2 = db.clone();
        let opts2 = opts.clone();
        tokio::task::spawn_blocking(move || summary_queue(&db2, &opts2)).await??
    };
    progress(&format!(
        "summaries: {} sessions queued ({} workers)",
        sessions.len(),
        workers
    ));
    log.info(format!(
        "Fast index: {} sessions needing summaries",
        sessions.len()
    ));
    let futures: Vec<BoxFuture<'_, ItemOutcome>> = sessions
        .iter()
        .map(|s| Box::pin(summarize_item(db, log, &instance_id, s)) as BoxFuture<'_, _>)
        .collect();
    report.summaries = drive(futures, workers, &mut |c| {
        progress(&format!("summaries {}", c.render()));
    })
    .await;

    // ---- 5. Snapshots ------------------------------------------------------
    let sessions = {
        let db2 = db.clone();
        let opts2 = opts.clone();
        tokio::task::spawn_blocking(move || snapshot_queue(&db2, &opts2)).await??
    };
    progress(&format!(
        "snapshots: {} sessions queued ({} workers)",
        sessions.len(),
        workers
    ));
    log.info(format!(
        "Fast index: {} sessions needing snapshots",
        sessions.len()
    ));
    let futures: Vec<BoxFuture<'_, ItemOutcome>> = sessions
        .iter()
        .map(|s| Box::pin(snapshot_item(db, log, &instance_id, s)) as BoxFuture<'_, _>)
        .collect();
    report.snapshots = drive(futures, workers, &mut |c| {
        progress(&format!("snapshots {}", c.render()));
    })
    .await;

    // ---- 6. Housekeeping ---------------------------------------------------
    {
        let db = db.clone();
        tokio::task::spawn_blocking(move || db.cleanup_expired_claims()).await??;
    }

    report.elapsed = started.elapsed();
    report.usage = get_usage_tracker().to_json();
    Ok(report)
}

#[cfg(test)]
mod tests;
