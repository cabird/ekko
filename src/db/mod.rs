//! SQLite storage layer for ekko sessions, search, claims, and telemetry.
//!
//! The schema is created fresh in its final shape; there is no migration
//! code, and the database is a rebuildable derived index.
//!
//! # Connection strategy
//!
//! `Database` stores only the database path and opens a new
//! [`rusqlite::Connection`] per call with a 30 s busy timeout. This keeps
//! `Database` trivially `Send + Sync` (it is just a `PathBuf`) so the
//! background worker can wrap calls in `spawn_blocking` without a `Mutex`,
//! and WAL plus the busy timeout arbitrate between multiple server
//! instances sharing one file.
//!
//! Pragmas. `journal_mode=WAL` is persisted in the file itself;
//! `synchronous` and `foreign_keys` are *per-connection* and so must be set
//! on every connection that cares about them.
//!
//! **Foreign keys are declared in the schema but never enforced.**
//! `connection()` explicitly sets `foreign_keys=OFF` — SQLite's own default,
//! which rusqlite otherwise overrides by enabling enforcement on every
//! connection it opens. So `add_file_reference("no-such-session", …)`
//! succeeds and leaves an orphan `session_files` row; callers must not rely
//! on the database to reject dangling references.
//!
//! `connection()` also applies `synchronous=NORMAL` to every per-operation
//! connection. This is a deliberate durability trade, not a correctness one:
//! under WAL, `NORMAL` means a commit is not fsynced to the WAL before
//! returning, so a machine crash (not a process crash) can lose the most
//! recent commits. Chosen for indexing throughput; the data is rebuildable.
//!
//! # Timestamp storage format
//!
//! All timestamps are stored as TEXT in ISO 8601 shape and compared as
//! strings (or via `julianday`) by the staleness queries, so format
//! consistency is load-bearing:
//!
//! - `created_at` / `modified_at` (from parsers): `format_iso8601` of the
//!   parsed `DateTime<FixedOffset>` — `YYYY-MM-DDTHH:MM:SS[.ffffff]+HH:MM`,
//!   fractional seconds only when the microsecond component is nonzero.
//!   Note that ekko's parsed timestamps are always offset-aware, including
//!   the vscode_copilot ones, which are local wall-clock times carrying the
//!   machine's current UTC offset. Two consequences worth knowing:
//!   * The `julianday(...)`-based staleness queries
//!     ([`Database::get_sessions_needing_transcript_index`],
//!     [`Database::get_sessions_needing_summary_stable`]) convert those
//!     stamps to true UTC, since `julianday` honours the offset (and treats
//!     an offset-less stamp as UTC). A staleness verdict can therefore flip
//!     when `modified_at` and the `*_at` stamp are within one local UTC
//!     offset of each other.
//!   * The plain string comparisons (`REPLACE(a,'T',' ') > REPLACE(b,'T',' ')`
//!     in `get_sessions_needing_summary` / `get_sessions_needing_snapshot`,
//!     and `< datetime('now', ?)`) compare the leading `YYYY-MM-DD HH:MM:SS`
//!     text first, so they are unaffected except in the tie case where two
//!     stamps share an identical datetime prefix — there the
//!     offset-suffixed value sorts greater (longer string wins).
//! - `summary_at` / `snapshot_at` / `transcript_indexed_at` / telemetry
//!   `timestamp`: current UTC instant — always `+00:00`-suffixed with
//!   microseconds.
//! - `indexed_at`: on INSERT, the current local time (naive, no offset); on
//!   UPDATE conflict, SQL `CURRENT_TIMESTAMP` (UTC, space-separated).

mod claims;
pub(crate) mod events;
pub mod leases;
mod search;
mod sessions;
mod similar;
mod stats;
mod telemetry;
#[cfg(test)]
mod tests;
mod web;

pub use events::{EVENT_CONTENT_CAP, EVENT_TOOL_INPUT_CAP, EVENT_TOOL_OUTPUT_CAP};
pub use search::{KeywordHit, TranscriptHit};
pub use sessions::{SessionFile, SessionRow};
pub use stats::{DbStats, ProjectRow, TranscriptIndexStats};
pub use telemetry::{
    CostBucket, CostStats, LlmUsage, OperationCost, OperationStats, TelemetryEvent,
    TelemetrySummary,
};
pub use web::{WebCursor, WebListFilter, WebSessionPage, WebSort};

use std::path::{Path, PathBuf};
use std::time::Duration;

use chrono::{DateTime, FixedOffset, Local, Timelike, Utc};
use rusqlite::Connection;

/// Errors from the database layer.
#[derive(Debug, thiserror::Error)]
pub enum DbError {
    /// Any underlying SQLite failure.
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// Stored JSON (e.g. telemetry `safe_args`) failed to parse.
    #[error("invalid JSON in {context}: {source}")]
    Json {
        context: &'static str,
        #[source]
        source: serde_json::Error,
    },
}

/// Convenience alias used throughout the module.
pub type DbResult<T> = Result<T, DbError>;

/// Handle to the sessions database. See module docs for the
/// connection-per-call rationale.
#[derive(Debug, Clone)]
pub struct Database {
    db_path: PathBuf,
}

/// Schema DDL, created in one shot — there is no migration facility.
///
/// Column order in `sessions` is load-bearing: `sessions_fts` is an
/// external-content FTS5 table over it, so the rowid mapping depends on the
/// declared order. Append new columns at the end; never reorder or remove
/// one without rebuilding the FTS index.
const SCHEMA_SQL: &str = "
    -- Claims table for coordinating background summarization
    CREATE TABLE IF NOT EXISTS summary_claims (
        session_id TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        claimed_until INTEGER NOT NULL,
        claimed_by TEXT NOT NULL,
        PRIMARY KEY (session_id, content_hash)
    );

    -- Telemetry table for anonymous usage tracking
    -- Records operation names + safe args only (no free-text, no paths)
    CREATE TABLE IF NOT EXISTS telemetry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        instance_id TEXT NOT NULL,  -- GUID per MCP server instance
        timestamp TEXT NOT NULL,
        operation TEXT NOT NULL,
        duration_ms INTEGER,
        safe_args TEXT,  -- JSON of enum/boolean/numeric args only
        -- LLM usage tracking (v0.6.2+)
        model TEXT,           -- Model used (e.g., 'gpt-5-mini')
        input_tokens INTEGER, -- Prompt/input tokens
        output_tokens INTEGER,-- Completion/output tokens
        thinking_tokens INTEGER, -- Reasoning tokens (if applicable)
        cost REAL             -- Computed cost in dollars
    );

    CREATE INDEX IF NOT EXISTS idx_telemetry_instance
        ON telemetry(instance_id);

    CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp
        ON telemetry(timestamp);
    CREATE INDEX IF NOT EXISTS idx_telemetry_operation
        ON telemetry(operation);

    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        cli_source TEXT NOT NULL DEFAULT 'claude_code',
        title TEXT,
        summary TEXT,
        first_prompt TEXT,
        project_path TEXT,
        git_branch TEXT,
        git_remote TEXT,
        transcript_path TEXT,
        created_at TEXT,
        modified_at TEXT,
        message_count INTEGER DEFAULT 0,
        event_count INTEGER DEFAULT 0,
        files_touched_count INTEGER DEFAULT 0,
        parser_version TEXT DEFAULT '1.0',
        normalized_version INTEGER DEFAULT 1,
        source_hash TEXT,
        model TEXT,
        duration_ms INTEGER,
        tool_timeline TEXT,
        tags TEXT,
        token_count INTEGER DEFAULT 0,
        keywords TEXT,
        indexed_at TEXT DEFAULT CURRENT_TIMESTAMP,
        context_snapshot TEXT,
        summary_at TEXT,
        snapshot_at TEXT,
        transcript_indexed_at TEXT
    );

    -- Singleton role leases (db/leases.rs) — who hosts the web UI. Unlike
    -- summary_claims, a holder can renew before expiry (heartbeat), which
    -- is what long-lived hosting needs. Additive table (v0.1+web).
    CREATE TABLE IF NOT EXISTS leases (
        name TEXT PRIMARY KEY,
        holder TEXT NOT NULL,
        expires_at INTEGER NOT NULL
    );

    -- Structured transcript events, capped at ingest (db/events.rs). The
    -- database is the source of truth for the transcript view: once a
    -- session is ingested, the .jsonl on disk can be reclaimed without
    -- losing the browsable transcript. Additive table (v0.1+web).
    -- pos is the storage ordinal (0..n in parse order); the parser's own
    -- `sequence` lives inside the JSON and is NOT unique (an assistant
    -- message and its tool calls from one transcript entry share it).
    CREATE TABLE IF NOT EXISTS session_events (
        session_id TEXT NOT NULL,
        pos INTEGER NOT NULL,
        event TEXT NOT NULL,
        PRIMARY KEY (session_id, pos)
    );

    CREATE TABLE IF NOT EXISTS session_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        file_path TEXT NOT NULL,
        operation TEXT DEFAULT 'read',
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    );

    CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_path);
    CREATE INDEX IF NOT EXISTS idx_sessions_cli ON sessions(cli_source);
    CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at);
    CREATE INDEX IF NOT EXISTS idx_session_files_session ON session_files(session_id);
    CREATE INDEX IF NOT EXISTS idx_session_files_path ON session_files(file_path);

    -- FTS5 for full-text search on sessions
    CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
        session_id,
        title,
        first_prompt,
        summary,
        tool_timeline,
        keywords,
        content='sessions',
        content_rowid='rowid'
    );

    -- FTS5 for full-text search on transcript content
    CREATE VIRTUAL TABLE IF NOT EXISTS transcripts_fts USING fts5(
        session_id,
        transcript_text,
        tokenize='porter unicode61'
    );

    -- Triggers to keep FTS in sync
    CREATE TRIGGER IF NOT EXISTS sessions_ai AFTER INSERT ON sessions BEGIN
        INSERT INTO sessions_fts(rowid, session_id, title, first_prompt, summary, tool_timeline, keywords)
        VALUES (new.rowid, new.session_id, new.title, new.first_prompt, new.summary, new.tool_timeline, new.keywords);
    END;

    CREATE TRIGGER IF NOT EXISTS sessions_ad AFTER DELETE ON sessions BEGIN
        INSERT INTO sessions_fts(sessions_fts, rowid, session_id, title, first_prompt, summary, tool_timeline, keywords)
        VALUES ('delete', old.rowid, old.session_id, old.title, old.first_prompt, old.summary, old.tool_timeline, old.keywords);
    END;

    CREATE TRIGGER IF NOT EXISTS sessions_au AFTER UPDATE ON sessions BEGIN
        INSERT INTO sessions_fts(sessions_fts, rowid, session_id, title, first_prompt, summary, tool_timeline, keywords)
        VALUES ('delete', old.rowid, old.session_id, old.title, old.first_prompt, old.summary, old.tool_timeline, old.keywords);
        INSERT INTO sessions_fts(rowid, session_id, title, first_prompt, summary, tool_timeline, keywords)
        VALUES (new.rowid, new.session_id, new.title, new.first_prompt, new.summary, new.tool_timeline, new.keywords);
    END;
";

impl Database {
    /// Open (creating if necessary) the database at `db_path` and ensure
    /// the schema exists. Idempotent — safe to call on an existing file.
    pub fn new(db_path: impl Into<PathBuf>) -> DbResult<Self> {
        let db = Database {
            db_path: db_path.into(),
        };
        db.init_db()?;
        Ok(db)
    }

    /// The path this database lives at.
    pub fn path(&self) -> &Path {
        &self.db_path
    }

    /// Open a fresh connection with a 30 s busy timeout. Deliberately does
    /// NOT set `foreign_keys=ON`, so FK constraints are never enforced and
    /// orphan `session_files` rows are legal; `synchronous=NORMAL` is the
    /// deliberate durability trade. Both are explained in the module docs.
    pub(crate) fn connection(&self) -> DbResult<Connection> {
        let conn = Connection::open(&self.db_path)?;
        conn.busy_timeout(Duration::from_millis(30_000))?;
        // `foreign_keys=OFF` is SQLite's own default, but rusqlite turns FK
        // enforcement ON for every connection it opens, so it has to be
        // turned back off explicitly.
        conn.execute_batch("PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=OFF;")?;
        Ok(conn)
    }

    /// Create the schema on a fresh database. Idempotent.
    fn init_db(&self) -> DbResult<()> {
        let conn = self.connection()?;
        // WAL is persistent in the file; returns a row, so query it.
        conn.query_row("PRAGMA journal_mode=WAL;", [], |_| Ok(()))?;
        // Both of these are per-connection and this connection is dropped at
        // the end of the function, so they have no lasting effect.
        conn.execute_batch("PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON;")?;
        conn.execute_batch(SCHEMA_SQL)?;
        Ok(())
    }
}

// ─── Timestamp helpers ────────────────────────────────────────────────────

/// Format an offset-aware value as ISO 8601:
/// `YYYY-MM-DDTHH:MM:SS[.ffffff]±HH:MM` — fractional seconds printed as
/// exactly six digits only when the microsecond component is nonzero.
pub(crate) fn format_iso8601(dt: &DateTime<FixedOffset>) -> String {
    // Leap seconds (unreachable in practice): chrono encodes one as
    // nanosecond >= 1_000_000_000 while keeping :59 in the seconds field.
    // The `% 1_000_000` below wraps that extra second away, printing
    // `…:59.xxxxxx`. No parser in ekko can produce a leap-second instant.
    let micros = dt.nanosecond() / 1_000;
    let micros = micros % 1_000_000;
    if micros == 0 {
        dt.format("%Y-%m-%dT%H:%M:%S%:z").to_string()
    } else {
        format!(
            "{}.{:06}{}",
            dt.format("%Y-%m-%dT%H:%M:%S"),
            micros,
            dt.format("%:z")
        )
    }
}

/// The current instant in UTC, ISO 8601 with a `+00:00` offset.
pub(crate) fn utc_now_iso() -> String {
    format_iso8601(&Utc::now().fixed_offset())
}

/// The current local wall-clock time, ISO 8601 with no offset suffix.
pub(crate) fn local_now_iso_naive() -> String {
    let now = Local::now();
    let micros = (now.nanosecond() / 1_000) % 1_000_000;
    if micros == 0 {
        now.format("%Y-%m-%dT%H:%M:%S").to_string()
    } else {
        format!("{}.{:06}", now.format("%Y-%m-%dT%H:%M:%S"), micros)
    }
}

/// Format a float for the modifier strings passed to SQLite `datetime()`
/// (e.g. `1.0` → `"1.0"`), keeping a `.0` on whole values.
///
/// Edge cases, all unreachable from the call sites (`min_age_hours` is a
/// small positive constant — 1.0 or 24.0): `|f| >= 1e16` falls back to the
/// plain `{f}` rendering, and `nan` prints as `NaN`. SQLite's modifier
/// parser rejects all of those anyway, so the query result is unaffected.
pub(crate) fn float_str(f: f64) -> String {
    if f == f.trunc() && f.is_finite() && f.abs() < 1e16 {
        format!("{f:.1}")
    } else {
        format!("{f}")
    }
}

/// Serialize JSON the way this layer persists it (keyword lists and
/// telemetry safe-args): ASCII-only escaping, and `", "` / `": "`
/// separators rather than the compact ones.
pub(crate) fn json_dumps(value: &serde_json::Value) -> String {
    use serde_json::Value;
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        Value::Number(n) => {
            if n.is_f64() {
                float_repr(n.as_f64().unwrap_or(f64::NAN))
            } else {
                n.to_string()
            }
        }
        Value::String(s) => json_dumps_str(s),
        Value::Array(items) => {
            let inner: Vec<String> = items.iter().map(json_dumps).collect();
            format!("[{}]", inner.join(", "))
        }
        Value::Object(map) => {
            let inner: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}: {}", json_dumps_str(k), json_dumps(v)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
    }
}

/// Shortest round-trip rendering of a float, as used inside [`json_dumps`].
/// Rust's `{:?}` already produces it; exponent notation is fixed up to a
/// signed, zero-padded two-digit exponent (`1e16` → `1e+16`, `1e-5` →
/// `1e-05`). `models.rs` has a near-identical private helper that differs
/// only in how it spells the non-finite values.
pub(crate) fn float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".to_string(); // unreachable via serde_json
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    let s = format!("{f:?}");
    if let Some(pos) = s.find('e') {
        let (mantissa, exp) = s.split_at(pos);
        let exp = &exp[1..];
        let (sign, digits) = match exp.strip_prefix('-') {
            Some(d) => ("-", d),
            None => ("+", exp.strip_prefix('+').unwrap_or(exp)),
        };
        let digits = if digits.len() < 2 {
            format!("0{digits}")
        } else {
            digits.to_string()
        };
        return format!("{mantissa}e{sign}{digits}");
    }
    s
}

/// Serialize a single JSON string with ASCII-only escaping: every non-ASCII
/// character becomes `\uXXXX` (a surrogate pair for astral planes).
pub(crate) fn json_dumps_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c if c.is_ascii() => out.push(c),
            c => {
                let cp = c as u32;
                if cp > 0xFFFF {
                    let v = cp - 0x10000;
                    let hi = 0xD800 + (v >> 10);
                    let lo = 0xDC00 + (v & 0x3FF);
                    out.push_str(&format!("\\u{hi:04x}\\u{lo:04x}"));
                } else {
                    out.push_str(&format!("\\u{cp:04x}"));
                }
            }
        }
    }
    out.push('"');
    out
}
