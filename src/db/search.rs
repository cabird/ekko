//! FTS5 query preparation, keyword search over `sessions_fts`, and
//! transcript search/indexing over `transcripts_fts`.

use std::sync::LazyLock;

use regex::Regex;
use rusqlite::params;
use rusqlite::types::Value as SqlValue;

use super::sessions::SessionRow;
use super::{Database, DbError, DbResult, utc_now_iso};

/// Matches any character outside `[\p{L}\p{N}_*]`. Tokens containing such
/// characters get double-quoted before they reach FTS5.
static NON_WORD_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[^\p{L}\p{N}_*]").expect("valid regex"));

/// A `search_keyword` hit: the full session row plus the FTS snippet.
#[derive(Debug, Clone, PartialEq)]
pub struct KeywordHit {
    pub session: SessionRow,
    /// `snippet(sessions_fts, 3, '>>>', '<<<', '...', 30)` over the
    /// summary column.
    pub summary_excerpt: Option<String>,
}

/// A `search_transcripts` hit: the full session row plus the FTS snippet.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptHit {
    pub session: SessionRow,
    /// `snippet(transcripts_fts, 1, '>>>', '<<<', '...', 40)` over the
    /// transcript text.
    pub excerpt: Option<String>,
}

/// The whitespace set queries are tokenized on: the Unicode `White_Space`
/// property PLUS the four "file/group/record/unit separator" controls
/// U+001C..U+001F, which Rust's [`char::is_whitespace`] excludes.
use crate::utils::is_extended_whitespace as is_split_space;

/// Prepare a query for FTS5: split on whitespace (see [`is_split_space`]);
/// tokens containing characters outside `[\p{L}\p{N}_*]` are double-quoted
/// with inner quotes doubled; joined with spaces (implicit AND in FTS5).
pub(crate) fn prepare_fts_query(query: &str) -> String {
    let mut prepared: Vec<String> = Vec::new();
    for token in query.split(is_split_space).filter(|t| !t.is_empty()) {
        if NON_WORD_RE.is_match(token) {
            let escaped = token.replace('"', "\"\"");
            prepared.push(format!("\"{escaped}\""));
        } else {
            prepared.push(token.to_string());
        }
    }
    prepared.join(" ")
}

/// Quote the whole query as one literal phrase — the fallback for when
/// FTS5 rejects the prepared query.
fn quote_whole(query: &str) -> String {
    format!("\"{}\"", query.replace('"', "\"\""))
}

/// True for the operational errors worth retrying with a quoted query:
/// FTS5 syntax errors surface as a generic SQLITE_ERROR from prepare/step.
fn is_operational(err: &DbError) -> bool {
    matches!(err, DbError::Sqlite(rusqlite::Error::SqliteFailure(_, _)))
}

const SEARCH_KEYWORD_SQL: &str = "
    SELECT s.*,
           snippet(sessions_fts, 3, '>>>', '<<<', '...', 30) as summary_excerpt
    FROM sessions s
    JOIN sessions_fts fts ON s.session_id = fts.session_id
    WHERE sessions_fts MATCH ?
    ORDER BY bm25(sessions_fts)
    LIMIT ?
";

const SEARCH_TRANSCRIPTS_SQL: &str = "
    SELECT
        s.*,
        snippet(transcripts_fts, 1, '>>>', '<<<', '...', 40) as excerpt
    FROM sessions s
    JOIN transcripts_fts t ON s.session_id = t.session_id
    WHERE transcripts_fts MATCH ?
    ORDER BY bm25(transcripts_fts)
    LIMIT ?
";

impl Database {
    /// Full-text search on sessions: implicit AND matching, bm25 ordering,
    /// quoted-literal fallback when FTS5 rejects the query (reserved
    /// operators like `OR`, stray syntax).
    pub fn search_keyword(&self, query: &str, limit: i64) -> DbResult<Vec<KeywordHit>> {
        let prepared = prepare_fts_query(query);
        match self.run_keyword_search(&prepared, limit) {
            Ok(rows) => Ok(rows),
            Err(err) if is_operational(&err) => self.run_keyword_search(&quote_whole(query), limit),
            Err(err) => Err(err),
        }
    }

    fn run_keyword_search(&self, match_expr: &str, limit: i64) -> DbResult<Vec<KeywordHit>> {
        let conn = self.connection()?;
        let mut stmt = conn.prepare(SEARCH_KEYWORD_SQL)?;
        let rows = stmt.query_map(params![match_expr, limit], |row| {
            Ok(KeywordHit {
                session: SessionRow::from_row(row)?,
                summary_excerpt: row.get("summary_excerpt")?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    /// Full-text search across indexed transcripts; same fallback behavior
    /// as `search_keyword`.
    pub fn search_transcripts(&self, query: &str, limit: i64) -> DbResult<Vec<TranscriptHit>> {
        let prepared = prepare_fts_query(query);
        match self.run_transcript_search(&prepared, limit) {
            Ok(rows) => Ok(rows),
            Err(err) if is_operational(&err) => {
                self.run_transcript_search(&quote_whole(query), limit)
            }
            Err(err) => Err(err),
        }
    }

    fn run_transcript_search(&self, match_expr: &str, limit: i64) -> DbResult<Vec<TranscriptHit>> {
        let conn = self.connection()?;
        let mut stmt = conn.prepare(SEARCH_TRANSCRIPTS_SQL)?;
        let rows = stmt.query_map(params![match_expr, limit], |row| {
            Ok(TranscriptHit {
                session: SessionRow::from_row(row)?,
                excerpt: row.get("excerpt")?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    /// Rebuild the sessions FTS index.
    pub fn rebuild_fts(&self) -> DbResult<()> {
        let conn = self.connection()?;
        conn.execute(
            "INSERT INTO sessions_fts(sessions_fts) VALUES('rebuild')",
            [],
        )?;
        Ok(())
    }

    /// Index (or re-index) a transcript in the FTS table and stamp
    /// `transcript_indexed_at`.
    pub fn index_transcript(&self, session_id: &str, transcript_text: &str) -> DbResult<()> {
        // DELETE + INSERT + UPDATE must be one transaction: without it each
        // statement autocommits, so a concurrent indexer can slip an INSERT
        // between our DELETE and INSERT and leave two FTS rows for one
        // session.
        //
        // BEGIN IMMEDIATE, not DEFERRED: under WAL a deferred transaction
        // takes its read snapshot on the DELETE and then fails the write with
        // SQLITE_BUSY_SNAPSHOT if another writer committed in between — an
        // error the busy handler is NOT allowed to retry, so `busy_timeout`
        // does nothing there. IMMEDIATE takes the write lock up front, where
        // the timeout does apply, so concurrent indexers do not see spurious
        // "database is locked".
        let mut conn = self.connection()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        tx.execute(
            "DELETE FROM transcripts_fts WHERE session_id = ?",
            params![session_id],
        )?;
        tx.execute(
            "INSERT INTO transcripts_fts (session_id, transcript_text) VALUES (?, ?)",
            params![session_id, transcript_text],
        )?;
        tx.execute(
            "UPDATE sessions SET transcript_indexed_at = ? WHERE session_id = ?",
            params![utc_now_iso(), session_id],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Stamp `transcript_indexed_at` without adding FTS content — used when
    /// the transcript file is missing/unreadable so the session exits the
    /// needs-indexing queue.
    pub fn mark_transcript_indexed(&self, session_id: &str) -> DbResult<()> {
        let conn = self.connection()?;
        conn.execute(
            "UPDATE sessions SET transcript_indexed_at = ? WHERE session_id = ?",
            params![utc_now_iso(), session_id],
        )?;
        Ok(())
    }

    /// Retrieve indexed transcript text.
    pub fn get_transcript_text(&self, session_id: &str) -> DbResult<Option<String>> {
        let conn = self.connection()?;
        let mut stmt =
            conn.prepare("SELECT transcript_text FROM transcripts_fts WHERE session_id = ?")?;
        let mut rows =
            stmt.query_map(params![session_id], |row| row.get::<_, Option<String>>(0))?;
        match rows.next() {
            Some(row) => Ok(row?),
            None => Ok(None),
        }
    }

    /// Whether a transcript is already in the FTS index.
    pub fn is_transcript_indexed(&self, session_id: &str) -> DbResult<bool> {
        let conn = self.connection()?;
        let mut stmt = conn.prepare("SELECT 1 FROM transcripts_fts WHERE session_id = ?")?;
        let mut rows = stmt.query_map(params![session_id], |_| Ok(()))?;
        Ok(rows.next().transpose()?.is_some())
    }

    /// Sessions needing transcript indexing — never indexed, or file
    /// modified after last indexing. `Some(0)` for `limit` means no limit.
    pub fn get_sessions_needing_transcript_index(
        &self,
        limit: Option<i64>,
    ) -> DbResult<Vec<SessionRow>> {
        let mut query = String::from(
            "SELECT s.* FROM sessions s
             WHERE s.transcript_path != ''
             AND s.message_count > 0
             AND (
                 s.transcript_indexed_at IS NULL
                 OR julianday(REPLACE(s.modified_at, 'T', ' '))
                    > julianday(REPLACE(s.transcript_indexed_at, 'T', ' '))
             )
             ORDER BY s.modified_at DESC",
        );
        let mut sql_params: Vec<SqlValue> = Vec::new();
        if let Some(l) = limit.filter(|l| *l != 0) {
            query.push_str(" LIMIT ?");
            sql_params.push(SqlValue::Integer(l));
        }
        self.query_sessions(&query, &sql_params)
    }
}
