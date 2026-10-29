//! Session row storage: upserts, getters, updates, staleness queries, and
//! `session_files` bookkeeping.

use std::collections::HashMap;

use rusqlite::types::Value as SqlValue;
use rusqlite::{Row, params};

use super::{Database, DbResult, local_now_iso_naive, float_str, format_iso8601, utc_now_iso};
use crate::models::SessionMetadata;

/// One row from the `sessions` table. Nullable columns are `Option`, and
/// `Some("")` is distinct from `None` — an empty `git_branch` is not the
/// same as a missing one.
///
/// The count columns have SQL defaults of 0 and are always written by this
/// layer, but the schema allows NULL; a NULL reads back as 0 here.
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize)]
pub struct SessionRow {
    pub session_id: String,
    pub cli_source: String,
    pub title: Option<String>,
    pub summary: Option<String>,
    pub first_prompt: Option<String>,
    pub project_path: Option<String>,
    pub git_branch: Option<String>,
    pub git_remote: Option<String>,
    pub transcript_path: Option<String>,
    pub created_at: Option<String>,
    pub modified_at: Option<String>,
    pub message_count: i64,
    pub event_count: i64,
    pub files_touched_count: i64,
    pub parser_version: Option<String>,
    pub normalized_version: i64,
    pub source_hash: Option<String>,
    pub model: Option<String>,
    pub duration_ms: Option<i64>,
    pub tool_timeline: Option<String>,
    pub tags: Option<String>,
    pub token_count: i64,
    pub keywords: Option<String>,
    pub indexed_at: Option<String>,
    pub context_snapshot: Option<String>,
    pub summary_at: Option<String>,
    pub snapshot_at: Option<String>,
    pub transcript_indexed_at: Option<String>,
}

impl SessionRow {
    /// Build from any row that includes the full `sessions` column set
    /// (works for `SELECT *` and `SELECT s.*, extra…` shapes because
    /// columns are read by name).
    pub(crate) fn from_row(row: &Row<'_>) -> rusqlite::Result<Self> {
        Ok(SessionRow {
            session_id: row.get("session_id")?,
            cli_source: row.get("cli_source")?,
            title: row.get("title")?,
            summary: row.get("summary")?,
            first_prompt: row.get("first_prompt")?,
            project_path: row.get("project_path")?,
            git_branch: row.get("git_branch")?,
            git_remote: row.get("git_remote")?,
            transcript_path: row.get("transcript_path")?,
            created_at: row.get("created_at")?,
            modified_at: row.get("modified_at")?,
            message_count: row.get::<_, Option<i64>>("message_count")?.unwrap_or(0),
            event_count: row.get::<_, Option<i64>>("event_count")?.unwrap_or(0),
            files_touched_count: row
                .get::<_, Option<i64>>("files_touched_count")?
                .unwrap_or(0),
            parser_version: row.get("parser_version")?,
            normalized_version: row
                .get::<_, Option<i64>>("normalized_version")?
                .unwrap_or(1),
            source_hash: row.get("source_hash")?,
            model: row.get("model")?,
            duration_ms: row.get("duration_ms")?,
            tool_timeline: row.get("tool_timeline")?,
            tags: row.get("tags")?,
            token_count: row.get::<_, Option<i64>>("token_count")?.unwrap_or(0),
            keywords: row.get("keywords")?,
            indexed_at: row.get("indexed_at")?,
            context_snapshot: row.get("context_snapshot")?,
            summary_at: row.get("summary_at")?,
            snapshot_at: row.get("snapshot_at")?,
            transcript_indexed_at: row.get("transcript_indexed_at")?,
        })
    }
}

/// A file touched by a session (`session_files` row payload).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionFile {
    pub file_path: String,
    pub operation: String,
}

const UPSERT_SQL: &str = "
    INSERT INTO sessions (
        session_id, cli_source, title, first_prompt, project_path,
        git_branch, git_remote, transcript_path, created_at, modified_at,
        message_count, event_count, parser_version, normalized_version,
        source_hash, model, tool_timeline, indexed_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(session_id) DO UPDATE SET
        cli_source = excluded.cli_source,
        first_prompt = COALESCE(excluded.first_prompt, first_prompt),
        project_path = excluded.project_path,
        git_branch = excluded.git_branch,
        git_remote = excluded.git_remote,
        transcript_path = excluded.transcript_path,
        created_at = COALESCE(excluded.created_at, created_at),
        modified_at = COALESCE(excluded.modified_at, modified_at),
        message_count = excluded.message_count,
        event_count = excluded.event_count,
        parser_version = excluded.parser_version,
        normalized_version = excluded.normalized_version,
        source_hash = excluded.source_hash,
        model = COALESCE(excluded.model, model),
        tool_timeline = COALESCE(excluded.tool_timeline, tool_timeline),
        indexed_at = CURRENT_TIMESTAMP
";

const UPSERT_WITH_FILES_SQL: &str = "
    INSERT INTO sessions (
        session_id, cli_source, title, first_prompt, project_path,
        git_branch, git_remote, transcript_path, created_at, modified_at,
        message_count, event_count, files_touched_count, token_count,
        parser_version, normalized_version, source_hash, model, tool_timeline, indexed_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(session_id) DO UPDATE SET
        cli_source = excluded.cli_source,
        first_prompt = COALESCE(excluded.first_prompt, first_prompt),
        project_path = excluded.project_path,
        git_branch = excluded.git_branch,
        git_remote = excluded.git_remote,
        transcript_path = excluded.transcript_path,
        created_at = COALESCE(excluded.created_at, created_at),
        modified_at = COALESCE(excluded.modified_at, modified_at),
        message_count = excluded.message_count,
        event_count = excluded.event_count,
        files_touched_count = excluded.files_touched_count,
        token_count = excluded.token_count,
        parser_version = excluded.parser_version,
        normalized_version = excluded.normalized_version,
        source_hash = excluded.source_hash,
        model = COALESCE(excluded.model, model),
        tool_timeline = COALESCE(excluded.tool_timeline, tool_timeline),
        indexed_at = CURRENT_TIMESTAMP
";

impl Database {
    /// Insert or update a session, preserving summary and title if not
    /// provided.
    pub fn upsert_session(&self, metadata: &SessionMetadata) -> DbResult<()> {
        let conn = self.connection()?;
        conn.execute(
            UPSERT_SQL,
            params![
                metadata.session_id,
                metadata.cli_source,
                metadata.title,
                metadata.first_prompt,
                metadata.project_path,
                metadata.git_branch,
                metadata.git_remote,
                metadata.transcript_path,
                metadata.created_at.as_ref().map(format_iso8601),
                metadata.modified_at.as_ref().map(format_iso8601),
                metadata.message_count,
                metadata.event_count,
                metadata.parser_version,
                metadata.normalized_version,
                metadata.source_hash,
                metadata.model,
                metadata.tool_timeline,
                local_now_iso_naive(),
            ],
        )?;
        Ok(())
    }

    /// Atomically insert/update a session AND replace its file references
    /// in one transaction.
    pub fn upsert_session_with_files(
        &self,
        metadata: &SessionMetadata,
        files: &[(String, String)],
    ) -> DbResult<()> {
        // BEGIN IMMEDIATE, not DEFERRED — see the note in
        // `search.rs::index_transcript`: under WAL a deferred transaction can
        // fail with a non-retryable SQLITE_BUSY_SNAPSHOT when a concurrent
        // writer commits between our snapshot and our first write.
        let mut conn = self.connection()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        tx.execute(
            UPSERT_WITH_FILES_SQL,
            params![
                metadata.session_id,
                metadata.cli_source,
                metadata.title,
                metadata.first_prompt,
                metadata.project_path,
                metadata.git_branch,
                metadata.git_remote,
                metadata.transcript_path,
                metadata.created_at.as_ref().map(format_iso8601),
                metadata.modified_at.as_ref().map(format_iso8601),
                metadata.message_count,
                metadata.event_count,
                metadata.files_touched_count,
                metadata.token_count,
                metadata.parser_version,
                metadata.normalized_version,
                metadata.source_hash,
                metadata.model,
                metadata.tool_timeline,
                local_now_iso_naive(),
            ],
        )?;
        tx.execute(
            "DELETE FROM session_files WHERE session_id = ?",
            params![metadata.session_id],
        )?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO session_files (session_id, file_path, operation) VALUES (?, ?, ?)",
            )?;
            for (file_path, operation) in files {
                stmt.execute(params![metadata.session_id, file_path, operation])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Update session title and summary from LLM summarization; stamps
    /// `summary_at` with UTC now.
    pub fn update_summary(&self, session_id: &str, title: &str, summary: &str) -> DbResult<()> {
        self.update_summary_opt(session_id, Some(title), Some(summary))
    }

    /// `update_summary` with nullability preserved: `None` binds SQL NULL,
    /// which is what a model emitting a present-but-null `title`/`summary`
    /// key must produce. A NULL summary leaves the session in the
    /// needs-summary queue (see `summarize::SessionSummary`), so the
    /// background indexer must not substitute `""`.
    pub fn update_summary_opt(
        &self,
        session_id: &str,
        title: Option<&str>,
        summary: Option<&str>,
    ) -> DbResult<()> {
        let conn = self.connection()?;
        conn.execute(
            "UPDATE sessions SET title = ?, summary = ?, summary_at = ? WHERE session_id = ?",
            params![title, summary, utc_now_iso(), session_id],
        )?;
        Ok(())
    }

    /// Update the context snapshot; stamps `snapshot_at` with UTC now.
    pub fn update_snapshot(&self, session_id: &str, snapshot: &str) -> DbResult<()> {
        let conn = self.connection()?;
        conn.execute(
            "UPDATE sessions SET context_snapshot = ?, snapshot_at = ? WHERE session_id = ?",
            params![snapshot, utc_now_iso(), session_id],
        )?;
        Ok(())
    }

    /// Update session keywords for FTS indexing. An empty list stores NULL
    /// rather than an empty JSON array.
    pub fn update_keywords(&self, session_id: &str, keywords: &[String]) -> DbResult<()> {
        let keywords_json = if keywords.is_empty() {
            None
        } else {
            let value = serde_json::Value::Array(
                keywords
                    .iter()
                    .map(|k| serde_json::Value::String(k.clone()))
                    .collect(),
            );
            Some(super::json_dumps(&value))
        };
        let conn = self.connection()?;
        conn.execute(
            "UPDATE sessions SET keywords = ? WHERE session_id = ?",
            params![keywords_json, session_id],
        )?;
        Ok(())
    }

    /// Remove a session and everything hanging off it: its files,
    /// transcript FTS row and any summary claim. `sessions_fts` follows via
    /// the `sessions_ad` trigger. Returns whether a row was deleted.
    pub fn delete_session(&self, session_id: &str) -> DbResult<bool> {
        let mut conn = self.connection()?;
        let tx = conn.transaction()?;
        tx.execute(
            "DELETE FROM session_files WHERE session_id = ?",
            params![session_id],
        )?;
        tx.execute(
            "DELETE FROM transcripts_fts WHERE session_id = ?",
            params![session_id],
        )?;
        tx.execute(
            "DELETE FROM summary_claims WHERE session_id = ?",
            params![session_id],
        )?;
        tx.execute(
            "DELETE FROM session_events WHERE session_id = ?",
            params![session_id],
        )?;
        let deleted = tx.execute(
            "DELETE FROM sessions WHERE session_id = ?",
            params![session_id],
        )?;
        tx.commit()?;
        Ok(deleted > 0)
    }

    /// Get a session by ID.
    pub fn get_session(&self, session_id: &str) -> DbResult<Option<SessionRow>> {
        let conn = self.connection()?;
        let mut stmt = conn.prepare("SELECT * FROM sessions WHERE session_id = ?")?;
        let mut rows = stmt.query_map(params![session_id], SessionRow::from_row)?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    /// Get sessions with optional filters.
    ///
    /// An empty `cli_source`/`project_path` filter is treated as absent, so
    /// `Some("")` behaves like `None`. `limit` applies whenever it is `Some`;
    /// an `offset` of 0 is skipped.
    pub fn get_sessions(
        &self,
        cli_source: Option<&str>,
        project_path: Option<&str>,
        limit: Option<i64>,
        offset: i64,
    ) -> DbResult<Vec<SessionRow>> {
        let mut query = String::from("SELECT * FROM sessions WHERE 1=1");
        let mut sql_params: Vec<SqlValue> = Vec::new();

        if let Some(cli) = cli_source.filter(|s| !s.is_empty()) {
            query.push_str(" AND cli_source = ?");
            sql_params.push(SqlValue::Text(cli.to_string()));
        }
        if let Some(pp) = project_path.filter(|s| !s.is_empty()) {
            query.push_str(" AND project_path LIKE ?");
            sql_params.push(SqlValue::Text(format!("{pp}%")));
        }
        query.push_str(" ORDER BY modified_at DESC");
        if let Some(l) = limit {
            query.push_str(" LIMIT ?");
            sql_params.push(SqlValue::Integer(l));
        }
        if offset != 0 {
            if limit.is_none() {
                query.push_str(" LIMIT -1"); // SQLite requires LIMIT before OFFSET
            }
            query.push_str(" OFFSET ?");
            sql_params.push(SqlValue::Integer(offset));
        }

        self.query_sessions(&query, &sql_params)
    }

    /// Sessions without summaries or with stale summaries. `Some(0)` for
    /// `limit` means no limit.
    pub fn get_sessions_needing_summary(
        &self,
        limit: Option<i64>,
        include_stale: bool,
    ) -> DbResult<Vec<SessionRow>> {
        let mut query = if include_stale {
            String::from(
                "SELECT * FROM sessions
                 WHERE transcript_path != ''
                 AND message_count > 0
                 AND (
                     summary IS NULL OR summary = ''
                     OR (summary_at IS NOT NULL AND REPLACE(modified_at, 'T', ' ') > REPLACE(summary_at, 'T', ' '))
                 )",
            )
        } else {
            String::from(
                "SELECT * FROM sessions
                 WHERE (summary IS NULL OR summary = '')
                 AND transcript_path != ''
                 AND message_count > 0",
            )
        };
        let mut sql_params: Vec<SqlValue> = Vec::new();
        if let Some(l) = limit.filter(|l| *l != 0) {
            query.push_str(" LIMIT ?");
            sql_params.push(SqlValue::Integer(l));
        }
        self.query_sessions(&query, &sql_params)
    }

    /// Sessions needing summaries whose transcripts have been stable for at
    /// least `min_age_hours`.
    pub fn get_sessions_needing_summary_stable(
        &self,
        min_age_hours: f64,
        limit: Option<i64>,
    ) -> DbResult<Vec<SessionRow>> {
        let mut query = String::from(
            "SELECT * FROM sessions
             WHERE transcript_path != ''
             AND message_count > 0
             AND (
                 summary IS NULL OR summary = ''
                 OR julianday(REPLACE(modified_at, 'T', ' '))
                    > julianday(REPLACE(summary_at, 'T', ' '))
             )
             AND REPLACE(modified_at, 'T', ' ') < datetime('now', ?)
             ORDER BY modified_at DESC",
        );
        let mut sql_params: Vec<SqlValue> = vec![SqlValue::Text(format!(
            "-{} hours",
            float_str(min_age_hours)
        ))];
        if let Some(l) = limit.filter(|l| *l != 0) {
            query.push_str(" LIMIT ?");
            sql_params.push(SqlValue::Integer(l));
        }
        self.query_sessions(&query, &sql_params)
    }

    /// Sessions without snapshots or with stale snapshots.
    pub fn get_sessions_needing_snapshot(
        &self,
        limit: Option<i64>,
        include_stale: bool,
        min_age_hours: Option<f64>,
        require_summary: bool,
    ) -> DbResult<Vec<SessionRow>> {
        let mut conditions: Vec<String> = vec![
            "transcript_path != ''".to_string(),
            "message_count > 0".to_string(),
        ];
        let mut sql_params: Vec<SqlValue> = Vec::new();

        if include_stale {
            conditions.push(
                "(
                    context_snapshot IS NULL OR context_snapshot = ''
                    OR (snapshot_at IS NOT NULL AND REPLACE(modified_at, 'T', ' ') > REPLACE(snapshot_at, 'T', ' '))
                )"
                .to_string(),
            );
        } else {
            conditions.push("(context_snapshot IS NULL OR context_snapshot = '')".to_string());
        }

        if require_summary {
            conditions.push("summary IS NOT NULL AND summary != ''".to_string());
        }

        if let Some(hours) = min_age_hours {
            conditions.push("REPLACE(modified_at, 'T', ' ') < datetime('now', ?)".to_string());
            sql_params.push(SqlValue::Text(format!("-{} hours", float_str(hours))));
        }

        let mut query = format!("SELECT * FROM sessions WHERE {}", conditions.join(" AND "));
        if min_age_hours.is_some() {
            query.push_str(" ORDER BY modified_at DESC");
        }
        if let Some(l) = limit.filter(|l| *l != 0) {
            query.push_str(" LIMIT ?");
            sql_params.push(SqlValue::Integer(l));
        }
        self.query_sessions(&query, &sql_params)
    }

    /// All sessions that have transcript files and actual messages.
    pub fn get_all_sessions_with_transcripts(
        &self,
        limit: Option<i64>,
    ) -> DbResult<Vec<SessionRow>> {
        let mut query = String::from(
            "SELECT * FROM sessions
             WHERE transcript_path != ''
             AND message_count > 0",
        );
        let mut sql_params: Vec<SqlValue> = Vec::new();
        if let Some(l) = limit.filter(|l| *l != 0) {
            query.push_str(" LIMIT ?");
            sql_params.push(SqlValue::Integer(l));
        }
        self.query_sessions(&query, &sql_params)
    }

    /// Sessions that have a non-empty summary.
    pub fn get_sessions_with_summaries(&self, limit: Option<i64>) -> DbResult<Vec<SessionRow>> {
        let mut query = String::from(
            "SELECT * FROM sessions
             WHERE summary IS NOT NULL AND summary != ''",
        );
        let mut sql_params: Vec<SqlValue> = Vec::new();
        if let Some(l) = limit.filter(|l| *l != 0) {
            query.push_str(" LIMIT ?");
            sql_params.push(SqlValue::Integer(l));
        }
        self.query_sessions(&query, &sql_params)
    }

    /// Add one file reference.
    pub fn add_file_reference(
        &self,
        session_id: &str,
        file_path: &str,
        operation: &str,
    ) -> DbResult<()> {
        let conn = self.connection()?;
        conn.execute(
            "INSERT INTO session_files (session_id, file_path, operation) VALUES (?, ?, ?)",
            params![session_id, file_path, operation],
        )?;
        Ok(())
    }

    /// Files touched by a session.
    pub fn get_session_files(&self, session_id: &str) -> DbResult<Vec<SessionFile>> {
        let conn = self.connection()?;
        let mut stmt =
            conn.prepare("SELECT file_path, operation FROM session_files WHERE session_id = ?")?;
        let rows = stmt.query_map(params![session_id], |row| {
            Ok(SessionFile {
                file_path: row.get(0)?,
                operation: row.get(1)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    /// Files for multiple sessions in one query. Every requested ID gets an
    /// entry, empty when the session has no files (or does not exist).
    ///
    /// The result is keyed by session id: **duplicate ids collapse to a
    /// single entry** carrying that session's files once, and entries come
    /// back in first-occurrence order of `session_ids`.
    pub fn get_files_for_sessions(
        &self,
        session_ids: &[String],
    ) -> DbResult<Vec<(String, Vec<SessionFile>)>> {
        if session_ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = vec!["?"; session_ids.len()].join(",");
        let query = format!(
            "SELECT session_id, file_path, operation
             FROM session_files
             WHERE session_id IN ({placeholders})
             ORDER BY session_id, file_path"
        );
        let conn = self.connection()?;
        let mut stmt = conn.prepare(&query)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(session_ids.iter()), |row| {
            Ok((
                row.get::<_, String>(0)?,
                SessionFile {
                    file_path: row.get(1)?,
                    operation: row.get(2)?,
                },
            ))
        })?;

        // Every requested id present exactly once, in first-occurrence
        // order.
        let mut index: HashMap<&str, usize> = HashMap::with_capacity(session_ids.len());
        let mut result: Vec<(String, Vec<SessionFile>)> = Vec::with_capacity(session_ids.len());
        for sid in session_ids {
            if !index.contains_key(sid.as_str()) {
                index.insert(sid.as_str(), result.len());
                result.push((sid.clone(), Vec::new()));
            }
        }
        for row in rows {
            let (sid, file) = row?;
            if let Some(&slot) = index.get(sid.as_str()) {
                result[slot].1.push(file);
            }
        }
        Ok(result)
    }

    /// Clear file references for a session.
    pub fn clear_session_files(&self, session_id: &str) -> DbResult<()> {
        let conn = self.connection()?;
        conn.execute(
            "DELETE FROM session_files WHERE session_id = ?",
            params![session_id],
        )?;
        Ok(())
    }

    /// Shared SELECT-into-`SessionRow` helper.
    pub(crate) fn query_sessions(
        &self,
        query: &str,
        sql_params: &[SqlValue],
    ) -> DbResult<Vec<SessionRow>> {
        let conn = self.connection()?;
        let mut stmt = conn.prepare(query)?;
        let rows = stmt.query_map(
            rusqlite::params_from_iter(sql_params.iter()),
            SessionRow::from_row,
        )?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }
}
