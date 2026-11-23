//! Database statistics and project listings.

use rusqlite::params;

use super::{Database, DbResult};

/// Result of [`Database::get_stats`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DbStats {
    /// Non-empty sessions only (`message_count > 0`).
    pub total_sessions: i64,
    /// All rows in the sessions table.
    pub total_indexed: i64,
    pub with_summary: i64,
    pub with_snapshot: i64,
    pub with_transcript: i64,
    pub file_references: i64,
    /// `(cli_source, count)` for non-empty sessions, in GROUP BY order.
    pub by_cli: Vec<(String, i64)>,
}

/// One row of [`Database::get_projects`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectRow {
    pub project_path: String,
    pub cli_source: String,
    pub session_count: i64,
    pub last_activity: Option<String>,
}

/// Result of [`Database::get_summary_queue_stats`]: how the LLM queue
/// stands right now, for `ekko check` and `get_status`. Distinguishes the
/// backlog that means "the indexer is behind" (`eligible_now`) from the
/// normal steady state (`awaiting_quiet`, `empty`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SummaryQueueStats {
    /// Sessions the summarize queue would hand out right now: needing a
    /// summary and quiet for at least the stability window. Nonzero for
    /// long means the indexer is not keeping up.
    pub eligible_now: i64,
    /// Needing a summary but modified inside the stability window —
    /// probably still being written. Normal steady state.
    pub awaiting_quiet: i64,
    /// Rows with no messages. Never summarized, by design.
    pub empty: i64,
    /// Most recent summary write (UTC ISO-8601), if any.
    pub last_summary_at: Option<String>,
    /// Most recent snapshot write (UTC ISO-8601), if any.
    pub last_snapshot_at: Option<String>,
}

/// Result of [`Database::get_transcript_index_stats`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptIndexStats {
    pub indexed: i64,
    pub total_with_transcripts: i64,
    /// `"indexed/total"`, or `"0/0"` when there are no transcripts.
    pub coverage: String,
}

impl Database {
    /// List of projects with session counts, most recently active first.
    pub fn get_projects(&self) -> DbResult<Vec<ProjectRow>> {
        let conn = self.connection()?;
        let mut stmt = conn.prepare(
            "SELECT
                 project_path,
                 cli_source,
                 COUNT(*) as session_count,
                 MAX(modified_at) as last_activity
             FROM sessions
             WHERE project_path IS NOT NULL AND project_path != ''
             GROUP BY project_path, cli_source
             ORDER BY last_activity DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(ProjectRow {
                project_path: row.get(0)?,
                cli_source: row.get(1)?,
                session_count: row.get(2)?,
                last_activity: row.get(3)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    /// Database statistics.
    pub fn get_stats(&self) -> DbResult<DbStats> {
        let conn = self.connection()?;
        let count = |sql: &str| -> DbResult<i64> {
            conn.query_row(sql, [], |row| row.get(0))
                .map_err(Into::into)
        };
        let total = count("SELECT COUNT(*) FROM sessions")?;
        let non_empty = count("SELECT COUNT(*) FROM sessions WHERE message_count > 0")?;
        let with_summary =
            count("SELECT COUNT(*) FROM sessions WHERE summary IS NOT NULL AND summary != ''")?;
        let with_transcript = count("SELECT COUNT(*) FROM sessions WHERE transcript_path != ''")?;
        let with_snapshot = count(
            "SELECT COUNT(*) FROM sessions WHERE context_snapshot IS NOT NULL AND context_snapshot != ''",
        )?;
        let file_refs = count("SELECT COUNT(*) FROM session_files")?;

        let mut stmt = conn.prepare(
            "SELECT cli_source, COUNT(*) as count
             FROM sessions
             WHERE message_count > 0
             GROUP BY cli_source",
        )?;
        let by_cli = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        Ok(DbStats {
            total_sessions: non_empty, // Only count non-empty sessions
            total_indexed: total,
            with_summary,
            with_snapshot,
            with_transcript,
            file_references: file_refs,
            by_cli,
        })
    }

    /// Statistics about the transcript FTS index.
    /// Summary-queue standing (see [`SummaryQueueStats`]). The pending
    /// predicate and the stability cutoff are the same shapes
    /// [`Database::get_sessions_needing_summary_stable`] uses, so these
    /// counts agree with what the queue would actually hand out.
    pub fn get_summary_queue_stats(&self, min_age_hours: f64) -> DbResult<SummaryQueueStats> {
        let conn = self.connection()?;
        const PENDING: &str = "transcript_path != ''
             AND message_count > 0
             AND (
                 summary IS NULL OR summary = ''
                 OR (summary_at IS NOT NULL AND julianday(REPLACE(modified_at, 'T', ' '))
                    > julianday(REPLACE(summary_at, 'T', ' ')))
             )";
        let cutoff = format!("-{} hours", super::float_str(min_age_hours));
        let eligible_now: i64 = conn.query_row(
            &format!(
                "SELECT COUNT(*) FROM sessions WHERE {PENDING}
                 AND REPLACE(modified_at, 'T', ' ') < datetime('now', ?)"
            ),
            params![cutoff],
            |row| row.get(0),
        )?;
        let pending_total: i64 = conn.query_row(
            &format!("SELECT COUNT(*) FROM sessions WHERE {PENDING}"),
            [],
            |row| row.get(0),
        )?;
        let empty: i64 = conn.query_row(
            "SELECT COUNT(*) FROM sessions WHERE message_count <= 0",
            [],
            |row| row.get(0),
        )?;
        let (last_summary_at, last_snapshot_at): (Option<String>, Option<String>) = conn
            .query_row(
                "SELECT MAX(summary_at), MAX(snapshot_at) FROM sessions",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?;
        Ok(SummaryQueueStats {
            eligible_now,
            awaiting_quiet: pending_total - eligible_now,
            empty,
            last_summary_at,
            last_snapshot_at,
        })
    }

    pub fn get_transcript_index_stats(&self) -> DbResult<TranscriptIndexStats> {
        let conn = self.connection()?;
        let indexed: i64 =
            conn.query_row("SELECT COUNT(*) FROM transcripts_fts", params![], |row| {
                row.get(0)
            })?;
        let total: i64 = conn.query_row(
            "SELECT COUNT(*) FROM sessions WHERE transcript_path != '' AND message_count > 0",
            params![],
            |row| row.get(0),
        )?;
        Ok(TranscriptIndexStats {
            indexed,
            total_with_transcripts: total,
            coverage: if total > 0 {
                format!("{indexed}/{total}")
            } else {
                "0/0".to_string()
            },
        })
    }
}
