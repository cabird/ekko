//! Unified memory API over indexed sessions.
//!
//! [`EkkoMemory`] is the layer every MCP tool calls: tiered access to
//! session history (summary → snapshot → transcript), with directory and
//! time scoping applied consistently by [`EkkoMemory::filter_sessions`].
//!
//! Every public method returns a [`serde_json::Value`] in the exact shape the
//! client sees. `serde_json` is built with `preserve_order`, so key insertion
//! order is part of the contract; the MCP layer serializes these straight
//! through.
//!
//! # Surface
//!
//! - [`EkkoMemory::recent_activity`]
//! - [`EkkoMemory::search_sessions`] (keyword, transcript, semantic, hybrid)
//! - [`EkkoMemory::get_session_snapshot`]
//! - [`EkkoMemory::ask_question`] (summaries tier → transcript tier)
//! - [`EkkoMemory::read_transcript`] and all its mode handlers
//! - [`EkkoMemory::list_projects`] / [`EkkoMemory::sessions_in_directory`]
//!
//! # LLM availability
//!
//! Nothing here gates on the LLM being configured: every LLM call site is
//! wrapped, so a missing API key degrades to "no semantic results" /
//! "couldn't extract a useful answer" rather than failing the tool.
//! [`crate::llm::LlmClient`] construction failure is caught in the same
//! places. The user-facing "configure an API key" message is produced one
//! layer up by the MCP tools via [`crate::llm::check_llm_available`].
//!
//! # Behaviour worth knowing
//!
//! - **File-ish queries** get no special routing: every keyword query goes
//!   through FTS. Same call as `search.rs` — see its module docs.
//! - **`operations` ordering**: `recent_activity` emits a file's operations in
//!   first-seen order, which is deterministic and therefore testable.
//! - **Missing values never fail a query**: `title or first_prompt[:60]`
//!   yields `""` when both columns are SQL NULL, and a NULL `project_path`
//!   yields the `unknown` label rather than an error.
//! - **Transcript fan-out order**: see [`qa`].

use std::collections::HashSet;
use std::path::PathBuf;

use serde_json::{Map, Value, json};

use crate::db::{Database, DbResult, SessionRow};
use crate::models::truncate_chars;
use crate::utils::{is_path_under, parse_date_filter, parse_datetime, paths_equal, resolve_path};

pub mod activity;
pub mod qa;
pub mod search;
pub mod transcript;

#[cfg(test)]
mod tests;

pub use activity::RecentActivityArgs;
pub use qa::AskArgs;
pub use search::SearchArgs;
pub use transcript::ReadTranscriptArgs;

/// Unified interface to the session archive.
///
/// Owns the [`Database`] handle — cheap, since it is just a path with a
/// connection opened per call — and constructs a
/// [`crate::search::SearchEngine`] borrowing it where needed.
#[derive(Debug, Clone)]
pub struct EkkoMemory {
    db: Database,
}

/// The filter set shared by every scoped query.
#[derive(Debug, Clone, Default)]
pub struct SessionFilter<'a> {
    /// Scope to sessions in/under this directory.
    pub directory: Option<&'a str>,
    /// Include sessions in subdirectories (default: true).
    pub include_subdirs: bool,
    /// Look back this many days.
    pub days: Option<i64>,
    /// ISO 8601 start date (inclusive); takes precedence over `days`.
    pub start_date: Option<&'a str>,
    /// ISO 8601 end date (inclusive); takes precedence over `days`.
    pub end_date: Option<&'a str>,
    /// Restrict to one CLI source.
    pub cli_source: Option<&'a str>,
    /// Minimum message count (default: 1).
    pub min_messages: i64,
}

impl<'a> SessionFilter<'a> {
    /// The default filter: `include_subdirs = true`, `min_messages = 1`,
    /// everything else absent.
    pub fn new() -> Self {
        SessionFilter {
            include_subdirs: true,
            min_messages: 1,
            ..SessionFilter::default()
        }
    }

    /// Scope to a directory.
    pub fn directory(mut self, directory: Option<&'a str>, include_subdirs: bool) -> Self {
        self.directory = directory;
        self.include_subdirs = include_subdirs;
        self
    }

    /// Look back `days` days.
    pub fn days(mut self, days: Option<i64>) -> Self {
        self.days = days;
        self
    }

    /// Restrict to an explicit date range.
    pub fn dates(mut self, start_date: Option<&'a str>, end_date: Option<&'a str>) -> Self {
        self.start_date = start_date;
        self.end_date = end_date;
        self
    }
}

impl EkkoMemory {
    /// Wrap a database handle.
    pub fn new(db: Database) -> Self {
        EkkoMemory { db }
    }

    /// The underlying database handle.
    pub fn db(&self) -> &Database {
        &self.db
    }

    // =====================================================================
    // Filtering helpers
    // =====================================================================

    /// Apply the common filters to a session list.
    pub fn filter_sessions(
        &self,
        sessions: Vec<SessionRow>,
        f: &SessionFilter<'_>,
    ) -> Vec<SessionRow> {
        let state = FilterState::new(f);
        sessions.into_iter().filter(|s| state.matches(s)).collect()
    }

    // =====================================================================
    // Tier 2: snapshot-level operations
    // =====================================================================

    /// Context snapshot for a session.
    ///
    /// # Errors
    ///
    /// Propagates database errors. A missing session is *not* an error: it
    /// returns `{'error': 'Session not found: …'}`.
    pub fn get_session_snapshot(&self, session_id: &str) -> DbResult<Value> {
        let Some(session) = self.db.get_session(session_id)? else {
            return Ok(json!({ "error": format!("Session not found: {session_id}") }));
        };
        self.snapshot_payload(&session)
    }

    /// [`Self::get_session_snapshot`] for a row the caller already fetched
    /// (the web layer resolves ids first and must not re-query).
    pub fn snapshot_payload(&self, session: &crate::db::SessionRow) -> DbResult<Value> {
        let session_id = session.session_id.as_str();

        // The column always exists on a row, so a NULL stays absent and the
        // fence-stripper returns it unchanged.
        let snapshot = strip_markdown_fences(session.context_snapshot.as_deref());
        let has_snapshot = snapshot.as_deref().is_some_and(|s| !s.is_empty());
        let files = self.db.get_session_files(session_id)?;

        let mut out = Map::new();
        out.insert("session_id".into(), json!(session_id));
        out.insert("snapshot".into(), opt_str(snapshot.as_deref()));
        out.insert("has_snapshot".into(), json!(has_snapshot));
        out.insert(
            "metadata".into(),
            json!({
                "title": opt_str(session.title.as_deref()),
                "project_path": opt_str(session.project_path.as_deref()),
                "git_branch": opt_str(session.git_branch.as_deref()),
                "cli_source": json!(session.cli_source),
                "created_at": opt_str(session.created_at.as_deref()),
                "modified_at": opt_str(session.modified_at.as_deref()),
                "message_count": json!(session.message_count),
            }),
        );
        out.insert(
            "transcript_path".into(),
            opt_str(session.transcript_path.as_deref()),
        );
        out.insert("files_touched".into(), files_touched_json(&files));
        out.insert(
            "suggestion".into(),
            if has_snapshot {
                Value::Null
            } else {
                json!(
                    "No snapshot available. Use read_transcript() for full detail, \
or generate a snapshot via the web UI."
                )
            },
        );
        Ok(Value::Object(out))
    }

    // =====================================================================
    // Project-level operations
    // =====================================================================

    /// All projects with session counts.
    ///
    /// # Errors
    ///
    /// Propagates database errors.
    pub fn list_projects(&self) -> DbResult<Value> {
        let projects = self.db.get_projects()?;
        Ok(Value::Array(
            projects
                .into_iter()
                .map(|p| {
                    json!({
                        "project_path": p.project_path,
                        "cli_source": p.cli_source,
                        "session_count": p.session_count,
                        "last_activity": opt_str(p.last_activity.as_deref()),
                    })
                })
                .collect(),
        ))
    }

    /// Sessions in or under a directory.
    ///
    /// # Errors
    ///
    /// Propagates database errors.
    pub fn sessions_in_directory(
        &self,
        directory: &str,
        include_subdirs: bool,
        days: Option<i64>,
        limit: i64,
    ) -> DbResult<Value> {
        let sessions =
            self.db
                .get_sessions(None, Some(directory), Some(limit.saturating_mul(2)), 0)?;
        let sessions = self.filter_sessions(
            sessions,
            &SessionFilter::new()
                .directory(Some(directory), include_subdirs)
                .days(days),
        );

        Ok(Value::Array(
            sessions
                .into_iter()
                .take(limit.max(0) as usize)
                .map(|s| {
                    let summary = s.summary.as_deref().filter(|v| !v.is_empty());
                    json!({
                        "session_id": s.session_id,
                        "title": json!(title_or_first_prompt(&s)),
                        "summary": match summary {
                            Some(v) => json!(truncate_chars(v, 200)),
                            None => Value::Null,
                        },
                        "project_path": opt_str(s.project_path.as_deref()),
                        "git_branch": opt_str(s.git_branch.as_deref()),
                        "cli_source": json!(s.cli_source),
                        "modified_at": opt_str(s.modified_at.as_deref()),
                        "message_count": json!(s.message_count),
                        "has_snapshot": json!(
                            s.context_snapshot.as_deref().is_some_and(|v| !v.is_empty())
                        ),
                        "transcript_path": opt_str(s.transcript_path.as_deref()),
                    })
                })
                .collect(),
        ))
    }
}

/// A [`SessionFilter`] with its directory and date bounds resolved once, so
/// a list *or* a list of hit structs can be filtered without cloning rows.
pub(crate) struct FilterState<'a> {
    base_path: Option<PathBuf>,
    include_subdirs: bool,
    start_dt: Option<chrono::DateTime<chrono::FixedOffset>>,
    end_dt: Option<chrono::DateTime<chrono::FixedOffset>>,
    cli_source: Option<&'a str>,
    min_messages: i64,
}

impl<'a> FilterState<'a> {
    pub(crate) fn new(f: &SessionFilter<'a>) -> Self {
        let (start_dt, end_dt) = parse_date_filter(f.days, f.start_date, f.end_date);
        FilterState {
            base_path: f
                .directory
                .filter(|d| !d.is_empty())
                .map(|d| resolve_path(d).unwrap_or_else(|| PathBuf::from(d))),
            include_subdirs: f.include_subdirs,
            start_dt,
            end_dt,
            cli_source: f.cli_source.filter(|c| !c.is_empty()),
            min_messages: f.min_messages,
        }
    }

    /// Filter order: empty sessions → directory → time → CLI.
    pub(crate) fn matches(&self, s: &SessionRow) -> bool {
        if s.message_count < self.min_messages {
            return false;
        }
        if let Some(base) = &self.base_path {
            let project_path = s.project_path.as_deref().unwrap_or("");
            let ok = if self.include_subdirs {
                is_path_under(project_path, base)
            } else {
                paths_equal(project_path, base)
            };
            if !ok {
                return false;
            }
        }
        if self.start_dt.is_some() || self.end_dt.is_some() {
            // `modified_at` if set, else `created_at`, else unset.
            let raw = s
                .modified_at
                .as_deref()
                .filter(|v| !v.is_empty())
                .or_else(|| s.created_at.as_deref().filter(|v| !v.is_empty()))
                .unwrap_or("");
            let Some(dt) = parse_datetime(raw) else {
                return false;
            };
            if let Some(start) = self.start_dt
                && dt < start
            {
                return false;
            }
            if let Some(end) = self.end_dt
                && dt > end
            {
                return false;
            }
        }
        if let Some(cli) = self.cli_source
            && s.cli_source != cli
        {
            return false;
        }
        true
    }
}

// =========================================================================
// Shared formatting helpers
// =========================================================================

/// `Option<&str>` → JSON string, or `null` for a NULL column.
pub(crate) fn opt_str(v: Option<&str>) -> Value {
    match v {
        Some(s) => Value::String(s.to_string()),
        None => Value::Null,
    }
}

/// The session title, falling back to the first 60 characters of
/// `first_prompt`.
///
/// When both columns are NULL this yields `""` rather than failing.
pub(crate) fn title_or_first_prompt(s: &SessionRow) -> String {
    match s.title.as_deref().filter(|t| !t.is_empty()) {
        Some(t) => t.to_string(),
        None => truncate_chars(s.first_prompt.as_deref().unwrap_or(""), 60).to_string(),
    }
}

/// `[{'path': …, 'operation': …}]` for a session's files.
pub(crate) fn files_touched_json(files: &[crate::db::SessionFile]) -> Value {
    Value::Array(
        files
            .iter()
            .map(|f| json!({"path": f.file_path, "operation": f.operation}))
            .collect(),
    )
}

/// Strip markdown code fences from LLM output. `None`/empty pass through
/// untouched.
pub fn strip_markdown_fences(text: Option<&str>) -> Option<String> {
    let text = text?;
    if text.is_empty() {
        return Some(String::new());
    }
    let mut text = crate::parsers::strip_ws(text).to_string();

    if text.starts_with("```") {
        // Drop the opening fence line.
        if let Some(nl) = text.find('\n') {
            text = text[nl + 1..].to_string();
        }
        // Drop a trailing fence.
        let trimmed = rstrip_ws(&text);
        if let Some(cut) = trimmed.strip_suffix("```") {
            text = rstrip_ws(cut).to_string();
        }
    }
    Some(text)
}

use crate::utils::rstrip_ws;

/// Extra per-tier fields a search hit can carry into
/// [`format_session_result`].
#[derive(Debug, Clone, Default)]
pub struct ResultExtras {
    /// Model relevance score (semantic tier only).
    pub score: Option<Value>,
    /// Model rationale (semantic tier only).
    pub relevance: Option<String>,
    /// FTS snippet (transcript tier only). Emitted only when truthy.
    pub excerpt: Option<String>,
}

/// Format a session for search results.
///
/// `excerpt` is present only when non-empty.
pub fn format_session_result(session: &SessionRow, extras: &ResultExtras) -> Value {
    let mut out = Map::new();
    out.insert("session_id".into(), json!(session.session_id));
    out.insert("title".into(), json!(title_or_first_prompt(session)));
    out.insert(
        "summary".into(),
        json!(truncate_chars(
            session.summary.as_deref().unwrap_or(""),
            300
        )),
    );
    out.insert(
        "project_path".into(),
        opt_str(session.project_path.as_deref()),
    );
    out.insert("git_branch".into(), opt_str(session.git_branch.as_deref()));
    out.insert("cli_source".into(), json!(session.cli_source));
    out.insert(
        "modified_at".into(),
        opt_str(session.modified_at.as_deref()),
    );
    out.insert("message_count".into(), json!(session.message_count));
    out.insert("score".into(), extras.score.clone().unwrap_or(Value::Null));
    out.insert("relevance".into(), opt_str(extras.relevance.as_deref()));
    out.insert(
        "transcript_path".into(),
        opt_str(session.transcript_path.as_deref()),
    );
    if let Some(excerpt) = extras.excerpt.as_deref().filter(|e| !e.is_empty()) {
        out.insert("excerpt".into(), json!(excerpt));
    }
    Value::Object(out)
}

/// Format a session as a compact tagged block for LLM context.
///
/// ```text
/// [[S_AB12]] myproject | Fix auth redirect bug
/// dir:/home/user/myproject | cli:claude_code | branch:main | msgs:23 | modified:2024-01-15
/// Outcome: Fixed redirect loops caused by...
/// ```
pub fn session_to_summary_line(
    session: &SessionRow,
    tag: &str,
    include_full_summary: bool,
) -> String {
    let project_path = session.project_path.as_deref().unwrap_or("");
    // Last path segment, or `unknown` when there is no project path.
    let project_name = if project_path.is_empty() {
        "unknown"
    } else {
        project_path.rsplit('/').next().unwrap_or(project_path)
    };
    let title = match session.title.as_deref().filter(|t| !t.is_empty()) {
        Some(t) => t.to_string(),
        None => {
            let fp = truncate_chars(session.first_prompt.as_deref().unwrap_or(""), 60);
            if fp.is_empty() {
                "Untitled".to_string()
            } else {
                fp.to_string()
            }
        }
    };
    let modified: String = match session.modified_at.as_deref().filter(|m| !m.is_empty()) {
        Some(m) => truncate_chars(m, 10).to_string(),
        None => String::new(),
    };

    let mut line = format!("[[{tag}]] {project_name} | {title}");

    let mut meta_parts: Vec<String> = Vec::new();
    if !project_path.is_empty() {
        meta_parts.push(format!("dir:{project_path}"));
    }
    if !session.cli_source.is_empty() {
        meta_parts.push(format!("cli:{}", session.cli_source));
    }
    if let Some(branch) = session.git_branch.as_deref().filter(|b| !b.is_empty()) {
        meta_parts.push(format!("branch:{branch}"));
    }
    meta_parts.push(format!("msgs:{}", session.message_count));
    if !modified.is_empty() {
        meta_parts.push(format!("modified:{modified}"));
    }
    line.push('\n');
    line.push_str(&meta_parts.join(" | "));

    if include_full_summary
        && let Some(summary) = session.summary.as_deref().filter(|s| !s.is_empty())
    {
        line.push('\n');
        line.push_str(summary);
    }
    line
}

/// Tag every session with a unique `[[S_XX00]]` tag and render the block the
/// summary-level prompts consume. Returns the tag→session mapping alongside.
pub(crate) fn build_tagged_summaries(
    sessions: &[SessionRow],
) -> (Vec<(String, &SessionRow)>, String) {
    let mut used: HashSet<String> = HashSet::new();
    let mut tagged: Vec<(String, &SessionRow)> = Vec::with_capacity(sessions.len());
    let mut lines: Vec<String> = Vec::with_capacity(sessions.len());
    for s in sessions {
        let tag = crate::utils::generate_unique_tag(&used);
        used.insert(tag.clone());
        lines.push(session_to_summary_line(s, &tag, true));
        tagged.push((tag, s));
    }
    (tagged, lines.join("\n"))
}

/// Last segment of `project_path`, or `unknown` — the source label the Q&A
/// prompts and sources use.
pub(crate) fn project_label(project_path: Option<&str>) -> String {
    let path = project_path.unwrap_or("");
    let last = path.rsplit('/').next().unwrap_or("");
    if last.is_empty() {
        "unknown".to_string()
    } else {
        last.to_string()
    }
}
