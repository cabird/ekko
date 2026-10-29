//! Tool parameter structs.
//!
//! One struct per tool. Doc comments become the JSON-Schema `description` of
//! each property, so they are what the model reads when deciding how to call
//! the tool; the defaults here are the documented ones.
//!
//! # Schema shape
//!
//! Every optional parameter is an `Option<T>` with `#[serde(default)]`, so a
//! client may omit it *or* send `null`. schemars would render that as
//! `"type": ["string", "null"]` — the shape that has been observed to make
//! Claude clients silently drop the whole tool list — so
//! [`crate::mcp::schema::sanitize`] rewrites those unions (and strips
//! integer `format`s) after the router is built. Deserialization is
//! unaffected: `null` still parses to `None`.

use schemars::JsonSchema;
use serde::Deserialize;

fn r#true() -> bool {
    true
}
fn seven() -> i64 {
    7
}
fn ten() -> i64 {
    10
}
fn twenty() -> i64 {
    20
}
fn fifty() -> i64 {
    50
}
fn five() -> i64 {
    5
}
fn three() -> i64 {
    3
}
fn keyword() -> String {
    "keyword".to_string()
}
fn info() -> String {
    "info".to_string()
}

/// `recent_activity` parameters.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct RecentActivityParams {
    /// Scope to sessions in/under this directory (e.g., current project)
    #[serde(default)]
    pub directory: Option<String>,
    /// Include sessions in subdirectories (default True)
    #[serde(default = "r#true")]
    pub include_subdirs: bool,
    /// Look back this many days (default 7)
    #[serde(default = "seven")]
    pub days: i64,
    /// Maximum sessions to include (default 10)
    #[serde(default = "ten")]
    pub max_sessions: i64,
}

/// `search_sessions` parameters.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct SearchSessionsParams {
    /// Search query (keywords, file paths, or natural language)
    pub query: String,
    /// Scope to sessions in/under this directory
    #[serde(default)]
    pub directory: Option<String>,
    /// Include sessions in subdirectories (default True)
    #[serde(default = "r#true")]
    pub include_subdirs: bool,
    /// Limit to sessions modified within N days
    #[serde(default)]
    pub days: Option<i64>,
    /// ISO 8601 start date (inclusive), e.g. "2026-02-01". Takes precedence
    /// over `days` when provided.
    #[serde(default)]
    pub start_date: Option<String>,
    /// ISO 8601 end date (inclusive), e.g. "2026-02-28". Takes precedence
    /// over `days` when provided.
    #[serde(default)]
    pub end_date: Option<String>,
    /// Search mode: "keyword" (default) full-text search on titles,
    /// summaries and keywords; "transcript" full-text search across all
    /// session transcripts; "semantic" [LLM] concept matching on summaries;
    /// "hybrid" cascades keyword → transcript → semantic until results are
    /// found.
    #[serde(default = "keyword")]
    pub mode: String,
    /// Maximum results (default 20)
    #[serde(default = "twenty")]
    pub limit: i64,
}

/// `get_session_snapshot` parameters.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct GetSessionSnapshotParams {
    /// The session ID (from search results or recent_activity)
    pub session_id: String,
}

/// `read_transcript` parameters.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct ReadTranscriptParams {
    /// The session ID
    pub session_id: String,
    /// Access mode (default "info"): "info", "head", "tail", "lines",
    /// "grep", "full" or "save".
    #[serde(default = "info")]
    pub mode: String,
    /// Number of messages for head/tail (default 5)
    #[serde(default = "five")]
    pub count: i64,
    /// Start line for lines mode (1-based, inclusive)
    #[serde(default)]
    pub start: Option<i64>,
    /// End line for lines mode (1-based, inclusive)
    #[serde(default)]
    pub end: Option<i64>,
    /// Text pattern for grep mode (case-insensitive)
    #[serde(default)]
    pub pattern: Option<String>,
    /// Lines of context before/after each grep match (default 3)
    #[serde(default = "three")]
    pub context_lines: i64,
    /// Maximum grep matches to return (default 20)
    #[serde(default = "twenty")]
    pub max_matches: i64,
    /// Include tool call messages in full/save/head/tail (default True)
    #[serde(default = "r#true")]
    pub include_tools: bool,
    /// Relative name for save mode (e.g. "my-export.txt"). Saved under
    /// `<data_dir>/exports/`. Absolute paths are rejected.
    #[serde(default)]
    pub output_path: Option<String>,
}

/// `ask_question` parameters.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct AskQuestionParams {
    /// Your question
    pub question: String,
    /// Scope to sessions in/under this directory
    #[serde(default)]
    pub directory: Option<String>,
    /// Include sessions in subdirectories (default True)
    #[serde(default = "r#true")]
    pub include_subdirs: bool,
    /// Limit to sessions modified within N days
    #[serde(default)]
    pub days: Option<i64>,
    /// Max transcripts to query if needed (default 7)
    #[serde(default = "seven")]
    pub max_sessions: i64,
    /// Skip first N sessions for pagination (default 0). Use this to search
    /// through more sessions if `pagination.has_more` is True.
    #[serde(default)]
    pub offset: i64,
}

/// `sessions_in_directory` parameters.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct SessionsInDirectoryParams {
    /// Directory path to search
    pub directory: String,
    /// Include sessions in subdirectories (default True)
    #[serde(default = "r#true")]
    pub include_subdirs: bool,
    /// Limit to sessions modified within N days
    #[serde(default)]
    pub days: Option<i64>,
    /// Maximum results (default 50)
    #[serde(default = "fifty")]
    pub limit: i64,
}

/// `export_telemetry` parameters.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct ExportTelemetryParams {
    /// Relative name for the export (e.g. "telemetry-2026-04"). Saved under
    /// `<data_dir>/exports/`. `.jsonl` extension is added if missing.
    /// Absolute paths and `..` traversal are rejected.
    pub filename: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn omitted_fields_take_the_documented_defaults() {
        let p: RecentActivityParams = serde_json::from_value(json!({})).unwrap();
        assert_eq!(p.directory, None);
        assert!(p.include_subdirs);
        assert_eq!(p.days, 7);
        assert_eq!(p.max_sessions, 10);

        let p: SearchSessionsParams = serde_json::from_value(json!({"query": "x"})).unwrap();
        assert_eq!(p.mode, "keyword");
        assert_eq!(p.limit, 20);
        assert_eq!(p.days, None);

        let p: ReadTranscriptParams = serde_json::from_value(json!({"session_id": "s"})).unwrap();
        assert_eq!(p.mode, "info");
        assert_eq!(p.count, 5);
        assert_eq!(p.context_lines, 3);
        assert_eq!(p.max_matches, 20);
        assert!(p.include_tools);

        let p: AskQuestionParams = serde_json::from_value(json!({"question": "q"})).unwrap();
        assert_eq!(p.max_sessions, 7);
        assert_eq!(p.offset, 0);

        let p: SessionsInDirectoryParams =
            serde_json::from_value(json!({"directory": "/x"})).unwrap();
        assert_eq!(p.limit, 50);
    }

    /// The sanitized schema no longer advertises `null`, but clients that
    /// send it anyway must not get a deserialization error.
    #[test]
    fn explicit_nulls_still_deserialize() {
        let p: SearchSessionsParams =
            serde_json::from_value(json!({"query": "x", "directory": null, "days": null})).unwrap();
        assert_eq!(p.directory, None);
        assert_eq!(p.days, None);
    }
}
