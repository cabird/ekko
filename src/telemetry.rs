//! Telemetry for the ekko MCP server.
//!
//! Records anonymous usage data for understanding feature usage patterns.
//! Data is stored locally and only shared when the user explicitly exports.
//!
//! # PRIVACY CONTRACT
//!
//! - Records: operation names, timestamps, durations
//! - Records: instance_id (random GUID per server start, for correlating
//!   events)
//! - Records: enum/boolean/numeric arguments (mode, limit, days,
//!   include_subdirs)
//! - Does **NOT** record: queries, file paths, directories, session content,
//!   any free text
//! - Data stays local until the user runs `export_telemetry`
//!
//! The allowlist is the enforcement mechanism: [`filter_safe_args`] drops
//! every argument that is not named in [`SAFE_ARGS`] for that operation, and
//! every string that is not one of the [`ENUM_VALUES`] for its key.

use std::path::PathBuf;
use std::time::Instant;

use serde_json::{Map, Value, json};

use crate::db::{Database, LlmUsage, TelemetryEvent, TelemetrySummary};
use crate::paths;

/// Arguments that are safe to record (enums, booleans, numbers), per
/// operation. Entries for operations ekko no longer exposes
/// (`ask_session`, `ask_sessions`, `find_sessions_by_file`,
/// `get_file_history`) are kept so that old telemetry rows and any
/// reintroduced tool are still filtered rather than silently unguarded.
pub const SAFE_ARGS: &[(&str, &[&str])] = &[
    (
        "search_sessions",
        &["mode", "include_subdirs", "days", "limit"],
    ),
    ("recent_activity", &["include_subdirs", "days", "max_sessions"]),
    (
        "find_sessions_by_file",
        &["operation", "include_subdirs", "days", "limit"],
    ),
    // session_id is not safe
    ("get_session_snapshot", &[]),
    (
        "read_transcript",
        &["mode", "context_lines", "max_matches", "include_tools"],
    ),
    // question is not safe
    ("ask_session", &[]),
    ("ask_sessions", &[]),
    (
        "ask_question",
        &["include_subdirs", "days", "max_sessions", "offset"],
    ),
    // file_path is not safe
    ("get_file_history", &["include_subdirs", "days", "limit"]),
    ("list_projects", &[]),
    // directory is not safe
    ("sessions_in_directory", &["include_subdirs", "days", "limit"]),
    ("telemetry_export", &["days", "format"]),
];

/// Argument values that indicate enums (for validation).
pub const ENUM_VALUES: &[(&str, &[&str])] = &[
    ("mode", &["keyword", "semantic", "transcript", "hybrid"]),
    ("operation", &["read", "write", "edit"]),
    ("format", &["json", "jsonl", "summary"]),
];

fn safe_arg_names(operation: &str) -> &'static [&'static str] {
    SAFE_ARGS
        .iter()
        .find(|(op, _)| *op == operation)
        .map(|(_, names)| *names)
        .unwrap_or(&[])
}

fn enum_values(key: &str) -> Option<&'static [&'static str]> {
    ENUM_VALUES
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, values)| *values)
}

/// Filter arguments down to the ones safe to record.
///
/// Safe: booleans, numbers, and strings that are known enum values for their
/// key. Not safe: arbitrary strings (queries, paths, session ids), arrays and
/// objects. Returns `None` when nothing survives, so an event with no safe
/// arguments records no args column at all rather than an empty map.
pub fn filter_safe_args(operation: &str, args: &Map<String, Value>) -> Option<Map<String, Value>> {
    let allowed = safe_arg_names(operation);
    let mut filtered = Map::new();

    for (key, value) in args {
        if !allowed.contains(&key.as_str()) {
            continue;
        }
        match value {
            // Always safe: booleans and numbers.
            Value::Bool(_) | Value::Number(_) => {
                filtered.insert(key.clone(), value.clone());
            }
            // Strings only if they're known enums; unknown strings skipped.
            Value::String(s)
                if enum_values(key).is_some_and(|vals| vals.contains(&s.as_str())) =>
            {
                filtered.insert(key.clone(), value.clone());
            }
            // Nothing else is recordable: null, arrays and objects all fall
            // through unrecorded.
            _ => {}
        }
    }

    if filtered.is_empty() {
        None
    } else {
        Some(filtered)
    }
}

/// Unpack an LLM usage object, falling back from `thinking_tokens` to
/// `reasoning_tokens` when the former is missing, null or zero.
pub fn usage_from_json(usage: &Value) -> LlmUsage {
    let get = |key: &str| usage.get(key).filter(|v| !v.is_null());
    let as_int = |v: Option<&Value>| v.and_then(Value::as_i64);
    let thinking = as_int(get("thinking_tokens"))
        .filter(|t| *t != 0)
        .or_else(|| as_int(get("reasoning_tokens")));
    LlmUsage {
        model: get("model").and_then(Value::as_str).map(str::to_string),
        input_tokens: as_int(get("input_tokens")),
        output_tokens: as_int(get("output_tokens")),
        thinking_tokens: thinking,
        cost: get("cost").and_then(Value::as_f64),
    }
}

/// Result of [`TelemetryRecorder::export_to_file`].
#[derive(Debug, Clone, PartialEq)]
pub struct TelemetryExport {
    pub file_path: PathBuf,
    pub event_count: usize,
    pub summary: TelemetrySummary,
}

/// Records telemetry events for MCP tool calls.
///
/// ```no_run
/// # use ekko::db::Database;
/// # use ekko::telemetry::TelemetryRecorder;
/// # fn demo(db: Database) {
/// let recorder = TelemetryRecorder::new(db);
///
/// // Manual recording
/// recorder.record("search_sessions", Some(45), None, None);
///
/// // Timed
/// let t = recorder.track("list_projects");
/// // … do the work …
/// t.finish();
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct TelemetryRecorder {
    db: Database,
    instance_id: String,
}

impl TelemetryRecorder {
    /// Create a recorder with a fresh random instance id: 32 lowercase hex
    /// characters.
    pub fn new(db: Database) -> Self {
        let mut instance_id = String::with_capacity(32);
        for _ in 0..16 {
            let byte: u8 = rand::random();
            use std::fmt::Write as _;
            let _ = write!(instance_id, "{byte:02x}");
        }
        TelemetryRecorder { db, instance_id }
    }

    /// Create a recorder with a caller-supplied instance id.
    pub fn with_instance_id(db: Database, instance_id: impl Into<String>) -> Self {
        TelemetryRecorder {
            db,
            instance_id: instance_id.into(),
        }
    }

    /// The instance id stamped on every event (`get_status` reports costs for
    /// this id, not the background indexer's).
    pub fn instance_id(&self) -> &str {
        &self.instance_id
    }

    /// The database this recorder writes to.
    pub fn db(&self) -> &Database {
        &self.db
    }

    /// Record a telemetry event. Failures are logged and swallowed —
    /// telemetry must never break the actual operation.
    pub fn record(
        &self,
        operation: &str,
        duration_ms: Option<i64>,
        args: Option<&Map<String, Value>>,
        usage: Option<&LlmUsage>,
    ) {
        let safe_args = args.and_then(|a| filter_safe_args(operation, a));
        let usage = usage.cloned().unwrap_or_default();
        if let Err(e) = self.db.record_telemetry(
            &self.instance_id,
            operation,
            duration_ms,
            safe_args.as_ref(),
            &usage,
        ) {
            tracing::warn!(error = %e, "Failed to record telemetry");
        }
    }

    /// Start timing an operation. Call [`Track::finish`] (or
    /// [`Track::finish_with`]) when it completes.
    pub fn track<'a>(&'a self, operation: &str) -> Track<'a> {
        Track {
            recorder: self,
            operation: operation.to_string(),
            start: Instant::now(),
        }
    }

    /// Export telemetry to a JSONL file under `<data_dir>/exports/`.
    ///
    /// `filename` must be a relative name (`.jsonl` is appended when
    /// missing); absolute paths and `..` traversal are rejected by the export
    /// sandbox. The first line is a privacy header, then one JSON object per
    /// event.
    pub fn export_to_file(&self, filename: &str) -> Result<TelemetryExport, String> {
        let filename = if filename.ends_with(".jsonl") {
            filename.to_string()
        } else {
            format!("{filename}.jsonl")
        };

        let out = paths::resolve_export_path(&filename).map_err(|e| e.to_string())?;

        let events = self.db.export_telemetry(None).map_err(|e| e.to_string())?;
        let summary = self
            .db
            .get_telemetry_summary(None)
            .map_err(|e| e.to_string())?;

        std::fs::write(&out, export_body(&events, &summary)).map_err(|e| e.to_string())?;

        Ok(TelemetryExport {
            file_path: out,
            event_count: events.len(),
            summary,
        })
    }
}

/// A timed operation. Dropping one without finishing records nothing, so an
/// early return or a panic never lands a bogus event.
#[derive(Debug)]
pub struct Track<'a> {
    recorder: &'a TelemetryRecorder,
    operation: String,
    start: Instant,
}

impl Track<'_> {
    /// Milliseconds since the operation started, truncated to an integer.
    pub fn elapsed_ms(&self) -> i64 {
        self.start.elapsed().as_millis() as i64
    }

    /// Record the event with no arguments and no LLM usage.
    pub fn finish(self) {
        self.finish_with(None, None);
    }

    /// Record the event with the tool's arguments (filtered by the
    /// allowlist) and optional LLM usage.
    pub fn finish_with(self, args: Option<&Map<String, Value>>, usage: Option<&LlmUsage>) {
        let duration_ms = self.elapsed_ms();
        self.recorder
            .record(&self.operation, Some(duration_ms), args, usage);
    }
}

/// The full JSONL export body: the privacy header, then one line per event
/// (separated from the file write so it can be asserted on directly).
pub fn export_body(events: &[TelemetryEvent], summary: &TelemetrySummary) -> String {
    let mut body = String::new();
    body.push_str(&crate::db::json_dumps(&export_header(summary)));
    body.push('\n');
    for event in events {
        body.push_str(&crate::db::json_dumps(&event_to_json(event)));
        body.push('\n');
    }
    body
}

/// The privacy header written as the first line of an export.
fn export_header(summary: &TelemetrySummary) -> Value {
    json!({
        "_comment": "TELEMETRY EXPORT - Please review before sharing",
        "_privacy": {
            "contains": [
                "operation names",
                "timestamps",
                "durations",
                "instance IDs (random per server start)",
                "safe arguments (booleans, numbers, enums)"
            ],
            "does_not_contain": [
                "search queries",
                "file paths",
                "session content",
                "any free-form text"
            ]
        },
        "_summary": summary_to_json(summary),
    })
}

/// A telemetry row as `dict(sqlite3.Row)` renders it: column order, with
/// `safe_args` already parsed back into an object.
fn event_to_json(event: &TelemetryEvent) -> Value {
    json!({
        "id": event.id,
        "instance_id": event.instance_id,
        "timestamp": event.timestamp,
        "operation": event.operation,
        "duration_ms": event.duration_ms,
        "safe_args": event.safe_args,
        "model": event.model,
        "input_tokens": event.input_tokens,
        "output_tokens": event.output_tokens,
        "thinking_tokens": event.thinking_tokens,
        "cost": event.cost,
    })
}

/// The summary dict from `database.get_telemetry_summary`.
pub fn summary_to_json(summary: &TelemetrySummary) -> Value {
    let mut operations = Map::new();
    for (name, stats) in &summary.operations {
        operations.insert(
            name.clone(),
            json!({
                "count": stats.count,
                "avg_duration_ms": stats.avg_duration_ms,
                "min_duration_ms": stats.min_duration_ms,
                "max_duration_ms": stats.max_duration_ms,
            }),
        );
    }
    json!({
        "total_events": summary.total_events,
        "first_event": summary.first_event,
        "last_event": summary.last_event,
        "operations": Value::Object(operations),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn args(pairs: &[(&str, Value)]) -> Map<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    // ── filter_safe_args ───────────────────────────────────────────────

    #[test]
    fn allowlisted_numbers_and_booleans_pass() {
        let filtered = filter_safe_args(
            "search_sessions",
            &args(&[
                ("limit", json!(10)),
                ("days", json!(7)),
                ("include_subdirs", json!(true)),
            ]),
        )
        .unwrap();
        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered["limit"], json!(10));
        assert_eq!(filtered["include_subdirs"], json!(true));
    }

    /// The whole point of the privacy contract: free text never lands in the
    /// database, whether or not its key is on the allowlist.
    #[test]
    fn free_text_is_always_dropped() {
        // `query` is not on any allowlist.
        assert_eq!(
            filter_safe_args("search_sessions", &args(&[("query", json!("secret terms"))])),
            None
        );
        // `session_id` is not safe for get_session_snapshot (empty allowlist).
        assert_eq!(
            filter_safe_args("get_session_snapshot", &args(&[("session_id", json!("abc"))])),
            None
        );
        // `directory` is not safe even though the operation has an allowlist.
        let filtered = filter_safe_args(
            "sessions_in_directory",
            &args(&[("directory", json!("/home/me/secret")), ("days", json!(3))]),
        )
        .unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered["days"], json!(3));
    }

    #[test]
    fn enum_strings_are_validated() {
        // Known enum value passes.
        let filtered =
            filter_safe_args("search_sessions", &args(&[("mode", json!("hybrid"))])).unwrap();
        assert_eq!(filtered["mode"], json!("hybrid"));
        // Unknown value for an enum key is dropped.
        assert_eq!(
            filter_safe_args("search_sessions", &args(&[("mode", json!("sneaky query"))])),
            None
        );
        // Allowlisted key with no enum table never accepts strings.
        assert_eq!(
            filter_safe_args("search_sessions", &args(&[("limit", json!("10"))])),
            None
        );
        // `operation` and `format` have their own tables.
        assert!(
            filter_safe_args("find_sessions_by_file", &args(&[("operation", json!("write"))]))
                .is_some()
        );
        assert!(
            filter_safe_args("find_sessions_by_file", &args(&[("operation", json!("delete"))]))
                .is_none()
        );
        assert!(filter_safe_args("telemetry_export", &args(&[("format", json!("jsonl"))])).is_some());
    }

    #[test]
    fn nulls_arrays_and_objects_are_dropped() {
        assert_eq!(
            filter_safe_args(
                "search_sessions",
                &args(&[
                    ("limit", json!(null)),
                    ("days", json!([1, 2])),
                    ("mode", json!({"a": 1}))
                ])
            ),
            None
        );
    }

    #[test]
    fn unknown_operation_records_nothing() {
        assert_eq!(
            filter_safe_args("not_a_tool", &args(&[("limit", json!(5))])),
            None
        );
    }

    /// The allowlists are the privacy contract: pin their exact contents so a
    /// new argument cannot be recorded without this test being updated.
    #[test]
    fn allowlists_have_the_expected_shape() {
        assert_eq!(SAFE_ARGS.len(), 12);
        assert_eq!(safe_arg_names("recent_activity"), ["include_subdirs", "days", "max_sessions"]);
        assert_eq!(safe_arg_names("read_transcript"), ["mode", "context_lines", "max_matches", "include_tools"]);
        assert_eq!(safe_arg_names("ask_question"), ["include_subdirs", "days", "max_sessions", "offset"]);
        assert!(safe_arg_names("list_projects").is_empty());
        assert!(safe_arg_names("ask_session").is_empty());
        assert!(safe_arg_names("ask_sessions").is_empty());
        assert_eq!(ENUM_VALUES.len(), 3);
        assert_eq!(
            enum_values("mode").unwrap(),
            ["keyword", "semantic", "transcript", "hybrid"]
        );
        assert_eq!(enum_values("operation").unwrap(), ["read", "write", "edit"]);
        assert_eq!(enum_values("format").unwrap(), ["json", "jsonl", "summary"]);
    }

    // ── usage ──────────────────────────────────────────────────────────

    #[test]
    fn usage_falls_back_from_thinking_to_reasoning_tokens() {
        let u = usage_from_json(&json!({
            "model": "gpt-5-mini", "input_tokens": 100, "output_tokens": 20,
            "reasoning_tokens": 7, "cost": 0.0012
        }));
        assert_eq!(u.model.as_deref(), Some("gpt-5-mini"));
        assert_eq!(u.thinking_tokens, Some(7));
        assert_eq!(u.cost, Some(0.0012));

        // A present-but-zero thinking count still falls back to reasoning.
        let u = usage_from_json(&json!({"thinking_tokens": 0, "reasoning_tokens": 5}));
        assert_eq!(u.thinking_tokens, Some(5));
        let u = usage_from_json(&json!({"thinking_tokens": 3, "reasoning_tokens": 5}));
        assert_eq!(u.thinking_tokens, Some(3));
        let u = usage_from_json(&json!({}));
        assert_eq!(u.thinking_tokens, None);
        assert_eq!(u.model, None);
    }

    // ── recording / export ─────────────────────────────────────────────

    fn recorder() -> (TelemetryRecorder, TempDir) {
        let dir = TempDir::new().unwrap();
        let db = Database::new(dir.path().join("sessions.db")).unwrap();
        (TelemetryRecorder::with_instance_id(db, "inst-1"), dir)
    }

    #[test]
    fn record_stores_filtered_args_and_usage() {
        let (rec, _dir) = recorder();
        rec.record(
            "search_sessions",
            Some(45),
            Some(&args(&[
                ("mode", json!("keyword")),
                ("query", json!("do not store me")),
            ])),
            Some(&LlmUsage {
                model: Some("gpt-5-mini".to_string()),
                input_tokens: Some(10),
                output_tokens: Some(2),
                thinking_tokens: None,
                cost: Some(0.5),
            }),
        );

        let events = rec.db().export_telemetry(None).unwrap();
        assert_eq!(events.len(), 1);
        let e = &events[0];
        assert_eq!(e.instance_id, "inst-1");
        assert_eq!(e.operation, "search_sessions");
        assert_eq!(e.duration_ms, Some(45));
        assert_eq!(e.safe_args, Some(json!({"mode": "keyword"})));
        assert_eq!(e.model.as_deref(), Some("gpt-5-mini"));
        assert_eq!(e.cost, Some(0.5));
    }

    #[test]
    fn record_stores_null_args_when_nothing_is_safe() {
        let (rec, _dir) = recorder();
        rec.record(
            "get_session_snapshot",
            Some(3),
            Some(&args(&[("session_id", json!("abc"))])),
            None,
        );
        let events = rec.db().export_telemetry(None).unwrap();
        assert_eq!(events[0].safe_args, None);
        assert_eq!(events[0].model, None);
    }

    #[test]
    fn track_times_the_operation() {
        let (rec, _dir) = recorder();
        let t = rec.track("list_projects");
        std::thread::sleep(std::time::Duration::from_millis(5));
        t.finish();
        let events = rec.db().export_telemetry(None).unwrap();
        assert_eq!(events[0].operation, "list_projects");
        assert!(events[0].duration_ms.unwrap() >= 5);
    }

    #[test]
    fn dropping_a_track_without_finishing_records_nothing() {
        let (rec, _dir) = recorder();
        drop(rec.track("list_projects"));
        assert!(rec.db().export_telemetry(None).unwrap().is_empty());
    }

    #[test]
    fn instance_ids_are_32_hex_chars() {
        let dir = TempDir::new().unwrap();
        let db = Database::new(dir.path().join("sessions.db")).unwrap();
        let id = TelemetryRecorder::new(db).instance_id().to_string();
        assert_eq!(id.len(), 32);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()));
    }

    #[test]
    fn export_body_starts_with_the_privacy_header() {
        let (rec, _dir) = recorder();
        rec.record("list_projects", Some(1), None, None);
        let events = rec.db().export_telemetry(None).unwrap();
        let summary = rec.db().get_telemetry_summary(None).unwrap();
        let body = export_body(&events, &summary);

        let mut lines = body.lines();
        let header: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
        assert_eq!(
            header["_comment"],
            json!("TELEMETRY EXPORT - Please review before sharing")
        );
        assert_eq!(header["_privacy"]["does_not_contain"][0], json!("search queries"));
        assert_eq!(header["_summary"]["total_events"], json!(1));
        assert_eq!(
            header["_summary"]["operations"]["list_projects"]["count"],
            json!(1)
        );

        let event: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
        assert_eq!(event["operation"], json!("list_projects"));
        assert_eq!(event["safe_args"], Value::Null);
        // Column order, as `dict(sqlite3.Row)` produces it.
        let keys: Vec<&str> = event.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            vec![
                "id", "instance_id", "timestamp", "operation", "duration_ms", "safe_args",
                "model", "input_tokens", "output_tokens", "thinking_tokens", "cost"
            ]
        );
        assert_eq!(lines.next(), None);
        assert!(body.ends_with('\n'));
    }

    #[test]
    fn export_rejects_paths_outside_the_sandbox() {
        let (rec, _dir) = recorder();
        assert!(rec.export_to_file("../escape").is_err());
        assert!(rec.export_to_file("/etc/passwd").is_err());
    }
}
