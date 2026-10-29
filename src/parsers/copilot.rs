//! GitHub Copilot CLI transcript parser.
//!
//! Handles transcripts from `~/.copilot/session-state/` — each session is a
//! UUID directory containing `events.jsonl` and optionally `workspace.yaml`.

use std::fs;
use std::path::{Path, PathBuf};

use chrono::{DateTime, FixedOffset};
use serde_json::{Map, Value, json};

use super::{Parser, read_text, take_chars};
use crate::models::{
    Event, EventType, MessageRole, ParsedTranscript, SessionMetadata, is_truthy, display_str,
};
use crate::utils::parse_datetime;

/// Parser for GitHub Copilot CLI session transcripts.
pub struct CopilotParser;

const CLI_SOURCE: &str = "github_copilot";
const PARSER_VERSION: &str = "1.0";

/// Shared empty object used as the default for a missing sub-object.
static EMPTY_MAP: std::sync::LazyLock<Map<String, Value>> = std::sync::LazyLock::new(Map::new);

/// Look up an object-valued key, distinguishing "absent" from "present but
/// not an object". The distinction is load-bearing:
///
/// * key ABSENT  → `Some(&{})` — the default empty object, so the caller's
///   block still runs (e.g. a `tool.execution_complete` line with no `data`
///   key still emits an event).
/// * key PRESENT and an object → `Some(object)`.
/// * key PRESENT but not an object → `None` — the caller skips the block.
fn as_dict_or_default(v: Option<&Value>) -> Option<&Map<String, Value>> {
    match v {
        None => Some(&EMPTY_MAP),
        Some(Value::Object(m)) => Some(m),
        Some(_) => None,
    }
}

impl Parser for CopilotParser {
    fn cli_source(&self) -> &'static str {
        CLI_SOURCE
    }

    fn parser_version(&self) -> &'static str {
        PARSER_VERSION
    }

    /// Each session is a UUID directory containing `events.jsonl`.
    fn discover_sessions(&self, base_path: &Path) -> Vec<PathBuf> {
        let Ok(read_dir) = fs::read_dir(base_path) else {
            return Vec::new();
        };
        let mut dirs: Vec<PathBuf> = read_dir.flatten().map(|e| e.path()).collect();
        dirs.sort();
        dirs.into_iter()
            .filter(|d| d.is_dir())
            .map(|d| d.join("events.jsonl"))
            .filter(|f| f.exists())
            .collect()
    }

    /// Reads `workspace.yaml` for metadata, `events.jsonl` for timestamps.
    fn extract_metadata(&self, transcript_path: &Path) -> SessionMetadata {
        let session_dir = transcript_path.parent().unwrap_or(Path::new(""));
        let session_id = session_dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();

        // Read workspace.yaml for metadata.
        //
        // DIVERGENCE (YAML 1.1 vs 1.2): serde_yaml implements YAML 1.2 core,
        // so unquoted scalars that a YAML 1.1 resolver would coerce stay
        // strings here:
        //   * `summary: yes` / `no` / `on` / `off` / `y` / `n`
        //     → the strings "yes"/"no"/… , not booleans.
        //   * sexagesimals, e.g. `summary: 12:30` → "12:30", not 750.
        //   * unquoted dates, e.g. `summary: 2025-01-01` → the string, not a
        //     date value.
        // Keeping the literal text the file contains is what we want for a
        // title, so this is deliberate and we do NOT switch YAML libraries.
        // Non-string YAML values that do survive (numbers, bools) are dropped
        // to "" here because `title` is a String; pinned by
        // `copilot_yaml_11_vs_12_divergence`.
        let workspace_data: Value = read_text(&session_dir.join("workspace.yaml"))
            .and_then(|text| serde_yaml::from_str::<Value>(&text).ok())
            .unwrap_or(Value::Null);

        let mut project_path = workspace_data
            .get("cwd")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let title = workspace_data
            .get("summary")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let mut git_branch: Option<String> = None; // Extract from session.start event

        // Parse events for timestamps and first prompt.
        let mut first_prompt = String::new();
        let mut created_at: Option<DateTime<FixedOffset>> = None;
        let mut modified_at: Option<DateTime<FixedOffset>> = None;
        let mut message_count: i64 = 0;

        if let Some(text) = read_text(transcript_path) {
            for line in text.lines() {
                let Ok(event) = serde_json::from_str::<Value>(line) else {
                    continue;
                };
                let Some(event) = event.as_object() else {
                    continue;
                };
                let event_type = event.get("type").and_then(Value::as_str).unwrap_or("");

                if let Some(ts) = event
                    .get("timestamp")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .and_then(parse_datetime)
                {
                    if created_at.is_none() {
                        created_at = Some(ts);
                    }
                    modified_at = Some(ts);
                }

                // Extract from session.start. `data` and `data.context` each
                // use the absent-vs-non-object rule above: an absent key acts
                // like an empty object, a present non-object skips the block.
                // So a `session.start` with `data: {}` (or no `data` key at
                // all) still enters the block and RESETS git_branch to None.
                if event_type == "session.start"
                    && let Some(data) = as_dict_or_default(event.get("data"))
                    && let Some(context) = as_dict_or_default(data.get("context"))
                {
                    if project_path.is_empty() {
                        project_path = context
                            .get("cwd")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                    }
                    // Assigned unconditionally — a session.start without a
                    // branch resets it to None.
                    git_branch = context
                        .get("branch")
                        .and_then(Value::as_str)
                        .map(String::from);
                }

                // Extract first user message. Same absent-vs-non-object rule:
                // absent `data` behaves like an empty object (no content, so
                // no assignment), present non-object `data` is skipped. Both
                // leave `first_prompt` untouched here, so the plain lookup
                // chain below is equivalent.
                if event_type == "user.message"
                    && first_prompt.is_empty()
                    && let Some(content) = event
                        .get("data")
                        .and_then(Value::as_object)
                        .and_then(|d| d.get("content"))
                        .and_then(Value::as_str)
                {
                    first_prompt = take_chars(content, 500);
                }

                // Count messages.
                if event_type == "user.message" || event_type == "assistant.message" {
                    message_count += 1;
                }
            }
        }

        let mut meta = SessionMetadata::new(
            session_id,
            CLI_SOURCE,
            transcript_path.to_string_lossy().into_owned(),
        );
        meta.title = Some(title);
        meta.first_prompt = Some(first_prompt);
        meta.project_path = Some(project_path);
        meta.git_branch = git_branch;
        meta.created_at = created_at;
        meta.modified_at = modified_at;
        meta.message_count = message_count;
        meta.parser_version = PARSER_VERSION.to_string();
        meta.source_hash = self.get_source_hash(transcript_path);
        meta
    }

    fn parse_transcript(&self, transcript_path: &Path) -> ParsedTranscript {
        let mut metadata = self.extract_metadata(transcript_path);
        let mut events: Vec<Event> = Vec::new();
        let mut sequence: i64 = 0;

        if let Some(text) = read_text(transcript_path) {
            for line in text.lines() {
                let Ok(entry) = serde_json::from_str::<Value>(line) else {
                    continue;
                };
                for event in parse_event(&entry, sequence) {
                    events.push(event);
                    sequence += 1;
                }
            }
        }

        metadata.event_count = events.len() as i64;
        ParsedTranscript { metadata, events }
    }
}

/// Parse a single Copilot event into unified events.
fn parse_event(entry: &Value, start_sequence: i64) -> Vec<Event> {
    let mut events: Vec<Event> = Vec::new();
    let Some(entry) = entry.as_object() else {
        return events;
    };
    let event_type = entry.get("type").and_then(Value::as_str).unwrap_or("");
    // A missing OR present-but-null `id` falls back to the generated
    // `event_{start_sequence}`, because NULL event ids must never reach the
    // database. Pinned by `copilot_null_id_generates_fallback`. A present
    // non-string id is stringified (e.g. the int 5 → "5"); everything
    // downstream treats ids as text anyway.
    let event_id = match entry.get("id") {
        None | Some(Value::Null) => format!("event_{start_sequence}"),
        Some(v) => display_str(v),
    };
    let timestamp = entry
        .get("timestamp")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .and_then(parse_datetime);

    // CRITICAL: an ABSENT `data` key becomes an empty object and FALLS
    // THROUGH — a bare `{"type": "tool.execution_complete"}` line still emits
    // a TOOL_RESULT (id "result_None", tool_output "", tool_success False) and
    // therefore still advances the sequence counter. Only a PRESENT
    // non-object `data` returns early.
    let Some(data) = as_dict_or_default(entry.get("data")) else {
        return events;
    };

    if event_type == "user.message" {
        if let Some(content) = data
            .get("content")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            let mut e = Event::new(event_id, start_sequence, EventType::Message);
            e.timestamp = timestamp;
            e.role = Some(MessageRole::User);
            e.content = content.to_string();
            events.push(e);
        }
    } else if event_type == "assistant.message" {
        // Add message event.
        if let Some(content) = data
            .get("content")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            let mut e = Event::new(event_id, start_sequence, EventType::Message);
            e.timestamp = timestamp;
            e.role = Some(MessageRole::Assistant);
            e.content = content.to_string();
            events.push(e);
        }

        // Add tool call events. Note the sequence offset uses the enumerate
        // index `i`, so skipped non-object items still advance it.
        let empty: Vec<Value> = Vec::new();
        let tool_requests = data
            .get("toolRequests")
            .and_then(Value::as_array)
            .unwrap_or(&empty);
        for (i, tool_req) in tool_requests.iter().enumerate() {
            let Some(tool_req) = tool_req.as_object() else {
                continue;
            };
            // arguments can be an object or a raw string (e.g. apply_patch).
            let args = match tool_req.get("arguments") {
                Some(Value::String(s)) => json!({ "content": s }),
                Some(v @ Value::Object(_)) => v.clone(),
                _ => json!({}),
            };
            // A missing or present-but-null `toolCallId` leaves
            // `tool_call_id` None but still gives the event the generated
            // `tool_{start}_{i}` id, so no NULL id reaches the DB. Pinned by
            // `copilot_null_tool_call_id_generates_fallback`. Non-string
            // values are stringified (the int 5 is stored as "5").
            let tool_call_id = match tool_req.get("toolCallId") {
                None | Some(Value::Null) => None,
                Some(v) => Some(display_str(v)),
            };
            let id = tool_call_id
                .clone()
                .unwrap_or_else(|| format!("tool_{start_sequence}_{i}"));
            let mut e = Event::new(id, start_sequence + i as i64 + 1, EventType::ToolCall);
            e.timestamp = timestamp;
            // A non-string tool name survives, stringified for the same
            // reason as `toolCallId` above; absent/null stays None.
            e.tool_name = match tool_req.get("name") {
                None | Some(Value::Null) => None,
                Some(v) => Some(display_str(v)),
            };
            e.tool_input = Some(args);
            e.tool_call_id = tool_call_id;
            events.push(e);
        }
    } else if event_type == "tool.execution_complete" {
        // The event id is `result_<toolCallId>`; absent or null renders the
        // literal "None", so those events are still distinguishable.
        let tcid_display = data
            .get("toolCallId")
            .map(display_str)
            .unwrap_or_else(|| "None".to_string());
        // A non-string id survives, stringified, so the stored
        // `tool_call_id` and the "result_5" event id agree. Absent or null
        // stays None.
        let tool_call_id = match data.get("toolCallId") {
            None | Some(Value::Null) => None,
            Some(v) => Some(display_str(v)),
        };
        // An absent `success` means false; a present null stays None (unknown);
        // other non-bool values collapse to their truthiness.
        let tool_success = match data.get("success") {
            None => Some(false),
            Some(Value::Null) => None,
            Some(v) => Some(is_truthy(v)),
        };
        // An object `result` contributes its `content` (absent → ""; present
        // null → None; non-string → stringified). A non-object `result` is
        // stringified when truthy and becomes "" otherwise.
        let tool_output: Option<String> = match data.get("result") {
            Some(v @ Value::Object(_)) => match v.get("content") {
                None => Some(String::new()),
                Some(Value::Null) => None,
                Some(Value::String(s)) => Some(s.clone()),
                Some(other) => Some(display_str(other)),
            },
            Some(v) if is_truthy(v) => Some(display_str(v)),
            _ => Some(String::new()),
        };

        let mut e = Event::new(
            format!("result_{tcid_display}"),
            start_sequence,
            EventType::ToolResult,
        );
        e.timestamp = timestamp;
        e.tool_output = tool_output;
        e.tool_success = tool_success;
        e.tool_call_id = tool_call_id;
        events.push(e);
    }

    events
}
