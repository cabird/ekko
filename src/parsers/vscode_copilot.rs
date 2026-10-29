//! VS Code Copilot workspace storage parser.
//!
//! Handles chat sessions from VS Code's workspaceStorage:
//! `<workspaceStorage>/<hash>/chatSessions/*.json` with the workspace path in
//! a sibling `workspace.json`.
//!
//! Determinism note: `extract_tool_input` scans `ARG_KEYS` in declaration
//! order for its top-level and result fallbacks, so the key order of the
//! resulting `tool_input` object is fixed rather than dependent on iteration
//! order.

use std::fs;
use std::path::{Path, PathBuf};

use chrono::{DateTime, FixedOffset, Local};
use serde_json::{Map, Value, json};

use super::{Parser, read_text, take_chars};
use crate::models::{
    Event, EventType, MessageRole, ParsedTranscript, SessionMetadata, is_truthy, display_str,
};

/// Parser for VS Code Copilot chat sessions.
pub struct VSCodeCopilotParser;

const CLI_SOURCE: &str = "vscode_copilot";
const PARSER_VERSION: &str = "1.0";

/// Keys that typically contain tool arguments. The order is load-bearing:
/// it fixes the key order of the extracted `tool_input`.
const ARG_KEYS: [&str; 20] = [
    "query",
    "pattern",
    "command",
    "path",
    "content",
    "text",
    "searchQuery",
    "searchPattern",
    "filePath",
    "fileName",
    "patch",
    "diff",
    "code",
    "message",
    "description",
    "directory",
    "folder",
    "input",
    "args",
    "arguments",
];

impl Parser for VSCodeCopilotParser {
    fn cli_source(&self) -> &'static str {
        CLI_SOURCE
    }

    fn parser_version(&self) -> &'static str {
        PARSER_VERSION
    }

    /// Each workspace hash directory may contain `chatSessions/{uuid}.json`.
    fn discover_sessions(&self, base_path: &Path) -> Vec<PathBuf> {
        let Ok(read_dir) = fs::read_dir(base_path) else {
            return Vec::new();
        };
        let mut workspace_dirs: Vec<PathBuf> = read_dir.flatten().map(|e| e.path()).collect();
        workspace_dirs.sort();

        let mut found = Vec::new();
        for workspace_dir in workspace_dirs {
            if !workspace_dir.is_dir() {
                continue;
            }
            let chat_sessions_dir = workspace_dir.join("chatSessions");
            let Ok(sessions) = fs::read_dir(&chat_sessions_dir) else {
                continue;
            };
            let mut files: Vec<PathBuf> = sessions
                .flatten()
                .map(|e| e.path())
                .filter(|p| {
                    p.extension().and_then(|e| e.to_str()) == Some("json") && p.is_file()
                })
                .collect();
            files.sort();
            found.extend(files);
        }
        found
    }

    fn extract_metadata(&self, transcript_path: &Path) -> SessionMetadata {
        let workspace_dir = transcript_path
            .parent()
            .and_then(Path::parent)
            .unwrap_or(Path::new(""));

        // Session ID from filename.
        let session_id = transcript_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();

        // Read workspace.json for project path.
        let mut project_path = String::new();
        if let Some(text) = read_text(&workspace_dir.join("workspace.json"))
            && let Ok(data) = serde_json::from_str::<Value>(&text)
        {
            let folder = data.get("folder").and_then(Value::as_str).unwrap_or("");
            project_path = decode_folder_uri(folder);
        }

        // Read the chat session file.
        let mut first_prompt = String::new();
        let mut created_at: Option<DateTime<FixedOffset>> = None;
        let mut modified_at: Option<DateTime<FixedOffset>> = None;
        let mut message_count: i64 = 0;
        let mut model: Option<String> = None;

        if let Some(text) = read_text(transcript_path)
            && let Ok(session) = serde_json::from_str::<Value>(&text)
        {
            // Timestamps (stored as milliseconds).
            if let Some(v) = session.get("creationDate").filter(|v| is_truthy(v)) {
                created_at = timestamp_from_ms(v);
            }
            if let Some(v) = session.get("lastMessageDate").filter(|v| is_truthy(v)) {
                modified_at = timestamp_from_ms(v);
            }

            // Model info: `identifier`, falling back to `modelId`. The test
            // is truthiness over the raw value, so a truthy non-string does
            // NOT fall through to the next candidate.
            if let Some(sm) = session.get("selectedModel").and_then(Value::as_object) {
                model = sm
                    .get("identifier")
                    .filter(|v| is_truthy(v))
                    .or_else(|| sm.get("modelId").filter(|v| is_truthy(v)))
                    .and_then(Value::as_str)
                    .map(String::from);
            }

            // Extract first prompt and count messages.
            let empty: Vec<Value> = Vec::new();
            let requests = session
                .get("requests")
                .and_then(Value::as_array)
                .unwrap_or(&empty);
            for req in requests {
                let Some(req) = req.as_object() else {
                    continue;
                };
                let text = req
                    .get("message")
                    .and_then(Value::as_object)
                    .and_then(|m| m.get("text"))
                    .and_then(Value::as_str)
                    .unwrap_or("");

                if !text.is_empty() && first_prompt.is_empty() {
                    first_prompt = take_chars(text, 500);
                }

                // Count: 1 for user message + 1 for response.
                if !text.is_empty() {
                    message_count += 1;
                }
                if req.get("response").is_some_and(is_truthy) {
                    message_count += 1;
                }
            }
        }

        let mut meta = SessionMetadata::new(
            session_id,
            CLI_SOURCE,
            transcript_path.to_string_lossy().into_owned(),
        );
        meta.title = if first_prompt.is_empty() {
            None
        } else {
            Some(take_chars(&first_prompt, 100))
        };
        meta.first_prompt = Some(first_prompt);
        meta.project_path = Some(project_path);
        meta.git_branch = None; // Not stored in VS Code Copilot sessions
        meta.created_at = created_at;
        meta.modified_at = modified_at;
        meta.message_count = message_count;
        meta.model = model;
        meta.parser_version = PARSER_VERSION.to_string();
        meta.source_hash = self.get_source_hash(transcript_path);
        meta
    }

    fn parse_transcript(&self, transcript_path: &Path) -> ParsedTranscript {
        let mut metadata = self.extract_metadata(transcript_path);
        let mut events: Vec<Event> = Vec::new();
        let mut sequence: i64 = 0;

        if let Some(text) = read_text(transcript_path)
            && let Ok(session) = serde_json::from_str::<Value>(&text)
        {
            let empty: Vec<Value> = Vec::new();
            let requests = session
                .get("requests")
                .and_then(Value::as_array)
                .unwrap_or(&empty);
            for req in requests {
                for event in parse_request(req, sequence) {
                    events.push(event);
                    sequence += 1;
                }
            }
        }

        metadata.event_count = events.len() as i64;
        ParsedTranscript { metadata, events }
    }
}

/// Convert epoch milliseconds to the local wall-clock time, with the local
/// UTC offset attached. VS Code records these as UTC epoch values, but the
/// stored timestamp is the machine-local reading of that instant.
fn timestamp_from_ms(v: &Value) -> Option<DateTime<FixedOffset>> {
    let ms = v.as_f64()?;
    let micros = (ms * 1000.0).round();
    if !micros.is_finite() || micros.abs() >= i64::MAX as f64 {
        return None;
    }
    let utc = DateTime::from_timestamp_micros(micros as i64)?;
    Some(utc.with_timezone(&Local).fixed_offset())
}

/// Parse a single request/response pair into events.
fn parse_request(req: &Value, start_sequence: i64) -> Vec<Event> {
    let mut events: Vec<Event> = Vec::new();
    let Some(req) = req.as_object() else {
        return events;
    };
    // The generated default applies only when the key is absent; a present
    // non-string value is stringified ("None" for null, "5" for 5).
    let request_id = match req.get("requestId") {
        Some(Value::String(s)) => s.clone(),
        Some(v) => display_str(v),
        None => format!("req_{start_sequence}"),
    };

    // Get timestamp.
    let timestamp = req
        .get("timestamp")
        .filter(|v| is_truthy(v))
        .and_then(timestamp_from_ms);

    // User message.
    let user_text = req
        .get("message")
        .and_then(Value::as_object)
        .and_then(|m| m.get("text"))
        .and_then(Value::as_str)
        .unwrap_or("");
    if !user_text.is_empty() {
        let mut e = Event::new(
            format!("user_{request_id}"),
            start_sequence,
            EventType::Message,
        );
        e.timestamp = timestamp;
        e.role = Some(MessageRole::User);
        e.content = user_text.to_string();
        events.push(e);
    }

    // Process response items.
    let empty: Vec<Value> = Vec::new();
    let response_items = req
        .get("response")
        .and_then(Value::as_array)
        .unwrap_or(&empty);
    let mut text_parts: Vec<String> = Vec::new();

    for (i, item) in response_items.iter().enumerate() {
        if let Some(item_obj) = item.as_object() {
            let kind = item_obj.get("kind").and_then(Value::as_str);

            match kind {
                Some("toolInvocationSerialized") => {
                    // Tool call. An absent `toolId` defaults to "" (not None);
                    // a present-but-null one stays None. The three cases are
                    // distinct and pinned by tests. A non-string `toolId`
                    // survives, stringified ("7"), since `tool_name` is a
                    // String and everything downstream treats it as text.
                    let tool_name = match item_obj.get("toolId") {
                        None => Some(String::new()),
                        Some(Value::Null) => None,
                        Some(v) => Some(display_str(v)),
                    };
                    // `toolCallId` is used for BOTH the event id and
                    // `tool_call_id`. Absent OR present-but-null falls back to
                    // the generated `tool_{start}_{i}`, so no NULL id ever
                    // reaches the DB. Non-string values are stringified.
                    let tool_call_id = match item_obj.get("toolCallId") {
                        None | Some(Value::Null) => format!("tool_{start_sequence}_{i}"),
                        Some(v) => display_str(v),
                    };

                    let tool_input = extract_tool_input(item_obj);

                    let mut e = Event::new(
                        tool_call_id.clone(),
                        start_sequence + events.len() as i64,
                        EventType::ToolCall,
                    );
                    e.timestamp = timestamp;
                    e.tool_name = tool_name;
                    e.tool_input = Some(Value::Object(tool_input));
                    e.tool_call_id = Some(tool_call_id);
                    events.push(e);
                }
                Some("textEditGroup") => {
                    // Code edit — treated as a tool call. A present-but-null
                    // `uri.path` stays null; absent or non-object gives "".
                    let file_path = match item_obj.get("uri").and_then(Value::as_object) {
                        Some(uri) => uri.get("path").cloned().unwrap_or_else(|| json!("")),
                        None => json!(""),
                    };
                    let empty_edits: Vec<Value> = Vec::new();
                    let edits = item_obj
                        .get("edits")
                        .and_then(Value::as_array)
                        .unwrap_or(&empty_edits);

                    // Summarize edits.
                    let mut edit_summary: Vec<String> = Vec::new();
                    for edit_group in edits {
                        let Some(edit_group) = edit_group.as_array() else {
                            continue;
                        };
                        for edit in edit_group {
                            let Some(edit) = edit.as_object() else {
                                continue;
                            };
                            let text_truthy =
                                edit.get("text").map(is_truthy).unwrap_or(false);
                            if text_truthy {
                                let range = edit.get("range").and_then(Value::as_object);
                                let start_line = range
                                    .and_then(|r| r.get("startLineNumber"))
                                    .map(display_str)
                                    .unwrap_or_else(|| "0".to_string());
                                let end_line = range
                                    .and_then(|r| r.get("endLineNumber"))
                                    .map(display_str)
                                    .unwrap_or_else(|| "0".to_string());
                                edit_summary.push(format!("Lines {start_line}-{end_line}"));
                            }
                        }
                    }

                    let edits_str = if edit_summary.is_empty() {
                        "code changes".to_string()
                    } else {
                        edit_summary.join(", ")
                    };
                    let mut e = Event::new(
                        format!("edit_{start_sequence}_{i}"),
                        start_sequence + events.len() as i64,
                        EventType::ToolCall,
                    );
                    e.timestamp = timestamp;
                    e.tool_name = Some("Edit".to_string());
                    e.tool_input = Some(json!({
                        "file_path": file_path,
                        "edits": edits_str,
                    }));
                    events.push(e);
                }
                Some("inlineReference")
                | Some("undoStop")
                | Some("prepareToolInvocation")
                | Some("confirmation")
                | Some("codeblockUri") => {
                    // Skip these internal events.
                }
                _ => {
                    // Plain text response (no kind, just value).
                    if let Some(value) = item_obj
                        .get("value")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                    {
                        text_parts.push(value.to_string());
                    }
                }
            }
        } else if let Value::String(s) = item {
            // Direct string response.
            text_parts.push(s.clone());
        }
    }

    // Combine text parts into assistant message.
    if !text_parts.is_empty() {
        let mut e = Event::new(
            format!("assistant_{request_id}"),
            start_sequence + events.len() as i64,
            EventType::Message,
        );
        e.timestamp = timestamp;
        e.role = Some(MessageRole::Assistant);
        e.content = text_parts.join("\n");
        events.push(e);
    }

    events
}

/// Extract tool input from a `toolInvocationSerialized` item. VS Code Copilot
/// stores tool args in various places depending on the tool.
pub fn extract_tool_input(item: &Map<String, Value>) -> Map<String, Value> {
    let mut tool_input: Map<String, Value> = Map::new();

    // 1. Check invocationMessage.
    if let Some(inv_msg) = item.get("invocationMessage").and_then(Value::as_object) {
        // Extract file path from URIs (first entry with a truthy path wins).
        if let Some(uris) = inv_msg.get("uris").and_then(Value::as_object) {
            for uri_obj in uris.values() {
                if let Some(obj) = uri_obj.as_object()
                    && let Some(path) = obj.get("path").filter(|v| is_truthy(v))
                {
                    tool_input.insert("file_path".to_string(), path.clone());
                    break;
                }
            }
        }

        // Extract all relevant string fields (case-insensitive key match).
        for (key, val) in inv_msg {
            if key == "uris" {
                continue;
            }
            let key_lower = key.to_lowercase();
            if !ARG_KEYS.iter().any(|k| k.to_lowercase() == key_lower) {
                continue;
            }
            match val {
                Value::String(s) if !s.is_empty() => {
                    tool_input.insert(key.clone(), json!(take_chars(s, 500)));
                }
                v if is_truthy(v) => {
                    tool_input.insert(key.clone(), v.clone());
                }
                _ => {}
            }
        }
    }

    // 2. Check inputArgs.
    if let Some(input_args) = item.get("inputArgs").and_then(Value::as_object) {
        for (key, val) in input_args {
            if !tool_input.contains_key(key) && is_truthy(val) {
                match val {
                    Value::String(s) if s.chars().count() > 500 => {
                        tool_input
                            .insert(key.clone(), json!(format!("{}...", take_chars(s, 500))));
                    }
                    v => {
                        tool_input.insert(key.clone(), v.clone());
                    }
                }
            }
        }
    }

    // 3. Check top-level item for args (some tools put them here).
    for key in ARG_KEYS {
        if let Some(val) = item.get(key) {
            if tool_input.contains_key(key) {
                continue;
            }
            match val {
                Value::String(s) if !s.is_empty() => {
                    tool_input.insert(key.to_string(), json!(take_chars(s, 500)));
                }
                v if is_truthy(v) => {
                    tool_input.insert(key.to_string(), v.clone());
                }
                _ => {}
            }
        }
    }

    // 4. Check pastAction for terminal commands.
    if let Some(past_action) = item.get("pastAction").and_then(Value::as_object) {
        if let Some(cmd) = past_action.get("command").filter(|v| is_truthy(v))
            && !tool_input.contains_key("command")
        {
            tool_input.insert("command".to_string(), cmd.clone());
        }
        if let Some(cmd) = past_action.get("commandLine").filter(|v| is_truthy(v))
            && !tool_input.contains_key("command")
        {
            tool_input.insert("command".to_string(), cmd.clone());
        }
    }

    // 5. Check result for context (sometimes has the input echoed).
    if tool_input.is_empty()
        && let Some(result) = item.get("result").and_then(Value::as_object)
    {
        for key in ARG_KEYS {
            if let Some(Value::String(s)) = result.get(key)
                && !s.is_empty()
            {
                tool_input.insert(key.to_string(), json!(take_chars(s, 500)));
            }
        }
    }

    tool_input
}

/// Decode a VS Code folder URI to a path.
///
/// Examples:
/// - `vscode-remote://wsl%2Bubuntu/home/cbird/repos/project`
/// - `file:///c:/Users/cbird/project`
pub fn decode_folder_uri(uri: &str) -> String {
    if uri.is_empty() {
        return String::new();
    }

    // Handle vscode-remote:// URIs (WSL, SSH, etc.)
    if let Some(rest) = uri.strip_prefix("vscode-remote://") {
        // URL decode, then skip the authority (wsl+ubuntu, ssh+host, ...).
        let rest = percent_decode(rest);
        if rest.contains('/') {
            let mut parts = rest.splitn(2, '/');
            let _authority = parts.next();
            if let Some(path) = parts.next() {
                return format!("/{path}");
            }
        }
        return rest;
    }

    // Handle file:// URIs.
    if let Some(path) = uri.strip_prefix("file:///") {
        let path = percent_decode(path);
        // On Windows, keep c:/path as-is.
        if path.chars().nth(1) == Some(':') {
            return path;
        }
        return format!("/{path}");
    }

    uri.to_string()
}

/// Permissive percent-decode: valid `%XX` escapes become bytes (decoded as
/// UTF-8 with replacement), invalid escapes are left literal rather than
/// rejected.
fn percent_decode(s: &str) -> String {
    fn hex_val(b: u8) -> Option<u8> {
        match b {
            b'0'..=b'9' => Some(b - b'0'),
            b'a'..=b'f' => Some(b - b'a' + 10),
            b'A'..=b'F' => Some(b - b'A' + 10),
            _ => None,
        }
    }
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let (Some(hi), Some(lo)) = (hex_val(bytes[i + 1]), hex_val(bytes[i + 2]))
        {
            out.push(hi * 16 + lo);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}
