//! OpenAI Codex transcript parser.
//!
//! Codex (CLI and Desktop) writes one "rollout" per thread under
//! `~/.codex/sessions/YYYY/MM/DD/rollout-<timestamp>-<thread-id>.jsonl`
//! (archived threads move to `~/.codex/archived_sessions/`). The format is
//! undocumented; it was derived from Codex 0.153 rollouts on 2026-09-07,
//! with fallbacks for the older `shell` / `apply_patch` function-call shapes.
//!
//! # Record model
//!
//! Every line is `{"timestamp", "type", "payload", ...}`. Only three record
//! types carry conversation content:
//!
//! * `session_meta` (first line): thread id, `cwd`, `git.branch`,
//!   `source` (how the thread was started) and `thread_source`.
//! * `response_item`: the model-facing history — `message` (roles
//!   `user` / `assistant` / `developer`), `function_call` +
//!   `function_call_output`, `custom_tool_call` + `custom_tool_call_output`,
//!   `reasoning` (encrypted; skipped), `compaction` (encrypted; skipped),
//!   `agent_message` (inter-agent traffic).
//! * `turn_context`: per-turn `cwd` and `model`.
//!
//! `event_msg` records mirror the response items for the UI (`item_completed`
//! duplicates every message) and are skipped so nothing is double counted;
//! `compacted` records replay history that was already parsed from the
//! original lines and are skipped for the same reason.
//!
//! # Gotchas this parser exists to handle
//!
//! * **Injected user-role messages.** Codex sends environment context,
//!   plugin recommendations and AGENTS.md as `role: "user"` messages. The
//!   passthrough metadata's `content_item_kinds` marks what a person typed
//!   (`user.text`); older rollouts without that field fall back to a shape
//!   heuristic (a message that is one XML-ish block is not typed input).
//! * **Subagent and one-shot threads.** Every subagent gets its own rollout
//!   file with `source: {"subagent": …}` (this Mac had 12 subagent files for
//!   3 human threads); `codex exec` one-shots have `source: "exec"`. Both are
//!   surfaced through [`SessionMetadata::entrypoint`] so
//!   [`crate::indexer::is_interactive`] can skip them like `sdk-cli`.
//! * **Tool calls are JavaScript.** The current `exec` tool takes a script
//!   (`text(await tools.exec_command({cmd:"ls"}))`), not a command. The
//!   parser lifts every `cmd` literal out so timelines see shell commands;
//!   the raw script is kept alongside them. File edits are
//!   `tools.apply_patch("*** Begin Patch…")` calls inside the same scripts
//!   (57 of them across this Mac's rollouts, none as a top-level tool call),
//!   so each nested patch becomes an `apply_patch` tool-call event of its
//!   own — that is what feeds the files-touched list.
//! * **Forked history.** A subagent or forked thread's rollout begins with
//!   its own `session_meta`, then replays the parent's records — the
//!   parent's `session_meta` included. Only the first one describes the
//!   file; taking the last would give every subagent its parent's id.
//! * **Titles live elsewhere.** `<codex-home>/session_index.jsonl` maps
//!   thread ids to names, like Claude's `sessions-index.json`.

use std::fs;
use std::path::{Path, PathBuf};

use chrono::{DateTime, FixedOffset};
use serde_json::{Map, Value, json};

use super::{Parser, strip_ws, read_text, take_chars};
use crate::models::{Event, EventType, MessageRole, ParsedTranscript, SessionMetadata, display_str};
use crate::utils::parse_datetime;

/// Parser for OpenAI Codex rollouts.
pub struct CodexParser;

const CLI_SOURCE: &str = "codex";
const PARSER_VERSION: &str = "1.0";

/// Rollout files are `rollout-<timestamp>-<uuid>.jsonl`.
const ROLLOUT_PREFIX: &str = "rollout-";
/// `<base>/YYYY/MM/DD/rollout-*.jsonl` is three directories deep; the cap
/// leaves headroom without letting a stray symlink farm recurse forever.
const MAX_DISCOVER_DEPTH: usize = 6;
/// How far above the transcript to look for `session_index.jsonl`
/// (`<home>/sessions/YYYY/MM/DD/file`: the home is the fifth ancestor).
const MAX_INDEX_ANCESTORS: usize = 6;
/// A standard UUID rendering, which is what a rollout file name ends with.
const UUID_LEN: usize = 36;

impl Parser for CodexParser {
    fn cli_source(&self) -> &'static str {
        CLI_SOURCE
    }

    fn parser_version(&self) -> &'static str {
        PARSER_VERSION
    }

    /// Every `rollout-*.jsonl` under the base, however it is nested (dated
    /// directories under `sessions/`, flat under `archived_sessions/`),
    /// sorted for determinism. Directory symlinks are not followed.
    fn discover_sessions(&self, base_path: &Path) -> Vec<PathBuf> {
        let mut out = Vec::new();
        walk_rollouts(base_path, &mut out, 0);
        out.sort();
        out
    }

    /// The file name (`rollout-<timestamp>-<uuid>.jsonl`, unique per
    /// thread). `~/.codex` and `$CODEX_HOME` can hold the same rollouts
    /// (Orca hard-links them on this machine); keying on the name means a
    /// thread is discovered once, from the first candidate base path.
    fn discovery_key(&self, transcript_path: &Path) -> String {
        transcript_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| transcript_path.to_string_lossy().into_owned())
    }

    fn extract_metadata(&self, transcript_path: &Path) -> SessionMetadata {
        let mut session_id = session_id_from_filename(transcript_path);
        let mut project_path: Option<String> = None;
        let mut git_branch: Option<String> = None;
        let mut entrypoint: Option<String> = None;
        let mut model: Option<String> = None;
        let mut first_prompt = String::new();
        let mut created_at: Option<DateTime<FixedOffset>> = None;
        let mut modified_at: Option<DateTime<FixedOffset>> = None;
        let mut message_count: i64 = 0;
        let mut meta_seen = false;

        if let Some(text) = read_text(transcript_path) {
            for line in text.lines() {
                let Some(Record {
                    record_type,
                    timestamp,
                    payload,
                }) = parse_record(line)
                else {
                    continue;
                };

                if let Some(ts) = timestamp {
                    if created_at.is_none() {
                        created_at = Some(ts);
                    }
                    modified_at = Some(ts);
                }

                match record_type.as_str() {
                    // Only the FIRST session_meta describes this thread. A
                    // subagent or forked thread carries its parent's history
                    // after its own meta — including the parent's
                    // session_meta, whose id must not replace this one.
                    "session_meta" if meta_seen => {}
                    "session_meta" => {
                        meta_seen = true;
                        if let Some(id) = payload
                            .get("id")
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                        {
                            session_id = id.to_string();
                        }
                        if project_path.is_none()
                            && let Some(cwd) = payload.get("cwd").and_then(Value::as_str)
                        {
                            project_path = Some(cwd.to_string());
                        }
                        if let Some(branch) = payload
                            .get("git")
                            .and_then(Value::as_object)
                            .and_then(|g| g.get("branch"))
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                        {
                            git_branch = Some(branch.to_string());
                        }
                        if entrypoint.is_none() {
                            entrypoint = entrypoint_of(&payload);
                        }
                        // The meta's own timestamp is when the thread was
                        // created; the record's is when it was written.
                        if let Some(ts) = payload
                            .get("timestamp")
                            .and_then(Value::as_str)
                            .and_then(parse_datetime)
                        {
                            created_at = Some(ts);
                        }
                    }
                    "turn_context" => {
                        if project_path.is_none()
                            && let Some(cwd) = payload.get("cwd").and_then(Value::as_str)
                        {
                            project_path = Some(cwd.to_string());
                        }
                        if model.is_none()
                            && let Some(m) = payload
                                .get("model")
                                .and_then(Value::as_str)
                                .filter(|s| !s.is_empty())
                        {
                            model = Some(m.to_string());
                        }
                    }
                    "response_item" => {
                        if payload.get("type").and_then(Value::as_str) != Some("message") {
                            continue;
                        }
                        match payload.get("role").and_then(Value::as_str) {
                            Some("user") => {
                                let Some(text) = typed_user_text(&payload) else {
                                    continue;
                                };
                                message_count += 1;
                                if first_prompt.is_empty() {
                                    let stripped = strip_ws(&text);
                                    if !stripped.is_empty() {
                                        first_prompt = take_chars(stripped, 500);
                                    }
                                }
                            }
                            Some("assistant") => message_count += 1,
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
        }

        let title = lookup_thread_name(transcript_path, &session_id);

        let mut meta = SessionMetadata::new(
            session_id,
            CLI_SOURCE,
            transcript_path.to_string_lossy().into_owned(),
        );
        meta.title = title;
        meta.first_prompt = Some(first_prompt);
        meta.project_path = Some(project_path.unwrap_or_default());
        meta.git_branch = git_branch;
        meta.entrypoint = entrypoint;
        meta.model = model;
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
                let Some(Record {
                    record_type,
                    timestamp,
                    payload,
                }) = parse_record(line)
                else {
                    continue;
                };
                if record_type != "response_item" {
                    continue;
                }
                for event in parse_response_item(&payload, timestamp, sequence) {
                    events.push(event);
                    sequence += 1;
                }
            }
        }

        metadata.event_count = events.len() as i64;
        ParsedTranscript { metadata, events }
    }
}

// ---------------------------------------------------------------------------
// Discovery helpers
// ---------------------------------------------------------------------------

fn is_rollout_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.starts_with(ROLLOUT_PREFIX) && n.ends_with(".jsonl"))
        && path.is_file()
}

fn walk_rollouts(dir: &Path, out: &mut Vec<PathBuf>, depth: usize) {
    if depth >= MAX_DISCOVER_DEPTH {
        return;
    }
    let Ok(read_dir) = fs::read_dir(dir) else {
        return;
    };
    for entry in read_dir.flatten() {
        let path = entry.path();
        // `symlink_metadata` reports a symlinked directory as a symlink, so
        // it is neither descended into nor (being no file) collected.
        let is_real_dir = fs::symlink_metadata(&path).is_ok_and(|m| m.is_dir());
        if is_real_dir {
            walk_rollouts(&path, out, depth + 1);
        } else if is_rollout_file(&path) {
            out.push(path);
        }
    }
}

/// The thread id from `rollout-<timestamp>-<uuid>.jsonl`; the whole stem
/// when the name does not end in a UUID (a hand-renamed file still gets a
/// stable, unique id).
pub(crate) fn session_id_from_filename(path: &Path) -> String {
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    if stem.len() > UUID_LEN + 1 && stem.is_char_boundary(stem.len() - UUID_LEN) {
        let (head, tail) = stem.split_at(stem.len() - UUID_LEN);
        if head.ends_with('-') && looks_like_uuid(tail) {
            return tail.to_string();
        }
    }
    stem
}

fn looks_like_uuid(s: &str) -> bool {
    s.len() == UUID_LEN
        && s.bytes().enumerate().all(|(i, b)| match i {
            8 | 13 | 18 | 23 => b == b'-',
            _ => b.is_ascii_hexdigit(),
        })
}

/// Thread name from `session_index.jsonl`, found by walking up from the
/// transcript (`<codex-home>/sessions/YYYY/MM/DD/file` → `<codex-home>`).
/// `None` when there is no index or no entry — not `Some("")`, so the
/// upsert keeps any title an LLM summary produced.
fn lookup_thread_name(transcript_path: &Path, session_id: &str) -> Option<String> {
    let index_path = transcript_path
        .ancestors()
        .skip(1)
        .take(MAX_INDEX_ANCESTORS)
        .map(|dir| dir.join("session_index.jsonl"))
        .find(|p| p.is_file())?;
    let text = read_text(&index_path)?;
    for line in text.lines() {
        let Ok(Value::Object(entry)) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if entry.get("id").and_then(Value::as_str) == Some(session_id) {
            return entry
                .get("thread_name")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(String::from);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Record helpers
// ---------------------------------------------------------------------------

/// One parsed JSONL line: the record's `type`, its timestamp, and its
/// payload (moved out of the record — an 18 MB rollout is parsed three
/// times per index pass, so payloads are not cloned).
struct Record {
    record_type: String,
    timestamp: Option<DateTime<FixedOffset>>,
    payload: Map<String, Value>,
}

/// `None` for malformed lines, non-object records, or records without an
/// object payload.
fn parse_record(line: &str) -> Option<Record> {
    let Value::Object(mut record) = serde_json::from_str::<Value>(line).ok()? else {
        return None;
    };
    let payload = match record.remove("payload") {
        Some(Value::Object(p)) => p,
        _ => return None,
    };
    let record_type = record
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let timestamp = record
        .get("timestamp")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .and_then(parse_datetime);
    Some(Record {
        record_type,
        timestamp,
        payload,
    })
}

/// How the thread was started, from `session_meta.source`: a string
/// (`"cli"`, `"vscode"`, `"exec"`, `"mcp"`) is kept as is; the object form
/// `{"subagent": {...}}` becomes `"subagent"`. Older rollouts without
/// `source` fall back to `thread_source`, where anything but `user`
/// (`subagent`, `guardian_review`, …) is a spawned thread → `"subagent"`.
fn entrypoint_of(meta: &Map<String, Value>) -> Option<String> {
    match meta.get("source") {
        Some(Value::String(s)) if !s.is_empty() => return Some(s.clone()),
        Some(Value::Object(o)) => {
            if o.contains_key("subagent") {
                return Some("subagent".to_string());
            }
            if let Some(key) = o.keys().next() {
                return Some(key.clone());
            }
        }
        _ => {}
    }
    match meta.get("thread_source").and_then(Value::as_str) {
        Some(ts) if !ts.is_empty() && ts != "user" => Some("subagent".to_string()),
        _ => None,
    }
}

/// What a person typed in a `role: "user"` message, or `None` when the
/// message is context Codex injected (environment, plugin list, AGENTS.md).
///
/// `content_item_kinds` in the passthrough metadata runs parallel to
/// `content` (one kind per item; verified 82/82 on real rollouts), so when it
/// is present only the items marked `user.text` are kept — a turn that ships
/// context and typed text in one message yields just the typed text. Older
/// rollouts have no kinds; there a message that is one XML-ish block
/// (`<environment_context>…</environment_context>`) is taken as injected.
fn typed_user_text(payload: &Map<String, Value>) -> Option<String> {
    let kinds = payload
        .get("internal_chat_message_metadata_passthrough")
        .and_then(Value::as_object)
        .and_then(|m| m.get("content_item_kinds"))
        .and_then(Value::as_array);
    let Some(kinds) = kinds else {
        let text = message_text(payload);
        let trimmed = text.trim();
        let injected = trimmed.starts_with('<') && trimmed.ends_with('>') && trimmed.contains("</");
        return if injected { None } else { Some(text) };
    };
    if !kinds.iter().any(|k| k.as_str() == Some("user.text")) {
        return None;
    }
    let items = payload.get("content").and_then(Value::as_array);
    match items {
        // Parallel arrays: keep the typed items only.
        Some(items) if items.len() == kinds.len() => {
            let mut out = String::new();
            for (item, kind) in items.iter().zip(kinds) {
                if kind.as_str() == Some("user.text") {
                    push_item_text(&mut out, item);
                }
            }
            Some(out)
        }
        // Lengths disagree (or string content): cannot attribute per item;
        // the message is typed, so keep all of it.
        _ => Some(message_text(payload)),
    }
}

/// Text of a message's `input_text` / `output_text` / `text` content items
/// (images and refusals contribute nothing), items separated by a newline
/// unless the previous one already ends with one.
fn message_text(payload: &Map<String, Value>) -> String {
    let mut out = String::new();
    match payload.get("content") {
        Some(Value::String(s)) => out.push_str(s),
        Some(Value::Array(items)) => {
            for item in items {
                push_item_text(&mut out, item);
            }
        }
        _ => {}
    }
    out
}

fn push_item_text(out: &mut String, item: &Value) {
    let Some(item) = item.as_object() else {
        return;
    };
    if !matches!(
        item.get("type").and_then(Value::as_str),
        Some("input_text") | Some("output_text") | Some("text")
    ) {
        return;
    }
    let text = item.get("text").and_then(Value::as_str).unwrap_or("");
    if text.is_empty() {
        return;
    }
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(text);
}

/// Text of a tool output, which is a plain string in `function_call_output`
/// and a list of `input_text` items in `custom_tool_call_output`.
fn output_text(output: Option<&Value>) -> String {
    match output {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(items)) => {
            let mut out = String::new();
            for item in items {
                if let Some(item) = item.as_object() {
                    if let Some(t) = item.get("text").and_then(Value::as_str) {
                        out.push_str(t);
                    }
                } else if let Value::String(s) = item {
                    out.push_str(s);
                }
            }
            out
        }
        Some(Value::Null) | None => String::new(),
        Some(other) => display_str(other),
    }
}

/// `Some(true)` when every `"exit_code":N` envelope in the output is zero,
/// `Some(false)` when any is not, `None` when there is no verdict. One
/// `exec` script can run several commands, and each gets its own envelope
/// (75 multi-envelope outputs on real data, 2 with a failure after a
/// success). An escaped `\"exit_code\"` inside an envelope's own `output`
/// string is a command's text, not a verdict, and is skipped.
fn success_from_output(text: &str) -> Option<bool> {
    const KEY: &str = "\"exit_code\":";
    let mut verdict: Option<bool> = None;
    let mut from = 0;
    while let Some(rel) = text[from..].find(KEY) {
        let idx = from + rel;
        from = idx + KEY.len();
        if text[..idx].ends_with('\\') {
            continue;
        }
        let digits: String = text[from..]
            .trim_start()
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '-')
            .collect();
        if let Ok(code) = digits.parse::<i64>() {
            verdict = Some(verdict.unwrap_or(true) && code == 0);
        }
    }
    verdict
}

// ---------------------------------------------------------------------------
// Response items → events
// ---------------------------------------------------------------------------

fn parse_response_item(
    payload: &Map<String, Value>,
    timestamp: Option<DateTime<FixedOffset>>,
    start_sequence: i64,
) -> Vec<Event> {
    let mut events: Vec<Event> = Vec::new();
    let item_type = payload.get("type").and_then(Value::as_str).unwrap_or("");

    match item_type {
        "message" => {
            let (role, text) = match payload.get("role").and_then(Value::as_str) {
                Some("user") => match typed_user_text(payload) {
                    Some(text) => (MessageRole::User, text),
                    // Injected context is not part of the conversation.
                    None => return events,
                },
                Some("assistant") => (MessageRole::Assistant, message_text(payload)),
                // `developer` instructions are not part of the conversation.
                _ => return events,
            };
            let stripped = strip_ws(&text);
            if stripped.is_empty() {
                return events;
            }
            let prefix = match role {
                MessageRole::User => "user",
                _ => "assistant",
            };
            let mut e = Event::new(
                format!("{prefix}_{start_sequence}"),
                start_sequence,
                EventType::Message,
            );
            e.timestamp = timestamp;
            e.role = Some(role);
            e.content = stripped.to_string();
            events.push(e);
        }
        "function_call" => {
            // `arguments` is a JSON-encoded string; keep it raw when it is
            // not an object so nothing is lost.
            let tool_input = match payload.get("arguments") {
                Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
                    Ok(v @ Value::Object(_)) => v,
                    _ => json!({ "content": s }),
                },
                Some(v @ Value::Object(_)) => v.clone(),
                _ => json!({}),
            };
            let name = payload.get("name").and_then(Value::as_str).unwrap_or("");
            let tool_input = normalize_tool_input(name, tool_input);
            events.push(tool_call_event(
                payload,
                name,
                tool_input,
                timestamp,
                start_sequence,
            ));
        }
        "custom_tool_call" => {
            // `input` is a free-form string whose meaning depends on the
            // tool: a JavaScript program for `exec`, a patch for
            // `apply_patch`.
            let name = payload.get("name").and_then(Value::as_str).unwrap_or("");
            let raw = match payload.get("input") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Null) | None => String::new(),
                Some(other) => display_str(other),
            };
            let tool_input = normalize_tool_input(name, json!({ "content": raw }));
            let exec_event = tool_call_event(payload, name, tool_input, timestamp, start_sequence);
            let exec_id = exec_event.id.clone();
            events.push(exec_event);

            // An `exec` script can call `tools.apply_patch(...)` itself; on
            // real rollouts that is how most file edits happen. Each nested
            // patch is a tool call in its own right (the script invoked the
            // tool), so it gets an event of its own — that is what puts its
            // files in the files-touched list as writes.
            if name == "exec" {
                for (i, patch) in extract_exec_patches(&raw).into_iter().enumerate() {
                    let input = normalize_tool_input("apply_patch", json!({ "content": patch }));
                    let mut e = Event::new(
                        format!("{exec_id}_patch_{i}"),
                        start_sequence + i as i64 + 1,
                        EventType::ToolCall,
                    );
                    e.timestamp = timestamp;
                    e.tool_name = Some("apply_patch".to_string());
                    e.tool_input = Some(input);
                    events.push(e);
                }
            }
        }
        "function_call_output" | "custom_tool_call_output" => {
            let tool_call_id = payload
                .get("call_id")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(String::from);
            let text = output_text(payload.get("output"));
            let mut e = Event::new(
                format!("result_{}", tool_call_id.as_deref().unwrap_or("None")),
                start_sequence,
                EventType::ToolResult,
            );
            e.timestamp = timestamp;
            e.tool_success = success_from_output(&text);
            e.tool_output = Some(text);
            e.tool_call_id = tool_call_id;
            events.push(e);
        }
        "agent_message" => {
            // Traffic between a thread and its subagents. The subagents'
            // own rollouts are skipped as non-interactive and the parent's
            // `spawn_agent` / `send_message` payloads are encrypted, so this
            // is the only readable record of what the agents found. It is
            // assistant-side text, prefixed with who sent it to whom, so it
            // reaches conversation text (summaries, FTS, Q&A).
            let text = message_text(payload);
            let stripped = strip_ws(&text);
            if stripped.is_empty() {
                return events;
            }
            let author = payload.get("author").and_then(Value::as_str).unwrap_or("?");
            let recipient = payload
                .get("recipient")
                .and_then(Value::as_str)
                .unwrap_or("?");
            let mut e = Event::new(
                format!("agent_{start_sequence}"),
                start_sequence,
                EventType::Message,
            );
            e.timestamp = timestamp;
            e.role = Some(MessageRole::Assistant);
            e.content = format!("[agent message {author} → {recipient}]\n{stripped}");
            events.push(e);
        }
        // `reasoning` and `compaction` are encrypted blobs; anything else is
        // unknown and skipped rather than guessed at.
        _ => {}
    }

    events
}

fn tool_call_event(
    payload: &Map<String, Value>,
    name: &str,
    tool_input: Value,
    timestamp: Option<DateTime<FixedOffset>>,
    sequence: i64,
) -> Event {
    let tool_call_id = payload
        .get("call_id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(String::from);
    let id = tool_call_id
        .clone()
        .or_else(|| {
            payload
                .get("id")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(String::from)
        })
        .unwrap_or_else(|| format!("tool_{sequence}"));
    let mut e = Event::new(id, sequence, EventType::ToolCall);
    e.timestamp = timestamp;
    e.tool_name = if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    };
    e.tool_input = Some(tool_input);
    e.tool_call_id = tool_call_id;
    e
}

/// Give Codex's tool inputs the keys the rest of ekko already understands
/// (`command` for shell-like calls, `files` for patches) without discarding
/// the raw form.
fn normalize_tool_input(name: &str, mut input: Value) -> Value {
    let Some(obj) = input.as_object_mut() else {
        return input;
    };
    match name {
        "exec" => {
            // Every `exec_command({cmd: …})` in the script, newline-joined
            // (the timeline flattens newlines to `; `).
            if let Some(script) = obj.get("content").and_then(Value::as_str) {
                let cmds = extract_exec_commands(script);
                if !cmds.is_empty() {
                    obj.insert("command".to_string(), Value::String(cmds.join("\n")));
                }
            }
        }
        "apply_patch" => {
            // Function-call form carries the patch under `input` or `patch`;
            // the custom-tool form arrives as `content` already.
            let patch = ["content", "input", "patch"]
                .iter()
                .find_map(|k| obj.get(*k).and_then(Value::as_str))
                .map(String::from);
            if let Some(patch) = patch {
                let files: Vec<Value> = patch_file_paths(&patch)
                    .into_iter()
                    .map(Value::String)
                    .collect();
                obj.insert("content".to_string(), Value::String(patch));
                obj.remove("input");
                obj.remove("patch");
                if !files.is_empty() {
                    obj.insert("files".to_string(), Value::Array(files));
                }
            }
        }
        _ => {}
    }
    input
}

/// File paths named by an `apply_patch` envelope's `*** Add File:` /
/// `*** Update File:` / `*** Delete File:` headers, in order, deduplicated.
pub(crate) fn patch_file_paths(patch: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in patch.lines() {
        let line = line.trim_end();
        let path = ["*** Add File: ", "*** Update File: ", "*** Delete File: "]
            .iter()
            .find_map(|prefix| line.strip_prefix(prefix));
        if let Some(path) = path.map(str::trim).filter(|p| !p.is_empty())
            && !out.iter().any(|p| p == path)
        {
            out.push(path.to_string());
        }
    }
    out
}

/// The `cmd` string literal of every `exec_command({cmd: …})` call in an
/// `exec` script, in order. Empty when the script runs no command (it may
/// only read files or view images) or no literal can be parsed (a variable,
/// a template with substitutions).
pub(crate) fn extract_exec_commands(script: &str) -> Vec<String> {
    call_argument_texts(script, "exec_command(")
        .into_iter()
        .filter_map(|args| find_key_literal(args, "cmd"))
        .collect()
}

/// The patch text of every `apply_patch(…)` call in an `exec` script:
/// `tools.apply_patch("*** Begin Patch…")` (a bare string, the shape Codex
/// emits) or `apply_patch({patch: "…"})`.
pub(crate) fn extract_exec_patches(script: &str) -> Vec<String> {
    call_argument_texts(script, "apply_patch(")
        .into_iter()
        .filter_map(|args| {
            let trimmed = args.trim_start();
            if trimmed.starts_with('{') {
                find_key_literal(trimmed, "patch").or_else(|| find_key_literal(trimmed, "input"))
            } else {
                parse_js_string_literal(trimmed)
            }
        })
        .collect()
}

/// The argument text of each `call` (e.g. `exec_command(`) in a script: from
/// the character after the opening paren to its matching close paren. A
/// single left-to-right pass tracks string literals, so a call token that
/// appears *inside* a string (`cmd:"rg 'exec_command(' src"`, or a patch
/// whose body mentions `apply_patch(`) is neither a call nor a boundary.
/// An unterminated call runs to the end of the script.
fn call_argument_texts<'a>(script: &'a str, call: &str) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut in_string: Option<char> = None;
    let mut escaped = false;
    // (start of the argument text, paren depth) while inside a call.
    let mut current: Option<(usize, usize)> = None;

    for (i, c) in script.char_indices() {
        if let Some(quote) = in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == quote {
                in_string = None;
            }
            continue;
        }
        match c {
            '"' | '\'' | '`' => in_string = Some(c),
            '(' => {
                if let Some((_, depth)) = current.as_mut() {
                    *depth += 1;
                }
            }
            ')' => {
                if let Some((start, depth)) = current.as_mut() {
                    *depth -= 1;
                    if *depth == 0 {
                        out.push(&script[*start..i]);
                        current = None;
                    }
                }
            }
            _ => {}
        }
        if current.is_none() && script[i..].starts_with(call) {
            // `call` ends with the opening paren; the loop will not see
            // that paren again because we jump past it via the depth start.
            current = Some((i + call.len(), 1));
        }
    }
    if let Some((start, _)) = current {
        out.push(&script[start..]);
    }
    out
}

/// The string literal assigned to `key` in a JavaScript object literal
/// (`{cmd: "…"}`, `{"cmd": '…'}`, `{ cmd : `…` }`). `None` when the key is
/// absent, is only part of a longer identifier, or its value is not a
/// literal.
fn find_key_literal(body: &str, key: &str) -> Option<String> {
    let mut search_from = 0;
    while let Some(rel) = body[search_from..].find(key) {
        let key_start = search_from + rel;
        let key_end = key_start + key.len();
        search_from = key_end;

        // The key must be a whole identifier (or a quoted key), not part of
        // one like `cmdline`.
        let before = body[..key_start].chars().next_back();
        let quoted_key = matches!(before, Some('"') | Some('\''));
        if !quoted_key && before.is_some_and(|c| c.is_alphanumeric() || c == '_') {
            continue;
        }
        let mut rest = &body[key_end..];
        if quoted_key {
            let closing = before.unwrap();
            rest = match rest.strip_prefix(closing) {
                Some(r) => r,
                None => continue,
            };
        } else if rest
            .chars()
            .next()
            .is_some_and(|c| c.is_alphanumeric() || c == '_')
        {
            continue;
        }
        let rest = rest.trim_start();
        let Some(rest) = rest.strip_prefix(':') else {
            continue;
        };
        // A non-literal value (a variable, a template with substitutions
        // we cannot evaluate) — nothing to lift.
        return parse_js_string_literal(rest.trim_start());
    }
    None
}

/// Decode a JavaScript string literal (`"…"`, `'…'`, or `` `…` ``) at the
/// start of `s`. Handles the escapes Codex actually emits (`\n`, `\t`,
/// `\"`, `\\`, `\u{XXXX}`/`\uXXXX`, `\xXX`, line continuations); unknown
/// escapes drop the backslash. Template literals keep `${…}` verbatim.
fn parse_js_string_literal(s: &str) -> Option<String> {
    let mut chars = s.chars().peekable();
    let quote = chars.next()?;
    if !matches!(quote, '"' | '\'' | '`') {
        return None;
    }
    let mut out = String::new();
    loop {
        let c = chars.next()?; // unterminated → None
        if c == quote {
            return Some(out);
        }
        if c != '\\' {
            out.push(c);
            continue;
        }
        let e = chars.next()?;
        match e {
            'n' => out.push('\n'),
            't' => out.push('\t'),
            'r' => out.push('\r'),
            'b' => out.push('\u{8}'),
            'f' => out.push('\u{c}'),
            'v' => out.push('\u{b}'),
            '0' => out.push('\0'),
            '\n' => {} // line continuation
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
            }
            'x' => {
                let hex: String = (0..2).filter_map(|_| chars.next()).collect();
                match u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                    Some(ch) => out.push(ch),
                    None => {
                        out.push('x');
                        out.push_str(&hex);
                    }
                }
            }
            'u' => {
                let hex: String = if chars.peek() == Some(&'{') {
                    chars.next();
                    let h: String = chars.by_ref().take_while(|c| *c != '}').collect();
                    h
                } else {
                    (0..4).filter_map(|_| chars.next()).collect()
                };
                match u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                    Some(ch) => out.push(ch),
                    None => {
                        out.push('u');
                        out.push_str(&hex);
                    }
                }
            }
            other => out.push(other), // \" \' \` \\ \$ and anything else
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn first_cmd(script: &str) -> Option<String> {
        extract_exec_commands(script).into_iter().next()
    }

    #[test]
    fn exec_command_double_quoted_with_escapes() {
        let script = r#"text(await tools.exec_command({cmd:"grep -n \"foo\" src/*.rs\nls -la",max_output_tokens:5000}));"#;
        assert_eq!(
            first_cmd(script).as_deref(),
            Some("grep -n \"foo\" src/*.rs\nls -la")
        );
    }

    #[test]
    fn exec_commands_all_calls_in_order() {
        let script = "text(await tools.exec_command({cmd:\"ls\"}));\ntext(await tools.exec_command({cmd:'pwd', workdir:'/x'}));\nawait tools.exec_command({cmd: v});\n";
        assert_eq!(extract_exec_commands(script), vec!["ls", "pwd"]);
    }

    #[test]
    fn exec_patches_bare_string_and_object_forms() {
        let script = "text(await tools.apply_patch(\"*** Begin Patch\\n*** Add File: a.md\\n+hi\\n*** End Patch\\n\"));\ntext(await tools.exec_command({cmd:\"ls\"}));\nawait tools.apply_patch({patch: '*** Begin Patch\\n*** Update File: b.rs\\n*** End Patch'});";
        let patches = extract_exec_patches(script);
        assert_eq!(patches.len(), 2);
        assert_eq!(patch_file_paths(&patches[0]), vec!["a.md"]);
        assert_eq!(patch_file_paths(&patches[1]), vec!["b.rs"]);
        assert!(extract_exec_patches("tools.exec_command({cmd:\"ls\"})").is_empty());
        // A patch argument that is not a literal is skipped, not mis-parsed.
        assert!(extract_exec_patches("tools.apply_patch(patchText)").is_empty());
    }

    #[test]
    fn exec_command_single_quoted_and_spaced_key() {
        let script =
            "const r = await tools.exec_command({ cmd : 'echo it\\'s fine', workdir: '/x' });";
        assert_eq!(first_cmd(script).as_deref(), Some("echo it's fine"));
    }

    #[test]
    fn exec_command_quoted_key_and_template_literal() {
        let script = "await tools.exec_command({\"cmd\": `cat ${f}`});";
        assert_eq!(first_cmd(script).as_deref(), Some("cat ${f}"));
    }

    #[test]
    fn exec_command_ignores_cmdline_identifier() {
        let script = "await tools.exec_command({cmdline: 1, cmd: \"true\"});";
        assert_eq!(first_cmd(script).as_deref(), Some("true"));
    }

    #[test]
    fn exec_command_absent_or_unparseable() {
        assert!(
            extract_exec_commands("image((await tools.view_image({path:'/a.png'})).image_url);")
                .is_empty()
        );
        assert!(extract_exec_commands("await tools.exec_command({cmd: someVar});").is_empty());
        assert!(extract_exec_commands("await tools.exec_command({cmd: \"unterminated").is_empty());
        assert!(extract_exec_commands("").is_empty());
    }

    #[test]
    fn exec_command_unicode_escapes() {
        let script = "tools.exec_command({cmd:\"echo \\u00e9\\x41\\u{1F600}\"})";
        assert_eq!(first_cmd(script).as_deref(), Some("echo éA😀"));
    }

    #[test]
    fn patch_paths_in_order_deduplicated() {
        let patch = "*** Begin Patch\n*** Update File: src/a.rs\n@@\n-x\n+y\n*** Add File: docs/b.md\n+hi\n*** Update File: src/a.rs\n*** Delete File: old.txt\n*** End Patch\n";
        assert_eq!(
            patch_file_paths(patch),
            vec!["src/a.rs", "docs/b.md", "old.txt"]
        );
        assert!(patch_file_paths("no headers here").is_empty());
    }

    #[test]
    fn session_id_from_rollout_name_or_stem() {
        let p = Path::new(
            "/x/sessions/2026/09/07/rollout-2026-09-07T13-19-45-01a07d86-dc97-76c3-adcc-dd2942bbd177.jsonl",
        );
        assert_eq!(
            session_id_from_filename(p),
            "01a07d86-dc97-76c3-adcc-dd2942bbd177"
        );
        assert_eq!(
            session_id_from_filename(Path::new("/x/renamed-thread.jsonl")),
            "renamed-thread"
        );
        // Ends in something UUID-shaped but not preceded by a dash.
        assert_eq!(
            session_id_from_filename(Path::new(
                "/x/rollout_01a07d86-dc97-76c3-adcc-dd2942bbd177.jsonl"
            )),
            "rollout_01a07d86-dc97-76c3-adcc-dd2942bbd177"
        );
    }

    #[test]
    fn success_needs_every_envelope_to_be_zero() {
        let two = "{\"exit_code\":0,\"output\":\"a\"}\n{\"exit_code\":1,\"output\":\"b\"}";
        assert_eq!(success_from_output(two), Some(false));
        let both_ok = "{\"exit_code\":0}{\"exit_code\": 0}";
        assert_eq!(success_from_output(both_ok), Some(true));
        // An escaped key inside a command's own output is not a verdict.
        let nested = "{\"exit_code\":0,\"output\":\"{\\\"exit_code\\\":2}\"}";
        assert_eq!(success_from_output(nested), Some(true));
    }

    #[test]
    fn call_token_inside_a_string_is_not_a_call() {
        let script = "text(await tools.exec_command({cmd:\"rg -n 'exec_command(' src\"}));\ntext(await tools.exec_command({cmd:\"ls\"}));";
        assert_eq!(
            extract_exec_commands(script),
            vec!["rg -n 'exec_command(' src", "ls"]
        );
        let patch_script = "text(await tools.apply_patch(\"*** Begin Patch\\n*** Update File: codex.rs\\n+    call_argument_texts(script, \\\"apply_patch(\\\")\\n*** End Patch\\n\"));\ntext(await tools.exec_command({cmd:\"cargo test\"}));";
        let patches = extract_exec_patches(patch_script);
        assert_eq!(patches.len(), 1);
        assert_eq!(patch_file_paths(&patches[0]), vec!["codex.rs"]);
        assert_eq!(extract_exec_commands(patch_script), vec!["cargo test"]);
        // Unterminated call still yields its literal.
        assert_eq!(
            extract_exec_commands("tools.exec_command({cmd:\"pwd\""),
            vec!["pwd"]
        );
    }

    #[test]
    fn success_parsed_from_exec_envelope() {
        assert_eq!(
            success_from_output(
                "Script completed\n{\"chunk_id\":\"a\",\"exit_code\":0,\"output\":\"\"}"
            ),
            Some(true)
        );
        assert_eq!(success_from_output("{\"exit_code\": 127}"), Some(false));
        assert_eq!(success_from_output("{\"accepted\":true}"), None);
    }

    #[test]
    fn entrypoint_from_source_shapes() {
        let m = |v: Value| v.as_object().unwrap().clone();
        assert_eq!(
            entrypoint_of(&m(json!({"source": "vscode"}))).as_deref(),
            Some("vscode")
        );
        assert_eq!(
            entrypoint_of(&m(
                json!({"source": {"subagent": {"depth": 1}}, "thread_source": "guardian_review"})
            ))
            .as_deref(),
            Some("subagent")
        );
        assert_eq!(
            entrypoint_of(&m(json!({"thread_source": "subagent"}))).as_deref(),
            Some("subagent")
        );
        // Legacy guardian threads (no `source`) are spawned threads too.
        assert_eq!(
            entrypoint_of(&m(json!({"thread_source": "guardian_review"}))).as_deref(),
            Some("subagent")
        );
        assert_eq!(entrypoint_of(&m(json!({"thread_source": "user"}))), None);
        assert_eq!(entrypoint_of(&m(json!({}))), None);
    }

    #[test]
    fn typed_text_keeps_only_user_text_items_when_kinds_are_parallel() {
        let m = |v: Value| v.as_object().unwrap().clone();
        let mixed = m(json!({
            "content": [
                {"type": "input_text", "text": "<environment_context>\n<cwd>/p</cwd>\n</environment_context>"},
                {"type": "input_text", "text": "fix the bug"},
                {"type": "input_image", "image_url": "data:..."}
            ],
            "internal_chat_message_metadata_passthrough": {
                "content_item_kinds": ["environments.environment_context", "user.text", "user.image"]
            }
        }));
        assert_eq!(typed_user_text(&mixed).as_deref(), Some("fix the bug"));
        // Several typed chunks are newline-separated, never glued.
        let chunks = m(json!({
            "content": [
                {"type": "input_text", "text": "line one"},
                {"type": "input_text", "text": "line two\n"},
                {"type": "input_text", "text": "line three"}
            ],
            "internal_chat_message_metadata_passthrough": {
                "content_item_kinds": ["user.text", "user.text", "user.text"]
            }
        }));
        assert_eq!(
            typed_user_text(&chunks).as_deref(),
            Some("line one\nline two\nline three")
        );
        // Kinds that do not line up with content: keep everything typed.
        let skewed = m(json!({
            "content": [{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}],
            "internal_chat_message_metadata_passthrough": {"content_item_kinds": ["user.text"]}
        }));
        assert_eq!(typed_user_text(&skewed).as_deref(), Some("a\nb"));
    }

    #[test]
    fn typed_user_detection_with_and_without_kinds() {
        let m = |v: Value| v.as_object().unwrap().clone();
        assert!(
            typed_user_text(&m(json!({
                "content": [{"type": "input_text", "text": "<x>"}],
                "internal_chat_message_metadata_passthrough": {"content_item_kinds": ["user.text"]}
            })))
            .is_some()
        );
        assert!(!typed_user_text(&m(json!({
            "content": [{"type": "input_text", "text": "hello"}],
            "internal_chat_message_metadata_passthrough": {"content_item_kinds": ["environments.environment_context"]}
        }))).is_some());
        // Legacy (no kinds): a lone XML block is injected context…
        assert!(!typed_user_text(&m(json!({
            "content": [{"type": "input_text", "text": "<environment_context>\n<cwd>/p</cwd>\n</environment_context>"}]
        }))).is_some());
        // …but typed text is typed text, even when it mentions a tag.
        assert!(
            typed_user_text(&m(json!({
                "content": [{"type": "input_text", "text": "why does <div> render twice?"}]
            })))
            .is_some()
        );
    }
}
