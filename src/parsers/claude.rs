//! Claude Code transcript parser.
//!
//! Handles transcripts from `~/.claude/projects/`.

use std::fs;
use std::path::{Path, PathBuf};

use chrono::{DateTime, FixedOffset};
use serde_json::{Map, Value, json};

use super::{Parser, strip_ws, read_text, take_chars};
use crate::models::{Event, EventType, MessageRole, ParsedTranscript, SessionMetadata, display_str};
use crate::utils::parse_datetime;

/// Parser for Claude Code session transcripts.
pub struct ClaudeParser;

const CLI_SOURCE: &str = "claude_code";
/// Bumped to 1.1 when the parser started extracting `cwd` from transcript
/// entries; a bump re-indexes sessions parsed by an older version.
const PARSER_VERSION: &str = "1.1";

impl Parser for ClaudeParser {
    fn cli_source(&self) -> &'static str {
        CLI_SOURCE
    }

    fn parser_version(&self) -> &'static str {
        PARSER_VERSION
    }

    /// Discover Claude Code session transcripts: every `*.jsonl` under the
    /// base whose path does not contain "subagents" and which sits exactly at
    /// `project-dir/session.jsonl` (two path components below the base).
    fn discover_sessions(&self, base_path: &Path) -> Vec<PathBuf> {
        let mut all = Vec::new();
        rglob_jsonl(base_path, &mut all, 0);
        all.into_iter()
            .filter(|p| {
                // Substring match over the FULL path string, base included —
                // not just the file name.
                if p.to_string_lossy().contains("subagents") {
                    return false;
                }
                match p.strip_prefix(base_path) {
                    Ok(rel) => rel.components().count() == 2,
                    Err(_) => false,
                }
            })
            .collect()
    }

    /// Extract session metadata. Project path and git branch come from the
    /// `cwd`/`gitBranch` fields of transcript entries; a sibling
    /// `sessions-index.json` may override first prompt, message count, and
    /// timestamps.
    fn extract_metadata(&self, transcript_path: &Path) -> SessionMetadata {
        let session_id = transcript_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();

        // Check for sessions-index.json for additional metadata.
        let index_data = load_index_entry(transcript_path, &session_id);

        let mut project_path: Option<String> = None;
        let mut git_branch: Option<String> = index_data
            .as_ref()
            .and_then(|d| d.get("gitBranch"))
            .and_then(Value::as_str)
            .map(String::from);
        let mut entrypoint: Option<String> = None;
        let mut first_prompt = String::new();
        let mut created_at: Option<DateTime<FixedOffset>> = None;
        let mut modified_at: Option<DateTime<FixedOffset>> = None;
        let mut message_count: i64 = 0;

        if let Some(text) = read_text(transcript_path) {
            let mut first_timestamp_seen = false;
            let mut last_timestamp: Option<DateTime<FixedOffset>> = None;

            for line in text.lines() {
                let Ok(entry) = serde_json::from_str::<Value>(line) else {
                    continue;
                };
                let Some(entry) = entry.as_object() else {
                    // A JSON line that is not an object is skipped.
                    continue;
                };
                let entry_type = entry.get("type").and_then(Value::as_str).unwrap_or("");

                // Extract cwd from the first entry that has it (the actual
                // project path). Only an unset value is replaceable, so a
                // present-but-null cwd leaves it unset while "" sticks.
                if project_path.is_none()
                    && let Some(s) = entry.get("cwd").and_then(Value::as_str)
                {
                    project_path = Some(s.to_string());
                }

                // How the session was started (`cli`, `claude-desktop`,
                // `sdk-cli`); first non-empty value wins.
                if entrypoint.is_none()
                    && let Some(s) = entry.get("entrypoint").and_then(Value::as_str)
                    && !s.is_empty()
                {
                    entrypoint = Some(s.to_string());
                }

                // Extract gitBranch from the entry-level field. Unlike cwd,
                // an empty value is still replaceable by a later entry.
                if git_branch.as_deref().is_none_or(str::is_empty)
                    && let Some(s) = entry.get("gitBranch").and_then(Value::as_str)
                {
                    git_branch = Some(s.to_string());
                }

                if let Some(ts) = entry
                    .get("timestamp")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .and_then(parse_datetime)
                {
                    if !first_timestamp_seen {
                        created_at = Some(ts);
                        first_timestamp_seen = true;
                    }
                    last_timestamp = Some(ts);
                }

                // Extract first user message.
                if entry_type == "user" && first_prompt.is_empty() {
                    let msg = entry.get("message").and_then(Value::as_object);
                    match msg.and_then(|m| m.get("content")) {
                        Some(Value::String(s)) => first_prompt = take_chars(s, 500),
                        Some(Value::Array(blocks)) => {
                            for block in blocks {
                                if block.get("type").and_then(Value::as_str) == Some("text") {
                                    let text =
                                        block.get("text").and_then(Value::as_str).unwrap_or("");
                                    first_prompt = take_chars(text, 500);
                                    break;
                                }
                            }
                        }
                        _ => {}
                    }
                }

                // Count user/assistant messages.
                if entry_type == "user" || entry_type == "assistant" {
                    message_count += 1;
                }
            }

            modified_at = last_timestamp;
        }

        // Use index data if available (can override transcript-derived values).
        if let Some(index) = &index_data {
            if let Some(fp) = index
                .get("firstPrompt")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                first_prompt = take_chars(fp, 500);
            }
            if let Some(mc) = index
                .get("messageCount")
                .and_then(Value::as_i64)
                .filter(|&n| n != 0)
            {
                message_count = mc;
            }
            if let Some(ts) = index
                .get("created")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .and_then(parse_datetime)
            {
                created_at = Some(ts);
            }
            if let Some(ts) = index
                .get("lastModified")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .and_then(parse_datetime)
            {
                modified_at = Some(ts);
            }
        }

        let mut meta = SessionMetadata::new(
            session_id,
            CLI_SOURCE,
            transcript_path.to_string_lossy().into_owned(),
        );
        meta.first_prompt = Some(first_prompt);
        meta.project_path = Some(project_path.unwrap_or_default());
        meta.git_branch = git_branch;
        meta.entrypoint = entrypoint;
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
                for event in parse_entry(&entry, sequence) {
                    events.push(event);
                    sequence += 1;
                }
            }
        }

        metadata.event_count = events.len() as i64;
        ParsedTranscript { metadata, events }
    }
}

/// Maximum directory nesting `rglob_jsonl` will descend. Real Claude Code
/// layouts are two levels deep; the cap is pure belt-and-braces against
/// pathological trees (and, historically, symlink cycles) blowing the stack.
const MAX_RGLOB_DEPTH: usize = 32;

/// Recursively collect `*.jsonl` files, sorted so discovery order does not
/// depend on the filesystem.
///
/// Directory symlinks are NOT followed: the `is_dir` test uses
/// `symlink_metadata` rather than the symlink-following `Path::is_dir`, so a
/// `.jsonl` inside a symlinked directory is not yielded. Symlinks *to files*
/// are still yielded.
///
/// Because directory symlinks are never followed, a symlink cycle cannot loop
/// here; `MAX_RGLOB_DEPTH` is nonetheless enforced so no future change (or
/// exotic filesystem) can recurse without bound.
fn rglob_jsonl(dir: &Path, out: &mut Vec<PathBuf>, depth: usize) {
    if depth >= MAX_RGLOB_DEPTH {
        return;
    }
    let Ok(read_dir) = fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = read_dir.flatten().map(|e| e.path()).collect();
    entries.sort();
    for path in entries {
        // `symlink_metadata` does not traverse the final symlink, so a
        // symlinked directory reports as a symlink (not a dir) and is skipped.
        let is_real_dir = fs::symlink_metadata(&path).is_ok_and(|m| m.is_dir());
        if is_real_dir {
            rglob_jsonl(&path, out, depth + 1);
        } else if path.extension().and_then(|e| e.to_str()) == Some("jsonl") && path.is_file() {
            out.push(path);
        }
    }
}

/// Load the entry for `session_id` from a sibling `sessions-index.json`, if
/// the file exists, parses to a list, and contains a matching dict.
fn load_index_entry(transcript_path: &Path, session_id: &str) -> Option<Map<String, Value>> {
    let index_path = transcript_path.parent()?.join("sessions-index.json");
    let text = read_text(&index_path)?;
    let Ok(Value::Array(sessions)) = serde_json::from_str::<Value>(&text) else {
        return None;
    };
    for s in sessions {
        if let Value::Object(obj) = s
            && obj.get("sessionId").and_then(Value::as_str) == Some(session_id)
        {
            return Some(obj);
        }
    }
    None
}

/// Parse a single JSONL entry into events.
fn parse_entry(entry: &Value, start_sequence: i64) -> Vec<Event> {
    let mut events: Vec<Event> = Vec::new();
    let Some(entry) = entry.as_object() else {
        return events;
    };
    let entry_type = entry.get("type").and_then(Value::as_str).unwrap_or("");
    let timestamp = entry
        .get("timestamp")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .and_then(parse_datetime);

    if entry_type == "user" {
        // User message. An absent `message` key and a present non-object one
        // are both treated as "no message": an empty object would give
        // `content=''`, which strips to empty and emits no events — exactly
        // what the early return below produces, so the shortcut is safe here.
        // (Contrast copilot's `data`, where an empty object *does* emit an
        // event, so the two cases must stay distinct there.)
        let msg = match entry.get("message") {
            Some(Value::Object(m)) => m,
            Some(_) => return events,
            None => {
                return events;
            }
        };
        let content = msg.get("content").unwrap_or(&Value::Null);

        // Handle both string and array content.
        let mut text_content = String::new();
        match content {
            Value::String(s) => text_content = s.clone(),
            Value::Array(blocks) => {
                for block in blocks {
                    if block.is_object()
                        && block.get("type").and_then(Value::as_str) == Some("text")
                    {
                        text_content += block.get("text").and_then(Value::as_str).unwrap_or("");
                    }
                    // tool_result blocks in user messages are skipped.
                }
            }
            _ => {}
        }

        let stripped = strip_ws(&text_content);
        if !stripped.is_empty() {
            let mut e = Event::new(
                format!("user_{start_sequence}"),
                start_sequence,
                EventType::Message,
            );
            e.timestamp = timestamp;
            e.role = Some(MessageRole::User);
            e.content = stripped.to_string();
            events.push(e);
        }
    } else if entry_type == "assistant" {
        // Same shape as above: an absent or non-object `message` would give
        // an empty content list, which emits nothing — identical to the early
        // return here.
        let msg = match entry.get("message") {
            Some(Value::Object(m)) => m,
            Some(_) => return events,
            None => return events,
        };
        // A missing `content` key takes the (empty) list branch.
        let content = msg.get("content").cloned().unwrap_or_else(|| json!([]));

        match content {
            Value::String(s) => {
                // String content is emitted unconditionally (even empty).
                let mut e = Event::new(
                    format!("assistant_{start_sequence}"),
                    start_sequence,
                    EventType::Message,
                );
                e.timestamp = timestamp;
                e.role = Some(MessageRole::Assistant);
                e.content = s;
                events.push(e);
            }
            Value::Array(blocks) => {
                let mut text_parts: Vec<String> = Vec::new();
                let mut tool_events: Vec<Event> = Vec::new();

                for block in &blocks {
                    let Some(block) = block.as_object() else {
                        continue;
                    };
                    match block.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            text_parts.push(
                                block
                                    .get("text")
                                    .and_then(Value::as_str)
                                    .unwrap_or("")
                                    .to_string(),
                            );
                        }
                        Some("tool_use") => {
                            let tool_input = match block.get("input") {
                                Some(Value::String(s)) => json!({ "content": s }),
                                Some(v @ Value::Object(_)) => v.clone(),
                                _ => json!({}),
                            };
                            // A missing OR present-but-null `id` falls back to
                            // the generated `tool_{start}_{n}`, because NULL
                            // event ids must never reach the database. Pinned
                            // by `claude_null_tool_use_id_generates_fallback`.
                            // A non-string id is stringified — everything
                            // downstream treats ids as text anyway.
                            let tool_call_id = match block.get("id") {
                                None | Some(Value::Null) => None,
                                Some(v) => Some(display_str(v)),
                            };
                            let id = tool_call_id.clone().unwrap_or_else(|| {
                                format!("tool_{}_{}", start_sequence, tool_events.len())
                            });
                            let mut e = Event::new(
                                id,
                                start_sequence + tool_events.len() as i64 + 1,
                                EventType::ToolCall,
                            );
                            e.timestamp = timestamp;
                            // A non-string tool name survives, stringified for
                            // the same reason as `id`; absent/null stays None.
                            e.tool_name = match block.get("name") {
                                None | Some(Value::Null) => None,
                                Some(v) => Some(display_str(v)),
                            };
                            e.tool_input = Some(tool_input);
                            e.tool_call_id = tool_call_id;
                            tool_events.push(e);
                        }
                        _ => {}
                    }
                }

                // Add text message first.
                if !text_parts.is_empty() {
                    let mut e = Event::new(
                        format!("assistant_{start_sequence}"),
                        start_sequence,
                        EventType::Message,
                    );
                    e.timestamp = timestamp;
                    e.role = Some(MessageRole::Assistant);
                    e.content = text_parts.join("\n");
                    events.push(e);
                }

                // Then add tool calls.
                events.extend(tool_events);
            }
            _ => {}
        }
    }
    // 'system' and 'progress' entries carry no conversation content.

    events
}
