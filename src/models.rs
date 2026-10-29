//! Unified data models.
//!
//! The canonical representation is the [`Event`]; UI/LLM "messages" are
//! derived from events. The strings produced here (conversation text, tool
//! one-liners) are embedded verbatim in LLM prompts, so their exact bytes
//! matter and are pinned by the golden tests — change them deliberately.
//!
//! Truncation limits are deliberately generous: full Bash/terminal commands
//! up to 500 chars rather than the first line capped at 80, and generic
//! string args capped at 200 rather than 40. Tighter caps were measured to
//! discard most of the command/argument text before it ever reached the
//! summarizer, Q&A, or FTS.

use chrono::{DateTime, FixedOffset};
use serde_json::Value;

/// Types of events in a transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventType {
    /// User or assistant message
    Message,
    /// Tool invocation
    ToolCall,
    /// Tool execution result
    ToolResult,
    /// Files created, patches, diffs
    Artifact,
    /// Model switch, environment info
    Meta,
    /// Errors during execution
    Error,
    /// Progress updates
    Progress,
}

impl EventType {
    /// The string value persisted and rendered for this variant.
    pub fn as_str(&self) -> &'static str {
        match self {
            EventType::Message => "message",
            EventType::ToolCall => "tool_call",
            EventType::ToolResult => "tool_result",
            EventType::Artifact => "artifact",
            EventType::Meta => "meta",
            EventType::Error => "error",
            EventType::Progress => "progress",
        }
    }
}

/// Roles for message events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

impl MessageRole {
    /// The string value persisted and rendered for this variant.
    pub fn as_str(&self) -> &'static str {
        match self {
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::System => "system",
        }
    }
}

/// A single event in a transcript.
///
/// This is the canonical representation — UI "messages" are derived from
/// events.
#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    /// Unique event ID
    pub id: String,
    /// Monotonic ordering (timestamps unreliable across CLIs)
    pub sequence: i64,
    /// Type of event
    pub event_type: EventType,
    pub timestamp: Option<DateTime<FixedOffset>>,

    // For MESSAGE events
    pub role: Option<MessageRole>,
    pub content: String,

    // For TOOL_CALL events. `tool_input` is a JSON value rather than a map
    // because some tools carry non-object inputs (e.g. apply_patch raw strings).
    pub tool_name: Option<String>,
    pub tool_input: Option<Value>,
    pub tool_call_id: Option<String>,

    // For TOOL_RESULT events
    pub tool_output: Option<String>,
    pub tool_success: Option<bool>,

    // For ARTIFACT events ('file', 'diff', 'image')
    pub artifact_path: Option<String>,
    pub artifact_type: Option<String>,

    /// Traceability back to raw transcript: line number, byte offset, or
    /// source event ID.
    pub raw_ref: Option<String>,
}

impl Event {
    /// Construct an event with the required fields; everything else is empty
    /// or absent.
    pub fn new(id: impl Into<String>, sequence: i64, event_type: EventType) -> Self {
        Event {
            id: id.into(),
            sequence,
            event_type,
            timestamp: None,
            role: None,
            content: String::new(),
            tool_name: None,
            tool_input: None,
            tool_call_id: None,
            tool_output: None,
            tool_success: None,
            artifact_path: None,
            artifact_type: None,
            raw_ref: None,
        }
    }
}

/// Session index record — what we store in the database.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionMetadata {
    pub session_id: String,
    /// 'claude_code', 'github_copilot', etc.
    pub cli_source: String,
    /// Path to raw transcript file
    pub transcript_path: String,

    // Display fields
    pub title: Option<String>,
    pub summary: Option<String>,
    pub first_prompt: Option<String>,

    // Project context
    pub project_path: Option<String>,
    pub git_branch: Option<String>,
    /// For cross-machine identification
    pub git_remote: Option<String>,

    // Timestamps
    pub created_at: Option<DateTime<FixedOffset>>,
    pub modified_at: Option<DateTime<FixedOffset>>,

    // Counts
    pub message_count: i64,
    pub event_count: i64,
    pub files_touched_count: i64,
    pub token_count: i64,

    // Versioning / provenance
    pub parser_version: String,
    pub normalized_version: i64,
    /// Hash of raw transcript for change detection
    pub source_hash: Option<String>,

    // Optional metadata
    /// Primary model used
    pub model: Option<String>,
    /// Session duration
    pub duration_ms: Option<i64>,
    pub tags: Vec<String>,

    /// Derived search text (for FTS)
    pub tool_timeline: Option<String>,

    /// How the session was started, when the transcript says: Claude Code
    /// records `"cli"` (interactive terminal), `"claude-desktop"`, or
    /// `"sdk-cli"` (`claude -p` / Agent SDK one-shots). Not persisted —
    /// the indexer uses it to skip non-interactive sessions
    /// ([`crate::indexer::is_interactive`]).
    pub entrypoint: Option<String>,
}

impl SessionMetadata {
    /// Construct metadata with the required fields; everything else takes its
    /// default (`parser_version` "1.0", `normalized_version` 1, counts 0).
    pub fn new(
        session_id: impl Into<String>,
        cli_source: impl Into<String>,
        transcript_path: impl Into<String>,
    ) -> Self {
        SessionMetadata {
            session_id: session_id.into(),
            cli_source: cli_source.into(),
            transcript_path: transcript_path.into(),
            title: None,
            summary: None,
            first_prompt: None,
            project_path: None,
            git_branch: None,
            git_remote: None,
            created_at: None,
            modified_at: None,
            message_count: 0,
            event_count: 0,
            files_touched_count: 0,
            token_count: 0,
            parser_version: "1.0".to_string(),
            normalized_version: 1,
            source_hash: None,
            model: None,
            duration_ms: None,
            tags: Vec::new(),
            tool_timeline: None,
            entrypoint: None,
        }
    }
}

/// A displayable message derived from events. `role` is one of "user",
/// "assistant", "system", "tools", or "unknown".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Result of parsing a transcript file. Contains both metadata and events.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedTranscript {
    pub metadata: SessionMetadata,
    pub events: Vec<Event>,
}

impl ParsedTranscript {
    /// Derive displayable messages from events.
    /// Groups tool calls with their parent assistant message.
    pub fn get_messages(&self) -> Vec<Message> {
        let mut messages: Vec<Message> = Vec::new();
        let mut current_tools: Vec<String> = Vec::new();

        for event in &self.events {
            match event.event_type {
                EventType::Message => {
                    // Flush any pending tool calls first
                    if !current_tools.is_empty() {
                        messages.push(Message {
                            role: "tools".to_string(),
                            content: current_tools.join("\n"),
                        });
                        current_tools.clear();
                    }
                    messages.push(Message {
                        role: event
                            .role
                            .map(|r| r.as_str().to_string())
                            .unwrap_or_else(|| "unknown".to_string()),
                        content: event.content.clone(),
                    });
                }
                EventType::ToolCall => {
                    current_tools.push(format_tool_call(event));
                }
                // Tool results are usually implicit in the flow; every other
                // event type is ignored here.
                _ => {}
            }
        }

        // Flush remaining tool calls
        if !current_tools.is_empty() {
            messages.push(Message {
                role: "tools".to_string(),
                content: current_tools.join("\n"),
            });
        }

        messages
    }

    /// Get plain text representation for LLM context.
    pub fn get_conversation_text(&self, include_tools: bool) -> String {
        let mut lines: Vec<String> = Vec::new();
        for msg in self.get_messages() {
            match msg.role.as_str() {
                "user" => lines.push(format!("Human:\n{}\n", msg.content)),
                "assistant" => lines.push(format!("Assistant:\n{}\n", msg.content)),
                "tools" if include_tools => {
                    lines.push(format!("[Tool Calls]\n{}\n", msg.content));
                }
                _ => {}
            }
        }
        lines.join("\n")
    }
}

/// Truncate to at most `n` characters — characters, never bytes.
pub(crate) fn truncate_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// Truthiness for JSON values: null, `false`, zero, the empty string and
/// empty containers are falsy; everything else is truthy.
pub(crate) fn is_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Length of a JSON value: strings count characters, containers count
/// elements. Anything else has no meaningful length and reports 0.
fn value_len(v: &Value) -> usize {
    match v {
        Value::String(s) => s.chars().count(),
        Value::Array(a) => a.len(),
        Value::Object(o) => o.len(),
        _ => 0,
    }
}

/// Quoted, escaped form of a string, used inside container reprs. Prefers
/// single quotes, switching to double quotes when the string contains a
/// single quote but no double quote; control characters escape to `\xNN`.
fn string_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        if c == '\\' {
            out.push_str("\\\\");
        } else if c == '\n' {
            out.push_str("\\n");
        } else if c == '\r' {
            out.push_str("\\r");
        } else if c == '\t' {
            out.push_str("\\t");
        } else if c == quote {
            out.push('\\');
            out.push(c);
        } else if (c as u32) < 0x20 || c as u32 == 0x7f {
            out.push_str(&format!("\\x{:02x}", c as u32));
        } else {
            out.push(c);
        }
    }
    out.push(quote);
    out
}

/// Shortest round-trip rendering of a float, keeping the trailing `.0` on
/// whole values. Rust's `{:?}` already produces that; only exponent notation
/// is normalized (`1e16` → `1e+16`, `1e-5` → `1e-05`).
fn float_repr(f: f64) -> String {
    if f.is_nan() {
        return "nan".to_string();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    let s = format!("{f:?}");
    if let Some(pos) = s.find('e') {
        let (mantissa, exp) = s.split_at(pos);
        let exp = &exp[1..];
        let (sign, digits) = match exp.strip_prefix('-') {
            Some(d) => ("-", d),
            None => ("+", exp.strip_prefix('+').unwrap_or(exp)),
        };
        let digits = if digits.len() < 2 {
            format!("0{digits}")
        } else {
            digits.to_string()
        };
        return format!("{mantissa}e{sign}{digits}");
    }
    s
}

/// Unambiguous rendering of a JSON value for prompt text: `None` / `True` /
/// `False` for null and booleans, [`string_repr`] for strings, and
/// `[a, b]` / `{k: v}` for containers.
fn value_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                float_repr(n.as_f64().unwrap_or(f64::NAN))
            }
        }
        Value::String(s) => string_repr(s),
        Value::Array(a) => {
            let items: Vec<String> = a.iter().map(value_repr).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Object(o) => {
            let items: Vec<String> = o
                .iter()
                .map(|(k, val)| format!("{}: {}", string_repr(k), value_repr(val)))
                .collect();
            format!("{{{}}}", items.join(", "))
        }
    }
}

/// Render a JSON value for interpolation into prompt text: strings render
/// raw (no quotes), everything else falls back to [`value_repr`].
pub(crate) fn display_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => value_repr(other),
    }
}

/// Look a key up in `tool_input` — key presence, not truthiness. A
/// non-object input yields `None` rather than an error (unreachable via the
/// real parsers, but tool inputs are not guaranteed to be objects).
fn get<'a>(input: &'a Value, key: &str) -> Option<&'a Value> {
    input.as_object().and_then(|o| o.get(key))
}

/// `tool_input.get(key, default)` rendered as a string.
fn get_or(input: &Value, key: &str, default: &str) -> String {
    match get(input, key) {
        Some(v) => display_str(v),
        None => default.to_string(),
    }
}

/// Render a shell command for the one-line tool timeline: full text up to
/// 500 chars, newlines joined with `; ` so the line-per-tool structure (and
/// per-line FTS grep) survives. Keeping only the first line capped at 80
/// chars was measured to drop 73% of real Bash command text before it
/// reached any LLM or index, so the whole command is kept instead.
fn flatten_command(cmd: &str) -> String {
    let joined = cmd
        .split('\n')
        .map(str::trim_end)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("; ");
    truncate_chars(&joined, 500).to_string()
}

/// Fernet tokens (`gAAAAA` + base64) are how Codex Desktop encrypts
/// inter-agent messages at rest; they carry nothing a reader or an index
/// can use.
fn looks_encrypted(s: &str) -> bool {
    s.len() > 32
        && s.starts_with("gAAAAA")
        && s
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'=')
}

/// A Codex agent-message payload for the timeline: "(encrypted)" for a
/// Fernet blob, else the text capped at 200 chars.
fn agent_payload_text(s: &str) -> String {
    if looks_encrypted(s) {
        "(encrypted)".to_string()
    } else {
        truncate_chars(s, 200).to_string()
    }
}

/// Format a tool call event as a compact one-liner.
pub fn format_tool_call(event: &Event) -> String {
    // An absent *or empty* tool name falls back to "Unknown".
    let tool_name = event
        .tool_name
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or("Unknown");
    // Any falsy input (null, empty string, empty container) collapses to an
    // empty object, so every lookup below just misses.
    let empty = Value::Object(serde_json::Map::new());
    let tool_input: &Value = match &event.tool_input {
        Some(v) if is_truthy(v) => v,
        _ => &empty,
    };

    // Claude Code tools
    if tool_name == "Read" {
        return format!("Read {}", get_or(tool_input, "file_path", "?"));
    } else if tool_name == "Edit" {
        return format!("Edit {}", get_or(tool_input, "file_path", "?"));
    } else if tool_name == "Write" {
        let size = match get(tool_input, "content") {
            Some(v) => value_len(v),
            None => 0,
        };
        let size_str = if size > 1024 {
            format!("{:.1}K", size as f64 / 1024.0)
        } else {
            size.to_string()
        };
        return format!(
            "Write {} ({})",
            get_or(tool_input, "file_path", "?"),
            size_str
        );
    } else if tool_name == "Bash" {
        // Keeping only the first line's first 80 chars was measured on real
        // data to discard 73% of all Bash command text (flags, paths,
        // pipeline stages, script bodies) before it reached the summarizer,
        // Q&A, or FTS. "What was that command?" is a primary use case, so
        // keep the full command up to 500 chars, with newlines flattened so
        // the timeline stays one line per tool.
        let cmd = get_or(tool_input, "command", "");
        let cmd_short = flatten_command(&cmd);
        let desc = get(tool_input, "description");
        if let Some(d) = desc
            && is_truthy(d)
        {
            return format!("Bash: {} > {}", display_str(d), cmd_short);
        }
        return format!("Bash: {cmd_short}");
    } else if tool_name == "Grep" {
        let pattern = get_or(tool_input, "pattern", "?");
        let path = get_or(tool_input, "path", ".");
        return format!("Grep \"{pattern}\" in {path}");
    } else if tool_name == "Glob" {
        return format!("Glob {}", get_or(tool_input, "pattern", "?"));
    } else if tool_name == "Task" {
        let subagent = get_or(tool_input, "subagent_type", "agent");
        let desc = get_or(tool_input, "description", "");
        return format!("Task[{}]: {}", subagent, truncate_chars(&desc, 40));
    } else if tool_name.starts_with("mcp__pal__") {
        // The prompt is the semantic payload of the call, so it is capped at
        // 200 chars rather than the 40 used for generic string args.
        let model = get_or(tool_input, "model", "unknown");
        let prompt = get_or(tool_input, "prompt", "");
        return format!("Chat[{}]: {}...", model, truncate_chars(&prompt, 200));
    }
    // OpenAI Codex tools. The Codex parser lifts the
    // shell command out of the `exec` script into `command`, and lists the
    // files an `apply_patch` touches under `files`.
    else if tool_name == "exec" {
        if let Some(cmd) = get(tool_input, "command").filter(|v| is_truthy(v)) {
            return format!("Exec: {}", flatten_command(&display_str(cmd)));
        }
        // A script that runs no command (reads a file, views an image):
        // show the script itself, one-lined.
        let script = get_or(tool_input, "content", "");
        if script.is_empty() {
            return "Exec: ?".to_string();
        }
        return format!("Exec[script]: {}", flatten_command(&script));
    } else if matches!(
        tool_name,
        "shell" | "shell_command" | "local_shell" | "container.exec"
    ) {
        // Legacy `shell` sends `command` as an argv list, usually
        // `["bash", "-lc", "<script>"]`; `shell_command` sends a string.
        let cmd = match get(tool_input, "command") {
            Some(Value::Array(argv)) => {
                let parts: Vec<String> = argv.iter().map(display_str).collect();
                match parts.as_slice() {
                    [sh, flag, script] if flag == "-lc" || flag == "-c" => {
                        let _ = sh;
                        script.clone()
                    }
                    _ => parts.join(" "),
                }
            }
            Some(v) if is_truthy(v) => display_str(v),
            _ => String::new(),
        };
        if cmd.is_empty() {
            return "Shell: ?".to_string();
        }
        return format!("Shell: {}", flatten_command(&cmd));
    } else if tool_name == "apply_patch" && tool_input.is_object() {
        // A raw (non-object) patch string keeps the generic rendering that
        // `non_dict_tool_input` pins.
        if let Some(Value::Array(files)) = get(tool_input, "files")
            && !files.is_empty()
        {
            let names: Vec<String> = files.iter().map(display_str).collect();
            return format!("Patch {}", names.join(", "));
        }
        return "Patch (inline)".to_string();
    } else if tool_name == "spawn_agent" {
        // Real shape: {"task_name","fork_turns","message"}; the message is
        // a Fernet blob (`gAAAA…`) in Codex Desktop, so it renders as
        // "(encrypted)" rather than 200 chars of ciphertext.
        let name = ["task_name", "agent_path", "nickname"]
            .iter()
            .find_map(|k| get(tool_input, k).filter(|v| is_truthy(v)))
            .map(display_str)
            .unwrap_or_else(|| "agent".to_string());
        let prompt = ["message", "prompt", "task"]
            .iter()
            .find_map(|k| get(tool_input, k).filter(|v| is_truthy(v)))
            .map(|v| agent_payload_text(&display_str(v)))
            .unwrap_or_default();
        return format!("Agent[{name}]: {prompt}");
    } else if tool_name == "send_message" || tool_name == "followup_task" {
        // Real shape: {"target","message"}; same encrypted message.
        let to = ["target", "recipient"]
            .iter()
            .find_map(|k| get(tool_input, k).filter(|v| is_truthy(v)))
            .map(display_str)
            .unwrap_or_else(|| "?".to_string());
        let text = ["message", "content"]
            .iter()
            .find_map(|k| get(tool_input, k).filter(|v| is_truthy(v)))
            .map(|v| agent_payload_text(&display_str(v)))
            .unwrap_or_default();
        let label = if tool_name == "send_message" {
            "Message"
        } else {
            "Followup"
        };
        return format!("{label}[{to}]: {text}");
    } else if tool_name == "update_plan" {
        return "Plan".to_string();
    }
    // VS Code Copilot tools
    else if tool_name == "copilot_readFile" {
        return format!("Read {}", get_or(tool_input, "file_path", "?"));
    } else if tool_name == "copilot_createFile" {
        return format!("Create {}", get_or(tool_input, "file_path", "?"));
    } else if tool_name == "copilot_editFile" {
        return format!("Edit {}", get_or(tool_input, "file_path", "?"));
    } else if tool_name == "copilot_findTextInFiles" {
        // Nested dict.get defaults: presence of 'query' wins even if falsy.
        let query = match get(tool_input, "query").or_else(|| get(tool_input, "pattern")) {
            Some(v) => display_str(v),
            None => "?".to_string(),
        };
        let path = get_or(tool_input, "path", ".");
        return format!("Search \"{query}\" in {path}");
    } else if tool_name == "copilot_listDirectory" {
        let target = match get(tool_input, "file_path").or_else(|| get(tool_input, "path")) {
            Some(v) => display_str(v),
            None => "?".to_string(),
        };
        return format!("List {target}");
    } else if tool_name == "run_in_terminal" {
        // Try multiple possible keys for the command; a present-but-falsy
        // value falls through to the next key.
        let cmd = ["command", "commandLine", "input", "text"]
            .iter()
            .find_map(|k| get(tool_input, k).filter(|v| is_truthy(v)));
        return match cmd {
            Some(v) => {
                // Full command up to 500 chars, like the Bash branch.
                let s = display_str(v);
                format!("Terminal: {}", flatten_command(&s))
            }
            None => "Terminal: ?".to_string(),
        };
    } else if tool_name == "applyPatch" {
        let fp = get(tool_input, "file_path").or_else(|| get(tool_input, "path"));
        if let Some(v) = fp
            && is_truthy(v)
        {
            return format!("Patch {}", display_str(v));
        }
        return "Patch (inline)".to_string();
    } else if tool_name == "getErrors" {
        let fp = get(tool_input, "file_path").or_else(|| get(tool_input, "path"));
        if let Some(v) = fp
            && is_truthy(v)
        {
            return format!("GetErrors {}", display_str(v));
        }
        return "GetErrors".to_string();
    } else if tool_name == "manage_todo_list" {
        return "TodoList".to_string();
    } else if tool_name.starts_with("copilot_") {
        // Generic copilot tool
        let short_name = tool_name.replace("copilot_", "");
        let fp = get(tool_input, "file_path").or_else(|| get(tool_input, "path"));
        if let Some(v) = fp
            && is_truthy(v)
        {
            return format!("{} {}", short_name, display_str(v));
        }
        let query = get(tool_input, "query")
            .or_else(|| get(tool_input, "pattern"))
            .or_else(|| get(tool_input, "searchQuery"));
        if let Some(v) = query
            && is_truthy(v)
        {
            return format!("{} \"{}\"", short_name, display_str(v));
        }
        return format!("{}: {}", short_name, format_args(tool_input));
    }

    // Generic format
    format!("{}: {}", tool_name, format_args(tool_input))
}

/// Format tool arguments, showing file_path in full but truncating others.
fn format_args(tool_input: &Value) -> String {
    if !is_truthy(tool_input) {
        return "(no args)".to_string();
    }

    // Handle non-dict arguments (e.g. apply_patch sends a raw string)
    if let Value::String(s) = tool_input {
        let n = s.chars().count();
        if n > 80 {
            return format!("({n} chars)");
        }
        return s.clone();
    }
    let obj = match tool_input.as_object() {
        Some(o) => o,
        // Anything else (number, bool, array) renders as its display form,
        // capped at 80 chars.
        None => return truncate_chars(&display_str(tool_input), 80).to_string(),
    };

    let mut parts: Vec<String> = Vec::new();
    for (key, val) in obj {
        if key == "file_path" || key == "path" {
            parts.push(display_str(val));
        } else if key == "content" {
            let size = if is_truthy(val) { value_len(val) } else { 0 };
            parts.push(format!("({size} chars)"));
        } else if let Value::String(s) = val {
            // Generic string args are capped at 200 chars, not 40: on real
            // data a 40-char cap truncated an argument in 61% of generic/MCP
            // tool calls, and those arguments (questions, prompts, queries)
            // are the semantic payload of the call.
            if s.chars().count() > 200 {
                parts.push(format!("{}={}...", key, truncate_chars(s, 200)));
            } else if !s.is_empty() {
                parts.push(format!("{key}={s}"));
            }
        } else if is_truthy(val) {
            parts.push(format!("{}={}", key, display_str(val)));
        }
    }

    if parts.is_empty() {
        "(no args)".to_string()
    } else {
        parts.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool(name: &str, input: Value) -> Event {
        let mut e = Event::new("e", 0, EventType::ToolCall);
        e.tool_name = Some(name.to_string());
        e.tool_input = Some(input);
        e
    }

    fn msg(role: Option<MessageRole>, content: &str) -> Event {
        let mut e = Event::new("m", 0, EventType::Message);
        e.role = role;
        e.content = content.to_string();
        e
    }

    // The expected strings below are pinned rendered output: they are
    // embedded verbatim in LLM prompts and indexed for FTS, so update them
    // only when the rendering is changed deliberately.

    #[test]
    fn empty_tool_name_is_unknown() {
        // An empty tool name is treated the same as an absent one.
        assert_eq!(
            format_tool_call(&tool("", json!({"a": "b"}))),
            "Unknown: a=b"
        );
    }

    #[test]
    fn write_content_absent() {
        assert_eq!(format_tool_call(&tool("Write", json!({}))), "Write ? (0)");
    }

    #[test]
    fn string_repr_quotes_and_escapes() {
        // Top-level strings render raw; nested ones go through the quoted,
        // escaped repr form.
        assert_eq!(
            format_tool_call(&tool("X", json!({"k": "it's"}))),
            "X: k=it's"
        );
        assert_eq!(
            format_tool_call(&tool("X", json!({"o": {"k": "it's"}}))),
            "X: o={'k': \"it's\"}"
        );
        assert_eq!(
            format_tool_call(&tool("X", json!({"o": {"k": "a\"b"}}))),
            "X: o={'k': 'a\"b'}"
        );
        assert_eq!(
            format_tool_call(&tool("X", json!({"o": {"k": "a\nb\t\\c"}}))),
            "X: o={'k': 'a\\nb\\t\\\\c'}"
        );
        assert_eq!(
            format_tool_call(&tool("X", json!({"o": {"k": "\u{1}"}}))),
            "X: o={'k': '\\x01'}"
        );
    }

    #[test]
    fn float_repr_exponent_fixup() {
        assert_eq!(
            format_tool_call(&tool("X", json!({"num": 1e16}))),
            "X: num=1e+16"
        );
        assert_eq!(
            format_tool_call(&tool("X", json!({"num": 1e-5}))),
            "X: num=1e-05"
        );
        assert_eq!(
            format_tool_call(&tool("X", json!({"num": 1e300}))),
            "X: num=1e+300"
        );
        assert_eq!(
            format_tool_call(&tool("X", json!({"num": -2.5e-8}))),
            "X: num=-2.5e-08"
        );
    }

    #[test]
    fn non_object_input_for_named_tool() {
        // A malformed tool input must never fail the whole conversation:
        // a non-object input, or a null field, renders a placeholder
        // instead of erroring. Pinned deliberately.
        assert_eq!(
            format_tool_call(&tool("Read", json!("raw string"))),
            "Read ?"
        );
        let e = tool("Task", json!({"description": null}));
        assert_eq!(format_tool_call(&e), "Task[agent]: None");
    }

    #[test]
    fn read_edit_write() {
        assert_eq!(
            format_tool_call(&tool("Read", json!({"file_path": "/tmp/α.txt"}))),
            "Read /tmp/α.txt"
        );
        assert_eq!(format_tool_call(&tool("Read", json!({}))), "Read ?");
        assert_eq!(
            format_tool_call(&tool("Edit", json!({"file_path": "/a/b.rs"}))),
            "Edit /a/b.rs"
        );
    }

    #[test]
    fn write_size_formatting() {
        // Multibyte content: size counts CHARACTERS (600), not bytes (1200).
        let e = tool(
            "Write",
            json!({"file_path": "/x", "content": "é".repeat(600)}),
        );
        assert_eq!(format_tool_call(&e), "Write /x (600)");
        // Boundary: 1024 chars is NOT > 1024 → plain number.
        let e = tool(
            "Write",
            json!({"file_path": "/x", "content": "a".repeat(1024)}),
        );
        assert_eq!(format_tool_call(&e), "Write /x (1024)");
        let e = tool(
            "Write",
            json!({"file_path": "/x", "content": "b".repeat(2000)}),
        );
        assert_eq!(format_tool_call(&e), "Write /x (2.0K)");
        let e = tool("Write", json!({"content": "hi"}));
        assert_eq!(format_tool_call(&e), "Write ? (2)");
    }

    #[test]
    fn bash_variants() {
        // Pins the generous rendering: full command up to 500 chars, with
        // newlines joined by "; ".
        let e = tool(
            "Bash",
            json!({"command": format!("{}\nsecond line", "x".repeat(100)), "description": "Run thing"}),
        );
        assert_eq!(
            format_tool_call(&e),
            format!("Bash: Run thing > {}; second line", "x".repeat(100))
        );
        assert_eq!(
            format_tool_call(&tool("Bash", json!({"command": "ls -la"}))),
            "Bash: ls -la"
        );
        // Multibyte command truncated to 500 CHARS (would panic if byte-sliced).
        let e = tool("Bash", json!({"command": "é".repeat(600)}));
        assert_eq!(format_tool_call(&e), format!("Bash: {}", "é".repeat(500)));
        // Blank lines vanish; trailing whitespace per line is trimmed.
        let e = tool("Bash", json!({"command": "a \n\nb"}));
        assert_eq!(format_tool_call(&e), "Bash: a; b");
    }

    #[test]
    fn grep_glob_task() {
        assert_eq!(
            format_tool_call(&tool("Grep", json!({"pattern": "fn main", "path": "/src"}))),
            "Grep \"fn main\" in /src"
        );
        assert_eq!(
            format_tool_call(&tool("Grep", json!({}))),
            "Grep \"?\" in ."
        );
        assert_eq!(
            format_tool_call(&tool("Glob", json!({"pattern": "**/*.rs"}))),
            "Glob **/*.rs"
        );
        // Description is 90 chars of multibyte text → truncated at 40 chars.
        let e = tool(
            "Task",
            json!({"subagent_type": "Explore", "description": "ページ".repeat(30)}),
        );
        assert_eq!(
            format_tool_call(&e),
            "Task[Explore]: ページページページページページページページページページページページページページペ"
        );
        assert_eq!(format_tool_call(&tool("Task", json!({}))), "Task[agent]: ");
    }

    #[test]
    fn mcp_pal_chat() {
        let e = tool(
            "mcp__pal__chat",
            json!({"model": "gpt-5", "prompt": "How do I frobnicate the widget in a way that works"}),
        );
        // The prompt cap is 200 chars, so this short prompt survives whole.
        assert_eq!(
            format_tool_call(&e),
            "Chat[gpt-5]: How do I frobnicate the widget in a way that works..."
        );
        assert_eq!(
            format_tool_call(&tool("mcp__pal__thinkdeep", json!({}))),
            "Chat[unknown]: ..."
        );
    }

    #[test]
    fn copilot_named_tools() {
        assert_eq!(
            format_tool_call(&tool("copilot_readFile", json!({"file_path": "/f"}))),
            "Read /f"
        );
        assert_eq!(
            format_tool_call(&tool("copilot_createFile", json!({"file_path": "/f"}))),
            "Create /f"
        );
        assert_eq!(
            format_tool_call(&tool("copilot_editFile", json!({"file_path": "/f"}))),
            "Edit /f"
        );
        // 'query' key PRESENT but empty: dict.get returns it — pattern is NOT
        // consulted (unlike an `or` chain).
        let e = tool(
            "copilot_findTextInFiles",
            json!({"query": "", "pattern": "zzz"}),
        );
        assert_eq!(format_tool_call(&e), "Search \"\" in .");
        assert_eq!(
            format_tool_call(&tool("copilot_findTextInFiles", json!({"pattern": "todo"}))),
            "Search \"todo\" in ."
        );
        assert_eq!(
            format_tool_call(&tool("copilot_listDirectory", json!({"path": "/dir"}))),
            "List /dir"
        );
    }

    #[test]
    fn terminal_patch_errors_todo() {
        // Falsy 'command' falls through to 'commandLine' (or-chain semantics);
        // DIVERGENCE pin (approved 2026-08-11): all lines survive, joined
        // with "; " (was first line only).
        let e = tool(
            "run_in_terminal",
            json!({"command": "", "commandLine": "echo hi\nmore"}),
        );
        assert_eq!(format_tool_call(&e), "Terminal: echo hi; more");
        assert_eq!(
            format_tool_call(&tool("run_in_terminal", json!({}))),
            "Terminal: ?"
        );
        assert_eq!(
            format_tool_call(&tool("applyPatch", json!({"path": "/p.diff"}))),
            "Patch /p.diff"
        );
        assert_eq!(
            format_tool_call(&tool("applyPatch", json!({}))),
            "Patch (inline)"
        );
        assert_eq!(
            format_tool_call(&tool("getErrors", json!({"file_path": "/e.rs"}))),
            "GetErrors /e.rs"
        );
        assert_eq!(format_tool_call(&tool("getErrors", json!({}))), "GetErrors");
        assert_eq!(
            format_tool_call(&tool("manage_todo_list", json!({"operation": "write"}))),
            "TodoList"
        );
    }

    #[test]
    fn copilot_generic_fallbacks() {
        assert_eq!(
            format_tool_call(&tool(
                "copilot_getNotebookSummary",
                json!({"path": "/nb.ipynb"})
            )),
            "getNotebookSummary /nb.ipynb"
        );
        assert_eq!(
            format_tool_call(&tool(
                "copilot_searchWorkspaceSymbols",
                json!({"searchQuery": "MyClass"})
            )),
            "searchWorkspaceSymbols \"MyClass\""
        );
        assert_eq!(
            format_tool_call(&tool(
                "copilot_runTests",
                json!({"files": ["a", "b"], "verbose": true})
            )),
            "runTests: files=['a', 'b'] verbose=True"
        );
    }

    #[test]
    fn codex_exec_prefers_lifted_command_over_script() {
        let script = "text(await tools.exec_command({cmd:\"ls -la\\ngit status\"}));\n";
        assert_eq!(
            format_tool_call(&tool(
                "exec",
                json!({"content": script, "command": "ls -la\ngit status"})
            )),
            "Exec: ls -la; git status"
        );
        // No command lifted: the script itself, one-lined.
        assert_eq!(
            format_tool_call(&tool("exec", json!({"content": "image(x);\n"}))),
            "Exec[script]: image(x);"
        );
        assert_eq!(format_tool_call(&tool("exec", json!({}))), "Exec: ?");
    }

    #[test]
    fn codex_shell_variants() {
        // Legacy argv form unwraps the `bash -lc` wrapper.
        assert_eq!(
            format_tool_call(&tool(
                "shell",
                json!({"command": ["bash", "-lc", "cargo test"], "workdir": "/p"})
            )),
            "Shell: cargo test"
        );
        // Any other argv is joined.
        assert_eq!(
            format_tool_call(&tool("shell", json!({"command": ["ls", "-la"]}))),
            "Shell: ls -la"
        );
        // String form.
        assert_eq!(
            format_tool_call(&tool("shell_command", json!({"command": "pwd"}))),
            "Shell: pwd"
        );
        assert_eq!(
            format_tool_call(&tool("local_shell", json!({}))),
            "Shell: ?"
        );
    }

    #[test]
    fn codex_patch_and_agent_tools() {
        assert_eq!(
            format_tool_call(&tool(
                "apply_patch",
                json!({"content": "*** Begin Patch", "files": ["a.rs", "b.md"]})
            )),
            "Patch a.rs, b.md"
        );
        assert_eq!(
            format_tool_call(&tool("apply_patch", json!({"content": "*** Begin Patch"}))),
            "Patch (inline)"
        );
        // Real Codex Desktop shapes: task_name/target, encrypted message.
        let blob = format!("gAAAAABqnPk4V3fQ{}", "x".repeat(120));
        assert_eq!(
            format_tool_call(&tool(
                "spawn_agent",
                json!({"task_name": "tutorial_design_factcheck", "fork_turns": "none", "message": blob})
            )),
            "Agent[tutorial_design_factcheck]: (encrypted)"
        );
        assert_eq!(
            format_tool_call(&tool(
                "send_message",
                json!({"target": "/root", "message": blob})
            )),
            "Message[/root]: (encrypted)"
        );
        assert_eq!(
            format_tool_call(&tool(
                "followup_task",
                json!({"target": "final_tutorial_redteam", "message": blob})
            )),
            "Followup[final_tutorial_redteam]: (encrypted)"
        );
        // Plain-text payloads (CLI, older versions) still show.
        assert_eq!(
            format_tool_call(&tool(
                "spawn_agent",
                json!({"agent_path": "/root/research", "message": "Find the prior art"})
            )),
            "Agent[/root/research]: Find the prior art"
        );
        assert_eq!(
            format_tool_call(&tool(
                "send_message",
                json!({"recipient": "/root", "message": "done"})
            )),
            "Message[/root]: done"
        );
        assert_eq!(
            format_tool_call(&tool("update_plan", json!({"plan": [{"step": "x"}]}))),
            "Plan"
        );
    }

    #[test]
    fn generic_format_args() {
        // Pins the generic rendering: the string cap is 200 chars, so the
        // 80-char url survives whole; falsy values (false, 0, "", [], null)
        // are skipped; truthy bools and numbers render as `True` / `3`.
        let e = tool(
            "WebFetch",
            json!({
                "url": format!("https://example.com/{}", "x".repeat(60)),
                "prompt": "short", "flag": false, "n": 0, "empty": "",
                "lst": [], "none": null, "ok": true, "num": 3
            }),
        );
        assert_eq!(
            format_tool_call(&e),
            format!(
                "WebFetch: url=https://example.com/{} prompt=short ok=True num=3",
                "x".repeat(60)
            )
        );
        // Strings over 200 chars still truncate with "...".
        let e = tool("WebFetch", json!({"prompt": "y".repeat(250)}));
        assert_eq!(
            format_tool_call(&e),
            format!("WebFetch: prompt={}...", "y".repeat(200))
        );
        // content → char count, notebook_path is not special-cased.
        let e = tool(
            "NotebookEdit",
            json!({"notebook_path": "/n.ipynb", "content": "αβγ".repeat(20)}),
        );
        assert_eq!(
            format_tool_call(&e),
            "NotebookEdit: notebook_path=/n.ipynb (60 chars)"
        );
        assert_eq!(
            format_tool_call(&tool("SomeTool", json!({}))),
            "SomeTool: (no args)"
        );
        let mut e = Event::new("e", 0, EventType::ToolCall);
        e.tool_name = Some("SomeTool".to_string());
        e.tool_input = None;
        assert_eq!(format_tool_call(&e), "SomeTool: (no args)");
    }

    #[test]
    fn non_dict_tool_input() {
        assert_eq!(
            format_tool_call(&tool("apply_patch", json!("raw patch text"))),
            "apply_patch: raw patch text"
        );
        assert_eq!(
            format_tool_call(&tool("apply_patch", json!("p".repeat(200)))),
            "apply_patch: (200 chars)"
        );
        // A list renders with each element repr'd and a ", " separator.
        assert_eq!(
            format_tool_call(&tool("WeirdTool", json!([1, 2, "three"]))),
            "WeirdTool: [1, 2, 'three']"
        );
    }

    #[test]
    fn missing_tool_name() {
        let mut e = Event::new("e", 0, EventType::ToolCall);
        e.tool_name = None;
        e.tool_input = Some(json!({"a": "b"}));
        assert_eq!(format_tool_call(&e), "Unknown: a=b");
    }

    #[test]
    fn format_args_use_scalar_reprs() {
        assert_eq!(
            format_args(&json!({"ratio": 2.5, "whole": 3.0})),
            "ratio=2.5 whole=3.0"
        );
        assert_eq!(
            format_args(&json!({"obj": {"k": "v"}, "arr": ["x", 1]})),
            "obj={'k': 'v'} arr=['x', 1]"
        );
        // path shown even when falsy: str(None) == "None".
        assert_eq!(format_args(&json!({"path": null})), "None");
    }

    fn sample_transcript() -> ParsedTranscript {
        ParsedTranscript {
            metadata: SessionMetadata::new("s", "claude_code", "/t"),
            events: vec![
                msg(Some(MessageRole::User), "Hello\nthere"),
                msg(Some(MessageRole::Assistant), "Hi! Let me look."),
                tool("Read", json!({"file_path": "/a"})),
                tool("Bash", json!({"command": "ls"})),
                msg(Some(MessageRole::Assistant), "Done."),
                {
                    let mut e = Event::new("r", 9, EventType::ToolResult);
                    e.tool_output = Some("ignored".to_string());
                    e
                },
                tool("Glob", json!({"pattern": "*.py"})),
                msg(None, "mystery"),
                msg(Some(MessageRole::System), "sys prompt"),
                tool("Grep", json!({"pattern": "x"})),
            ],
        }
    }

    #[test]
    fn conversation_text_full() {
        // Tool grouping, flush-before-message, tool results ignored, unknown
        // and system roles dropped from conversation text, trailing group kept.
        assert_eq!(
            sample_transcript().get_conversation_text(true),
            "Human:\nHello\nthere\n\nAssistant:\nHi! Let me look.\n\n[Tool Calls]\nRead /a\nBash: ls\n\nAssistant:\nDone.\n\n[Tool Calls]\nGlob *.py\n\n[Tool Calls]\nGrep \"x\" in .\n"
        );
    }

    #[test]
    fn conversation_text_without_tools() {
        assert_eq!(
            sample_transcript().get_conversation_text(false),
            "Human:\nHello\nthere\n\nAssistant:\nHi! Let me look.\n\nAssistant:\nDone.\n"
        );
    }

    #[test]
    fn messages_include_unknown_and_system() {
        let msgs = sample_transcript().get_messages();
        let expected: Vec<(&str, &str)> = vec![
            ("user", "Hello\nthere"),
            ("assistant", "Hi! Let me look."),
            ("tools", "Read /a\nBash: ls"),
            ("assistant", "Done."),
            ("tools", "Glob *.py"),
            ("unknown", "mystery"),
            ("system", "sys prompt"),
            ("tools", "Grep \"x\" in ."),
        ];
        let got: Vec<(&str, &str)> = msgs
            .iter()
            .map(|m| (m.role.as_str(), m.content.as_str()))
            .collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn empty_and_tools_only_transcripts() {
        let empty = ParsedTranscript {
            metadata: SessionMetadata::new("s", "claude_code", "/t"),
            events: vec![],
        };
        assert_eq!(empty.get_conversation_text(true), "");
        let only_tools = ParsedTranscript {
            metadata: SessionMetadata::new("s", "claude_code", "/t"),
            events: vec![tool("Read", json!({"file_path": "/only"}))],
        };
        assert_eq!(
            only_tools.get_conversation_text(true),
            "[Tool Calls]\nRead /only\n"
        );
    }

    #[test]
    fn enum_string_values() {
        assert_eq!(EventType::ToolCall.as_str(), "tool_call");
        assert_eq!(EventType::Message.as_str(), "message");
        assert_eq!(MessageRole::User.as_str(), "user");
        assert_eq!(MessageRole::Assistant.as_str(), "assistant");
        assert_eq!(MessageRole::System.as_str(), "system");
    }
}
