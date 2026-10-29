//! Tests for `src/parsers/` — registry, Claude, Copilot, VS Code Copilot and
//! Codex parsers, plus BOM, malformed-JSON and golden-output cases.
//!
//! GOLDENS: every string marked "golden" below is pinned expected output for
//! the exact fixture bytes defined in this file. Do not edit a golden to make
//! a test pass — update it deliberately, only when the behavior change is the
//! intended one.

use std::fs;
use std::path::{Path, PathBuf};

use chrono::{DateTime, FixedOffset, TimeZone, Utc};
use ekko::models::{EventType, MessageRole};
use ekko::parsers::vscode_copilot::{decode_folder_uri, extract_tool_input};
use ekko::parsers::{
    ClaudeParser, CodexParser, CopilotParser, Parser, VSCodeCopilotParser, parser_for,
};
use serde_json::{Value, json};
use tempfile::TempDir;

fn utc(y: i32, mo: u32, d: u32, h: u32, mi: u32, s: u32) -> DateTime<FixedOffset> {
    Utc.with_ymd_and_hms(y, mo, d, h, mi, s)
        .unwrap()
        .fixed_offset()
}

// ═══════════════════════════════════════════════════════════════════════
// Registry
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn registry_known_sources() {
    let p = parser_for("claude_code").unwrap();
    assert_eq!(p.cli_source(), "claude_code");
    assert_eq!(p.parser_version(), "1.1");
    let p = parser_for("github_copilot").unwrap();
    assert_eq!(p.cli_source(), "github_copilot");
    assert_eq!(p.parser_version(), "1.0");
    let p = parser_for("vscode_copilot").unwrap();
    assert_eq!(p.cli_source(), "vscode_copilot");
    assert_eq!(p.parser_version(), "1.0");
    let p = parser_for("codex").unwrap();
    assert_eq!(p.cli_source(), "codex");
    assert_eq!(p.parser_version(), "1.0");
}

#[test]
fn registry_unknown_source_is_none_never_fallback() {
    assert!(parser_for("openai_codex").is_none());
    assert!(parser_for("Codex").is_none());
    assert!(parser_for("gemini_cli").is_none());
    assert!(parser_for("").is_none());
    assert!(parser_for("Claude_Code").is_none());
    assert!(parser_for("unknown").is_none());
}

// ═══════════════════════════════════════════════════════════════════════
// ClaudeParser
// ═══════════════════════════════════════════════════════════════════════

/// Byte-exact fixture the goldens below are pinned against.
const CLAUDE_RICH: &str = r#"{"type":"system","subtype":"init","cwd":"/home/user/project","gitBranch":"main","timestamp":"2025-06-15T10:00:00Z"}
{"type":"user","timestamp":"2025-06-15T10:00:05Z","message":{"role":"user","content":"Hello, can you help me?"}}
{"type":"assistant","timestamp":"2025-06-15T10:00:10Z","message":{"role":"assistant","content":[{"type":"text","text":"Sure! How can I help?"},{"type":"tool_use","id":"tc_1","name":"Read","input":{"file_path":"/home/user/project/main.py"}}]}}
not json
{"type":"user","timestamp":"2025-06-15T10:00:20Z","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tc_1","content":"print('hi')"},{"type":"text","text":"  please continue  "}]}}
{"type":"assistant","timestamp":"2025-06-15T10:00:30Z","message":{"role":"assistant","content":"I see the file."}}
{"type":"assistant","timestamp":"2025-06-15T10:00:35Z","message":{"role":"assistant","content":[{"type":"tool_use","id":"tc_2","name":"applyPatch","input":"raw patch content"}]}}
{"type":"progress","timestamp":"2025-06-15T10:00:40Z"}
"#;

fn write_claude(dir: &Path, name: &str, content: &str) -> PathBuf {
    let proj = dir.join("proj-dir");
    fs::create_dir_all(&proj).unwrap();
    let p = proj.join(name);
    fs::write(&p, content).unwrap();
    p
}

#[test]
fn claude_discover_finds_jsonl_at_correct_depth() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().join("project-abc");
    fs::create_dir_all(&proj).unwrap();
    let session = proj.join("abc123.jsonl");
    fs::write(&session, "{\"type\":\"system\"}\n").unwrap();

    let found = ClaudeParser.discover_sessions(tmp.path());
    assert_eq!(found, vec![session]);
}

#[test]
fn claude_discover_ignores_subagent_dirs() {
    let tmp = TempDir::new().unwrap();
    let base = tmp.path().join("projects");
    let proj = base.join("project-abc");
    fs::create_dir_all(proj.join("subagents/agent1")).unwrap();
    fs::write(proj.join("subagents/agent1/session.jsonl"), "{}\n").unwrap();
    fs::write(proj.join("main.jsonl"), "{}\n").unwrap();

    let found = ClaudeParser.discover_sessions(&base);
    assert_eq!(found.len(), 1);
    assert!(!found[0].to_string_lossy().contains("subagents"));
}

#[test]
fn claude_discover_skips_wrong_depth() {
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("top.jsonl"), "{}\n").unwrap(); // depth 1
    let deep = tmp.path().join("a/b");
    fs::create_dir_all(&deep).unwrap();
    fs::write(deep.join("deep.jsonl"), "{}\n").unwrap(); // depth 3

    assert!(ClaudeParser.discover_sessions(tmp.path()).is_empty());
}

#[test]
fn claude_discover_nonexistent_base() {
    let tmp = TempDir::new().unwrap();
    assert!(
        ClaudeParser
            .discover_sessions(&tmp.path().join("nonexistent"))
            .is_empty()
    );
}

#[test]
fn claude_metadata_golden() {
    // GOLDEN: field values from `ClaudeParser::extract_metadata` on
    // CLAUDE_RICH.
    let tmp = TempDir::new().unwrap();
    let path = write_claude(tmp.path(), "rich.jsonl", CLAUDE_RICH);
    let meta = ClaudeParser.extract_metadata(&path);

    assert_eq!(meta.session_id, "rich");
    assert_eq!(meta.cli_source, "claude_code");
    assert_eq!(meta.title, None);
    assert_eq!(meta.first_prompt.as_deref(), Some("Hello, can you help me?"));
    assert_eq!(meta.project_path.as_deref(), Some("/home/user/project"));
    assert_eq!(meta.git_branch.as_deref(), Some("main"));
    assert_eq!(meta.created_at, Some(utc(2025, 6, 15, 10, 0, 0)));
    assert_eq!(meta.modified_at, Some(utc(2025, 6, 15, 10, 0, 40)));
    assert_eq!(meta.message_count, 5);
    assert_eq!(meta.parser_version, "1.1");
    assert_eq!(meta.model, None);
    assert_eq!(
        meta.source_hash.as_deref(),
        Some("2fc8ff061be01346b311ae6cf415f4758e4ded1ea0926bd243028aab67f83aec")
    );
}

#[test]
fn claude_conversation_text_golden() {
    // GOLDEN: `parse_transcript(..).get_conversation_text(true/false)` on
    // CLAUDE_RICH.
    let tmp = TempDir::new().unwrap();
    let path = write_claude(tmp.path(), "rich.jsonl", CLAUDE_RICH);
    let t = ClaudeParser.parse_transcript(&path);

    assert_eq!(
        t.get_conversation_text(true),
        "Human:\nHello, can you help me?\n\nAssistant:\nSure! How can I help?\n\n[Tool Calls]\nRead /home/user/project/main.py\n\nHuman:\nplease continue\n\nAssistant:\nI see the file.\n\n[Tool Calls]\nPatch (inline)\n"
    );
    assert_eq!(
        t.get_conversation_text(false),
        "Human:\nHello, can you help me?\n\nAssistant:\nSure! How can I help?\n\nHuman:\nplease continue\n\nAssistant:\nI see the file.\n"
    );
}

#[test]
fn claude_events_golden() {
    // GOLDEN: event ids/sequences/types. Note the sequence quirk: a
    // text-less tool_use entry emits sequence start+i+1, so the last tool
    // call has sequence 6 while only six events exist.
    let tmp = TempDir::new().unwrap();
    let path = write_claude(tmp.path(), "rich.jsonl", CLAUDE_RICH);
    let t = ClaudeParser.parse_transcript(&path);

    assert_eq!(t.metadata.event_count, 6);
    let summary: Vec<(&str, i64, EventType)> = t
        .events
        .iter()
        .map(|e| (e.id.as_str(), e.sequence, e.event_type))
        .collect();
    assert_eq!(
        summary,
        vec![
            ("user_0", 0, EventType::Message),
            ("assistant_1", 1, EventType::Message),
            ("tc_1", 2, EventType::ToolCall),
            ("user_3", 3, EventType::Message),
            ("assistant_4", 4, EventType::Message),
            ("tc_2", 6, EventType::ToolCall),
        ]
    );
    assert_eq!(t.events[2].tool_name.as_deref(), Some("Read"));
    assert_eq!(t.events[2].tool_call_id.as_deref(), Some("tc_1"));
    assert_eq!(
        t.events[2].tool_input,
        Some(json!({"file_path": "/home/user/project/main.py"}))
    );
    assert_eq!(t.events[2].timestamp, Some(utc(2025, 6, 15, 10, 0, 10)));
    // String tool input is wrapped in {'content': ...}.
    assert_eq!(
        t.events[5].tool_input,
        Some(json!({"content": "raw patch content"}))
    );
    assert_eq!(t.events[0].role, Some(MessageRole::User));
    assert_eq!(t.events[0].content, "Hello, can you help me?");
    // User list content is stripped: "  please continue  " → "please continue".
    assert_eq!(t.events[3].content, "please continue");
}

#[test]
fn claude_user_message_list_content() {
    let tmp = TempDir::new().unwrap();
    let path = write_claude(
        tmp.path(),
        "s.jsonl",
        "{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Help me with this\"}]}}\n",
    );
    let meta = ClaudeParser.extract_metadata(&path);
    assert_eq!(meta.first_prompt.as_deref(), Some("Help me with this"));

    let t = ClaudeParser.parse_transcript(&path);
    let user_msgs: Vec<_> = t
        .events
        .iter()
        .filter(|e| e.role == Some(MessageRole::User))
        .collect();
    assert_eq!(user_msgs.len(), 1);
    assert_eq!(user_msgs[0].content, "Help me with this");
}

#[test]
fn claude_sessions_index_overrides() {
    let tmp = TempDir::new().unwrap();
    let path = write_claude(
        tmp.path(),
        "sess1.jsonl",
        "{\"type\":\"human\",\"message\":{\"role\":\"user\",\"content\":\"original prompt\"}}\n",
    );
    let index = json!([{
        "sessionId": "sess1", "firstPrompt": "overridden prompt",
        "messageCount": 99, "created": "2025-06-01T00:00:00Z",
        "lastModified": "2025-06-02T00:00:00Z"
    }]);
    fs::write(
        path.parent().unwrap().join("sessions-index.json"),
        serde_json::to_string(&index).unwrap(),
    )
    .unwrap();

    let meta = ClaudeParser.extract_metadata(&path);
    assert_eq!(meta.first_prompt.as_deref(), Some("overridden prompt"));
    assert_eq!(meta.message_count, 99);
    assert_eq!(meta.created_at, Some(utc(2025, 6, 1, 0, 0, 0)));
    assert_eq!(meta.modified_at, Some(utc(2025, 6, 2, 0, 0, 0)));
}

#[test]
fn claude_empty_file() {
    let tmp = TempDir::new().unwrap();
    let path = write_claude(tmp.path(), "empty.jsonl", "");
    let meta = ClaudeParser.extract_metadata(&path);
    assert_eq!(meta.session_id, "empty");
    assert_eq!(meta.message_count, 0);
    assert_eq!(meta.first_prompt.as_deref(), Some(""));
    assert_eq!(meta.project_path.as_deref(), Some(""));
    assert_eq!(meta.created_at, None);

    let t = ClaudeParser.parse_transcript(&path);
    assert!(t.events.is_empty());
    assert_eq!(t.metadata.event_count, 0);
    assert_eq!(t.get_conversation_text(true), "");
}

#[test]
fn claude_malformed_json_lines_skipped() {
    let tmp = TempDir::new().unwrap();
    let path = write_claude(
        tmp.path(),
        "bad.jsonl",
        "not json\n{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"hello\"}}\n{broken json\n",
    );
    let meta = ClaudeParser.extract_metadata(&path);
    assert_eq!(meta.first_prompt.as_deref(), Some("hello"));
    assert_eq!(meta.message_count, 1);
}

#[test]
fn claude_string_content_assistant() {
    let tmp = TempDir::new().unwrap();
    let path = write_claude(
        tmp.path(),
        "s.jsonl",
        "{\"type\":\"assistant\",\"message\":{\"role\":\"assistant\",\"content\":\"plain string\"}}\n",
    );
    let t = ClaudeParser.parse_transcript(&path);
    let asst: Vec<_> = t
        .events
        .iter()
        .filter(|e| e.role == Some(MessageRole::Assistant))
        .collect();
    assert_eq!(asst.len(), 1);
    assert_eq!(asst[0].content, "plain string");
}

#[test]
fn claude_bom_stripped() {
    // A leading UTF-8 BOM is stripped before parsing, so the first JSONL
    // line still parses rather than being skipped as malformed JSON.
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().join("proj-dir");
    fs::create_dir_all(&proj).unwrap();
    let path = proj.join("bom.jsonl");
    let mut bytes = vec![0xEF, 0xBB, 0xBF];
    bytes.extend_from_slice(
        b"{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"bom line\"}}\n{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"second\"}}\n",
    );
    fs::write(&path, &bytes).unwrap();

    let meta = ClaudeParser.extract_metadata(&path);
    assert_eq!(meta.first_prompt.as_deref(), Some("bom line"));
    assert_eq!(meta.message_count, 2);

    // The source hash still covers the raw bytes, BOM included — the file is
    // hashed as-is, not the decoded text.
    let hash_with_bom = meta.source_hash.clone().unwrap();
    let no_bom = proj.join("nobom.jsonl");
    fs::write(&no_bom, &bytes[3..]).unwrap();
    let other = ClaudeParser.extract_metadata(&no_bom);
    assert_ne!(Some(hash_with_bom), other.source_hash);
}

#[test]
fn claude_bad_timestamp_tolerated() {
    // An unparseable timestamp is treated as absent rather than failing the
    // whole session.
    let tmp = TempDir::new().unwrap();
    let path = write_claude(
        tmp.path(),
        "badts.jsonl",
        "{\"type\":\"user\",\"timestamp\":\"not a date\",\"message\":{\"role\":\"user\",\"content\":\"x\"}}\n",
    );
    let meta = ClaudeParser.extract_metadata(&path);
    assert_eq!(meta.created_at, None);
    assert_eq!(meta.modified_at, None);
    assert_eq!(meta.first_prompt.as_deref(), Some("x"));
}

#[test]
fn claude_no_messages_session() {
    let tmp = TempDir::new().unwrap();
    let path = write_claude(
        tmp.path(),
        "sysonly.jsonl",
        "{\"type\":\"system\",\"subtype\":\"init\",\"cwd\":\"/p\",\"timestamp\":\"2025-06-15T10:00:00Z\"}\n{\"type\":\"progress\"}\n",
    );
    let t = ClaudeParser.parse_transcript(&path);
    assert!(t.events.is_empty());
    assert_eq!(t.metadata.message_count, 0);
    assert_eq!(t.metadata.event_count, 0);
    assert_eq!(t.metadata.project_path.as_deref(), Some("/p"));
    assert_eq!(t.get_conversation_text(true), "");
}

#[test]
fn claude_validate_and_hash() {
    let tmp = TempDir::new().unwrap();
    let path = write_claude(tmp.path(), "v.jsonl", "abc");
    assert!(ClaudeParser.validate_transcript(&path));
    assert!(!ClaudeParser.validate_transcript(&tmp.path().join("missing.jsonl")));
    assert!(!ClaudeParser.validate_transcript(path.parent().unwrap()));
    // sha256("abc") — well-known digest.
    assert_eq!(
        ClaudeParser.get_source_hash(&path).as_deref(),
        Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    );
    assert_eq!(
        ClaudeParser.get_source_hash(&tmp.path().join("missing.jsonl")),
        None
    );
}

// ═══════════════════════════════════════════════════════════════════════
// CopilotParser
// ═══════════════════════════════════════════════════════════════════════

/// Byte-exact fixture the goldens below are pinned against.
const COPILOT_EVENTS: &str = r#"{"type":"session.start","timestamp":"2025-01-15T10:00:00Z","data":{"context":{"cwd":"/proj","branch":"feature-x"}}}
{"type":"user.message","timestamp":"2025-01-15T10:01:00Z","data":{"content":"Read the file"}}
{"type":"assistant.message","timestamp":"2025-01-15T10:01:30Z","data":{"content":"Here it is.","toolRequests":[{"name":"copilot_readFile","toolCallId":"tc1","arguments":{"file_path":"/src/main.py"}},{"name":"applyPatch","toolCallId":"tc2","arguments":"raw patch string"}]}}
{"type":"tool.execution_complete","timestamp":"2025-01-15T10:01:31Z","data":{"toolCallId":"tc1","success":true,"result":{"content":"print('hi')"}}}
{"type":"user.message","data":"not a dict"}
{"type":"user.message","timestamp":"2025-01-15T10:02:00Z","data":{"content":"thanks"}}
"#;

const COPILOT_WORKSPACE: &str = "cwd: /home/user/myapp\nsummary: Fixing auth\n";

fn write_copilot(dir: &Path, session_name: &str, events: &str, workspace: Option<&str>) -> PathBuf {
    let sdir = dir.join(session_name);
    fs::create_dir_all(&sdir).unwrap();
    if let Some(ws) = workspace {
        fs::write(sdir.join("workspace.yaml"), ws).unwrap();
    }
    let p = sdir.join("events.jsonl");
    fs::write(&p, events).unwrap();
    p
}

#[test]
fn copilot_discover_finds_events_jsonl() {
    let tmp = TempDir::new().unwrap();
    let events = write_copilot(tmp.path(), "uuid-session-1", "{}\n", None);
    fs::create_dir_all(tmp.path().join("empty-dir")).unwrap();
    fs::write(tmp.path().join("loose-file.txt"), "x").unwrap();

    let found = CopilotParser.discover_sessions(tmp.path());
    assert_eq!(found, vec![events]);
}

#[test]
fn copilot_discover_nonexistent_base() {
    let tmp = TempDir::new().unwrap();
    assert!(
        CopilotParser
            .discover_sessions(&tmp.path().join("nope"))
            .is_empty()
    );
}

#[test]
fn copilot_metadata_golden() {
    // GOLDEN: field values from `CopilotParser::extract_metadata` on
    // COPILOT_EVENTS + COPILOT_WORKSPACE.
    let tmp = TempDir::new().unwrap();
    let path = write_copilot(
        tmp.path(),
        "session-uuid-1",
        COPILOT_EVENTS,
        Some(COPILOT_WORKSPACE),
    );
    let meta = CopilotParser.extract_metadata(&path);

    assert_eq!(meta.session_id, "session-uuid-1");
    assert_eq!(meta.cli_source, "github_copilot");
    assert_eq!(meta.title.as_deref(), Some("Fixing auth"));
    assert_eq!(meta.first_prompt.as_deref(), Some("Read the file"));
    // workspace.yaml cwd wins; session.start only fills an empty path.
    assert_eq!(meta.project_path.as_deref(), Some("/home/user/myapp"));
    assert_eq!(meta.git_branch.as_deref(), Some("feature-x"));
    assert_eq!(meta.created_at, Some(utc(2025, 1, 15, 10, 0, 0)));
    assert_eq!(meta.modified_at, Some(utc(2025, 1, 15, 10, 2, 0)));
    // The user.message with non-object `data` still counts — messages are
    // counted by event type, not by extracted content.
    assert_eq!(meta.message_count, 4);
    assert_eq!(meta.parser_version, "1.0");
    assert_eq!(
        meta.source_hash.as_deref(),
        Some("54cddc5325ca47ab7f65dac113336a99d5be58a28cfc0adafeb6681f0c012fcb")
    );
}

#[test]
fn copilot_conversation_and_events_golden() {
    // GOLDEN: `parse_transcript(..).get_conversation_text(true)` and event
    // fields on COPILOT_EVENTS.
    let tmp = TempDir::new().unwrap();
    let path = write_copilot(
        tmp.path(),
        "session-uuid-1",
        COPILOT_EVENTS,
        Some(COPILOT_WORKSPACE),
    );
    let t = CopilotParser.parse_transcript(&path);

    assert_eq!(
        t.get_conversation_text(true),
        "Human:\nRead the file\n\nAssistant:\nHere it is.\n\n[Tool Calls]\nRead /src/main.py\nPatch (inline)\n\nHuman:\nthanks\n"
    );

    assert_eq!(t.metadata.event_count, 6);
    let summary: Vec<(&str, i64, EventType)> = t
        .events
        .iter()
        .map(|e| (e.id.as_str(), e.sequence, e.event_type))
        .collect();
    assert_eq!(
        summary,
        vec![
            ("event_0", 0, EventType::Message),
            ("event_1", 1, EventType::Message),
            ("tc1", 2, EventType::ToolCall),
            ("tc2", 3, EventType::ToolCall),
            ("result_tc1", 4, EventType::ToolResult),
            ("event_5", 5, EventType::Message),
        ]
    );
    // Tool call / result pairing via tool_call_id.
    assert_eq!(t.events[2].tool_name.as_deref(), Some("copilot_readFile"));
    assert_eq!(t.events[2].tool_call_id.as_deref(), Some("tc1"));
    assert_eq!(
        t.events[2].tool_input,
        Some(json!({"file_path": "/src/main.py"}))
    );
    // String arguments (apply_patch style) wrapped in {'content': ...}.
    assert_eq!(
        t.events[3].tool_input,
        Some(json!({"content": "raw patch string"}))
    );
    assert_eq!(t.events[4].tool_call_id.as_deref(), Some("tc1"));
    assert_eq!(t.events[4].tool_success, Some(true));
    assert_eq!(t.events[4].tool_output.as_deref(), Some("print('hi')"));
    assert_eq!(t.events[4].timestamp, Some(utc(2025, 1, 15, 10, 1, 31)));
}

#[test]
fn copilot_session_start_context() {
    let tmp = TempDir::new().unwrap();
    let line = json!({
        "type": "session.start", "timestamp": "2025-01-01T00:00:00Z",
        "data": {"context": {"cwd": "/proj", "branch": "feature-x"}}
    });
    let path = write_copilot(
        tmp.path(),
        "sess",
        &format!("{}\n", serde_json::to_string(&line).unwrap()),
        None,
    );
    let meta = CopilotParser.extract_metadata(&path);
    assert_eq!(meta.project_path.as_deref(), Some("/proj"));
    assert_eq!(meta.git_branch.as_deref(), Some("feature-x"));
}

#[test]
fn copilot_non_dict_data_handled() {
    let tmp = TempDir::new().unwrap();
    let path = write_copilot(
        tmp.path(),
        "sess",
        "{\"type\":\"user.message\",\"data\":\"not a dict\"}\n{\"type\":\"user.message\",\"data\":{\"content\":\"valid\"}}\n",
        None,
    );
    let meta = CopilotParser.extract_metadata(&path);
    assert_eq!(meta.first_prompt.as_deref(), Some("valid"));

    // Non-dict data produces no events.
    let single = write_copilot(
        tmp.path(),
        "sess2",
        "{\"type\":\"user.message\",\"data\":\"just a string\"}\n",
        None,
    );
    let t = CopilotParser.parse_transcript(&single);
    assert!(t.events.is_empty());
}

#[test]
fn copilot_missing_fields() {
    let tmp = TempDir::new().unwrap();
    let path = write_copilot(tmp.path(), "sess", "{\"type\":\"unknown\"}\n", None);
    let meta = CopilotParser.extract_metadata(&path);
    assert_eq!(meta.first_prompt.as_deref(), Some(""));
    assert_eq!(meta.message_count, 0);
    assert_eq!(meta.title.as_deref(), Some(""));
    assert_eq!(meta.project_path.as_deref(), Some(""));
}

#[test]
fn copilot_bom_workspace_and_events() {
    // A BOM is stripped from both the YAML and the JSONL, so the first JSONL
    // line parses and the YAML key reads as `cwd`, not `\u{feff}cwd`.
    let tmp = TempDir::new().unwrap();
    let sdir = tmp.path().join("sess");
    fs::create_dir_all(&sdir).unwrap();
    let mut yaml = vec![0xEF, 0xBB, 0xBF];
    yaml.extend_from_slice(b"cwd: /w\n");
    fs::write(sdir.join("workspace.yaml"), yaml).unwrap();
    let mut events = vec![0xEF, 0xBB, 0xBF];
    events.extend_from_slice(
        b"{\"type\":\"user.message\",\"timestamp\":\"2025-01-01T00:00:00Z\",\"data\":{\"content\":\"hello\"}}\n",
    );
    let path = sdir.join("events.jsonl");
    fs::write(&path, events).unwrap();

    let meta = CopilotParser.extract_metadata(&path);
    assert_eq!(meta.project_path.as_deref(), Some("/w"));
    assert_eq!(meta.first_prompt.as_deref(), Some("hello"));
}

// ═══════════════════════════════════════════════════════════════════════
// VSCodeCopilotParser
// ═══════════════════════════════════════════════════════════════════════

/// Byte-exact fixture the goldens below are pinned against.
const VSCODE_SESSION: &str = r#"{"creationDate":1700000000000,"lastMessageDate":1700003600000,"selectedModel":{"identifier":"gpt-4o"},"requests":[{"requestId":"r1","timestamp":1700000000000,"message":{"text":"What does this do?"},"response":[{"kind":"toolInvocationSerialized","toolId":"copilot_readFile","toolCallId":"tcA","inputArgs":{"file_path":"/src/app.py"}},{"kind":"textEditGroup","uri":{"path":"/src/main.py"},"edits":[[{"text":"new code","range":{"startLineNumber":1,"endLineNumber":5}}]]},{"value":"It prints hello.","kind":"markdownContent"},"direct string tail"]},{"message":{"text":"thanks"},"response":[]}]}"#;

const VSCODE_WORKSPACE: &str =
    r#"{"folder":"vscode-remote://wsl%2Bubuntu/home/cbird/repos/project"}"#;

fn write_vscode(dir: &Path, session_name: &str, session: &str, workspace: Option<&str>) -> PathBuf {
    let ws = dir.join("ws-hash");
    let chat = ws.join("chatSessions");
    fs::create_dir_all(&chat).unwrap();
    if let Some(w) = workspace {
        fs::write(ws.join("workspace.json"), w).unwrap();
    }
    let p = chat.join(session_name);
    fs::write(&p, session).unwrap();
    p
}

/// Expected local wall-clock time for an epoch-ms value. The parser converts
/// epoch milliseconds to machine-local time, so this helper must too — the
/// absolute instant is fixed, the printed wall clock is not.
fn local_ts(ms: i64) -> DateTime<FixedOffset> {
    DateTime::from_timestamp_millis(ms)
        .unwrap()
        .with_timezone(&chrono::Local)
        .fixed_offset()
}

#[test]
fn vscode_discover_finds_chat_sessions() {
    let tmp = TempDir::new().unwrap();
    let session = write_vscode(tmp.path(), "session1.json", "{}", None);
    fs::write(tmp.path().join("not-a-dir.txt"), "x").unwrap();
    fs::create_dir_all(tmp.path().join("no-chats-dir")).unwrap();

    let found = VSCodeCopilotParser.discover_sessions(tmp.path());
    assert_eq!(found, vec![session]);
}

#[test]
fn vscode_discover_nonexistent_base() {
    let tmp = TempDir::new().unwrap();
    assert!(
        VSCodeCopilotParser
            .discover_sessions(&tmp.path().join("nope"))
            .is_empty()
    );
}

#[test]
fn vscode_metadata_golden() {
    // GOLDEN: field values from `VSCodeCopilotParser::extract_metadata` on
    // VSCODE_SESSION + VSCODE_WORKSPACE. Timestamps are epoch-ms converted
    // to local wall-clock time.
    let tmp = TempDir::new().unwrap();
    let path = write_vscode(
        tmp.path(),
        "abc-123.json",
        VSCODE_SESSION,
        Some(VSCODE_WORKSPACE),
    );
    let meta = VSCodeCopilotParser.extract_metadata(&path);

    assert_eq!(meta.session_id, "abc-123");
    assert_eq!(meta.cli_source, "vscode_copilot");
    assert_eq!(meta.title.as_deref(), Some("What does this do?"));
    assert_eq!(meta.first_prompt.as_deref(), Some("What does this do?"));
    assert_eq!(meta.project_path.as_deref(), Some("/home/cbird/repos/project"));
    assert_eq!(meta.git_branch, None);
    assert_eq!(meta.created_at, Some(local_ts(1_700_000_000_000)));
    assert_eq!(meta.modified_at, Some(local_ts(1_700_003_600_000)));
    // The wall-clock date is 2023-11-14 in any plausible timezone, so the
    // year is a tz-stable assertion where the full timestamp would not be.
    use chrono::Datelike;
    assert_eq!(meta.created_at.unwrap().year(), 2023);
    assert_eq!(meta.message_count, 3);
    assert_eq!(meta.model.as_deref(), Some("gpt-4o"));
    assert_eq!(meta.parser_version, "1.0");
    assert_eq!(
        meta.source_hash.as_deref(),
        Some("1d30712af87534f7970d3fb1c5a97c8db8be0a34fac1605274fd43f2a6f244e0")
    );
}

#[test]
fn vscode_conversation_and_events_golden() {
    // GOLDEN: `parse_transcript(..).get_conversation_text(true)` and event
    // ids/sequences on VSCODE_SESSION.
    let tmp = TempDir::new().unwrap();
    let path = write_vscode(
        tmp.path(),
        "abc-123.json",
        VSCODE_SESSION,
        Some(VSCODE_WORKSPACE),
    );
    let t = VSCodeCopilotParser.parse_transcript(&path);

    assert_eq!(
        t.get_conversation_text(true),
        "Human:\nWhat does this do?\n\n[Tool Calls]\nRead /src/app.py\nEdit /src/main.py\n\nAssistant:\nIt prints hello.\ndirect string tail\n\nHuman:\nthanks\n"
    );

    assert_eq!(t.metadata.event_count, 5);
    let summary: Vec<(&str, i64, EventType)> = t
        .events
        .iter()
        .map(|e| (e.id.as_str(), e.sequence, e.event_type))
        .collect();
    assert_eq!(
        summary,
        vec![
            ("user_r1", 0, EventType::Message),
            ("tcA", 1, EventType::ToolCall),
            ("edit_0_1", 2, EventType::ToolCall),
            ("assistant_r1", 3, EventType::Message),
            // Second request has no requestId → default "req_4".
            ("user_req_4", 4, EventType::Message),
        ]
    );
    assert_eq!(t.events[1].tool_name.as_deref(), Some("copilot_readFile"));
    assert_eq!(t.events[1].tool_call_id.as_deref(), Some("tcA"));
    assert_eq!(
        t.events[1].tool_input,
        Some(json!({"file_path": "/src/app.py"}))
    );
    assert_eq!(t.events[2].tool_name.as_deref(), Some("Edit"));
    assert_eq!(
        t.events[2].tool_input,
        Some(json!({"file_path": "/src/main.py", "edits": "Lines 1-5"}))
    );
    // Second request had no timestamp.
    assert_eq!(t.events[4].timestamp, None);
    assert_eq!(t.events[0].timestamp, Some(local_ts(1_700_000_000_000)));
}

#[test]
fn vscode_non_dict_message_handled() {
    // A `message` that is not an object is treated as no message: empty
    // first_prompt, no title, no events — rather than an error.
    let tmp = TempDir::new().unwrap();
    let path = write_vscode(
        tmp.path(),
        "s.json",
        r#"{"requests":[{"message":"not a dict","response":[]}]}"#,
        None,
    );
    let meta = VSCodeCopilotParser.extract_metadata(&path);
    assert_eq!(meta.first_prompt.as_deref(), Some(""));
    assert_eq!(meta.title, None);
}

#[test]
fn vscode_non_list_requests_handled() {
    let tmp = TempDir::new().unwrap();
    let path = write_vscode(tmp.path(), "s.json", r#"{"requests":"not a list"}"#, None);
    // A non-list `requests` is treated as an empty list → message_count 0.
    let meta = VSCodeCopilotParser.extract_metadata(&path);
    assert_eq!(meta.message_count, 0);
    // `parse_transcript` tolerates it the same way: no events, rather than
    // failing the whole session on a wrong-typed field.
    let t = VSCodeCopilotParser.parse_transcript(&path);
    assert!(t.events.is_empty());
}

#[test]
fn vscode_non_dict_uri_in_text_edit() {
    let tmp = TempDir::new().unwrap();
    let path = write_vscode(
        tmp.path(),
        "s.json",
        r#"{"requests":[{"message":{"text":"edit"},"response":[{"kind":"textEditGroup","uri":"not-a-dict","edits":[]}]}]}"#,
        None,
    );
    let t = VSCodeCopilotParser.parse_transcript(&path);
    let tools: Vec<_> = t
        .events
        .iter()
        .filter(|e| e.event_type == EventType::ToolCall)
        .collect();
    assert_eq!(tools.len(), 1);
    assert_eq!(
        tools[0].tool_input,
        Some(json!({"file_path": "", "edits": "code changes"}))
    );
}

#[test]
fn vscode_non_list_response_handled() {
    let tmp = TempDir::new().unwrap();
    let path = write_vscode(
        tmp.path(),
        "s.json",
        r#"{"requests":[{"message":{"text":"hi"},"response":"not a list"}]}"#,
        None,
    );
    let t = VSCodeCopilotParser.parse_transcript(&path);
    let user_msgs: Vec<_> = t
        .events
        .iter()
        .filter(|e| e.role == Some(MessageRole::User))
        .collect();
    assert_eq!(user_msgs.len(), 1);
}

#[test]
fn vscode_malformed_session_json() {
    let tmp = TempDir::new().unwrap();
    let path = write_vscode(tmp.path(), "s.json", "{broken", None);
    let meta = VSCodeCopilotParser.extract_metadata(&path);
    assert_eq!(meta.first_prompt.as_deref(), Some(""));
    assert_eq!(meta.message_count, 0);
    let t = VSCodeCopilotParser.parse_transcript(&path);
    assert!(t.events.is_empty());
}

// ───────────────────────────────────────────────────────────────────────
// decode_folder_uri
// ───────────────────────────────────────────────────────────────────────

#[test]
fn decode_uri_vscode_remote_wsl() {
    assert_eq!(
        decode_folder_uri("vscode-remote://wsl%2Bubuntu/home/cbird/repos/project"),
        "/home/cbird/repos/project"
    );
}

#[test]
fn decode_uri_file_windows() {
    assert_eq!(
        decode_folder_uri("file:///c:/Users/cbird/project"),
        "c:/Users/cbird/project"
    );
}

#[test]
fn decode_uri_file_unix() {
    assert_eq!(
        decode_folder_uri("file:///home/user/project"),
        "/home/user/project"
    );
}

#[test]
fn decode_uri_plain_path_and_empty() {
    assert_eq!(decode_folder_uri("/home/user/project"), "/home/user/project");
    assert_eq!(decode_folder_uri(""), "");
}

#[test]
fn decode_uri_percent_escapes() {
    // Space and non-ASCII percent-escapes decode as UTF-8.
    assert_eq!(
        decode_folder_uri("file:///home/u/my%20pro%C3%A9ject"),
        "/home/u/my proéject"
    );
    // Invalid escapes stay literal rather than being rejected.
    assert_eq!(decode_folder_uri("file:///home/u/100%zz"), "/home/u/100%zz");
}

// ───────────────────────────────────────────────────────────────────────
// extract_tool_input
// ───────────────────────────────────────────────────────────────────────

fn obj(v: Value) -> serde_json::Map<String, Value> {
    v.as_object().unwrap().clone()
}

#[test]
fn tool_input_invocation_message() {
    let item = obj(json!({"invocationMessage": {
        "query": "search term",
        "uris": {"file1": {"path": "/src/main.py"}}
    }}));
    let result = extract_tool_input(&item);
    assert_eq!(result.get("file_path"), Some(&json!("/src/main.py")));
    assert_eq!(result.get("query"), Some(&json!("search term")));
}

#[test]
fn tool_input_input_args() {
    let item = obj(json!({"inputArgs": {"file_path": "/a.py", "mode": "r"}}));
    let result = extract_tool_input(&item);
    assert_eq!(result.get("file_path"), Some(&json!("/a.py")));
    assert_eq!(result.get("mode"), Some(&json!("r")));
}

#[test]
fn tool_input_top_level_keys() {
    let item = obj(json!({"command": "ls -la", "path": "/src"}));
    let result = extract_tool_input(&item);
    assert_eq!(result.get("command"), Some(&json!("ls -la")));
    assert_eq!(result.get("path"), Some(&json!("/src")));
}

#[test]
fn tool_input_top_level_order_deterministic() {
    // The extracted keys come out in ARG_KEYS declaration order, not in the
    // order they appear in the item — so `tool_input` is byte-stable.
    let item = obj(json!({"command": "ls -la", "path": "/src", "query": "q1", "text": "t1"}));
    let result = extract_tool_input(&item);
    let keys: Vec<&String> = result.keys().collect();
    assert_eq!(keys, ["query", "command", "path", "text"]);
}

#[test]
fn tool_input_past_action() {
    let item = obj(json!({"pastAction": {"command": "npm test"}}));
    let result = extract_tool_input(&item);
    assert_eq!(result.get("command"), Some(&json!("npm test")));
    // commandLine only fills when command absent.
    let item = obj(json!({"pastAction": {"commandLine": "yarn build"}}));
    let result = extract_tool_input(&item);
    assert_eq!(result.get("command"), Some(&json!("yarn build")));
}

#[test]
fn tool_input_result_fallback() {
    let item = obj(json!({"result": {"query": "fallback query"}}));
    let result = extract_tool_input(&item);
    assert_eq!(result.get("query"), Some(&json!("fallback query")));
    // Result is only consulted when nothing else matched.
    let item = obj(json!({"command": "x", "result": {"query": "ignored"}}));
    let result = extract_tool_input(&item);
    assert_eq!(result.get("query"), None);
}

#[test]
fn tool_input_empty_item() {
    assert!(extract_tool_input(&obj(json!({}))).is_empty());
}

#[test]
fn tool_input_long_strings_truncated() {
    // inputArgs adds "..." after 500 chars; invocationMessage does not.
    let long = "x".repeat(600);
    let item = obj(json!({"inputArgs": {"content": long}}));
    let result = extract_tool_input(&item);
    let got = result.get("content").unwrap().as_str().unwrap();
    assert_eq!(got.len(), 503);
    assert!(got.ends_with("..."));

    let long = "y".repeat(600);
    let item = obj(json!({"invocationMessage": {"query": long}}));
    let result = extract_tool_input(&item);
    let got = result.get("query").unwrap().as_str().unwrap();
    assert_eq!(got.len(), 500);
    assert!(!got.ends_with("..."));
}

// ───────────────────────────────────────────────────────────────────────
// Edge branches — odd but real transcript shapes, and what each yields
// ───────────────────────────────────────────────────────────────────────

#[test]
fn claude_empty_string_assistant_content_emits_event() {
    // String content "" still emits an assistant message, which shows up in
    // conversation text as "Assistant:\n\n".
    let tmp = TempDir::new().unwrap();
    let path = write_claude(
        tmp.path(),
        "s.jsonl",
        "{\"type\":\"assistant\",\"message\":{\"role\":\"assistant\",\"content\":\"\"}}\n",
    );
    let t = ClaudeParser.parse_transcript(&path);
    assert_eq!(t.events.len(), 1);
    assert_eq!(t.events[0].content, "");
    assert_eq!(t.get_conversation_text(true), "Assistant:\n\n");
}

#[test]
fn copilot_tool_result_variants() {
    // Non-object truthy result → stringified; falsy → ""; object with
    // content:null → tool_output None; object without content → "".
    let tmp = TempDir::new().unwrap();
    let cases: Vec<(&str, Option<&str>)> = vec![
        (r#"{"type":"tool.execution_complete","data":{"toolCallId":"t1","result":"text result"}}"#, Some("text result")),
        (r#"{"type":"tool.execution_complete","data":{"toolCallId":"t1","result":0}}"#, Some("")),
        (r#"{"type":"tool.execution_complete","data":{"toolCallId":"t1","result":null}}"#, Some("")),
        (r#"{"type":"tool.execution_complete","data":{"toolCallId":"t1","result":["a"]}}"#, Some("['a']")),
        (r#"{"type":"tool.execution_complete","data":{"toolCallId":"t1","result":{"content":null}}}"#, None),
        (r#"{"type":"tool.execution_complete","data":{"toolCallId":"t1","result":{"x":1}}}"#, Some("")),
    ];
    for (i, (line, expected)) in cases.iter().enumerate() {
        let path = write_copilot(tmp.path(), &format!("sess{i}"), &format!("{line}\n"), None);
        let t = CopilotParser.parse_transcript(&path);
        assert_eq!(t.events.len(), 1, "case {i}");
        assert_eq!(t.events[0].event_type, EventType::ToolResult, "case {i}");
        assert_eq!(t.events[0].tool_output.as_deref(), *expected, "case {i}");
        // An absent `success` key means false.
        assert_eq!(t.events[0].tool_success, Some(false), "case {i}");
        assert_eq!(t.events[0].id, "result_t1", "case {i}");
    }
}

#[test]
fn vscode_text_edit_missing_range_defaults_to_zero() {
    // Edits without a range render "Lines 0-0" rather than being skipped.
    let tmp = TempDir::new().unwrap();
    let path = write_vscode(
        tmp.path(),
        "s.json",
        r#"{"requests":[{"message":{"text":"e"},"response":[{"kind":"textEditGroup","uri":{"path":"/f"},"edits":[[{"text":"zz"}]]}]}]}"#,
        None,
    );
    let t = VSCodeCopilotParser.parse_transcript(&path);
    let tool = t
        .events
        .iter()
        .find(|e| e.event_type == EventType::ToolCall)
        .unwrap();
    assert_eq!(
        tool.tool_input,
        Some(json!({"file_path": "/f", "edits": "Lines 0-0"}))
    );
}

#[test]
fn vscode_tool_invocation_defaults() {
    // The three `toolId` cases are distinct: absent → tool_name "";
    // present-but-null → tool_name None. An absent `toolCallId` falls back to
    // the generated id "tool_{request}_{i}".
    let tmp = TempDir::new().unwrap();
    let path = write_vscode(
        tmp.path(),
        "s.json",
        r#"{"requests":[{"response":[{"kind":"toolInvocationSerialized"}]},{"response":[{"kind":"toolInvocationSerialized","toolId":null}]}]}"#,
        None,
    );
    let t = VSCodeCopilotParser.parse_transcript(&path);
    assert_eq!(t.events.len(), 2);
    assert_eq!(t.events[0].tool_name.as_deref(), Some(""));
    assert_eq!(t.events[0].id, "tool_0_0");
    assert_eq!(t.events[0].tool_call_id.as_deref(), Some("tool_0_0"));
    assert_eq!(t.events[1].tool_name, None);
    assert_eq!(t.events[1].id, "tool_1_0");
}

// ═══════════════════════════════════════════════════════════════════════
// Absent-vs-null-vs-wrong-type fields — the cases that are easy to collapse
// by accident. Each expected shape below is pinned deliberately.
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn copilot_tool_execution_complete_without_data_key_still_emits() {
    // CRITICAL: an ABSENT `data` key acts like an empty object and falls
    // through, so the bare tool.execution_complete line still emits a
    // TOOL_RESULT *and* still advances the sequence counter.
    //
    // Expected:
    //   event_count = 2
    //   [0] id='result_None' seq=0 tool_result tool_output='' success=False
    //       tool_call_id=None
    //   [1] id='event_1'     seq=1 message user content='after'
    let tmp = TempDir::new().unwrap();
    let path = write_copilot(
        tmp.path(),
        "sess",
        "{\"type\":\"tool.execution_complete\",\"timestamp\":\"2025-01-01T00:00:00Z\"}\n\
         {\"type\":\"user.message\",\"data\":{\"content\":\"after\"}}\n",
        None,
    );
    let t = CopilotParser.parse_transcript(&path);
    assert_eq!(t.metadata.event_count, 2);
    assert_eq!(t.events.len(), 2);

    assert_eq!(t.events[0].id, "result_None");
    assert_eq!(t.events[0].sequence, 0);
    assert_eq!(t.events[0].event_type, EventType::ToolResult);
    assert_eq!(t.events[0].tool_output.as_deref(), Some(""));
    assert_eq!(t.events[0].tool_success, Some(false));
    assert_eq!(t.events[0].tool_call_id, None);
    assert_eq!(t.events[0].timestamp, Some(utc(2025, 1, 1, 0, 0, 0)));

    assert_eq!(t.events[1].id, "event_1");
    assert_eq!(t.events[1].sequence, 1);
    assert_eq!(t.events[1].role, Some(MessageRole::User));
    assert_eq!(t.events[1].content, "after");
}

#[test]
fn copilot_present_non_dict_data_returns_early() {
    // The other half of the absent-vs-wrong-type pair: a PRESENT but
    // non-object `data` is skipped entirely and emits nothing (and
    // does NOT advance the counter).
    //
    // Expected: event_count = 1,
    //   [0] id='event_0' seq=0 message user content='after'
    let tmp = TempDir::new().unwrap();
    let path = write_copilot(
        tmp.path(),
        "sess",
        "{\"type\":\"tool.execution_complete\",\"data\":\"oops\"}\n\
         {\"type\":\"user.message\",\"data\":{\"content\":\"after\"}}\n",
        None,
    );
    let t = CopilotParser.parse_transcript(&path);
    assert_eq!(t.metadata.event_count, 1);
    assert_eq!(t.events[0].id, "event_0");
    assert_eq!(t.events[0].sequence, 0);
    assert_eq!(t.events[0].content, "after");
}

#[test]
fn copilot_session_start_empty_data_resets_branch() {
    // An empty `data` still enters the block, and an empty `context` yields
    // no branch, so a later session.start with no context RESETS the branch.
    //
    // Expected: project='/a', branch=None.
    let tmp = TempDir::new().unwrap();
    let path = write_copilot(
        tmp.path(),
        "sess",
        "{\"type\":\"session.start\",\"data\":{\"context\":{\"cwd\":\"/a\",\"branch\":\"main\"}}}\n\
         {\"type\":\"session.start\",\"data\":{}}\n",
        None,
    );
    let meta = CopilotParser.extract_metadata(&path);
    assert_eq!(meta.project_path.as_deref(), Some("/a"));
    assert_eq!(meta.git_branch, None);
}

#[test]
fn copilot_session_start_missing_data_key_resets_branch() {
    // Same reset, reached via the absent-key default rather than an explicit
    // empty object. Expected: project='/a', branch=None.
    let tmp = TempDir::new().unwrap();
    let path = write_copilot(
        tmp.path(),
        "sess",
        "{\"type\":\"session.start\",\"data\":{\"context\":{\"cwd\":\"/a\",\"branch\":\"main\"}}}\n\
         {\"type\":\"session.start\"}\n",
        None,
    );
    let meta = CopilotParser.extract_metadata(&path);
    assert_eq!(meta.project_path.as_deref(), Some("/a"));
    assert_eq!(meta.git_branch, None);
}

#[test]
fn copilot_session_start_non_dict_context_keeps_branch() {
    // A PRESENT non-object `context` skips the whole block, so the previous
    // branch survives. Expected: project='/a', branch='main'.
    let tmp = TempDir::new().unwrap();
    let path = write_copilot(
        tmp.path(),
        "sess",
        "{\"type\":\"session.start\",\"data\":{\"context\":{\"cwd\":\"/a\",\"branch\":\"main\"}}}\n\
         {\"type\":\"session.start\",\"data\":{\"context\":\"nope\"}}\n",
        None,
    );
    let meta = CopilotParser.extract_metadata(&path);
    assert_eq!(meta.project_path.as_deref(), Some("/a"));
    assert_eq!(meta.git_branch.as_deref(), Some("main"));
}

#[test]
fn copilot_null_id_generates_fallback() {
    // A present-but-null `id` gets the same generated fallback as an absent
    // key, so no NULL id can reach the DB.
    let tmp = TempDir::new().unwrap();
    let path = write_copilot(
        tmp.path(),
        "sess",
        "{\"type\":\"user.message\",\"id\":null,\"data\":{\"content\":\"hi\"}}\n",
        None,
    );
    let t = CopilotParser.parse_transcript(&path);
    assert_eq!(t.events.len(), 1);
    assert_eq!(t.events[0].id, "event_0");
    assert_eq!(t.events[0].sequence, 0);
    assert_eq!(t.events[0].content, "hi");
}

#[test]
fn copilot_null_tool_call_id_generates_fallback() {
    // A null `toolCallId` leaves `tool_call_id` None but still gives the
    // event the generated id:
    //   [0] id='event_0'  seq=0 message assistant 'x'
    //   [1] id='tool_0_0' seq=1 tool_call name='bash' tool_call_id=None
    let tmp = TempDir::new().unwrap();
    let path = write_copilot(
        tmp.path(),
        "sess",
        "{\"type\":\"assistant.message\",\"data\":{\"content\":\"x\",\"toolRequests\":[{\"toolCallId\":null,\"name\":\"bash\",\"arguments\":{\"a\":1}}]}}\n",
        None,
    );
    let t = CopilotParser.parse_transcript(&path);
    assert_eq!(t.events.len(), 2);
    assert_eq!(t.events[1].id, "tool_0_0");
    assert_eq!(t.events[1].sequence, 1);
    assert_eq!(t.events[1].tool_name.as_deref(), Some("bash"));
    assert_eq!(t.events[1].tool_call_id, None);
}

#[test]
fn copilot_non_string_tool_call_id_coerced() {
    // A numeric `toolCallId` is stringified, so the stored tool_call_id "5"
    // agrees with the "result_5" event id instead of dropping to None.
    let tmp = TempDir::new().unwrap();
    let path = write_copilot(
        tmp.path(),
        "sess",
        "{\"type\":\"tool.execution_complete\",\"data\":{\"toolCallId\":5,\"success\":true,\"result\":{\"content\":\"out\"}}}\n",
        None,
    );
    let t = CopilotParser.parse_transcript(&path);
    assert_eq!(t.events.len(), 1);
    assert_eq!(t.events[0].id, "result_5");
    assert_eq!(t.events[0].tool_call_id.as_deref(), Some("5"));
    assert_eq!(t.events[0].tool_output.as_deref(), Some("out"));
    assert_eq!(t.events[0].tool_success, Some(true));
}

#[test]
fn copilot_non_string_tool_request_fields_coerced() {
    // Numeric `toolCallId` and `name` are both stringified:
    //   [0] id="5" seq=1 tool_call tool_name="7" tool_input={} tool_call_id="5"
    // (the assistant.message itself has no content, so no message event).
    let tmp = TempDir::new().unwrap();
    let path = write_copilot(
        tmp.path(),
        "sess",
        "{\"type\":\"assistant.message\",\"data\":{\"toolRequests\":[{\"toolCallId\":5,\"name\":7,\"arguments\":{}}]}}\n",
        None,
    );
    let t = CopilotParser.parse_transcript(&path);
    assert_eq!(t.events.len(), 1);
    assert_eq!(t.events[0].id, "5");
    assert_eq!(t.events[0].sequence, 1);
    assert_eq!(t.events[0].tool_name.as_deref(), Some("7"));
    assert_eq!(t.events[0].tool_call_id.as_deref(), Some("5"));
}

#[test]
fn copilot_yaml_11_vs_12_divergence() {
    // DIVERGENCE (documented at the parse site in copilot.rs): serde_yaml
    // implements the YAML 1.2 core schema, so the unquoted scalars a YAML 1.1
    // resolver would coerce — `yes`/`no` to booleans, `12:30` to the
    // sexagesimal 750 — stay the strings "yes"/"no"/"12:30" and survive as
    // the title text. Non-string YAML scalars that do reach here end up as ""
    // because `title` is a String. We do not switch YAML libraries for this.
    let tmp = TempDir::new().unwrap();
    let path = write_copilot(
        tmp.path(),
        "sess",
        "{\"type\":\"user.message\",\"data\":{\"content\":\"hi\"}}\n",
        Some("summary: yes\ncwd: /repo\n"),
    );
    let meta = CopilotParser.extract_metadata(&path);
    assert_eq!(meta.title.as_deref(), Some("yes"));
    assert_eq!(meta.project_path.as_deref(), Some("/repo"));
}

#[test]
fn claude_null_tool_use_id_generates_fallback() {
    // A null `tool_use` id leaves `tool_call_id` None but still gives the
    // event the generated fallback id.
    let tmp = TempDir::new().unwrap();
    let path = write_claude(
        tmp.path(),
        "s.jsonl",
        "{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":null,\"name\":\"Read\",\"input\":{}}]}}\n",
    );
    let t = ClaudeParser.parse_transcript(&path);
    assert_eq!(t.events.len(), 1);
    assert_eq!(t.events[0].id, "tool_0_0");
    assert_eq!(t.events[0].sequence, 1);
    assert_eq!(t.events[0].tool_name.as_deref(), Some("Read"));
    assert_eq!(t.events[0].tool_call_id, None);
}

#[test]
fn claude_duplicate_sequence_quirk() {
    // Fixture A1: a text-less assistant entry emits ONE event whose sequence
    // is start_sequence + 1, but the caller only advances the counter by the
    // number of events emitted (1). The next entry therefore starts at
    // sequence 1 again — two events share sequence 1. This pins the quirk so
    // nobody "fixes" it by accident.
    //
    // Expected: event_count = 2
    //   [0] id='t1'          seq=1 tool_call name='Read'
    //                        input={'file_path': '/a'} tool_call_id='t1'
    //   [1] id='assistant_1' seq=1 message assistant content='done'
    let tmp = TempDir::new().unwrap();
    let path = write_claude(
        tmp.path(),
        "s.jsonl",
        "{\"type\":\"assistant\",\"timestamp\":\"2025-01-01T00:00:00Z\",\"message\":{\"content\":[{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"Read\",\"input\":{\"file_path\":\"/a\"}}]}}\n\
         {\"type\":\"assistant\",\"timestamp\":\"2025-01-01T00:00:01Z\",\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"done\"}]}}\n",
    );
    let t = ClaudeParser.parse_transcript(&path);
    assert_eq!(t.metadata.event_count, 2);
    assert_eq!(t.events[0].id, "t1");
    assert_eq!(t.events[0].sequence, 1);
    assert_eq!(t.events[0].event_type, EventType::ToolCall);
    assert_eq!(t.events[0].tool_name.as_deref(), Some("Read"));
    assert_eq!(t.events[0].tool_input, Some(json!({"file_path": "/a"})));
    assert_eq!(t.events[0].tool_call_id.as_deref(), Some("t1"));

    assert_eq!(t.events[1].id, "assistant_1");
    assert_eq!(t.events[1].sequence, 1); // DUPLICATE of events[0]
    assert_eq!(t.events[1].event_type, EventType::Message);
    assert_eq!(t.events[1].role, Some(MessageRole::Assistant));
    assert_eq!(t.events[1].content, "done");
}

#[cfg(unix)]
#[test]
fn claude_discover_does_not_follow_directory_symlinks() {
    // With base/proj/a.jsonl real and base/linked -> ../elsewhere
    // (containing b.jsonl), discovery yields only proj/a.jsonl — it does not
    // descend into the symlinked directory.
    let tmp = TempDir::new().unwrap();
    let base = tmp.path().join("projects");
    let proj = base.join("proj");
    fs::create_dir_all(&proj).unwrap();
    let real_session = proj.join("a.jsonl");
    fs::write(&real_session, "{}\n").unwrap();

    let elsewhere = tmp.path().join("elsewhere");
    fs::create_dir_all(&elsewhere).unwrap();
    fs::write(elsewhere.join("b.jsonl"), "{}\n").unwrap();
    std::os::unix::fs::symlink(&elsewhere, base.join("linked")).unwrap();

    let found = ClaudeParser.discover_sessions(&base);
    assert_eq!(found, vec![real_session]);
}

#[cfg(unix)]
#[test]
fn claude_discover_symlink_cycle_terminates() {
    // A symlink cycle cannot be walked at all now that directory symlinks are
    // skipped; MAX_RGLOB_DEPTH is the second line of defence. This test just
    // proves discovery terminates and still finds the real session.
    let tmp = TempDir::new().unwrap();
    let base = tmp.path().join("projects");
    let proj = base.join("proj");
    fs::create_dir_all(&proj).unwrap();
    let real_session = proj.join("a.jsonl");
    fs::write(&real_session, "{}\n").unwrap();
    std::os::unix::fs::symlink(&base, proj.join("loop")).unwrap();

    let found = ClaudeParser.discover_sessions(&base);
    assert_eq!(found, vec![real_session]);
}

#[test]
fn vscode_non_string_tool_ids_coerced() {
    // Numeric `toolId` and `toolCallId` are stringified, not dropped. For
    //   response=[{kind:toolInvocationSerialized, toolId:7, toolCallId:5}]
    //   [0] id='user_r1' seq=0 message
    //   [1] id="5" seq=1 tool_call tool_name="7" tool_input={} tool_call_id="5"
    let tmp = TempDir::new().unwrap();
    let path = write_vscode(
        tmp.path(),
        "s.json",
        r#"{"requests":[{"requestId":"r1","message":{"text":"hi"},"response":[{"kind":"toolInvocationSerialized","toolId":7,"toolCallId":5}]}]}"#,
        None,
    );
    let t = VSCodeCopilotParser.parse_transcript(&path);
    assert_eq!(t.events.len(), 2);
    assert_eq!(t.events[1].id, "5");
    assert_eq!(t.events[1].sequence, 1);
    assert_eq!(t.events[1].tool_name.as_deref(), Some("7"));
    assert_eq!(t.events[1].tool_call_id.as_deref(), Some("5"));
}

#[test]
fn vscode_null_tool_call_id_generates_fallback() {
    // A null `toolId` keeps tool_name None, but a null `toolCallId` still
    // gets the generated fallback id so no NULL id reaches the DB.
    let tmp = TempDir::new().unwrap();
    let path = write_vscode(
        tmp.path(),
        "s.json",
        r#"{"requests":[{"requestId":"r1","response":[{"kind":"toolInvocationSerialized","toolId":null,"toolCallId":null}]}]}"#,
        None,
    );
    let t = VSCodeCopilotParser.parse_transcript(&path);
    assert_eq!(t.events.len(), 1);
    assert_eq!(t.events[0].id, "tool_0_0");
    assert_eq!(t.events[0].tool_name, None);
    assert_eq!(t.events[0].tool_call_id.as_deref(), Some("tool_0_0"));
}

// ═══════════════════════════════════════════════════════════════════════
// CodexParser
// ═══════════════════════════════════════════════════════════════════════
//
// The fixture below is assembled from the record shapes of real Codex 0.153
// rollouts (2026-09-07); expected values were verified by hand against those
// files.

const CODEX_THREAD_ID: &str = "01a074eb-3c11-71c3-9524-3587c8a5495a";

const CODEX_RICH: &str = r##"{"timestamp":"2026-09-06T04:13:22.000Z","ordinal":0,"type":"session_meta","payload":{"session_id":"01a074eb-3c11-71c3-9524-3587c8a5495a","id":"01a074eb-3c11-71c3-9524-3587c8a5495a","timestamp":"2026-09-06T04:13:20.000Z","cwd":"/Users/u/proj","originator":"Codex Desktop","cli_version":"0.153.4","source":"vscode","thread_source":"user","model_provider":"openai","git":{"commit_hash":"889d6260","branch":"main"},"base_instructions":{"text":"You are Codex."}}}
{"timestamp":"2026-09-06T04:13:22.100Z","ordinal":1,"type":"turn_context","payload":{"turn_id":"t1","cwd":"/Users/u/proj","model":"gpt-5.6-sol","effort":"medium"}}
{"timestamp":"2026-09-06T04:13:22.200Z","ordinal":2,"type":"response_item","payload":{"type":"message","id":"msg_dev","role":"developer","content":[{"type":"input_text","text":"# Permissions\nYou may write to /Users/u/proj."}],"internal_chat_message_metadata_passthrough":{"content_item_kinds":["permissions.instructions"]}}}
{"timestamp":"2026-09-06T04:13:22.300Z","ordinal":3,"type":"response_item","payload":{"type":"message","id":"msg_ctx","role":"user","content":[{"type":"input_text","text":"<recommended_plugins>\n- Airtable\n</recommended_plugins>"},{"type":"input_text","text":"<environment_context>\n<cwd>/Users/u/proj</cwd>\n</environment_context>"}],"internal_chat_message_metadata_passthrough":{"content_item_kinds":["plugins.recommendations","environments.environment_context"]}}}
{"timestamp":"2026-09-06T04:13:22.400Z","ordinal":4,"type":"response_item","payload":{"type":"message","id":"msg_u1","role":"user","content":[{"type":"input_text","text":"Read the patents and the signed declaration.  \n"}],"internal_chat_message_metadata_passthrough":{"turn_id":"t1","content_item_kinds":["user.text"]}}}
{"timestamp":"2026-09-06T04:13:22.500Z","ordinal":5,"type":"event_msg","payload":{"type":"item_completed","thread_id":"01a074eb-3c11-71c3-9524-3587c8a5495a","turn_id":"t1","item":{"type":"UserMessage","id":"ui1","content":[{"type":"text","text":"Read the patents and the signed declaration.  \n"}]}}}
{"timestamp":"2026-09-06T04:13:25.000Z","ordinal":6,"type":"response_item","payload":{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"gAAAAAB"}}
{"timestamp":"2026-09-06T04:13:26.000Z","ordinal":7,"type":"response_item","payload":{"type":"custom_tool_call","id":"ctc_1","status":"completed","call_id":"call_1","name":"exec","input":"text(await tools.exec_command({cmd:\"rg --files -g '*.pdf'\",max_output_tokens:5000}));\n"}}
{"timestamp":"2026-09-06T04:13:26.300Z","ordinal":8,"type":"response_item","payload":{"type":"custom_tool_call_output","id":"ctco_1","call_id":"call_1","output":[{"type":"input_text","text":"Script completed\nWall time 0.2 seconds\nOutput:\n"},{"type":"input_text","text":"{\"chunk_id\":\"208e90\",\"exit_code\":0,\"output\":\"decl.pdf\\n\"}"}]}}
{"timestamp":"2026-09-06T04:13:30.000Z","ordinal":9,"type":"response_item","payload":{"type":"message","id":"msg_a1","role":"assistant","content":[{"type":"output_text","text":"Found one PDF."}],"phase":"commentary"}}
{"timestamp":"2026-09-06T04:13:31.000Z","ordinal":10,"type":"response_item","payload":{"type":"function_call","id":"fc_1","name":"request_user_input_async","arguments":"{\"questions\":[{\"title\":\"Which patent first?\"}]}","call_id":"call_2"}}
{"timestamp":"2026-09-06T04:13:31.100Z","ordinal":11,"type":"response_item","payload":{"type":"function_call_output","id":"fco_1","call_id":"call_2","output":"{\"accepted\":true}"}}
{"timestamp":"2026-09-06T04:13:40.000Z","ordinal":12,"type":"response_item","payload":{"type":"custom_tool_call","id":"ctc_2","status":"completed","call_id":"call_3","name":"apply_patch","input":"*** Begin Patch\n*** Update File: notes.md\n@@\n+read decl.pdf\n*** Add File: todo.md\n+later\n*** End Patch\n"}}
{"timestamp":"2026-09-06T04:13:40.200Z","ordinal":13,"type":"response_item","payload":{"type":"custom_tool_call_output","id":"ctco_2","call_id":"call_3","output":[{"type":"input_text","text":"Done"}]}}
{"timestamp":"2026-09-06T04:13:45.000Z","ordinal":14,"type":"response_item","payload":{"type":"message","id":"msg_a2","role":"assistant","content":[{"type":"output_text","text":"Patched notes.md and added todo.md."}],"phase":"final_answer"}}
{"timestamp":"2026-09-06T04:14:00.000Z","ordinal":15,"type":"response_item","payload":{"type":"agent_message","id":"amsg_1","author":"/root/helper","recipient":"/root","content":[{"type":"input_text","text":"Message Type: FINAL_ANSWER\nPayload:\nAll clear."}]}}
{"timestamp":"2026-09-06T04:15:00.000Z","ordinal":16,"type":"compacted","payload":{"message":"","replacement_history":[{"type":"message","id":"msg_u1","role":"user","content":[{"type":"input_text","text":"Read the patents and the signed declaration.  \n"}],"internal_chat_message_metadata_passthrough":{"content_item_kinds":["user.text"]}}]}}
not json
{"timestamp":"2026-09-06T04:20:00.000Z","ordinal":17,"type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"total_tokens":25790}}}}
"##;

const CODEX_SESSION_INDEX: &str = r#"{"id":"01a06965-1af7-7561-8bba-b26965ad5217","thread_name":"Diagnose MacBook fan issue","updated_at":"2026-09-03T22:30:43.815553Z"}
{"id":"01a074eb-3c11-71c3-9524-3587c8a5495a","thread_name":"Review patents and declaration","updated_at":"2026-09-06T04:13:20.536907Z"}
"#;

/// A rollout in Codex's own layout: `<codex_home>/sessions/YYYY/MM/DD/`.
/// Returns `(codex_home, transcript_path)`.
fn write_codex(root: &Path, thread_id: &str, content: &str, index: Option<&str>) -> (PathBuf, PathBuf) {
    let codex_home = root.join(".codex");
    let day = codex_home.join("sessions").join("2026").join("09").join("06");
    fs::create_dir_all(&day).unwrap();
    let path = day.join(format!("rollout-2026-09-06T04-13-20-{thread_id}.jsonl"));
    fs::write(&path, content).unwrap();
    if let Some(index) = index {
        fs::write(codex_home.join("session_index.jsonl"), index).unwrap();
    }
    (codex_home, path)
}

#[test]
fn codex_discover_nested_dated_dirs_and_flat_archive() {
    let tmp = TempDir::new().unwrap();
    let (codex_home, nested) = write_codex(tmp.path(), CODEX_THREAD_ID, "{}\n", None);
    let sessions = codex_home.join("sessions");
    // A second dated dir, sorted after the first.
    let later = sessions.join("2026").join("09").join("07");
    fs::create_dir_all(&later).unwrap();
    let later_file = later.join("rollout-2026-09-07T13-19-45-01a07d86-dc97-76c3-adcc-dd2942bbd177.jsonl");
    fs::write(&later_file, "{}\n").unwrap();
    // Noise: a jsonl that is not a rollout, and a non-jsonl rollout-ish file.
    fs::write(sessions.join("2026").join("09").join("06").join("notes.jsonl"), "{}").unwrap();
    fs::write(sessions.join("rollout-stray.txt"), "x").unwrap();
    // A flat file directly under the base (archived_sessions layout).
    let flat = sessions.join("rollout-2026-08-01T00-00-00-0a0a0a0a-0000-0000-0000-000000000000.jsonl");
    fs::write(&flat, "{}\n").unwrap();

    let found = CodexParser.discover_sessions(&sessions);
    let mut expected = vec![nested, later_file, flat];
    expected.sort();
    assert_eq!(found, expected);
}

#[test]
fn codex_discover_nonexistent_base() {
    let tmp = TempDir::new().unwrap();
    assert!(CodexParser.discover_sessions(&tmp.path().join("nope")).is_empty());
}

#[test]
fn codex_metadata() {
    let tmp = TempDir::new().unwrap();
    let (_, path) = write_codex(tmp.path(), CODEX_THREAD_ID, CODEX_RICH, Some(CODEX_SESSION_INDEX));
    let meta = CodexParser.extract_metadata(&path);

    assert_eq!(meta.session_id, CODEX_THREAD_ID);
    assert_eq!(meta.cli_source, "codex");
    assert_eq!(meta.parser_version, "1.0");
    assert_eq!(meta.transcript_path, path.to_string_lossy());
    // Title from session_index.jsonl four levels up.
    assert_eq!(meta.title.as_deref(), Some("Review patents and declaration"));
    // First prompt is what was typed, stripped — not the injected context
    // that precedes it in the file.
    assert_eq!(
        meta.first_prompt.as_deref(),
        Some("Read the patents and the signed declaration.")
    );
    assert_eq!(meta.project_path.as_deref(), Some("/Users/u/proj"));
    assert_eq!(meta.git_branch.as_deref(), Some("main"));
    assert_eq!(meta.model.as_deref(), Some("gpt-5.6-sol"));
    assert_eq!(meta.entrypoint.as_deref(), Some("vscode"));
    // created_at is the thread's own timestamp (session_meta.payload),
    // modified_at the last record written — including non-content records.
    assert_eq!(meta.created_at, Some(utc(2026, 9, 6, 4, 13, 20)));
    assert_eq!(meta.modified_at, Some(utc(2026, 9, 6, 4, 20, 0)));
    // 1 typed user + 2 assistant; the injected context, the developer
    // message, the item_completed echo and the compacted replay do not count.
    assert_eq!(meta.message_count, 3);
    assert!(meta.source_hash.is_some());
    assert!(CodexParser.validate_transcript(&path));
}

#[test]
fn codex_metadata_without_index_has_no_title() {
    let tmp = TempDir::new().unwrap();
    let (_, path) = write_codex(tmp.path(), CODEX_THREAD_ID, CODEX_RICH, None);
    let meta = CodexParser.extract_metadata(&path);
    // None, not Some("") — the upsert must not clobber an LLM title.
    assert_eq!(meta.title, None);
}

#[test]
fn codex_events() {
    let tmp = TempDir::new().unwrap();
    let (_, path) = write_codex(tmp.path(), CODEX_THREAD_ID, CODEX_RICH, None);
    let parsed = CodexParser.parse_transcript(&path);
    assert_eq!(parsed.metadata.event_count, 10);
    let ev = &parsed.events;

    let kinds: Vec<(i64, &str, &str)> = ev
        .iter()
        .map(|e| (e.sequence, e.event_type.as_str(), e.id.as_str()))
        .collect();
    assert_eq!(
        kinds,
        vec![
            (0, "message", "user_0"),
            (1, "tool_call", "call_1"),
            (2, "tool_result", "result_call_1"),
            (3, "message", "assistant_3"),
            (4, "tool_call", "call_2"),
            (5, "tool_result", "result_call_2"),
            (6, "tool_call", "call_3"),
            (7, "tool_result", "result_call_3"),
            (8, "message", "assistant_8"),
            // agent_message is emitted as assistant text (see the next test).
            (9, "message", "agent_9"),
        ]
    );

    // User message: stripped, role user.
    assert_eq!(ev[0].role, Some(MessageRole::User));
    assert_eq!(ev[0].content, "Read the patents and the signed declaration.");
    // Record timestamps carry milliseconds; they survive parsing.
    assert_eq!(
        ev[0].timestamp,
        Some(DateTime::parse_from_rfc3339("2026-09-06T04:13:22.400Z").unwrap())
    );

    // exec: raw script kept under `content`, command lifted out.
    assert_eq!(ev[1].tool_name.as_deref(), Some("exec"));
    assert_eq!(ev[1].tool_call_id.as_deref(), Some("call_1"));
    let input = ev[1].tool_input.as_ref().unwrap();
    assert_eq!(input["command"], json!("rg --files -g '*.pdf'"));
    assert!(input["content"].as_str().unwrap().starts_with("text(await tools.exec_command("));

    // exec output: text concatenated, success read from the exit_code.
    assert_eq!(ev[2].tool_call_id.as_deref(), Some("call_1"));
    assert_eq!(ev[2].tool_success, Some(true));
    assert!(ev[2].tool_output.as_deref().unwrap().starts_with("Script completed\n"));
    assert!(ev[2].tool_output.as_deref().unwrap().contains("decl.pdf"));

    assert_eq!(ev[3].role, Some(MessageRole::Assistant));
    assert_eq!(ev[3].content, "Found one PDF.");

    // function_call: JSON-string arguments decoded into an object.
    assert_eq!(ev[4].tool_name.as_deref(), Some("request_user_input_async"));
    assert_eq!(ev[4].tool_input.as_ref().unwrap()["questions"][0]["title"], json!("Which patent first?"));
    // Plain string output with no exit code: no success verdict.
    assert_eq!(ev[5].tool_output.as_deref(), Some("{\"accepted\":true}"));
    assert_eq!(ev[5].tool_success, None);

    // apply_patch: patch under `content`, touched files listed.
    assert_eq!(ev[6].tool_name.as_deref(), Some("apply_patch"));
    assert_eq!(ev[6].tool_input.as_ref().unwrap()["files"], json!(["notes.md", "todo.md"]));
    assert_eq!(ev[7].tool_output.as_deref(), Some("Done"));

    assert_eq!(ev[8].content, "Patched notes.md and added todo.md.");
}

#[test]
fn codex_agent_message_is_assistant_text_with_provenance() {
    // Subagent rollouts are skipped and spawn/send payloads are encrypted,
    // so the inter-agent message is the only readable record of what the
    // agents found: it must reach conversation text.
    let tmp = TempDir::new().unwrap();
    let (_, path) = write_codex(tmp.path(), CODEX_THREAD_ID, CODEX_RICH, None);
    let parsed = CodexParser.parse_transcript(&path);
    let agent = parsed.events.last().unwrap();
    assert_eq!(agent.id, "agent_9");
    assert_eq!(agent.event_type, EventType::Message);
    assert_eq!(agent.role, Some(MessageRole::Assistant));
    assert_eq!(
        agent.content,
        "[agent message /root/helper → /root]\nMessage Type: FINAL_ANSWER\nPayload:\nAll clear."
    );
}

#[test]
fn codex_conversation_text() {
    let tmp = TempDir::new().unwrap();
    let (_, path) = write_codex(tmp.path(), CODEX_THREAD_ID, CODEX_RICH, None);
    let parsed = CodexParser.parse_transcript(&path);
    // Developer instructions, injected context, reasoning and every
    // event_msg are absent; the agent message is assistant text; tool calls
    // render through the Codex branches of format_tool_call.
    let expected = "Human:\nRead the patents and the signed declaration.\n\n\
[Tool Calls]\nExec: rg --files -g '*.pdf'\n\n\
Assistant:\nFound one PDF.\n\n\
[Tool Calls]\nrequest_user_input_async: questions=[{'title': 'Which patent first?'}]\nPatch notes.md, todo.md\n\n\
Assistant:\nPatched notes.md and added todo.md.\n\n\
Assistant:\n[agent message /root/helper → /root]\nMessage Type: FINAL_ANSWER\nPayload:\nAll clear.\n";
    assert_eq!(parsed.get_conversation_text(true), expected);
}

#[test]
fn codex_files_touched_from_patch_only() {
    let tmp = TempDir::new().unwrap();
    let (_, path) = write_codex(tmp.path(), CODEX_THREAD_ID, CODEX_RICH, None);
    let parsed = CodexParser.parse_transcript(&path);
    let files = ekko::indexer::extract_file_paths(&parsed.events);
    // apply_patch is a write tool; the exec call names no file.
    assert_eq!(
        files,
        vec![
            ("notes.md".to_string(), "write".to_string()),
            ("todo.md".to_string(), "write".to_string()),
        ]
    );
}

#[test]
fn codex_subagent_and_exec_threads_are_non_interactive() {
    let tmp = TempDir::new().unwrap();
    let meta_line = |source: &str, thread_source: &str| {
        format!(
            r#"{{"timestamp":"2026-09-07T20:29:36.132Z","type":"session_meta","payload":{{"id":"{CODEX_THREAD_ID}","cwd":"/p","source":{source},"thread_source":"{thread_source}"}}}}"#
        )
    };
    let cases = [
        (r#"{"subagent":{"thread_spawn":{"parent_thread_id":"x","depth":1}}}"#, "subagent", Some("subagent"), false),
        (r#"{"subagent":{"thread_spawn":{"parent_thread_id":"x","depth":1}}}"#, "guardian_review", Some("subagent"), false),
        (r#""exec""#, "user", Some("exec"), false),
        (r#""vscode""#, "user", Some("vscode"), true),
        (r#""cli""#, "user", Some("cli"), true),
        (r#""mcp""#, "user", Some("mcp"), false),
    ];
    for (i, (source, thread_source, entrypoint, interactive)) in cases.iter().enumerate() {
        let dir = tmp.path().join(format!("case{i}"));
        let (_, path) = write_codex(&dir, CODEX_THREAD_ID, &meta_line(source, thread_source), None);
        let meta = CodexParser.extract_metadata(&path);
        assert_eq!(meta.entrypoint.as_deref(), *entrypoint, "case {i}");
        assert_eq!(ekko::indexer::is_interactive(&meta), *interactive, "case {i}");
    }
    // No `source` at all (older Codex): assumed interactive.
    let (_, path) = write_codex(
        &tmp.path().join("legacy"),
        CODEX_THREAD_ID,
        &format!(r#"{{"timestamp":"2026-09-07T20:29:36.132Z","type":"session_meta","payload":{{"id":"{CODEX_THREAD_ID}","cwd":"/p"}}}}"#),
        None,
    );
    let meta = CodexParser.extract_metadata(&path);
    assert_eq!(meta.entrypoint, None);
    assert!(ekko::indexer::is_interactive(&meta));
    // No `source` but a guardian `thread_source`: still a spawned thread.
    let (_, path) = write_codex(
        &tmp.path().join("legacy-guardian"),
        CODEX_THREAD_ID,
        &format!(r#"{{"timestamp":"2026-09-07T20:29:36.132Z","type":"session_meta","payload":{{"id":"{CODEX_THREAD_ID}","cwd":"/p","thread_source":"guardian_review"}}}}"#),
        None,
    );
    let meta = CodexParser.extract_metadata(&path);
    assert_eq!(meta.entrypoint.as_deref(), Some("subagent"));
    assert!(!ekko::indexer::is_interactive(&meta));
}

#[test]
fn codex_session_id_falls_back_to_filename() {
    let tmp = TempDir::new().unwrap();
    // No session_meta line at all.
    let body = r#"{"timestamp":"2026-09-06T04:13:22.400Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}],"internal_chat_message_metadata_passthrough":{"content_item_kinds":["user.text"]}}}
"#;
    let (_, path) = write_codex(tmp.path(), CODEX_THREAD_ID, body, None);
    let meta = CodexParser.extract_metadata(&path);
    assert_eq!(meta.session_id, CODEX_THREAD_ID);
    assert_eq!(meta.project_path.as_deref(), Some(""));
    assert_eq!(meta.first_prompt.as_deref(), Some("hi"));
    assert_eq!(meta.message_count, 1);
}

#[test]
fn codex_legacy_rollout_without_kinds_uses_shape_heuristic_and_shell_tool() {
    let tmp = TempDir::new().unwrap();
    let body = r#"{"timestamp":"2025-11-01T10:00:00.000Z","type":"session_meta","payload":{"id":"01a074eb-3c11-71c3-9524-3587c8a5495a","cwd":"/p","originator":"codex_cli_rs","cli_version":"0.45.0"}}
{"timestamp":"2025-11-01T10:00:01.000Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"<user_instructions>\nBe terse.\n</user_instructions>"}]}}
{"timestamp":"2025-11-01T10:00:01.500Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"<environment_context>\n<cwd>/p</cwd>\n</environment_context>"}]}}
{"timestamp":"2025-11-01T10:00:02.000Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"why does <div> render twice?"}]}}
{"timestamp":"2025-11-01T10:00:03.000Z","type":"response_item","payload":{"type":"function_call","name":"shell","arguments":"{\"command\":[\"bash\",\"-lc\",\"cargo test\"],\"workdir\":\"/p\"}","call_id":"c1"}}
{"timestamp":"2025-11-01T10:00:04.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"c1","output":"{\"output\":\"ok\",\"metadata\":{\"exit_code\":0,\"duration_seconds\":1.2}}"}}
{"timestamp":"2025-11-01T10:00:05.000Z","type":"response_item","payload":{"type":"function_call","name":"apply_patch","arguments":"{\"input\":\"*** Begin Patch\\n*** Update File: src/lib.rs\\n@@\\n-a\\n+b\\n*** End Patch\\n\"}","call_id":"c2"}}
{"timestamp":"2025-11-01T10:00:06.000Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Because it mounts twice."}]}}
"#;
    let (_, path) = write_codex(tmp.path(), CODEX_THREAD_ID, body, None);
    let meta = CodexParser.extract_metadata(&path);
    assert_eq!(meta.first_prompt.as_deref(), Some("why does <div> render twice?"));
    assert_eq!(meta.message_count, 2);
    assert_eq!(meta.entrypoint, None);

    let parsed = CodexParser.parse_transcript(&path);
    let expected = "Human:\nwhy does <div> render twice?\n\n\
[Tool Calls]\nShell: cargo test\nPatch src/lib.rs\n\n\
Assistant:\nBecause it mounts twice.\n";
    assert_eq!(parsed.get_conversation_text(true), expected);
    // exit_code inside the legacy metadata envelope is still found.
    let result = parsed.events.iter().find(|e| e.event_type == EventType::ToolResult).unwrap();
    assert_eq!(result.tool_success, Some(true));
    // The function-call form of apply_patch normalizes like the custom form.
    let patch = parsed.events.iter().find(|e| e.tool_name.as_deref() == Some("apply_patch")).unwrap();
    let input = patch.tool_input.as_ref().unwrap();
    assert_eq!(input["files"], json!(["src/lib.rs"]));
    assert!(input.get("input").is_none());
    assert!(input["content"].as_str().unwrap().starts_with("*** Begin Patch"));
}

#[test]
fn codex_bom_and_malformed_lines_tolerated() {
    let tmp = TempDir::new().unwrap();
    let body = format!(
        "\u{FEFF}{}\n{{\"timestamp\":\"2026-09-06T04:13:22.400Z\",\"type\":\"response_item\",\"payload\":\"not an object\"}}\n[1,2]\n{{\"type\":\"response_item\"}}\n{{\"timestamp\":\"2026-09-06T04:13:23.000Z\",\"type\":\"response_item\",\"payload\":{{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{{\"type\":\"output_text\",\"text\":\"ok\"}}]}}}}\n",
        CODEX_RICH.lines().next().unwrap()
    );
    let (_, path) = write_codex(tmp.path(), CODEX_THREAD_ID, &body, None);
    let meta = CodexParser.extract_metadata(&path);
    assert_eq!(meta.session_id, CODEX_THREAD_ID);
    assert_eq!(meta.message_count, 1);
    let parsed = CodexParser.parse_transcript(&path);
    assert_eq!(parsed.events.len(), 1);
    assert_eq!(parsed.events[0].content, "ok");
}

#[test]
fn codex_empty_file() {
    let tmp = TempDir::new().unwrap();
    let (_, path) = write_codex(tmp.path(), CODEX_THREAD_ID, "", None);
    let meta = CodexParser.extract_metadata(&path);
    assert_eq!(meta.session_id, CODEX_THREAD_ID);
    assert_eq!(meta.message_count, 0);
    assert_eq!(meta.created_at, None);
    assert_eq!(meta.first_prompt.as_deref(), Some(""));
    assert!(CodexParser.parse_transcript(&path).events.is_empty());
}

#[test]
fn codex_exec_without_command_keeps_script() {
    let tmp = TempDir::new().unwrap();
    let body = r#"{"timestamp":"2026-09-06T04:13:26.000Z","type":"response_item","payload":{"type":"custom_tool_call","call_id":"c1","name":"exec","input":"image((await tools.view_image({path:'/tmp/a.png'})).image_url);\n"}}
"#;
    let (_, path) = write_codex(tmp.path(), CODEX_THREAD_ID, body, None);
    let parsed = CodexParser.parse_transcript(&path);
    let input = parsed.events[0].tool_input.as_ref().unwrap();
    assert!(input.get("command").is_none());
    assert_eq!(
        parsed.get_conversation_text(true),
        "[Tool Calls]\nExec[script]: image((await tools.view_image({path:'/tmp/a.png'})).image_url);\n"
    );
}

#[test]
fn codex_first_session_meta_wins_over_copied_parent_history() {
    // A subagent (or forked) rollout starts with its own session_meta and
    // then replays the parent's history — parent session_meta included.
    let tmp = TempDir::new().unwrap();
    let own = "01a07d8f-dfbf-78d2-a72f-87272d8d86f7";
    let parent = "01a07d86-dc97-76c3-adcc-dd2942bbd177";
    let body = format!(
        r#"{{"timestamp":"2026-09-07T20:29:36.132Z","ordinal":0,"type":"session_meta","payload":{{"id":"{own}","session_id":"{parent}","forked_from_id":"{parent}","timestamp":"2026-09-07T20:29:36.106Z","cwd":"/Users/u/fork","source":{{"subagent":{{"thread_spawn":{{"parent_thread_id":"{parent}","depth":1}}}}}},"thread_source":"subagent","git":{{"branch":"topic"}},"subagent_history_start_ordinal":30}}}}
{{"timestamp":"2026-09-07T20:29:36.133Z","ordinal":1,"type":"session_meta","payload":{{"id":"{parent}","session_id":"{parent}","timestamp":"2026-09-07T20:19:45.000Z","cwd":"/Users/u/proj","source":"vscode","thread_source":"user","git":{{"branch":"main"}}}}}}
"#
    );
    let (_, path) = write_codex(tmp.path(), own, &body, None);
    let meta = CodexParser.extract_metadata(&path);
    assert_eq!(meta.session_id, own);
    assert_eq!(meta.project_path.as_deref(), Some("/Users/u/fork"));
    assert_eq!(meta.git_branch.as_deref(), Some("topic"));
    assert_eq!(meta.entrypoint.as_deref(), Some("subagent"));
    assert!(!ekko::indexer::is_interactive(&meta));
    assert_eq!(
        meta.created_at,
        Some(DateTime::parse_from_rfc3339("2026-09-07T20:29:36.106Z").unwrap())
    );
}

#[test]
fn codex_exec_nested_apply_patch_becomes_patch_events() {
    // On real rollouts nearly every file edit is `tools.apply_patch(...)`
    // called from inside an exec script, so it must reach the timeline and
    // the files-touched list as a write.
    let tmp = TempDir::new().unwrap();
    let body = r#"{"timestamp":"2026-09-06T04:13:26.000Z","type":"response_item","payload":{"type":"custom_tool_call","call_id":"c9","name":"exec","input":"text(await tools.apply_patch(\"*** Begin Patch\\n*** Add File: /p/docs/a.md\\n+hello\\n*** End Patch\\n\"));\ntext(await tools.exec_command({cmd:\"ls /p/docs\"}));\ntext(await tools.apply_patch(\"*** Begin Patch\\n*** Update File: /p/src/b.rs\\n@@\\n-x\\n+y\\n*** End Patch\\n\"));\n"}}
{"timestamp":"2026-09-06T04:13:26.300Z","type":"response_item","payload":{"type":"custom_tool_call_output","call_id":"c9","output":[{"type":"input_text","text":"{\"exit_code\":0}"}]}}
{"timestamp":"2026-09-06T04:13:27.000Z","type":"response_item","payload":{"type":"custom_tool_call","call_id":"c10","name":"exec","input":"text(await tools.exec_command({cmd:\"cat /p/src/b.rs\"}));\n"}}
"#;
    let (_, path) = write_codex(tmp.path(), CODEX_THREAD_ID, body, None);
    let parsed = CodexParser.parse_transcript(&path);

    let kinds: Vec<(i64, &str, Option<&str>)> = parsed
        .events
        .iter()
        .map(|e| (e.sequence, e.id.as_str(), e.tool_name.as_deref()))
        .collect();
    assert_eq!(
        kinds,
        vec![
            (0, "c9", Some("exec")),
            (1, "c9_patch_0", Some("apply_patch")),
            (2, "c9_patch_1", Some("apply_patch")),
            (3, "result_c9", None),
            (4, "c10", Some("exec")),
        ]
    );
    // Nested patches have no call id of their own — nothing answers them.
    assert_eq!(parsed.events[1].tool_call_id, None);
    assert_eq!(
        parsed.events[1].tool_input.as_ref().unwrap()["files"],
        json!(["/p/docs/a.md"])
    );
    assert_eq!(
        parsed.get_conversation_text(true),
        "[Tool Calls]\nExec: ls /p/docs\nPatch /p/docs/a.md\nPatch /p/src/b.rs\nExec: cat /p/src/b.rs\n"
    );
    assert_eq!(
        ekko::indexer::extract_file_paths(&parsed.events),
        vec![
            ("/p/docs/a.md".to_string(), "write".to_string()),
            ("/p/src/b.rs".to_string(), "write".to_string()),
        ]
    );
}

#[test]
fn codex_exec_with_several_commands_joins_them() {
    let tmp = TempDir::new().unwrap();
    let body = r#"{"timestamp":"2026-09-06T04:13:26.000Z","type":"response_item","payload":{"type":"custom_tool_call","call_id":"c1","name":"exec","input":"text(await tools.exec_command({cmd:\"git status\"}));\ntext(await tools.exec_command({cmd:\"git diff --stat\",max_output_tokens:2000}));\n"}}
"#;
    let (_, path) = write_codex(tmp.path(), CODEX_THREAD_ID, body, None);
    let parsed = CodexParser.parse_transcript(&path);
    assert_eq!(
        parsed.events[0].tool_input.as_ref().unwrap()["command"],
        json!("git status\ngit diff --stat")
    );
    assert_eq!(
        parsed.get_conversation_text(true),
        "[Tool Calls]\nExec: git status; git diff --stat\n"
    );
}

#[test]
fn codex_exec_with_several_envelopes_fails_if_any_failed() {
    let tmp = TempDir::new().unwrap();
    let body = r#"{"timestamp":"2026-09-06T04:13:26.000Z","type":"response_item","payload":{"type":"custom_tool_call","call_id":"c1","name":"exec","input":"text(await tools.exec_command({cmd:\"true\"}));\ntext(await tools.exec_command({cmd:\"false\"}));\n"}}
{"timestamp":"2026-09-06T04:13:26.300Z","type":"response_item","payload":{"type":"custom_tool_call_output","call_id":"c1","output":[{"type":"input_text","text":"{\"chunk_id\":\"a\",\"exit_code\":0,\"output\":\"\"}"},{"type":"input_text","text":"{\"chunk_id\":\"b\",\"exit_code\":1,\"output\":\"\"}"}]}}
"#;
    let (_, path) = write_codex(tmp.path(), CODEX_THREAD_ID, body, None);
    let parsed = CodexParser.parse_transcript(&path);
    assert_eq!(parsed.events[1].tool_success, Some(false));
}

#[test]
fn codex_command_mentioning_a_call_token_is_kept_whole() {
    // `exec_command(` inside a string is not a call boundary, and a patch
    // whose body mentions `apply_patch(` is still one patch.
    let tmp = TempDir::new().unwrap();
    let body = r#"{"timestamp":"2026-09-06T04:13:26.000Z","type":"response_item","payload":{"type":"custom_tool_call","call_id":"c1","name":"exec","input":"text(await tools.exec_command({cmd:\"rg -n 'exec_command(' src\"}));\ntext(await tools.apply_patch(\"*** Begin Patch\\n*** Update File: src/parsers/codex.rs\\n@@\\n+    call_argument_texts(script, \\\"apply_patch(\\\")\\n*** End Patch\\n\"));\n"}}
"#;
    let (_, path) = write_codex(tmp.path(), CODEX_THREAD_ID, body, None);
    let parsed = CodexParser.parse_transcript(&path);
    assert_eq!(
        parsed.get_conversation_text(true),
        "[Tool Calls]\nExec: rg -n 'exec_command(' src\nPatch src/parsers/codex.rs\n"
    );
    assert_eq!(
        ekko::indexer::extract_file_paths(&parsed.events),
        vec![("src/parsers/codex.rs".to_string(), "write".to_string())]
    );
}

#[test]
fn codex_mixed_context_and_typed_items_keep_only_typed() {
    let tmp = TempDir::new().unwrap();
    let body = r#"{"timestamp":"2026-09-06T04:13:22.400Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"<environment_context>\n<cwd>/p</cwd>\n</environment_context>"},{"type":"input_text","text":"why is the build red?"}],"internal_chat_message_metadata_passthrough":{"content_item_kinds":["environments.environment_context","user.text"]}}}
"#;
    let (_, path) = write_codex(tmp.path(), CODEX_THREAD_ID, body, None);
    let meta = CodexParser.extract_metadata(&path);
    assert_eq!(meta.first_prompt.as_deref(), Some("why is the build red?"));
    let parsed = CodexParser.parse_transcript(&path);
    assert_eq!(parsed.events[0].content, "why is the build red?");
}
