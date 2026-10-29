//! Integration tests for the MCP server: tool inventory and FAST-tool calls
//! through the real tool handlers.
//!
//! # Network / cost / data isolation
//!
//! Same pattern as `background_test.rs`: `HOME`, `XDG_DATA_HOME` and
//! `APPDATA` are pinned into a temp directory and the API-key variables are
//! removed **before** anything reads the config singleton, with `EKKO_MODEL`
//! left at a model whose provider is known (`gpt-5-mini`). Consequences:
//!
//! - `check_llm_available` returns `false`, so the `[LLM]` tools return
//!   their guidance payload without opening a socket;
//! - `get_status` resolves the data dir inside the sandbox, so it never
//!   reports on (or touches) the developer's real `~/.local/share/ekko`;
//! - no test starts the background indexer, so nothing scans session
//!   directories.
//!
//! The calls go through the `#[tool]`-annotated handler methods (the same
//! functions the router dispatches to), not over a stdio transport — a
//! round-trip test would add a process boundary without exercising anything
//! these do not.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ekko::config::Config;
use ekko::db::Database;
use ekko::mcp::EkkoServer;
use ekko::mcp::params::*;
use ekko::memory::{ReadTranscriptArgs, RecentActivityArgs, EkkoMemory};
use ekko::models::SessionMetadata;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock};
use serde_json::{Value, json};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Environment sandbox
// ---------------------------------------------------------------------------

/// The process-wide fake home. Created exactly once, before any test does
/// work, so the environment mutation is race-free.
fn sandbox() -> &'static Path {
    static SANDBOX: OnceLock<TempDir> = OnceLock::new();
    SANDBOX
        .get_or_init(|| {
            let dir = TempDir::new().expect("tempdir");
            fs::create_dir_all(dir.path().join("share/ekko")).unwrap();
            // SAFETY: every test in this binary blocks on this `OnceLock`
            // before touching the environment or the config singleton.
            unsafe {
                std::env::set_var("HOME", dir.path());
                std::env::set_var("XDG_DATA_HOME", dir.path().join("share"));
                std::env::remove_var("APPDATA");
                std::env::set_var("WIN_HOME", "/mnt/c/Users/ekko-test-no-such-user");
                // `CODEX_HOME` would otherwise point `scan_all` at a real Codex home
                // (launchers such as Orca set it) from inside the sandbox.
                std::env::remove_var("CODEX_HOME");
                std::env::remove_var("OPENAI_API_KEY");
                std::env::remove_var("ANTHROPIC_API_KEY");
                // Pin the API backend: with no key, auto-detection would pick a
                // locally installed `claude` and the LLM gate would open.
                std::env::set_var("EKKO_BACKEND", "api");
                std::env::set_var("EKKO_QUERY_BACKEND", "api");
                std::env::set_var("EKKO_QUERY_MODEL", "gpt-5-mini");
                std::env::set_var("EKKO_MODEL", "gpt-5-mini");
            }
            dir
        })
        .path()
}

/// A keyless configuration, leaked so it satisfies the server's `'static`
/// bound (the real binary uses the process-wide singleton).
fn config() -> &'static Config {
    static CONFIG: OnceLock<&'static Config> = OnceLock::new();
    CONFIG.get_or_init(|| {
        sandbox();
        Box::leak(Box::new(Config {
            llm_model: "gpt-5-mini".to_string(),
            // Hand-built config bypasses EKKO_BACKEND; pin the API backend here
            // too so a locally installed `claude` is not auto-detected.
            llm_backend: Some("api".to_string()),
            query_backend: Some("api".to_string()),
            ..Config::default()
        }))
    })
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

/// A Claude Code transcript with two user turns and one tool call.
const TRANSCRIPT: &str = concat!(
    r#"{"type":"system","subtype":"init","sessionId":"s1","cwd":"/home/user/webapp","gitBranch":"main","timestamp":"2026-01-05T10:00:00Z"}"#,
    "\n",
    r#"{"type":"user","timestamp":"2026-01-05T10:00:05Z","message":{"role":"user","content":"Why is the auth redirect looping?"}}"#,
    "\n",
    r#"{"type":"assistant","timestamp":"2026-01-05T10:00:09Z","message":{"role":"assistant","content":[{"type":"text","text":"Reading the auth module."},{"type":"tool_use","id":"tc_1","name":"Read","input":{"file_path":"/home/user/webapp/auth.py"}}]}}"#,
    "\n",
    r#"{"type":"user","timestamp":"2026-01-05T10:01:00Z","message":{"role":"user","content":"Fix it please"}}"#,
    "\n",
    r#"{"type":"assistant","timestamp":"2026-01-05T10:01:20Z","message":{"role":"assistant","content":[{"type":"text","text":"Fixed: the session cookie was never set on the redirect response."}]}}"#,
    "\n",
);

struct Fixture {
    server: EkkoServer,
    memory: EkkoMemory,
    _dir: TempDir,
}

fn fixture() -> Fixture {
    let config = config();
    let dir = TempDir::new().unwrap();
    let transcript_path: PathBuf = dir.path().join("s1.jsonl");
    fs::write(&transcript_path, TRANSCRIPT).unwrap();

    let db = Database::new(dir.path().join("sessions.db")).unwrap();

    let mut s1 = SessionMetadata::new("s1", "claude_code", transcript_path.to_string_lossy());
    s1.title = Some("Fix auth redirect loop".to_string());
    s1.first_prompt = Some("Why is the auth redirect looping?".to_string());
    s1.project_path = Some("/home/user/webapp".to_string());
    s1.git_branch = Some("main".to_string());
    s1.created_at = Some((chrono::Utc::now() - chrono::Duration::days(1)).fixed_offset());
    s1.modified_at = Some((chrono::Utc::now() - chrono::Duration::days(1)).fixed_offset());
    s1.message_count = 4;
    db.upsert_session(&s1).unwrap();
    db.update_summary(
        "s1",
        "Fix auth redirect loop",
        "Set the session cookie on redirects",
    )
    .unwrap();
    db.add_file_reference("s1", "/home/user/webapp/auth.py", "write")
        .unwrap();

    let mut s2 = SessionMetadata::new("s2", "github_copilot", "");
    s2.title = Some("Refactor DB".to_string());
    s2.project_path = Some("/home/user/backend".to_string());
    s2.created_at = Some((chrono::Utc::now() - chrono::Duration::days(2)).fixed_offset());
    s2.modified_at = Some((chrono::Utc::now() - chrono::Duration::days(2)).fixed_offset());
    s2.message_count = 12;
    db.upsert_session(&s2).unwrap();
    db.update_summary("s2", "Refactor DB", "Split the database module")
        .unwrap();

    Fixture {
        memory: EkkoMemory::new(db.clone()),
        server: EkkoServer::new(db, config),
        _dir: dir,
    }
}

/// The JSON payload a client would read out of a successful tool result.
#[track_caller]
fn payload(result: &CallToolResult) -> Value {
    assert_eq!(
        result.is_error,
        Some(false),
        "tool returned an error result"
    );
    let text = match &result.content[0] {
        ContentBlock::Text(t) => t.text.clone(),
        other => panic!("expected a text content block, got {other:?}"),
    };
    let value: Value = serde_json::from_str(&text).expect("tool text content is JSON");
    // Value equality with preserve_order maps is order-INsensitive, so also
    // pin the exact serialized bytes: the text block must be the pretty
    // printing of the structured value, key order included (P9 review gap).
    assert_eq!(
        text,
        serde_json::to_string_pretty(&value).expect("re-serialize"),
        "text block is not to_string_pretty of its own value"
    );
    value
}

/// Assert the text block is byte-identical to the pretty printing of
/// `expected` — a key-order-sensitive comparison, unlike `assert_eq!` on
/// `Value`s.
#[track_caller]
fn assert_payload_text(result: &CallToolResult, expected: &Value) {
    let text = match &result.content[0] {
        ContentBlock::Text(t) => t.text.as_str(),
        other => panic!("expected a text content block, got {other:?}"),
    };
    assert_eq!(
        text,
        serde_json::to_string_pretty(expected).expect("serialize expected"),
        "text block diverges from expected value (byte comparison)"
    );
}

// ---------------------------------------------------------------------------
// Inventory
// ---------------------------------------------------------------------------

#[test]
fn server_lists_the_nine_tools() {
    let f = fixture();
    let names: Vec<String> = f
        .server
        .tools()
        .into_iter()
        .map(|t| t.name.to_string())
        .collect();
    assert_eq!(
        names,
        [
            "ask_question",
            "export_telemetry",
            "get_session_snapshot",
            "get_status",
            "list_projects",
            "read_transcript",
            "recent_activity",
            "search_sessions",
            "sessions_in_directory",
        ]
    );
    for tool in f.server.tools() {
        assert!(
            tool.description.as_deref().is_some_and(|d| !d.is_empty()),
            "{} has no description",
            tool.name
        );
    }
}

// ---------------------------------------------------------------------------
// FAST tools
// ---------------------------------------------------------------------------

#[tokio::test]
async fn recent_activity_matches_the_memory_layer() {
    let f = fixture();
    let result = f
        .server
        .recent_activity(Parameters(
            serde_json::from_value::<RecentActivityParams>(json!({"days": 30})).unwrap(),
        ))
        .await
        .unwrap();

    let expected = f
        .memory
        .recent_activity(&RecentActivityArgs {
            days: 30,
            ..RecentActivityArgs::default()
        })
        .unwrap();

    assert_eq!(payload(&result), expected);
    assert_payload_text(&result, &expected);
    assert_eq!(result.structured_content.as_ref(), Some(&expected));
    let sessions = expected["sessions"].as_array().unwrap();
    assert_eq!(sessions.len(), 2);
    assert_eq!(sessions[0]["session_id"], json!("s1"));
}

#[tokio::test]
async fn list_projects_matches_the_memory_layer_and_wraps_the_array() {
    let f = fixture();
    let result = f.server.list_projects().await.unwrap();
    let expected = f.memory.list_projects().unwrap();

    assert_eq!(payload(&result), expected);
    assert_payload_text(&result, &expected);
    // MCP requires structuredContent to be an object, so an array result is
    // wrapped as `{"result": …}`.
    assert_eq!(
        result.structured_content,
        Some(json!({ "result": expected }))
    );
    assert_eq!(expected.as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn sessions_in_directory_matches_the_memory_layer() {
    let f = fixture();
    let result = f
        .server
        .sessions_in_directory(Parameters(
            serde_json::from_value::<SessionsInDirectoryParams>(
                json!({"directory": "/home/user/webapp"}),
            )
            .unwrap(),
        ))
        .await
        .unwrap();

    let expected = f
        .memory
        .sessions_in_directory("/home/user/webapp", true, None, 50)
        .unwrap();
    assert_eq!(payload(&result), expected);
    assert_payload_text(&result, &expected);
    assert_eq!(expected.as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn read_transcript_info_matches_the_memory_layer() {
    let f = fixture();
    let result = f
        .server
        .read_transcript(Parameters(
            serde_json::from_value::<ReadTranscriptParams>(json!({"session_id": "s1"})).unwrap(),
        ))
        .await
        .unwrap();

    let expected = f
        .memory
        .read_transcript(&ReadTranscriptArgs::new("s1"))
        .unwrap();

    assert_eq!(payload(&result), expected);
    assert_payload_text(&result, &expected);
    let value = payload(&result);
    assert_eq!(value["session_id"], json!("s1"));
    assert!(value["message_count"].as_i64().unwrap() > 0);
}

#[tokio::test]
async fn get_session_snapshot_reports_a_missing_snapshot_in_band() {
    let f = fixture();
    let result = f
        .server
        .get_session_snapshot(Parameters(
            serde_json::from_value::<GetSessionSnapshotParams>(json!({"session_id": "s1"}))
                .unwrap(),
        ))
        .await
        .unwrap();

    let value = payload(&result);
    assert_eq!(value["has_snapshot"], json!(false));
    assert_eq!(value["metadata"]["title"], json!("Fix auth redirect loop"));

    // A missing session is an in-band `error` key, not a failed call: the
    // tool result is successful and carries the message.
    let result = f
        .server
        .get_session_snapshot(Parameters(
            serde_json::from_value::<GetSessionSnapshotParams>(json!({"session_id": "nope"}))
                .unwrap(),
        ))
        .await
        .unwrap();
    assert_eq!(result.is_error, Some(false));
    assert_eq!(payload(&result)["error"], json!("Session not found: nope"));
}

#[tokio::test]
async fn keyword_search_needs_no_llm() {
    let f = fixture();
    let result = f
        .server
        .search_sessions(Parameters(
            serde_json::from_value::<SearchSessionsParams>(json!({"query": "auth"})).unwrap(),
        ))
        .await
        .unwrap();

    let value = payload(&result);
    let results = value["results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["session_id"], json!("s1"));
    assert!(value.get("mode_fallback").is_none());
}

#[tokio::test]
async fn get_status_reports_the_database_indexer_and_config() {
    let f = fixture();
    let value = payload(&f.server.get_status().await.unwrap());

    assert_eq!(value["version"], json!(env!("CARGO_PKG_VERSION")));
    assert_eq!(value["sessions"]["total"], json!(2));
    assert_eq!(value["sessions"]["with_summary"], json!(2));
    assert_eq!(value["sessions"]["with_snapshot"], json!(0));
    assert_eq!(value["sessions"]["needing_snapshot"], json!(1)); // only s1 has a transcript
    assert_eq!(value["sessions"]["by_source"]["claude_code"], json!(1));

    // Not started until the lifespan starts it.
    assert_eq!(value["indexer"]["status"], json!("not_started"));
    assert_eq!(value["indexer"]["instance_id"], Value::Null);
    assert_eq!(value["indexer"]["startup_delay_seconds"], json!(60));

    assert_eq!(value["config"]["model"], json!("gpt-5-mini"));
    assert_eq!(value["config"]["api_keys_available"], json!([]));
    assert!(
        value["config"]["missing_key_warning"]
            .as_str()
            .unwrap()
            .contains("OPENAI_API_KEY")
    );

    // Costs come from telemetry; nothing priced yet.
    assert_eq!(value["costs"]["this_instance"]["calls"], json!(0));
    assert_eq!(value["costs"]["by_operation_7d"], json!({}));

    // The status flag is shared with the lifespan.
    f.server.indexer_status().mark_running();
    let value = payload(&f.server.get_status().await.unwrap());
    assert_eq!(value["indexer"]["status"], json!("running"));
    assert!(value["indexer"]["instance_id"].is_string());
}

// ---------------------------------------------------------------------------
// LLM gating
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ask_question_returns_the_setup_guidance_without_a_key() {
    let f = fixture();
    let result = f
        .server
        .ask_question(Parameters(
            serde_json::from_value::<AskQuestionParams>(json!({"question": "why?"})).unwrap(),
        ))
        .await
        .unwrap();

    // A successful call carrying an error key, not a protocol error.
    assert_eq!(result.is_error, Some(false));
    let value = payload(&result);
    let error = value["error"].as_str().unwrap();
    assert!(error.starts_with("ask_question requires LLM access."));
    assert!(error.contains("OPENAI_API_KEY"));
    assert_eq!(
        value["suggestion"],
        json!(
            "Use search_sessions(mode=\"keyword\") or search_sessions(mode=\"transcript\") for text-based search (no LLM needed)."
        )
    );
}

#[tokio::test]
async fn semantic_search_is_refused_and_hybrid_falls_back_to_keyword() {
    let f = fixture();

    let result = f
        .server
        .search_sessions(Parameters(
            serde_json::from_value::<SearchSessionsParams>(
                json!({"query": "auth", "mode": "semantic"}),
            )
            .unwrap(),
        ))
        .await
        .unwrap();
    let value = payload(&result);
    assert!(
        value["error"]
            .as_str()
            .unwrap()
            .starts_with("Semantic search requires LLM access.")
    );
    assert_eq!(
        value["suggestion"],
        json!("Use mode=\"keyword\" or mode=\"transcript\" instead (no LLM needed).")
    );

    let result = f
        .server
        .search_sessions(Parameters(
            serde_json::from_value::<SearchSessionsParams>(
                json!({"query": "auth", "mode": "hybrid"}),
            )
            .unwrap(),
        ))
        .await
        .unwrap();
    let value = payload(&result);
    assert_eq!(
        value["mode_fallback"],
        json!({
            "requested": "hybrid",
            "used": "keyword",
            "reason": "LLM not configured, fell back to keyword search"
        })
    );
    assert_eq!(value["results"].as_array().unwrap().len(), 1);
}

// ---------------------------------------------------------------------------
// Telemetry
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tool_calls_record_allowlisted_arguments_only() {
    let f = fixture();

    f.server
        .search_sessions(Parameters(
            serde_json::from_value::<SearchSessionsParams>(
                json!({"query": "top secret query", "mode": "keyword", "limit": 5, "days": 30}),
            )
            .unwrap(),
        ))
        .await
        .unwrap();
    f.server
        .read_transcript(Parameters(
            serde_json::from_value::<ReadTranscriptParams>(
                json!({"session_id": "s1", "mode": "info"}),
            )
            .unwrap(),
        ))
        .await
        .unwrap();
    f.server.list_projects().await.unwrap();

    let events = f.server.db().export_telemetry(None).unwrap();
    let by_op: std::collections::HashMap<&str, &ekko::db::TelemetryEvent> =
        events.iter().map(|e| (e.operation.as_str(), e)).collect();
    assert_eq!(events.len(), 3);

    let search = by_op["search_sessions"];
    assert_eq!(
        search.safe_args,
        Some(json!({"mode": "keyword", "include_subdirs": true, "days": 30, "limit": 5}))
    );
    assert!(search.duration_ms.unwrap() >= 0);
    // The query text and the requested_mode (not on the allowlist) are gone.
    let dumped = serde_json::to_string(&search.safe_args).unwrap();
    assert!(!dumped.contains("top secret"));
    assert!(!dumped.contains("requested_mode"));

    // `mode` is allowlisted for read_transcript, but "info" is not one of
    // the ENUM_VALUES for the `mode` key (those are the *search* modes), so
    // it is dropped — the allowlist is shared across operations.
    assert_eq!(
        by_op["read_transcript"].safe_args,
        Some(json!({"context_lines": 3, "max_matches": 20, "include_tools": true}))
    );
    assert_eq!(by_op["list_projects"].safe_args, None);

    // Every event carries this server's instance id.
    for event in &events {
        assert_eq!(event.instance_id, f.server.telemetry().instance_id());
        assert!(event.cost.is_none(), "no LLM ran, so nothing is priced");
    }
}

#[tokio::test]
async fn export_telemetry_writes_into_the_export_sandbox() {
    let f = fixture();
    f.server.list_projects().await.unwrap();

    let result = f
        .server
        .export_telemetry(Parameters(
            serde_json::from_value::<ExportTelemetryParams>(json!({"filename": "usage"})).unwrap(),
        ))
        .await
        .unwrap();

    let value = payload(&result);
    let path = PathBuf::from(value["file_path"].as_str().unwrap());
    // The platform data dir (XDG share/ekko on Linux, Library/Application
    // Support/ekko on macOS) — but always inside the sandbox fake home.
    // Canonicalize both sides: the export resolver canonicalizes, and on
    // macOS the tempdir sits behind the /var -> /private/var symlink.
    let path = path.canonicalize().unwrap();
    let exports = ekko::paths::get_data_dir()
        .unwrap()
        .join("exports")
        .canonicalize()
        .unwrap();
    assert!(path.starts_with(&exports), "{path:?}");
    assert!(path.starts_with(sandbox().canonicalize().unwrap()), "{path:?}");
    assert_eq!(path.extension().unwrap(), "jsonl");
    assert_eq!(value["event_count"], json!(1));
    assert_eq!(value["summary"]["total_events"], json!(1));

    let body = fs::read_to_string(&path).unwrap();
    assert!(body.lines().next().unwrap().contains("TELEMETRY EXPORT"));

    // Traversal is rejected, and the rejection is a readable error result.
    let result = f
        .server
        .export_telemetry(Parameters(
            serde_json::from_value::<ExportTelemetryParams>(json!({"filename": "../escape"}))
                .unwrap(),
        ))
        .await
        .unwrap();
    assert_eq!(result.is_error, Some(true));
}
