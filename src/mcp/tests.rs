//! Unit tests for the server assembly: the tool inventory, the schemas it
//! advertises, and the result envelope.
//!
//! Tool-call behaviour against a real database lives in
//! `tests/mcp_test.rs`, which can pin the process environment before the
//! config singleton is read.

use super::*;
use rmcp::model::Tool;

/// The nine tools ekko exposes, in `list_all` (sorted) order.
const EXPECTED_TOOLS: [&str; 9] = [
    "ask_question",
    "export_telemetry",
    "get_session_snapshot",
    "get_status",
    "list_projects",
    "read_transcript",
    "recent_activity",
    "search_sessions",
    "sessions_in_directory",
];

fn tools() -> Vec<Tool> {
    EkkoServer::tool_router().list_all()
}

fn tool_names() -> Vec<String> {
    tools().into_iter().map(|t| t.name.to_string()).collect()
}

#[test]
fn exactly_the_nine_kept_tools_are_registered() {
    assert_eq!(tool_names(), EXPECTED_TOOLS);
}

#[test]
fn dropped_tools_are_absent() {
    let names = tool_names();
    for gone in [
        "ask_session",
        "ask_sessions",
        "find_sessions_by_file",
        "get_file_history",
        "index_sessions",
        "submit_index_result",
    ] {
        assert!(
            !names.iter().any(|n| n == gone),
            "{gone} should not be registered"
        );
    }
}

#[test]
fn every_tool_carries_its_ported_docstring() {
    for tool in tools() {
        let expected = descriptions::description(&tool.name)
            .unwrap_or_else(|| panic!("no description table entry for {}", tool.name));
        assert_eq!(
            tool.description.as_deref(),
            Some(expected),
            "{} description drifted from the table",
            tool.name
        );
    }
}

/// A schema containing `type: [T, "null"]` or an integer `format` has been
/// observed to make Claude clients silently drop **every** tool. This test
/// is the guard.
#[test]
fn no_tool_schema_contains_type_arrays_or_numeric_formats() {
    for tool in tools() {
        let schema = Value::Object((*tool.input_schema).clone());
        let problems = schema::forbidden_shapes(&schema);
        assert!(
            problems.is_empty(),
            "{} advertises a client-breaking schema: {problems:?}\n{}",
            tool.name,
            serde_json::to_string_pretty(&schema).unwrap()
        );
        if let Some(output) = &tool.output_schema {
            let schema = Value::Object((**output).clone());
            assert!(
                schema::forbidden_shapes(&schema).is_empty(),
                "{} advertises a client-breaking output schema",
                tool.name
            );
        }
    }
}

/// The sanitizer must not have destroyed the schema in the process: required
/// arguments stay required and optional ones stay absent from `required`.
#[test]
fn schemas_keep_their_required_arguments() {
    let by_name: std::collections::HashMap<String, Tool> = tools()
        .into_iter()
        .map(|t| (t.name.to_string(), t))
        .collect();

    let required = |tool: &str| -> Vec<String> {
        by_name[tool]
            .input_schema
            .get("required")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    };

    assert_eq!(required("search_sessions"), ["query"]);
    assert_eq!(required("get_session_snapshot"), ["session_id"]);
    assert_eq!(required("read_transcript"), ["session_id"]);
    assert_eq!(required("ask_question"), ["question"]);
    assert_eq!(required("sessions_in_directory"), ["directory"]);
    assert_eq!(required("export_telemetry"), ["filename"]);
    assert!(required("recent_activity").is_empty());

    // Optional scalars survive as concrete types.
    let props = by_name["search_sessions"].input_schema["properties"].clone();
    assert_eq!(props["directory"]["type"], json!("string"));
    assert_eq!(props["days"]["type"], json!("integer"));
    assert_eq!(props["limit"]["type"], json!("integer"));
    assert_eq!(props["include_subdirs"]["type"], json!("boolean"));
    assert!(
        props["mode"]["description"]
            .as_str()
            .unwrap()
            .contains("keyword")
    );
}

/// `list_projects` and `get_status` take no arguments; their schema must
/// still be a well-formed object schema.
#[test]
fn no_argument_tools_have_empty_object_schemas() {
    for name in ["list_projects", "get_status"] {
        let tool = tools().into_iter().find(|t| t.name == name).unwrap();
        assert_eq!(tool.input_schema.get("type"), Some(&json!("object")));
        let props = tool.input_schema.get("properties");
        assert!(props.is_none_or(|p| p.as_object().is_some_and(|o| o.is_empty())));
    }
}

#[test]
fn server_info_names_ekko_and_carries_the_instructions() {
    let config: &'static Config = Box::leak(Box::new(Config::default()));
    let dir = tempfile::TempDir::new().unwrap();
    let db = Database::new(dir.path().join("sessions.db")).unwrap();
    let server = EkkoServer::new(db, config);

    let info = server.get_info();
    assert_eq!(info.server_info.name, "ekko");
    assert_eq!(info.server_info.version, VERSION);
    let instructions = info.instructions.unwrap();
    assert!(instructions.starts_with(&format!(
        "ekko v{VERSION} - Archive of past coding sessions"
    )));
    assert!(!instructions.contains("AGENT-ASSISTED INDEXING"));
    assert!(info.capabilities.tools.is_some());
}

// ── result envelope ───────────────────────────────────────────────────

#[test]
fn object_results_keep_key_order_in_text_and_structured_content() {
    let value = json!({"sessions": [], "files_touched": [], "summary": "none"});
    let result = ok(&value);
    let text = match &result.content[0] {
        ContentBlock::Text(t) => t.text.clone(),
        other => panic!("expected text content, got {other:?}"),
    };
    // Pretty-printed, in insertion order (serde_json `preserve_order`).
    assert!(text.starts_with("{\n  \"sessions\": []"));
    assert_eq!(serde_json::from_str::<Value>(&text).unwrap(), value);
    assert_eq!(result.structured_content, Some(value));
    assert_eq!(result.is_error, Some(false));
}

#[test]
fn array_results_are_wrapped_for_structured_content_only() {
    let value = json!([{"project_path": "/repo"}]);
    let result = ok(&value);
    let text = match &result.content[0] {
        ContentBlock::Text(t) => t.text.clone(),
        other => panic!("expected text content, got {other:?}"),
    };
    // The text block carries the raw list; only structuredContent is wrapped.
    assert_eq!(serde_json::from_str::<Value>(&text).unwrap(), value);
    assert_eq!(result.structured_content, Some(json!({"result": value})));
}

#[test]
fn failures_come_back_as_readable_error_results() {
    let result = failed("Session not found: abc");
    assert_eq!(result.is_error, Some(true));
    match &result.content[0] {
        ContentBlock::Text(t) => assert_eq!(t.text, "Session not found: abc"),
        other => panic!("expected text content, got {other:?}"),
    }
}

// ── telemetry usage extraction ────────────────────────────────────────

#[test]
fn usage_extraction_is_pinned() {
    // Pin the API backend: with no key in this config, auto-detection would
    // pick a CLI backend on machines that have one installed.
    let config = Config {
        llm_model: "gpt-5-mini".to_string(),
        llm_backend: Some("api".to_string()),
        ..Config::default()
    };

    // No usage key, empty usage, null usage → None.
    assert!(extract_usage_for_telemetry(&json!({"answer": "hi"}), &config).is_none());
    assert!(extract_usage_for_telemetry(&json!({"usage": {}}), &config).is_none());
    assert!(extract_usage_for_telemetry(&json!({"usage": null}), &config).is_none());

    // ekko's own naming (UsageStats::to_json).
    let usage = extract_usage_for_telemetry(
        &json!({"usage": {"input_tokens": 1000, "output_tokens": 200, "reasoning_tokens": 50}}),
        &config,
    )
    .unwrap();
    assert_eq!(usage.model.as_deref(), Some("gpt-5-mini"));
    assert_eq!(usage.input_tokens, Some(1000));
    assert_eq!(usage.output_tokens, Some(200));
    assert_eq!(usage.thinking_tokens, Some(50));
    assert_eq!(
        usage.cost,
        Some(crate::pricing::get_cost("gpt-5-mini", 1000, 200, 50))
    );

    // The prompt/completion naming is accepted too.
    let usage = extract_usage_for_telemetry(
        &json!({"usage": {"prompt_tokens": 7, "completion_tokens": 3, "thinking_tokens": 1}}),
        &config,
    )
    .unwrap();
    assert_eq!(usage.input_tokens, Some(7));
    assert_eq!(usage.output_tokens, Some(3));
    assert_eq!(usage.thinking_tokens, Some(1));
}

/// Documents *why* the sanitizer exists: schemars' raw output for these
/// parameter structs does contain the client-breaking shapes.
#[test]
fn raw_schemars_output_would_break_clients() {
    let raw = EkkoServer::tool_router_internal();
    let mut problems = Vec::new();
    for route in raw.map.values() {
        let schema = Value::Object((*route.attr.input_schema).clone());
        problems.extend(schema::forbidden_shapes(&schema));
    }
    assert!(
        !problems.is_empty(),
        "schemars no longer emits the bad shapes — the sanitizer may be removable"
    );
    eprintln!("raw schemas contained {} forbidden shapes:", problems.len());
    for p in &problems {
        eprintln!("  {p}");
    }
}
