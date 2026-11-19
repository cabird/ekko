//! Tests for the memory layer.
//!
//! Behavioural coverage spans session filtering, markdown-fence stripping,
//! session/result formatting, summary lines, `recent_activity`,
//! `search_sessions`, `get_session_snapshot`, `read_transcript` (including
//! the FTS-cached fallback), `sessions_in_directory` and `list_projects`.
//!
//! `read_transcript`'s output is additionally pinned byte-for-byte against
//! `tests/goldens/read_transcript.json`, rendered from the fixture transcript
//! in `tests/goldens/rt_transcript.jsonl`. Those are expected bytes: update
//! them deliberately when the output shape changes on purpose.
//!
//! The Q&A pipeline is tested through its pure pieces (prompt packing,
//! response interpretation, reduction), following `summarize.rs`'s pattern —
//! the LLM boundary itself is covered by `llm/tests.rs`.

use std::collections::HashSet;

use chrono::{DateTime, FixedOffset, Utc};
use serde_json::{Value, json};
use tempfile::TempDir;

use super::qa::{self, SummaryOutcome, TranscriptAnswer};
use super::transcript::{self, ReadTranscriptArgs};
use super::*;
use crate::db::Database;
use crate::models::SessionMetadata;
use crate::paths::testenv::{EnvVarGuard, env_lock};

const GOLDENS: &str = include_str!("../../tests/goldens/read_transcript.json");
const FIXTURE_TRANSCRIPT: &str = include_str!("../../tests/goldens/rt_transcript.jsonl");

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

struct TestMem {
    mem: EkkoMemory,
    _dir: TempDir,
}

impl std::ops::Deref for TestMem {
    type Target = EkkoMemory;
    fn deref(&self) -> &EkkoMemory {
        &self.mem
    }
}

fn days_ago(days: i64) -> DateTime<FixedOffset> {
    (Utc::now() - chrono::Duration::days(days)).fixed_offset()
}

fn session(
    sid: &str,
    title: &str,
    project: &str,
    cli: &str,
    days: i64,
    msgs: i64,
) -> SessionMetadata {
    let mut m = SessionMetadata::new(sid, cli, format!("/data/{sid}.jsonl"));
    m.title = Some(title.to_string());
    m.first_prompt = Some(format!("Prompt for {sid}"));
    m.project_path = Some(project.to_string());
    m.created_at = Some(days_ago(days));
    m.modified_at = Some(days_ago(days));
    m.message_count = msgs;
    m
}

fn mem_db() -> TestMem {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::new(dir.path().join("mem.db")).expect("db");

    for s in [
        session(
            "m1",
            "Fix auth bug",
            "/home/user/webapp",
            "claude_code",
            1,
            10,
        ),
        session(
            "m2",
            "Add dashboard",
            "/home/user/webapp",
            "claude_code",
            3,
            25,
        ),
        session(
            "m3",
            "Refactor DB",
            "/home/user/backend",
            "github_copilot",
            5,
            15,
        ),
        session(
            "m4",
            "Write tests",
            "/home/user/webapp/api",
            "claude_code",
            10,
            8,
        ),
        session("m5", "Setup CI", "/home/user/infra", "claude_code", 30, 12),
        session(
            "m6",
            "Empty session",
            "/home/user/webapp",
            "claude_code",
            2,
            0,
        ),
    ] {
        db.upsert_session(&s).expect("upsert");
    }

    db.update_summary(
        "m1",
        "Fix auth bug",
        "Fixed redirect loop in authentication flow",
    )
    .expect("summary");
    db.update_summary("m2", "Add dashboard", "Created user dashboard with charts")
        .expect("summary");
    db.update_summary(
        "m3",
        "Refactor DB",
        "Refactored database layer for performance",
    )
    .expect("summary");

    for (sid, path, op) in [
        ("m1", "/home/user/webapp/auth.py", "write"),
        ("m1", "/home/user/webapp/tests/test_auth.py", "write"),
        ("m2", "/home/user/webapp/dashboard.py", "write"),
        ("m2", "/home/user/webapp/auth.py", "read"),
    ] {
        db.add_file_reference(sid, path, op).expect("file ref");
    }

    TestMem {
        mem: EkkoMemory::new(db),
        _dir: dir,
    }
}

fn all_sessions(mem: &EkkoMemory) -> Vec<crate::db::SessionRow> {
    mem.db()
        .get_sessions(None, None, None, 0)
        .expect("sessions")
}

fn ids(sessions: &[crate::db::SessionRow]) -> HashSet<String> {
    sessions.iter().map(|s| s.session_id.clone()).collect()
}

fn result_ids(v: &Value) -> HashSet<String> {
    v["results"]
        .as_array()
        .expect("results array")
        .iter()
        .map(|r| r["session_id"].as_str().expect("session_id").to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// _filter_sessions
// ---------------------------------------------------------------------------

#[test]
fn filters_zero_message_sessions() {
    let m = mem_db();
    let got = m.filter_sessions(all_sessions(&m), &SessionFilter::new());
    assert!(!ids(&got).contains("m6"));
}

#[test]
fn directory_filter_with_subdirs() {
    let m = mem_db();
    let got = m.filter_sessions(
        all_sessions(&m),
        &SessionFilter::new().directory(Some("/home/user/webapp"), true),
    );
    let got = ids(&got);
    assert!(got.contains("m1"));
    assert!(got.contains("m4"), "subdirectory session must be included");
    assert!(!got.contains("m3"));
}

#[test]
fn directory_filter_exact_match() {
    let m = mem_db();
    let got = m.filter_sessions(
        all_sessions(&m),
        &SessionFilter::new().directory(Some("/home/user/webapp"), false),
    );
    let got = ids(&got);
    assert!(got.contains("m1"));
    assert!(!got.contains("m4"));
}

#[test]
fn days_filter() {
    let m = mem_db();
    let got = ids(&m.filter_sessions(all_sessions(&m), &SessionFilter::new().days(Some(7))));
    assert!(got.contains("m1") && got.contains("m2") && got.contains("m3"));
    assert!(!got.contains("m4") && !got.contains("m5"));
}

#[test]
fn cli_source_filter() {
    let m = mem_db();
    let mut f = SessionFilter::new();
    f.cli_source = Some("github_copilot");
    let got = ids(&m.filter_sessions(all_sessions(&m), &f));
    assert_eq!(got, HashSet::from(["m3".to_string()]));
}

#[test]
fn combined_filters() {
    let m = mem_db();
    let mut f = SessionFilter::new()
        .directory(Some("/home/user/webapp"), true)
        .days(Some(7));
    f.cli_source = Some("claude_code");
    let got = ids(&m.filter_sessions(all_sessions(&m), &f));
    assert!(got.contains("m1") && got.contains("m2"));
    assert!(!got.contains("m3") && !got.contains("m4"));
}

#[test]
fn min_messages_filter() {
    let m = mem_db();
    let mut f = SessionFilter::new();
    f.min_messages = 15;
    let got = ids(&m.filter_sessions(all_sessions(&m), &f));
    assert!(got.contains("m2") && got.contains("m3"));
    assert!(!got.contains("m1"));
}

#[test]
fn date_range_filter() {
    let m = mem_db();
    let start = (Utc::now() - chrono::Duration::days(6))
        .format("%Y-%m-%d")
        .to_string();
    let end = (Utc::now() - chrono::Duration::days(2))
        .format("%Y-%m-%d")
        .to_string();
    let got = ids(&m.filter_sessions(
        all_sessions(&m),
        &SessionFilter::new().dates(Some(&start), Some(&end)),
    ));
    assert!(got.contains("m2") && got.contains("m3"));
    assert!(!got.contains("m1"), "1 day old is after the end date");
}

// ---------------------------------------------------------------------------
// _strip_markdown_fences
// ---------------------------------------------------------------------------

#[test]
fn strips_markdown_fences() {
    assert_eq!(
        strip_markdown_fences(Some("```markdown\n# Hello\nWorld\n```")).as_deref(),
        Some("# Hello\nWorld")
    );
    assert_eq!(
        strip_markdown_fences(Some("```\ncode here\n```")).as_deref(),
        Some("code here")
    );
    assert_eq!(
        strip_markdown_fences(Some("Just plain text")).as_deref(),
        Some("Just plain text")
    );
    assert_eq!(strip_markdown_fences(Some("")).as_deref(), Some(""));
    assert_eq!(strip_markdown_fences(None), None);
}

#[test]
fn strips_opening_fence_without_closing_fence() {
    assert_eq!(
        strip_markdown_fences(Some("```json\n{\"a\": 1}")).as_deref(),
        Some("{\"a\": 1}")
    );
}

// ---------------------------------------------------------------------------
// _format_session_result
// ---------------------------------------------------------------------------

fn row(id: &str) -> crate::db::SessionRow {
    crate::db::SessionRow {
        session_id: id.to_string(),
        cli_source: "claude_code".to_string(),
        ..Default::default()
    }
}

#[test]
fn format_session_result_basic_fields() {
    let s = crate::db::SessionRow {
        title: Some("Test Session".into()),
        summary: Some("A summary".into()),
        first_prompt: Some("Hello".into()),
        project_path: Some("/home/user/proj".into()),
        git_branch: Some("main".into()),
        modified_at: Some("2025-01-15T10:00:00Z".into()),
        message_count: 5,
        transcript_path: Some("/data/x1.jsonl".into()),
        ..row("x1")
    };
    let r = format_session_result(&s, &ResultExtras::default());
    assert_eq!(r["session_id"], json!("x1"));
    assert_eq!(r["title"], json!("Test Session"));
    assert_eq!(r["summary"], json!("A summary"));
    assert_eq!(r["project_path"], json!("/home/user/proj"));
    assert_eq!(r["score"], Value::Null);
    assert_eq!(r["relevance"], Value::Null);
    assert!(r.get("excerpt").is_none(), "no excerpt key when absent");
}

#[test]
fn format_session_result_truncates_summary_to_300() {
    let s = crate::db::SessionRow {
        summary: Some("a".repeat(500)),
        first_prompt: Some("hi".into()),
        ..row("x1")
    };
    let r = format_session_result(&s, &ResultExtras::default());
    assert_eq!(r["summary"].as_str().expect("summary").len(), 300);
}

#[test]
fn format_session_result_falls_back_to_first_prompt() {
    let s = crate::db::SessionRow {
        title: None,
        first_prompt: Some("Fix the bug please".into()),
        ..row("x1")
    };
    let r = format_session_result(&s, &ResultExtras::default());
    assert_eq!(r["title"], json!("Fix the bug please"));
}

#[test]
fn format_session_result_includes_excerpt_and_score() {
    let s = crate::db::SessionRow {
        first_prompt: Some(String::new()),
        ..row("x1")
    };
    let r = format_session_result(
        &s,
        &ResultExtras {
            score: Some(json!(87)),
            relevance: Some("about auth".into()),
            excerpt: Some("...matching text...".into()),
        },
    );
    assert_eq!(r["excerpt"], json!("...matching text..."));
    assert_eq!(r["score"], json!(87));
    assert_eq!(r["relevance"], json!("about auth"));
}

// ---------------------------------------------------------------------------
// _session_to_summary_line
// ---------------------------------------------------------------------------

fn summary_line_session() -> crate::db::SessionRow {
    crate::db::SessionRow {
        project_path: Some("/home/user/webapp".into()),
        title: Some("Fix auth bug".into()),
        summary: Some("Detailed summary here".into()),
        modified_at: Some("2025-01-15T10:00:00Z".into()),
        message_count: 10,
        git_branch: Some("main".into()),
        ..row("x1")
    }
}

#[test]
fn summary_line_exact_bytes() {
    let line = session_to_summary_line(&summary_line_session(), "S_AB12", false);
    assert_eq!(
        line,
        "[[S_AB12]] webapp | Fix auth bug\n\
         dir:/home/user/webapp | cli:claude_code | branch:main | msgs:10 | modified:2025-01-15"
    );
}

#[test]
fn summary_line_includes_full_summary_when_requested() {
    let line = session_to_summary_line(&summary_line_session(), "S_XY99", true);
    assert!(line.ends_with("\nDetailed summary here"));
}

#[test]
fn summary_line_untitled_and_unknown_project() {
    let s = crate::db::SessionRow {
        message_count: 3,
        ..row("x1")
    };
    let line = session_to_summary_line(&s, "S_ZZ00", true);
    assert_eq!(
        line,
        "[[S_ZZ00]] unknown | Untitled\ncli:claude_code | msgs:3"
    );
}

// ---------------------------------------------------------------------------
// recent_activity
// ---------------------------------------------------------------------------

#[test]
fn recent_activity_returns_sessions_with_summaries() {
    let m = mem_db();
    let r = m
        .recent_activity(&RecentActivityArgs::default())
        .expect("ok");
    let got: HashSet<&str> = r["sessions"]
        .as_array()
        .expect("sessions")
        .iter()
        .map(|s| s["session_id"].as_str().expect("id"))
        .collect();
    assert!(got.contains("m1") && got.contains("m2"));
    assert!(!got.contains("m4"), "10 days old is outside the window");
    assert_eq!(r["summary"], Value::Null);
}

#[test]
fn recent_activity_returns_files_touched_with_counts() {
    let m = mem_db();
    let r = m
        .recent_activity(&RecentActivityArgs::default())
        .expect("ok");
    let files = r["files_touched"].as_array().expect("files");
    let auth = files
        .iter()
        .find(|f| f["path"] == json!("/home/user/webapp/auth.py"))
        .expect("auth.py present");
    // Written by m1, read by m2.
    assert_eq!(auth["count"], json!(2));
    let ops: HashSet<&str> = auth["operations"]
        .as_array()
        .expect("operations")
        .iter()
        .map(|o| o.as_str().expect("op"))
        .collect();
    assert_eq!(ops, HashSet::from(["write", "read"]));
    // Most-touched file sorts first.
    assert_eq!(files[0]["path"], json!("/home/user/webapp/auth.py"));
}

#[test]
fn recent_activity_directory_scoping() {
    let m = mem_db();
    let r = m
        .recent_activity(&RecentActivityArgs {
            directory: Some("/home/user/backend"),
            days: 30,
            ..RecentActivityArgs::default()
        })
        .expect("ok");
    let got: HashSet<&str> = r["sessions"]
        .as_array()
        .expect("sessions")
        .iter()
        .map(|s| s["session_id"].as_str().expect("id"))
        .collect();
    assert_eq!(got, HashSet::from(["m3"]));
}

#[test]
fn recent_activity_empty_result_message() {
    let m = mem_db();
    let r = m
        .recent_activity(&RecentActivityArgs {
            directory: Some("/nonexistent/path"),
            ..RecentActivityArgs::default()
        })
        .expect("ok");
    assert_eq!(r["sessions"], json!([]));
    assert_eq!(r["files_touched"], json!([]));
    assert_eq!(
        r["summary"],
        json!("No sessions found in the last 7 days under /nonexistent/path")
    );
}

#[test]
fn recent_activity_empty_result_message_without_directory() {
    let m = mem_db();
    let r = m
        .recent_activity(&RecentActivityArgs {
            days: 0,
            ..RecentActivityArgs::default()
        })
        .expect("ok");
    assert_eq!(r["summary"], json!("No sessions found in the last 0 days"));
}

#[test]
fn recent_activity_respects_max_sessions() {
    let m = mem_db();
    let r = m
        .recent_activity(&RecentActivityArgs {
            days: 365,
            max_sessions: 1,
            ..RecentActivityArgs::default()
        })
        .expect("ok");
    assert_eq!(r["sessions"].as_array().expect("sessions").len(), 1);
}

/// Sessions come back newest first, and `max_sessions` keeps the newest ones,
/// whatever order the rows were stored in. The rows here are inserted oldest
/// first so that SQLite's storage order disagrees with recency (the `mem_db`
/// fixture inserts newest first, which hid this).
#[test]
fn recent_activity_returns_newest_first_regardless_of_insert_order() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::new(dir.path().join("order.db")).expect("db");
    for (sid, days) in [("old", 20), ("mid", 10), ("new", 1), ("newer", 0)] {
        db.upsert_session(&session(sid, sid, "/home/user/webapp", "claude_code", days, 5))
            .expect("upsert");
        db.update_summary(sid, sid, &format!("Summary of {sid}"))
            .expect("summary");
    }
    let m = TestMem {
        mem: EkkoMemory::new(db),
        _dir: dir,
    };

    let ids = |r: &Value| -> Vec<String> {
        r["sessions"]
            .as_array()
            .expect("sessions")
            .iter()
            .map(|s| s["session_id"].as_str().expect("id").to_string())
            .collect()
    };

    let all = m
        .recent_activity(&RecentActivityArgs {
            days: 30,
            ..RecentActivityArgs::default()
        })
        .expect("ok");
    assert_eq!(ids(&all), ["newer", "new", "mid", "old"]);

    let top2 = m
        .recent_activity(&RecentActivityArgs {
            days: 30,
            max_sessions: 2,
            ..RecentActivityArgs::default()
        })
        .expect("ok");
    assert_eq!(ids(&top2), ["newer", "new"]);
}

// ---------------------------------------------------------------------------
// search_sessions
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_keyword_mode() {
    let m = mem_db();
    let r = m
        .search_sessions(&SearchArgs::new("auth"))
        .await
        .expect("ok");
    assert!(result_ids(&r).contains("m1"));
    assert!(r["usage"].is_object());
    let first = &r["results"][0];
    for key in ["session_id", "title", "summary", "project_path", "score"] {
        assert!(first.get(key).is_some(), "missing key {key}");
    }
}

#[tokio::test]
async fn search_transcript_mode_carries_excerpt() {
    let m = mem_db();
    m.db()
        .index_transcript("m4", "The rare_parrot function handles edge cases")
        .expect("index");
    let mut args = SearchArgs::new("rare_parrot");
    args.mode = "transcript";
    let r = m.search_sessions(&args).await.expect("ok");
    assert!(result_ids(&r).contains("m4"));
    assert!(
        r["results"][0]["excerpt"]
            .as_str()
            .expect("excerpt")
            .contains("rare_parrot")
    );
}

#[tokio::test]
async fn search_directory_filter() {
    let m = mem_db();
    let mut args = SearchArgs::new("auth");
    args.directory = Some("/home/user/webapp");
    let r = m.search_sessions(&args).await.expect("ok");
    for result in r["results"].as_array().expect("results") {
        assert!(
            result["project_path"]
                .as_str()
                .expect("project_path")
                .starts_with("/home/user/webapp")
        );
    }
}

#[tokio::test]
async fn search_days_filter() {
    let m = mem_db();
    let mut args = SearchArgs::new("Setup");
    args.days = Some(7);
    let r = m.search_sessions(&args).await.expect("ok");
    assert!(!result_ids(&r).contains("m5"), "30 days old is excluded");
}

#[tokio::test]
async fn search_no_results_suggests_similar_paths() {
    let m = mem_db();
    let mut args = SearchArgs::new("xyzzy_no_such_token");
    args.directory = Some("/home/user/webap");
    let r = m.search_sessions(&args).await.expect("ok");
    assert_eq!(r["results"], json!([]));
    let suggestions = &r["suggestions"];
    assert!(
        !suggestions["similar_paths"]
            .as_array()
            .expect("similar_paths")
            .is_empty()
    );
    assert_eq!(
        suggestions["hint"],
        json!("No sessions found in '/home/user/webap'. Did you mean one of these?")
    );
}

#[tokio::test]
async fn search_no_suggestions_without_directory() {
    let m = mem_db();
    let r = m
        .search_sessions(&SearchArgs::new("xyzzy_no_such_token"))
        .await
        .expect("ok");
    assert_eq!(r["results"], json!([]));
    assert!(r.get("suggestions").is_none());
}

#[test]
fn semantic_response_maps_tags_back_to_sessions() {
    let sessions = vec![row("a"), row("b")];
    let (tagged, _) = build_tagged_summaries(&sessions);
    let tag_a = tagged[0].0.clone();
    let data = json!({"results": [
        {"tag": tag_a, "score": 91, "reason": "mentions auth"},
        {"tag": "S_ZZ99", "score": 50, "reason": "unknown tag is dropped"},
        {"score": 10},
    ]});
    let hits = search::semantic_hits_from_response(&data, &tagged);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0.session_id, "a");
    assert_eq!(hits[0].1.score, Some(json!(91)));
    assert_eq!(hits[0].1.relevance.as_deref(), Some("mentions auth"));
}

#[test]
fn semantic_response_defaults_score_and_reason() {
    let sessions = vec![row("a")];
    let (tagged, _) = build_tagged_summaries(&sessions);
    let data = json!({"results": [{"tag": tagged[0].0.clone()}]});
    let hits = search::semantic_hits_from_response(&data, &tagged);
    assert_eq!(hits[0].1.score, Some(json!(0)));
    assert_eq!(hits[0].1.relevance.as_deref(), Some(""));
}

// ---------------------------------------------------------------------------
// get_session_snapshot
// ---------------------------------------------------------------------------

#[test]
fn snapshot_returns_content_and_metadata() {
    let m = mem_db();
    m.db()
        .update_snapshot("m1", "## Key Decisions\n- Used Redis for caching")
        .expect("snapshot");
    let r = m.get_session_snapshot("m1").expect("ok");
    assert_eq!(r["has_snapshot"], json!(true));
    assert!(r["snapshot"].as_str().expect("snapshot").contains("Redis"));
    assert_eq!(r["metadata"]["title"], json!("Fix auth bug"));
    assert_eq!(r["suggestion"], Value::Null);
}

#[test]
fn snapshot_absent_gives_suggestion() {
    let m = mem_db();
    let r = m.get_session_snapshot("m2").expect("ok");
    assert_eq!(r["has_snapshot"], json!(false));
    assert_eq!(
        r["suggestion"],
        json!(
            "No snapshot available. Use read_transcript() for full detail, \
or generate a snapshot via the web UI."
        )
    );
}

#[test]
fn snapshot_missing_session() {
    let m = mem_db();
    let r = m.get_session_snapshot("nonexistent").expect("ok");
    assert_eq!(r["error"], json!("Session not found: nonexistent"));
}

#[test]
fn snapshot_strips_markdown_fences() {
    let m = mem_db();
    m.db()
        .update_snapshot("m2", "```markdown\n## Decisions\n- Chose React\n```")
        .expect("snapshot");
    let r = m.get_session_snapshot("m2").expect("ok");
    let snapshot = r["snapshot"].as_str().expect("snapshot");
    assert!(!snapshot.starts_with("```"));
    assert!(snapshot.starts_with("## Decisions"));
}

#[test]
fn snapshot_includes_files_touched() {
    let m = mem_db();
    let r = m.get_session_snapshot("m1").expect("ok");
    let paths: HashSet<&str> = r["files_touched"]
        .as_array()
        .expect("files")
        .iter()
        .map(|f| f["path"].as_str().expect("path"))
        .collect();
    assert!(paths.contains("/home/user/webapp/auth.py"));
}

// ---------------------------------------------------------------------------
// read_transcript — pinned goldens
// ---------------------------------------------------------------------------

struct GoldenMem {
    mem: EkkoMemory,
    goldens: Value,
    _dir: TempDir,
}

/// Keys the goldens intentionally omit (absolute paths / row timestamps).
const GOLDEN_DROP: [&str; 7] = [
    "session_id",
    "metadata",
    "transcript_path",
    "files_touched",
    "saved_to",
    "created_at",
    "modified_at",
];

fn golden_mem() -> GoldenMem {
    let dir = tempfile::tempdir().expect("tempdir");
    let transcript = dir.path().join("rt_transcript.jsonl");
    std::fs::write(&transcript, FIXTURE_TRANSCRIPT).expect("write fixture");

    let db = Database::new(dir.path().join("golden.db")).expect("db");
    let mk = |sid: &str, path: &str| {
        let mut s = SessionMetadata::new(sid, "claude_code", path);
        s.title = Some(format!("Session {sid}"));
        s.first_prompt = Some(format!("Prompt for {sid}"));
        s.project_path = Some("/home/user/project".to_string());
        s.created_at = Some(days_ago(0));
        s.modified_at = Some(days_ago(0));
        s.message_count = 10;
        db.upsert_session(&s).expect("upsert");
    };
    mk("rt1", transcript.to_str().expect("utf-8 path"));
    mk("cached_simple", "/nonexistent/gone.jsonl");
    mk("cached_tools", "/nonexistent/gone2.jsonl");
    mk("nocache", "/gone.jsonl");

    let goldens: Value = serde_json::from_str(GOLDENS).expect("goldens parse");
    let inputs = &goldens["_inputs"];
    db.index_transcript(
        "cached_simple",
        inputs["cached_simple"].as_str().expect("cached_simple"),
    )
    .expect("index");
    db.index_transcript(
        "cached_tools",
        inputs["cached_tools"].as_str().expect("cached_tools"),
    )
    .expect("index");

    GoldenMem {
        mem: EkkoMemory::new(db),
        goldens,
        _dir: dir,
    }
}

impl GoldenMem {
    fn run(&self, args: &ReadTranscriptArgs<'_>) -> Value {
        let mut v = self.mem.read_transcript(args).expect("read_transcript");
        let obj = v.as_object_mut().expect("object");
        for key in GOLDEN_DROP {
            // With preserve_order, `remove` is a swap_remove that reorders
            // the map; shift_remove keeps the remaining keys in order.
            obj.shift_remove(key);
        }
        v
    }

    fn check(&self, name: &str, args: &ReadTranscriptArgs<'_>) {
        let expected = self.goldens.get(name).unwrap_or_else(|| {
            panic!("no golden named {name}");
        });
        let actual = self.run(args);
        assert_eq!(&actual, expected, "golden `{name}` diverged");
        // Key order is part of the serialization contract. Value equality
        // above is order-insensitive with preserve_order maps, so pin the
        // top-level order explicitly against the golden file's own order.
        if let (Some(a), Some(e)) = (actual.as_object(), expected.as_object()) {
            let actual_keys: Vec<&str> = a.keys().map(String::as_str).collect();
            let expected_keys: Vec<&str> = e.keys().map(String::as_str).collect();
            assert_eq!(
                actual_keys, expected_keys,
                "golden `{name}` key order diverged"
            );
        }
    }
}

#[test]
fn read_transcript_goldens_from_file() {
    let g = golden_mem();
    let rt = |mode: &'static str| ReadTranscriptArgs::new("rt1").mode(mode);

    g.check("file_info", &rt("info"));
    g.check(
        "file_head_2",
        &ReadTranscriptArgs {
            count: 2,
            ..rt("head")
        },
    );
    g.check(
        "file_head_10_no_tools",
        &ReadTranscriptArgs {
            count: 10,
            include_tools: false,
            ..rt("head")
        },
    );
    g.check(
        "file_head_10_tools",
        &ReadTranscriptArgs {
            count: 10,
            ..rt("head")
        },
    );
    g.check(
        "file_tail_2",
        &ReadTranscriptArgs {
            count: 2,
            ..rt("tail")
        },
    );
    g.check(
        "file_tail_100",
        &ReadTranscriptArgs {
            count: 100,
            ..rt("tail")
        },
    );
    g.check(
        "file_lines_1_5",
        &ReadTranscriptArgs {
            start: Some(1),
            end: Some(5),
            ..rt("lines")
        },
    );
    g.check(
        "file_lines_0_5",
        &ReadTranscriptArgs {
            start: Some(0),
            end: Some(5),
            ..rt("lines")
        },
    );
    g.check(
        "file_lines_1_big",
        &ReadTranscriptArgs {
            start: Some(1),
            end: Some(999_999),
            ..rt("lines")
        },
    );
    g.check("file_lines_none", &rt("lines"));
    g.check(
        "file_lines_empty_range",
        &ReadTranscriptArgs {
            start: Some(4),
            end: Some(2),
            ..rt("lines")
        },
    );
    g.check(
        "file_grep_help",
        &ReadTranscriptArgs {
            pattern: Some("help"),
            ..rt("grep")
        },
    );
    g.check(
        "file_grep_ci",
        &ReadTranscriptArgs {
            pattern: Some("HELLO"),
            ..rt("grep")
        },
    );
    g.check(
        "file_grep_ctx0",
        &ReadTranscriptArgs {
            pattern: Some("Read"),
            context_lines: 0,
            ..rt("grep")
        },
    );
    g.check(
        "file_grep_truncated",
        &ReadTranscriptArgs {
            pattern: Some("e"),
            max_matches: 2,
            ..rt("grep")
        },
    );
    g.check(
        "file_grep_none",
        &ReadTranscriptArgs {
            pattern: Some("definitely_not_present_xyz"),
            ..rt("grep")
        },
    );
    g.check("file_grep_no_pattern", &rt("grep"));
    g.check("file_full", &rt("full"));
    g.check(
        "file_full_no_tools",
        &ReadTranscriptArgs {
            include_tools: false,
            ..rt("full")
        },
    );
    g.check("file_save_no_path", &rt("save"));
    g.check("bad_mode", &rt("bogus"));
    g.check(
        "missing_session",
        &ReadTranscriptArgs::new("does-not-exist").mode("info"),
    );
}

#[test]
fn read_transcript_goldens_from_cache() {
    let g = golden_mem();
    let simple = |mode: &'static str| ReadTranscriptArgs::new("cached_simple").mode(mode);
    let tools = |mode: &'static str| ReadTranscriptArgs::new("cached_tools").mode(mode);

    g.check("cached_info", &simple("info"));
    g.check(
        "cached_head_2",
        &ReadTranscriptArgs {
            count: 2,
            ..simple("head")
        },
    );
    g.check(
        "cached_tail_1",
        &ReadTranscriptArgs {
            count: 1,
            ..simple("tail")
        },
    );
    g.check(
        "cached_lines_1_3",
        &ReadTranscriptArgs {
            start: Some(1),
            end: Some(3),
            ..simple("lines")
        },
    );
    g.check(
        "cached_grep",
        &ReadTranscriptArgs {
            pattern: Some("redirect"),
            ..simple("grep")
        },
    );
    g.check("cached_full", &simple("full"));
    g.check("cached_tools_info", &tools("info"));
    g.check(
        "cached_tools_head_4",
        &ReadTranscriptArgs {
            count: 4,
            ..tools("head")
        },
    );
    g.check(
        "cached_tools_head_no_tools",
        &ReadTranscriptArgs {
            count: 4,
            include_tools: false,
            ..tools("head")
        },
    );
    g.check(
        "cached_tools_tail_2",
        &ReadTranscriptArgs {
            count: 2,
            ..tools("tail")
        },
    );
    g.check(
        "no_file_no_cache",
        &ReadTranscriptArgs {
            pattern: Some("x"),
            ..ReadTranscriptArgs::new("nocache").mode("grep")
        },
    );
}

#[test]
fn read_transcript_save_goldens() {
    let _lock = env_lock();
    let home = tempfile::tempdir().expect("tempdir");
    let _guard = EnvVarGuard::set("XDG_DATA_HOME", home.path().to_str().expect("utf-8"));

    let g = golden_mem();
    for (name, sid, output) in [
        ("file_save", "rt1", "ekko-golden-save.txt"),
        ("cached_save", "cached_simple", "ekko-golden-cached.txt"),
    ] {
        let args = ReadTranscriptArgs {
            output_path: Some(output),
            ..ReadTranscriptArgs::new(sid).mode("save")
        };
        let full = g.mem.read_transcript(&args).expect("save");
        let saved_to = full["saved_to"].as_str().expect("saved_to");
        assert!(saved_to.ends_with(output));
        assert!(std::path::Path::new(saved_to).exists());
        g.check(name, &args);
    }
}

#[test]
fn read_transcript_save_rejects_escaping_paths() {
    let _lock = env_lock();
    let home = tempfile::tempdir().expect("tempdir");
    let _guard = EnvVarGuard::set("XDG_DATA_HOME", home.path().to_str().expect("utf-8"));

    let g = golden_mem();
    let absolute = ReadTranscriptArgs {
        output_path: Some("/tmp/ekko-abs.txt"),
        ..ReadTranscriptArgs::new("rt1").mode("save")
    };
    let r = g.mem.read_transcript(&absolute).expect("save");
    assert!(
        r["error"]
            .as_str()
            .expect("error")
            .to_lowercase()
            .contains("absolute")
    );

    let traversal = ReadTranscriptArgs {
        output_path: Some("../escape.txt"),
        ..ReadTranscriptArgs::new("rt1").mode("save")
    };
    let r = g.mem.read_transcript(&traversal).expect("save");
    let error = r["error"].as_str().expect("error").to_lowercase();
    assert!(error.contains("escape") || error.contains("sandbox"));
}

#[test]
fn read_transcript_attaches_session_id_and_metadata() {
    let g = golden_mem();
    let r = g
        .mem
        .read_transcript(&ReadTranscriptArgs::new("rt1"))
        .expect("info");
    assert_eq!(r["session_id"], json!("rt1"));
    assert_eq!(r["metadata"]["title"], json!("Session rt1"));
    assert_eq!(r["metadata"]["project_path"], json!("/home/user/project"));
    // `info` reports the row's timestamps.
    assert_eq!(r["created_at"], r["metadata"]["created_at"]);
    assert_eq!(r["modified_at"], r["metadata"]["modified_at"]);
    assert!(r["created_at"].is_string());
}

#[test]
fn read_transcript_cached_marks_source() {
    let g = golden_mem();
    let r = g
        .mem
        .read_transcript(&ReadTranscriptArgs::new("cached_simple").mode("full"))
        .expect("full");
    assert_eq!(r["source"], json!("cached"));
    assert!(
        r["transcript"]
            .as_str()
            .expect("transcript")
            .contains("auth bug")
    );
}

#[test]
fn read_transcript_unknown_cli_source_is_a_parse_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let transcript = dir.path().join("t.jsonl");
    std::fs::write(&transcript, FIXTURE_TRANSCRIPT).expect("write");
    let db = Database::new(dir.path().join("x.db")).expect("db");
    let s = SessionMetadata::new("weird", "atari_basic", transcript.to_str().expect("path"));
    db.upsert_session(&s).expect("upsert");
    let mem = EkkoMemory::new(db);
    let r = mem
        .read_transcript(&ReadTranscriptArgs::new("weird"))
        .expect("read");
    assert_eq!(
        r["error"],
        json!("Parse error: Unknown CLI source: atari_basic")
    );
    assert_eq!(r["session_id"], json!("weird"));
}

#[test]
fn parse_messages_from_text_handles_all_markers() {
    let msgs = transcript::parse_messages_from_text(
        "Human:\nhi\n\nA:\nshort form\n\n[Tool Calls]\nBash: ls\n\nAssistant:\ndone\n",
    );
    let roles: Vec<&str> = msgs.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, vec!["user", "assistant", "tools", "assistant"]);
    assert_eq!(msgs[1].content, "short form");
    assert!(transcript::parse_messages_from_text("no markers at all").is_empty());
}

#[test]
fn format_messages_labels_unknown_roles() {
    let msgs = vec![
        crate::models::Message {
            role: "system".into(),
            content: "boot".into(),
        },
        crate::models::Message {
            role: "tools".into(),
            content: "Bash: ls".into(),
        },
    ];
    assert_eq!(
        transcript::format_messages(&msgs, true),
        "[system]\nboot\n\n[Tool Calls]\nBash: ls\n"
    );
    assert_eq!(
        transcript::format_messages(&msgs, false),
        "[system]\nboot\n"
    );
}

#[test]
fn tail_with_zero_count_returns_everything() {
    // A count of zero means "no tail limit", so everything is returned.
    let msgs = vec![
        crate::models::Message {
            role: "user".into(),
            content: "a".into(),
        },
        crate::models::Message {
            role: "assistant".into(),
            content: "b".into(),
        },
    ];
    let out = transcript::tail(&msgs, 0, true);
    assert_eq!(out["showing"], json!("last 2 of 2 messages"));
    let out = transcript::head(&msgs, 0, true);
    assert_eq!(out["showing"], json!("first 0 of 2 messages"));
}

// ---------------------------------------------------------------------------
// sessions_in_directory / list_projects
// ---------------------------------------------------------------------------

#[test]
fn sessions_in_directory_lists_sessions() {
    let m = mem_db();
    let r = m
        .sessions_in_directory("/home/user/webapp", true, None, 50)
        .expect("ok");
    let got: HashSet<&str> = r
        .as_array()
        .expect("array")
        .iter()
        .map(|s| s["session_id"].as_str().expect("id"))
        .collect();
    assert!(got.contains("m1") && got.contains("m2") && got.contains("m4"));
    assert!(!got.contains("m3"));
    assert!(!got.contains("m6"), "zero-message sessions are filtered");
}

#[test]
fn sessions_in_directory_excludes_subdirs() {
    let m = mem_db();
    let r = m
        .sessions_in_directory("/home/user/webapp", false, None, 50)
        .expect("ok");
    let got: HashSet<&str> = r
        .as_array()
        .expect("array")
        .iter()
        .map(|s| s["session_id"].as_str().expect("id"))
        .collect();
    assert!(got.contains("m1"));
    assert!(!got.contains("m4"));
}

#[test]
fn sessions_in_directory_days_filter_and_fields() {
    let m = mem_db();
    m.db()
        .update_snapshot("m1", "## Decisions")
        .expect("snapshot");
    let r = m
        .sessions_in_directory("/home/user/webapp", true, Some(7), 50)
        .expect("ok");
    let arr = r.as_array().expect("array");
    assert!(arr.iter().all(|s| s["session_id"] != json!("m4")));
    let m1 = arr
        .iter()
        .find(|s| s["session_id"] == json!("m1"))
        .expect("m1");
    assert_eq!(m1["has_snapshot"], json!(true));
    assert_eq!(m1["title"], json!("Fix auth bug"));
    assert_eq!(m1["project_path"], json!("/home/user/webapp"));
    assert!(
        m1["summary"]
            .as_str()
            .expect("summary")
            .starts_with("Fixed")
    );
    // m2 has a summary but no snapshot.
    let m2 = arr
        .iter()
        .find(|s| s["session_id"] == json!("m2"))
        .expect("m2");
    assert_eq!(m2["has_snapshot"], json!(false));
}

#[test]
fn sessions_in_directory_empty() {
    let m = mem_db();
    let r = m
        .sessions_in_directory("/nowhere/at/all", true, None, 50)
        .expect("ok");
    assert_eq!(r, json!([]));
}

#[test]
fn sessions_in_directory_null_summary_is_null() {
    let m = mem_db();
    let r = m
        .sessions_in_directory("/home/user/webapp/api", true, None, 50)
        .expect("ok");
    assert_eq!(r[0]["session_id"], json!("m4"));
    assert_eq!(r[0]["summary"], Value::Null);
}

#[test]
fn list_projects_returns_counts() {
    let m = mem_db();
    let r = m.list_projects().expect("ok");
    let arr = r.as_array().expect("array");
    let webapp = arr
        .iter()
        .find(|p| p["project_path"] == json!("/home/user/webapp"))
        .expect("webapp project");
    assert_eq!(webapp["session_count"], json!(3));
    assert_eq!(webapp["cli_source"], json!("claude_code"));
    assert!(webapp["last_activity"].is_string());
}

// ---------------------------------------------------------------------------
// ask_question — pure pipeline pieces
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ask_question_without_candidates_short_circuits() {
    let m = mem_db();
    let r = m
        .ask_question(&AskArgs {
            directory: Some("/nowhere"),
            ..AskArgs::new("what did we do?")
        })
        .await
        .expect("ok");
    assert_eq!(r["answer"], json!("No sessions found matching your scope."));
    assert_eq!(r["tier_used"], json!("none"));
    assert_eq!(r["sources"], json!([]));
    assert_eq!(
        r["pagination"],
        json!({"offset": 0, "returned": 0, "total": 0, "has_more": false})
    );
}

#[tokio::test]
async fn ask_question_pagination_counts_all_candidates() {
    let m = mem_db();
    let r = m
        .ask_question(&AskArgs {
            offset: 99,
            ..AskArgs::new("what did we do?")
        })
        .await
        .expect("ok");
    // 3 sessions have summaries; offset past the end yields nothing.
    assert_eq!(r["pagination"]["total"], json!(3));
    assert_eq!(r["pagination"]["returned"], json!(0));
    assert_eq!(r["pagination"]["offset"], json!(99));
    assert_eq!(r["tier_used"], json!("none"));
}

#[test]
fn priority_sort_uses_the_inverted_key_order() {
    let mut sessions = vec![
        crate::db::SessionRow {
            project_path: Some("/home/user/webapp".into()),
            modified_at: Some("2025-01-01T00:00:00+00:00".into()),
            ..row("in_dir_old")
        },
        crate::db::SessionRow {
            project_path: Some("/home/user/other".into()),
            modified_at: Some("2024-01-01T00:00:00+00:00".into()),
            ..row("out_old")
        },
        crate::db::SessionRow {
            project_path: Some("/home/user/other".into()),
            modified_at: Some("2026-01-01T00:00:00+00:00".into()),
            ..row("out_new")
        },
    ];
    qa::sort_by_priority(&mut sessions, Some("/home/user/webapp"));
    let order: Vec<&str> = sessions.iter().map(|s| s.session_id.as_str()).collect();
    // `reverse=True` flips the whole `(-in_directory, modified)` key, so
    // out-of-directory sessions come first, each group newest-first.
    assert_eq!(order, vec!["out_new", "out_old", "in_dir_old"]);
}

#[test]
fn priority_sort_without_directory_is_recency_desc() {
    let mut sessions = vec![
        crate::db::SessionRow {
            modified_at: Some("2024-01-01T00:00:00+00:00".into()),
            ..row("old")
        },
        crate::db::SessionRow {
            modified_at: None,
            ..row("null")
        },
        crate::db::SessionRow {
            modified_at: Some("2026-01-01T00:00:00+00:00".into()),
            ..row("new")
        },
    ];
    qa::sort_by_priority(&mut sessions, None);
    let order: Vec<&str> = sessions.iter().map(|s| s.session_id.as_str()).collect();
    assert_eq!(order, vec!["new", "old", "null"]);
}

fn summary_session(id: &str, summary: &str) -> crate::db::SessionRow {
    crate::db::SessionRow {
        title: Some(format!("Title {id}")),
        summary: Some(summary.to_string()),
        project_path: Some("/home/user/webapp".into()),
        modified_at: Some("2025-02-11T09:00:00+00:00".into()),
        created_at: Some("2025-02-10T09:00:00+00:00".into()),
        message_count: 4,
        ..row(id)
    }
}

#[test]
fn pack_summaries_respects_the_token_budget() {
    let sessions: Vec<_> = (0..40)
        .map(|i| summary_session(&format!("s{i}"), &"word ".repeat(200)))
        .collect();
    // Budget below the reserve: nothing fits.
    let (tagged, block) = qa::pack_summaries("q", &sessions, qa::RESERVED_TOKENS);
    assert!(tagged.is_empty());
    assert_eq!(block, "");

    // A budget that fits a handful of records.
    let (tagged, block) = qa::pack_summaries("q", &sessions, qa::RESERVED_TOKENS + 2_000);
    assert!(!tagged.is_empty() && tagged.len() < sessions.len());
    assert_eq!(block.matches("[[S_").count(), tagged.len());
    assert!(crate::tokens::count_tokens(&block) <= 2_000);

    // The full budget packs everything, one record per session.
    let (tagged, _) = qa::pack_summaries("q", &sessions, qa::MAX_CONTEXT_TOKENS);
    assert_eq!(tagged.len(), sessions.len());
    let tags: HashSet<&str> = tagged.iter().map(|(t, _)| t.as_str()).collect();
    assert_eq!(tags.len(), sessions.len(), "tags must be unique");
}

#[test]
fn summary_outcome_answered_builds_sources() {
    let sessions = vec![
        summary_session("a", "Chose Redis"),
        summary_session("b", "Other"),
    ];
    let (tagged, _) = qa::pack_summaries("q", &sessions, qa::MAX_CONTEXT_TOKENS);
    let data = json!({
        "answered": true,
        "answer": "We chose Redis.",
        "sources": [tagged[0].0.clone(), "S_NOPE"],
    });
    match qa::summary_outcome(&data, &tagged) {
        SummaryOutcome::Answered { answer, sources } => {
            assert_eq!(answer, "We chose Redis.");
            assert_eq!(sources.len(), 1, "unknown tags are dropped");
            assert_eq!(sources[0]["session_id"], json!("a"));
            assert_eq!(sources[0]["title"], json!("Title a"));
            assert_eq!(sources[0]["project"], json!("webapp"));
            assert_eq!(sources[0]["date"], json!("2025-02-11"));
            assert_eq!(sources[0]["excerpt"], json!("Chose Redis"));
        }
        other => panic!("expected an answer, got {other:?}"),
    }
}

#[test]
fn summary_outcome_candidates_accepts_dicts_and_bare_tags() {
    let sessions = vec![summary_session("a", "A"), summary_session("b", "B")];
    let (tagged, _) = qa::pack_summaries("q", &sessions, qa::MAX_CONTEXT_TOKENS);
    let data = json!({
        "answered": false,
        "candidates": [
            {"tag": tagged[1].0.clone(), "why": "discusses it"},
            tagged[0].0.clone(),
            {"tag": "S_NOPE"},
        ],
    });
    match qa::summary_outcome(&data, &tagged) {
        SummaryOutcome::Candidates(c) => {
            let order: Vec<&str> = c.iter().map(|s| s.session_id.as_str()).collect();
            assert_eq!(order, vec!["b", "a"], "model order is preserved");
        }
        other => panic!("expected candidates, got {other:?}"),
    }
}

#[test]
fn summary_outcome_missing_fields_yield_no_candidates() {
    let sessions = vec![summary_session("a", "A")];
    let (tagged, _) = qa::pack_summaries("q", &sessions, qa::MAX_CONTEXT_TOKENS);
    assert_eq!(
        qa::summary_outcome(&json!({}), &tagged),
        SummaryOutcome::Candidates(vec![])
    );
}

#[test]
fn not_relevant_sentinel_blanks_the_answer() {
    assert_eq!(qa::apply_not_relevant("NOT_RELEVANT"), "");
    assert_eq!(
        qa::apply_not_relevant("Some text\nNOT_RELEVANT\nmore"),
        "",
        "the sentinel is matched anywhere in the reply"
    );
    assert_eq!(qa::apply_not_relevant("A real answer"), "A real answer");
}

#[test]
fn transcript_prompt_includes_context_and_skips_missing_summary() {
    let mut s = summary_session("a", "The summary");
    let with_summary = qa::transcript_prompt("Why Redis?", "Human:\nhi\n", &s).expect("prompt");
    assert!(with_summary.contains("\"Why Redis?\""));
    assert!(with_summary.contains("- **Project:** webapp"));
    assert!(with_summary.contains("- **Date:** 2025-02-10"));
    assert!(with_summary.contains("- **Summary:** The summary"));
    assert!(with_summary.contains("Human:\nhi\n"));

    s.summary = None;
    s.created_at = None;
    let without = qa::transcript_prompt("Why Redis?", "Human:\nhi\n", &s).expect("prompt");
    assert!(!without.contains("**Summary:**"));
    assert!(without.contains("- **Date:** unknown"));
}

fn answer(id: &str, project: &str, text: &str) -> TranscriptAnswer {
    TranscriptAnswer {
        session_id: id.to_string(),
        title: json!(format!("Title {id}")),
        project: project.to_string(),
        answer: text.to_string(),
        error: None,
    }
}

#[test]
fn synthesize_prompt_numbers_sessions_in_input_order() {
    let a = answer("a", "webapp", "First answer");
    let b = answer("b", "backend", "Second answer");
    let prompt = qa::synthesize_prompt("How?", &[&a, &b]).expect("prompt");
    let first = prompt.find("### Session 1: webapp").expect("session 1");
    let second = prompt.find("### Session 2: backend").expect("session 2");
    assert!(first < second);
    assert!(prompt.contains("- **Title:** Title a"));
    assert!(prompt.contains("First answer") && prompt.contains("Second answer"));
    // `date`/`summary` are deliberately absent from the context.
    assert!(prompt.contains("### Session 1: webapp ()"));
    assert!(!prompt.contains("**Summary:**"));
}

#[test]
fn synthesize_context_has_the_pinned_keys() {
    let a = answer("a", "webapp", "text");
    let ctx = qa::synthesize_answers_context(&[&a]);
    let keys: Vec<&str> = ctx[0]
        .as_object()
        .expect("obj")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(keys, vec!["session_id", "title", "project", "answer"]);
}

#[test]
fn combine_answers_fallback_format() {
    let a = answer("a", "webapp", "First");
    let b = answer("b", "backend", "Second");
    assert_eq!(
        qa::combine_answers(&[&a, &b]),
        "From webapp:\nFirst\n\nFrom backend:\nSecond"
    );
}

#[test]
fn build_source_carries_the_per_session_answer_as_excerpt() {
    let a = answer("a", "webapp", "It was Redis");
    assert_eq!(
        qa::build_source(&a),
        json!({
            "session_id": "a",
            "title": "Title a",
            "project": "webapp",
            "excerpt": "It was Redis",
        })
    );
}

#[tokio::test]
async fn reduce_without_answers_reports_low_confidence() {
    let tracker = crate::llm::UsageTracker::new();
    let results = vec![
        TranscriptAnswer {
            answer: String::new(),
            error: Some("no_transcript".into()),
            ..answer("a", "webapp", "")
        },
        TranscriptAnswer {
            answer: String::new(),
            error: Some("empty".into()),
            ..answer("b", "backend", "")
        },
    ];
    let out = qa::finalize_transcript_answers("q", results, &tracker).await;
    assert_eq!(
        out.answer,
        "I found relevant sessions but couldn't extract a useful answer."
    );
    assert_eq!(out.confidence, "low");
    assert_eq!(
        out.sources,
        vec![json!({"session_id": "a"}), json!({"session_id": "b"})]
    );
}

#[tokio::test]
async fn reduce_with_one_answer_skips_synthesis() {
    let tracker = crate::llm::UsageTracker::new();
    let results = vec![
        answer("a", "webapp", "The answer"),
        TranscriptAnswer {
            answer: String::new(),
            ..answer("b", "backend", "")
        },
    ];
    let out = qa::finalize_transcript_answers("q", results, &tracker).await;
    assert_eq!(out.answer, "The answer");
    assert_eq!(out.confidence, "high");
    assert_eq!(out.sources.len(), 1);
    assert_eq!(out.sources[0]["excerpt"], json!("The answer"));
}
