//! Behavioral tests for the storage layer, plus pins for FTS query prep,
//! claims contention, telemetry, porter stemming and `suggest_similar_paths`.
//!
//! The expected values here are pinned; update them deliberately when
//! behavior changes intentionally.

use chrono::{DateTime, FixedOffset};
use serde_json::json;
use tempfile::TempDir;

use super::search::prepare_fts_query;
use super::similar::{path_signature, sequence_ratio};
use super::telemetry::LlmUsage;
use super::{Database, format_iso8601, json_dumps};
use crate::models::SessionMetadata;

struct TestDb {
    db: Database,
    _dir: TempDir,
}

fn test_db() -> TestDb {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::new(dir.path().join("test.db")).expect("create db");
    TestDb { db, _dir: dir }
}

fn make_session(session_id: &str) -> SessionMetadata {
    let mut m = SessionMetadata::new(
        session_id,
        "claude_code",
        format!("/data/{session_id}.jsonl"),
    );
    m.title = Some(format!("Session {session_id}"));
    m.first_prompt = Some(format!("prompt for {session_id}"));
    m.project_path = Some("/home/user/project".to_string());
    m.message_count = 5;
    m
}

fn dt(s: &str) -> DateTime<FixedOffset> {
    DateTime::parse_from_rfc3339(s).expect("valid datetime")
}

/// Shared fixture: 6 sessions, one of them empty.
fn populated_db() -> TestDb {
    let t = test_db();
    let specs: &[(&str, &str, &str, &str, &str, i64, &str)] = &[
        (
            "s1",
            "claude_code",
            "Fix authentication bug",
            "The login authentication is broken",
            "/home/user/webapp",
            10,
            "2025-01-05T10:00:00+00:00",
        ),
        (
            "s2",
            "claude_code",
            "Add dashboard widgets",
            "Create a new dashboard with widgets",
            "/home/user/webapp",
            8,
            "2025-01-04T10:00:00+00:00",
        ),
        (
            "s3",
            "github_copilot",
            "Refactor database layer",
            "Refactor the session database",
            "/home/user/tool",
            6,
            "2025-01-03T10:00:00+00:00",
        ),
        (
            "s4",
            "claude_code",
            "Write API tests",
            "Add tests for the API session endpoints",
            "/home/user/webapp/api",
            4,
            "2025-01-02T10:00:00+00:00",
        ),
        (
            "s5",
            "github_copilot",
            "Update documentation",
            "Update the session README",
            "/home/user/tool",
            2,
            "2025-01-01T10:00:00+00:00",
        ),
        (
            "s6",
            "claude_code",
            "Empty session",
            "nothing here",
            "/home/user/webapp",
            0,
            "2024-12-31T10:00:00+00:00",
        ),
    ];
    for (sid, cli, title, prompt, project, count, modified) in specs {
        let mut m = SessionMetadata::new(*sid, *cli, format!("/data/{sid}.jsonl"));
        m.title = Some(title.to_string());
        m.first_prompt = Some(prompt.to_string());
        m.project_path = Some(project.to_string());
        m.message_count = *count;
        m.modified_at = Some(dt(modified));
        t.db.upsert_session(&m).expect("upsert");
    }
    t
}

// ── upsert_session ──────────────────────────────────────────────────────

#[test]
fn upsert_insert_new() {
    let t = test_db();
    t.db.upsert_session(&make_session("new1")).unwrap();
    let row = t.db.get_session("new1").unwrap().expect("row");
    assert_eq!(row.session_id, "new1");
    assert_eq!(row.title.as_deref(), Some("Session new1"));
    assert_eq!(row.cli_source, "claude_code");
}

#[test]
fn upsert_update_preserves_title_updates_counts() {
    let t = test_db();
    let mut m = make_session("u1");
    m.title = Some("Original".to_string());
    t.db.upsert_session(&m).unwrap();

    m.title = Some("Ignored".to_string());
    m.message_count = 20;
    t.db.upsert_session(&m).unwrap();

    let row = t.db.get_session("u1").unwrap().unwrap();
    // title is NOT updated by upsert (only by update_summary)
    assert_eq!(row.title.as_deref(), Some("Original"));
    assert_eq!(row.message_count, 20);
}

#[test]
fn upsert_preserves_summary_on_update() {
    let t = test_db();
    t.db.upsert_session(&make_session("p1")).unwrap();
    t.db.update_summary("p1", "LLM Title", "LLM Summary")
        .unwrap();

    let mut m = make_session("p1");
    m.title = None;
    t.db.upsert_session(&m).unwrap();
    let row = t.db.get_session("p1").unwrap().unwrap();
    assert_eq!(row.summary.as_deref(), Some("LLM Summary"));
}

#[test]
fn upsert_coalesce_first_prompt_none_keeps_old() {
    let t = test_db();
    t.db.upsert_session(&make_session("c1")).unwrap();
    let mut m = make_session("c1");
    m.first_prompt = None;
    t.db.upsert_session(&m).unwrap();
    let row = t.db.get_session("c1").unwrap().unwrap();
    assert_eq!(row.first_prompt.as_deref(), Some("prompt for c1"));
}

#[test]
fn upsert_preserves_empty_string_vs_none() {
    // git_branch Some("") must round-trip as '' (distinct from NULL):
    // parsers rely on the two being distinguishable.
    let t = test_db();
    let mut m = make_session("es1");
    m.git_branch = Some(String::new());
    m.git_remote = None;
    t.db.upsert_session(&m).unwrap();
    let row = t.db.get_session("es1").unwrap().unwrap();
    assert_eq!(row.git_branch.as_deref(), Some(""));
    assert_eq!(row.git_remote, None);

    // And a re-upsert with None overwrites (no COALESCE on git_branch).
    m.git_branch = None;
    t.db.upsert_session(&m).unwrap();
    let row = t.db.get_session("es1").unwrap().unwrap();
    assert_eq!(row.git_branch, None);
}

#[test]
fn upsert_datetime_round_trip() {
    let t = test_db();
    let mut m = make_session("dt1");
    m.created_at = Some(dt("2025-06-15T12:00:00+00:00"));
    m.modified_at = Some(dt("2025-06-15T12:00:00+00:00"));
    t.db.upsert_session(&m).unwrap();
    let row = t.db.get_session("dt1").unwrap().unwrap();
    // Exact bytes: no fractional part when µs == 0.
    assert_eq!(row.created_at.as_deref(), Some("2025-06-15T12:00:00+00:00"));
}

#[test]
fn format_iso8601_is_pinned() {
    // Pinned expected bytes.
    assert_eq!(
        format_iso8601(&dt("2025-06-15T12:00:00+00:00")),
        "2025-06-15T12:00:00+00:00"
    );
    assert_eq!(
        format_iso8601(&dt("2025-06-15T12:00:00.123456+00:00")),
        "2025-06-15T12:00:00.123456+00:00"
    );
    assert_eq!(
        format_iso8601(&dt("2025-06-15T12:00:00.000123-07:00")),
        "2025-06-15T12:00:00.000123-07:00"
    );
}

// ── get_session / get_sessions ──────────────────────────────────────────

#[test]
fn get_session_missing_returns_none() {
    let t = test_db();
    assert!(t.db.get_session("nonexistent").unwrap().is_none());
}

#[test]
fn get_sessions_returns_all_and_limit() {
    let t = populated_db();
    assert_eq!(t.db.get_sessions(None, None, None, 0).unwrap().len(), 6);
    assert_eq!(t.db.get_sessions(None, None, Some(2), 0).unwrap().len(), 2);
}

#[test]
fn get_sessions_filter_by_cli_source() {
    let t = populated_db();
    let sessions =
        t.db.get_sessions(Some("github_copilot"), None, None, 0)
            .unwrap();
    assert_eq!(sessions.len(), 2);
    assert!(sessions.iter().all(|s| s.cli_source == "github_copilot"));
}

#[test]
fn get_sessions_filter_by_project_path_prefix() {
    let t = populated_db();
    let sessions =
        t.db.get_sessions(None, Some("/home/user/webapp"), None, 0)
            .unwrap();
    let mut ids: Vec<&str> = sessions.iter().map(|s| s.session_id.as_str()).collect();
    ids.sort_unstable();
    assert_eq!(ids, vec!["s1", "s2", "s4", "s6"]);
}

#[test]
fn get_sessions_ordered_by_modified_desc() {
    let t = populated_db();
    let sessions = t.db.get_sessions(None, None, None, 0).unwrap();
    let dates: Vec<_> = sessions
        .iter()
        .filter_map(|s| s.modified_at.clone())
        .collect();
    let mut sorted = dates.clone();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    assert_eq!(dates, sorted);
}

#[test]
fn get_sessions_offset_paging() {
    let t = populated_db();
    let all = t.db.get_sessions(None, None, None, 0).unwrap();

    let offset = t.db.get_sessions(None, None, None, 2).unwrap();
    assert_eq!(offset.len(), all.len() - 2);
    assert_eq!(offset[0].session_id, all[2].session_id);

    let page = t.db.get_sessions(None, None, Some(2), 2).unwrap();
    assert_eq!(page.len(), 2);
    assert_eq!(page[0].session_id, all[2].session_id);
    assert_eq!(page[1].session_id, all[3].session_id);

    assert!(t.db.get_sessions(None, None, None, 100).unwrap().is_empty());
    // offset without limit uses LIMIT -1
    assert_eq!(
        t.db.get_sessions(None, None, None, 1).unwrap().len(),
        all.len() - 1
    );
}

// ── _prepare_fts_query ──────────────────────────────────────────────────

#[test]
fn prepare_fts_query_goldens() {
    // Pinned expected output.
    assert_eq!(
        prepare_fts_query("GenAI survey paper"),
        "GenAI survey paper"
    );
    assert_eq!(
        prepare_fts_query("drill-down terms"),
        "\"drill-down\" terms"
    );
    assert_eq!(prepare_fts_query("path/to/file.py"), "\"path/to/file.py\"");
    assert_eq!(
        prepare_fts_query("say \"hi\" now"),
        "say \"\"\"hi\"\"\" now"
    );
    assert_eq!(prepare_fts_query("wild* card"), "wild* card");
    assert_eq!(prepare_fts_query("naïve café"), "naïve café");
    assert_eq!(prepare_fts_query("a.b c_d"), "\"a.b\" c_d");
    assert_eq!(prepare_fts_query(""), "");
}

// ── search_keyword (FTS5) ──────────────────────────────────────────────

#[test]
fn search_keyword_matches_title() {
    let t = populated_db();
    let results = t.db.search_keyword("authentication", 20).unwrap();
    assert!(!results.is_empty());
    assert!(results.iter().any(|r| {
        r.session
            .title
            .as_deref()
            .unwrap_or("")
            .to_lowercase()
            .contains("authentication")
    }));
}

#[test]
fn search_keyword_matches_first_prompt() {
    let t = populated_db();
    assert!(!t.db.search_keyword("dashboard", 20).unwrap().is_empty());
}

#[test]
fn search_keyword_no_results_for_nonsense() {
    let t = populated_db();
    assert!(t.db.search_keyword("xyzzyplugh42", 20).unwrap().is_empty());
}

#[test]
fn search_keyword_respects_limit() {
    let t = populated_db();
    assert!(t.db.search_keyword("session", 1).unwrap().len() <= 1);
}

#[test]
fn search_keyword_special_characters_handled() {
    let t = populated_db();
    // Must not error (quoted-token path).
    t.db.search_keyword("path/to/file.py", 20).unwrap();
}

#[test]
fn search_keyword_matches_summary_after_update() {
    // Also pins that the sessions_au trigger keeps sessions_fts in sync.
    let t = populated_db();
    t.db.update_summary("s1", "Auth Fix", "Fixed the authentication middleware")
        .unwrap();
    let results = t.db.search_keyword("middleware", 20).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].session.session_id, "s1");
}

#[test]
fn search_keyword_snippet_shape() {
    let t = populated_db();
    t.db.update_summary(
        "s1",
        "Auth Fix",
        "Fixed the authentication middleware in the session handler",
    )
    .unwrap();
    let results = t.db.search_keyword("middleware", 20).unwrap();
    let excerpt = results[0].summary_excerpt.as_deref().unwrap();
    assert!(excerpt.contains(">>>middleware<<<"), "excerpt: {excerpt}");
}

#[test]
fn search_keyword_bm25_orders_better_match_first() {
    let t = test_db();
    let mut a = make_session("bm1");
    a.first_prompt = Some("redis redis redis caching with redis".to_string());
    t.db.upsert_session(&a).unwrap();
    let mut b = make_session("bm2");
    b.first_prompt = Some("we briefly mentioned redis once in a long discussion about many other unrelated topics and tools".to_string());
    t.db.upsert_session(&b).unwrap();

    let results = t.db.search_keyword("redis", 20).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].session.session_id, "bm1");
}

#[test]
fn search_keyword_fallback_on_fts_syntax_error() {
    let t = test_db();
    let mut m = make_session("fts1");
    m.title = Some("OR session".to_string());
    t.db.upsert_session(&m).unwrap();
    // "OR" alone is an FTS5 syntax error; the fallback quotes it.
    let results = t.db.search_keyword("OR", 20).unwrap();
    // Fallback treats it as the literal token "or" — title matches.
    assert!(results.iter().any(|r| r.session.session_id == "fts1"));
}

#[test]
fn search_keyword_finds_keywords_column() {
    let t = test_db();
    t.db.upsert_session(&make_session("k1")).unwrap();
    t.db.update_keywords(
        "k1",
        &[
            "authentication".to_string(),
            "middleware".to_string(),
            "redis".to_string(),
        ],
    )
    .unwrap();
    let results = t.db.search_keyword("redis", 20).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].session.session_id, "k1");

    // Stored JSON uses the ", " separator, not the compact one.
    let row = t.db.get_session("k1").unwrap().unwrap();
    assert_eq!(
        row.keywords.as_deref(),
        Some(r#"["authentication", "middleware", "redis"]"#)
    );

    // Empty list stores NULL, not an empty JSON array.
    t.db.update_keywords("k1", &[]).unwrap();
    let row = t.db.get_session("k1").unwrap().unwrap();
    assert_eq!(row.keywords, None);
}

// ── update_summary / update_snapshot ───────────────────────────────────

#[test]
fn update_summary_sets_fields_and_stamp() {
    let t = test_db();
    t.db.upsert_session(&make_session("sum1")).unwrap();
    t.db.update_summary("sum1", "New Title", "New Summary")
        .unwrap();
    let row = t.db.get_session("sum1").unwrap().unwrap();
    assert_eq!(row.title.as_deref(), Some("New Title"));
    assert_eq!(row.summary.as_deref(), Some("New Summary"));
    let stamp = row.summary_at.expect("summary_at stamped");
    assert!(stamp.ends_with("+00:00"), "UTC stamp: {stamp}");
}

#[test]
fn update_snapshot_sets_fields_and_stamp() {
    let t = test_db();
    t.db.upsert_session(&make_session("snap1")).unwrap();
    t.db.update_snapshot("snap1", "## Key Decisions\n- Used Redis")
        .unwrap();
    let row = t.db.get_session("snap1").unwrap().unwrap();
    assert_eq!(
        row.context_snapshot.as_deref(),
        Some("## Key Decisions\n- Used Redis")
    );
    assert!(row.snapshot_at.is_some());
}

#[test]
fn summary_retrievable_after_update() {
    let t = test_db();
    t.db.upsert_session(&make_session("sr1")).unwrap();
    t.db.update_summary("sr1", "Title", "Summary text").unwrap();
    let summaries = t.db.get_sessions_with_summaries(None).unwrap();
    assert!(summaries.iter().any(|s| s.session_id == "sr1"));
}

// ── session files ──────────────────────────────────────────────────────

#[test]
fn upsert_with_files_roundtrip() {
    let t = test_db();
    let mut m = make_session("f1");
    m.files_touched_count = 2;
    let files = vec![
        ("/src/main.py".to_string(), "write".to_string()),
        ("/src/utils.py".to_string(), "read".to_string()),
    ];
    t.db.upsert_session_with_files(&m, &files).unwrap();
    let session_files = t.db.get_session_files("f1").unwrap();
    assert_eq!(session_files.len(), 2);
    let paths: Vec<&str> = session_files.iter().map(|f| f.file_path.as_str()).collect();
    assert!(paths.contains(&"/src/main.py"));
    assert!(paths.contains(&"/src/utils.py"));
    let row = t.db.get_session("f1").unwrap().unwrap();
    assert_eq!(row.files_touched_count, 2);
}

#[test]
fn files_replaced_on_re_upsert() {
    let t = test_db();
    let m = make_session("f2");
    t.db.upsert_session_with_files(&m, &[("/a.py".to_string(), "read".to_string())])
        .unwrap();
    t.db.upsert_session_with_files(&m, &[("/b.py".to_string(), "write".to_string())])
        .unwrap();
    let files = t.db.get_session_files("f2").unwrap();
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].file_path, "/b.py");
    assert_eq!(files[0].operation, "write");
}

#[test]
fn get_files_for_sessions_batched() {
    let t = test_db();
    t.db.upsert_session_with_files(
        &make_session("m1"),
        &[
            ("/a.py".to_string(), "read".to_string()),
            ("/b.py".to_string(), "write".to_string()),
        ],
    )
    .unwrap();
    t.db.upsert_session_with_files(
        &make_session("m2"),
        &[("/c.py".to_string(), "read".to_string())],
    )
    .unwrap();

    let result =
        t.db.get_files_for_sessions(&["m1".to_string(), "m2".to_string(), "m3".to_string()])
            .unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].0, "m1");
    assert_eq!(result[0].1.len(), 2);
    assert_eq!(result[1].1.len(), 1);
    // Nonexistent session still gets an (empty) entry.
    assert_eq!(result[2].0, "m3");
    assert!(result[2].1.is_empty());
}

#[test]
fn empty_files_list_ok() {
    let t = test_db();
    t.db.upsert_session_with_files(&make_session("f5"), &[])
        .unwrap();
    assert!(t.db.get_session_files("f5").unwrap().is_empty());
}

// ── schema idempotency ─────────────────────────────────────────────────

#[test]
fn double_init_is_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("idem.db");
    let _a = Database::new(&path).unwrap();
    let b = Database::new(&path).unwrap();
    b.upsert_session(&make_session("idem1")).unwrap();
    assert!(b.get_session("idem1").unwrap().is_some());
}

// ── get_stats ──────────────────────────────────────────────────────────

#[test]
fn stats_reflect_data() {
    let t = populated_db();
    let stats = t.db.get_stats().unwrap();
    assert_eq!(stats.total_indexed, 6);
    assert_eq!(stats.total_sessions, 5); // s6 has 0 messages
    let clis: Vec<&str> = stats.by_cli.iter().map(|(c, _)| c.as_str()).collect();
    assert!(clis.contains(&"claude_code"));
    assert!(clis.contains(&"github_copilot"));
}

#[test]
fn empty_db_stats() {
    let t = test_db();
    let stats = t.db.get_stats().unwrap();
    assert_eq!(stats.total_sessions, 0);
    assert_eq!(stats.total_indexed, 0);
}

// ── get_projects ───────────────────────────────────────────────────────

#[test]
fn get_projects_groups_and_orders() {
    let t = populated_db();
    let projects = t.db.get_projects().unwrap();
    // (webapp, claude_code), (webapp/api, claude_code), (tool, github_copilot)
    assert_eq!(projects.len(), 3);
    // Ordered by last_activity DESC — webapp's newest session is s1.
    assert_eq!(projects[0].project_path, "/home/user/webapp");
    assert_eq!(projects[0].session_count, 3);
}

// ── transcript FTS ─────────────────────────────────────────────────────

#[test]
fn transcript_index_and_search() {
    let t = test_db();
    t.db.upsert_session(&make_session("t1")).unwrap();
    t.db.index_transcript("t1", "The user asked about authentication middleware")
        .unwrap();
    assert!(t.db.is_transcript_indexed("t1").unwrap());
    let results = t.db.search_transcripts("authentication", 20).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].session.session_id, "t1");
    let excerpt = results[0].excerpt.as_deref().unwrap();
    assert!(excerpt.contains(">>>"), "excerpt: {excerpt}");

    // transcript_indexed_at got stamped as an offset-aware UTC timestamp.
    let row = t.db.get_session("t1").unwrap().unwrap();
    assert!(row.transcript_indexed_at.unwrap().ends_with("+00:00"));

    // Retrievable text.
    assert_eq!(
        t.db.get_transcript_text("t1").unwrap().as_deref(),
        Some("The user asked about authentication middleware")
    );
}

#[test]
fn transcript_not_indexed() {
    let t = test_db();
    t.db.upsert_session(&make_session("t2")).unwrap();
    assert!(!t.db.is_transcript_indexed("t2").unwrap());
    assert!(t.db.get_transcript_text("t2").unwrap().is_none());
}

#[test]
fn transcript_search_no_results() {
    let t = test_db();
    t.db.upsert_session(&make_session("t3")).unwrap();
    t.db.index_transcript("t3", "some text about databases")
        .unwrap();
    assert!(
        t.db.search_transcripts("xyznonexistent", 20)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn transcript_porter_stemming() {
    // transcripts_fts uses tokenize='porter unicode61': "running" ↔ "run".
    let t = test_db();
    t.db.upsert_session(&make_session("stem1")).unwrap();
    t.db.index_transcript("stem1", "the tests were running quickly")
        .unwrap();
    let results = t.db.search_transcripts("run", 20).unwrap();
    assert_eq!(results.len(), 1);
    // Sessions FTS (default tokenizer) does NOT stem — control check.
    let mut m = make_session("stem2");
    m.first_prompt = Some("the tests were running quickly".to_string());
    t.db.upsert_session(&m).unwrap();
    assert!(t.db.search_keyword("run", 20).unwrap().is_empty());
}

#[test]
fn transcript_reindex_replaces() {
    let t = test_db();
    t.db.upsert_session(&make_session("ri1")).unwrap();
    t.db.index_transcript("ri1", "old words alpha").unwrap();
    t.db.index_transcript("ri1", "new words beta").unwrap();
    assert!(t.db.search_transcripts("alpha", 20).unwrap().is_empty());
    assert_eq!(t.db.search_transcripts("beta", 20).unwrap().len(), 1);
}

#[test]
fn transcript_search_fallback_on_fts_syntax_error() {
    let t = test_db();
    t.db.upsert_session(&make_session("fts2")).unwrap();
    t.db.index_transcript("fts2", "The user typed OR into the search box")
        .unwrap();
    let results = t.db.search_transcripts("OR", 20).unwrap();
    assert!(results.iter().any(|r| r.session.session_id == "fts2"));
}

#[test]
fn transcript_index_stats_and_needing() {
    let t = test_db();
    let mut m = make_session("ti1");
    m.modified_at = Some(dt("2025-01-05T10:00:00+00:00"));
    t.db.upsert_session(&m).unwrap();

    let stats = t.db.get_transcript_index_stats().unwrap();
    assert_eq!(stats.indexed, 0);
    assert_eq!(stats.total_with_transcripts, 1);
    assert_eq!(stats.coverage, "0/1");

    // Never indexed → needs indexing.
    let needing = t.db.get_sessions_needing_transcript_index(None).unwrap();
    assert_eq!(needing.len(), 1);

    t.db.index_transcript("ti1", "hello world").unwrap();
    assert!(
        t.db.get_sessions_needing_transcript_index(None)
            .unwrap()
            .is_empty()
    );
    assert_eq!(t.db.get_transcript_index_stats().unwrap().coverage, "1/1");

    // File modified after indexing → stale again.
    let conn = t.db.connection().unwrap();
    conn.execute(
        "UPDATE sessions SET modified_at = '2099-01-01T00:00:00+00:00' WHERE session_id = 'ti1'",
        [],
    )
    .unwrap();
    let needing = t.db.get_sessions_needing_transcript_index(None).unwrap();
    assert_eq!(needing.len(), 1);

    // mark_transcript_indexed stamps without content.
    t.db.upsert_session(&make_session("ti2")).unwrap();
    t.db.mark_transcript_indexed("ti2").unwrap();
    assert!(!t.db.is_transcript_indexed("ti2").unwrap());
    let needing = t.db.get_sessions_needing_transcript_index(None).unwrap();
    assert!(needing.iter().all(|s| s.session_id != "ti2"));
}

// ── staleness queries ──────────────────────────────────────────────────

#[test]
fn needing_summary_basics() {
    let t = populated_db();
    // 5 non-empty sessions lack summaries.
    assert_eq!(
        t.db.get_sessions_needing_summary(None, true).unwrap().len(),
        5
    );

    t.db.update_summary("s1", "Title", "Summary").unwrap();
    let results = t.db.get_sessions_needing_summary(None, false).unwrap();
    assert!(results.iter().all(|s| s.session_id != "s1"));

    // limit
    assert_eq!(
        t.db.get_sessions_needing_summary(Some(2), true)
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn needing_summary_includes_stale() {
    let t = populated_db();
    t.db.update_summary("s1", "Title", "Summary").unwrap();
    // Not stale yet (summary_at just stamped > modified_at).
    let results = t.db.get_sessions_needing_summary(None, true).unwrap();
    assert!(results.iter().all(|s| s.session_id != "s1"));

    let conn = t.db.connection().unwrap();
    conn.execute(
        "UPDATE sessions SET modified_at = '2099-01-01T00:00:00+00:00' WHERE session_id = 's1'",
        [],
    )
    .unwrap();
    let results = t.db.get_sessions_needing_summary(None, true).unwrap();
    assert!(results.iter().any(|s| s.session_id == "s1"));
}

#[test]
fn needing_summary_excludes_zero_message_sessions() {
    let t = test_db();
    let mut m = make_session("empty");
    m.message_count = 0;
    t.db.upsert_session(&m).unwrap();
    assert!(
        t.db.get_sessions_needing_summary(None, true)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn needing_summary_stable_min_age() {
    let t = test_db();
    // Old session (modified 2025) qualifies for min_age 1h; a session
    // modified "now" does not.
    let mut old = make_session("old1");
    old.modified_at = Some(dt("2025-01-01T10:00:00+00:00"));
    t.db.upsert_session(&old).unwrap();

    let mut fresh = make_session("fresh1");
    fresh.modified_at = Some(chrono::Utc::now().fixed_offset());
    t.db.upsert_session(&fresh).unwrap();

    let results = t.db.get_sessions_needing_summary_stable(1.0, None).unwrap();
    let ids: Vec<&str> = results.iter().map(|s| s.session_id.as_str()).collect();
    assert!(ids.contains(&"old1"));
    assert!(!ids.contains(&"fresh1"));
}

#[test]
fn needing_snapshot_basics() {
    let t = populated_db();
    assert_eq!(
        t.db.get_sessions_needing_snapshot(None, true, None, false)
            .unwrap()
            .len(),
        5
    );

    t.db.update_snapshot("s1", "## Snapshot content").unwrap();
    let results =
        t.db.get_sessions_needing_snapshot(None, false, None, false)
            .unwrap();
    assert!(results.iter().all(|s| s.session_id != "s1"));

    assert_eq!(
        t.db.get_sessions_needing_snapshot(Some(2), true, None, false)
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn needing_snapshot_includes_stale() {
    let t = populated_db();
    t.db.update_snapshot("s1", "## Snapshot").unwrap();
    let conn = t.db.connection().unwrap();
    conn.execute(
        "UPDATE sessions SET modified_at = '2099-01-01T00:00:00+00:00' WHERE session_id = 's1'",
        [],
    )
    .unwrap();
    let results =
        t.db.get_sessions_needing_snapshot(None, true, None, false)
            .unwrap();
    assert!(results.iter().any(|s| s.session_id == "s1"));
}

#[test]
fn needing_snapshot_require_summary() {
    let t = populated_db();
    assert!(
        t.db.get_sessions_needing_snapshot(None, true, None, true)
            .unwrap()
            .is_empty()
    );
    t.db.update_summary("s1", "Title", "Summary").unwrap();
    let results =
        t.db.get_sessions_needing_snapshot(None, true, None, true)
            .unwrap();
    let ids: Vec<&str> = results.iter().map(|s| s.session_id.as_str()).collect();
    assert_eq!(ids, vec!["s1"]);
}

#[test]
fn needing_snapshot_min_age_orders_desc() {
    let t = populated_db();
    let results =
        t.db.get_sessions_needing_snapshot(None, true, Some(1.0), false)
            .unwrap();
    // All populated sessions are old; ordered modified_at DESC.
    assert_eq!(results.len(), 5);
    let dates: Vec<_> = results
        .iter()
        .filter_map(|s| s.modified_at.clone())
        .collect();
    let mut sorted = dates.clone();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    assert_eq!(dates, sorted);
}

// ── claims ─────────────────────────────────────────────────────────────

#[test]
fn claims_contention_and_release() {
    let t = test_db();
    // First instance wins.
    assert!(
        t.db.try_claim_summary("s1", "hash1", "inst-a", 600)
            .unwrap()
    );
    // Second instance loses while the claim is live.
    assert!(
        !t.db
            .try_claim_summary("s1", "hash1", "inst-b", 600)
            .unwrap()
    );
    // Same instance also cannot re-claim: the UPSERT's WHERE clause fails.
    assert!(
        !t.db
            .try_claim_summary("s1", "hash1", "inst-a", 600)
            .unwrap()
    );
    // Different content hash is an independent claim.
    assert!(
        t.db.try_claim_summary("s1", "hash2", "inst-b", 600)
            .unwrap()
    );

    // After release, claimable again.
    t.db.release_claim("s1", "hash1").unwrap();
    assert!(
        t.db.try_claim_summary("s1", "hash1", "inst-b", 600)
            .unwrap()
    );
}

#[test]
fn claims_expired_reclaimable() {
    let t = test_db();
    // ttl <= 0 → claimed_until in the past → immediately expired.
    assert!(t.db.try_claim_summary("s2", "h", "inst-a", -10).unwrap());
    assert!(t.db.try_claim_summary("s2", "h", "inst-b", 600).unwrap());

    // cleanup_expired_claims removes dead rows.
    assert!(t.db.try_claim_summary("s3", "h", "inst-a", -10).unwrap());
    t.db.cleanup_expired_claims().unwrap();
    let conn = t.db.connection().unwrap();
    let remaining: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM summary_claims WHERE session_id = 's3'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(remaining, 0);
}

// ── telemetry ──────────────────────────────────────────────────────────

#[test]
fn telemetry_record_and_export() {
    let t = test_db();
    let mut args = serde_json::Map::new();
    args.insert("limit".to_string(), json!(10));
    args.insert("verbose".to_string(), json!(true));
    t.db.record_telemetry(
        "inst-1",
        "search_sessions",
        Some(42),
        Some(&args),
        &LlmUsage::default(),
    )
    .unwrap();
    t.db.record_telemetry("inst-1", "get_status", None, None, &LlmUsage::default())
        .unwrap();
    // An empty safe_args map is stored as NULL.
    t.db.record_telemetry(
        "inst-2",
        "get_status",
        Some(1),
        Some(&serde_json::Map::new()),
        &LlmUsage::default(),
    )
    .unwrap();

    let events = t.db.export_telemetry(None).unwrap();
    assert_eq!(events.len(), 3);
    let first = events
        .iter()
        .find(|e| e.operation == "search_sessions")
        .unwrap();
    assert_eq!(first.instance_id, "inst-1");
    assert_eq!(first.duration_ms, Some(42));
    assert_eq!(first.safe_args, Some(json!({"limit": 10, "verbose": true})));
    assert!(first.timestamp.ends_with("+00:00"));
    assert!(
        events
            .iter()
            .filter(|e| e.operation == "get_status")
            .all(|e| e.safe_args.is_none())
    );

    // Days filter: everything is recent, so days=7 keeps all.
    assert_eq!(t.db.export_telemetry(Some(7)).unwrap().len(), 3);
}

#[test]
fn telemetry_summary_aggregates() {
    let t = test_db();
    for d in [10, 20, 30] {
        t.db.record_telemetry("i", "op_a", Some(d), None, &LlmUsage::default())
            .unwrap();
    }
    t.db.record_telemetry("i", "op_b", None, None, &LlmUsage::default())
        .unwrap();

    let summary = t.db.get_telemetry_summary(None).unwrap();
    assert_eq!(summary.total_events, 4);
    assert!(summary.first_event.is_some());
    // Ordered by count DESC: op_a first.
    assert_eq!(summary.operations[0].0, "op_a");
    let stats = &summary.operations[0].1;
    assert_eq!(stats.count, 3);
    assert_eq!(stats.avg_duration_ms, Some(20));
    assert_eq!(stats.min_duration_ms, Some(10));
    assert_eq!(stats.max_duration_ms, Some(30));
    // op_b has all-NULL durations → avg falsy → None.
    let b = &summary.operations[1].1;
    assert_eq!(b.avg_duration_ms, None);
}

#[test]
fn telemetry_cost_stats() {
    let t = test_db();
    let usage = LlmUsage {
        model: Some("gpt-5-mini".to_string()),
        input_tokens: Some(1000),
        output_tokens: Some(200),
        thinking_tokens: Some(50),
        cost: Some(0.0123),
    };
    t.db.record_telemetry("inst-1", "summarize", Some(5), None, &usage)
        .unwrap();
    let usage2 = LlmUsage {
        cost: Some(0.0002),
        ..LlmUsage::default()
    };
    t.db.record_telemetry("inst-2", "ask_question", Some(5), None, &usage2)
        .unwrap();
    // No-cost event must not count as a call.
    t.db.record_telemetry("inst-1", "get_status", Some(1), None, &LlmUsage::default())
        .unwrap();

    let stats = t.db.get_cost_stats(Some("inst-1")).unwrap();
    let mine = stats.this_instance.unwrap();
    assert_eq!(mine.calls, 1);
    assert_eq!(mine.cost, 0.0123);
    assert_eq!(mine.cost_display, "$0.0123");
    assert_eq!(mine.input_tokens, 1000);

    assert_eq!(stats.today.calls, 2);
    assert!((stats.today.cost - 0.0125).abs() < 1e-9);
    assert_eq!(stats.last_7_days.calls, 2);

    assert_eq!(stats.by_operation_7d.len(), 2);
    // Sorted by cost DESC.
    assert_eq!(stats.by_operation_7d[0].0, "summarize");
    assert_eq!(stats.by_operation_7d[1].0, "ask_question");
    assert_eq!(stats.by_operation_7d[1].1.cost_display, "$0.0002");

    let none = t.db.get_cost_stats(None).unwrap();
    assert!(none.this_instance.is_none());
}

#[test]
fn telemetry_cleanup_old() {
    let t = test_db();
    t.db.record_telemetry("i", "recent", None, None, &LlmUsage::default())
        .unwrap();
    // Insert an old event directly.
    let conn = t.db.connection().unwrap();
    conn.execute(
        "INSERT INTO telemetry (instance_id, timestamp, operation)
         VALUES ('i', '2020-01-01T00:00:00+00:00', 'ancient')",
        [],
    )
    .unwrap();
    drop(conn);

    assert_eq!(t.db.export_telemetry(None).unwrap().len(), 2);
    t.db.cleanup_old_telemetry(365).unwrap();
    let events = t.db.export_telemetry(None).unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].operation, "recent");
}

// ── suggest_similar_paths ──────────────────────────────────────────────

#[test]
fn sequence_ratio_matches_goldens() {
    // Pinned similarity ratios.
    assert!((sequence_ratio("abcdef", "abdef") - 0.909_090_909_090_909_1).abs() < 1e-12);
    assert_eq!(sequence_ratio("", ""), 1.0);
    assert_eq!(
        sequence_ratio("/home/user/webapp", "/home/user/webapp"),
        1.0
    );
}

#[test]
fn suggest_similar_paths_goldens() {
    let t = test_db();
    let known = [
        "/home/user/webapp",
        "/home/user/web-app-2",
        "/home/cbird/repos/session-explorer",
        "/tmp/xyz",
    ];
    for (i, p) in known.iter().enumerate() {
        let mut m = make_session(&format!("proj{i}"));
        m.project_path = Some(p.to_string());
        m.modified_at = Some(dt("2025-01-01T10:00:00+00:00"));
        t.db.upsert_session(&m).unwrap();
    }

    // Pinned ratios: 1.0, 0.9729…, 0.4230…, 0.2962….
    let result = t.db.suggest_similar_paths("/home/user/web_app", 3).unwrap();
    assert_eq!(result, vec!["/home/user/webapp", "/home/user/web-app-2"]);

    let result =
        t.db.suggest_similar_paths("/home/cbird/repos/session_explorer", 3)
            .unwrap();
    assert_eq!(result, vec!["/home/cbird/repos/session-explorer"]);

    // No known paths → empty.
    let empty = test_db();
    assert!(
        empty
            .db
            .suggest_similar_paths("/anything", 3)
            .unwrap()
            .is_empty()
    );
}

// ── serialization helpers ──────────────────────────────────────────────

#[test]
fn json_dumps_matches_goldens() {
    // Pinned output: ASCII-only escaping, ", " / ": " separators. Non-ASCII
    // characters become \uXXXX escapes.
    assert_eq!(
        json_dumps(&json!(["a", "b", "naïve"])),
        "[\"a\", \"b\", \"na\\u00efve\"]"
    );
    assert_eq!(
        json_dumps(&json!({"k": 1, "f": true, "s": "x\ny"})),
        "{\"k\": 1, \"f\": true, \"s\": \"x\\ny\"}"
    );
    // Astral plane → surrogate pair.
    assert_eq!(json_dumps(&json!("🎉")), "\"\\ud83c\\udf89\"");
}

#[test]
fn float_str_is_pinned() {
    assert_eq!(super::float_str(1.0), "1.0");
    assert_eq!(super::float_str(0.5), "0.5");
    assert_eq!(super::float_str(24.0), "24.0");
}

// ── behavioral edge cases ──────────────────────────────────────────────

/// `foreign_keys=ON` is set only on the schema-creation connection, and that
/// pragma is per-connection — so no FK is ever enforced during normal
/// operation and an orphan `session_files` row is legal.
#[test]
fn add_file_reference_with_absent_parent_succeeds() {
    let t = test_db();
    t.db.add_file_reference("nonexistent-session", "/orphan.py", "read")
        .expect("allowed: foreign_keys is never ON per-operation");
    let files = t.db.get_session_files("nonexistent-session").unwrap();
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].file_path, "/orphan.py");
    assert_eq!(files[0].operation, "read");
    // And it really is an orphan — no such session row exists.
    assert!(t.db.get_session("nonexistent-session").unwrap().is_none());
}

/// `clear_session_files` removes only the target session's rows.
#[test]
fn clear_session_files_removes_only_that_session() {
    let t = test_db();
    t.db.upsert_session_with_files(
        &make_session("cs1"),
        &[
            ("/a.py".to_string(), "read".to_string()),
            ("/b.py".to_string(), "write".to_string()),
        ],
    )
    .unwrap();
    t.db.upsert_session_with_files(
        &make_session("cs2"),
        &[("/c.py".to_string(), "read".to_string())],
    )
    .unwrap();

    t.db.clear_session_files("cs1").unwrap();
    assert!(t.db.get_session_files("cs1").unwrap().is_empty());
    assert_eq!(t.db.get_session_files("cs2").unwrap().len(), 1);
    // Clearing an unknown session is a no-op, not an error.
    t.db.clear_session_files("no-such").unwrap();
}

/// `index_transcript` must leave exactly ONE `transcripts_fts` row
/// per session, sequentially...
#[test]
fn index_transcript_twice_leaves_one_fts_row() {
    let t = test_db();
    t.db.upsert_session(&make_session("atom1")).unwrap();
    t.db.index_transcript("atom1", "first body text").unwrap();
    t.db.index_transcript("atom1", "second body text").unwrap();
    assert_eq!(fts_row_count(&t.db, "atom1"), 1);
    assert_eq!(
        t.db.get_transcript_text("atom1").unwrap().as_deref(),
        Some("second body text")
    );
}

/// ...and under real concurrency (the DELETE+INSERT pair is one transaction,
/// so a second writer cannot interleave between them).
#[test]
fn index_transcript_concurrent_writers_leave_one_fts_row() {
    let t = test_db();
    t.db.upsert_session(&make_session("atom2")).unwrap();

    let barrier = std::sync::Arc::new(std::sync::Barrier::new(4));
    let handles: Vec<_> = (0..4)
        .map(|n| {
            let db = t.db.clone();
            let barrier = std::sync::Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                for i in 0..25 {
                    db.index_transcript("atom2", &format!("body from thread {n} round {i}"))
                        .expect("index_transcript");
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("thread");
    }
    assert_eq!(fts_row_count(&t.db, "atom2"), 1);
}

fn fts_row_count(db: &Database, session_id: &str) -> i64 {
    let conn = db.connection().unwrap();
    conn.query_row(
        "SELECT COUNT(*) FROM transcripts_fts WHERE session_id = ?",
        [session_id],
        |r| r.get(0),
    )
    .unwrap()
}

/// The result is keyed by session id, so duplicate ids collapse to a single
/// entry: `['m1','m1','m2']` yields two entries, not three.
#[test]
fn get_files_for_sessions_duplicate_ids_collapse() {
    let t = test_db();
    t.db.upsert_session_with_files(
        &make_session("m1"),
        &[
            ("/a.py".to_string(), "read".to_string()),
            ("/b.py".to_string(), "write".to_string()),
        ],
    )
    .unwrap();
    t.db.upsert_session_with_files(
        &make_session("m2"),
        &[("/c.py".to_string(), "read".to_string())],
    )
    .unwrap();

    let result =
        t.db.get_files_for_sessions(&["m1".to_string(), "m1".to_string()])
            .unwrap();
    assert_eq!(result.len(), 1, "duplicate ids collapse to one entry");
    assert_eq!(result[0].0, "m1");
    assert_eq!(result[0].1.len(), 2, "the single entry carries both files");

    // First-occurrence order is preserved across a mixed duplicate list.
    let result =
        t.db.get_files_for_sessions(&[
            "m2".to_string(),
            "m1".to_string(),
            "m2".to_string(),
            "m3".to_string(),
            "m1".to_string(),
        ])
        .unwrap();
    let ids: Vec<&str> = result.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(ids, vec!["m2", "m1", "m3"]);
    assert_eq!(result[0].1.len(), 1);
    assert_eq!(result[1].1.len(), 2);
    assert!(result[2].1.is_empty());
}

/// Query tokenization also splits on U+001C..U+001F, which Rust's
/// `char::is_whitespace` excludes. The full 0..0x100 split set is
/// {9,10,11,12,13,28,29,30,31,32,133,160}.
#[test]
fn prepare_fts_query_splits_on_c1_separators() {
    assert_eq!(prepare_fts_query("a\u{1c}b"), "a b");
    assert_eq!(prepare_fts_query("a\u{1d}b"), "a b");
    assert_eq!(prepare_fts_query("a\u{1e}b"), "a b");
    assert_eq!(prepare_fts_query("a\u{1f}b"), "a b");
    // Also the White_Space members Rust already handles.
    assert_eq!(prepare_fts_query("a\u{b}b\u{85}c\u{a0}d"), "a b c d");
    assert_eq!(prepare_fts_query("a\u{3000}b"), "a b");
    // U+200B (ZWSP) is NOT whitespace in either language — one token, and
    // it is outside [\w*], so it gets quoted.
    assert_eq!(prepare_fts_query("a\u{200b}b"), "\"a\u{200b}b\"");
    // Leading/trailing/repeated separators produce no empty tokens.
    assert_eq!(prepare_fts_query("\u{1c}\u{1c} a \u{1f}"), "a");
}

/// POSIX keeps a doubled leading slash as the root component but collapses
/// three or more: `"//foo/bar"` splits as `("//","foo","bar")` while
/// `"///foo/bar"` splits as `("/","foo","bar")`.
#[test]
fn path_signature_keeps_the_double_slash_quirk() {
    // Pinned signatures.
    assert_eq!(path_signature("/home/user/my-proj"), "//home/user/myproj");
    assert_eq!(path_signature("//foo/bar"), "///foo/bar");
    assert_eq!(path_signature("///foo/bar"), "//foo/bar");
    assert_eq!(path_signature("////foo/bar"), "//foo/bar");
    assert_eq!(path_signature("/"), "//");
    assert_eq!(path_signature("//"), "///");
    assert_eq!(path_signature("foo"), "/foo");
    assert_eq!(path_signature(""), "");
    assert_eq!(path_signature("a/b_c"), "a/bc");
    // '.' dropped, duplicate/trailing slashes collapsed, '..' kept.
    assert_eq!(path_signature("./a/./b"), "a/b");
    assert_eq!(path_signature("a//b/"), "a/b");
    assert_eq!(path_signature("../a"), "../a");
}

/// The limit-zero split. `get_sessions` applies any `Some` limit (so
/// `Some(0)` → `LIMIT 0` → empty), while every other query treats `Some(0)`
/// as no limit at all (no LIMIT clause → all rows).
#[test]
fn limit_zero_truthiness_split() {
    let t = populated_db();
    // `is not None` → LIMIT 0 → no rows.
    assert!(
        t.db.get_sessions(None, None, Some(0), 0)
            .unwrap()
            .is_empty()
    );
    // `if limit:` → 0 is falsy → clause omitted → all 5 non-empty sessions.
    assert_eq!(
        t.db.get_sessions_needing_summary(Some(0), true)
            .unwrap()
            .len(),
        5
    );
    assert_eq!(
        t.db.get_sessions_needing_snapshot(Some(0), true, None, false)
            .unwrap()
            .len(),
        5
    );
    assert_eq!(
        t.db.get_all_sessions_with_transcripts(Some(0))
            .unwrap()
            .len(),
        5
    );
    assert_eq!(
        t.db.get_sessions_needing_transcript_index(Some(0))
            .unwrap()
            .len(),
        5
    );
    assert_eq!(
        t.db.get_sessions_needing_summary_stable(1.0, Some(0))
            .unwrap()
            .len(),
        5
    );
}

/// An empty `cli_source` / `project_path` filter behaves exactly like
/// `None`, and that composes with `offset`.
#[test]
fn get_sessions_empty_string_filters_are_falsy_with_offset() {
    let t = populated_db();
    let all = t.db.get_sessions(None, None, None, 0).unwrap();
    let empty_filters = t.db.get_sessions(Some(""), Some(""), None, 0).unwrap();
    assert_eq!(empty_filters, all);

    // Combined with offset (offset itself is truthiness-gated: 0 = no clause).
    let offset_none = t.db.get_sessions(None, None, None, 2).unwrap();
    let offset_empty = t.db.get_sessions(Some(""), Some(""), None, 2).unwrap();
    assert_eq!(offset_empty, offset_none);
    assert_eq!(offset_empty.len(), all.len() - 2);

    // Some("") is not treated as a real filter that would match nothing.
    assert!(
        !t.db
            .get_sessions(Some(""), None, Some(3), 1)
            .unwrap()
            .is_empty()
    );
}

/// `indexed_at` serialization. INSERT writes local wall-clock time (naive,
/// `T` separator, no offset); the ON CONFLICT path writes SQL
/// `CURRENT_TIMESTAMP` (UTC, space-separated, no fractional seconds).
#[test]
fn indexed_at_insert_vs_conflict_serialization() {
    let t = test_db();
    t.db.upsert_session(&make_session("ia1")).unwrap();
    let inserted =
        t.db.get_session("ia1")
            .unwrap()
            .unwrap()
            .indexed_at
            .unwrap();
    assert!(
        inserted.contains('T') && !inserted.contains(' '),
        "insert path is naive local time: {inserted}"
    );
    assert!(
        !inserted.contains('+') && !inserted[11..].contains('-'),
        "naive: no UTC offset suffix: {inserted}"
    );

    t.db.upsert_session(&make_session("ia1")).unwrap();
    let updated =
        t.db.get_session("ia1")
            .unwrap()
            .unwrap()
            .indexed_at
            .unwrap();
    assert!(
        !updated.contains('T') && updated.contains(' '),
        "conflict path is CURRENT_TIMESTAMP: {updated}"
    );
    assert_eq!(updated.len(), "2025-01-01 00:00:00".len(), "got {updated}");

    // Same split for the with-files variant.
    let t2 = test_db();
    t2.db
        .upsert_session_with_files(&make_session("ia2"), &[])
        .unwrap();
    let inserted = t2
        .db
        .get_session("ia2")
        .unwrap()
        .unwrap()
        .indexed_at
        .unwrap();
    assert!(inserted.contains('T'), "{inserted}");
    t2.db
        .upsert_session_with_files(&make_session("ia2"), &[])
        .unwrap();
    let updated = t2
        .db
        .get_session("ia2")
        .unwrap()
        .unwrap()
        .indexed_at
        .unwrap();
    assert!(updated.contains(' ') && !updated.contains('T'), "{updated}");
}

/// The candidate list is truncated to `n` *before* the ratio filter runs, so
/// `n` caps the result even when more candidates clear the threshold.
#[test]
fn suggest_similar_paths_slices_before_filtering() {
    let t = test_db();
    for (i, p) in ["/home/user/webapp", "/home/user/web-app-2", "/tmp/xyz"]
        .iter()
        .enumerate()
    {
        let mut m = make_session(&format!("sl{i}"));
        m.project_path = Some(p.to_string());
        m.modified_at = Some(dt("2025-01-01T10:00:00+00:00"));
        t.db.upsert_session(&m).unwrap();
    }
    // Ratios: webapp 1.0, web-app-2 0.9729…, xyz 0.2962…
    assert_eq!(
        t.db.suggest_similar_paths("/home/user/web_app", 2).unwrap(),
        vec!["/home/user/webapp", "/home/user/web-app-2"]
    );
    // n=1 drops the second >0.5 candidate.
    assert_eq!(
        t.db.suggest_similar_paths("/home/user/web_app", 1).unwrap(),
        vec!["/home/user/webapp"]
    );
    // n=0 → empty slice → empty result.
    assert!(
        t.db.suggest_similar_paths("/home/user/web_app", 0)
            .unwrap()
            .is_empty()
    );
    // n larger than the candidate set still filters out ratio <= 0.5.
    assert_eq!(
        t.db.suggest_similar_paths("/home/user/web_app", 10)
            .unwrap()
            .len(),
        2
    );
}

/// `get_projects` groups by `(project_path, cli_source)`, so the same path
/// recorded under two CLIs appears twice in `known_paths` — and the
/// suggestion list does not deduplicate, so it repeats the path.
#[test]
fn suggest_similar_paths_repeats_path_present_under_two_cli_sources() {
    let t = test_db();
    for (i, cli) in ["claude_code", "github_copilot"].iter().enumerate() {
        let mut m = SessionMetadata::new(format!("dup{i}"), *cli, format!("/data/dup{i}.jsonl"));
        m.message_count = 5;
        m.project_path = Some("/home/user/webapp".to_string());
        m.modified_at = Some(dt("2025-01-01T10:00:00+00:00"));
        t.db.upsert_session(&m).unwrap();
    }
    assert_eq!(
        t.db.get_projects().unwrap().len(),
        2,
        "grouped per cli_source"
    );
    assert_eq!(
        t.db.suggest_similar_paths("/home/user/web_app", 3).unwrap(),
        vec!["/home/user/webapp", "/home/user/webapp"]
    );
}

/// `rebuild_fts` restores the external-content sessions index after
/// it has been emptied out of band.
#[test]
fn rebuild_fts_restores_sessions_index() {
    let t = populated_db();
    assert!(
        !t.db
            .search_keyword("authentication", 20)
            .unwrap()
            .is_empty()
    );

    let conn = t.db.connection().unwrap();
    conn.execute(
        "INSERT INTO sessions_fts(sessions_fts) VALUES('delete-all')",
        [],
    )
    .unwrap();
    drop(conn);
    assert!(
        t.db.search_keyword("authentication", 20)
            .unwrap()
            .is_empty()
    );

    t.db.rebuild_fts().unwrap();
    assert!(
        !t.db
            .search_keyword("authentication", 20)
            .unwrap()
            .is_empty()
    );
}

/// `get_all_sessions_with_transcripts` filters empty sessions and honors a
/// nonzero limit.
#[test]
fn get_all_sessions_with_transcripts_filters_and_limits() {
    let t = populated_db();
    let all = t.db.get_all_sessions_with_transcripts(None).unwrap();
    assert_eq!(all.len(), 5, "s6 has message_count 0");
    assert!(all.iter().all(|s| s.session_id != "s6"));
    assert_eq!(all[0].session_id, "s1", "newest first");
    assert_eq!(
        t.db.get_all_sessions_with_transcripts(Some(2))
            .unwrap()
            .len(),
        2
    );

    // A session with an empty transcript_path is excluded too.
    let mut m = make_session("nopath");
    m.transcript_path = String::new();
    t.db.upsert_session(&m).unwrap();
    let all = t.db.get_all_sessions_with_transcripts(None).unwrap();
    assert!(all.iter().all(|s| s.session_id != "nopath"));
}

/// The `_stable` variant honors a nonzero limit (and keeps its
/// `ORDER BY modified_at DESC`, so the limit takes the newest rows).
#[test]
fn needing_summary_stable_with_limit() {
    let t = populated_db();
    let all = t.db.get_sessions_needing_summary_stable(1.0, None).unwrap();
    assert_eq!(all.len(), 5);
    let limited =
        t.db.get_sessions_needing_summary_stable(1.0, Some(2))
            .unwrap();
    assert_eq!(limited.len(), 2);
    let ids: Vec<&str> = limited.iter().map(|s| s.session_id.as_str()).collect();
    assert_eq!(ids, vec!["s1", "s2"], "newest first");
    // Also pins get_sessions_with_summaries' limit branch.
    t.db.update_summary("s1", "T", "S").unwrap();
    t.db.update_summary("s2", "T", "S").unwrap();
    assert_eq!(t.db.get_sessions_with_summaries(Some(1)).unwrap().len(), 1);
    assert_eq!(t.db.get_sessions_with_summaries(Some(0)).unwrap().len(), 2);
}

/// `search_transcripts` respects its LIMIT, and
/// `get_sessions_needing_transcript_index` respects a nonzero limit.
#[test]
fn search_transcripts_and_needing_index_respect_limit() {
    let t = test_db();
    for i in 0..3 {
        let mut m = make_session(&format!("lim{i}"));
        m.modified_at = Some(dt(&format!("2025-01-0{}T10:00:00+00:00", i + 1)));
        t.db.upsert_session(&m).unwrap();
        t.db.index_transcript(&format!("lim{i}"), "shared authentication token text")
            .unwrap();
    }
    assert_eq!(
        t.db.search_transcripts("authentication", 20).unwrap().len(),
        3
    );
    assert_eq!(
        t.db.search_transcripts("authentication", 2).unwrap().len(),
        2
    );
    assert!(
        t.db.search_transcripts("authentication", 0)
            .unwrap()
            .is_empty()
    );

    let conn = t.db.connection().unwrap();
    conn.execute("UPDATE sessions SET transcript_indexed_at = NULL", [])
        .unwrap();
    drop(conn);
    assert_eq!(
        t.db.get_sessions_needing_transcript_index(None)
            .unwrap()
            .len(),
        3
    );
    assert_eq!(
        t.db.get_sessions_needing_transcript_index(Some(1))
            .unwrap()
            .len(),
        1
    );
}

/// `get_telemetry_summary(Some(days))` applies the time filter to
/// both the per-operation aggregate and the totals row.
#[test]
fn telemetry_summary_days_filter() {
    let t = test_db();
    t.db.record_telemetry("i", "recent_op", Some(10), None, &LlmUsage::default())
        .unwrap();
    let conn = t.db.connection().unwrap();
    conn.execute(
        "INSERT INTO telemetry (instance_id, timestamp, operation, duration_ms)
         VALUES ('i', '2020-01-01T00:00:00+00:00', 'ancient_op', 5)",
        [],
    )
    .unwrap();
    drop(conn);

    let all = t.db.get_telemetry_summary(None).unwrap();
    assert_eq!(all.total_events, 2);
    assert_eq!(all.operations.len(), 2);

    let recent = t.db.get_telemetry_summary(Some(7)).unwrap();
    assert_eq!(recent.total_events, 1);
    assert_eq!(recent.operations.len(), 1);
    assert_eq!(recent.operations[0].0, "recent_op");
    assert_eq!(recent.operations[0].1.avg_duration_ms, Some(10));

    // days = 0 means no filter at all.
    assert_eq!(t.db.get_telemetry_summary(Some(0)).unwrap().total_events, 2);
    assert_eq!(t.db.export_telemetry(Some(0)).unwrap().len(), 2);
}

/// A stored empty `safe_args` string is left unparsed and stays `None`,
/// rather than surfacing as a JSON error.
#[test]
fn export_telemetry_empty_safe_args_is_falsy() {
    let t = test_db();
    let conn = t.db.connection().unwrap();
    conn.execute(
        "INSERT INTO telemetry (instance_id, timestamp, operation, safe_args)
         VALUES ('i', '2099-01-01T00:00:00+00:00', 'empty_args', '')",
        [],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO telemetry (instance_id, timestamp, operation, safe_args)
         VALUES ('i', '2099-01-02T00:00:00+00:00', 'null_args', NULL)",
        [],
    )
    .unwrap();
    drop(conn);

    let events = t.db.export_telemetry(None).unwrap();
    assert_eq!(events.len(), 2);
    assert!(
        events.iter().all(|e| e.safe_args.is_none()),
        "'' and NULL both yield None, with no JSON error"
    );
}

/// A failing file insert rolls the whole `upsert_session_with_files` back —
/// the pre-existing session row and its files are untouched. Forced here
/// with a `RAISE(ABORT)` trigger on `session_files`.
#[test]
fn upsert_with_files_rolls_back_on_file_insert_failure() {
    let t = test_db();
    let mut m = make_session("rb1");
    m.message_count = 5;
    t.db.upsert_session_with_files(&m, &[("/ok.py".to_string(), "read".to_string())])
        .unwrap();

    let conn = t.db.connection().unwrap();
    conn.execute_batch(
        "CREATE TRIGGER boom BEFORE INSERT ON session_files
         WHEN NEW.file_path = 'BOOM'
         BEGIN SELECT RAISE(ABORT, 'boom'); END;",
    )
    .unwrap();
    drop(conn);

    let mut bad = make_session("rb1");
    bad.message_count = 999;
    let err = t.db.upsert_session_with_files(
        &bad,
        &[
            ("/new.py".to_string(), "read".to_string()),
            ("BOOM".to_string(), "read".to_string()),
        ],
    );
    assert!(
        err.is_err(),
        "the aborting trigger must surface as an error"
    );

    let row = t.db.get_session("rb1").unwrap().unwrap();
    assert_eq!(row.message_count, 5, "session row rolled back");
    let files = t.db.get_session_files("rb1").unwrap();
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].file_path, "/ok.py", "files rolled back");
}

/// Claim contention from two real threads — exactly one winner.
#[test]
fn claims_contention_across_threads() {
    let t = test_db();
    for round in 0..20 {
        let hash = format!("hash-{round}");
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let handles: Vec<_> = (0..2)
            .map(|n| {
                let db = t.db.clone();
                let hash = hash.clone();
                let barrier = std::sync::Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    db.try_claim_summary("contended", &hash, &format!("inst-{n}"), 600)
                        .expect("try_claim_summary")
                })
            })
            .collect();
        let wins: Vec<bool> = handles
            .into_iter()
            .map(|h| h.join().expect("thread"))
            .collect();
        assert_eq!(
            wins.iter().filter(|w| **w).count(),
            1,
            "round {round}: exactly one winner, got {wins:?}"
        );
        assert!(wins.iter().any(|w| !*w), "round {round}: one loser");
    }
}
