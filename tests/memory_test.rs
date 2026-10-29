//! Integration tests for the LLM-facing paths of `memory/`
//! (`search_sessions` in semantic/hybrid mode, and `ask_question`'s two-tier
//! pipeline) with the LLM deliberately unreachable.
//!
//! Every LLM call is wrapped so a missing or unusable model degrades instead
//! of failing the tool: semantic search returns no results and `ask_question`
//! falls back to routing the first five sessions into the transcript tier.
//! These tests pin that degradation.
//!
//! Like `search_test.rs`, [`isolate_llm`] pins `EKKO_MODEL` to a name no
//! provider claims, so `LlmClient::from_config` fails with `UnknownProvider`
//! before any socket is opened — no test here can reach the network or spend
//! money, whatever the user's config says.

use chrono::{DateTime, FixedOffset, Utc};
use ekko::db::Database;
use ekko::memory::{AskArgs, SearchArgs, EkkoMemory};
use ekko::models::SessionMetadata;
use serde_json::json;
use tempfile::TempDir;

struct TestMem {
    mem: EkkoMemory,
    _dir: TempDir,
}

fn isolate_llm() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // SAFETY: no other thread has touched the environment yet — every
        // test blocks on this `Once` before doing any work.
        unsafe {
            std::env::set_var("EKKO_MODEL", "ekko-test-no-such-provider");
            // The query role has its own settings in the user's config.json;
            // pin those (and the backends) so no LLM call can escape.
            std::env::set_var("EKKO_BACKEND", "api");
            std::env::set_var("EKKO_QUERY_BACKEND", "api");
            std::env::set_var("EKKO_QUERY_MODEL", "ekko-test-no-such-provider");
        }
    });
}

fn days_ago(days: i64) -> DateTime<FixedOffset> {
    (Utc::now() - chrono::Duration::days(days)).fixed_offset()
}

fn populated() -> TestMem {
    isolate_llm();
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::new(dir.path().join("mem.db")).expect("db");

    let specs = [
        ("m1", "Fix auth bug", "/home/user/webapp", 1_i64, 10_i64),
        ("m2", "Add dashboard", "/home/user/webapp", 3, 25),
        ("m3", "Refactor DB", "/home/user/backend", 5, 15),
    ];
    for (sid, title, project, days, msgs) in specs {
        let mut s = SessionMetadata::new(sid, "claude_code", format!("/data/{sid}.jsonl"));
        s.title = Some(title.to_string());
        s.first_prompt = Some(format!("Prompt for {sid}"));
        s.project_path = Some(project.to_string());
        s.created_at = Some(days_ago(days));
        s.modified_at = Some(days_ago(days));
        s.message_count = msgs;
        db.upsert_session(&s).expect("upsert");
        db.update_summary(sid, title, &format!("Summary of {title}"))
            .expect("summary");
    }

    TestMem {
        mem: EkkoMemory::new(db),
        _dir: dir,
    }
}

#[tokio::test]
async fn semantic_mode_degrades_to_no_results_without_an_llm() {
    let t = populated();
    let mut args = SearchArgs::new("authentication");
    args.mode = "semantic";
    let r = t.mem.search_sessions(&args).await.expect("search");
    // `semantic_search` swallows the failure and returns [].
    assert_eq!(r["results"], json!([]));
    assert!(r["usage"].is_object());
}

#[tokio::test]
async fn hybrid_mode_reports_tiers_and_formats_results() {
    let t = populated();
    let mut args = SearchArgs::new("auth");
    args.mode = "hybrid";
    let r = t.mem.search_sessions(&args).await.expect("search");

    let tiers: Vec<&str> = r["tiers_used"]
        .as_array()
        .expect("tiers_used")
        .iter()
        .map(|v| v.as_str().expect("tier"))
        .collect();
    // Fewer than MIN_RESULTS hits, so the cascade runs all three tiers.
    assert_eq!(tiers, vec!["keyword", "transcript", "semantic"]);

    let results = r["results"].as_array().expect("results");
    assert!(!results.is_empty(), "the keyword tier matches 'auth'");
    let first = &results[0];
    assert_eq!(first["session_id"], json!("m1"));
    for key in [
        "session_id",
        "title",
        "summary",
        "project_path",
        "git_branch",
        "cli_source",
        "modified_at",
        "message_count",
        "score",
        "relevance",
        "transcript_path",
    ] {
        assert!(first.get(key).is_some(), "missing key {key}");
    }
}

#[tokio::test]
async fn ask_question_falls_through_to_the_transcript_tier() {
    let t = populated();
    let r = t
        .mem
        .ask_question(&AskArgs::new("Why did we choose Redis?"))
        .await
        .expect("ask");

    // Tier 1 fails → the first five sessions are routed to tier 2; every
    // transcript path is missing, so no answer survives.
    assert_eq!(r["tier_used"], json!("transcript"));
    assert_eq!(
        r["answer"],
        json!("I found relevant sessions but couldn't extract a useful answer.")
    );
    let sources = r["sources"].as_array().expect("sources");
    assert_eq!(sources.len(), 3, "at most the first three results");
    assert_eq!(
        sources[0].as_object().expect("source").keys().count(),
        1,
        "the low-confidence branch reports session ids only"
    );
    assert_eq!(
        r["pagination"],
        json!({"offset": 0, "returned": 3, "total": 3, "has_more": false})
    );
    assert!(r["usage"].is_object());
}

#[tokio::test]
async fn ask_question_scopes_by_directory() {
    let t = populated();
    let r = t
        .mem
        .ask_question(&AskArgs {
            directory: Some("/home/user/backend"),
            ..AskArgs::new("What changed?")
        })
        .await
        .expect("ask");
    assert_eq!(r["pagination"]["total"], json!(1));
    let sources = r["sources"].as_array().expect("sources");
    assert_eq!(sources[0]["session_id"], json!("m3"));
}
