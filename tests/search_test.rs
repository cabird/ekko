//! Integration tests for `search.rs`, driven against the six-session fixture
//! built below.
//!
//! Semantic search is never exercised here — it needs an LLM — but the
//! hybrid cascade is driven to the point where tier 3 fires, and the test
//! asserts the tier list and dedup/rerank behaviour around it.
//!
//! Tier 3 builds its client from the process-wide config, which on a real
//! machine can carry the user's API key. [`isolate_llm`] pins `EKKO_MODEL`
//! to a provider-less name so `LlmClient::from_config` fails locally with
//! `UnknownProvider` before any socket is opened: no test here can reach the
//! network or spend money, whatever `~/.local/share/ekko/config.json` and
//! the environment say.

use chrono::{DateTime, FixedOffset};
use ekko::db::Database;
use ekko::models::SessionMetadata;
use ekko::search::{MatchTier, SearchEngine};
use tempfile::TempDir;

struct TestDb {
    db: Database,
    _dir: TempDir,
}

fn dt(s: &str) -> DateTime<FixedOffset> {
    DateTime::parse_from_rfc3339(s).expect("valid datetime")
}

/// `(session_id, cli_source, title, first_prompt, project_path,
/// message_count, modified_at, summary)`.
type SessionSpec = (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    i64,
    &'static str,
    &'static str,
);

/// Pin the LLM model to a name no provider claims, before anything in this
/// binary can initialize the config singleton. Every test goes through
/// `populated_db()` first, and `Once` makes the ordering total, so the
/// semantic tier can only ever fail closed.
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

/// Shared fixture: 6 sessions with summaries.
fn populated_db() -> TestDb {
    isolate_llm();
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::new(dir.path().join("test.db")).expect("create db");

    let specs: &[SessionSpec] = &[
        (
            "s1",
            "claude_code",
            "Fix authentication bug",
            "The login authentication is broken",
            "/home/user/webapp",
            10,
            "2025-01-05T10:00:00+00:00",
            "Outcome: repaired the auth redirect loop in AuthProvider.",
        ),
        (
            "s2",
            "claude_code",
            "Add dashboard widgets",
            "Create a new dashboard with widgets",
            "/home/user/webapp",
            8,
            "2025-01-04T10:00:00+00:00",
            "Outcome: dashboard widgets render.",
        ),
        (
            "s3",
            "github_copilot",
            "Refactor database layer",
            "Refactor the session database",
            "/home/user/tool",
            6,
            "2025-01-03T10:00:00+00:00",
            "Outcome: database layer refactored.",
        ),
        (
            "s4",
            "claude_code",
            "Write API tests",
            "Add tests for the API session endpoints",
            "/home/user/webapp/api",
            4,
            "2025-01-02T10:00:00+00:00",
            "Outcome: API tests written for authentication endpoints.",
        ),
        (
            "s5",
            "github_copilot",
            "Update documentation",
            "Update the session README",
            "/home/user/tool",
            2,
            "2025-01-01T10:00:00+00:00",
            "Outcome: docs updated.",
        ),
        (
            "s6",
            "claude_code",
            "Empty session",
            "nothing here",
            "/home/user/webapp",
            0,
            "2024-12-31T10:00:00+00:00",
            "Outcome: nothing happened.",
        ),
    ];
    for (sid, cli, title, prompt, project, count, modified, summary) in specs {
        let mut m = SessionMetadata::new(*sid, *cli, format!("/data/{sid}.jsonl"));
        m.title = Some(title.to_string());
        m.first_prompt = Some(prompt.to_string());
        m.project_path = Some(project.to_string());
        m.message_count = *count;
        m.modified_at = Some(dt(modified));
        m.created_at = Some(dt(modified));
        db.upsert_session(&m).expect("upsert");
        db.update_summary(sid, title, summary).expect("summary");
    }
    TestDb { db, _dir: dir }
}

fn ids(rows: &[ekko::db::KeywordHit]) -> Vec<&str> {
    rows.iter().map(|r| r.session.session_id.as_str()).collect()
}

#[test]
fn keyword_search_finds_matching_sessions() {
    let t = populated_db();
    let engine = SearchEngine::new(&t.db);
    let results = engine
        .keyword_search("authentication", None, None, 20)
        .unwrap();
    assert!(!results.is_empty());
    assert!(ids(&results).contains(&"s1"));
}

#[test]
fn keyword_search_filters_by_project_path_prefix() {
    let t = populated_db();
    let engine = SearchEngine::new(&t.db);
    let results = engine
        .keyword_search("session", Some("/home/user/tool"), None, 20)
        .unwrap();
    assert!(!results.is_empty());
    assert!(
        results
            .iter()
            .all(|r| r.session.project_path.as_deref() == Some("/home/user/tool"))
    );
}

#[test]
fn keyword_search_filters_by_cli_source() {
    let t = populated_db();
    let engine = SearchEngine::new(&t.db);
    let results = engine
        .keyword_search("session", None, Some("github_copilot"), 20)
        .unwrap();
    assert!(!results.is_empty());
    assert!(
        results
            .iter()
            .all(|r| r.session.cli_source == "github_copilot")
    );

    // A filter that excludes everything yields nothing.
    let none = engine
        .keyword_search("authentication", None, Some("github_copilot"), 20)
        .unwrap();
    assert!(none.is_empty());
}

#[test]
fn keyword_search_respects_limit() {
    let t = populated_db();
    let engine = SearchEngine::new(&t.db);
    let results = engine.keyword_search("session", None, None, 2).unwrap();
    assert!(results.len() <= 2);
}

#[test]
fn keyword_search_no_matches() {
    let t = populated_db();
    let engine = SearchEngine::new(&t.db);
    assert!(
        engine
            .keyword_search("nonexistentxyz", None, None, 20)
            .unwrap()
            .is_empty()
    );
}

/// DIVERGENCE pin: a file-ish query no longer routes to `search_by_file`
/// (dropped in ekko) — it goes through FTS like everything else and must
/// not error.
#[test]
fn file_like_query_falls_through_to_keyword_search() {
    let t = populated_db();
    let engine = SearchEngine::new(&t.db);
    let results = engine.keyword_search("auth.py", None, None, 20).unwrap();
    assert!(results.is_empty() || !ids(&results).is_empty());
    let slashed = engine
        .keyword_search("src/auth/AuthProvider.tsx", None, None, 20)
        .unwrap();
    assert!(slashed.is_empty() || !ids(&slashed).is_empty());
}

#[tokio::test]
async fn hybrid_returns_keyword_tier_first() {
    let t = populated_db();
    let engine = SearchEngine::new(&t.db);
    let result = engine
        .search_hybrid("outcome", None, true, None, None, None, 20)
        .await
        .unwrap();
    assert!(!result.results.is_empty());
    assert_eq!(result.tiers_used, vec!["keyword"]);
    assert!(
        result
            .results
            .iter()
            .all(|r| r.match_tier == MatchTier::Summary)
    );
    // Reranked by recency within the tier.
    let order: Vec<&str> = result
        .results
        .iter()
        .map(|r| r.session.session_id.as_str())
        .collect();
    // s1 is the most recently modified, s6 the oldest.
    assert_eq!(order, vec!["s1", "s2", "s3", "s4", "s5", "s6"]);
}

#[tokio::test]
async fn hybrid_dedupes_by_session_id() {
    let t = populated_db();
    let engine = SearchEngine::new(&t.db);
    let result = engine
        .search_hybrid("outcome", None, true, None, None, None, 20)
        .await
        .unwrap();
    let mut seen = std::collections::HashSet::new();
    for r in &result.results {
        assert!(seen.insert(r.session.session_id.clone()), "duplicate hit");
    }
    assert_eq!(result.total_found, result.results.len());
}

#[tokio::test]
async fn hybrid_filters_by_directory() {
    let t = populated_db();
    let engine = SearchEngine::new(&t.db);
    let result = engine
        .search_hybrid(
            "outcome",
            Some("/home/user/tool"),
            true,
            None,
            None,
            None,
            20,
        )
        .await
        .unwrap();
    assert!(
        result
            .results
            .iter()
            .all(|r| r.session.project_path.as_deref() == Some("/home/user/tool"))
    );
    // Only 2 sessions live under that directory, so the cascade runs out of
    // tiers; the semantic tier fails closed (see `isolate_llm`) and adds
    // nothing rather than reaching the network.
    assert_eq!(result.tiers_used, vec!["keyword", "transcript", "semantic"]);
    assert!(
        result
            .results
            .iter()
            .all(|r| r.match_tier != MatchTier::Semantic)
    );
}

#[tokio::test]
async fn hybrid_filters_by_date_range() {
    let t = populated_db();
    let engine = SearchEngine::new(&t.db);
    let result = engine
        .search_hybrid(
            "outcome",
            None,
            true,
            None,
            Some("2025-01-04"),
            Some("2025-01-06"),
            20,
        )
        .await
        .unwrap();
    assert!(!result.results.is_empty());
    for r in &result.results {
        let id = r.session.session_id.as_str();
        assert!(id == "s1" || id == "s2", "unexpected session {id} in range");
    }
    // Two in-range hits is below MIN_RESULTS, so tier 3 runs and fails
    // closed — the date filter is not applied to the semantic tier, so a
    // live LLM here would smuggle out-of-range sessions into the result.
    assert_eq!(result.tiers_used, vec!["keyword", "transcript", "semantic"]);
}

#[tokio::test]
async fn hybrid_cascades_to_transcript_tier_when_short() {
    let t = populated_db();
    // A query that matches nothing in summaries but exists in a transcript.
    t.db.index_transcript("s3", "the exotic flamingo migration notes")
        .unwrap();
    let engine = SearchEngine::new(&t.db);
    let result = engine
        .search_hybrid("flamingo", None, true, None, None, None, 20)
        .await
        .unwrap();
    assert_eq!(result.tiers_used, vec!["keyword", "transcript", "semantic"]);
    assert_eq!(result.results.len(), 1);
    assert_eq!(result.results[0].session.session_id, "s3");
    assert_eq!(result.results[0].match_tier, MatchTier::Transcript);
    assert!(result.results[0].excerpt.is_some());
}
