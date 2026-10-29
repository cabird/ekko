//! Tests for `fastindex.rs` — the `ekko index` foreground driver.
//!
//! # Network / cost isolation
//!
//! Same pattern as `background_test.rs`: `HOME`, `XDG_DATA_HOME`, `APPDATA`
//! and `WIN_HOME` are pinned into a temp directory and the API-key variables
//! are removed before anything reads the config singleton, with `EKKO_MODEL`
//! left at a model whose provider is known (`gpt-5-mini`). `check_llm_available`
//! therefore returns `false`, `run()`'s LLM gate closes, and the summarize and
//! snapshot passes are never reached. No test here can open a socket, and
//! `scan_all` cannot see the machine's real `~/.claude/projects`.
//!
//! The per-item tests exercise the claim → work → release path with a session
//! whose transcript file does not exist, so `summarize_session` fails before
//! any LLM client is constructed.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ekko::background::{CLAIM_TTL_SECONDS, FileLog, summary_claim_key};
use ekko::db::{Database, SessionRow};
use ekko::fastindex::{IndexOptions, ItemOutcome, run, summarize_item};
use tempfile::TempDir;

/// The process-wide fake home. Created exactly once, before any test does
/// work, so the environment mutation is race-free.
fn sandbox() -> &'static Path {
    static SANDBOX: OnceLock<TempDir> = OnceLock::new();
    SANDBOX
        .get_or_init(|| {
            let dir = TempDir::new().expect("tempdir");
            fs::create_dir_all(dir.path().join(".claude/projects")).unwrap();
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

fn claude_projects() -> PathBuf {
    sandbox().join(".claude/projects")
}

fn write_session(project: &str, name: &str, content: &str) -> PathBuf {
    let dir = claude_projects().join(project);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    fs::write(&path, content).unwrap();
    path
}

/// A transcript old enough for the staleness queries, with real tool calls
/// and identifier-shaped words for the keyword extractor.
fn transcript(session: &str) -> String {
    format!(
        "{{\"type\":\"system\",\"subtype\":\"init\",\"sessionId\":\"{session}\",\"cwd\":\"/repo\",\"gitBranch\":\"main\",\"timestamp\":\"2025-06-15T10:00:00Z\"}}\n\
{{\"type\":\"user\",\"timestamp\":\"2025-06-15T10:00:05Z\",\"message\":{{\"role\":\"user\",\"content\":\"Please refactor search_sessions to use ThreadPoolExecutor and sqlite3.connect\"}}}}\n\
{{\"type\":\"assistant\",\"timestamp\":\"2025-06-15T10:00:10Z\",\"message\":{{\"role\":\"assistant\",\"content\":[{{\"type\":\"text\",\"text\":\"On it — reading the file first.\"}},{{\"type\":\"tool_use\",\"id\":\"tc_1\",\"name\":\"Read\",\"input\":{{\"file_path\":\"/repo/search.py\"}}}}]}}}}\n\
{{\"type\":\"assistant\",\"timestamp\":\"2025-06-15T10:00:20Z\",\"message\":{{\"role\":\"assistant\",\"content\":[{{\"type\":\"tool_use\",\"id\":\"tc_2\",\"name\":\"Edit\",\"input\":{{\"file_path\":\"/repo/search.py\",\"old_string\":\"a\",\"new_string\":\"b\"}}}}]}}}}\n"
    )
}

fn temp_db() -> (Database, TempDir) {
    let dir = TempDir::new().unwrap();
    let db = Database::new(dir.path().join("sessions.db")).unwrap();
    (db, dir)
}

/// Fails the test if the environment could actually reach an LLM.
fn assert_llm_gate_closed() {
    let config = ekko::config::get_config();
    let (available, _) = ekko::llm::check_llm_available(config, &config.llm_model);
    assert!(
        !available,
        "test environment leaked an API key — refusing to run something that could call an LLM"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// End-to-end: --no-llm
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn no_llm_run_scans_and_indexes_fts_and_keywords() {
    sandbox();
    write_session("repo-fast-a", "fast-a.jsonl", &transcript("fast-a"));
    write_session("repo-fast-b", "fast-b.jsonl", &transcript("fast-b"));
    assert_llm_gate_closed();

    let (db, _dir) = temp_db();
    let opts = IndexOptions {
        workers: 4,
        no_llm: true,
        ..IndexOptions::default()
    };

    let mut lines: Vec<String> = Vec::new();
    let report = run(&db, &FileLog::none(), &opts, &mut |l| {
        lines.push(l.to_string())
    })
    .await
    .expect("index run");

    // Everything local ran.
    assert!(report.scanned >= 2, "scanned {}", report.scanned);
    assert!(report.fts.indexed >= 2, "fts {:?}", report.fts);
    assert!(
        report.keywords.indexed >= 2,
        "keywords {:?}",
        report.keywords
    );

    // The LLM passes gated off cleanly, for the requested reason.
    assert_eq!(report.llm_skipped.as_deref(), Some("--no-llm"));
    assert_eq!(report.summaries.total, 0);
    assert_eq!(report.snapshots.total, 0);
    assert!(lines.iter().any(|l| l.contains("skipping summaries")));

    // And the database reflects it.
    // `>= 2`: the sandbox home is shared by every test in this binary, so a
    // concurrently running test's fixture may also be on disk by now.
    let sessions = db.get_sessions(None, None, None, 0).unwrap();
    assert!(sessions.len() >= 2, "scanned {} sessions", sessions.len());
    for s in &sessions {
        assert!(s.transcript_indexed_at.is_some(), "no FTS stamp");
        assert!(
            s.keywords
                .as_deref()
                .unwrap_or("")
                .contains("ThreadPoolExecutor"),
            "keywords missing: {:?}",
            s.keywords
        );
        assert!(s.summary.is_none());
        assert!(s.context_snapshot.is_none());
    }

    // Transcript FTS is searchable.
    let hits = db.search_transcripts("ThreadPoolExecutor", 10).unwrap();
    assert!(!hits.is_empty(), "transcript FTS returned nothing");

    // The report line is one line and mentions the wall time.
    let summary = report.render_summary();
    assert!(!summary.contains('\n'));
    assert!(summary.starts_with("done in "));
    // Usage totals come from the global tracker; nothing was spent here.
    assert!(
        summary.ends_with("llm 0 calls, 0 tokens, $0.0000"),
        "unexpected summary line: {summary}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// End-to-end: LLM passes requested but no API key
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn llm_passes_gate_off_when_no_api_key_is_configured() {
    sandbox();
    write_session("repo-fast-c", "fast-c.jsonl", &transcript("fast-c"));
    assert_llm_gate_closed();

    let (db, _dir) = temp_db();
    let opts = IndexOptions {
        workers: 3,
        no_llm: false,
        ..IndexOptions::default()
    };

    let report = run(&db, &FileLog::none(), &opts, &mut |_| {})
        .await
        .expect("index run");

    // Local work still happened; the LLM passes were skipped with the
    // availability message, not with `--no-llm`.
    assert!(report.fts.indexed >= 1);
    let reason = report.llm_skipped.expect("gate should be closed");
    assert!(reason.contains("OPENAI_API_KEY"), "reason: {reason}");
    assert_eq!(report.summaries.total, 0);
    assert_eq!(report.snapshots.total, 0);
}

// ═══════════════════════════════════════════════════════════════════════
// Per-item: claim → work → release
// ═══════════════════════════════════════════════════════════════════════

/// A session whose transcript file is gone fails inside `summarize_one`
/// (before any LLM client exists) — and the item still releases its claim.
#[tokio::test(flavor = "multi_thread")]
async fn a_failing_item_releases_its_claim() {
    sandbox();
    let (db, _dir) = temp_db();
    let session = SessionRow {
        session_id: "missing-transcript".to_string(),
        cli_source: "claude_code".to_string(),
        transcript_path: Some("/nonexistent/nope.jsonl".to_string()),
        source_hash: Some("hash-1".to_string()),
        ..SessionRow::default()
    };

    let outcome = summarize_item(&db, &FileLog::none(), "inst-a", &session).await;
    assert_eq!(outcome, ItemOutcome::Failed);

    // Released: another instance can take the claim immediately, without
    // waiting out the TTL.
    let key = summary_claim_key(&session);
    assert!(
        db.try_claim_summary(&session.session_id, &key, "inst-b", CLAIM_TTL_SECONDS)
            .unwrap(),
        "the failed item did not release its claim"
    );
}

/// A session already claimed by another instance is skipped, not retried —
/// this is what makes `ekko index` safe alongside a running `ekko mcp`.
#[tokio::test(flavor = "multi_thread")]
async fn a_claimed_item_is_skipped() {
    sandbox();
    let (db, _dir) = temp_db();
    let session = SessionRow {
        session_id: "claimed-elsewhere".to_string(),
        cli_source: "claude_code".to_string(),
        transcript_path: Some("/nonexistent/nope.jsonl".to_string()),
        ..SessionRow::default()
    };
    let key = summary_claim_key(&session);
    assert!(
        db.try_claim_summary(&session.session_id, &key, "other-server", CLAIM_TTL_SECONDS)
            .unwrap()
    );

    let outcome = summarize_item(&db, &FileLog::none(), "inst-a", &session).await;
    assert_eq!(outcome, ItemOutcome::Skipped);

    // The other instance still holds it (we did not release someone else's
    // claim on the way out).
    assert!(
        !db.try_claim_summary(&session.session_id, &key, "inst-a", CLAIM_TTL_SECONDS)
            .unwrap()
    );
}
