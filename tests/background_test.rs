//! Tests for `background.rs`.
//!
//! # Network / cost isolation
//!
//! This binary pins `HOME`, `XDG_DATA_HOME` and `APPDATA` into a temp
//! directory and **removes** `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` before
//! anything reads the config singleton, then leaves `EKKO_MODEL` at a model
//! with a known provider (`gpt-5-mini`). The result is that
//! `check_llm_available` returns `false`, the cycle's LLM gate closes, and
//! the summarize/snapshot batches are never reached — no socket can be
//! opened whatever the developer's real `~/.local/share/ekko/config.json`
//! contains. Redirecting `HOME` also guarantees `scan_all` cannot walk the
//! machine's real `~/.claude/projects`.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ekko::background::{
    BackgroundIndexer, CLAIM_TTL_SECONDS, FileLog, snapshot_claim_key, summary_claim_key,
};
use ekko::db::{Database, SessionRow};
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
                // On WSL, `get_session_paths` also probes
                // `/mnt/c/Users/{win_user}`; `WIN_HOME` pins that name to a
                // directory that cannot exist, so the real Windows-side
                // Claude/Copilot session stores stay out of the scan.
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

/// `<sandbox>/.claude/projects` — where `scan_all` will look.
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

/// A transcript whose `modified_at` is old enough for the staleness queries
/// and whose conversation is comfortably longer than 50 characters.
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

// ═══════════════════════════════════════════════════════════════════════
// Constants and pure helpers
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn constants_are_pinned() {
    assert_eq!(ekko::background::STARTUP_DELAY_SECONDS, 60);
    assert_eq!(ekko::background::MIN_AGE_HOURS, 1.0);
    assert_eq!(ekko::background::SCAN_INTERVAL_SECONDS, 300);
    assert_eq!(CLAIM_TTL_SECONDS, 600);
    assert_eq!(ekko::background::BATCH_SIZE, 10);
    assert_eq!(ekko::background::ERROR_BACKOFF_SECONDS, 60);
}

#[test]
fn instance_id_has_pid_and_eight_hex() {
    let id = ekko::background::new_instance_id();
    let (pid, hex) = id.split_once('-').expect("pid-hex shape");
    assert_eq!(pid, std::process::id().to_string());
    assert_eq!(hex.len(), 8);
    assert!(
        hex.chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase())
    );

    // Process-wide id is stable.
    assert_eq!(
        ekko::background::instance_id(),
        ekko::background::instance_id()
    );
}

#[test]
fn claim_keys_skip_empty_values() {
    let mut row = SessionRow {
        session_id: "sess-1".to_string(),
        ..SessionRow::default()
    };
    // No source_hash → session_id.
    assert_eq!(summary_claim_key(&row), "sess-1");
    assert_eq!(snapshot_claim_key(&row), "snapshot:sess-1");

    // An empty source_hash counts as absent, so the key falls back.
    row.source_hash = Some(String::new());
    assert_eq!(summary_claim_key(&row), "sess-1");

    row.source_hash = Some("deadbeef".to_string());
    assert_eq!(summary_claim_key(&row), "deadbeef");
    assert_eq!(snapshot_claim_key(&row), "snapshot:deadbeef");
}

/// Summary and snapshot claims for the same session must not collide.
#[test]
fn summary_and_snapshot_claims_are_independent() {
    let (db, _dir) = temp_db();
    let row = SessionRow {
        session_id: "s1".to_string(),
        source_hash: Some("hash1".to_string()),
        ..SessionRow::default()
    };
    let (sum, snap) = (summary_claim_key(&row), snapshot_claim_key(&row));

    assert!(
        db.try_claim_summary("s1", &sum, "inst-a", CLAIM_TTL_SECONDS)
            .unwrap()
    );
    // Same key, different instance → loses.
    assert!(
        !db.try_claim_summary("s1", &sum, "inst-b", CLAIM_TTL_SECONDS)
            .unwrap()
    );
    // Snapshot key is free.
    assert!(
        db.try_claim_summary("s1", &snap, "inst-b", CLAIM_TTL_SECONDS)
            .unwrap()
    );

    // Release frees only the summary claim.
    db.release_claim("s1", &sum).unwrap();
    assert!(
        db.try_claim_summary("s1", &sum, "inst-b", CLAIM_TTL_SECONDS)
            .unwrap()
    );
    assert!(
        !db.try_claim_summary("s1", &snap, "inst-a", CLAIM_TTL_SECONDS)
            .unwrap()
    );
}

// ═══════════════════════════════════════════════════════════════════════
// One end-to-end cycle, LLM gate closed
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn one_cycle_scans_indexes_and_stops_at_the_llm_gate() {
    let sandbox = sandbox();
    write_session("repo-a", "sess-a.jsonl", &transcript("sess-a"));
    write_session("repo-b", "sess-b.jsonl", &transcript("sess-b"));

    // The gate must actually be closed, or this test could spend money.
    let config = ekko::config::get_config();
    let (available, _) = ekko::llm::check_llm_available(config, &config.llm_model);
    assert!(
        !available,
        "test environment leaked an API key — refusing to run a cycle that could call an LLM"
    );

    let (db, _dir) = temp_db();
    let log_path = sandbox.join("indexer-test.log");
    let mut indexer = BackgroundIndexer::new(db.clone())
        .with_log(FileLog::to_path(&log_path))
        .with_instance_id("test-instance");

    // Cycle 1: scan finds both sessions and the FTS pass indexes them.
    let work_done = indexer.run_cycle().await.expect("cycle");
    assert!(work_done, "indexing two new sessions counts as work");

    // Other tests in this binary write into the same sandbox, so count only
    // this test's two sessions (and one-shot fixtures never appear at all).
    let mine = |s: &ekko::db::SessionRow| s.session_id == "sess-a" || s.session_id == "sess-b";
    let sessions: Vec<ekko::db::SessionRow> = db
        .get_sessions(None, None, None, 0)
        .unwrap()
        .into_iter()
        .filter(mine)
        .collect();
    assert_eq!(sessions.len(), 2);
    assert!(
        db.get_sessions(None, None, None, 0)
            .unwrap()
            .iter()
            .all(|s| !s.session_id.starts_with("sess-p") && s.session_id != "sess-f2"),
        "one-shot transcripts must never be indexed"
    );

    // Sessions were fully indexed: timeline, files and tokens all present.
    for session in &sessions {
        assert!(
            session
                .tool_timeline
                .as_deref()
                .unwrap_or("")
                .starts_with("Tool Calls:\n"),
            "missing tool timeline for {}",
            session.session_id
        );
        assert_eq!(session.files_touched_count, 1);
        assert!(session.token_count > 0);
        assert!(session.transcript_indexed_at.is_some());
        // The gate is closed, so nothing was summarized or snapshotted.
        assert!(session.summary.is_none());
        assert!(session.context_snapshot.is_none());
    }

    // Transcript text really landed in FTS.
    let hits = db.search_transcripts("ThreadPoolExecutor", 10).unwrap();
    assert!(hits.len() >= 2, "{}", hits.len());

    // Cycle 2: nothing left to do (the scan is rate-limited and the FTS
    // queue is empty), so the loop would sleep.
    let work_done = indexer.run_cycle().await.expect("cycle");
    assert!(!work_done);

    // The LLM warning is logged exactly once across cycles — a latch keeps
    // it from repeating every 5 minutes for the life of the process.
    let log = fs::read_to_string(&log_path).unwrap();
    assert_eq!(
        log.matches("Skipping summarization/snapshots").count(),
        1,
        "LLM warning should be logged once, got:\n{log}"
    );
    // The count includes interactive fixtures written by sibling tests, so
    // only the line's shape is asserted.
    assert!(
        log.lines()
            .any(|l| l.contains("[INFO] ekko.background: Indexed ") && l.ends_with(" sessions")),
        "{log}"
    );
    assert!(log.lines().all(|l| l.contains("ekko.background:")));
}

/// A stop signal raised before the loop starts makes it exit without work.
#[tokio::test(flavor = "multi_thread")]
async fn stop_signal_short_circuits_the_startup_delay() {
    sandbox();
    let (db, _dir) = temp_db();
    let handle = ekko::background::start(db.clone());
    // `start` waits STARTUP_DELAY_SECONDS; stopping must not block on it.
    tokio::time::timeout(std::time::Duration::from_secs(5), handle.stop())
        .await
        .expect("stop returned promptly");
    // Nothing was indexed.
    assert_eq!(db.get_stats().unwrap().total_sessions, 0);
}

/// A claim held by another instance makes the cycle skip that session; this
/// exercises the claim/release sequencing without any LLM involvement.
#[tokio::test(flavor = "multi_thread")]
async fn foreign_claims_block_and_expire() {
    sandbox();
    let (db, _dir) = temp_db();
    let row = SessionRow {
        session_id: "s-claim".to_string(),
        ..SessionRow::default()
    };
    let key = summary_claim_key(&row);

    // Another instance holds a live claim.
    assert!(
        db.try_claim_summary("s-claim", &key, "other", CLAIM_TTL_SECONDS)
            .unwrap()
    );
    assert!(
        !db.try_claim_summary("s-claim", &key, "mine", CLAIM_TTL_SECONDS)
            .unwrap()
    );

    // An already-expired claim is takeable, and cleanup removes it.
    assert!(
        db.try_claim_summary("s-expired", &key, "other", -1)
            .unwrap()
    );
    assert!(
        db.try_claim_summary("s-expired", &key, "mine", CLAIM_TTL_SECONDS)
            .unwrap()
    );
    db.release_claim("s-expired", &key).unwrap();
    db.cleanup_expired_claims().unwrap();
}

/// P6 note, enforced here: the summarize write path must pass `Option`
/// through to SQL. A model that emits `"title": null` / `"summary": null`
/// leaves NULL columns, so the session stays in the needs-summary queue —
/// substituting `""` would mark it done with an empty summary.
#[test]
fn null_title_and_summary_reach_sql_null_and_stay_queued() {
    use chrono::{DateTime, FixedOffset};
    let (db, _dir) = temp_db();

    let mut meta = ekko::models::SessionMetadata::new("s-null", "claude_code", "/tmp/t.jsonl");
    meta.message_count = 3;
    let old: DateTime<FixedOffset> = "2020-01-01T00:00:00+00:00".parse().unwrap();
    meta.created_at = Some(old);
    meta.modified_at = Some(old);
    db.upsert_session(&meta).unwrap();

    // Queued to begin with.
    assert_eq!(
        db.get_sessions_needing_summary_stable(1.0, Some(10))
            .unwrap()
            .len(),
        1
    );

    // A present-but-null `title` in the model's response.
    db.update_summary_opt("s-null", None, None).unwrap();
    let row = db.get_session("s-null").unwrap().unwrap();
    assert_eq!(
        row.title, None,
        "title must be SQL NULL, not the empty string"
    );
    assert_eq!(row.summary, None, "summary must be SQL NULL");
    assert!(row.summary_at.is_some(), "summary_at is still stamped");

    // Still queued: a NULL summary leaves the session needing one.
    assert_eq!(
        db.get_sessions_needing_summary_stable(1.0, Some(10))
            .unwrap()
            .len(),
        1
    );

    // A real summary takes it out of the queue.
    db.update_summary_opt("s-null", Some("Title"), Some("Body"))
        .unwrap();
    assert!(
        db.get_sessions_needing_summary_stable(1.0, Some(10))
            .unwrap()
            .is_empty()
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Non-interactive sessions (`claude -p`, Agent SDK) are not indexed
// ═══════════════════════════════════════════════════════════════════════

/// A one-shot transcript as Claude Code persists it for `claude -p`:
/// `"entrypoint":"sdk-cli"` on every record that carries a `cwd`.
fn one_shot_transcript(session: &str) -> String {
    format!(
        "{{\"type\":\"user\",\"entrypoint\":\"sdk-cli\",\"cwd\":\"/repo\",\"sessionId\":\"{session}\",\"timestamp\":\"2025-06-15T10:00:05Z\",\"message\":{{\"role\":\"user\",\"content\":\"Summarize this transcript as JSON, please, with a title and summary field\"}}}}\n\
{{\"type\":\"assistant\",\"entrypoint\":\"sdk-cli\",\"cwd\":\"/repo\",\"sessionId\":\"{session}\",\"timestamp\":\"2025-06-15T10:00:10Z\",\"message\":{{\"role\":\"assistant\",\"content\":[{{\"type\":\"text\",\"text\":\"{{\\\"title\\\":\\\"x\\\"}}\"}}]}}}}\n"
    )
}

#[test]
fn parser_reads_entrypoint_and_is_interactive_classifies_it() {
    use ekko::parsers::{Parser, claude::ClaudeParser};
    let _ = sandbox();
    let interactive = write_session("repo-e", "sess-i.jsonl", &transcript("sess-i"));
    let one_shot = write_session("repo-e", "sess-p.jsonl", &one_shot_transcript("sess-p"));

    let meta = ClaudeParser.extract_metadata(&interactive);
    assert_eq!(
        meta.entrypoint, None,
        "fixture transcripts predate the field"
    );
    assert!(ekko::indexer::is_interactive(&meta));

    let meta = ClaudeParser.extract_metadata(&one_shot);
    assert_eq!(meta.entrypoint.as_deref(), Some("sdk-cli"));
    assert!(!ekko::indexer::is_interactive(&meta));

    let mut desktop = meta.clone();
    desktop.entrypoint = Some("claude-desktop".to_string());
    assert!(ekko::indexer::is_interactive(&desktop));
}

#[test]
fn scan_skips_one_shot_sessions_and_evicts_previously_indexed_ones() {
    let _ = sandbox();
    write_session("repo-f", "sess-f1.jsonl", &transcript("sess-f1"));
    let one_shot = write_session("repo-f", "sess-f2.jsonl", &one_shot_transcript("sess-f2"));
    let (db, _dir) = temp_db();

    // Pretend an older ekko had indexed the one-shot: put a row in by hand.
    let mut stale = ekko::models::SessionMetadata::new(
        "sess-f2",
        "claude_code",
        one_shot.to_string_lossy().into_owned(),
    );
    stale.project_path = Some("/repo".to_string());
    db.upsert_session_with_files(&stale, &[("/repo/x.py".to_string(), "read".to_string())])
        .unwrap();
    db.index_transcript("sess-f2", "some transcript text")
        .unwrap();
    assert!(db.get_session("sess-f2").unwrap().is_some());

    let stats = ekko::indexer::scan_all(&db);
    assert!(stats.total_skipped_non_interactive >= 1, "{stats:?}");

    let ids: Vec<String> = db
        .get_sessions(None, None, None, 0)
        .unwrap()
        .into_iter()
        .map(|s| s.session_id)
        .collect();
    assert!(ids.contains(&"sess-f1".to_string()), "{ids:?}");
    assert!(
        !ids.contains(&"sess-f2".to_string()),
        "one-shot must be evicted: {ids:?}"
    );
    assert!(db.get_transcript_text("sess-f2").unwrap().is_none());

    // Idempotent: a second scan finds nothing to evict and still skips it.
    let stats = ekko::indexer::scan_all(&db);
    assert!(stats.total_skipped_non_interactive >= 1);
    assert!(db.get_session("sess-f2").unwrap().is_none());
}

// ═══════════════════════════════════════════════════════════════════════
// Structured events: stored on scan, stable when unchanged, refreshed on change
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn scan_stores_events_idempotently_and_refreshes_on_change() {
    let _ = sandbox();
    let path = write_session("repo-g", "sess-g1.jsonl", &transcript("sess-g1"));
    let (db, _dir) = temp_db();

    // First scan ingests the events.
    ekko::indexer::scan_all(&db);
    let events = db.get_events("sess-g1").unwrap();
    assert!(events.len() >= 4, "expected parsed events, got {}", events.len());
    assert_eq!(events[0]["type"], "message");
    assert!(
        events.iter().any(|e| e["type"] == "tool_call" && e["tool_name"] == "Read"),
        "tool calls must be stored"
    );
    assert!(events.iter().all(|e| e["truncated"] == false));

    // Unchanged transcript: a rescan must not rewrite the rows. Plant a
    // marker in the stored JSON and prove it survives.
    {
        let conn = rusqlite::Connection::open(db.path()).unwrap();
        conn.execute(
            "UPDATE session_events SET event = '{\"marker\":true}' \
             WHERE session_id = 'sess-g1' AND pos = 0",
            [],
        )
        .unwrap();
    }
    ekko::indexer::scan_all(&db);
    let events = db.get_events("sess-g1").unwrap();
    assert_eq!(events[0]["marker"], true, "unchanged transcript was rewritten");

    // Changed transcript: the marker is replaced and the new event appears.
    let mut content = transcript("sess-g1");
    content.push_str(
        "{\"type\":\"user\",\"timestamp\":\"2025-06-15T10:05:00Z\",\"message\":{\"role\":\"user\",\"content\":\"one more thing about sqlite\"}}\n",
    );
    fs::write(&path, content).unwrap();
    ekko::indexer::scan_all(&db);
    let events = db.get_events("sess-g1").unwrap();
    assert!(events[0].get("marker").is_none(), "changed transcript must re-ingest");
    assert!(
        events
            .iter()
            .any(|e| e["content"].as_str().unwrap_or("").contains("one more thing")),
        "new event missing after re-ingest"
    );
}

#[test]
fn oversized_payloads_are_capped_at_ingest() {
    let _ = sandbox();
    let big = "y".repeat(60_000);
    let content = format!(
        "{{\"type\":\"user\",\"timestamp\":\"2025-06-15T10:00:05Z\",\"message\":{{\"role\":\"user\",\"content\":\"start\"}}}}\n\
{{\"type\":\"assistant\",\"timestamp\":\"2025-06-15T10:00:10Z\",\"message\":{{\"role\":\"assistant\",\"content\":[{{\"type\":\"tool_use\",\"id\":\"tc_1\",\"name\":\"Write\",\"input\":{{\"file_path\":\"/repo/big.txt\",\"content\":\"{big}\"}}}}]}}}}\n"
    );
    write_session("repo-h", "sess-h1.jsonl", &content);
    let (db, _dir) = temp_db();
    ekko::indexer::scan_all(&db);

    let events = db.get_events("sess-h1").unwrap();
    let call = events
        .iter()
        .find(|e| e["type"] == "tool_call")
        .expect("tool call stored");
    let input = call["tool_input"].as_str().expect("oversized input stored as capped text");
    assert_eq!(input.chars().count(), ekko::db::EVENT_TOOL_INPUT_CAP);
    assert_eq!(call["truncated"], true);
    // What reached the DB is what the web serves; nothing near 60K remains.
    let total: usize = events.iter().map(|e| e.to_string().len()).sum();
    assert!(total < 10_000, "stored events unexpectedly large: {total}");
}
