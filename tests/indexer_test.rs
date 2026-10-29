//! Integration tests for `indexer.rs`.
//!
//! The keyword, tool-timeline and file-path expectations in
//! `tests/goldens/keywords.json` and `tests/goldens/indexer.json` are pinned
//! expected output; update them deliberately when the extraction behaviour
//! changes on purpose. Nothing here touches the network or the user's real
//! session directories.

use std::fs;
use std::path::{Path, PathBuf};

use ekko::db::Database;
use ekko::indexer;
use ekko::parsers::{ClaudeParser, Parser};
use serde_json::Value;
use tempfile::TempDir;

fn goldens(name: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/goldens")
        .join(name);
    serde_json::from_str(&fs::read_to_string(&path).expect("golden file")).expect("golden json")
}

/// Write a claude transcript into `<dir>/proj/<name>` and return its path.
fn write_claude(dir: &Path, name: &str, content: &str) -> PathBuf {
    let proj = dir.join("proj");
    fs::create_dir_all(&proj).unwrap();
    let p = proj.join(name);
    fs::write(&p, content).unwrap();
    p
}

fn parse_events(dir: &Path, content: &str) -> Vec<ekko::models::Event> {
    let path = write_claude(dir, "s.jsonl", content);
    ClaudeParser.parse_transcript(&path).events
}

// ═══════════════════════════════════════════════════════════════════════
// Keyword extraction — goldens
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn keyword_extraction_matches_goldens() {
    let g = goldens("keywords.json");
    let cases = [
        "code_discussion",
        "acronyms_and_dots",
        "snake_and_camel",
        "mixed_prose",
        "unicode_and_edges",
        "empty",
        "short",
    ];
    for case in cases {
        let entry = &g[case];
        let text = entry["text"].as_str().unwrap();
        let expected: Vec<String> = entry["keywords"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        let actual = indexer::extract_keywords(text, indexer::MAX_KEYWORDS);
        assert_eq!(actual, expected, "keyword mismatch for golden case {case}");
    }
}

/// The `max_keywords` cap applies after the codepoint sort, not to insertion
/// order.
#[test]
fn keyword_extraction_truncates_after_sorting() {
    let g = goldens("keywords.json");
    let entry = &g["truncate5"];
    let expected: Vec<String> = entry["keywords"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let actual = indexer::extract_keywords(entry["text"].as_str().unwrap(), 5);
    assert_eq!(actual.len(), 5);
    assert_eq!(actual, expected);
}

/// The exclusion set must be exactly the pinned `EXCLUDED_ACRONYMS` list.
#[test]
fn excluded_acronyms_are_pinned() {
    let g = goldens("keywords.json");
    let expected: Vec<&str> = g["_meta"]["excluded_acronyms"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    let mut actual: Vec<&str> = indexer::EXCLUDED_ACRONYMS.to_vec();
    actual.sort_unstable();
    assert_eq!(actual, expected);
    assert_eq!(
        indexer::EXCLUDED_ACRONYMS.len(),
        g["_meta"]["excluded_acronyms_count"].as_u64().unwrap() as usize
    );
    // No duplicates crept in while transcribing the literal.
    let mut deduped = actual.clone();
    deduped.dedup();
    assert_eq!(deduped.len(), actual.len());
}

/// Spot-checks of each pattern's boundary conditions, independent of the
/// golden corpus.
#[test]
fn keyword_pattern_boundaries() {
    // snake_case shorter than 5 characters is dropped, 5 is kept.
    let kw = indexer::extract_keywords("a_cd ab_cd", 50);
    assert_eq!(kw, vec!["ab_cd".to_string()]);
    // Acronyms longer than 10 characters are dropped.
    let kw = indexer::extract_keywords("ABCDEFGHIJK ABCDEFGHIJ", 50);
    assert_eq!(kw, vec!["ABCDEFGHIJ".to_string()]);
    // Excluded acronyms never appear.
    assert!(indexer::extract_keywords("THE AND FOR", 50).is_empty());
    // Dotted identifiers are case-insensitive.
    assert_eq!(
        indexer::extract_keywords("FOO.BAR", 50),
        // "BAR" and "FOO" also match the acronym pattern.
        vec!["BAR".to_string(), "FOO".to_string(), "FOO.BAR".to_string()]
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Tool timeline + file paths — goldens
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn tool_timeline_and_file_paths_match_goldens() {
    let g = goldens("indexer.json");
    for case in ["rich", "files", "notools"] {
        let tmp = TempDir::new().unwrap();
        let entry = &g[case];
        let events = parse_events(tmp.path(), entry["jsonl"].as_str().unwrap());

        assert_eq!(
            events.len(),
            entry["event_count"].as_u64().unwrap() as usize,
            "event count for {case}"
        );
        assert_eq!(
            indexer::generate_tool_timeline(&events),
            entry["tool_timeline"].as_str().unwrap(),
            "tool timeline for {case}"
        );

        let expected: Vec<(String, String)> = entry["file_paths"]
            .as_array()
            .unwrap()
            .iter()
            .map(|pair| {
                let a = pair.as_array().unwrap();
                (
                    a[0].as_str().unwrap().to_string(),
                    a[1].as_str().unwrap().to_string(),
                )
            })
            .collect();
        assert_eq!(
            indexer::extract_file_paths(&events),
            expected,
            "file paths for {case}"
        );
    }
}

/// The golden `files` case pins the whole rule set; assert the individual
/// claims explicitly so a regression names itself.
#[test]
fn write_beats_read_regardless_of_order() {
    let g = goldens("indexer.json");
    let tmp = TempDir::new().unwrap();
    let events = parse_events(tmp.path(), g["files"]["jsonl"].as_str().unwrap());
    let files = indexer::extract_file_paths(&events);
    let op = |path: &str| {
        files
            .iter()
            .find(|(p, _)| p == path)
            .map(|(_, o)| o.as_str())
    };

    // Read then Edit → write.
    assert_eq!(op("/repo/a.py"), Some("write"));
    // Write then Read → still write (read never downgrades).
    assert_eq!(op("/repo/b.py"), Some("write"));
    // `view` uses the `path` key and counts as a read.
    assert_eq!(op("/repo/c.txt"), Some("read"));
    // NotebookEdit uses `notebook_path` and counts as a write.
    assert_eq!(op("/repo/n.ipynb"), Some("write"));
    // Unknown tool carrying a path → read.
    assert_eq!(op("/repo/d.txt"), Some("read"));
    // Lowercased-contains… no: exact lowercased name match. `create` is a write.
    assert_eq!(op("/repo/e.md"), Some("write"));
    // Bash has no path key, and an empty file_path is falsy.
    assert_eq!(files.len(), 6);

    // First-seen insertion order is preserved.
    let order: Vec<&str> = files.iter().map(|(p, _)| p.as_str()).collect();
    assert_eq!(
        order,
        vec![
            "/repo/a.py",
            "/repo/b.py",
            "/repo/c.txt",
            "/repo/n.ipynb",
            "/repo/d.txt",
            "/repo/e.md"
        ]
    );
}

#[test]
fn empty_timeline_is_empty_string_not_header() {
    assert_eq!(indexer::generate_tool_timeline(&[]), "");
}

/// Write/read classification is case-insensitive on the tool name.
#[test]
fn tool_name_matching_is_case_insensitive() {
    let tmp = TempDir::new().unwrap();
    let jsonl = r#"{"type":"assistant","timestamp":"2025-06-15T10:00:02Z","message":{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"WRITE","input":{"file_path":"/x"}},{"type":"tool_use","id":"t2","name":"VIEW","input":{"path":"/y"}}]}}
"#;
    let events = parse_events(tmp.path(), jsonl);
    assert_eq!(
        indexer::extract_file_paths(&events),
        vec![
            ("/x".to_string(), "write".to_string()),
            ("/y".to_string(), "read".to_string())
        ]
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Database-backed indexing passes
// ═══════════════════════════════════════════════════════════════════════

struct Fixture {
    db: Database,
    _dir: TempDir,
}

fn fixture() -> Fixture {
    let dir = TempDir::new().unwrap();
    let db = Database::new(dir.path().join("sessions.db")).unwrap();
    Fixture { db, _dir: dir }
}

/// Index a transcript through the same path `scan_all` uses, without going
/// near the machine's real session directories.
fn index_file(db: &Database, path: &Path) {
    let mut metadata = ClaudeParser.extract_metadata(path);
    let parsed = ClaudeParser.parse_transcript(path);
    metadata.tool_timeline = Some(indexer::generate_tool_timeline(&parsed.events));
    metadata.event_count = parsed.events.len() as i64;
    let files = indexer::extract_file_paths(&parsed.events);
    metadata.files_touched_count = files.len() as i64;
    db.upsert_session_with_files(&metadata, &files).unwrap();
}

#[test]
fn index_transcripts_fts_indexes_and_stamps() {
    let f = fixture();
    let g = goldens("indexer.json");
    let tmp = TempDir::new().unwrap();
    // A long conversation (>50 chars) gets indexed…
    let long = write_claude(tmp.path(), "long.jsonl", g["rich"]["jsonl"].as_str().unwrap());
    index_file(&f.db, &long);
    // …a trivial one is stamped and skipped.
    let short = write_claude(
        tmp.path(),
        "short.jsonl",
        "{\"type\":\"user\",\"timestamp\":\"2025-06-15T10:00:05Z\",\"message\":{\"role\":\"user\",\"content\":\"hi\"}}\n",
    );
    index_file(&f.db, &short);

    let stats = indexer::index_transcripts_fts(&f.db, Some(10));
    assert_eq!(stats.total_processed, 2);
    assert_eq!(stats.indexed, 1);
    assert_eq!(stats.skipped, 1);
    assert_eq!(stats.errors, 0);

    // Both left the queue.
    assert!(
        indexer::index_transcripts_fts(&f.db, Some(10))
            .total_processed
            == 0
    );
}

#[test]
fn index_transcripts_fts_stamps_missing_files() {
    let f = fixture();
    let g = goldens("indexer.json");
    let tmp = TempDir::new().unwrap();
    let path = write_claude(tmp.path(), "gone.jsonl", g["rich"]["jsonl"].as_str().unwrap());
    index_file(&f.db, &path);
    fs::remove_file(&path).unwrap();

    let stats = indexer::index_transcripts_fts(&f.db, Some(10));
    assert_eq!((stats.indexed, stats.skipped, stats.errors), (0, 1, 0));
    // Stamped, so it does not come back.
    assert_eq!(
        indexer::index_transcripts_fts(&f.db, Some(10)).total_processed,
        0
    );
}

#[test]
fn index_keywords_writes_keywords_column() {
    let f = fixture();
    let tmp = TempDir::new().unwrap();
    let jsonl = "{\"type\":\"user\",\"timestamp\":\"2025-06-15T10:00:05Z\",\"message\":{\"role\":\"user\",\"content\":\"Please refactor search_sessions and ThreadPoolExecutor via sqlite3.connect today\"}}\n";
    let path = write_claude(tmp.path(), "kw.jsonl", jsonl);
    index_file(&f.db, &path);

    let stats = indexer::index_keywords(&f.db, Some(10));
    assert_eq!((stats.indexed, stats.skipped, stats.errors), (1, 0, 0));

    let row = f
        .db
        .get_session(&ClaudeParser.extract_metadata(&path).session_id)
        .unwrap()
        .unwrap();
    let keywords: Vec<String> = serde_json::from_str(row.keywords.as_deref().unwrap()).unwrap();
    assert!(keywords.contains(&"search_sessions".to_string()));
    assert!(keywords.contains(&"ThreadPoolExecutor".to_string()));
    assert!(keywords.contains(&"sqlite3.connect".to_string()));
}

#[test]
fn index_keywords_skips_missing_and_trivial() {
    let f = fixture();
    let tmp = TempDir::new().unwrap();
    let gone = write_claude(
        tmp.path(),
        "gone.jsonl",
        "{\"type\":\"user\",\"timestamp\":\"2025-06-15T10:00:05Z\",\"message\":{\"role\":\"user\",\"content\":\"a longer prompt so the row survives message_count filtering\"}}\n",
    );
    index_file(&f.db, &gone);
    fs::remove_file(&gone).unwrap();

    let stats = indexer::index_keywords(&f.db, Some(10));
    assert_eq!((stats.indexed, stats.skipped, stats.errors), (0, 1, 0));
}

#[test]
fn reindex_session_refreshes_timeline_and_counts() {
    let f = fixture();
    let g = goldens("indexer.json");
    let tmp = TempDir::new().unwrap();
    let path = write_claude(tmp.path(), "s.jsonl", g["notools"]["jsonl"].as_str().unwrap());
    index_file(&f.db, &path);
    let session_id = ClaudeParser.extract_metadata(&path).session_id;
    assert_eq!(
        f.db.get_session(&session_id)
            .unwrap()
            .unwrap()
            .tool_timeline
            .as_deref(),
        Some("")
    );

    // Rewrite the file with tool calls and reindex.
    fs::write(&path, g["files"]["jsonl"].as_str().unwrap()).unwrap();
    assert!(indexer::reindex_session(&f.db, &session_id));

    let row = f.db.get_session(&session_id).unwrap().unwrap();
    assert_eq!(
        row.tool_timeline.as_deref(),
        Some(g["files"]["tool_timeline"].as_str().unwrap())
    );
    assert_eq!(row.files_touched_count, 6);
    assert!(row.token_count > 0);
    assert_eq!(f.db.get_session_files(&session_id).unwrap().len(), 6);
}

#[test]
fn reindex_session_returns_false_for_unknown_or_missing() {
    let f = fixture();
    assert!(!indexer::reindex_session(&f.db, "no-such-session"));

    let tmp = TempDir::new().unwrap();
    let path = write_claude(
        tmp.path(),
        "s.jsonl",
        "{\"type\":\"user\",\"timestamp\":\"2025-06-15T10:00:05Z\",\"message\":{\"role\":\"user\",\"content\":\"hello there friend\"}}\n",
    );
    index_file(&f.db, &path);
    let session_id = ClaudeParser.extract_metadata(&path).session_id;
    fs::remove_file(&path).unwrap();
    assert!(!indexer::reindex_session(&f.db, &session_id));
}

#[test]
fn parser_order_is_pinned() {
    assert_eq!(
        indexer::PARSER_ORDER,
        ["claude_code", "github_copilot", "vscode_copilot", "codex"]
    );
}

/// An `apply_patch` event's `files` list (built by the Codex parser from the
/// patch headers) contributes every entry as a write; `files` on any other
/// tool is ignored.
#[test]
fn files_list_contributes_every_path() {
    use ekko::models::{Event, EventType};
    let mut patch = Event::new("p", 0, EventType::ToolCall);
    patch.tool_name = Some("apply_patch".to_string());
    patch.tool_input = Some(serde_json::json!({
        "content": "*** Begin Patch",
        "files": ["/repo/a.rs", "/repo/b.rs", "", 7, "/repo/a.rs"]
    }));
    let mut read = Event::new("r", 1, EventType::ToolCall);
    read.tool_name = Some("Read".to_string());
    read.tool_input = Some(serde_json::json!({"file_path": "/repo/b.rs"}));
    // A non-patch tool carrying a `files` argument (VS Code's
    // copilot_runTests, MCP tools) is NOT expanded — only the path
    // arguments those tools genuinely name are collected.
    let mut other = Event::new("b", 2, EventType::ToolCall);
    other.tool_name = Some("copilot_runTests".to_string());
    other.tool_input = Some(serde_json::json!({"path": "/repo/c.rs", "files": ["/repo/d.rs", "/repo/c.rs"]}));

    let files = indexer::extract_file_paths(&[patch, read, other]);
    assert_eq!(
        files,
        vec![
            ("/repo/a.rs".to_string(), "write".to_string()),
            ("/repo/b.rs".to_string(), "write".to_string()),
            ("/repo/c.rs".to_string(), "read".to_string()),
        ]
    );
}
