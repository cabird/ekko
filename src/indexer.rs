//! Session discovery and indexing.
//!
//! Three jobs, all synchronous/blocking (they parse files and write SQLite):
//!
//! * [`scan_all`] walks every configured session directory for every parser,
//!   extracts metadata + tool timeline + touched files + token count, and
//!   upserts each session.
//! * [`index_transcripts_fts`] renders full conversations into the
//!   `transcripts_fts` table (no LLM involved).
//! * [`index_keywords`] extracts identifier-shaped keywords from
//!   conversations into `sessions.keywords` (which feeds `sessions_fts`).
//!
//! These are free functions taking `&Database` rather than methods on an
//! indexer object: the only state any of them needs is the database handle.
//!
//! # Robustness
//!
//! Malformed tool inputs never fail a whole session. [`extract_file_paths`]
//! reads `tool_input` as a JSON object and simply contributes no file when it
//! is something else, and a file-path value that is truthy but not a string
//! is stringified on the way into the TEXT column.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;

use crate::db::Database;
use crate::models::{Event, EventType, format_tool_call, display_str};
use crate::parsers::{self, Parser};
use crate::paths;
use crate::tokens::count_tokens;

/// Common acronyms excluded from keyword extraction — too generic to be
/// useful as search terms.
pub const EXCLUDED_ACRONYMS: &[&str] = &[
    "THE", "AND", "FOR", "NOT", "BUT", "ARE", "WAS", "HAS", "HAD", "HER", "HIS", "ITS", "LET",
    "MAY", "NEW", "NOW", "OLD", "SEE", "WAY", "WHO", "BOY", "DID", "GET", "HIM", "HOW", "MAN",
    "OUR", "OUT", "SAY", "SHE", "TOO", "USE", "ALL", "ANY", "CAN", "DAY", "END", "FEW", "GOT",
    "OWN", "PUT", "RUN", "SET", "TRY", "TWO", "WHY", "YES", "YET", "ALSO", "BACK", "BEEN", "CALL",
    "COME", "COULD", "EACH", "EVEN", "FIND", "FIRST", "FROM", "GIVE", "GOOD", "HAVE", "HERE",
    "INTO", "JUST", "KNOW", "LAST", "LIKE", "LONG", "LOOK", "MADE", "MAKE", "MORE", "MOST", "MUCH",
    "MUST", "NAME", "NEED", "ONLY", "OVER", "PART", "SAME", "SOME", "SUCH", "TAKE", "TELL", "THAN",
    "THAT", "THEM", "THEN", "THEY", "THIS", "TIME", "VERY", "WANT", "WELL", "WHAT", "WHEN", "WILL",
    "WITH", "WORK", "WOULD", "YEAR", "YOUR", "TODO", "NOTE", "DONE", "EDIT", "FILE", "READ",
    "SAVE", "OPEN", "CLOSE", "TRUE", "FALSE", "NULL", "NONE",
];

/// Default keyword cap.
pub const MAX_KEYWORDS: usize = 50;

/// Parser registry iteration order. NOTE this differs from
/// `get_session_paths()`'s order (`claude_code`, `vscode_copilot`,
/// `github_copilot`, `codex`); [`scan_all`] iterates this registry and looks
/// paths up by key, so this order is the one that reaches the stats.
pub const PARSER_ORDER: [&str; 4] = ["claude_code", "github_copilot", "vscode_copilot", "codex"];

/// Tool names that indicate write operations (case-insensitive).
/// `apply_patch` is Codex's edit tool; its files arrive as a `files` list.
pub const WRITE_TOOLS: [&str; 5] = ["write", "edit", "create", "notebookedit", "apply_patch"];
/// Tool names that indicate read operations (case-insensitive). Listed for
/// documentation only: reads and unknown tools are handled identically, so
/// [`extract_file_paths`] only needs to test for writes.
pub const READ_TOOLS: [&str; 2] = ["read", "view"];

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Per-CLI counters from [`scan_all`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CliScanStats {
    pub found: usize,
    pub indexed: usize,
}

/// Result of [`scan_all`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ScanStats {
    pub total_found: usize,
    pub total_indexed: usize,
    /// Transcripts left out because they are not interactive sessions
    /// (see [`is_interactive`]) — e.g. `claude -p` one-shots.
    pub total_skipped_non_interactive: usize,
    /// `(cli_source, counts)` in [`PARSER_ORDER`], omitting sources with no
    /// transcripts at all.
    pub by_cli: Vec<(&'static str, CliScanStats)>,
}

/// Result of [`index_transcripts_fts`] / [`index_keywords`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IndexStats {
    pub indexed: usize,
    pub errors: usize,
    pub skipped: usize,
    pub total_processed: usize,
}

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

/// Build the `tool_timeline` text stored on a session: one
/// `format_tool_call` line per TOOL_CALL event under a `Tool Calls:` header,
/// or `""` when there are none.
pub fn generate_tool_timeline(events: &[Event]) -> String {
    let mut lines: Vec<String> = Vec::new();
    for event in events {
        if event.event_type == EventType::ToolCall {
            lines.push(format_tool_call(event));
        }
    }
    if lines.is_empty() {
        return String::new();
    }
    format!("Tool Calls:\n{}", lines.join("\n"))
}

/// Extract `(file_path, operation)` pairs from events.
///
/// Each file appears once: `write` wins over `read`, and an unrecognized
/// tool that carries a file path counts as `read`. First-seen order is
/// preserved — a later write overwrites the operation in place rather than
/// moving the file to the end.
pub fn extract_file_paths(events: &[Event]) -> Vec<(String, String)> {
    let mut order: Vec<String> = Vec::new();
    let mut file_ops: HashMap<String, &'static str> = HashMap::new();

    for event in events {
        if event.event_type != EventType::ToolCall {
            continue;
        }
        let tool_name = event.tool_name.as_deref().unwrap_or("").to_lowercase();

        for file_path in file_paths_of(event) {
            if WRITE_TOOLS.contains(&tool_name.as_str()) {
                // Write always wins.
                if file_ops.insert(file_path.clone(), "write").is_none() {
                    order.push(file_path);
                }
            } else if !file_ops.contains_key(&file_path) {
                // Read tools and unknown-tool-with-a-file-path are handled
                // identically: record 'read' only when unseen.
                file_ops.insert(file_path.clone(), "read");
                order.push(file_path);
            }
        }
    }

    order
        .into_iter()
        .map(|p| {
            let op = file_ops[&p];
            (p, op.to_string())
        })
        .collect()
}

/// The file paths one tool call touched: the first truthy value among
/// `file_path`, `path` and `notebook_path` — at most one path.
///
/// For `apply_patch` only, the `files` list the Codex parser builds from the
/// patch headers additionally yields every file the patch touched. Other
/// tools' `files` arguments (VS Code's `copilot_runTests`, arbitrary MCP
/// tools) are deliberately ignored — they are not file edits.
fn file_paths_of(event: &Event) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let Some(input) = event.tool_input.as_ref().and_then(Value::as_object) else {
        return out;
    };
    for key in ["file_path", "path", "notebook_path"] {
        match input.get(key) {
            None => continue,
            Some(v) if !truthy(v) => continue,
            Some(v) => {
                out.push(display_str(v));
                break;
            }
        }
    }
    let is_patch = event
        .tool_name
        .as_deref()
        .is_some_and(|n| n.eq_ignore_ascii_case("apply_patch"));
    if is_patch && let Some(files) = input.get("files").and_then(Value::as_array) {
        for f in files {
            if let Value::String(s) = f
                && !s.is_empty()
                && !out.contains(s)
            {
                out.push(s.clone());
            }
        }
    }
    out
}

use crate::models::is_truthy as truthy;

struct KeywordPatterns {
    camel_case: Regex,
    pascal_case: Regex,
    snake_case: Regex,
    acronym: Regex,
    dotted: Regex,
}

/// The five extraction patterns. Changing one changes every stored keyword
/// set, so the goldens pin them.
static PATTERNS: LazyLock<KeywordPatterns> = LazyLock::new(|| KeywordPatterns {
    camel_case: Regex::new(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b").unwrap(),
    pascal_case: Regex::new(r"\b[A-Z][a-z]+[A-Z]\w*\b").unwrap(),
    snake_case: Regex::new(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b").unwrap(),
    acronym: Regex::new(r"\b[A-Z]{2,}\b").unwrap(),
    // Case-insensitive for this pattern only.
    dotted: Regex::new(r"(?i)\b[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*\b").unwrap(),
});

/// Extract significant identifiers from text.
///
/// Patterns: CamelCase (2+ humps), PascalCase with an internal capital,
/// snake_case of at least 5 characters, all-caps acronyms of 2..=10
/// characters that are not in [`EXCLUDED_ACRONYMS`], and dotted identifiers
/// (case-insensitive).
///
/// Matches accumulate in a set, so the result is **sorted by codepoint** and
/// only then truncated to `max_keywords` — the cap keeps the alphabetically
/// first keywords, not the most frequent ones.
pub fn extract_keywords(text: &str, max_keywords: usize) -> Vec<String> {
    let p = &*PATTERNS;
    let mut keywords: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();

    for m in p.camel_case.find_iter(text) {
        keywords.insert(m.as_str());
    }
    for m in p.pascal_case.find_iter(text) {
        keywords.insert(m.as_str());
    }
    for m in p.snake_case.find_iter(text) {
        let word = m.as_str();
        // The 5-character minimum counts characters; these matches are ASCII
        // by construction, so bytes == characters.
        if word.len() >= 5 {
            keywords.insert(word);
        }
    }
    for m in p.acronym.find_iter(text) {
        let word = m.as_str();
        if !EXCLUDED_ACRONYMS.contains(&word) && word.len() <= 10 {
            keywords.insert(word);
        }
    }
    for m in p.dotted.find_iter(text) {
        keywords.insert(m.as_str());
    }

    keywords
        .into_iter()
        .take(max_keywords)
        .map(str::to_string)
        .collect()
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

/// Discover every transcript for one CLI source, deduplicated by path
/// string. The candidate list (defaults, or config.json's `session_paths`
/// override) is authoritative — `paths::compute_session_paths` is the
/// single source of truth for per-CLI default directories.
fn discover_for_source(parser: &dyn Parser, candidate_paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut transcript_paths: Vec<PathBuf> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for base_path in candidate_paths {
        if !base_path.exists() {
            continue;
        }
        for path in parser.discover_sessions(base_path) {
            // Parsers whose transcripts can be mirrored across candidate
            // paths (Codex: `~/.codex` and `$CODEX_HOME`) key on the file
            // name, so the first candidate's copy wins and a thread is
            // indexed once per scan.
            if seen.insert(parser.discovery_key(&path)) {
                transcript_paths.push(path);
            }
        }
    }

    transcript_paths
}

// ---------------------------------------------------------------------------
// Indexing entry points
// ---------------------------------------------------------------------------

/// Scan every configured CLI directory for sessions and index them.
///
/// Per-session failures are logged and skipped, never propagated; a failure
/// to resolve session paths at all yields empty stats.
pub fn scan_all(db: &Database) -> ScanStats {
    let session_paths = match paths::get_configured_session_paths() {
        Ok(p) => p,
        Err(e) => {
            tracing::error!(error = %e, "could not resolve session paths");
            return ScanStats::default();
        }
    };
    let lookup = |cli_source: &str| -> &[PathBuf] {
        session_paths
            .iter()
            .find(|(name, _)| *name == cli_source)
            .map(|(_, p)| p)
            .unwrap_or(&[])
    };
    let mut stats = ScanStats::default();

    for cli_source in PARSER_ORDER {
        let Some(parser) = parsers::parser_for(cli_source) else {
            continue;
        };
        let transcript_paths = discover_for_source(parser, lookup(cli_source));
        if transcript_paths.is_empty() {
            continue;
        }

        let total = transcript_paths.len();
        let mut cli_stats = CliScanStats {
            found: total,
            indexed: 0,
        };

        for path in &transcript_paths {
            match index_one(db, parser, path) {
                Ok(IndexOutcome::Indexed) => {
                    cli_stats.indexed += 1;
                    stats.total_indexed += 1;
                }
                Ok(IndexOutcome::SkippedNonInteractive) => {
                    stats.total_skipped_non_interactive += 1;
                }
                Err(e) => tracing::error!(path = %path.display(), error = %e, "error indexing"),
            }
        }

        stats.total_found += total;
        stats.by_cli.push((cli_source, cli_stats));
    }

    stats
}

/// Entrypoints that are not a person working in a session. Claude Code
/// stamps `sdk-cli` on `claude -p` runs and Agent SDK programs — ekko's own
/// summarization calls would be one (they don't persist, but other
/// people's scripts, cron jobs and CI runs do). Codex marks `codex exec`
/// one-shots with source `exec`, and every subagent thread it spawns gets a
/// rollout file of its own with a `subagent` source (the Codex parser folds
/// `guardian_review` threads into that too), and `mcp` is Codex driven by
/// another program over MCP. A transcript without an `entrypoint` field
/// (older versions, other CLIs) is assumed interactive.
pub const NON_INTERACTIVE_ENTRYPOINTS: &[&str] = &["sdk-cli", "exec", "subagent", "mcp"];

/// Whether a session was driven by a person rather than a program.
pub fn is_interactive(metadata: &crate::models::SessionMetadata) -> bool {
    match metadata.entrypoint.as_deref() {
        Some(entrypoint) => !NON_INTERACTIVE_ENTRYPOINTS.contains(&entrypoint),
        None => true,
    }
}

/// What [`index_one`] did with a transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndexOutcome {
    Indexed,
    /// Not a person's session; not stored (and evicted if a previous
    /// version of ekko had stored it).
    SkippedNonInteractive,
}

/// Index a single transcript file: metadata, timeline, files, tokens, upsert.
fn index_one(
    db: &Database,
    parser: &dyn Parser,
    path: &Path,
) -> Result<IndexOutcome, crate::db::DbError> {
    let mut metadata = parser.extract_metadata(path);
    if !is_interactive(&metadata) {
        if db.delete_session(&metadata.session_id)? {
            tracing::info!(
                session_id = %metadata.session_id,
                entrypoint = metadata.entrypoint.as_deref().unwrap_or(""),
                "removed previously indexed non-interactive session"
            );
        }
        return Ok(IndexOutcome::SkippedNonInteractive);
    }
    let mut files: Vec<(String, String)> = Vec::new();

    // Every parser sets `transcript_path` to the file it was handed; an empty
    // value means "no transcript to parse".
    if !metadata.transcript_path.is_empty() {
        let parsed = parser.parse_transcript(path);
        metadata.tool_timeline = Some(generate_tool_timeline(&parsed.events));
        metadata.event_count = parsed.events.len() as i64;
        files = extract_file_paths(&parsed.events);
        metadata.files_touched_count = files.len() as i64;
        let conversation = parsed.get_conversation_text(true);
        metadata.token_count = count_tokens(&conversation) as i64;

        // Persist the capped structured events (the transcript view's
        // source of truth) — but only when the transcript changed or the
        // events are missing: the scan re-parses every file every cycle,
        // and rewriting identical rows each time would churn the WAL.
        let unchanged = metadata.source_hash.is_some()
            && db
                .get_session(&metadata.session_id)?
                .is_some_and(|row| row.source_hash == metadata.source_hash);
        if !unchanged || !db.has_events(&metadata.session_id)? {
            db.store_events(&metadata.session_id, &parsed.events)?;
        }
    }

    db.upsert_session_with_files(&metadata, &files)?;
    Ok(IndexOutcome::Indexed)
}

/// Reindex a single session by id.
/// Returns `false` when the session is unknown, has no transcript file on
/// disk, has an unknown `cli_source`, or the database write fails.
pub fn reindex_session(db: &Database, session_id: &str) -> bool {
    let session = match db.get_session(session_id) {
        Ok(Some(s)) => s,
        Ok(None) => return false,
        Err(e) => {
            tracing::error!(session_id, error = %e, "error reindexing");
            return false;
        }
    };

    let transcript_path = session.transcript_path.unwrap_or_default();
    if transcript_path.is_empty() || !Path::new(&transcript_path).exists() {
        return false;
    }

    let cli_source = crate::snapshot::effective_cli_source(&session.cli_source);
    let Some(parser) = parsers::parser_for(cli_source) else {
        tracing::error!(
            session_id,
            cli_source,
            "error reindexing: unknown CLI source"
        );
        return false;
    };

    match index_one(db, parser, Path::new(&transcript_path)) {
        Ok(IndexOutcome::Indexed) => true,
        // The row is gone now; "reindexed" would be a lie.
        Ok(IndexOutcome::SkippedNonInteractive) => false,
        Err(e) => {
            tracing::error!(session_id, error = %e, "error reindexing");
            false
        }
    }
}

/// Index transcript text into FTS for full-text search.
///
/// Conversations of 50 characters or fewer are stamped as indexed and
/// counted as skipped so they leave the queue permanently; missing files are
/// stamped too (they re-queue via `modified_at > transcript_indexed_at` if
/// the file returns).
pub fn index_transcripts_fts(db: &Database, limit: Option<i64>) -> IndexStats {
    let sessions = match db.get_sessions_needing_transcript_index(limit) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(error = %e, "error listing sessions needing transcript index");
            return IndexStats::default();
        }
    };
    let mut stats = IndexStats {
        total_processed: sessions.len(),
        ..IndexStats::default()
    };

    for session in &sessions {
        let session_id = &session.session_id;
        let transcript_path = session.transcript_path.clone().unwrap_or_default();

        if transcript_path.is_empty() || !Path::new(&transcript_path).exists() {
            if let Err(e) = db.mark_transcript_indexed(session_id) {
                tracing::error!(session_id, error = %e, "error stamping transcript index");
            }
            stats.skipped += 1;
            continue;
        }

        match render_conversation(session, &transcript_path) {
            Err(e) => {
                tracing::error!(session_id, error = %e, "error indexing transcript FTS");
                stats.errors += 1;
            }
            Ok(conversation) => {
                let result = if conversation.chars().count() > 50 {
                    db.index_transcript(session_id, &conversation)
                        .map(|()| true)
                } else {
                    // Empty/trivial transcript — stamp so we don't retry forever.
                    db.mark_transcript_indexed(session_id).map(|()| false)
                };
                match result {
                    Ok(true) => stats.indexed += 1,
                    Ok(false) => stats.skipped += 1,
                    Err(e) => {
                        tracing::error!(session_id, error = %e, "error indexing transcript FTS");
                        stats.errors += 1;
                    }
                }
            }
        }
    }

    stats
}

/// Extract and index keywords from transcript content.
pub fn index_keywords(db: &Database, limit: Option<i64>) -> IndexStats {
    let sessions = match db.get_all_sessions_with_transcripts(limit) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(error = %e, "error listing sessions with transcripts");
            return IndexStats::default();
        }
    };
    let mut stats = IndexStats {
        total_processed: sessions.len(),
        ..IndexStats::default()
    };

    for session in &sessions {
        let session_id = &session.session_id;
        let transcript_path = session.transcript_path.clone().unwrap_or_default();

        if transcript_path.is_empty() || !Path::new(&transcript_path).exists() {
            stats.skipped += 1;
            continue;
        }

        match render_conversation(session, &transcript_path) {
            Err(e) => {
                tracing::error!(session_id, error = %e, "error extracting keywords");
                stats.errors += 1;
            }
            Ok(conversation) => {
                if conversation.chars().count() > 50 {
                    let keywords = extract_keywords(&conversation, MAX_KEYWORDS);
                    if keywords.is_empty() {
                        stats.skipped += 1;
                    } else if let Err(e) = db.update_keywords(session_id, &keywords) {
                        tracing::error!(session_id, error = %e, "error extracting keywords");
                        stats.errors += 1;
                    } else {
                        stats.indexed += 1;
                    }
                } else {
                    stats.skipped += 1;
                }
            }
        }
    }

    stats
}

/// Parse a session's transcript and render the conversation text. The only
/// failure once the file exists is an unrecognized `cli_source`.
fn render_conversation(
    session: &crate::db::SessionRow,
    transcript_path: &str,
) -> Result<String, String> {
    let cli_source = crate::snapshot::effective_cli_source(&session.cli_source);
    let parser = parsers::parser_for(cli_source)
        .ok_or_else(|| format!("Unknown CLI source: {cli_source}"))?;
    let parsed = parser.parse_transcript(Path::new(transcript_path));
    Ok(parsed.get_conversation_text(true))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Codex rollouts mirrored between `~/.codex` and `$CODEX_HOME` are one
    /// thread, discovered once — from the first candidate path.
    #[test]
    fn mirrored_codex_rollouts_are_discovered_once() {
        let tmp = TempDir::new().unwrap();
        let name = "rollout-2026-09-07T13-19-45-01a07d86-dc97-76c3-adcc-dd2942bbd177.jsonl";
        let home = tmp.path().join("home").join("sessions").join("2026").join("09").join("07");
        let orca = tmp.path().join("orca").join("sessions").join("2026").join("09").join("07");
        std::fs::create_dir_all(&home).unwrap();
        std::fs::create_dir_all(&orca).unwrap();
        std::fs::write(home.join(name), "{}\n").unwrap();
        std::fs::write(orca.join(name), "{}\n").unwrap();
        // A thread only the second home has is still found.
        let only_orca = "rollout-2026-09-08T00-00-00-0a0a0a0a-0000-0000-0000-000000000001.jsonl";
        std::fs::write(orca.join(only_orca), "{}\n").unwrap();

        let parser = parsers::parser_for("codex").unwrap();
        let bases = [
            tmp.path().join("home").join("sessions"),
            tmp.path().join("orca").join("sessions"),
        ];
        let found = discover_for_source(parser, &bases);
        assert_eq!(found, vec![home.join(name), orca.join(only_orca)]);
    }

    /// The default key is the full path: identical file names under two
    /// Claude project directories are two sessions.
    #[test]
    fn claude_transcripts_keep_path_identity() {
        let tmp = TempDir::new().unwrap();
        let a = tmp.path().join("a").join("proj");
        let b = tmp.path().join("b").join("proj");
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        std::fs::write(a.join("s.jsonl"), "{}\n").unwrap();
        std::fs::write(b.join("s.jsonl"), "{}\n").unwrap();
        let parser = parsers::parser_for("claude_code").unwrap();
        let bases = [tmp.path().join("a"), tmp.path().join("b")];
        let found = discover_for_source(parser, &bases);
        assert_eq!(found.len(), 2);
    }
}
