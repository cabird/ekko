//! LLM-facing tool descriptions.
//!
//! Each one opens with a `[FAST]` or `[LLM]` speed tag, then describes the
//! tool and its arguments. Leading/trailing blank lines are stripped and
//! uniform indentation removed, but relative indentation (the nested bullets
//! under `mode:`) is load-bearing and preserved.
//!
//! They live here rather than in `#[tool(description = …)]` attributes for
//! two reasons: the attribute takes a single string literal (unreadable at
//! this length), and rmcp's doc-comment fallback trims every line, which
//! would flatten the nested indentation. [`crate::mcp::EkkoServer::tool_router`]
//! installs them onto the generated tool attributes.
//!
//! A description must only ever point at a tool ekko actually exposes; the
//! tests below pin that, and the speed tag on every entry.

/// `recent_activity`
pub const RECENT_ACTIVITY: &str = r#"[FAST] Get recent session activity.

Returns recent session summaries and frequently touched files.
Use this for a quick overview of what's been happening.

For deeper context on a specific session, use `get_session_snapshot`.

Args:
    directory: Scope to sessions in/under this directory (e.g., current project)
    include_subdirs: Include sessions in subdirectories (default True)
    days: Look back this many days (default 7)
    max_sessions: Maximum sessions to include (default 10)

Returns:
    - sessions: Recent sessions with titles and summaries
    - files_touched: Most frequently modified files
    - summary: Quick overview (if available)"#;

/// `search_sessions`
pub const SEARCH_SESSIONS: &str = r#"[FAST, or LLM for semantic/hybrid modes] Search across ALL sessions.

This is the primary discovery tool. It searches across your entire session
archive and returns matching SESSIONS (not line-level matches). Use this
to find which sessions discuss a topic, then use read_transcript for
line-level details within a specific session.

Args:
    query: Search query (keywords, file paths, or natural language)
    directory: Scope to sessions in/under this directory
    include_subdirs: Include sessions in subdirectories (default True)
    days: Limit to sessions modified within N days
    start_date: ISO 8601 start date (inclusive), e.g. "2026-02-01".
        Takes precedence over `days` when provided.
    end_date: ISO 8601 end date (inclusive), e.g. "2026-02-28".
        Takes precedence over `days` when provided.
    mode: Search mode:
        - "keyword" (default): [FAST] FTS5 over titles, first prompts,
          summaries, extracted keywords and tool timelines. Matches whole
          words (tokens, not substrings); multiple words are implicitly
          ANDed — they may appear anywhere in the session's indexed text,
          in any order. bm25-ranked. Prefix matching works (auth*), and a
          bare OR is honored (sqlite OR postgres). Quoted phrases are NOT
          reliable here — quotes are escaped to literal text, not phrase
          syntax; put distinctive words instead.
        - "transcript": [FAST] Same FTS semantics (whole words, implicit
          AND) over the rendered conversation text of ALL sessions: user
          and assistant messages plus one-line tool-call summaries (tool
          outputs are not indexed). Returns excerpts with >>>match<<<
          markers around hits.
        - "semantic": [LLM] AI-powered concept matching on summaries. Finds
          sessions by meaning even when exact words don't match. Takes 2-5 sec.
        - "hybrid": Cascades keyword → transcript → semantic until results found.
          Best recall, but may use LLM if earlier tiers return nothing.
    limit: Maximum results (default 20)

Returns:
    - results: Matching sessions with summaries (and excerpts for transcript mode)
    - tiers_used: Which search tiers found results (hybrid mode only)
    - usage: Token usage (for semantic/hybrid modes)
    - suggestions: Similar paths if directory filter returns empty

Workflow:
    1. Use search_sessions(query, mode="hybrid") to find relevant sessions
    2. Use read_transcript(session_id, mode="grep", pattern=...) for line-level details

Examples:
    - search_sessions("authentication", mode="hybrid") - find sessions about auth
    - search_sessions("INSERT OR REPLACE", mode="transcript") - find exact SQL text
    - search_sessions("how to deploy", mode="semantic") - conceptual search
    - search_sessions("auth", start_date="2026-02-01", end_date="2026-02-28") - Feb only"#;

/// `get_session_snapshot`
pub const GET_SESSION_SNAPSHOT: &str = r#"[FAST] Get the context snapshot for a session.

Snapshots provide all the context needed to get up to speed on a session:
- Key decisions and rationale
- Commands that were run
- Files modified and why
- Architecture notes
- Gotchas and sharp edges

Use this when you want to deeply understand what happened in a session
without reading the full transcript.

Args:
    session_id: The session ID (from search results or recent_activity)

Returns:
    - snapshot: The context snapshot (markdown)
    - has_snapshot: Whether a snapshot exists
    - metadata: Session metadata (title, project, dates)
    - files_touched: Files modified in this session
    - suggestion: Hint if snapshot not available"#;

/// `read_transcript`
pub const READ_TRANSCRIPT: &str = r#"[FAST] Read a session transcript with flexible access modes.

One tool for all transcript access — check size before reading, browse
head/tail, grep for patterns, read line ranges, or dump the full text.

Args:
    session_id: The session ID
    mode: Access mode (default "info"):
        - "info": line count, message count, token estimate, time range.
          Use this first to check size before reading large transcripts.
        - "head": first N messages (complete turns). Good for seeing what
          the session started with.
        - "tail": last N messages. Good for seeing the final outcome.
        - "lines": line range (1-based, numbered output). Use after grep
          to read surrounding context.
        - "grep": search for pattern with line numbers + context. Follow
          up with "lines" mode to read more around a match.
        - "full": entire transcript text. Same as the old
          get_session_transcript. Use for small/medium sessions.
        - "save": write transcript to a file, return stats only. Use for
          very large transcripts that would overflow the response.
          ``output_path`` is treated as a relative name under
          ``<data_dir>/exports/`` (absolute paths and ``..`` traversal
          are rejected for safety).
    count: Number of messages for head/tail (default 5)
    start: Start line for lines mode (1-based, inclusive)
    end: End line for lines mode (1-based, inclusive)
    pattern: Text pattern for grep mode (case-insensitive)
    context_lines: Lines of context before/after each grep match (default 3)
    max_matches: Maximum grep matches to return (default 20)
    include_tools: Include tool call messages in full/save/head/tail (default True)
    output_path: Relative name for save mode (e.g. "my-export.txt").
        Saved under ``<data_dir>/exports/``. Absolute paths are rejected.

Returns:
    - session_id: The session ID
    - metadata: Session metadata (title, project, dates, message count)
    - Mode-specific fields (see mode descriptions above)

Workflow:
    1. read_transcript(id, mode="info") — check size
    2. read_transcript(id, mode="head") — see first messages
    3. read_transcript(id, mode="grep", pattern="error") — find matches
    4. read_transcript(id, mode="lines", start=100, end=150) — read around match

Examples:
    - read_transcript(id) — get transcript stats (default info mode)
    - read_transcript(id, mode="tail", count=3) — last 3 messages
    - read_transcript(id, mode="grep", pattern="INSERT OR REPLACE")
    - read_transcript(id, mode="lines", start=42, end=60)
    - read_transcript(id, mode="full") — entire transcript
    - read_transcript(id, mode="save", output_path="session.txt")"#;

/// `ask_question`
pub const ASK_QUESTION: &str = r#"[LLM] Answer a question using session history.

This is the smart Q&A operation that:
1. First tries to answer from summaries (fast, cheap)
2. If needed, queries relevant transcripts in parallel

Use this for general questions across your session history.

Args:
    question: Your question
    directory: Scope to sessions in/under this directory
    include_subdirs: Include sessions in subdirectories (default True)
    days: Limit to sessions modified within N days
    max_sessions: Max transcripts to query if needed (default 7)
    offset: Skip first N sessions for pagination (default 0). Use this
        to search through more sessions if `pagination.has_more` is True.

Returns:
    - answer: The answer to your question
    - sources: Sessions that contributed to the answer
    - tier_used: 'summary' or 'transcript'
    - usage: Token usage
    - pagination: {offset, returned, total, has_more} - indicates how many
        sessions were searched. If has_more is True, call again with
        offset=offset+returned to search additional sessions.

Examples:
    - "Why did we choose React over Vue?"
    - "How do we run the test suite?"
    - "What was the deployment process for staging?""#;

/// `list_projects`
pub const LIST_PROJECTS: &str = r#"[FAST] List all projects with session counts and recent activity.

Use this to see what projects have been worked on and how recently.

Returns:
    List of projects with:
    - project_path: Directory path
    - cli_source: Which CLI (claude_code, vscode_copilot, github_copilot, codex)
    - session_count: Number of sessions
    - last_activity: Most recent session date"#;

/// `sessions_in_directory`
pub const SESSIONS_IN_DIRECTORY: &str = r#"[FAST] List all sessions in or under a directory.

Use this to find all past work in the current project directory.

Args:
    directory: Directory path to search
    include_subdirs: Include sessions in subdirectories (default True)
    days: Limit to sessions modified within N days
    limit: Maximum results (default 50)

Returns:
    List of sessions with metadata and whether they have snapshots"#;

/// `get_status`
pub const GET_STATUS: &str = r#"[FAST] Get current status of the ekko database and indexer.

Use this to check:
- How many sessions are indexed
- How many need summaries/snapshots
- Background indexer status
- Database location and size
- LLM cost tracking (this instance, today, last 7 days)
- Which completion backend each role resolved to (index = summaries/snapshots,
  query = ask_question/semantic search), and configuration issues

Returns:
    - database: Path and size of the database
    - sessions: Counts of indexed sessions, with/without summaries
    - indexer: Background indexer status
    - costs: LLM usage costs (this_instance, today, last_7_days, by_operation)
    - config: Current configuration — `index` and `query` (backend, model,
      reason, availability, cost basis), max_workers, reasoning level,
      api_keys_available
    - errors: Recent errors from the indexer log (if any)"#;

/// `export_telemetry`
pub const EXPORT_TELEMETRY: &str = r#"[FAST] Export anonymous usage telemetry to a file for sharing.

Writes a .jsonl file that the user can review and share with developers.
This helps improve ekko by understanding which features are used.

PRIVACY: Only contains operation names, timestamps, durations, instance IDs,
and safe arguments (booleans, numbers, enum values). Does NOT contain queries,
file paths, session content, or any free-form text.

Args:
    filename: Relative name for the export (e.g. "telemetry-2026-04").
        Saved under ``<data_dir>/exports/``. ``.jsonl`` extension is added
        if missing. Absolute paths and ``..`` traversal are rejected.

Returns:
    - file_path: Where the file was written
    - event_count: Number of events exported
    - summary: Quick stats about the data"#;

/// Tool name → description, in the order the tools are declared.
pub const DESCRIPTIONS: &[(&str, &str)] = &[
    ("recent_activity", RECENT_ACTIVITY),
    ("search_sessions", SEARCH_SESSIONS),
    ("get_session_snapshot", GET_SESSION_SNAPSHOT),
    ("read_transcript", READ_TRANSCRIPT),
    ("ask_question", ASK_QUESTION),
    ("list_projects", LIST_PROJECTS),
    ("sessions_in_directory", SESSIONS_IN_DIRECTORY),
    ("get_status", GET_STATUS),
    ("export_telemetry", EXPORT_TELEMETRY),
];

/// The description for a tool, or `None` if the name is unknown.
pub fn description(tool: &str) -> Option<&'static str> {
    DESCRIPTIONS
        .iter()
        .find(|(name, _)| *name == tool)
        .map(|(_, text)| *text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nine_descriptions_none_empty_none_mentioning_dropped_tools() {
        assert_eq!(DESCRIPTIONS.len(), 9);
        for (name, text) in DESCRIPTIONS {
            assert!(!text.is_empty(), "{name} has no description");
            assert!(!text.contains("\n\n\n"), "{name} has a stray blank block");
            for gone in [
                "ask_session",
                "ask_sessions",
                "find_sessions_by_file",
                "get_file_history",
            ] {
                assert!(!text.contains(gone), "{name} mentions dropped tool {gone}");
            }
        }
    }

    #[test]
    fn descriptions_open_with_a_speed_tag() {
        for (name, text) in DESCRIPTIONS {
            assert!(
                text.starts_with("[FAST]")
                    || text.starts_with("[LLM]")
                    || text.starts_with("[FAST,"),
                "{name} lost its speed tag"
            );
        }
    }
}
