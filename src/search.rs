//! Keyword / semantic / hybrid search.
//!
//! # Behaviour worth knowing
//!
//! - **File-ish queries get no special routing.** Every query goes through
//!   the ordinary FTS keyword search. FTS tokenization handles `auth.py` and
//!   `src/auth/AuthProvider.tsx` acceptably (the path segments become
//!   tokens); the loss is that a session that *touched* a file but never
//!   mentioned it in its summary does not match. [`is_file_like_query`] is
//!   kept so the heuristic stays available if a file-scoped search is ever
//!   added back.
//! - **NULL `project_path`** is treated as `''`, so a directory-scoped search
//!   simply does not match it rather than failing.
//! - **Semantic scores are not re-filtered.** The prompt asks the model for
//!   `score >= 50` and the answer is trusted: every returned tag that maps to
//!   a candidate is kept.

use std::collections::HashSet;

use serde_json::Value;

use crate::config;
use crate::db::SessionRow;
use crate::db::{Database, DbResult};
use crate::db::{KeywordHit, TranscriptHit};
use crate::llm::{DEFAULT_JSON_RETRIES, DEFAULT_MAX_RETRIES, LlmClient, Message, UsageTracker};
use crate::models::display_str;
use crate::prompts;
use crate::utils::{
    generate_unique_tag, is_path_under, parse_date_filter, parse_datetime, paths_equal,
    resolve_path,
};

/// Sort key for sessions with an unparseable `modified_at`: the earliest
/// representable UTC instant, so they rank last by recency.
const DATETIME_MIN_TIMESTAMP: f64 = -62_135_596_800.0;

/// Hybrid search stops cascading once it has this many results.
pub const MIN_RESULTS: usize = 5;

/// Which cascade tier produced a hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MatchTier {
    /// FTS over session titles/summaries/first prompts.
    Summary,
    /// FTS over full transcript text.
    Transcript,
    /// LLM relevance judgement over session summaries.
    Semantic,
}

impl MatchTier {
    /// The value stored in `match_tier` and reported to clients.
    pub fn as_str(self) -> &'static str {
        match self {
            MatchTier::Summary => "summary",
            MatchTier::Transcript => "transcript",
            MatchTier::Semantic => "semantic",
        }
    }

    /// Rerank rank (`{'summary': 0, 'transcript': 1, 'semantic': 2}`).
    fn rank(self) -> u8 {
        match self {
            MatchTier::Summary => 0,
            MatchTier::Transcript => 1,
            MatchTier::Semantic => 2,
        }
    }
}

/// A semantic-search hit: the candidate session plus the model's judgement.
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticHit {
    /// The candidate session the tag mapped back to.
    pub session: SessionRow,
    /// The model's `score` field verbatim (`0` when absent), kept as JSON
    /// because whatever the model produced is passed straight through.
    pub score: Value,
    /// The model's `reason` field (`''` when absent).
    pub relevance: String,
}

/// Result of [`SearchEngine::semantic_search`]: `results`, `usage`, and
/// `error` when the LLM step failed.
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticSearchResult {
    /// Matching sessions, capped at `limit`.
    pub results: Vec<SemanticHit>,
    /// `tracker.to_dict()` for this call only.
    pub usage: Value,
    /// Present when the LLM call failed; `results` is then empty.
    pub error: Option<String>,
}

/// A hybrid-search hit, carrying whichever tier-specific extras apply.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    /// The session row.
    pub session: SessionRow,
    /// Which tier found it.
    pub match_tier: MatchTier,
    /// FTS snippet over the summary column (`Summary` tier only).
    pub summary_excerpt: Option<String>,
    /// FTS snippet over the transcript (`Transcript` tier only).
    pub excerpt: Option<String>,
    /// Model score (`Semantic` tier only).
    pub score: Option<Value>,
    /// Model reason (`Semantic` tier only).
    pub relevance: Option<String>,
}

impl SearchHit {
    fn from_keyword(hit: KeywordHit) -> Self {
        SearchHit {
            session: hit.session,
            match_tier: MatchTier::Summary,
            summary_excerpt: hit.summary_excerpt,
            excerpt: None,
            score: None,
            relevance: None,
        }
    }

    fn from_transcript(hit: TranscriptHit) -> Self {
        SearchHit {
            session: hit.session,
            match_tier: MatchTier::Transcript,
            summary_excerpt: None,
            excerpt: hit.excerpt,
            score: None,
            relevance: None,
        }
    }

    fn from_semantic(hit: SemanticHit) -> Self {
        SearchHit {
            session: hit.session,
            match_tier: MatchTier::Semantic,
            summary_excerpt: None,
            excerpt: None,
            score: Some(hit.score),
            relevance: Some(hit.relevance),
        }
    }
}

/// Result of [`SearchEngine::search_hybrid`].
#[derive(Debug, Clone, PartialEq)]
pub struct HybridSearchResult {
    /// Reranked hits, capped at `limit`.
    pub results: Vec<SearchHit>,
    /// Tiers that were actually consulted, in order.
    pub tiers_used: Vec<&'static str>,
    /// Number of deduped hits found before the `limit` cap.
    pub total_found: usize,
}

/// Heuristic for "this query names a file". Nothing routes on it today; it is
/// retained for documentation and future use (see the module docs).
pub fn is_file_like_query(query: &str) -> bool {
    query.contains('/')
        || [".py", ".js", ".ts", ".jsx", ".tsx", ".md"]
            .iter()
            .any(|ext| query.ends_with(ext))
}

/// Search engine supporting keyword, semantic and hybrid search.
pub struct SearchEngine<'a> {
    db: &'a Database,
    /// Test-only client override so [`SearchEngine::semantic_search`] can be
    /// driven against a mock HTTP server instead of the process-wide config.
    #[cfg(test)]
    llm_override: Option<LlmClient>,
}

impl<'a> SearchEngine<'a> {
    /// Wrap a database handle.
    pub fn new(db: &'a Database) -> Self {
        SearchEngine {
            db,
            #[cfg(test)]
            llm_override: None,
        }
    }

    /// The client [`SearchEngine::semantic_search`] calls through: a JSON
    /// completion on the configured query model.
    fn llm_client(&self) -> Result<LlmClient, crate::llm::LlmError> {
        #[cfg(test)]
        if let Some(client) = &self.llm_override {
            return Ok(client.clone());
        }
        LlmClient::for_role(config::get_config(), crate::llm::Role::Query)
    }

    /// Full-text keyword search on session titles, summaries and first
    /// prompts, with optional post-filters.
    ///
    /// The database is over-fetched at `limit * 2` before filtering.
    ///
    /// # Errors
    ///
    /// Propagates database errors.
    pub fn keyword_search(
        &self,
        query: &str,
        project_path: Option<&str>,
        cli_source: Option<&str>,
        limit: i64,
    ) -> DbResult<Vec<KeywordHit>> {
        let mut results = self.db.search_keyword(query, limit.saturating_mul(2))?;

        if let Some(prefix) = project_path.filter(|p| !p.is_empty()) {
            results.retain(|r| {
                r.session
                    .project_path
                    .as_deref()
                    .unwrap_or("")
                    .starts_with(prefix)
            });
        }
        if let Some(cli) = cli_source.filter(|c| !c.is_empty()) {
            results.retain(|r| r.session.cli_source == cli);
        }

        results.truncate(limit.max(0) as usize);
        Ok(results)
    }

    /// Semantic search: ask the LLM which sessions are conceptually relevant.
    ///
    /// `prefilter` narrows candidates with a keyword search first (keeping
    /// only sessions that already have a summary); otherwise every session
    /// with a summary is a candidate. LLM failures never return `Err` — they
    /// are reported in [`SemanticSearchResult::error`] — but database errors
    /// do propagate.
    ///
    /// # Errors
    ///
    /// Propagates database errors from candidate selection.
    pub async fn semantic_search(
        &self,
        query: &str,
        prefilter: bool,
        limit: i64,
    ) -> DbResult<SemanticSearchResult> {
        let tracker = UsageTracker::new();

        let candidates: Vec<SessionRow> = if prefilter {
            self.db
                .search_keyword(query, 20)?
                .into_iter()
                .map(|h| h.session)
                .filter(|s| s.summary.as_deref().is_some_and(|v| !v.is_empty()))
                .collect()
        } else {
            self.db.get_sessions_with_summaries(None)?
        };

        if candidates.is_empty() {
            return Ok(SemanticSearchResult {
                results: Vec::new(),
                usage: tracker.to_json(),
                error: None,
            });
        }

        let (tagged, sessions_block) = build_sessions_block(&candidates);

        let prompt = match prompts::render(
            "semantic_search",
            serde_json::json!({
                "query": query,
                "sessions": sessions_block,
                "limit": limit,
            }),
        ) {
            Ok(p) => p,
            Err(e) => {
                return Ok(SemanticSearchResult {
                    results: Vec::new(),
                    usage: tracker.to_json(),
                    error: Some(e.to_string()),
                });
            }
        };

        let call = async {
            let client = self.llm_client()?;
            client
                .call_llm_json_with(
                    &[Message::user(prompt)],
                    DEFAULT_MAX_RETRIES,
                    DEFAULT_JSON_RETRIES,
                    Some(&tracker),
                )
                .await
        };

        let data = match call.await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, "semantic search failed");
                return Ok(SemanticSearchResult {
                    results: Vec::new(),
                    usage: tracker.to_json(),
                    error: Some(e.to_string()),
                });
            }
        };

        let mut results = Vec::new();
        if let Some(items) = data.get("results").and_then(Value::as_array) {
            for item in items {
                let tag = item.get("tag").map(display_str).unwrap_or_default();
                if let Some((_, session)) = tagged.iter().find(|(t, _)| *t == tag) {
                    results.push(SemanticHit {
                        session: (*session).clone(),
                        score: item.get("score").cloned().unwrap_or(Value::from(0)),
                        relevance: item.get("reason").map(display_str).unwrap_or_default(),
                    });
                }
            }
        }
        results.truncate(limit.max(0) as usize);

        Ok(SemanticSearchResult {
            results,
            usage: tracker.to_json(),
            error: None,
        })
    }

    /// Hybrid search: keyword → transcript FTS → semantic, cascading until
    /// [`MIN_RESULTS`] hits exist, deduping by `session_id` and reranking
    /// `summary > transcript > semantic`, then by recency.
    ///
    /// # Errors
    ///
    /// Propagates database errors.
    #[allow(clippy::too_many_arguments)]
    pub async fn search_hybrid(
        &self,
        query: &str,
        directory: Option<&str>,
        include_subdirs: bool,
        days: Option<i64>,
        start_date: Option<&str>,
        end_date: Option<&str>,
        limit: i64,
    ) -> DbResult<HybridSearchResult> {
        let mut results: Vec<SearchHit> = Vec::new();
        let mut seen_ids: HashSet<String> = HashSet::new();
        let mut tiers_used: Vec<&'static str> = Vec::new();

        let (start_dt, end_dt) = parse_date_filter(days, start_date, end_date);
        let base_path = directory
            .filter(|d| !d.is_empty())
            .map(|d| resolve_path(d).unwrap_or_else(|| std::path::PathBuf::from(d)));

        let keep = |session: &SessionRow| -> bool {
            if let Some(base) = &base_path {
                let project_path = session.project_path.as_deref().unwrap_or("");
                if project_path.is_empty() {
                    return false;
                }
                let ok = if include_subdirs {
                    is_path_under(project_path, base)
                } else {
                    paths_equal(project_path, base)
                };
                if !ok {
                    return false;
                }
            }
            if start_dt.is_some() || end_dt.is_some() {
                let raw = session
                    .modified_at
                    .as_deref()
                    .filter(|s| !s.is_empty())
                    .or(session.created_at.as_deref().filter(|s| !s.is_empty()))
                    .unwrap_or("");
                let Some(dt) = parse_datetime(raw) else {
                    return false;
                };
                if let Some(start) = start_dt
                    && dt < start
                {
                    return false;
                }
                if let Some(end) = end_dt
                    && dt > end
                {
                    return false;
                }
            }
            true
        };

        let overfetch = limit.saturating_mul(2);

        // Tier 1: keyword search on summaries.
        for hit in self.db.search_keyword(query, overfetch)? {
            if !keep(&hit.session) {
                continue;
            }
            if seen_ids.insert(hit.session.session_id.clone()) {
                results.push(SearchHit::from_keyword(hit));
            }
        }
        tiers_used.push("keyword");

        // Tier 2: transcript FTS.
        if results.len() < MIN_RESULTS {
            for hit in self.db.search_transcripts(query, overfetch)? {
                if !keep(&hit.session) {
                    continue;
                }
                if seen_ids.insert(hit.session.session_id.clone()) {
                    results.push(SearchHit::from_transcript(hit));
                }
            }
            tiers_used.push("transcript");
        }

        // Tier 3: semantic. Note the directory/date filters are deliberately
        // NOT applied here — `semantic_search` picks its own candidates.
        if results.len() < MIN_RESULTS {
            let semantic = self.semantic_search(query, false, limit).await?;
            for hit in semantic.results {
                if seen_ids.insert(hit.session.session_id.clone()) {
                    results.push(SearchHit::from_semantic(hit));
                }
            }
            tiers_used.push("semantic");
        }

        rerank(&mut results);
        let total_found = results.len();
        results.truncate(limit.max(0) as usize);

        Ok(HybridSearchResult {
            results,
            tiers_used,
            total_found,
        })
    }
}

/// Rerank by `(tier rank, -modified_at timestamp)`. The sort is stable, so
/// equal keys keep insertion order.
fn rerank(results: &mut [SearchHit]) {
    results.sort_by(|a, b| {
        a.match_tier.rank().cmp(&b.match_tier.rank()).then_with(|| {
            let ka = -recency_timestamp(&a.session);
            let kb = -recency_timestamp(&b.session);
            ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
        })
    });
}

/// `parse_datetime(r.get('modified_at')) or datetime.min(utc)` as a POSIX
/// timestamp.
fn recency_timestamp(session: &SessionRow) -> f64 {
    session
        .modified_at
        .as_deref()
        .and_then(parse_datetime)
        .map(|dt| dt.timestamp() as f64 + f64::from(dt.timestamp_subsec_micros()) / 1e6)
        .unwrap_or(DATETIME_MIN_TIMESTAMP)
}

/// Build the `{{ sessions }}` block: one `[[TAG]] title / metadata /
/// summary` record per candidate, joined with newlines. Returns the
/// tag→session mapping alongside it.
///
/// Byte format:
///
/// ```text
/// [[S_AB01]] {title}
/// dir:{project_path} | cli:{cli_source} | msgs:{message_count}[ | branch:…][ | created:…][ | modified:…]
/// {summary}
/// ```
///
/// `branch` is omitted when empty; `created`/`modified` are the first 10
/// characters of the timestamps and `modified` is omitted when it equals
/// `created`. A NULL column interpolates as the literal `None`.
fn build_sessions_block(candidates: &[SessionRow]) -> (Vec<(String, &SessionRow)>, String) {
    let mut used_tags: HashSet<String> = HashSet::new();
    let mut tagged: Vec<(String, &SessionRow)> = Vec::with_capacity(candidates.len());
    let mut lines: Vec<String> = Vec::with_capacity(candidates.len());

    for session in candidates {
        let tag = generate_unique_tag(&used_tags);
        used_tags.insert(tag.clone());

        let project_path = or_none(&session.project_path);
        let cli = &session.cli_source;
        let msg_count = session.message_count;
        let branch = session.git_branch.as_deref().unwrap_or("");
        let created = first10(&session.created_at);
        let modified = first10(&session.modified_at);

        let mut meta_parts = vec![
            format!("dir:{project_path}"),
            format!("cli:{cli}"),
            format!("msgs:{msg_count}"),
        ];
        if !branch.is_empty() {
            meta_parts.push(format!("branch:{branch}"));
        }
        if !created.is_empty() {
            meta_parts.push(format!("created:{created}"));
        }
        if !modified.is_empty() && modified != created {
            meta_parts.push(format!("modified:{modified}"));
        }
        let metadata = meta_parts.join(" | ");

        // A NULL title renders as the literal `None`; it deliberately does
        // not fall back to `first_prompt` in this block.
        let title = or_none(&session.title);
        let summary = or_none(&session.summary);

        lines.push(format!("[[{tag}]] {title}\n{metadata}\n{summary}"));
        tagged.push((tag, session));
    }

    (tagged, lines.join("\n"))
}

/// A SQL NULL interpolates as the literal `None`.
fn or_none(v: &Option<String>) -> String {
    match v {
        Some(s) => s.clone(),
        None => "None".to_string(),
    }
}

/// First 10 characters of a timestamp, or `''` when it is NULL or empty.
fn first10(v: &Option<String>) -> String {
    match v {
        Some(s) if !s.is_empty() => s.chars().take(10).collect(),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str) -> SessionRow {
        SessionRow {
            session_id: id.to_string(),
            cli_source: "claude_code".to_string(),
            ..SessionRow::default()
        }
    }

    #[test]
    fn file_like_query_detection_is_pinned() {
        assert!(is_file_like_query("src/auth.rs"));
        assert!(is_file_like_query("auth.py"));
        assert!(is_file_like_query("a.md"));
        assert!(!is_file_like_query("authentication"));
        assert!(!is_file_like_query("auth.rs"));
    }

    #[test]
    fn sessions_block_format() {
        let s = SessionRow {
            title: Some("Fix auth".to_string()),
            summary: Some("Outcome: fixed.".to_string()),
            project_path: Some("/home/u/app".to_string()),
            git_branch: Some("main".to_string()),
            message_count: 42,
            created_at: Some("2025-02-10T14:03:22+00:00".to_string()),
            modified_at: Some("2025-02-11T09:00:00+00:00".to_string()),
            ..row("a")
        };
        let rows = [s];
        let (tagged, block) = build_sessions_block(&rows);
        assert_eq!(tagged.len(), 1);
        let tag = &tagged[0].0;
        assert_eq!(
            block,
            format!(
                "[[{tag}]] Fix auth\ndir:/home/u/app | cli:claude_code | msgs:42 | \
branch:main | created:2025-02-10 | modified:2025-02-11\nOutcome: fixed."
            )
        );
    }

    #[test]
    fn sessions_block_omits_branch_and_same_day_modified() {
        let s = SessionRow {
            title: Some("T".to_string()),
            summary: Some("S".to_string()),
            project_path: Some("/p".to_string()),
            message_count: 1,
            created_at: Some("2025-02-10T14:03:22+00:00".to_string()),
            modified_at: Some("2025-02-10T18:00:00+00:00".to_string()),
            ..row("a")
        };
        let rows = [s];
        let (_, block) = build_sessions_block(&rows);
        assert!(block.contains("dir:/p | cli:claude_code | msgs:1 | created:2025-02-10\n"));
        assert!(!block.contains("branch:"));
        assert!(!block.contains("modified:"));
    }

    #[test]
    fn sessions_block_renders_nulls_as_none() {
        let rows = [row("a")];
        let (_, block) = build_sessions_block(&rows);
        let mut lines = block.lines();
        assert!(lines.next().unwrap().ends_with("]] None"));
        assert_eq!(lines.next().unwrap(), "dir:None | cli:claude_code | msgs:0");
        assert_eq!(lines.next().unwrap(), "None");
    }

    #[test]
    fn sessions_block_joins_records_with_single_newline() {
        let rows = [row("a"), row("b")];
        let (_, block) = build_sessions_block(&rows);
        assert_eq!(block.matches("[[S_").count(), 2);
        assert_eq!(block.lines().count(), 6);
    }

    #[test]
    fn rerank_orders_by_tier_then_recency() {
        let mk = |id: &str, tier: MatchTier, modified: Option<&str>| SearchHit {
            session: SessionRow {
                modified_at: modified.map(str::to_string),
                ..row(id)
            },
            match_tier: tier,
            summary_excerpt: None,
            excerpt: None,
            score: None,
            relevance: None,
        };
        let mut hits = vec![
            mk(
                "sem",
                MatchTier::Semantic,
                Some("2030-01-01T00:00:00+00:00"),
            ),
            mk(
                "tx-old",
                MatchTier::Transcript,
                Some("2020-01-01T00:00:00+00:00"),
            ),
            mk("kw-null", MatchTier::Summary, None),
            mk(
                "kw-new",
                MatchTier::Summary,
                Some("2025-06-01T00:00:00+00:00"),
            ),
            mk(
                "kw-old",
                MatchTier::Summary,
                Some("2024-06-01T00:00:00+00:00"),
            ),
        ];
        rerank(&mut hits);
        let ids: Vec<&str> = hits.iter().map(|h| h.session.session_id.as_str()).collect();
        assert_eq!(ids, vec!["kw-new", "kw-old", "kw-null", "tx-old", "sem"]);
    }

    #[test]
    fn tier_strings_are_pinned() {
        assert_eq!(MatchTier::Summary.as_str(), "summary");
        assert_eq!(MatchTier::Transcript.as_str(), "transcript");
        assert_eq!(MatchTier::Semantic.as_str(), "semantic");
    }

    #[test]
    fn first10_and_or_none_helpers() {
        assert_eq!(first10(&Some("2025-02-10T14:03".into())), "2025-02-10");
        assert_eq!(first10(&Some(String::new())), "");
        assert_eq!(first10(&None), "");
        assert_eq!(or_none(&None), "None");
        assert_eq!(or_none(&Some(String::new())), "");
    }

    // -----------------------------------------------------------------
    // semantic_search over a mock HTTP server
    // -----------------------------------------------------------------

    use crate::config::Config;
    use crate::models::SessionMetadata;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    /// Read one HTTP request off the socket and return its body (same
    /// canned-server pattern as `src/llm/tests.rs`, trimmed to what this
    /// test needs).
    fn read_body(stream: &mut std::net::TcpStream) -> String {
        let mut buf: Vec<u8> = Vec::new();
        let mut chunk = [0u8; 4096];
        let header_end = loop {
            if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                break pos;
            }
            let n = stream.read(&mut chunk).expect("read headers");
            assert!(n > 0, "connection closed before headers");
            buf.extend_from_slice(&chunk[..n]);
        };
        let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
        let content_length: usize = head
            .split("\r\n")
            .filter_map(|l| l.split_once(':'))
            .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
            .and_then(|(_, v)| v.trim().parse().ok())
            .unwrap_or(0);
        let body_start = header_end + 4;
        while buf.len() < body_start + content_length {
            let n = stream.read(&mut chunk).expect("read body");
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&chunk[..n]);
        }
        String::from_utf8_lossy(&buf[body_start..]).to_string()
    }

    /// `[[TAG]] title` pairs, in prompt order, recovered from the request the
    /// engine actually sent.
    fn tagged_titles(request_body: &str) -> Vec<(String, String)> {
        let req: Value = serde_json::from_str(request_body).expect("request is JSON");
        let prompt = req["messages"][0]["content"]
            .as_str()
            .expect("user message")
            .to_string();
        prompt
            .lines()
            .filter_map(|line| line.strip_prefix("[["))
            .filter_map(|rest| rest.split_once("]] "))
            .map(|(tag, title)| (tag.to_string(), title.to_string()))
            .collect()
    }

    fn session_with_summary(db: &Database, id: &str, title: &str) {
        let m = SessionMetadata::new(id, "claude_code", format!("/data/{id}.jsonl"));
        db.upsert_session(&m).expect("upsert");
        db.update_summary(id, title, &format!("Outcome: {title}."))
            .expect("summary");
    }

    /// Happy path for the post-LLM half of `semantic_search`: tags map back
    /// to their sessions, `score`/`reason` fall back to `0`/`''`, a tag the
    /// model invented is dropped, the list is cut to `limit`, and usage lands
    /// on the per-call tracker.
    #[tokio::test]
    async fn semantic_search_maps_tags_back_to_sessions() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = Database::new(dir.path().join("test.db")).expect("db");
        for (id, title) in [("s1", "alpha"), ("s2", "beta"), ("s3", "gamma")] {
            session_with_summary(&db, id, title);
        }

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = listener.local_addr().expect("addr");
        let seen: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
        let sink = Arc::clone(&seen);
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let body = read_body(&mut stream);
            let tags: Vec<String> = tagged_titles(&body).into_iter().map(|(t, _)| t).collect();
            // 1st: full record. 2nd: a tag that was never offered. 3rd: no
            // score/reason. 4th: a valid tag that `limit` must cut off.
            let content = serde_json::json!({
                "results": [
                    {"tag": tags[0], "score": 90, "reason": "best match"},
                    {"tag": "S_NOPE1", "score": 80, "reason": "hallucinated"},
                    {"tag": tags[2]},
                    {"tag": tags[1], "score": 70, "reason": "third"},
                ]
            })
            .to_string();
            let payload = serde_json::json!({
                "choices": [{"message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 7}
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
Content-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                payload.len()
            );
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
            *sink.lock().expect("sink") = body;
        });

        let config = Config {
            llm_model: "gpt-5-mini".to_string(),
            api_keys: vec![("OPENAI_API_KEY".to_string(), "sk-test".to_string())],
            ..Config::default()
        };
        let client = LlmClient::new(&config, "gpt-5-mini")
            .expect("client builds")
            .with_base_url(&format!("http://{addr}"));
        let engine = SearchEngine {
            db: &db,
            llm_override: Some(client),
        };

        let out = engine
            .semantic_search("anything", false, 2)
            .await
            .expect("no db error");
        assert_eq!(out.error, None);

        let request = seen.lock().expect("sink").clone();
        let offered = tagged_titles(&request);
        assert_eq!(offered.len(), 3, "all three candidates were offered");
        let title_of = |i: usize| offered[i].1.clone();

        // Two kept (the 4th valid hit is truncated by `limit`), the invented
        // tag silently dropped.
        assert_eq!(out.results.len(), 2);
        assert_eq!(
            out.results[0].session.title.as_deref(),
            Some(&title_of(0)[..])
        );
        assert_eq!(out.results[0].score, Value::from(90));
        assert_eq!(out.results[0].relevance, "best match");
        assert_eq!(
            out.results[1].session.title.as_deref(),
            Some(&title_of(2)[..])
        );
        assert_eq!(out.results[1].score, Value::from(0));
        assert_eq!(out.results[1].relevance, "");

        assert_eq!(out.usage["calls"], Value::from(1));
        assert_eq!(out.usage["input_tokens"], Value::from(10));
        assert_eq!(out.usage["output_tokens"], Value::from(7));
    }
}
