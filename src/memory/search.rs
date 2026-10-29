//! `search_sessions` and the memory-level semantic search.
//!
//! Note that this semantic search is *not* [`crate::search::SearchEngine`]'s:
//! it builds its candidate list from `get_sessions_with_summaries` plus the
//! shared session filter, tags them with [`super::session_to_summary_line`]
//! (a different byte format from the engine's block), and adds a keyword
//! fallback when the model returns fewer than
//! [`MIN_RESULTS_BEFORE_FALLBACK`] hits. Both formats are kept.
//!
//! Whether a hit came from the model or from the keyword fallback is not
//! reported to the client: the distinction survives only as the
//! presence/absence of `score`/`relevance`.

use serde_json::{Map, Value, json};

use super::{ResultExtras, SessionFilter, EkkoMemory, format_session_result};
use crate::config;
use crate::db::{DbResult, SessionRow};
use crate::llm::{DEFAULT_JSON_RETRIES, DEFAULT_MAX_RETRIES, LlmClient, Message, UsageTracker};
use crate::models::display_str;
use crate::prompts;
use crate::search::SearchEngine;

/// Below this many semantic hits, `_semantic_search` also runs a keyword
/// search and merges the extras in.
pub const MIN_RESULTS_BEFORE_FALLBACK: usize = 3;

/// How many similar paths to suggest when a scoped search finds nothing.
const SUGGESTED_PATHS: usize = 3;

/// Arguments for [`EkkoMemory::search_sessions`], with their defaults.
#[derive(Debug, Clone)]
pub struct SearchArgs<'a> {
    /// Search query.
    pub query: &'a str,
    /// Scope to sessions in/under this directory.
    pub directory: Option<&'a str>,
    /// Include sessions in subdirectories.
    pub include_subdirs: bool,
    /// Limit to sessions modified within N days.
    pub days: Option<i64>,
    /// ISO 8601 start date (inclusive); takes precedence over `days`.
    pub start_date: Option<&'a str>,
    /// ISO 8601 end date (inclusive); takes precedence over `days`.
    pub end_date: Option<&'a str>,
    /// `"keyword"` (default), `"transcript"`, `"semantic"` or `"hybrid"`.
    /// Any other value behaves like `"keyword"`.
    pub mode: &'a str,
    /// Maximum results.
    pub limit: i64,
}

impl<'a> SearchArgs<'a> {
    /// A keyword search for `query` with the default options.
    pub fn new(query: &'a str) -> Self {
        SearchArgs {
            query,
            directory: None,
            include_subdirs: true,
            days: None,
            start_date: None,
            end_date: None,
            mode: "keyword",
            limit: 20,
        }
    }
}

impl EkkoMemory {
    /// Search for sessions by keyword, transcript text or semantic
    /// similarity.
    ///
    /// # Errors
    ///
    /// Propagates database errors. LLM failures are swallowed: semantic mode
    /// then returns no results.
    pub async fn search_sessions(&self, args: &SearchArgs<'_>) -> DbResult<Value> {
        let tracker = UsageTracker::new();
        let filter = SessionFilter::new()
            .directory(args.directory, args.include_subdirs)
            .days(args.days)
            .dates(args.start_date, args.end_date);
        let limit = args.limit.max(0) as usize;

        let mut response = Map::new();

        match args.mode {
            "hybrid" => {
                let hybrid = SearchEngine::new(self.db())
                    .search_hybrid(
                        args.query,
                        args.directory,
                        args.include_subdirs,
                        args.days,
                        args.start_date,
                        args.end_date,
                        args.limit,
                    )
                    .await?;
                let results: Vec<Value> = hybrid
                    .results
                    .iter()
                    .map(|hit| {
                        format_session_result(
                            &hit.session,
                            &ResultExtras {
                                score: hit.score.clone(),
                                relevance: hit.relevance.clone(),
                                excerpt: hit.excerpt.clone(),
                            },
                        )
                    })
                    .collect();
                response.insert("results".into(), Value::Array(results));
                response.insert("tiers_used".into(), json!(hybrid.tiers_used));
                // The hybrid path uses its own tracker internally, so the
                // usage reported here is always zeroed.
                response.insert("usage".into(), tracker.to_json());
            }
            "semantic" => {
                let hits = self.semantic_search(args, &tracker).await?;
                let results: Vec<Value> = hits
                    .iter()
                    .map(|(s, extras)| format_session_result(s, extras))
                    .collect();
                response.insert("results".into(), Value::Array(results));
                response.insert("usage".into(), tracker.to_json());
            }
            "transcript" => {
                let raw = self
                    .db()
                    .search_transcripts(args.query, args.limit.saturating_mul(2))?;
                // The FTS excerpt travels with the row, so filter in lockstep.
                let state = super::FilterState::new(&filter);
                let results: Vec<Value> = raw
                    .into_iter()
                    .filter(|h| state.matches(&h.session))
                    .take(limit)
                    .map(|h| {
                        format_session_result(
                            &h.session,
                            &ResultExtras {
                                excerpt: h.excerpt,
                                ..ResultExtras::default()
                            },
                        )
                    })
                    .collect();
                response.insert("results".into(), Value::Array(results));
                response.insert("usage".into(), tracker.to_json());
            }
            _ => {
                // Keyword search — the default branch. File-ish queries are
                // not routed anywhere special here (see `search.rs`).
                let raw = self
                    .db()
                    .search_keyword(args.query, args.limit.saturating_mul(2))?;
                let rows: Vec<SessionRow> = raw.into_iter().map(|h| h.session).collect();
                let results: Vec<Value> = self
                    .filter_sessions(rows, &filter)
                    .iter()
                    .take(limit)
                    .map(|s| format_session_result(s, &ResultExtras::default()))
                    .collect();
                response.insert("results".into(), Value::Array(results));
                response.insert("usage".into(), tracker.to_json());
            }
        }

        // Path suggestions when a directory-scoped search finds nothing.
        let empty = response
            .get("results")
            .and_then(Value::as_array)
            .is_none_or(|r| r.is_empty());
        if empty && let Some(dir) = args.directory.filter(|d| !d.is_empty()) {
            let suggestions = self.db().suggest_similar_paths(dir, SUGGESTED_PATHS)?;
            if !suggestions.is_empty() {
                response.insert(
                    "suggestions".into(),
                    json!({
                        "similar_paths": suggestions,
                        "hint": format!(
                            "No sessions found in '{dir}'. Did you mean one of these?"
                        ),
                    }),
                );
            }
        }

        Ok(Value::Object(response))
    }

    /// LLM-powered semantic search over summaries.
    ///
    /// Returns each hit with its score/relevance extras. LLM and prompt
    /// failures are logged and return an empty list, including the
    /// missing-API-key case.
    async fn semantic_search(
        &self,
        args: &SearchArgs<'_>,
        tracker: &UsageTracker,
    ) -> DbResult<Vec<(SessionRow, ResultExtras)>> {
        let filter = SessionFilter::new()
            .directory(args.directory, args.include_subdirs)
            .days(args.days)
            .dates(args.start_date, args.end_date);
        let limit = args.limit.max(0) as usize;

        let candidates = self
            .db()
            .get_sessions_with_summaries(None)
            .map(|c| self.filter_sessions(c, &filter))?;
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let (tagged, sessions_block) = super::build_tagged_summaries(&candidates);
        let prompt = match semantic_prompt(args.query, &sessions_block, args.limit) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!(error = %e, "semantic search failed");
                return Ok(Vec::new());
            }
        };

        let call = async {
            let client = LlmClient::for_role(config::get_config(), crate::llm::Role::Query)?;
            client
                .call_llm_json_with(
                    &[Message::user(prompt)],
                    DEFAULT_MAX_RETRIES,
                    DEFAULT_JSON_RETRIES,
                    Some(tracker),
                )
                .await
        };
        let data = match call.await {
            Ok(v) => v,
            Err(e) => {
                tracing::error!(error = %e, "semantic search failed");
                return Ok(Vec::new());
            }
        };

        let mut results = semantic_hits_from_response(&data, &tagged);
        results.truncate(limit);

        // Fallback: top up with keyword hits when the model was unhelpful.
        if results.len() < MIN_RESULTS_BEFORE_FALLBACK {
            tracing::info!(
                found = results.len(),
                "semantic search returned few results, falling back to keyword search"
            );
            // DIVERGENCE: file-ish queries went to `db.search_by_file`.
            let keyword = self
                .db()
                .search_keyword(args.query, args.limit.saturating_mul(2))?;
            let rows: Vec<SessionRow> = keyword.into_iter().map(|h| h.session).collect();
            let keyword = self.filter_sessions(rows, &filter);

            for kr in keyword {
                if results.iter().any(|(s, _)| s.session_id == kr.session_id) {
                    continue;
                }
                // `match_type = 'keyword_fallback'`: no score/relevance.
                results.push((kr, ResultExtras::default()));
            }
            results.truncate(limit);
        }

        Ok(results)
    }
}

/// Render the `semantic_search` prompt (pure — covered by the P6 goldens).
pub fn semantic_prompt(
    query: &str,
    sessions_block: &str,
    limit: i64,
) -> Result<String, prompts::PromptError> {
    prompts::render(
        "semantic_search",
        json!({ "query": query, "sessions": sessions_block, "limit": limit }),
    )
}

/// Map a `semantic_search` JSON response back onto the tagged candidates
/// (pure). Unknown tags are dropped.
pub fn semantic_hits_from_response(
    data: &Value,
    tagged: &[(String, &SessionRow)],
) -> Vec<(SessionRow, ResultExtras)> {
    let mut results = Vec::new();
    let Some(items) = data.get("results").and_then(Value::as_array) else {
        return results;
    };
    for item in items {
        let tag = item.get("tag").map(display_str).unwrap_or_default();
        if let Some((_, session)) = tagged.iter().find(|(t, _)| *t == tag) {
            results.push((
                (*session).clone(),
                ResultExtras {
                    // Missing `score` defaults to 0, missing `reason` to "".
                    score: Some(item.get("score").cloned().unwrap_or(Value::from(0))),
                    relevance: Some(item.get("reason").map(display_str).unwrap_or_default()),
                    excerpt: None,
                },
            ));
        }
    }
    results
}
