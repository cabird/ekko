//! `ask_question` — the two-tier Q&A pipeline.
//!
//! Tier 1 ([`EkkoMemory::query_summaries`]) packs as many session summaries as fit in
//! 100K tokens into the `qa_from_summaries` prompt; the model either answers
//! outright or returns candidate tags. Tier 2
//! ([`EkkoMemory::answer_from_transcripts`]) runs `qa_transcript` over
//! each candidate's full conversation and reduces the results with
//! `qa_synthesize`.
//!
//! # Fan-out concurrency
//!
//! Tier 2 fans out with `futures_util::stream::buffered`, which caps
//! in-flight queries at `min(7, max_sessions)` **and** yields results in
//! input order. Input order matters: the `qa_synthesize` prompt numbers
//! sessions with `loop.index`, so completion-order collection would make the
//! rendered prompt vary run to run for the same inputs.

use futures_util::StreamExt;
use futures_util::future::BoxFuture;
use serde_json::{Map, Value, json};

use super::{SessionFilter, EkkoMemory, opt_str, project_label};
use crate::config;
use crate::db::{DbResult, SessionRow};
use crate::llm::{DEFAULT_JSON_RETRIES, DEFAULT_MAX_RETRIES, LlmClient, Message, UsageTracker};
use crate::models::{display_str, truncate_chars};
use crate::parsers;
use crate::prompts::{self, PromptError};

/// Token budget for the summaries block.
pub const MAX_CONTEXT_TOKENS: usize = 100_000;

/// Tokens held back for the prompt template and the model's reply.
pub const RESERVED_TOKENS: usize = 2000;

/// Upper bound on parallel transcript queries.
pub const MAX_WORKERS: usize = 7;

/// A transcript answer shorter than this many characters is treated as an
/// empty conversation.
const MIN_CONVERSATION_CHARS: usize = 50;

/// Arguments for [`EkkoMemory::ask_question`], with their defaults.
#[derive(Debug, Clone)]
pub struct AskArgs<'a> {
    /// The question to answer.
    pub question: &'a str,
    /// Scope to sessions in/under this directory.
    pub directory: Option<&'a str>,
    /// Include sessions in subdirectories.
    pub include_subdirs: bool,
    /// Limit to sessions modified within N days.
    pub days: Option<i64>,
    /// Maximum sessions to query at transcript level.
    pub max_sessions: usize,
    /// Skip the first N sessions (pagination).
    pub offset: i64,
    /// Maximum sessions to consider per call.
    pub router_limit: i64,
}

impl<'a> AskArgs<'a> {
    /// A question with the default scope and limits.
    pub fn new(question: &'a str) -> Self {
        AskArgs {
            question,
            directory: None,
            include_subdirs: true,
            days: None,
            max_sessions: 7,
            offset: 0,
            router_limit: 100,
        }
    }
}

/// What tier 1 produced.
#[derive(Debug, Clone, PartialEq)]
pub enum SummaryOutcome {
    /// The model answered from summaries alone.
    Answered {
        /// `data['answer']`.
        answer: String,
        /// Resolved source records.
        sources: Vec<Value>,
    },
    /// The model routed to candidate sessions for a transcript-level query.
    Candidates(Vec<SessionRow>),
}

impl EkkoMemory {
    /// Answer a question from session history.
    ///
    /// # Errors
    ///
    /// Propagates database errors. LLM failures degrade rather than fail —
    /// see the module docs.
    pub async fn ask_question(&self, args: &AskArgs<'_>) -> DbResult<Value> {
        let tracker = UsageTracker::new();

        let all_sessions = self.db().get_sessions_with_summaries(None)?;
        let mut candidates = self.filter_sessions(
            all_sessions,
            &SessionFilter::new()
                .directory(args.directory, args.include_subdirs)
                .days(args.days),
        );
        sort_by_priority(&mut candidates, args.directory);

        let total_sessions = candidates.len() as i64;
        let offset = args.offset.max(0);
        let router_limit = args.router_limit.max(0);
        // Both ends of the `offset`/`router_limit` window are clamped.
        let start = offset.min(total_sessions) as usize;
        let end = offset
            .saturating_add(router_limit)
            .clamp(start as i64, total_sessions) as usize;
        let candidates: Vec<SessionRow> = candidates[start..end].to_vec();

        let pagination = json!({
            "offset": offset,
            "returned": candidates.len(),
            "total": total_sessions,
            "has_more": offset.saturating_add(router_limit) < total_sessions,
        });

        if candidates.is_empty() {
            return Ok(json!({
                "answer": "No sessions found matching your scope.",
                "sources": [],
                "tier_used": "none",
                "usage": tracker.to_json(),
                "pagination": pagination,
            }));
        }

        // Step 1: summaries — answer or route.
        let outcome = self
            .query_summaries(args.question, &candidates, &tracker)
            .await;

        let candidate_sessions = match outcome {
            SummaryOutcome::Answered { answer, sources } => {
                return Ok(json!({
                    "answer": answer,
                    "sources": sources,
                    "tier_used": "summary",
                    "usage": tracker.to_json(),
                    "pagination": pagination,
                }));
            }
            SummaryOutcome::Candidates(sessions) => sessions,
        };

        if candidate_sessions.is_empty() {
            return Ok(json!({
                "answer": "I found sessions but none appeared relevant to your question.",
                "sources": [],
                "tier_used": "summary",
                "usage": tracker.to_json(),
                "pagination": pagination,
            }));
        }

        // Step 2: transcripts.
        let take = args.max_sessions.min(candidate_sessions.len());
        let transcript = self
            .answer_from_transcripts(
                args.question,
                &candidate_sessions[..take],
                args.max_sessions,
                &tracker,
            )
            .await;

        Ok(json!({
            "answer": transcript.answer,
            "sources": transcript.sources,
            "tier_used": "transcript",
            "pagination": pagination,
            "usage": tracker.to_json(),
        }))
    }

    /// Tier 1: query the summaries.
    ///
    /// Never fails: an LLM or prompt error falls back to the first five
    /// sessions as candidates.
    pub async fn query_summaries(
        &self,
        question: &str,
        sessions: &[SessionRow],
        tracker: &UsageTracker,
    ) -> SummaryOutcome {
        let (tagged, summaries) = pack_summaries(question, sessions, MAX_CONTEXT_TOKENS);

        let fallback = || SummaryOutcome::Candidates(sessions[..sessions.len().min(5)].to_vec());

        // A prompt-render failure is treated exactly like an LLM failure and
        // falls back rather than propagating. Unreachable in practice:
        // templates are embedded at compile time with fixed contexts.
        // (Applies here, to transcript_prompt/synthesize_prompt below, and to
        // memory::search's semantic prompt.)
        let prompt = match summaries_prompt(question, &summaries) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!(error = %e, "summary query failed");
                return fallback();
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
        match call.await {
            Ok(data) => summary_outcome(&data, &tagged),
            Err(e) => {
                tracing::error!(error = %e, "summary query failed");
                fallback()
            }
        }
    }

    /// Tier 2: query each candidate's transcript and reduce the answers.
    pub async fn answer_from_transcripts(
        &self,
        question: &str,
        sessions: &[SessionRow],
        max_sessions: usize,
        tracker: &UsageTracker,
    ) -> TranscriptResult {
        let selected = &sessions[..max_sessions.min(sessions.len())];
        let workers = MAX_WORKERS.min(max_sessions).max(1);

        // The per-session futures are boxed (rather than mapped inline into
        // `stream::iter`) so the combined future's higher-ranked lifetimes
        // stay provably `Send`: without the box, rustc reports
        // "implementation of `FnOnce` is not general enough" the moment this
        // future is required to be `Send` (as the MCP tool handlers require).
        let futures: Vec<BoxFuture<'_, TranscriptAnswer>> = selected
            .iter()
            .map(|s| Box::pin(query_one_transcript(question, s, tracker)) as BoxFuture<'_, _>)
            .collect();
        let results: Vec<TranscriptAnswer> = futures_util::stream::iter(futures)
            .buffered(workers)
            .collect()
            .await;

        finalize_transcript_answers(question, results, tracker).await
    }
}

/// The reduced tier-2 result: `answer`, `confidence` and `sources`.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptResult {
    /// Final answer text.
    pub answer: String,
    /// `'low'`, `'medium'` or `'high'`.
    pub confidence: &'static str,
    /// Source records (`session_id`, and for real answers `title`,
    /// `project`, `excerpt`).
    pub sources: Vec<Value>,
}

/// One transcript query's outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptAnswer {
    /// The session queried.
    pub session_id: String,
    /// The session title; `null` when the column is NULL.
    pub title: Value,
    /// Last path segment of `project_path`, or `'unknown'`.
    pub project: String,
    /// The model's answer; empty when the session was irrelevant or failed.
    pub answer: String,
    /// Why the query produced no answer (`'no_transcript'`, `'empty'`, or
    /// the error text). Never reaches the client; it is for logs only.
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// Pure helpers (LLM-free, unit tested)
// ---------------------------------------------------------------------------

/// Sort candidates before pagination.
///
/// The key is `(-in_directory, modified_at)`, sorted in reverse. Reversing
/// flips the *whole* tuple, so sessions **outside** the requested directory
/// sort first (`0 > -1`) — the opposite of what the key's shape suggests.
/// That is a known quirk, kept deliberately and pinned by tests: changing it
/// would change which sessions the router sees.
pub fn sort_by_priority(sessions: &mut [SessionRow], directory: Option<&str>) {
    let key = |s: &SessionRow| -> (i64, String) {
        let project = s.project_path.as_deref().unwrap_or("");
        let in_directory = match directory.filter(|d| !d.is_empty()) {
            Some(d) if project.starts_with(d) => 1,
            _ => 0,
        };
        // A NULL `modified_at` sorts as the empty string rather than
        // erroring, so such sessions land last.
        (-in_directory, s.modified_at.clone().unwrap_or_default())
    };
    // `sort_by_key` is stable, so equal keys keep their input order.
    sessions.sort_by_key(|s| std::cmp::Reverse(key(s)));
}

/// Pack summaries into the token budget, tagging each session.
///
/// A tag is generated for every session considered, but only *added* to the
/// used set when the session fits, so a rejected session's tag can be reused
/// later.
pub fn pack_summaries<'a>(
    question: &str,
    sessions: &'a [SessionRow],
    max_context_tokens: usize,
) -> (Vec<(String, &'a SessionRow)>, String) {
    use crate::tokens::count_tokens;

    let mut used_tags = std::collections::HashSet::new();
    let mut tagged: Vec<(String, &SessionRow)> = Vec::new();
    let mut lines: Vec<String> = Vec::new();
    let mut total_tokens = 0usize;

    let reserved = RESERVED_TOKENS + count_tokens(question);
    let available = max_context_tokens.saturating_sub(reserved);

    for s in sessions {
        let tag = crate::utils::generate_unique_tag(&used_tags);
        let line = super::session_to_summary_line(s, &tag, true);
        let line_tokens = count_tokens(&line);

        if total_tokens + line_tokens > available {
            // Skip the oversized entry and keep packing rather than stopping:
            // stopping would let one huge summary early in the (newest-first)
            // list hide every older session from the tier-1 router. Same
            // budget, strictly more context.
            tracing::info!(
                skipped = %s.session_id,
                tokens = line_tokens,
                limit = available,
                "summary too large to pack; skipping"
            );
            continue;
        }

        used_tags.insert(tag.clone());
        tagged.push((tag, s));
        lines.push(line);
        total_tokens += line_tokens;
    }

    (tagged, lines.join("\n"))
}

/// Render the tier-1 prompt.
pub fn summaries_prompt(question: &str, summaries: &str) -> Result<String, PromptError> {
    prompts::render(
        "qa_from_summaries",
        json!({ "question": question, "summaries": summaries }),
    )
}

/// Interpret the tier-1 JSON response (pure).
pub fn summary_outcome(data: &Value, tagged: &[(String, &SessionRow)]) -> SummaryOutcome {
    let lookup = |tag: &str| tagged.iter().find(|(t, _)| t == tag).map(|(_, s)| *s);

    if crate::models::is_truthy(data.get("answered").unwrap_or(&Value::Null)) {
        let mut sources = Vec::new();
        if let Some(tags) = data.get("sources").and_then(Value::as_array) {
            for tag in tags {
                let tag = display_str(tag);
                let Some(s) = lookup(&tag) else { continue };
                let date = s
                    .modified_at
                    .as_deref()
                    .filter(|v| !v.is_empty())
                    .or(s.created_at.as_deref())
                    .unwrap_or("");
                let mut m = Map::new();
                m.insert("session_id".into(), json!(s.session_id));
                m.insert("title".into(), opt_str(s.title.as_deref()));
                m.insert(
                    "project".into(),
                    json!(project_label(s.project_path.as_deref())),
                );
                m.insert("date".into(), json!(truncate_chars(date, 10)));
                m.insert("excerpt".into(), opt_str(s.summary.as_deref()));
                sources.push(Value::Object(m));
            }
        }
        return SummaryOutcome::Answered {
            answer: data.get("answer").map(display_str).unwrap_or_default(),
            sources,
        };
    }

    let mut candidates = Vec::new();
    if let Some(items) = data.get("candidates").and_then(Value::as_array) {
        for item in items {
            // A candidate is either `{"tag": …, "why": …}` or a bare tag
            // string. The `why` text is never read again, so it is dropped.
            let tag = match item {
                Value::Object(o) => o.get("tag").map(display_str).unwrap_or_default(),
                other => display_str(other),
            };
            if let Some(s) = lookup(&tag) {
                candidates.push(s.clone());
            }
        }
    }
    SummaryOutcome::Candidates(candidates)
}

/// Render the per-transcript prompt (pure).
pub fn transcript_prompt(
    question: &str,
    conversation: &str,
    session: &SessionRow,
) -> Result<String, PromptError> {
    let date = match session.created_at.as_deref().filter(|d| !d.is_empty()) {
        Some(d) => truncate_chars(d, 10).to_string(),
        None => "unknown".to_string(),
    };
    prompts::render(
        "qa_transcript",
        json!({
            "question": question,
            "conversation": conversation,
            "project": project_label(session.project_path.as_deref()),
            "date": date,
            // A NULL summary is falsy in the template: `{% if summary %}`
            // is false for both a missing value and `''`.
            "summary": opt_str(session.summary.as_deref()),
        }),
    )
}

/// Apply the `NOT_RELEVANT` sentinel to a model reply: a reply containing it
/// is treated as no answer at all.
pub fn apply_not_relevant(answer: &str) -> String {
    if answer.contains("NOT_RELEVANT") {
        String::new()
    } else {
        answer.to_string()
    }
}

/// The `answers` context for `qa_synthesize`: `session_id`, `title`,
/// `project` and `answer` only — no `date` or `summary`, so the template's
/// `{{ answer.date }}` renders empty and `{% if answer.summary %}` is
/// skipped.
pub fn synthesize_answers_context(answers: &[&TranscriptAnswer]) -> Vec<Value> {
    answers
        .iter()
        .map(|r| {
            json!({
                "session_id": r.session_id,
                "title": r.title,
                "project": r.project,
                "answer": r.answer,
            })
        })
        .collect()
}

/// Render the reduce prompt (pure).
pub fn synthesize_prompt(
    question: &str,
    answers: &[&TranscriptAnswer],
) -> Result<String, PromptError> {
    prompts::render(
        "qa_synthesize",
        json!({ "question": question, "answers": synthesize_answers_context(answers) }),
    )
}

/// The client-facing source record for one transcript answer.
pub fn build_source(r: &TranscriptAnswer) -> Value {
    json!({
        "session_id": r.session_id,
        "title": r.title,
        "project": r.project,
        "excerpt": r.answer,
    })
}

/// Fallback when synthesis fails: join the per-session answers.
pub fn combine_answers(answers: &[&TranscriptAnswer]) -> String {
    answers
        .iter()
        .map(|r| format!("From {}:\n{}", r.project, r.answer))
        .collect::<Vec<_>>()
        .join("\n\n")
}

// ---------------------------------------------------------------------------
// LLM-backed steps
// ---------------------------------------------------------------------------

/// Query one transcript.
async fn query_one_transcript(
    question: &str,
    session: &SessionRow,
    tracker: &UsageTracker,
) -> TranscriptAnswer {
    let project = project_label(session.project_path.as_deref());
    let base = |answer: String, error: Option<String>| TranscriptAnswer {
        session_id: session.session_id.clone(),
        title: opt_str(session.title.as_deref()),
        project: project.clone(),
        answer,
        error,
    };

    let transcript_path = session.transcript_path.as_deref().unwrap_or("");
    if transcript_path.is_empty() || !std::path::Path::new(transcript_path).exists() {
        return base(String::new(), Some("no_transcript".to_string()));
    }

    let cli_source = if session.cli_source.is_empty() {
        "claude_code"
    } else {
        &session.cli_source
    };
    let Some(parser) = parsers::parser_for(cli_source) else {
        return base(
            String::new(),
            Some(format!("Unknown CLI source: {cli_source}")),
        );
    };
    let parsed = parser.parse_transcript(std::path::Path::new(transcript_path));
    let conversation = parsed.get_conversation_text(true);

    if conversation.chars().count() < MIN_CONVERSATION_CHARS {
        return base(String::new(), Some("empty".to_string()));
    }

    let prompt = match transcript_prompt(question, &conversation, session) {
        Ok(p) => p,
        Err(e) => return base(String::new(), Some(e.to_string())),
    };

    let call = async {
        let client = LlmClient::for_role(config::get_config(), crate::llm::Role::Query)?;
        client
            .call_llm_with(&[Message::user(prompt)], DEFAULT_MAX_RETRIES, Some(tracker))
            .await
    };
    match call.await {
        Ok(response) => base(apply_not_relevant(response.content()), None),
        Err(e) => {
            tracing::error!(session_id = %session.session_id, error = %e, "LLM query failed");
            base(String::new(), Some(e.to_string()))
        }
    }
}

/// Reduce the per-transcript answers into a single result.
pub(crate) async fn finalize_transcript_answers(
    question: &str,
    results: Vec<TranscriptAnswer>,
    tracker: &UsageTracker,
) -> TranscriptResult {
    let valid: Vec<&TranscriptAnswer> = results.iter().filter(|r| !r.answer.is_empty()).collect();

    if valid.is_empty() {
        return TranscriptResult {
            answer: "I found relevant sessions but couldn't extract a useful answer.".to_string(),
            confidence: "low",
            sources: results
                .iter()
                .take(3)
                .map(|r| json!({ "session_id": r.session_id }))
                .collect(),
        };
    }

    if valid.len() == 1 {
        return TranscriptResult {
            answer: valid[0].answer.clone(),
            confidence: "high",
            sources: vec![build_source(valid[0])],
        };
    }

    let sources: Vec<Value> = valid.iter().map(|r| build_source(r)).collect();

    let prompt = match synthesize_prompt(question, &valid) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!(error = %e, "failed to synthesize answers");
            return TranscriptResult {
                answer: combine_answers(&valid),
                confidence: "medium",
                sources,
            };
        }
    };

    let call = async {
        let client = LlmClient::for_role(config::get_config(), crate::llm::Role::Query)?;
        client
            .call_llm_with(&[Message::user(prompt)], DEFAULT_MAX_RETRIES, Some(tracker))
            .await
    };
    match call.await {
        Ok(response) => TranscriptResult {
            answer: response.content().to_string(),
            confidence: "high",
            sources,
        },
        Err(e) => {
            tracing::error!(error = %e, "failed to synthesize answers");
            TranscriptResult {
                answer: combine_answers(&valid),
                confidence: "medium",
                sources,
            }
        }
    }
}
