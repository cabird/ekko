//! LLM summarization for sessions.
//!
//! Transcripts above [`CHUNK_THRESHOLD`] tokens are summarized map-reduce
//! style: split into chunks, summarize each with a "part i of n" note, then
//! merge the partial summaries with the inline reduce prompt.
//!
//! These are free functions: summarizing never touches the database. The
//! caller decides what to do with the result, and `background.rs` owns the
//! `update_summary` write — which is what lets a NULL title stay NULL.
//!
//! Usage accounting: the map/reduce/single calls all go through
//! [`crate::llm::call_llm_json`], which records on the **global** usage
//! tracker. Only `search.rs` uses a per-call tracker.

use std::path::Path;

use serde_json::{Value, json};

use crate::db::SessionRow;
use crate::llm::{self, Message};
use crate::models::display_str;
use crate::parsers;
use crate::prompts::{self, PromptError};
use crate::tokens::{count_tokens, split_text_by_tokens};

/// Transcripts above this token count trigger map-reduce chunking.
pub const CHUNK_THRESHOLD: usize = 200_000;

/// A finished session summary.
///
/// Both fields are `Option` because an **absent** key and a key whose value
/// is **null** are treated differently: an absent key becomes `Some("")`,
/// while an explicit `null` becomes `None`, which is written as SQL NULL and
/// leaves the session in the needs-summary queue. `None` here therefore means
/// "write NULL", not "write the string `None`".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionSummary {
    /// Action-oriented title from the model (`Some("")` if the key is
    /// absent, `None` if the model emitted `null`).
    pub title: Option<String>,
    /// Summary body, with `\n\nTopics: …` already appended when the model
    /// returned a truthy `topics` field. `None` only when `summary` was
    /// `null` **and** `topics` was falsy: once topics are appended the null
    /// summary is interpolated into the text as the literal `"None"`.
    pub summary: Option<String>,
}

/// Why summarization failed. The `Display` strings are pinned — the
/// background indexer logs them (truncated to 40 chars), so they are part of
/// the observable output.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SummarizeError {
    /// Missing/absent `transcript_path`.
    #[error("No transcript file")]
    NoTranscript,
    /// Parser lookup or transcript parsing failed.
    #[error("Parse error: {0}")]
    Parse(String),
    /// The LLM call failed; the message is the underlying error.
    #[error("{0}")]
    Llm(String),
    /// The reduce step of a map-reduce summary failed.
    #[error("Reduce step failed: {0}")]
    Reduce(String),
    /// Prompt rendering failed — a template or context bug, not bad input.
    #[error("Prompt error: {0}")]
    Prompt(String),
}

impl From<PromptError> for SummarizeError {
    fn from(e: PromptError) -> Self {
        SummarizeError::Prompt(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Prompt construction (pure — covered by the prompt goldens)
// ---------------------------------------------------------------------------

/// The single-shot summarize prompt: `render('summarize', conversation=…)`.
pub fn summarize_prompt(conversation: &str) -> Result<String, PromptError> {
    prompts::render("summarize", json!({ "conversation": conversation }))
}

/// The map-step prompt for chunk `index` of `num_chunks` (both 1-indexed):
/// the "NOTE:" preamble followed by the ordinary summarize prompt rendered
/// over the chunk.
pub fn map_prompt(index: usize, num_chunks: usize, chunk: &str) -> Result<String, PromptError> {
    let prompt = summarize_prompt(chunk)?;
    Ok(format!(
        "NOTE: This is part {index} of {num_chunks} of a large session transcript. \
Summarize only what is in this part.\n\n{prompt}"
    ))
}

/// Everything in the reduce prompt before the partial-summaries block.
const REDUCE_HEAD: &str = concat!(
    "You are merging partial summaries of a single coding session that was too large ",
    "to summarize at once. The session was split into parts and each part was summarized ",
    "independently. Combine them into one cohesive summary.\n\n",
    "Use the EXACT same JSON output format as the original summaries. ",
    "De-duplicate topics. Synthesize a single coherent title and summary that covers ",
    "the entire session.\n\n",
    "## Output Format\n\n",
    "```json\n",
    "{\n",
    "  \"title\": \"Action-oriented title (60 chars max)\",\n",
    "  \"summary\": \"Structured summary with Outcome, Goal, Approach, and optional Follow-ups sections\",\n",
    "  \"topics\": \"5-10 comma-separated key concepts\",\n",
    "  \"files\": [\"top 5-15 file paths\"],\n",
    "  \"key_entities\": [\"specific technical terms\"],\n",
    "  \"tags\": [\"1-5 canonical tags\"]\n",
    "}\n",
    "```\n\n",
    "## Partial Summaries\n\n",
);

/// The reduce-step prompt built from the raw partial-summary objects the map
/// steps produced. The objects are kept verbatim and their fields rendered
/// with [`display_str`], so a non-string field still renders sensibly.
pub fn reduce_prompt(parts: &[Value], num_chunks: usize) -> String {
    let mut parts_text = String::new();
    for (idx, ps) in parts.iter().enumerate() {
        let i = idx + 1;
        let title = ps.get("title");
        let summary = ps.get("summary");
        let topics = ps.get("topics");
        parts_text.push_str(&format!("--- Part {i} of {num_chunks} ---\n"));
        if truthy(title) {
            parts_text.push_str(&format!("Title: {}\n", field_str(title)));
        }
        parts_text.push_str(&format!("Summary: {}\n", field_str(summary)));
        if truthy(topics) {
            parts_text.push_str(&format!("Topics: {}\n", field_str(topics)));
        }
        parts_text.push('\n');
    }
    format!("{REDUCE_HEAD}{parts_text}\n\nReturn only valid JSON.")
}

/// A field rendered for interpolation; an absent key renders empty.
fn field_str(v: Option<&Value>) -> String {
    v.map(display_str).unwrap_or_default()
}

/// Truthiness of an optional JSON value — absent counts as falsy.
fn truthy(v: Option<&Value>) -> bool {
    v.is_some_and(crate::models::is_truthy)
}

/// A field as it reaches `db.update_summary`: an absent key becomes `""`, an
/// explicit JSON `null` stays `None` (SQL NULL), and anything else is
/// rendered with [`display_str`].
fn opt_field(data: &Value, key: &str) -> Option<String> {
    match data.get(key) {
        None => Some(String::new()),
        Some(Value::Null) => None,
        Some(v) => Some(display_str(v)),
    }
}

/// Post-process a summarize response: pull `title`/`summary`, and append
/// `\n\nTopics: …` when `topics` is truthy.
pub fn summary_from_response(data: &Value) -> SessionSummary {
    let mut summary = opt_field(data, "summary");
    let topics = data.get("topics");
    if truthy(topics) {
        // A null summary becomes the literal "None" once it is interpolated
        // into the topics line — it is no longer NULL after this point.
        summary = Some(format!(
            "{}\n\nTopics: {}",
            summary.as_deref().unwrap_or("None"),
            field_str(topics)
        ));
    }
    SessionSummary {
        title: opt_field(data, "title"),
        summary,
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Generate a summary for a single session.
///
/// # Errors
///
/// See [`SummarizeError`]. Note that an empty transcript is **not** an error:
/// it yields the title `"Empty session"` and an empty summary.
pub async fn summarize_session(session: &SessionRow) -> Result<SessionSummary, SummarizeError> {
    let transcript_path = session.transcript_path.as_deref().unwrap_or("");
    if transcript_path.is_empty() || !Path::new(transcript_path).exists() {
        return Err(SummarizeError::NoTranscript);
    }

    let cli_source = crate::snapshot::effective_cli_source(&session.cli_source);
    let parser = parsers::parser_for(cli_source)
        .ok_or_else(|| SummarizeError::Parse(format!("Unknown CLI source: {cli_source}")))?;
    let parsed = parser.parse_transcript(Path::new(transcript_path));
    let conversation = parsed.get_conversation_text(true);

    if parsers::strip_ws(&conversation).is_empty() {
        return Ok(SessionSummary {
            title: Some("Empty session".to_string()),
            summary: Some(String::new()),
        });
    }

    let token_count = count_tokens(&conversation);
    if token_count <= CHUNK_THRESHOLD {
        summarize_single(&conversation).await
    } else {
        summarize_mapreduce(&conversation).await
    }
}

/// Summarize a conversation that fits in one LLM call.
async fn summarize_single(conversation: &str) -> Result<SessionSummary, SummarizeError> {
    let prompt = summarize_prompt(conversation)?;
    let data = llm::call_llm_json(&[Message::user(prompt)])
        .await
        .map_err(|e| SummarizeError::Llm(e.to_string()))?;
    Ok(summary_from_response(&data))
}

/// Map-reduce summarization for large transcripts.
async fn summarize_mapreduce(conversation: &str) -> Result<SessionSummary, SummarizeError> {
    let chunks = split_text_by_tokens(conversation, CHUNK_THRESHOLD);
    let num_chunks = chunks.len();
    tracing::info!(chunks = num_chunks, "map-reduce summary");

    // Map: summarize each chunk. A failed chunk becomes a placeholder part
    // rather than failing the summary — the reduce step still runs.
    let mut partials: Vec<Value> = Vec::with_capacity(num_chunks);
    for (idx, chunk) in chunks.iter().enumerate() {
        let i = idx + 1;
        let prompt = map_prompt(i, num_chunks, chunk)?;
        match llm::call_llm_json(&[Message::user(prompt)]).await {
            Ok(data) => partials.push(data),
            Err(e) => {
                tracing::warn!(part = i, total = num_chunks, error = %e, "map step failed");
                partials.push(json!({ "summary": format!("[Part {i} failed: {e}]") }));
            }
        }
    }

    // Reduce.
    let prompt = reduce_prompt(&partials, num_chunks);
    let data = llm::call_llm_json(&[Message::user(prompt)])
        .await
        .map_err(|e| SummarizeError::Reduce(e.to_string()))?;
    Ok(summary_from_response(&data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topics_are_appended_to_summary() {
        let s = summary_from_response(&json!({
            "title": "T", "summary": "S", "topics": "a, b"
        }));
        assert_eq!(s.title.as_deref(), Some("T"));
        assert_eq!(s.summary.as_deref(), Some("S\n\nTopics: a, b"));
    }

    #[test]
    fn empty_topics_are_not_appended() {
        let s = summary_from_response(&json!({"title": "T", "summary": "S", "topics": ""}));
        assert_eq!(s.summary.as_deref(), Some("S"));
        let s = summary_from_response(&json!({"title": "T", "summary": "S"}));
        assert_eq!(s.summary.as_deref(), Some("S"));
        let s = summary_from_response(&json!({"title": "T", "summary": "S", "topics": null}));
        assert_eq!(s.summary.as_deref(), Some("S"));
    }

    #[test]
    fn missing_fields_default_to_empty_strings() {
        let s = summary_from_response(&json!({}));
        assert_eq!(s.title.as_deref(), Some(""));
        assert_eq!(s.summary.as_deref(), Some(""));
    }

    /// A present-but-null field stays `None`, which `background.rs` hands to
    /// `update_summary` as SQL NULL — the session stays in the needs-summary
    /// queue. It must NOT become the four-byte string "None".
    #[test]
    fn null_fields_stay_none() {
        let s = summary_from_response(&json!({"title": "X", "summary": null}));
        assert_eq!(s.title.as_deref(), Some("X"));
        assert_eq!(s.summary, None);

        let s = summary_from_response(&json!({"title": null, "summary": "S"}));
        assert_eq!(s.title, None);
        assert_eq!(s.summary.as_deref(), Some("S"));

        // But once topics are appended the null summary is interpolated into
        // the text, so it is no longer NULL.
        let s = summary_from_response(&json!({"title": "X", "summary": null, "topics": "a"}));
        assert_eq!(s.summary.as_deref(), Some("None\n\nTopics: a"));
    }

    #[test]
    fn non_string_fields_are_stringified() {
        let s = summary_from_response(&json!({
            "title": 7, "summary": "S", "topics": ["a", "b"]
        }));
        assert_eq!(s.title.as_deref(), Some("7"));
        assert_eq!(s.summary.as_deref(), Some("S\n\nTopics: ['a', 'b']"));
    }

    #[test]
    fn error_messages_are_pinned() {
        assert_eq!(
            SummarizeError::NoTranscript.to_string(),
            "No transcript file"
        );
        assert_eq!(
            SummarizeError::Parse("Unknown CLI source: nope".into()).to_string(),
            "Parse error: Unknown CLI source: nope"
        );
        assert_eq!(SummarizeError::Llm("boom".into()).to_string(), "boom");
        assert_eq!(
            SummarizeError::Reduce("boom".into()).to_string(),
            "Reduce step failed: boom"
        );
    }

    #[tokio::test]
    async fn missing_transcript_path_errors() {
        let mut row = SessionRow {
            cli_source: "claude_code".to_string(),
            ..SessionRow::default()
        };
        assert_eq!(
            summarize_session(&row).await.unwrap_err(),
            SummarizeError::NoTranscript
        );
        row.transcript_path = Some("/definitely/not/here.jsonl".to_string());
        assert_eq!(
            summarize_session(&row).await.unwrap_err(),
            SummarizeError::NoTranscript
        );
    }
}
