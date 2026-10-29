//! Context Snapshot generation.
//!
//! Same shape as [`crate::summarize`]: one LLM call under
//! [`CHUNK_THRESHOLD`] tokens, map-reduce above it. Unlike summarization the
//! model returns free-form markdown, so [`crate::llm::call_llm`] is used and
//! **the response is stored verbatim** — no fence stripping, no other
//! post-processing.
//!
//! As in `summarize.rs`, generating a snapshot never touches the database:
//! the caller writes the row. Usage is recorded on the global tracker.

use std::path::Path;

use serde_json::json;

use crate::db::SessionRow;
use crate::llm::{self, Message};
use crate::models::truncate_chars;
use crate::parsers;
use crate::prompts::{self, PromptError};
use crate::tokens::{count_tokens, split_text_by_tokens};

/// Transcripts above this token count trigger map-reduce chunking.
pub const CHUNK_THRESHOLD: usize = 200_000;

/// Session metadata interpolated into the snapshot prompts.
///
/// Field values are already resolved to their display form: `None`/empty
/// inputs have been replaced by `"Unknown"`/`"Session"` before they reach a
/// template, so a template never has to guard against a missing value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotMeta {
    /// `project_path` or `"Unknown"`.
    pub project_path: String,
    /// Human-readable CLI name (see [`cli_display`]).
    pub cli_source: String,
    /// `git_branch` or `"Unknown"`.
    pub git_branch: String,
    /// `created_at` (raw ISO string, as stored) or `"Unknown"`.
    pub session_date: String,
    /// `title`, else the first 50 characters of `first_prompt`, else
    /// `"Session"`.
    pub brief_title: String,
}

impl SnapshotMeta {
    /// Build the prompt metadata from a session row.
    ///
    /// A NULL `first_prompt` is treated as `""`, so a row with neither a
    /// title nor a first prompt falls through to `"Session"` rather than
    /// failing — snapshotting must not be blocked by a sparse row.
    pub fn from_session(session: &SessionRow) -> Self {
        let or_unknown = |v: &Option<String>| match v {
            Some(s) if !s.is_empty() => s.clone(),
            _ => "Unknown".to_string(),
        };
        let brief_title = match &session.title {
            Some(t) if !t.is_empty() => t.clone(),
            _ => {
                let head =
                    truncate_chars(session.first_prompt.as_deref().unwrap_or(""), 50).to_string();
                if head.is_empty() {
                    "Session".to_string()
                } else {
                    head
                }
            }
        };
        SnapshotMeta {
            project_path: or_unknown(&session.project_path),
            cli_source: cli_display(effective_cli_source(&session.cli_source)).to_string(),
            git_branch: or_unknown(&session.git_branch),
            session_date: or_unknown(&session.created_at),
            brief_title,
        }
    }
}

/// The `cli_source` to use for a row whose column can be empty or NULL,
/// defaulting to `claude_code`. Both the parser lookup and the
/// [`cli_display`] map must see the *same* value, so every caller goes
/// through this helper rather than defaulting on its own.
pub fn effective_cli_source(cli_source: &str) -> &str {
    if cli_source.is_empty() {
        "claude_code"
    } else {
        cli_source
    }
}

/// Human-readable CLI name. Unknown sources pass through unchanged.
pub fn cli_display(cli_source: &str) -> &str {
    match cli_source {
        "claude_code" => "Claude Code",
        "vscode_copilot" => "VS Code Copilot",
        "github_copilot" => "GitHub Copilot CLI",
        "codex" => "Codex",
        other => other,
    }
}

/// Why snapshot generation failed. The `Display` strings are pinned — the
/// background indexer logs them, so they are part of the observable output.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SnapshotError {
    /// Missing/absent `transcript_path`.
    #[error("No transcript file")]
    NoTranscript,
    /// Parser lookup or transcript parsing failed.
    #[error("Parse error: {0}")]
    Parse(String),
    /// The parsed conversation was blank.
    #[error("Empty transcript")]
    EmptyTranscript,
    /// The LLM call failed; the message is the underlying error.
    #[error("{0}")]
    Llm(String),
    /// The reduce step of a map-reduce snapshot failed.
    #[error("Reduce step failed: {0}")]
    Reduce(String),
    /// Prompt rendering failed — a template or context bug, not bad input.
    #[error("Prompt error: {0}")]
    Prompt(String),
}

impl From<PromptError> for SnapshotError {
    fn from(e: PromptError) -> Self {
        SnapshotError::Prompt(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Prompt construction (pure — covered by the prompt goldens)
// ---------------------------------------------------------------------------

/// The single-shot snapshot prompt:
/// `render('context_snapshot', transcript=…, **meta)`.
pub fn snapshot_prompt(transcript: &str, meta: &SnapshotMeta) -> Result<String, PromptError> {
    prompts::render(
        "context_snapshot",
        json!({
            "transcript": transcript,
            "project_path": meta.project_path,
            "cli_source": meta.cli_source,
            "git_branch": meta.git_branch,
            "session_date": meta.session_date,
            "brief_title": meta.brief_title,
        }),
    )
}

/// The map-step prompt for chunk `index` of `num_chunks` (1-indexed).
pub fn map_prompt(
    index: usize,
    num_chunks: usize,
    chunk: &str,
    meta: &SnapshotMeta,
) -> Result<String, PromptError> {
    let prompt = snapshot_prompt(chunk, meta)?;
    Ok(format!(
        "NOTE: This is part {index} of {num_chunks} of a large session transcript. \
Generate a context snapshot covering only what is in this part.\n\n{prompt}"
    ))
}

/// Everything in the reduce prompt before the interpolated metadata.
const REDUCE_HEAD: &str = concat!(
    "You are merging partial context snapshots of a single coding session that was ",
    "too large to process at once. The session was split into parts and each part ",
    "got its own context snapshot. Combine them into one cohesive context snapshot.\n\n",
    "Use the EXACT same markdown output format as the individual snapshots. ",
    "De-duplicate information. Reconcile any conflicts (later parts are more recent). ",
    "Produce a single comprehensive snapshot that would help a new assistant get up ",
    "to speed on the entire session.\n\n",
);

/// The reduce-step prompt built from the partial snapshots (raw model
/// markdown, in order).
pub fn reduce_prompt(parts: &[String], num_chunks: usize, meta: &SnapshotMeta) -> String {
    let mut parts_text = String::new();
    for (idx, ps) in parts.iter().enumerate() {
        let i = idx + 1;
        parts_text.push_str(&format!("=== Part {i} of {num_chunks} ===\n{ps}\n\n"));
    }
    format!(
        "{REDUCE_HEAD}**Project:** {project}\n**CLI Source:** {cli}\n**Git Branch:** {branch}\n\
**Session Date:** {date}\n\n## Partial Snapshots\n\n{parts_text}\n\n\
Generate the merged context snapshot now.",
        project = meta.project_path,
        cli = meta.cli_source,
        branch = meta.git_branch,
        date = meta.session_date,
    )
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Generate a Context Snapshot for a single session. Returns the model's
/// markdown verbatim.
///
/// # Errors
///
/// See [`SnapshotError`].
pub async fn generate_snapshot(session: &SessionRow) -> Result<String, SnapshotError> {
    let transcript_path = session.transcript_path.as_deref().unwrap_or("");
    if transcript_path.is_empty() || !Path::new(transcript_path).exists() {
        return Err(SnapshotError::NoTranscript);
    }

    let cli_source = effective_cli_source(&session.cli_source);
    let parser = parsers::parser_for(cli_source)
        .ok_or_else(|| SnapshotError::Parse(format!("Unknown CLI source: {cli_source}")))?;
    let parsed = parser.parse_transcript(Path::new(transcript_path));
    let transcript = parsed.get_conversation_text(true);

    if parsers::strip_ws(&transcript).is_empty() {
        return Err(SnapshotError::EmptyTranscript);
    }

    let meta = SnapshotMeta::from_session(session);
    let token_count = count_tokens(&transcript);
    if token_count <= CHUNK_THRESHOLD {
        snapshot_single(&transcript, &meta).await
    } else {
        snapshot_mapreduce(&transcript, &meta).await
    }
}

async fn snapshot_single(transcript: &str, meta: &SnapshotMeta) -> Result<String, SnapshotError> {
    let prompt = snapshot_prompt(transcript, meta)?;
    let response = llm::call_llm(&[Message::user(prompt)])
        .await
        .map_err(|e| SnapshotError::Llm(e.to_string()))?;
    Ok(response.content)
}

async fn snapshot_mapreduce(
    transcript: &str,
    meta: &SnapshotMeta,
) -> Result<String, SnapshotError> {
    let chunks = split_text_by_tokens(transcript, CHUNK_THRESHOLD);
    let num_chunks = chunks.len();
    tracing::info!(chunks = num_chunks, "map-reduce snapshot");

    let mut partials: Vec<String> = Vec::with_capacity(num_chunks);
    for (idx, chunk) in chunks.iter().enumerate() {
        let i = idx + 1;
        let prompt = map_prompt(i, num_chunks, chunk, meta)?;
        match llm::call_llm(&[Message::user(prompt)]).await {
            Ok(response) => partials.push(response.content),
            Err(e) => {
                tracing::warn!(part = i, total = num_chunks, error = %e, "map step failed");
                partials.push(format!("[Part {i} failed: {e}]"));
            }
        }
    }

    let prompt = reduce_prompt(&partials, num_chunks, meta);
    let response = llm::call_llm(&[Message::user(prompt)])
        .await
        .map_err(|e| SnapshotError::Reduce(e.to_string()))?;
    Ok(response.content)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_display_map_and_fallback() {
        assert_eq!(cli_display("claude_code"), "Claude Code");
        assert_eq!(cli_display("vscode_copilot"), "VS Code Copilot");
        assert_eq!(cli_display("github_copilot"), "GitHub Copilot CLI");
        assert_eq!(cli_display("codex"), "Codex");
        assert_eq!(cli_display("weird_cli"), "weird_cli");
        assert_eq!(cli_display(""), "");
    }

    /// One derivation for both uses: an empty `cli_source` column becomes
    /// `claude_code` for the parser lookup AND for `cli_display`, so the two
    /// can never disagree.
    #[test]
    fn empty_cli_source_defaults_for_parser_and_display() {
        assert_eq!(effective_cli_source(""), "claude_code");
        assert_eq!(effective_cli_source("weird_cli"), "weird_cli");
        let row = SessionRow::default();
        assert!(row.cli_source.is_empty());
        assert_eq!(SnapshotMeta::from_session(&row).cli_source, "Claude Code");
    }

    #[test]
    fn meta_falls_back_when_fields_are_missing() {
        let row = SessionRow {
            cli_source: "claude_code".to_string(),
            ..SessionRow::default()
        };
        let meta = SnapshotMeta::from_session(&row);
        assert_eq!(meta.project_path, "Unknown");
        assert_eq!(meta.git_branch, "Unknown");
        assert_eq!(meta.session_date, "Unknown");
        assert_eq!(meta.brief_title, "Session");
        assert_eq!(meta.cli_source, "Claude Code");
    }

    #[test]
    fn empty_strings_are_treated_as_missing() {
        let row = SessionRow {
            cli_source: "vscode_copilot".to_string(),
            project_path: Some(String::new()),
            git_branch: Some(String::new()),
            created_at: Some(String::new()),
            title: Some(String::new()),
            first_prompt: Some("x".repeat(80)),
            ..SessionRow::default()
        };
        let meta = SnapshotMeta::from_session(&row);
        assert_eq!(meta.project_path, "Unknown");
        assert_eq!(meta.brief_title, "x".repeat(50));
        assert_eq!(meta.cli_source, "VS Code Copilot");
    }

    #[test]
    fn brief_title_truncates_by_characters() {
        let row = SessionRow {
            first_prompt: Some("é".repeat(60)),
            ..SessionRow::default()
        };
        assert_eq!(SnapshotMeta::from_session(&row).brief_title, "é".repeat(50));
    }

    #[test]
    fn error_messages_are_pinned() {
        assert_eq!(
            SnapshotError::NoTranscript.to_string(),
            "No transcript file"
        );
        assert_eq!(
            SnapshotError::EmptyTranscript.to_string(),
            "Empty transcript"
        );
        assert_eq!(SnapshotError::Llm("boom".into()).to_string(), "boom");
        assert_eq!(
            SnapshotError::Reduce("boom".into()).to_string(),
            "Reduce step failed: boom"
        );
    }

    #[tokio::test]
    async fn missing_transcript_path_errors() {
        let row = SessionRow {
            cli_source: "claude_code".to_string(),
            ..SessionRow::default()
        };
        assert_eq!(
            generate_snapshot(&row).await.unwrap_err(),
            SnapshotError::NoTranscript
        );
    }
}
