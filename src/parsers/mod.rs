//! CLI-specific transcript parsers.
//!
//! Each CLI (Claude Code, GitHub Copilot CLI, VS Code Copilot, OpenAI Codex)
//! has its own parser that converts raw transcripts into the unified event
//! model.
//!
//! Shared tolerance rules — a malformed transcript degrades, it never fails
//! the session:
//!
//! - **BOM stripping**: all text reads strip a leading UTF-8 BOM, so a BOM'd
//!   first JSONL line still parses instead of being skipped as bad JSON.
//! - **Malformed timestamps / wrong-typed fields**: an unparseable timestamp
//!   or a field of the wrong type is treated as absent rather than as an
//!   error.
//! - **Invalid UTF-8**: files are read lossily (U+FFFD replacement) instead
//!   of being rejected.

use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::models::{ParsedTranscript, SessionMetadata};

pub mod claude;
pub mod codex;
pub mod copilot;
pub mod vscode_copilot;

pub use claude::ClaudeParser;
pub use codex::CodexParser;
pub use copilot::CopilotParser;
pub use vscode_copilot::VSCodeCopilotParser;

/// Abstract interface for CLI transcript parsers.
pub trait Parser: Sync {
    /// The `cli_source` value stored in session metadata.
    fn cli_source(&self) -> &'static str;

    /// Parser version string stored in session metadata.
    fn parser_version(&self) -> &'static str;

    /// Discover session transcript files under a base path.
    fn discover_sessions(&self, base_path: &Path) -> Vec<PathBuf>;

    /// Identity used to deduplicate a transcript discovered under several
    /// candidate base paths. The path itself by default; a parser whose
    /// files are mirrored between locations (Codex, see `codex.rs`)
    /// overrides this so one thread is indexed once.
    fn discovery_key(&self, transcript_path: &Path) -> String {
        transcript_path.to_string_lossy().into_owned()
    }

    /// Extract session metadata without building the full event list.
    /// Fast operation for indexing — reads minimal data.
    fn extract_metadata(&self, transcript_path: &Path) -> SessionMetadata;

    /// Parse a transcript file into the unified event model.
    fn parse_transcript(&self, transcript_path: &Path) -> ParsedTranscript;

    /// Hash of the raw transcript file (sha256 hex) for change detection.
    /// `None` if the file cannot be read.
    fn get_source_hash(&self, transcript_path: &Path) -> Option<String> {
        sha256_file_hex(transcript_path)
    }

    /// Check if a transcript file is valid for this parser.
    fn validate_transcript(&self, transcript_path: &Path) -> bool {
        transcript_path.is_file()
    }
}

/// Registry lookup. Returns `None` for unknown sources — never silently
/// falls back to another parser.
pub fn parser_for(cli_source: &str) -> Option<&'static dyn Parser> {
    match cli_source {
        "claude_code" => Some(&ClaudeParser),
        "github_copilot" => Some(&CopilotParser),
        "vscode_copilot" => Some(&VSCodeCopilotParser),
        "codex" => Some(&CodexParser),
        _ => None,
    }
}

/// Read a file as text, stripping a leading UTF-8 BOM and decoding invalid
/// UTF-8 lossily. `None` on I/O error — the caller skips the file.
pub(crate) fn read_text(path: &Path) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    let bytes = bytes
        .strip_prefix(&[0xEF, 0xBB, 0xBF][..])
        .unwrap_or(&bytes);
    Some(String::from_utf8_lossy(bytes).into_owned())
}

/// sha256 hex digest of a file's raw bytes — hashed as-is, BOM included, so
/// the hash tracks the file rather than the decoded text.
pub(crate) fn sha256_file_hex(path: &Path) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Some(
        hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect(),
    )
}

pub(crate) use crate::utils::strip_ws;

/// First `n` characters (not bytes) of `s`, as an owned string.
pub(crate) fn take_chars(s: &str, n: usize) -> String {
    crate::models::truncate_chars(s, n).to_string()
}
