//! `read_transcript` and its mode handlers.
//!
//! Every string this module produces reaches the MCP client verbatim, so the
//! formats (`"first N of M messages"`, `"{n}: {line}"`, `"{s}-{e} of
//! {total}"`, the error sentences) are load-bearing. They are pinned
//! byte-for-byte by `tests/goldens/read_transcript.json`; update those
//! goldens deliberately when the output shape changes on purpose.
//!
//! Two sources are supported: the transcript file (parsed via the CLI parser)
//! and, when the file is gone, the FTS-cached conversation text — whose
//! `Human:` / `Assistant:` / `A:` / `[Tool Calls]` markers are re-split into
//! messages by [`parse_messages_from_text`].

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value, json};

use super::{EkkoMemory, files_touched_json, opt_str};
use crate::db::{DbResult, SessionRow};
use crate::models::{Message, ParsedTranscript};
use crate::parsers;
use crate::paths;
use crate::tokens::count_tokens;

/// The modes `read_transcript` accepts, in dispatch order (the order is
/// user-visible in the unknown-mode error).
pub const MODES: [&str; 7] = ["info", "head", "tail", "lines", "grep", "full", "save"];

/// Arguments for [`EkkoMemory::read_transcript`], with their defaults.
#[derive(Debug, Clone)]
pub struct ReadTranscriptArgs<'a> {
    /// Session to read.
    pub session_id: &'a str,
    /// One of [`MODES`].
    pub mode: &'a str,
    /// Messages to show in `head`/`tail`.
    pub count: i64,
    /// First line for `lines` (1-based, inclusive).
    pub start: Option<i64>,
    /// Last line for `lines` (1-based, inclusive).
    pub end: Option<i64>,
    /// Literal pattern for `grep` (case-insensitive).
    pub pattern: Option<&'a str>,
    /// Context lines around each `grep` match.
    pub context_lines: i64,
    /// Maximum `grep` matches.
    pub max_matches: i64,
    /// Include `[Tool Calls]` blocks in rendered text.
    pub include_tools: bool,
    /// Relative export name for `save`.
    pub output_path: Option<&'a str>,
}

impl<'a> ReadTranscriptArgs<'a> {
    /// `info` mode for a session, with defaults for everything else.
    pub fn new(session_id: &'a str) -> Self {
        ReadTranscriptArgs {
            session_id,
            mode: "info",
            count: 5,
            start: None,
            end: None,
            pattern: None,
            context_lines: 3,
            max_matches: 20,
            include_tools: true,
            output_path: None,
        }
    }

    /// Set the mode.
    pub fn mode(mut self, mode: &'a str) -> Self {
        self.mode = mode;
        self
    }
}

impl EkkoMemory {
    /// Read a session transcript in one of the [`MODES`].
    ///
    /// User-facing failures (missing session, unknown mode, missing
    /// transcript, parse failure, bad export path) are reported in-band as
    /// `{'error': …}` values; only database failures are `Err`.
    ///
    /// # Errors
    ///
    /// Propagates database errors.
    pub fn read_transcript(&self, args: &ReadTranscriptArgs<'_>) -> DbResult<Value> {
        let Some(session) = self.db().get_session(args.session_id)? else {
            return Ok(json!({
                "error": format!("Session not found: {}", args.session_id)
            }));
        };

        let metadata = json!({
            "title": opt_str(session.title.as_deref()),
            "summary": opt_str(session.summary.as_deref()),
            "project_path": opt_str(session.project_path.as_deref()),
            "git_branch": opt_str(session.git_branch.as_deref()),
            "cli_source": session.cli_source,
            "created_at": opt_str(session.created_at.as_deref()),
            "modified_at": opt_str(session.modified_at.as_deref()),
            "message_count": session.message_count,
        });

        if !MODES.contains(&args.mode) {
            return Ok(json!({
                "error": format!(
                    "Unknown mode: {}. Valid modes: {}",
                    args.mode,
                    MODES.join(", ")
                ),
                "session_id": args.session_id,
            }));
        }

        let transcript_path = session.transcript_path.as_deref().unwrap_or("");
        let file_available =
            !transcript_path.is_empty() && std::path::Path::new(transcript_path).exists();

        let parsed: Option<ParsedTranscript> = if file_available {
            // The column always exists on a row, so only an empty string
            // falls back to the default source.
            let cli_source = if session.cli_source.is_empty() {
                "claude_code"
            } else {
                &session.cli_source
            };
            match parsers::parser_for(cli_source) {
                Some(parser) => {
                    Some(parser.parse_transcript(std::path::Path::new(transcript_path)))
                }
                None => {
                    // An unrecognized source is reported in-band as a
                    // `Parse error:` result, not as an `Err`.
                    return Ok(json!({
                        "error": format!("Parse error: Unknown CLI source: {cli_source}"),
                        "session_id": args.session_id,
                    }));
                }
            }
        } else {
            None
        };

        let mut result = match &parsed {
            Some(parsed) => self.handle_parsed_mode(parsed, &session, args)?,
            None => {
                let Some(cached) = self.db().get_transcript_text(args.session_id)? else {
                    return Ok(json!({
                        "error": "No transcript file available",
                        "session_id": args.session_id,
                        "metadata": metadata,
                    }));
                };
                self.handle_text_mode(&cached, &session, args)?
            }
        };
        let is_cached = parsed.is_none();

        // `session_id`, then `metadata`, then (on the cached path) `source`
        // are appended after the handler runs, so all three land at the end of
        // the map in that order.
        result.insert("session_id".into(), json!(args.session_id));
        result.insert("metadata".into(), metadata);
        if is_cached {
            result.insert("source".into(), json!("cached"));
        }
        Ok(Value::Object(result))
    }

    /// Dispatch a mode against a parsed transcript file.
    fn handle_parsed_mode(
        &self,
        parsed: &ParsedTranscript,
        session: &SessionRow,
        args: &ReadTranscriptArgs<'_>,
    ) -> DbResult<Map<String, Value>> {
        Ok(match args.mode {
            "info" => {
                let messages = parsed.get_messages();
                let text = parsed.get_conversation_text(true);
                transcript_info(&text, &messages, session)
            }
            "head" => {
                let messages = parsed.get_messages();
                head(&messages, args.count, args.include_tools)
            }
            "tail" => {
                let messages = parsed.get_messages();
                tail(&messages, args.count, args.include_tools)
            }
            "lines" => transcript_lines(&parsed.get_conversation_text(true), args.start, args.end),
            "grep" => transcript_grep(
                &parsed.get_conversation_text(true),
                args.pattern,
                args.context_lines,
                args.max_matches,
            ),
            "full" => {
                let transcript = parsed.get_conversation_text(args.include_tools);
                self.transcript_full(&transcript, session)?
            }
            // `save` re-renders the transcript, honouring `include_tools`.
            "save" => transcript_save(
                &parsed.get_conversation_text(args.include_tools),
                args.output_path,
            ),
            other => unreachable!("mode validated before dispatch: {other}"),
        })
    }

    /// Dispatch a mode against FTS-cached transcript text.
    fn handle_text_mode(
        &self,
        text: &str,
        session: &SessionRow,
        args: &ReadTranscriptArgs<'_>,
    ) -> DbResult<Map<String, Value>> {
        Ok(match args.mode {
            "info" => {
                let messages = parse_messages_from_text(text);
                transcript_info(text, &messages, session)
            }
            "head" => {
                let messages = parse_messages_from_text(text);
                head(&messages, args.count, args.include_tools)
            }
            "tail" => {
                let messages = parse_messages_from_text(text);
                tail(&messages, args.count, args.include_tools)
            }
            "lines" => transcript_lines(text, args.start, args.end),
            "grep" => transcript_grep(text, args.pattern, args.context_lines, args.max_matches),
            "full" => self.transcript_full(text, session)?,
            // The cached path ignores `include_tools`: the text is whatever
            // was indexed.
            "save" => transcript_save(text, args.output_path),
            other => unreachable!("mode validated before dispatch: {other}"),
        })
    }

    /// `full` mode body, shared by both sources.
    fn transcript_full(
        &self,
        transcript: &str,
        session: &SessionRow,
    ) -> DbResult<Map<String, Value>> {
        let files = self.db().get_session_files(&session.session_id)?;
        let mut m = Map::new();
        m.insert("transcript".into(), json!(transcript));
        m.insert(
            "transcript_path".into(),
            opt_str(session.transcript_path.as_deref()),
        );
        m.insert("files_touched".into(), files_touched_json(&files));
        Ok(m)
    }
}

// ---------------------------------------------------------------------------
// Mode handlers (pure)
// ---------------------------------------------------------------------------

/// `info` mode. The file-backed and cached paths share this, differing only
/// in where the messages come from.
pub fn transcript_info(
    text: &str,
    messages: &[Message],
    session: &SessionRow,
) -> Map<String, Value> {
    let line_count = text.split('\n').count();
    let token_est = count_tokens(text);

    let mut role_counts = Map::new();
    for m in messages {
        let role = if m.role.is_empty() {
            "unknown"
        } else {
            &m.role
        };
        let entry = role_counts
            .entry(role.to_string())
            .or_insert_with(|| json!(0));
        let next = entry.as_i64().unwrap_or(0) + 1;
        *entry = json!(next);
    }

    let mut out = Map::new();
    out.insert("line_count".into(), json!(line_count));
    out.insert("message_count".into(), json!(messages.len()));
    out.insert("role_counts".into(), Value::Object(role_counts));
    out.insert("token_estimate".into(), json!(token_est));
    out.insert("created_at".into(), opt_str(session.created_at.as_deref()));
    out.insert(
        "modified_at".into(),
        opt_str(session.modified_at.as_deref()),
    );
    out
}

/// `head` mode: the first `count` messages.
pub fn head(messages: &[Message], count: i64, include_tools: bool) -> Map<String, Value> {
    let n = count.max(0) as usize;
    let selected = &messages[..n.min(messages.len())];
    message_block(
        selected,
        include_tools,
        format!("first {} of {} messages", selected.len(), messages.len()),
    )
}

/// `tail` mode: the last `count` messages.
///
/// A `count` of zero means "no limit" and returns **every** message, rather
/// than none — pinned by tests.
pub fn tail(messages: &[Message], count: i64, include_tools: bool) -> Map<String, Value> {
    let n = count.max(0) as usize;
    let selected: &[Message] = if n < messages.len() && n > 0 {
        &messages[messages.len() - n..]
    } else {
        messages
    };
    message_block(
        selected,
        include_tools,
        format!("last {} of {} messages", selected.len(), messages.len()),
    )
}

fn message_block(selected: &[Message], include_tools: bool, showing: String) -> Map<String, Value> {
    let mut out = Map::new();
    out.insert(
        "messages".into(),
        Value::Array(
            selected
                .iter()
                .map(|m| json!({"role": m.role, "content": m.content}))
                .collect(),
        ),
    );
    out.insert(
        "text".into(),
        json!(format_messages(selected, include_tools)),
    );
    out.insert("showing".into(), json!(showing));
    out
}

/// Render messages as readable text.
///
/// Note this is *not* `ParsedTranscript.get_conversation_text`: unknown roles
/// render as `[role]` blocks here.
pub fn format_messages(messages: &[Message], include_tools: bool) -> String {
    let mut lines: Vec<String> = Vec::new();
    for msg in messages {
        let content = &msg.content;
        match msg.role.as_str() {
            "tools" if !include_tools => continue,
            "user" => lines.push(format!("Human:\n{content}\n")),
            "assistant" => lines.push(format!("Assistant:\n{content}\n")),
            "tools" => lines.push(format!("[Tool Calls]\n{content}\n")),
            other => lines.push(format!("[{other}]\n{content}\n")),
        }
    }
    lines.join("\n")
}

/// `lines` mode: a numbered slice of the rendered text.
pub fn transcript_lines(text: &str, start: Option<i64>, end: Option<i64>) -> Map<String, Value> {
    let lines: Vec<&str> = text.split('\n').collect();
    let total = lines.len() as i64;

    // Bounds are 1-based and clamped. Zero is treated as "unset", so
    // `start=0` means line 1 and `end=0` means the last line.
    let s = ((start.filter(|v| *v != 0).unwrap_or(1)) - 1).max(0);
    let e = end.filter(|v| *v != 0).unwrap_or(total).min(total);

    let mut numbered: Vec<String> = Vec::new();
    let mut i = s;
    while i < e {
        numbered.push(format!("{}: {}", i + 1, lines[i as usize]));
        i += 1;
    }

    let mut out = Map::new();
    out.insert("lines".into(), json!(numbered.join("\n")));
    out.insert(
        "range".into(),
        json!(format!("{}-{} of {}", s + 1, e, total)),
    );
    out
}

/// `grep` mode: literal, case-insensitive search with numbered context.
pub fn transcript_grep(
    text: &str,
    pattern: Option<&str>,
    context_lines: i64,
    max_matches: i64,
) -> Map<String, Value> {
    let Some(pattern) = pattern.filter(|p| !p.is_empty()) else {
        let mut out = Map::new();
        out.insert("error".into(), json!("pattern is required for grep mode"));
        out.insert("matches".into(), json!([]));
        return out;
    };

    let lines: Vec<&str> = text.split('\n').collect();
    let re = Regex::new(&format!("(?i){}", regex::escape(pattern)))
        .expect("escaped literal is a valid regex");

    let mut matches: Vec<Value> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if !re.is_match(line) {
            continue;
        }
        let i = i as i64;
        let ctx_start = (i - context_lines).max(0);
        let ctx_end = (i + context_lines + 1).min(lines.len() as i64);
        let mut context: Vec<String> = Vec::new();
        let mut j = ctx_start;
        while j < ctx_end {
            context.push(format!("{}: {}", j + 1, lines[j as usize]));
            j += 1;
        }
        matches.push(json!({
            "line_number": i + 1,
            "text": parsers::strip_ws(line),
            "context": context.join("\n"),
        }));

        // The cap is checked *after* appending, so `max_matches=0` still
        // yields one match.
        if matches.len() as i64 >= max_matches {
            break;
        }
    }

    // Computed at return time: with `max_matches <= 0` and zero matches this
    // is `0 >= 0` → true.
    let truncated = matches.len() as i64 >= max_matches;
    // Key order is load-bearing: pattern, matches, total_matches, truncated.
    let mut out = Map::new();
    out.insert("pattern".into(), json!(pattern));
    let total = matches.len();
    out.insert("matches".into(), Value::Array(matches));
    out.insert("total_matches".into(), json!(total));
    out.insert("truncated".into(), json!(truncated));
    out
}

/// `save` mode: write the rendered transcript under the exports sandbox and
/// report stats.
pub fn transcript_save(transcript: &str, output_path: Option<&str>) -> Map<String, Value> {
    let Some(output_path) = output_path.filter(|p| !p.is_empty()) else {
        let mut out = Map::new();
        out.insert(
            "error".into(),
            json!(
                "output_path is required for save mode (a relative name like \
\"my-export.txt\"; saved under <data_dir>/exports/)"
            ),
        );
        return out;
    };

    let out_path = match paths::resolve_export_path(output_path) {
        Ok(p) => p,
        Err(e) => {
            let mut out = Map::new();
            out.insert("error".into(), json!(e.to_string()));
            return out;
        }
    };

    let line_count = transcript.split('\n').count();
    let token_est = count_tokens(transcript);

    if let Err(e) = std::fs::write(&out_path, transcript) {
        let mut out = Map::new();
        out.insert("error".into(), json!(format!("Failed to write file: {e}")));
        return out;
    }
    let size_bytes = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);

    let mut out = Map::new();
    out.insert("saved_to".into(), json!(out_path.to_string_lossy()));
    out.insert("line_count".into(), json!(line_count));
    out.insert("token_estimate".into(), json!(token_est));
    out.insert("size_bytes".into(), json!(size_bytes));
    out
}

/// Splits cached transcript text on the role markers, at line starts.
static MARKER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^(Human:|Assistant:|A:|(?:\[Tool Calls\]))").expect("valid regex")
});

/// Re-derive messages from cached conversation text.
pub fn parse_messages_from_text(text: &str) -> Vec<Message> {
    // Split on the role markers, keeping them: [pre, sep, chunk, sep, …].
    let mut parts: Vec<&str> = Vec::new();
    let mut last = 0usize;
    for m in MARKER_RE.find_iter(text) {
        parts.push(&text[last..m.start()]);
        parts.push(m.as_str());
        last = m.end();
    }
    parts.push(&text[last..]);

    let mut messages = Vec::new();
    let mut i = 1;
    // Walk (marker, content) pairs, skipping the leading preamble.
    while i + 1 < parts.len() {
        let marker = parsers::strip_ws(parts[i]);
        let content = parsers::strip_ws(parts[i + 1]);
        let role = match marker {
            "Human:" => "user",
            "Assistant:" | "A:" => "assistant",
            "[Tool Calls]" => "tools",
            _ => "unknown",
        };
        messages.push(Message {
            role: role.to_string(),
            content: content.to_string(),
        });
        i += 2;
    }
    messages
}
