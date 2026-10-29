//! CLI completion backends.
//!
//! ekko's LLM calls are all the same shape — one prompt in, one answer out,
//! no tools — which is exactly what coding-assistant CLIs offer in their
//! non-interactive mode. Routing through one lets people who only have a
//! Claude Code subscription (no API key) get summaries and snapshots.
//!
//! Design: `docs/cli-backend/DESIGN.md`. Model choice: `EXPERIMENT.md` there.
//!
//! Only [`CliKind::Claude`] exists today. Adding another CLI means adding a
//! variant, its [`CliKind::command`] / [`CliKind::parse_output`] arms, and a
//! `cli_defaults` row — nothing above this module changes.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use serde_json::Value;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tracing::{debug, warn};

use super::{LlmError, Message, RawUsage};

/// Which CLI to drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliKind {
    /// Claude Code: `claude -p`.
    Claude,
}

impl CliKind {
    /// Executable name looked up on `PATH`.
    pub fn binary(self) -> &'static str {
        match self {
            CliKind::Claude => "claude",
        }
    }

    /// Backend name as written in config (`llm_backend`) and reported by
    /// status/check.
    pub fn name(self) -> &'static str {
        match self {
            CliKind::Claude => "claude-cli",
        }
    }

    /// Default model when the user did not set `llm_model`. Chosen by the
    /// experiment in `docs/cli-backend/EXPERIMENT.md`: haiku fails JSON
    /// summaries of sessions ≥100K tokens (continues the conversation or
    /// invents its own schema); sonnet was correct on every run.
    pub fn default_model(self) -> &'static str {
        match self {
            CliKind::Claude => "sonnet",
        }
    }

    /// Cost-accounting name for a model alias (`"sonnet"` is what goes on
    /// the command line; pricing wants a real model id). Best effort — the
    /// CLI reports its own notional cost, which [`CliOutcome::cost_usd`]
    /// carries through, so this only matters for the telemetry label.
    pub fn pricing_model(self, model: &str) -> String {
        match self {
            CliKind::Claude => match model.to_ascii_lowercase().as_str() {
                "haiku" => "claude-haiku-4-5".to_string(),
                "sonnet" => "claude-sonnet-4-5".to_string(),
                "opus" => "claude-opus-4-5".to_string(),
                _ => model.to_string(),
            },
        }
    }

    /// Locate the binary on `PATH`. `None` when it is not installed.
    pub fn find_binary(self) -> Option<PathBuf> {
        find_on_path(self.binary())
    }
}

/// A located CLI plus the per-call settings it needs.
#[derive(Debug, Clone)]
pub struct CliBackend {
    pub kind: CliKind,
    pub binary: PathBuf,
    pub model: String,
    /// Reasoning effort as configured; mapped per CLI (Claude accepts
    /// low/medium/high via `--effort`; anything else is dropped).
    pub reasoning_effort: String,
    pub timeout: Duration,
    /// Working directory for the child. An empty, dedicated directory so the
    /// CLI discovers no project instructions (CLAUDE.md, settings, MCP
    /// servers) and cannot mistake ekko's prompt for project work.
    pub workdir: PathBuf,
}

/// What one CLI run produced.
#[derive(Debug, Clone)]
pub struct CliOutcome {
    pub content: String,
    pub usage: Option<RawUsage>,
    /// The CLI's own notional cost, when it reports one. Subscription runs
    /// are not billed per call; this is what the call *would* have cost.
    pub cost_usd: Option<f64>,
}

impl CliBackend {
    /// Run one completion: the rendered prompt goes in on stdin, the answer
    /// comes back on stdout.
    pub async fn complete(&self, messages: &[Message]) -> Result<CliOutcome, LlmError> {
        let prompt = render_prompt(messages);
        let (program, args) = self.kind.command(self);

        std::fs::create_dir_all(&self.workdir).map_err(|e| {
            LlmError::ClientInit(format!(
                "cannot create CLI working directory {}: {e}",
                self.workdir.display()
            ))
        })?;

        debug!(cli = self.kind.name(), model = %self.model, bytes = prompt.len(), "spawning CLI completion");
        let mut child = Command::new(program)
            .args(&args)
            .current_dir(&self.workdir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| LlmError::Http {
                status: None,
                body: format!("failed to start {}: {e}", self.binary.display()),
            })?;

        // Feed stdin on its own task so a CLI that starts answering before it
        // has drained a large prompt cannot deadlock against us.
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| LlmError::InvalidRequest("CLI child has no stdin".to_string()))?;
        let feeder = tokio::spawn(async move {
            let result = stdin.write_all(prompt.as_bytes()).await;
            drop(stdin);
            result
        });

        let output = match tokio::time::timeout(self.timeout, child.wait_with_output()).await {
            Ok(Ok(output)) => output,
            Ok(Err(e)) => {
                return Err(LlmError::Http {
                    status: None,
                    body: format!("{} failed: {e}", self.kind.binary()),
                });
            }
            Err(_) => {
                // `kill_on_drop` reaps the child when `wait_with_output`'s
                // future is dropped here.
                return Err(LlmError::Timeout);
            }
        };
        if let Ok(Err(e)) = feeder.await {
            // A broken pipe here means the CLI exited early; its stdout /
            // stderr below say why, so only note it.
            warn!(error = %e, "writing prompt to CLI stdin failed");
        }

        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        self.kind
            .parse_output(output.status.success(), &stdout, &stderr)
    }
}

impl CliKind {
    /// Program and argument vector for one run.
    ///
    /// No `--json-schema`: a loose `{"type":"object"}` schema made Claude Code
    /// return wrapper objects (`{"input": …}`, a schema echo) in 3 of 4 runs,
    /// and ekko's callers do not pass their exact schemas down. The prompts
    /// already demand JSON and [`super::LlmClient::call_llm_json_with`]
    /// re-asks on a parse failure; the experiment in
    /// `docs/cli-backend/EXPERIMENT.md` found the default model reliable
    /// that way. Per-caller schemas are the route to supporting smaller
    /// models on long sessions.
    fn command(self, backend: &CliBackend) -> (PathBuf, Vec<String>) {
        match self {
            CliKind::Claude => {
                let mut args: Vec<String> = vec![
                    "-p".into(),
                    // Not written to ~/.claude/projects — so ekko never
                    // indexes its own summarization runs.
                    "--no-session-persistence".into(),
                    // No Read/Bash/…: the model can only answer.
                    "--tools".into(),
                    "".into(),
                    "--output-format".into(),
                    "json".into(),
                    "--model".into(),
                    backend.model.clone(),
                ];
                if matches!(backend.reasoning_effort.as_str(), "low" | "medium" | "high") {
                    args.push("--effort".into());
                    args.push(backend.reasoning_effort.clone());
                }
                // NOTE: `--bare` is deliberately absent — it skips keychain
                // reads, so a subscription login is not found.
                (backend.binary.clone(), args)
            }
        }
    }

    /// Turn the process result into content + usage, or a typed error that
    /// the retry loop can classify.
    fn parse_output(
        self,
        success: bool,
        stdout: &str,
        stderr: &str,
    ) -> Result<CliOutcome, LlmError> {
        match self {
            CliKind::Claude => parse_claude_output(success, stdout, stderr),
        }
    }
}

/// Claude Code `--output-format json` envelope → outcome.
///
/// Exposed for tests. The envelope is one JSON object with `result` (text),
/// `is_error`, `usage.{input_tokens,output_tokens,…}`, `total_cost_usd`, and
/// — if a `--json-schema` were ever passed — `structured_output`, which is
/// preferred when it is an object.
pub fn parse_claude_output(
    success: bool,
    stdout: &str,
    stderr: &str,
) -> Result<CliOutcome, LlmError> {
    let envelope: Option<Value> = serde_json::from_str(stdout.trim()).ok();

    // Error classification first: the envelope may carry the reason even on
    // a non-zero exit, and the exit code alone says nothing useful.
    let result_text = envelope
        .as_ref()
        .and_then(|v| v.get("result"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let is_error = envelope
        .as_ref()
        .and_then(|v| v.get("is_error"))
        .and_then(Value::as_bool)
        .unwrap_or(false);

    if !success || is_error {
        let detail = if !result_text.is_empty() {
            result_text.clone()
        } else if !stderr.trim().is_empty() {
            stderr.trim().to_string()
        } else {
            stdout.trim().to_string()
        };
        return Err(classify_claude_error(&detail));
    }

    let Some(envelope) = envelope else {
        return Err(LlmError::InvalidResponse(format!(
            "claude produced no JSON envelope: {}",
            super::truncate_chars(stdout.trim(), 200)
        )));
    };

    // Prefer the schema-validated object when present; it is the answer the
    // CLI itself checked. Fall back to the free-text result.
    let content = match envelope.get("structured_output") {
        Some(obj) if obj.is_object() => obj.to_string(),
        _ => result_text,
    };

    let usage = envelope.get("usage").and_then(|u| {
        let input = u.get("input_tokens").and_then(Value::as_i64);
        let output = u.get("output_tokens").and_then(Value::as_i64);
        // Claude Code counts cached prompt tokens separately; fold them in
        // so the prompt size is honest.
        let cached = u
            .get("cache_read_input_tokens")
            .and_then(Value::as_i64)
            .unwrap_or(0)
            + u.get("cache_creation_input_tokens")
                .and_then(Value::as_i64)
                .unwrap_or(0);
        if input.is_none() && output.is_none() {
            return None;
        }
        Some(RawUsage {
            prompt_tokens: Some(input.unwrap_or(0) + cached),
            completion_tokens: output,
            completion_tokens_details: None,
        })
    });
    let cost_usd = envelope.get("total_cost_usd").and_then(Value::as_f64);

    Ok(CliOutcome {
        content,
        usage,
        cost_usd,
    })
}

/// Map Claude Code's failure text onto the retry classifier's vocabulary.
/// Text matching is unavoidable here (there is no status code); the
/// patterns are the CLI's own user-facing strings.
fn classify_claude_error(detail: &str) -> LlmError {
    let lower = detail.to_ascii_lowercase();
    if lower.contains("not logged in") || lower.contains("please run /login") {
        return LlmError::NotConfigured(super::LlmNotConfiguredError {
            model: String::new(),
            provider: "claude-cli".to_string(),
            env_var: "claude login".to_string(),
        });
    }
    if lower.contains("rate limit")
        || lower.contains("usage limit")
        || lower.contains("limit reached")
    {
        // Same wait as an HTTP 429: sit out the window.
        return LlmError::Http {
            status: Some(429),
            body: super::truncate_chars(detail, 500),
        };
    }
    LlmError::Http {
        status: None,
        body: super::truncate_chars(detail, 500),
    }
}

/// Flatten a message list into one stdin prompt.
///
/// ekko's calls are a single user message, which goes through verbatim. The
/// JSON re-ask loop appends the model's previous answer and a nudge; the CLI
/// has no multi-turn input in `-p` mode, so those are rendered inline.
pub fn render_prompt(messages: &[Message]) -> String {
    if let [only] = messages
        && only.role == "user"
    {
        return only.content.clone();
    }
    let mut out = String::new();
    for (i, m) in messages.iter().enumerate() {
        if i > 0 {
            out.push_str("\n\n");
        }
        match m.role.as_str() {
            "user" => out.push_str(&m.content),
            "assistant" => {
                out.push_str("[Your previous answer]\n");
                out.push_str(&m.content);
            }
            "system" => {
                out.push_str("[Instructions]\n");
                out.push_str(&m.content);
            }
            other => {
                out.push('[');
                out.push_str(other);
                out.push_str("]\n");
                out.push_str(&m.content);
            }
        }
    }
    out
}

/// `which`, minus the dependency: first executable named `name` on `PATH`.
pub fn find_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| is_executable(candidate))
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    path.is_file()
        && std::fs::metadata(path)
            .map(|m| m.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
        || ["exe", "cmd", "bat"]
            .iter()
            .any(|ext| path.with_extension(ext).is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_prompt_passes_single_user_message_through() {
        let msgs = [Message::user("summarize this")];
        assert_eq!(render_prompt(&msgs), "summarize this");
    }

    #[test]
    fn render_prompt_inlines_reask_turns() {
        let msgs = [
            Message::user("give me json"),
            Message::assistant("sure, here you go"),
            Message::user("JSON only please"),
        ];
        let out = render_prompt(&msgs);
        assert!(out.starts_with("give me json\n\n[Your previous answer]\nsure, here you go"));
        assert!(out.ends_with("JSON only please"));
    }

    #[test]
    fn claude_output_prefers_structured_output() {
        let stdout = r#"{"type":"result","is_error":false,"result":"```json\n{\"a\":1}\n```","structured_output":{"a":1},"usage":{"input_tokens":10,"output_tokens":5,"cache_read_input_tokens":90},"total_cost_usd":0.01}"#;
        let out = parse_claude_output(true, stdout, "").expect("parses");
        assert_eq!(out.content, r#"{"a":1}"#);
        let usage = out.usage.expect("usage");
        assert_eq!(usage.prompt_tokens, Some(100));
        assert_eq!(usage.completion_tokens, Some(5));
        assert_eq!(out.cost_usd, Some(0.01));
    }

    #[test]
    fn claude_output_falls_back_to_result_text() {
        let stdout =
            r#"{"is_error":false,"result":"hello","usage":{"input_tokens":1,"output_tokens":1}}"#;
        let out = parse_claude_output(true, stdout, "").expect("parses");
        assert_eq!(out.content, "hello");
    }

    #[test]
    fn claude_not_logged_in_is_not_configured() {
        let stdout = r#"{"is_error":true,"result":"Not logged in · Please run /login"}"#;
        let err = parse_claude_output(false, stdout, "").expect_err("error");
        assert!(matches!(err, LlmError::NotConfigured(_)), "got {err:?}");
    }

    #[test]
    fn claude_usage_limit_maps_to_429() {
        let stdout =
            r#"{"is_error":true,"result":"You've hit your usage limit. Try again at 3pm."}"#;
        let err = parse_claude_output(true, stdout, "").expect_err("error");
        assert!(
            matches!(
                err,
                LlmError::Http {
                    status: Some(429),
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn claude_nonzero_exit_without_envelope_is_transient() {
        let err = parse_claude_output(false, "", "segfault or whatever").expect_err("error");
        assert!(
            matches!(err, LlmError::Http { status: None, .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn claude_success_without_envelope_is_invalid_response() {
        let err = parse_claude_output(true, "plain text", "").expect_err("error");
        assert!(matches!(err, LlmError::InvalidResponse(_)), "got {err:?}");
    }

    #[test]
    fn claude_command_line_shape() {
        let backend = CliBackend {
            kind: CliKind::Claude,
            binary: PathBuf::from("/usr/bin/claude"),
            model: "sonnet".into(),
            reasoning_effort: "low".into(),
            timeout: Duration::from_secs(1),
            workdir: PathBuf::from("/tmp/x"),
        };
        let (_, args) = CliKind::Claude.command(&backend);
        let joined = args.join(" ");
        assert!(
            joined.contains(
                "-p --no-session-persistence --tools  --output-format json --model sonnet"
            )
        );
        assert!(joined.contains("--effort low"));
        assert!(!joined.contains("--json-schema"));
        assert!(!joined.contains("--bare"));
    }

    #[test]
    fn unsupported_effort_is_dropped() {
        let backend = CliBackend {
            kind: CliKind::Claude,
            binary: PathBuf::from("claude"),
            model: "sonnet".into(),
            reasoning_effort: "minimal".into(),
            timeout: Duration::from_secs(1),
            workdir: PathBuf::from("/tmp/x"),
        };
        let (_, args) = CliKind::Claude.command(&backend);
        assert!(!args.iter().any(|a| a == "--effort"));
    }
}
