//! Self-check for ekko (`ekko check`).
//!
//! Verifies that every component is working and reports issues with
//! actionable guidance: gather facts, emit pass/warn/fail lines grouped in
//! sections, print a tally, exit non-zero iff anything failed.
//!
//! The gathering and the formatting are separated: [`probe`] does the I/O and
//! returns a [`CheckInputs`] snapshot, [`assemble`] turns that snapshot into a
//! [`CheckReport`] with no I/O at all (which is what the tests exercise).
//!
//! Notes on what is and is not checked:
//!
//! * The API-key list is OpenAI, Anthropic and Azure OpenAI, and keys are read
//!   through [`crate::config::Config::get_api_key`], because ekko never
//!   injects config.json keys into the process environment.
//! * Prompts are embedded at compile time, so "prompts directory missing" is
//!   unreachable; the check verifies instead that all expected templates are
//!   present and render.
//! * The connectivity probe makes one attempt and sends no `max_tokens` —
//!   ekko's call shape omits it.
//! * The data directory is probed for writability, not just existence.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::config::Config;
use crate::llm::{self, Message};
use crate::{paths, prompts};

/// The prompt templates the server expects; the health check proves every
/// one is embedded and renders.
pub const EXPECTED_PROMPTS: [&str; 6] = [
    "summarize",
    "context_snapshot",
    "semantic_search",
    "qa_from_summaries",
    "qa_transcript",
    "qa_synthesize",
];

/// API keys ekko can use, with the feature each enables.
pub const KEY_CHECKS: [(&str, &str); 3] = [
    ("OPENAI_API_KEY", "OpenAI (gpt-* models)"),
    ("ANTHROPIC_API_KEY", "Anthropic (claude-* models)"),
    (
        "AZURE_OPENAI_API_KEY",
        "Azure OpenAI (gpt-* models via AZURE_OPENAI_ENDPOINT)",
    ),
];

// ---------------------------------------------------------------------------
// Report model
// ---------------------------------------------------------------------------

/// Outcome of a single check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Pass,
    Warn,
    Fail,
}

impl Status {
    fn marker(self, color: bool) -> &'static str {
        match (self, color) {
            (Status::Pass, true) => "\u{1b}[92m✓\u{1b}[0m",
            (Status::Warn, true) => "\u{1b}[93m!\u{1b}[0m",
            (Status::Fail, true) => "\u{1b}[91m✗\u{1b}[0m",
            (Status::Pass, false) => "✓",
            (Status::Warn, false) => "!",
            (Status::Fail, false) => "✗",
        }
    }
}

/// One reported line, with an optional `→ fix` hint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckEntry {
    pub status: Status,
    pub message: String,
    pub fix: Option<String>,
}

impl CheckEntry {
    /// A passing entry; it never carries a fix hint.
    pub fn passed(message: impl Into<String>) -> Self {
        CheckEntry {
            status: Status::Pass,
            message: message.into(),
            fix: None,
        }
    }

    /// A warning, optionally with a fix hint.
    pub fn warned(message: impl Into<String>, fix: Option<String>) -> Self {
        CheckEntry {
            status: Status::Warn,
            message: message.into(),
            fix,
        }
    }

    /// A failure, optionally with a fix hint.
    pub fn failed(message: impl Into<String>, fix: Option<String>) -> Self {
        CheckEntry {
            status: Status::Fail,
            message: message.into(),
            fix,
        }
    }
}

/// A titled group of entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckSection {
    pub title: String,
    pub entries: Vec<CheckEntry>,
}

/// Pass/warn/fail tallies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CheckCounts {
    pub pass: usize,
    pub warn: usize,
    pub fail: usize,
}

impl CheckCounts {
    /// Total number of checks run.
    pub fn total(self) -> usize {
        self.pass + self.warn + self.fail
    }
}

/// The whole report.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CheckReport {
    pub sections: Vec<CheckSection>,
}

impl CheckReport {
    /// Tally entries by status.
    pub fn counts(&self) -> CheckCounts {
        let mut counts = CheckCounts::default();
        for entry in self.sections.iter().flat_map(|s| &s.entries) {
            match entry.status {
                Status::Pass => counts.pass += 1,
                Status::Warn => counts.warn += 1,
                Status::Fail => counts.fail += 1,
            }
        }
        counts
    }

    /// `1` when anything failed, else `0`.
    pub fn exit_code(&self) -> i32 {
        i32::from(self.counts().fail > 0)
    }

    /// Render the report as it is printed to stderr. `color` controls ANSI
    /// escapes so tests can assert on plain text.
    pub fn render(&self, color: bool) -> String {
        let bold = if color { "\u{1b}[1m" } else { "" };
        let reset = if color { "\u{1b}[0m" } else { "" };
        let mut out = String::new();

        for section in &self.sections {
            let _ = write!(out, "\n{bold}{}{reset}\n", section.title);
            for entry in &section.entries {
                let _ = writeln!(out, "  {} {}", entry.status.marker(color), entry.message);
                if let Some(fix) = &entry.fix {
                    let _ = writeln!(out, "      → {fix}");
                }
            }
        }

        let c = self.counts();
        let _ = write!(out, "\n{}\n", "─".repeat(50));
        let mut parts: Vec<String> = Vec::new();
        if c.pass > 0 {
            parts.push(colorize(&format!("{} passed", c.pass), "92", color));
        }
        if c.warn > 0 {
            parts.push(colorize(&format!("{} warnings", c.warn), "93", color));
        }
        if c.fail > 0 {
            parts.push(colorize(&format!("{} failed", c.fail), "91", color));
        }
        let _ = writeln!(out, "  {} ({} checks)", parts.join(", "), c.total());

        if c.fail == 0 && c.warn == 0 {
            let _ = write!(out, "\n  {bold}Everything looks good! 🎉{reset}\n");
        } else if c.fail == 0 {
            let _ = write!(
                out,
                "\n  {bold}Core functionality works. Warnings are optional improvements.{reset}\n"
            );
        } else {
            let _ = write!(
                out,
                "\n  {bold}Some issues need attention — see failed items above.{reset}\n"
            );
        }
        out
    }

    /// Print the report to stderr (stdout is reserved for JSON-RPC).
    pub fn print(&self) {
        eprint!("{}", self.render(true));
    }
}

fn colorize(text: &str, code: &str, color: bool) -> String {
    if color {
        format!("\u{1b}[{code}m{text}\u{1b}[0m")
    } else {
        text.to_string()
    }
}

// ---------------------------------------------------------------------------
// Gathered facts
// ---------------------------------------------------------------------------

/// State of the sessions database.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DbState {
    /// File not created yet.
    Missing,
    /// Opened; `sessions` row count.
    Ok(i64),
    /// File exists but could not be opened/queried.
    Broken(String),
}

/// What happened on the connectivity probe.
#[derive(Debug, Clone, PartialEq)]
pub enum LlmProbe {
    /// Skipped because no API key is configured.
    Skipped,
    /// Response text plus round-trip seconds.
    Ok { content: String, seconds: f64 },
    /// The error string from the failed call.
    Failed(String),
}

/// Per-CLI session-discovery result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoverySource {
    pub cli_source: &'static str,
    /// Every candidate path for this platform.
    pub candidates: Vec<PathBuf>,
    /// Candidates that exist on disk.
    pub existing: Vec<PathBuf>,
    /// Deduplicated transcript count across `existing`.
    pub session_count: usize,
}

/// Everything [`assemble`] needs, with no I/O of its own.
#[derive(Debug, Clone, PartialEq)]
pub struct CheckInputs {
    pub platform: String,
    pub data_dir: Result<PathBuf, String>,
    pub data_dir_writable: bool,
    pub config_path: PathBuf,
    pub config_exists: bool,
    /// `(env var name, description, present)` in [`KEY_CHECKS`] order, with
    /// the masked value when present.
    pub api_keys: Vec<(&'static str, &'static str, Option<String>)>,
    pub model: String,
    pub provider: Option<String>,
    /// Resolved completion backend (`api` / `claude-cli`) and why.
    pub backend: String,
    pub backend_reason: String,
    pub llm_available: (bool, String),
    pub llm_probe: LlmProbe,
    /// The interactive-query role (`ask_question`, semantic search) when it
    /// resolves differently from indexing: `(backend, model, reason)` and
    /// its own probe.
    pub query: Option<(String, String, String)>,
    pub query_probe: Option<LlmProbe>,
    /// Templates that are present and render.
    pub prompts_loadable: usize,
    /// Expected templates that are missing.
    pub prompts_missing: Vec<String>,
    pub db_path: PathBuf,
    pub db_state: DbState,
    pub discovery: Vec<DiscoverySource>,
}

impl CheckInputs {
    fn any_key(&self) -> bool {
        self.api_keys.iter().any(|(_, _, v)| v.is_some())
    }
}

/// Mask a secret: 8 leading + 4 trailing characters when longer than 16, else
/// the first 4 characters.
pub fn mask_key(value: &str) -> String {
    let chars: Vec<char> = value.chars().collect();
    if chars.len() > 16 {
        let head: String = chars[..8].iter().collect();
        let tail: String = chars[chars.len() - 4..].iter().collect();
        format!("{head}...{tail}")
    } else {
        let head: String = chars.iter().take(4).collect();
        format!("{head}...")
    }
}

// ---------------------------------------------------------------------------
// Assembly (pure)
// ---------------------------------------------------------------------------

/// Turn gathered facts into the printable report. No I/O.
pub fn assemble(inputs: &CheckInputs) -> CheckReport {
    let mut sections: Vec<CheckSection> = Vec::new();

    // ── 1. Platform ────────────────────────────────────────────────────
    sections.push(CheckSection {
        title: "Platform".to_string(),
        entries: vec![CheckEntry::passed(format!(
            "Platform detected: {}",
            inputs.platform
        ))],
    });

    // ── 2. Config & API keys ───────────────────────────────────────────
    let mut entries = Vec::new();
    if inputs.config_exists {
        entries.push(CheckEntry::passed(format!(
            "config.json found: {}",
            inputs.config_path.display()
        )));
    } else {
        entries.push(CheckEntry::warned(
            format!("No config.json at {}", inputs.config_path.display()),
            Some(
                "Create config.json with an API key or llm_backend=claude-cli (see README.md)"
                    .to_string(),
            ),
        ));
    }
    for (var, desc, value) in &inputs.api_keys {
        if let Some(masked) = value {
            entries.push(CheckEntry::passed(format!(
                "{var} is set ({masked}) — enables {desc}"
            )));
        }
    }
    if !inputs.any_key() {
        if inputs.backend == "api" {
            entries.push(CheckEntry::failed(
                "No API keys found — LLM features (summaries, Q&A, semantic search) will not work",
                Some(
                    "Add at least one API key to config.json or set via environment (see README.md), \
or install and log in to Claude Code to use it as the backend"
                        .to_string(),
                ),
            ));
        } else {
            entries.push(CheckEntry::passed(format!(
                "No API keys — using the {} backend instead",
                inputs.backend
            )));
        }
    }
    sections.push(CheckSection {
        title: "Environment & API Keys".to_string(),
        entries,
    });

    // ── 3. Configured model ────────────────────────────────────────────
    let provider = inputs.provider.as_deref().unwrap_or("unknown");
    let (available, msg) = &inputs.llm_available;
    let is_cli = inputs.backend != "api";
    let entry = if *available {
        if is_cli {
            CheckEntry::passed(format!(
                "Configured model: {} (backend: {}) — ready",
                inputs.model, inputs.backend
            ))
        } else {
            CheckEntry::passed(format!(
                "Configured model: {} (provider: {provider}) — ready",
                inputs.model
            ))
        }
    } else if is_cli {
        CheckEntry::failed(
            format!(
                "Configured model: {} (backend: {}) — CLI not found",
                inputs.model, inputs.backend
            ),
            Some(msg.clone()),
        )
    } else {
        CheckEntry::failed(
            format!(
                "Configured model: {} (provider: {provider}) — key missing",
                inputs.model
            ),
            Some(msg.clone()),
        )
    };
    let mut entries = vec![entry];
    entries.push(CheckEntry::passed(format!(
        "Indexing backend: {} — {}",
        inputs.backend, inputs.backend_reason
    )));
    match &inputs.query {
        Some((backend, model, reason)) => entries.push(CheckEntry::passed(format!(
            "Query backend (ask_question, semantic search): {backend}, model {model} — {reason}"
        ))),
        None => entries.push(CheckEntry::passed(
            "Query backend (ask_question, semantic search): same as indexing",
        )),
    }
    sections.push(CheckSection {
        title: "LLM Configuration".to_string(),
        entries,
    });

    // ── 4. LLM test call ───────────────────────────────────────────────
    let mut entries = vec![probe_entry(&inputs.llm_probe, "LLM test")];
    if let Some(probe) = &inputs.query_probe {
        entries.push(probe_entry(probe, "query-backend test"));
    }
    sections.push(CheckSection {
        title: "LLM Connectivity".to_string(),
        entries,
    });

    // ── 5. Prompt templates ────────────────────────────────────────────
    let mut entries = Vec::new();
    let total_prompts = prompts::list_prompts().len();
    if total_prompts > 0 {
        entries.push(CheckEntry::passed(format!(
            "Embedded prompts: {total_prompts} templates found"
        )));
    } else {
        entries.push(CheckEntry::failed(
            "No prompt templates are embedded in the binary",
            Some("Rebuild ekko from a tree containing prompts/*.md".to_string()),
        ));
    }
    if inputs.prompts_missing.is_empty() {
        entries.push(CheckEntry::passed(format!(
            "All {} expected prompt templates loadable",
            inputs.prompts_loadable
        )));
    } else {
        entries.push(CheckEntry::failed(
            format!(
                "Missing prompt templates: {}",
                inputs.prompts_missing.join(", ")
            ),
            Some("Rebuild ekko from a tree containing prompts/*.md".to_string()),
        ));
    }
    sections.push(CheckSection {
        title: "Prompt Templates".to_string(),
        entries,
    });

    // ── 6. Database ────────────────────────────────────────────────────
    let mut entries = Vec::new();
    match &inputs.data_dir {
        Ok(dir) => {
            entries.push(CheckEntry::passed(format!(
                "Data directory: {}",
                dir.display()
            )));
            if !inputs.data_dir_writable {
                entries.push(CheckEntry::failed(
                    format!("Data directory is not writable: {}", dir.display()),
                    Some("Fix the directory permissions or set XDG_DATA_HOME".to_string()),
                ));
            }
        }
        Err(e) => entries.push(CheckEntry::failed(
            format!("Database check failed: {e}"),
            Some("Could not determine the data directory".to_string()),
        )),
    }
    match &inputs.db_state {
        DbState::Ok(count) => {
            entries.push(CheckEntry::passed(format!(
                "Database: {count} sessions indexed"
            )));
        }
        DbState::Broken(e) => {
            entries.push(CheckEntry::warned(
                format!("Database exists but may need initialization: {e}"),
                None,
            ));
        }
        DbState::Missing => {
            entries.push(CheckEntry::warned(
                format!("Database not yet created at {}", inputs.db_path.display()),
                Some("It will be created on first use".to_string()),
            ));
        }
    }
    sections.push(CheckSection {
        title: "Database".to_string(),
        entries,
    });

    // ── 7. Session discovery ───────────────────────────────────────────
    let mut entries = Vec::new();
    let mut total_sessions = 0usize;
    for source in &inputs.discovery {
        let cli_source = source.cli_source;
        if source.candidates.is_empty() {
            entries.push(CheckEntry::warned(
                format!(
                    "{cli_source}: no candidate paths configured for {}",
                    inputs.platform
                ),
                None,
            ));
            continue;
        }
        if source.existing.is_empty() {
            entries.push(CheckEntry::warned(
                format!("{cli_source}: no session directories found"),
                Some(format!("Expected at: {}", join_paths(&source.candidates))),
            ));
            continue;
        }
        total_sessions += source.session_count;
        let path_str = join_paths(&source.existing);
        if source.session_count > 0 {
            entries.push(CheckEntry::passed(format!(
                "{cli_source}: {} sessions ({path_str})",
                source.session_count
            )));
        } else {
            entries.push(CheckEntry::warned(
                format!("{cli_source}: directory exists but no sessions yet ({path_str})"),
                None,
            ));
        }
    }
    if total_sessions > 0 {
        entries.push(CheckEntry::passed(format!(
            "Total: {total_sessions} sessions discoverable"
        )));
    } else {
        entries.push(CheckEntry::warned(
            "No sessions found from any source",
            Some(
                "Use Claude Code, GitHub Copilot CLI, VS Code Copilot, or Codex to create sessions"
                    .to_string(),
            ),
        ));
    }
    sections.push(CheckSection {
        title: "Session Discovery".to_string(),
        entries,
    });

    CheckReport { sections }
}

fn join_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

use crate::parsers::take_chars as truncate;

// ---------------------------------------------------------------------------
// Probing (I/O)
// ---------------------------------------------------------------------------

/// Gather every fact the report needs. Performs a real LLM call only when an
/// API key is configured.
pub async fn probe(config: &Config) -> CheckInputs {
    let platform = paths::get_platform().as_str().to_string();
    let data_dir = paths::get_data_dir().map_err(|e| e.to_string());

    let (config_path, data_dir_writable) = match &data_dir {
        Ok(dir) => (dir.join("config.json"), is_writable(dir)),
        Err(_) => (PathBuf::from("config.json"), false),
    };
    let config_exists = config_path.exists();

    let api_keys: Vec<(&'static str, &'static str, Option<String>)> = KEY_CHECKS
        .iter()
        .map(|(var, desc)| {
            let masked = config.get_api_key(var).map(|v| mask_key(&v));
            (*var, *desc, masked)
        })
        .collect();
    let any_key = api_keys.iter().any(|(_, _, v)| v.is_some());

    let resolved = llm::resolve_backend(config, llm::Role::Index);
    let model = resolved.model.clone();
    let provider = match resolved.backend {
        llm::Backend::Api => llm::get_required_provider(&model).map(|p| p.name().to_string()),
        llm::Backend::Cli(kind) => Some(kind.name().to_string()),
    };
    let llm_available = llm::backend_availability(config, &resolved);
    let backend = resolved.backend.name().to_string();
    let backend_reason = resolved.reason.clone();

    // Probe whenever something could answer: a key for the API, or a
    // located CLI (the probe is also how "installed but not logged in"
    // surfaces).
    let can_probe = |r: &llm::ResolvedBackend, available: bool| {
        any_key || (r.backend != llm::Backend::Api && available)
    };
    let llm_probe = if can_probe(&resolved, llm_available.0) {
        probe_llm(config, llm::Role::Index).await
    } else {
        LlmProbe::Skipped
    };

    // The query role gets its own line and probe only when it differs.
    let query_resolved = llm::resolve_backend(config, llm::Role::Query);
    let (query, query_probe) =
        if query_resolved.backend != resolved.backend || query_resolved.model != resolved.model {
            let query_available = llm::backend_availability(config, &query_resolved);
            let probe = if can_probe(&query_resolved, query_available.0) {
                probe_llm(config, llm::Role::Query).await
            } else {
                LlmProbe::Skipped
            };
            (
                Some((
                    query_resolved.backend.name().to_string(),
                    query_resolved.model.clone(),
                    query_resolved.reason.clone(),
                )),
                Some(probe),
            )
        } else {
            (None, None)
        };

    // Prompts are embedded, so "loadable" means "present and renders".
    let mut prompts_loadable = 0usize;
    let mut prompts_missing: Vec<String> = Vec::new();
    for name in EXPECTED_PROMPTS {
        if prompts::has_prompt(name) {
            prompts_loadable += 1;
        } else {
            prompts_missing.push(name.to_string());
        }
    }

    let db_path = match &data_dir {
        Ok(dir) => dir.join("sessions.db"),
        Err(_) => PathBuf::from("sessions.db"),
    };
    let db_state = if db_path.exists() {
        match crate::db::Database::new(&db_path).and_then(|db| db.get_stats()) {
            // Report the raw row count: total_sessions filters to
            // message_count > 0, total_indexed does not.
            Ok(stats) => DbState::Ok(stats.total_indexed),
            Err(e) => DbState::Broken(e.to_string()),
        }
    } else {
        DbState::Missing
    };

    let discovery = probe_discovery();

    CheckInputs {
        platform,
        data_dir,
        data_dir_writable,
        config_path,
        config_exists,
        api_keys,
        model,
        provider,
        backend,
        backend_reason,
        llm_available,
        llm_probe,
        query,
        query_probe,
        prompts_loadable,
        prompts_missing,
        db_path,
        db_state,
        discovery,
    }
}

/// Render one probe result. `what` is "LLM test" (the indexing backend —
/// wording preserved from before query backends existed) or
/// "query-backend test".
fn probe_entry(probe: &LlmProbe, what: &str) -> CheckEntry {
    // "LLM test" → "LLM call …", "query-backend test" → "Query-backend call …".
    let label = |what: &str| -> String {
        let base = what.trim_end_matches(" test");
        if base == "LLM" {
            base.to_string()
        } else {
            let mut c = base.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        }
    };
    match probe {
        LlmProbe::Skipped => CheckEntry::warned(
            format!("Skipping {what} — no API keys configured and no CLI backend"),
            None,
        ),
        LlmProbe::Ok { content, seconds } => CheckEntry::passed(format!(
            "{} call succeeded in {seconds:.1}s — response: \"{content}\"",
            label(what)
        )),
        LlmProbe::Failed(err) => {
            let lower = err.to_lowercase();
            if lower.contains("api key") || lower.contains("auth") {
                CheckEntry::failed(
                    format!("{} call failed — authentication error", label(what)),
                    Some(format!(
                        "Check that your API key is valid: {}",
                        truncate(err, 120)
                    )),
                )
            } else if lower.contains("rate") {
                CheckEntry::warned(
                    format!(
                        "{} call rate-limited — key works but may need a moment",
                        label(what)
                    ),
                    Some("Wait a minute and try again".to_string()),
                )
            } else {
                CheckEntry::failed(
                    format!("{} call failed: {}", label(what), truncate(err, 150)),
                    Some("Check your API key and network connectivity".to_string()),
                )
            }
        }
    }
}

async fn probe_llm(config: &Config, role: llm::Role) -> LlmProbe {
    let client = match llm::LlmClient::for_role(config, role) {
        Ok(c) => c,
        Err(e) => return LlmProbe::Failed(e.to_string()),
    };
    let start = std::time::Instant::now();
    // One attempt, no retries: this is a liveness probe, not real work. The
    // call shape carries no `max_tokens`.
    match client
        .call_llm_with(&[Message::user("Reply with exactly: OK")], 1, None)
        .await
    {
        Ok(response) => LlmProbe::Ok {
            content: response.content.trim().to_string(),
            seconds: start.elapsed().as_secs_f64(),
        },
        Err(e) => LlmProbe::Failed(e.to_string()),
    }
}

fn probe_discovery() -> Vec<DiscoverySource> {
    let Ok(session_paths) = paths::get_configured_session_paths() else {
        return Vec::new();
    };
    let mut sources = Vec::new();
    for cli_source in crate::indexer::PARSER_ORDER {
        let candidates: Vec<PathBuf> = session_paths
            .iter()
            .find(|(name, _)| *name == cli_source)
            .map(|(_, p)| p.to_vec())
            .unwrap_or_default();
        let existing: Vec<PathBuf> = candidates.iter().filter(|p| p.exists()).cloned().collect();

        let mut session_count = 0usize;
        if let Some(parser) = crate::parsers::parser_for(cli_source) {
            let mut seen = std::collections::HashSet::new();
            for base in &existing {
                for path in parser.discover_sessions(base) {
                    if seen.insert(parser.discovery_key(&path)) {
                        session_count += 1;
                    }
                }
            }
        }

        sources.push(DiscoverySource {
            cli_source,
            candidates,
            existing,
            session_count,
        });
    }
    sources
}

/// Can we create files in `dir`?
fn is_writable(dir: &Path) -> bool {
    let probe = dir.join(".ekko-write-probe");
    match std::fs::write(&probe, b"") {
        Ok(()) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

/// Run all checks, print them to stderr, and return the process exit code.
pub async fn run_checks() -> i32 {
    let inputs = probe(crate::config::get_config()).await;
    let report = assemble(&inputs);
    report.print();
    report.exit_code()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// A "everything is fine" snapshot, mutated per test.
    fn healthy(dir: &Path) -> CheckInputs {
        CheckInputs {
            platform: "linux".to_string(),
            data_dir: Ok(dir.to_path_buf()),
            data_dir_writable: true,
            config_path: dir.join("config.json"),
            config_exists: true,
            api_keys: vec![
                (
                    "OPENAI_API_KEY",
                    "OpenAI (gpt-* models)",
                    Some("sk-abcde...wxyz".to_string()),
                ),
                ("ANTHROPIC_API_KEY", "Anthropic (claude-* models)", None),
            ],
            model: "gpt-5-mini".to_string(),
            provider: Some("openai".to_string()),
            backend: "api".to_string(),
            backend_reason: "API key found for model 'gpt-5-mini'".to_string(),
            llm_available: (true, "API key found for openai".to_string()),
            query: None,
            query_probe: None,
            llm_probe: LlmProbe::Ok {
                content: "OK".to_string(),
                seconds: 0.42,
            },
            prompts_loadable: EXPECTED_PROMPTS.len(),
            prompts_missing: Vec::new(),
            db_path: dir.join("sessions.db"),
            db_state: DbState::Ok(37),
            discovery: vec![DiscoverySource {
                cli_source: "claude_code",
                candidates: vec![dir.join(".claude/projects")],
                existing: vec![dir.join(".claude/projects")],
                session_count: 12,
            }],
        }
    }

    #[test]
    fn healthy_environment_passes_everything() {
        let tmp = TempDir::new().unwrap();
        let report = assemble(&healthy(tmp.path()));
        let counts = report.counts();
        assert_eq!(counts.fail, 0, "{}", report.render(false));
        assert_eq!(counts.warn, 0, "{}", report.render(false));
        assert_eq!(report.exit_code(), 0);

        let text = report.render(false);
        assert!(text.contains("Platform detected: linux"));
        assert!(text.contains("OPENAI_API_KEY is set (sk-abcde...wxyz) — enables OpenAI"));
        assert!(text.contains("Configured model: gpt-5-mini (provider: openai) — ready"));
        assert!(text.contains("LLM call succeeded in 0.4s — response: \"OK\""));
        assert!(text.contains("All 6 expected prompt templates loadable"));
        assert!(text.contains("Database: 37 sessions indexed"));
        assert!(text.contains("claude_code: 12 sessions"));
        assert!(text.contains("Total: 12 sessions discoverable"));
        assert!(text.contains("Everything looks good!"));
        // Sections appear in this exact order.
        let titles: Vec<&str> = report.sections.iter().map(|s| s.title.as_str()).collect();
        assert_eq!(
            titles,
            vec![
                "Platform",
                "Environment & API Keys",
                "LLM Configuration",
                "LLM Connectivity",
                "Prompt Templates",
                "Database",
                "Session Discovery"
            ]
        );
    }

    #[test]
    fn missing_keys_fail_and_skip_the_llm_probe() {
        let tmp = TempDir::new().unwrap();
        let mut inputs = healthy(tmp.path());
        inputs.api_keys.iter_mut().for_each(|k| k.2 = None);
        inputs.llm_probe = LlmProbe::Skipped;
        inputs.llm_available = (
            false,
            "Model 'gpt-5-mini' requires OPENAI_API_KEY".to_string(),
        );
        inputs.config_exists = false;

        let report = assemble(&inputs);
        let text = report.render(false);
        assert!(text.contains("No API keys found"));
        assert!(text.contains("No config.json at"));
        assert!(text.contains("Skipping LLM test — no API keys configured"));
        assert!(text.contains("key missing"));
        assert!(text.contains("→ Model 'gpt-5-mini' requires OPENAI_API_KEY"));
        assert_eq!(report.exit_code(), 1);
        assert!(text.contains("Some issues need attention"));
    }

    #[test]
    fn llm_errors_are_classified_by_status() {
        let tmp = TempDir::new().unwrap();
        let mut inputs = healthy(tmp.path());

        inputs.llm_probe = LlmProbe::Failed("401 Invalid API key provided".to_string());
        let text = assemble(&inputs).render(false);
        assert!(text.contains("LLM call failed — authentication error"));

        inputs.llm_probe = LlmProbe::Failed("429 Rate limit exceeded".to_string());
        let report = assemble(&inputs);
        assert!(report.render(false).contains("rate-limited"));
        assert_eq!(report.counts().fail, 0);
        assert_eq!(report.exit_code(), 0);

        inputs.llm_probe = LlmProbe::Failed("connection refused".to_string());
        let text = assemble(&inputs).render(false);
        assert!(text.contains("LLM call failed: connection refused"));
        assert!(text.contains("Check your API key and network connectivity"));
    }

    #[test]
    fn missing_prompts_and_databases_are_reported() {
        let tmp = TempDir::new().unwrap();
        let mut inputs = healthy(tmp.path());
        inputs.prompts_missing = vec!["summarize".to_string()];
        inputs.prompts_loadable = 8;
        inputs.db_state = DbState::Missing;

        let report = assemble(&inputs);
        let text = report.render(false);
        assert!(text.contains("Missing prompt templates: summarize"));
        assert!(text.contains("Database not yet created at"));
        assert_eq!(report.exit_code(), 1);

        inputs.db_state = DbState::Broken("file is not a database".to_string());
        let text = assemble(&inputs).render(false);
        assert!(
            text.contains("Database exists but may need initialization: file is not a database")
        );
    }

    #[test]
    fn unwritable_data_dir_fails() {
        let tmp = TempDir::new().unwrap();
        let mut inputs = healthy(tmp.path());
        inputs.data_dir_writable = false;
        let report = assemble(&inputs);
        assert!(
            report
                .render(false)
                .contains("Data directory is not writable")
        );
        assert_eq!(report.exit_code(), 1);
    }

    #[test]
    fn discovery_states_warn_without_failing() {
        let tmp = TempDir::new().unwrap();
        let mut inputs = healthy(tmp.path());
        inputs.discovery = vec![
            DiscoverySource {
                cli_source: "claude_code",
                candidates: Vec::new(),
                existing: Vec::new(),
                session_count: 0,
            },
            DiscoverySource {
                cli_source: "github_copilot",
                candidates: vec![PathBuf::from("/nope")],
                existing: Vec::new(),
                session_count: 0,
            },
            DiscoverySource {
                cli_source: "vscode_copilot",
                candidates: vec![PathBuf::from("/here")],
                existing: vec![PathBuf::from("/here")],
                session_count: 0,
            },
        ];
        let report = assemble(&inputs);
        let text = report.render(false);
        assert!(text.contains("claude_code: no candidate paths configured for linux"));
        assert!(text.contains("github_copilot: no session directories found"));
        assert!(text.contains("Expected at: /nope"));
        assert!(text.contains("vscode_copilot: directory exists but no sessions yet (/here)"));
        assert!(text.contains("No sessions found from any source"));
        assert_eq!(report.counts().fail, 0);
        assert_eq!(report.counts().warn, 4);
        assert!(text.contains("Core functionality works."));
    }

    #[test]
    fn counts_and_exit_code_track_statuses() {
        let report = CheckReport {
            sections: vec![CheckSection {
                title: "T".to_string(),
                entries: vec![
                    CheckEntry::passed("a"),
                    CheckEntry::warned("b", None),
                    CheckEntry::failed("c", Some("fix it".to_string())),
                ],
            }],
        };
        let counts = report.counts();
        assert_eq!(
            (counts.pass, counts.warn, counts.fail, counts.total()),
            (1, 1, 1, 3)
        );
        assert_eq!(report.exit_code(), 1);
        let text = report.render(false);
        assert!(text.contains("1 passed, 1 warnings, 1 failed (3 checks)"));
        assert!(text.contains("      → fix it"));
        // ANSI escapes only when asked for.
        assert!(!text.contains('\u{1b}'));
        assert!(report.render(true).contains('\u{1b}'));
    }

    #[test]
    fn key_masking_is_pinned() {
        // Longer than 16 characters → first 8 + "..." + last 4.
        assert_eq!(mask_key("sk-abcdefghijklmnopqrstuv"), "sk-abcde...stuv");
        // 16 or fewer → first 4 + "...".
        assert_eq!(mask_key("short"), "shor...");
        assert_eq!(mask_key("0123456789abcdef"), "0123...");
        assert_eq!(mask_key("0123456789abcdefg"), "01234567...defg");
    }

    #[test]
    fn expected_prompts_are_all_embedded() {
        for name in EXPECTED_PROMPTS {
            assert!(prompts::has_prompt(name), "missing embedded prompt {name}");
        }
        assert_eq!(prompts::list_prompts().len(), EXPECTED_PROMPTS.len());
    }
}
