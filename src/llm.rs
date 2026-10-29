//! LLM call layer.
//!
//! ekko speaks two wire protocols directly — **OpenAI chat completions and
//! Anthropic messages** — with no provider-abstraction SDK in between. That
//! keeps the dependency tree small and startup instant.
//!
//! What this module owns:
//! - the provider-routing table and its startswith-then-substring lookup,
//! - the availability-check / `require_llm` messages,
//! - `escape_special_tokens` applied to every message before sending,
//! - retry policy: up to `max_retries` attempts, `min(2^attempt, 60)`s backoff,
//! - the JSON recovery trio (direct parse → strip fences → brace scan),
//! - the JSON re-ask loop and its nudge text,
//! - `UsageStats`/`UsageTracker` accounting, including subtracting reasoning
//!   tokens out of `completion_tokens`.
//!
//! Subscription-CLI backends (`claude -p`) live in [`cli`].

use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use tracing::{error, warn};

use crate::config::Config;
use crate::pricing::get_cost;
use crate::tokens::escape_special_tokens;

pub mod cli;
pub use cli::{CliBackend, CliKind};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default OpenAI API base (no trailing slash).
pub const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
/// Default Anthropic API base (no trailing slash).
pub const ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com/v1";
/// `anthropic-version` header value.
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Azure OpenAI: API key env var / config.json key.
pub const AZURE_KEY_VAR: &str = "AZURE_OPENAI_API_KEY";
/// Azure OpenAI: resource endpoint (e.g. `https://myres.openai.azure.com`).
/// When set (env or config.json), OpenAI-family models are sent to this
/// Azure resource instead of api.openai.com, authenticated with
/// [`AZURE_KEY_VAR`]. The model name must match an Azure *deployment* name.
pub const AZURE_ENDPOINT_VAR: &str = "AZURE_OPENAI_ENDPOINT";
/// Azure OpenAI: optional `api-version` query parameter. ekko calls the
/// version-free `/openai/v1/` surface, so this is only needed to opt into a
/// preview surface (e.g. `preview`).
pub const AZURE_API_VERSION_VAR: &str = "AZURE_OPENAI_API_VERSION";

/// Anthropic's messages API **requires** `max_tokens`; there is no
/// "unlimited" sentinel, so ekko must pick a value. 32_000 is below the
/// documented max output of the current Claude models (Sonnet 4.5 /
/// Opus 4.5+ allow 64k) and far above anything our prompts ask for, so it acts
/// as a safety ceiling rather than a truncation point. Older claude-3.x models
/// cap out lower — see [`anthropic_max_tokens`]. OpenAI requests carry no
/// token limit at all.
pub const ANTHROPIC_MAX_TOKENS: u32 = 32_000;

/// `max_tokens` to send for a given Claude model.
///
/// Anthropic rejects a `max_tokens` above the model's documented max output
/// with a **non-retryable 400**, so a single flat ceiling would hard-fail every
/// claude-3.x model. This is a deliberately dumb name-prefix heuristic (the
/// same shape as [`supports_reasoning_effort`]): claude-3-5/claude-3-7 cap at
/// 8192, the original claude-3 line (haiku/sonnet/opus) at 4096, everything
/// newer gets the [`ANTHROPIC_MAX_TOKENS`] ceiling. No config plumbing — if a
/// future model caps lower, add a line here.
fn anthropic_max_tokens(model: &str) -> u32 {
    let model = model.to_lowercase();
    if model.starts_with("claude-3-5") || model.starts_with("claude-3-7") {
        8192
    } else if model.starts_with("claude-3-") {
        // claude-3-haiku / claude-3-sonnet / claude-3-opus
        4096
    } else {
        ANTHROPIC_MAX_TOKENS
    }
}

/// Default retry attempts per call.
pub const DEFAULT_MAX_RETRIES: u32 = 3;
/// Default JSON re-ask attempts when a response will not parse.
pub const DEFAULT_JSON_RETRIES: u32 = 2;
/// Backoff cap — the `2^attempt` schedule saturates at 60 seconds.
const MAX_BACKOFF: Duration = Duration::from_secs(60);
/// Backoff base — 1 second before the first retry, doubling thereafter.
const DEFAULT_BACKOFF_BASE: Duration = Duration::from_secs(1);
/// Rate-limit (429) backoff, as a multiple of the backoff base: 60× — one
/// minute at the default base. Providers enforce rate limits over per-minute
/// windows (Azure admits in 1s/10s sub-windows of the minute), so the
/// exponential 1-2-4s schedule lands every retry inside the same throttle
/// window and burns the whole attempt; waiting out the window is the only
/// thing that works. Expressed as a factor so tests that shrink the base
/// shrink this wait proportionally.
const RATE_LIMIT_BACKOFF_FACTOR: u32 = 60;
/// Whole-request timeout: ample for the largest map-reduce chunk, and it
/// bounds a wedged connection.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

/// The message appended when a response would not parse as JSON and the call
/// is re-asked. Its exact bytes are pinned by tests.
pub const JSON_RETRY_NUDGE: &str =
    "Please respond with valid JSON only, no markdown fences or extra text.";

// ---------------------------------------------------------------------------
// Providers
// ---------------------------------------------------------------------------

/// A supported LLM provider. These two are the only wire protocols ekko
/// implements. Azure OpenAI is not a third provider: it re-routes
/// [`Provider::OpenAi`] traffic to an Azure resource when
/// [`AZURE_ENDPOINT_VAR`] is configured — see [`AzureSettings`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    OpenAi,
    Anthropic,
}

impl Provider {
    /// Provider name as it appears in user-facing messages
    /// (`"openai"` / `"anthropic"`).
    pub fn name(self) -> &'static str {
        match self {
            Provider::OpenAi => "openai",
            Provider::Anthropic => "anthropic",
        }
    }

    /// Env var (and config.json key) holding this provider's API key.
    pub fn env_var(self) -> &'static str {
        match self {
            Provider::OpenAi => "OPENAI_API_KEY",
            Provider::Anthropic => "ANTHROPIC_API_KEY",
        }
    }

    /// Default API base URL.
    pub fn base_url(self) -> &'static str {
        match self {
            Provider::OpenAi => OPENAI_BASE_URL,
            Provider::Anthropic => ANTHROPIC_BASE_URL,
        }
    }
}

/// Azure OpenAI routing, active when [`AZURE_ENDPOINT_VAR`] is configured.
///
/// ekko targets Azure's `v1` surface (`{endpoint}/openai/v1/chat/completions`,
/// `api-key` header), which speaks the exact same request/response schema as
/// api.openai.com — the deployment name travels in the body's `model` field,
/// so the rest of the client is unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AzureSettings {
    /// Resource endpoint, trailing slashes trimmed.
    pub endpoint: String,
    /// Optional `api-version` query parameter (the v1 surface needs none).
    pub api_version: Option<String>,
}

/// Read Azure settings from config/env, `None` when no endpoint is set.
pub fn azure_settings(config: &Config) -> Option<AzureSettings> {
    let endpoint = config.get_api_key(AZURE_ENDPOINT_VAR)?;
    Some(AzureSettings {
        endpoint: endpoint.trim_end_matches('/').to_string(),
        api_version: config.get_api_key(AZURE_API_VERSION_VAR),
    })
}

/// The key env var a model actually needs under the current config:
/// [`AZURE_KEY_VAR`] for OpenAI-family models when an Azure endpoint is
/// configured, the provider's own var otherwise. Availability checks and
/// client construction both resolve through here so they can't disagree.
fn required_key_var(config: &Config, provider: Provider) -> &'static str {
    if provider == Provider::OpenAi && azure_settings(config).is_some() {
        AZURE_KEY_VAR
    } else {
        provider.env_var()
    }
}

/// Model prefix → provider. The lookup below depends on this order: it takes
/// the first match, so moving a row can change which provider a model name
/// resolves to.
///
/// There are deliberately no `gemini`/`azure/` rows. `azure/gpt-4` falls
/// through to the substring pass and resolves to openai (Azure is a routing
/// option on the OpenAI provider, not a provider of its own); `gemini-*`
/// resolves to "unknown", since ekko cannot call it.
static PROVIDER_MAP: &[(&str, Provider)] = &[
    ("gpt", Provider::OpenAi),
    ("o1", Provider::OpenAi),
    ("o3", Provider::OpenAi),
    ("o4", Provider::OpenAi),
    ("claude", Provider::Anthropic),
];

/// Determine the provider required for a model name.
///
/// Case-insensitive; a prefix match wins over a substring match; `None` when
/// nothing matches. Callers always pass the model explicitly — there is no
/// hidden read of the configured model here.
pub fn get_required_provider(model: &str) -> Option<Provider> {
    let model_lower = model.to_lowercase();

    // Prefix match first: it is the more specific signal.
    for (prefix, provider) in PROVIDER_MAP {
        if model_lower.starts_with(prefix) {
            return Some(*provider);
        }
    }
    // fallback: prefix appearing anywhere in the name
    for (prefix, provider) in PROVIDER_MAP {
        if model_lower.contains(prefix) {
            return Some(*provider);
        }
    }
    None
}

/// Check whether the configured LLM is usable (API key present).
///
/// Returns `(is_available, message)`; the message explains what is missing.
/// The message text is pinned by tests.
///
/// An unrecognised model reports available with a "will attempt call"
/// message: a health check must not block a user over a model name we merely
/// fail to recognise. The actual [`LlmClient::new`] call will still fail with
/// [`LlmError::UnknownProvider`], since there is no generic routing fallback.
pub fn check_llm_available(config: &Config, model: &str) -> (bool, String) {
    match get_required_provider(model) {
        None => (
            true,
            format!("Model '{model}' has unknown provider; will attempt call"),
        ),
        Some(provider) => {
            let key_var = required_key_var(config, provider);
            if config.get_api_key(key_var).is_some() {
                (true, format!("API key found for {}", provider.name()))
            } else {
                (
                    false,
                    format!(
                        "Model '{model}' requires {key_var} to be set. \
Set this environment variable or add it to your .env file to enable \
LLM-powered features (summaries, snapshots, semantic search, Q&A)."
                    ),
                )
            }
        }
    }
}

/// Error out if the LLM is not available. An unrecognised provider is *not*
/// an error here — only a recognised provider with no API key is.
pub fn require_llm(config: &Config, model: &str) -> Result<(), LlmNotConfiguredError> {
    if let Some(provider) = get_required_provider(model) {
        let key_var = required_key_var(config, provider);
        if config.get_api_key(key_var).is_none() {
            return Err(LlmNotConfiguredError {
                model: model.to_string(),
                provider: provider.name().to_string(),
                env_var: key_var.to_string(),
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Backends (HTTP API vs. a coding-assistant CLI)
// ---------------------------------------------------------------------------

/// How completions are obtained. `Api` is the original path — OpenAI / Azure
/// / Anthropic over HTTP with an API key, the provider chosen by the model
/// name. `Cli` drives a coding-assistant CLI in single-prompt mode (see
/// [`cli`]) for people who have a subscription but no key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Api,
    Cli(CliKind),
}

impl Backend {
    /// Parse a config value. `api` (aliases `openai`, `anthropic`, `azure` —
    /// the provider still follows the model name) or a CLI name.
    pub fn parse(name: &str) -> Option<Backend> {
        match name.trim().to_ascii_lowercase().as_str() {
            "api" | "http" | "openai" | "anthropic" | "azure" => Some(Backend::Api),
            "claude-cli" | "claude_cli" | "claude" => Some(Backend::Cli(CliKind::Claude)),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Backend::Api => "api",
            Backend::Cli(kind) => kind.name(),
        }
    }

    /// Every value [`Backend::parse`] accepts as a canonical name, for
    /// error messages.
    pub const NAMES: &'static [&'static str] = &["api", "claude-cli"];
}

/// The outcome of [`resolve_backend`]: which backend, which model, and a
/// one-line explanation for `ekko check` / `get_status`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedBackend {
    pub backend: Backend,
    pub model: String,
    pub reason: String,
}

/// What the completion is for. Background indexing (summaries, snapshots)
/// and interactive queries (`ask_question`, semantic search) can run on
/// different backends: a subscription CLI is fine for a batch job that
/// nobody is waiting on, but adds 15–30 s to every interactive answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Summaries and context snapshots — the indexer. Configured by
    /// `llm_backend` / `llm_model`.
    Index,
    /// `ask_question`, semantic/hybrid search — someone is waiting.
    /// Configured by `query_backend` / `query_model`, falling back to the
    /// index settings.
    Query,
}


/// Pick the model for a chosen backend: the user's choice when the backend
/// can serve it, else the backend's default — with a note for the
/// resolution reason when a choice had to be overridden.
///
/// A CLI serves its own vendor's models only, so an OpenAI-family name
/// (typically left over from an API setup) is always replaced. The API path
/// keeps a model the user chose *for this role* verbatim even when ekko
/// cannot route it — the call then fails with [`LlmError::UnknownProvider`],
/// which is the honest outcome (and what tests rely on to keep LLM calls
/// from escaping). Only a model `inherited` from the other role (the query
/// role reusing `llm_model`, say a CLI alias like `sonnet`) is swapped for
/// the API default.
fn pick_model(backend: Backend, explicit: Option<&str>, inherited: bool) -> (String, String) {
    match (backend, explicit) {
        (Backend::Api, None) => (crate::config::DEFAULT_LLM_MODEL.to_string(), String::new()),
        (Backend::Api, Some(model)) => {
            if !inherited || get_required_provider(model).is_some() {
                (model.to_string(), String::new())
            } else {
                (
                    crate::config::DEFAULT_LLM_MODEL.to_string(),
                    format!("; llm_model '{model}' has no API provider, using the API default"),
                )
            }
        }
        (Backend::Cli(kind), None) => (kind.default_model().to_string(), String::new()),
        (Backend::Cli(kind), Some(model)) => {
            if get_required_provider(model) == Some(Provider::OpenAi) {
                (
                    kind.default_model().to_string(),
                    format!(
                        "; model '{model}' is an OpenAI-family name this CLI cannot serve, using the CLI default"
                    ),
                )
            } else {
                (model.to_string(), String::new())
            }
        }
    }
}

/// Decide the backend and model for `role`.
///
/// Precedence for [`Role::Index`]:
/// 1. `llm_backend` set explicitly (config.json or `EKKO_BACKEND`) wins. An
///    unrecognised value falls through to auto-detection with a warning in
///    the reason text.
/// 2. Otherwise, an API key usable for `llm_model` → `Api`.
/// 3. Otherwise, the first supported CLI found on `PATH`.
/// 4. Otherwise `Api`, so the existing "set OPENAI_API_KEY" guidance shows.
///
/// [`Role::Query`] uses `query_backend` / `query_model` when set, and the
/// index resolution otherwise. The model is always passed through
/// [`pick_model`], so a choice the backend cannot serve is replaced by that
/// backend's default and the reason says so.
pub fn resolve_backend(config: &Config, role: Role) -> ResolvedBackend {
    match role {
        Role::Index => resolve_index(config),
        Role::Query => resolve_query(config),
    }
}

/// [`Role::Query`]: `query_backend` / `query_model` when set, else the
/// index resolution (with a reason suffix saying so).
fn resolve_query(config: &Config) -> ResolvedBackend {
    let query_model = config.query_model.as_deref();
    match config
        .query_backend
        .as_deref()
        .map(|n| (n, Backend::parse(n)))
    {
        Some((_, Some(backend))) => {
            let (explicit, inherited) = match query_model {
                Some(model) => (Some(model), false),
                None if config.llm_model_explicit => (Some(config.llm_model.as_str()), true),
                None => (None, false),
            };
            let (model, note) = pick_model(backend, explicit, inherited);
            ResolvedBackend {
                backend,
                model,
                reason: format!("query_backend={} (explicit){note}", backend.name()),
            }
        }
        Some((name, None)) => {
            warn!(
                value = name,
                "unrecognised query_backend; using index backend"
            );
            let mut resolved = resolve_index(config);
            resolved.reason = format!(
                "query_backend '{name}' not recognised (expected one of {}); {}",
                Backend::NAMES.join(", "),
                resolved.reason
            );
            resolved
        }
        None => {
            let mut resolved = resolve_index(config);
            if let Some(model) = query_model {
                let (model, note) = pick_model(resolved.backend, Some(model), false);
                resolved.model = model;
                resolved.reason = format!("{} (query_model set){note}", resolved.reason);
            } else {
                resolved.reason = format!("{} (same as index)", resolved.reason);
            }
            resolved
        }
    }
}

/// [`Role::Index`]: the precedence ladder documented on [`resolve_backend`].
fn resolve_index(config: &Config) -> ResolvedBackend {
    let explicit_model = if config.llm_model_explicit {
        Some(config.llm_model.as_str())
    } else {
        None
    };

    // An explicit, recognised backend short-circuits; an unrecognised one
    // falls through to auto-detection carrying a note for the reason text.
    let unknown_note = match config
        .llm_backend
        .as_deref()
        .map(|name| (name, Backend::parse(name)))
    {
        Some((_, Some(backend))) => {
            let (model, note) = pick_model(backend, explicit_model, false);
            return ResolvedBackend {
                backend,
                model,
                reason: format!("llm_backend={} (explicit){note}", backend.name()),
            };
        }
        Some((name, None)) => {
            warn!(value = name, "unrecognised llm_backend; auto-detecting");
            format!(
                "llm_backend '{name}' not recognised (expected one of {}); ",
                Backend::NAMES.join(", ")
            )
        }
        None => String::new(),
    };

    let api_key_present = get_required_provider(&config.llm_model)
        .map(|provider| {
            config
                .get_api_key(required_key_var(config, provider))
                .is_some()
        })
        .unwrap_or(false);
    if api_key_present {
        return ResolvedBackend {
            backend: Backend::Api,
            model: config.llm_model.clone(),
            reason: format!(
                "{unknown_note}API key found for model '{}'",
                config.llm_model
            ),
        };
    }

    for kind in [CliKind::Claude] {
        if let Some(path) = kind.find_binary() {
            let (model, note) = pick_model(Backend::Cli(kind), explicit_model, false);
            return ResolvedBackend {
                backend: Backend::Cli(kind),
                model,
                reason: format!(
                    "{unknown_note}no API key configured; found {} at {}{note}",
                    kind.binary(),
                    path.display()
                ),
            };
        }
    }

    ResolvedBackend {
        backend: Backend::Api,
        model: config.llm_model.clone(),
        reason: format!(
            "{unknown_note}no API key configured and no supported CLI ({}) on PATH",
            [CliKind::Claude]
                .iter()
                .map(|k| k.binary())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// Backend-aware availability check: the CLI/API that [`resolve_backend`]
/// picks, and whether it can actually be used. This is what the indexer,
/// status and search gates should call; [`check_llm_available`] is the
/// API-only primitive underneath.
pub fn check_backend_available(config: &Config, role: Role) -> (bool, String) {
    backend_availability(config, &resolve_backend(config, role))
}

/// Availability of an already-resolved backend. Callers that show the
/// resolution next to its availability (`ekko check`, `get_status`) pass
/// the one they display, so the two can never describe different answers.
pub fn backend_availability(config: &Config, resolved: &ResolvedBackend) -> (bool, String) {
    match resolved.backend {
        Backend::Api => check_llm_available(config, &resolved.model),
        Backend::Cli(kind) => match kind.find_binary() {
            Some(path) => (
                true,
                format!(
                    "{} backend via {} (model {})",
                    kind.name(),
                    path.display(),
                    resolved.model
                ),
            ),
            None => (
                false,
                format!(
                    "llm_backend is {} but '{}' was not found on PATH. Install it \
and log in, or set an API key to use the HTTP backend instead.",
                    kind.name(),
                    kind.binary()
                ),
            ),
        },
    }
}

/// The semantic/hybrid gate shared by the MCP tool and the web route:
/// `semantic` without a query backend is refused (the message explains
/// what is missing), `hybrid` degrades to `keyword`, everything else
/// passes through. Returns the mode to actually run.
pub fn gate_search_mode(config: &Config, requested: &str) -> Result<String, String> {
    if requested != "semantic" && requested != "hybrid" {
        return Ok(requested.to_string());
    }
    let (available, msg) = check_backend_available(config, Role::Query);
    if available {
        Ok(requested.to_string())
    } else if requested == "semantic" {
        Err(msg)
    } else {
        Ok("keyword".to_string())
    }
}

/// Effective worker count: CLI backends default to one concurrent process
/// (subscription usage windows are opaque and per-call latency is high)
/// unless `max_workers` was set explicitly.
pub fn effective_max_workers(config: &Config) -> usize {
    if !config.max_workers_explicit
        && matches!(
            resolve_backend(config, Role::Index).backend,
            Backend::Cli(_)
        )
    {
        1
    } else {
        config.max_workers
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Raised when an LLM operation is attempted but no API key is configured.
/// The message text is pinned by tests.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "LLM not configured: model '{model}' requires {env_var} to be set. \
Add it to config.json in your data directory, or set the environment \
variable to enable LLM-powered features."
)]
pub struct LlmNotConfiguredError {
    pub model: String,
    pub provider: String,
    pub env_var: String,
}

/// Everything that can go wrong in the LLM layer.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error(transparent)]
    NotConfigured(#[from] LlmNotConfiguredError),

    /// ekko cannot route this model (no OpenAI/Anthropic prefix match).
    #[error(
        "LLM not available: model '{model}' has no known provider; \
ekko supports OpenAI (gpt*, o1*, o3*, o4*) and Anthropic (claude*) models only"
    )]
    UnknownProvider { model: String },

    /// `status` is `Some` for a non-success HTTP response and `None` for a
    /// transport failure (DNS, refused connection, mid-flight reset). The
    /// retry classifier keys off `status` only, never the body text, so a
    /// change in provider response shape cannot silently reclassify a
    /// retryable error.
    #[error("HTTP error{}: {body}", status.map(|s| format!(" status {s}")).unwrap_or_default())]
    Http { status: Option<u16>, body: String },

    #[error("LLM request timed out")]
    Timeout,

    #[error("invalid LLM response: {0}")]
    InvalidResponse(String),

    #[error("invalid LLM request: {0}")]
    InvalidRequest(String),

    #[error("HTTP client initialization failed: {0}")]
    ClientInit(String),
}

/// Retry policy. Keyed on transport-vs-status: transport failures, timeouts,
/// 429 and 5xx retry; every other 4xx (bad key, bad request, model not found)
/// propagates immediately.
///
/// Classification is structural, never text-based: the decision is made from
/// the HTTP status (or its absence) alone, so it cannot drift when a provider
/// reworks its error bodies.
fn is_retryable(err: &LlmError) -> bool {
    match err {
        LlmError::Timeout => true,
        LlmError::Http { status: None, .. } => true,
        LlmError::Http {
            status: Some(code), ..
        } => *code == 429 || (500..600).contains(code),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Messages / responses
// ---------------------------------------------------------------------------

/// One chat message. Serializes to `{"role": ..., "content": ...}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Message {
            role: role.into(),
            content: content.into(),
        }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Message::new("system", content)
    }
    pub fn user(content: impl Into<String>) -> Self {
        Message::new("user", content)
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Message::new("assistant", content)
    }
}

/// Escape special tokens in every message's content before sending.
fn prepare_messages(messages: &[Message]) -> Vec<Message> {
    messages
        .iter()
        .map(|m| Message {
            role: m.role.clone(),
            content: escape_special_tokens(&m.content),
        })
        .collect()
}

/// Raw usage numbers as reported by the provider, in OpenAI's shape.
/// Anthropic's `input_tokens`/`output_tokens` are mapped into it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawUsage {
    #[serde(default)]
    pub prompt_tokens: Option<i64>,
    #[serde(default)]
    pub completion_tokens: Option<i64>,
    #[serde(default)]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

/// OpenAI's reasoning-token breakdown.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: Option<i64>,
}

/// A completed LLM call. Deliberately minimal — `.content` is the assistant
/// text of the single choice ekko asks for.
#[derive(Debug, Clone)]
pub struct LlmResponse {
    /// Model the request was sent with.
    pub model: String,
    /// Assistant text (all text blocks concatenated for Anthropic).
    pub content: String,
    /// Cost/token accounting for this single call (also folded into the tracker).
    pub call_stats: UsageStats,
}

impl LlmResponse {
    /// Assistant text.
    pub fn content(&self) -> &str {
        &self.content
    }
}

// ---------------------------------------------------------------------------
// Usage tracking
// ---------------------------------------------------------------------------

fn round6(x: f64) -> f64 {
    crate::utils::round_decimal(x, 6)
}

/// Accumulated call, token and cost totals.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct UsageStats {
    pub calls: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub reasoning_tokens: u64,
    pub cost: f64,
}

impl UsageStats {
    /// Add another `UsageStats` into this one.
    pub fn add(&mut self, other: &UsageStats) {
        self.calls += other.calls;
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.reasoning_tokens += other.reasoning_tokens;
        self.cost += other.cost;
    }

    /// JSON form; the keys and their order are pinned by tests.
    ///
    /// `cost` uses [`round6`], which rounds to six decimal places with ties
    /// going to even by formatting and re-parsing. The naive
    /// `(v * 1e6).round() / 1e6` this used to do differs on perfectly ordinary
    /// token counts (e.g. `get_cost("gpt-5-nano", 70, 0, 0)` = 3.5e-06 →
    /// 3e-06 correctly, 4e-06 naively), because the midpoint case is common at
    /// these magnitudes.
    pub fn to_json(&self) -> Value {
        json!({
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.input_tokens + self.output_tokens + self.reasoning_tokens,
            "cost": round6(self.cost),
            "cost_display": format!("${:.4}", self.cost),
        })
    }
}

/// Thread-safe accumulator for [`UsageStats`] across calls.
#[derive(Debug, Default)]
pub struct UsageTracker {
    stats: Mutex<UsageStats>,
}

impl UsageTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record usage from an API response and return this call's stats.
    ///
    /// Reasoning tokens are subtracted out of `completion_tokens`, so
    /// `output_tokens` never double-counts them.
    ///
    /// A missing or empty usage payload must record nothing at all, not even
    /// `calls`. This function takes a typed `&RawUsage` that cannot express
    /// "absent", so that gate lives one level up, in
    /// [`LlmClient::call_llm_with`]: `send_once` returns `Option<RawUsage>`
    /// and `record` is simply not called for `None`.
    pub fn record(&self, model: &str, usage: &RawUsage) -> UsageStats {
        self.record_with_cost(model, usage, None)
    }

    /// [`UsageTracker::record`] with an authoritative cost from the backend
    /// (a CLI that reports `total_cost_usd`), falling back to the pricing
    /// table when `None`.
    pub fn record_with_cost(
        &self,
        model: &str,
        usage: &RawUsage,
        reported_cost: Option<f64>,
    ) -> UsageStats {
        let input_tokens = usage.prompt_tokens.unwrap_or(0).max(0) as u64;
        let mut output_tokens = usage.completion_tokens.unwrap_or(0).max(0) as u64;
        let mut reasoning_tokens: u64 = 0;

        if let Some(details) = &usage.completion_tokens_details
            && let Some(raw) = details.reasoning_tokens
        {
            reasoning_tokens = raw.max(0) as u64;
            output_tokens = output_tokens.saturating_sub(reasoning_tokens);
        }

        let cost = reported_cost
            .map(round6)
            .unwrap_or_else(|| get_cost(model, input_tokens, output_tokens, reasoning_tokens));
        let call_stats = UsageStats {
            calls: 1,
            input_tokens,
            output_tokens,
            reasoning_tokens,
            cost,
        };

        match self.stats.lock() {
            Ok(mut guard) => guard.add(&call_stats),
            // A poisoned tracker mutex must never take down an LLM call:
            // accounting is best-effort telemetry, not correctness.
            Err(poisoned) => poisoned.into_inner().add(&call_stats),
        }
        call_stats
    }

    /// Current accumulated stats.
    pub fn get_stats(&self) -> UsageStats {
        match self.stats.lock() {
            Ok(guard) => *guard,
            Err(poisoned) => *poisoned.into_inner(),
        }
    }

    /// Reset accumulated stats.
    pub fn reset(&self) {
        match self.stats.lock() {
            Ok(mut guard) => *guard = UsageStats::default(),
            Err(poisoned) => *poisoned.into_inner() = UsageStats::default(),
        }
    }

    /// Current stats as JSON — see [`UsageStats::to_json`].
    pub fn to_json(&self) -> Value {
        self.get_stats().to_json()
    }
}

static USAGE_TRACKER: OnceLock<UsageTracker> = OnceLock::new();

/// The process-wide usage tracker, used when a caller passes none.
pub fn get_usage_tracker() -> &'static UsageTracker {
    USAGE_TRACKER.get_or_init(UsageTracker::new)
}

// ---------------------------------------------------------------------------
// Reasoning-effort capability
// ---------------------------------------------------------------------------

/// Strip a leading vendor prefix from a model name for the wire.
///
/// Config and callers may name a model `openai/gpt-5-mini`, but the
/// OpenAI/Anthropic APIs themselves reject the prefixed form. ekko therefore
/// routes on the full name (so `openai/…` still resolves) but sends the bare
/// one. Same prefix list as [`crate::pricing::normalize_model_name`], so cost
/// lookup and the wire agree.
fn wire_model(model: &str) -> &str {
    for prefix in ["openai/", "anthropic/", "google/"] {
        if model.len() >= prefix.len() && model[..prefix.len()].eq_ignore_ascii_case(prefix) {
            return &model[prefix.len()..];
        }
    }
    model
}

/// Whether this model accepts OpenAI's `reasoning_effort` parameter.
///
/// A deliberately dumb name-prefix rule: the GPT-5 family and the o-series
/// (`o` followed by a digit) accept it, everything else has the field
/// stripped from the request rather than getting a failed call.
pub fn supports_reasoning_effort(model: &str) -> bool {
    let model = model.to_lowercase();
    if model.starts_with("gpt-5") {
        return true;
    }
    let bytes = model.as_bytes();
    bytes.len() >= 2 && bytes[0] == b'o' && bytes[1].is_ascii_digit()
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    #[serde(default)]
    choices: Vec<OpenAiChoice>,
    /// Left untyped so that "no usage to account for" — absent, `null` *and*
    /// `{}` alike — survives the round trip; see [`truthy_usage`].
    #[serde(default)]
    usage: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    #[serde(default)]
    message: OpenAiChoiceMessage,
}

#[derive(Debug, Default, Deserialize)]
struct OpenAiChoiceMessage {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    #[serde(default)]
    content: Vec<AnthropicBlock>,
    #[serde(default)]
    usage: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct AnthropicBlock {
    #[serde(default, rename = "type")]
    block_type: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: Option<i64>,
    #[serde(default)]
    output_tokens: Option<i64>,
}

/// Gate on a usage payload being present and non-empty, then type it.
///
/// Returns `None` when the provider sent no usage object, `null`, or an empty
/// one; nothing is recorded for those, so `calls` stays at 0. A
/// present-but-all-zero usage object is non-empty and therefore counts. A
/// usage value of an unexpected *shape* also yields `None` (nothing to
/// account for) rather than failing the whole call.
fn truthy_usage<T: serde::de::DeserializeOwned>(usage: Option<Value>) -> Option<T> {
    let usage = usage?;
    if !is_truthy(&usage) {
        return None;
    }
    serde_json::from_value(usage).ok()
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

fn shared_http_client() -> Result<reqwest::Client, LlmError> {
    static CLIENT: OnceLock<Result<reqwest::Client, String>> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            reqwest::Client::builder()
                .timeout(REQUEST_TIMEOUT)
                .build()
                .map_err(|e| e.to_string())
        })
        .clone()
        .map_err(LlmError::ClientInit)
}

/// A configured LLM client: model + provider + credential + transport.
///
/// Construction performs the [`require_llm`] check, so a client only exists
/// when a call can actually be attempted.
#[derive(Clone)]
pub struct LlmClient {
    http: reqwest::Client,
    provider: Provider,
    model: String,
    api_key: String,
    reasoning_effort: String,
    base_url: String,
    /// `Some` when this client talks to an Azure OpenAI resource instead of
    /// the provider's own API (OpenAI-family models only). Changes the URL
    /// and the auth header; the body and response schema are identical.
    azure: Option<AzureSettings>,
    /// `Some` when completions come from a coding-assistant CLI instead of
    /// HTTP. `provider`/`api_key`/`base_url` are then placeholders.
    cli: Option<CliBackend>,
    backoff_base: Duration,
}

/// Manual `Debug` so an accidental `{:?}` (tracing field, panic message,
/// error context) can never leak the API key.
impl std::fmt::Debug for LlmClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmClient")
            .field("provider", &self.provider)
            .field("model", &self.model)
            .field("api_key", &"<redacted>")
            .field("reasoning_effort", &self.reasoning_effort)
            .field("base_url", &self.base_url)
            .field("azure", &self.azure)
            .field("cli", &self.cli)
            .field("backoff_base", &self.backoff_base)
            .finish()
    }
}

impl LlmClient {
    /// Build a client for the [`Role::Index`] backend, taking model and
    /// reasoning effort from config. Interactive callers want
    /// [`LlmClient::for_role`] with [`Role::Query`].
    pub fn from_config(config: &Config) -> Result<Self, LlmError> {
        Self::for_role(config, Role::Index)
    }

    /// Build a client for the backend/model that [`resolve_backend`] picks
    /// for `role`.
    pub fn for_role(config: &Config, role: Role) -> Result<Self, LlmError> {
        let resolved = resolve_backend(config, role);
        match resolved.backend {
            Backend::Api => Self::new(config, &resolved.model),
            Backend::Cli(kind) => Self::new_cli(config, kind, &resolved.model),
        }
    }

    /// Build a client that drives a coding-assistant CLI.
    pub fn new_cli(config: &Config, kind: CliKind, model: &str) -> Result<Self, LlmError> {
        let binary = kind.find_binary().ok_or_else(|| LlmNotConfiguredError {
            model: model.to_string(),
            provider: kind.name().to_string(),
            env_var: format!("PATH (no '{}' executable)", kind.binary()),
        })?;
        let workdir = crate::paths::get_data_dir()
            .map(|d| d.join("cli-workdir"))
            .unwrap_or_else(|_| std::env::temp_dir().join("ekko-cli-workdir"));
        Ok(LlmClient {
            http: shared_http_client()?,
            provider: Provider::Anthropic,
            model: model.to_string(),
            api_key: String::new(),
            reasoning_effort: config.llm_reasoning_effort.clone(),
            base_url: String::new(),
            azure: None,
            cli: Some(CliBackend {
                kind,
                binary,
                model: model.to_string(),
                reasoning_effort: config.llm_reasoning_effort.clone(),
                timeout: Duration::from_secs(config.cli_timeout_secs),
                workdir,
            }),
            backoff_base: DEFAULT_BACKOFF_BASE,
        })
    }

    /// The backend this client completes through.
    pub fn backend(&self) -> Backend {
        match &self.cli {
            Some(cli) => Backend::Cli(cli.kind),
            None => Backend::Api,
        }
    }

    /// Build a client for an explicit model, taking credentials and reasoning
    /// effort from `config`.
    ///
    /// API keys are resolved via [`Config::get_api_key`] (process env first,
    /// `config.json` second). ekko never mutates the process environment; a
    /// key from `config.json` is passed to the client, not exported.
    pub fn new(config: &Config, model: &str) -> Result<Self, LlmError> {
        let provider = get_required_provider(model).ok_or_else(|| LlmError::UnknownProvider {
            model: model.to_string(),
        })?;
        // Azure routing: an OpenAI-family model with AZURE_OPENAI_ENDPOINT
        // configured goes to that resource with the Azure key instead.
        let azure = if provider == Provider::OpenAi {
            azure_settings(config)
        } else {
            None
        };
        let key_var = required_key_var(config, provider);
        let api_key = config
            .get_api_key(key_var)
            .ok_or_else(|| LlmNotConfiguredError {
                model: model.to_string(),
                provider: provider.name().to_string(),
                env_var: key_var.to_string(),
            })?;

        Ok(LlmClient {
            http: shared_http_client()?,
            provider,
            model: model.to_string(),
            api_key,
            reasoning_effort: config.llm_reasoning_effort.clone(),
            base_url: provider.base_url().to_string(),
            azure,
            cli: None,
            backoff_base: DEFAULT_BACKOFF_BASE,
        })
    }

    /// Override the API base URL (trailing slashes trimmed). Used by tests
    /// and by anyone fronting the provider with a proxy/gateway. An explicit
    /// override wins outright, so it also switches off Azure routing.
    pub fn with_base_url(mut self, base_url: &str) -> Self {
        self.base_url = base_url.trim_end_matches('/').to_string();
        self.azure = None;
        self
    }

    /// Override the retry backoff base (default 1s, giving 1, 2, 4, … capped
    /// at 60). Exists so tests can exercise the retry loop without real waits.
    pub fn with_backoff_base(mut self, base: Duration) -> Self {
        self.backoff_base = base;
        self
    }

    /// The model this client calls.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The provider this client routes to.
    pub fn provider(&self) -> Provider {
        self.provider
    }

    /// Call the LLM with the default retry budget, recording usage on the
    /// global tracker.
    pub async fn call_llm(&self, messages: &[Message]) -> Result<LlmResponse, LlmError> {
        self.call_llm_with(messages, DEFAULT_MAX_RETRIES, None)
            .await
    }

    /// Call the LLM. `tracker` defaults to the global tracker.
    pub async fn call_llm_with(
        &self,
        messages: &[Message],
        max_retries: u32,
        tracker: Option<&UsageTracker>,
    ) -> Result<LlmResponse, LlmError> {
        if max_retries == 0 {
            // Zero attempts is a caller bug, not a runtime condition: reject
            // it with a typed error rather than falling out of the loop with
            // nothing to report.
            return Err(LlmError::InvalidRequest(
                "max_retries must be at least 1".to_string(),
            ));
        }
        let tracker = tracker.unwrap_or_else(|| get_usage_tracker());
        let safe_messages = prepare_messages(messages);
        let body = if self.cli.is_none() {
            Some(self.build_body(&safe_messages))
        } else {
            None
        };

        let mut last_error: Option<LlmError> = None;
        for attempt in 0..max_retries {
            let outcome = match (&self.cli, &body) {
                (Some(cli), _) => cli
                    .complete(&safe_messages)
                    .await
                    .map(|o| (o.content, o.usage, o.cost_usd)),
                (None, Some(body)) => self.send_once(body).await.map(|(c, u)| (c, u, None)),
                (None, None) => unreachable!("HTTP client always builds a body"),
            };
            match outcome {
                Ok((content, usage, reported_cost)) => {
                    // A missing or empty usage object means the call is not
                    // counted at all — not even `calls`. A *present* usage
                    // object whose numbers are all zero is still non-empty,
                    // and does count.
                    let pricing_model = match &self.cli {
                        Some(cli) => cli.kind.pricing_model(&self.model),
                        None => self.model.clone(),
                    };
                    let call_stats = match &usage {
                        Some(usage) => {
                            tracker.record_with_cost(&pricing_model, usage, reported_cost)
                        }
                        None => UsageStats::default(),
                    };
                    return Ok(LlmResponse {
                        model: self.model.clone(),
                        content,
                        call_stats,
                    });
                }
                Err(e) if is_retryable(&e) => {
                    warn!(
                        attempt = attempt + 1,
                        max_retries,
                        error = %e,
                        "LLM API error; will retry"
                    );
                    // No sleep after the final attempt — that wait would be
                    // dead time before the error propagates. 429s wait a flat
                    // minute (see RATE_LIMIT_BACKOFF_FACTOR); other transient
                    // errors follow the 1, 2, 4, … (cap 60s) schedule.
                    let delay = self.backoff_delay(attempt, &e);
                    last_error = Some(e);
                    if attempt + 1 < max_retries {
                        tokio::time::sleep(delay).await;
                    }
                }
                Err(e) => {
                    error!(error = %e, "LLM call failed");
                    return Err(e);
                }
            }
        }

        Err(last_error
            .unwrap_or_else(|| LlmError::InvalidRequest("LLM call exhausted retries".to_string())))
    }

    /// Call the LLM expecting JSON, with the default retry budgets.
    pub async fn call_llm_json(&self, messages: &[Message]) -> Result<Value, LlmError> {
        self.call_llm_json_with(messages, DEFAULT_MAX_RETRIES, DEFAULT_JSON_RETRIES, None)
            .await
    }

    /// Call the LLM expecting JSON, re-asking on parse failure.
    ///
    /// Returns the parsed value, or an empty object once `json_retries` extra
    /// attempts are exhausted. API errors propagate.
    pub async fn call_llm_json_with(
        &self,
        messages: &[Message],
        max_retries: u32,
        json_retries: u32,
        tracker: Option<&UsageTracker>,
    ) -> Result<Value, LlmError> {
        let mut convo: Vec<Message> = messages.to_vec();
        let mut last_content = String::new();

        for json_attempt in 0..=json_retries {
            let response = self.call_llm_with(&convo, max_retries, tracker).await?;
            let content = response.content;
            last_content.clone_from(&content);

            let result = parse_json_response(&content);
            if is_truthy(&result) {
                return Ok(result);
            }

            if json_attempt < json_retries {
                warn!(
                    attempt = json_attempt + 1,
                    "JSON parse failed, retrying with nudge"
                );
                convo.push(Message::assistant(content));
                convo.push(Message::user(JSON_RETRY_NUDGE));
            }
        }

        error!(
            attempts = json_retries + 1,
            preview = %truncate_chars(&last_content, 200),
            "failed to parse JSON from LLM"
        );
        Ok(Value::Object(Map::new()))
    }

    fn backoff_delay(&self, attempt: u32, err: &LlmError) -> Duration {
        let factor = if matches!(
            err,
            LlmError::Http {
                status: Some(429),
                ..
            }
        ) {
            RATE_LIMIT_BACKOFF_FACTOR
        } else {
            2u32.checked_pow(attempt).unwrap_or(u32::MAX)
        };
        self.backoff_base.saturating_mul(factor).min(MAX_BACKOFF)
    }

    /// Build the provider-specific request body.
    ///
    /// The model goes out with any vendor prefix stripped ([`wire_model`]);
    /// capability checks look at the stripped name too, so
    /// `openai/gpt-5-mini` still sends `reasoning_effort`.
    fn build_body(&self, messages: &[Message]) -> Value {
        let model = wire_model(&self.model);
        match self.provider {
            Provider::OpenAi => {
                let mut body = json!({
                    "model": model,
                    "messages": messages,
                });
                // Deliberately no `temperature` and no `max_tokens`: the
                // provider defaults are what ekko wants.
                if !self.reasoning_effort.is_empty()
                    && supports_reasoning_effort(model)
                    && let Some(map) = body.as_object_mut()
                {
                    map.insert(
                        "reasoning_effort".to_string(),
                        Value::String(self.reasoning_effort.clone()),
                    );
                }
                body
            }
            Provider::Anthropic => {
                // System messages are not part of Anthropic's `messages`
                // array; they move to the top-level `system` field. Multiple
                // system messages are joined with a blank line (ekko's
                // prompts only ever produce one).
                let system: Vec<&str> = messages
                    .iter()
                    .filter(|m| m.role == "system")
                    .map(|m| m.content.as_str())
                    .collect();
                let rest: Vec<&Message> = messages.iter().filter(|m| m.role != "system").collect();

                let mut body = json!({
                    "model": model,
                    "messages": rest,
                    "max_tokens": anthropic_max_tokens(model),
                });
                if !system.is_empty()
                    && let Some(map) = body.as_object_mut()
                {
                    map.insert("system".to_string(), Value::String(system.join("\n\n")));
                }
                // `reasoning_effort` is never sent to Anthropic — its messages
                // API has no such parameter (it uses a `thinking` block with a
                // token budget instead), and sending it would be rejected.
                body
            }
        }
    }

    fn url(&self) -> String {
        if let Some(azure) = &self.azure {
            // Azure's OpenAI-compatible v1 surface; the deployment name is
            // the body's `model` field. `api-version` only when configured
            // (the v1 surface is version-free; `preview` opts into previews).
            let mut url = format!("{}/openai/v1/chat/completions", azure.endpoint);
            if let Some(version) = &azure.api_version {
                url.push_str("?api-version=");
                url.push_str(version);
            }
            return url;
        }
        match self.provider {
            Provider::OpenAi => format!("{}/chat/completions", self.base_url),
            Provider::Anthropic => format!("{}/messages", self.base_url),
        }
    }

    /// One HTTP round trip, returning `(content, usage)`. `usage` is `None`
    /// when the response carried no (truthy) usage object — see
    /// [`truthy_usage`].
    async fn send_once(&self, body: &Value) -> Result<(String, Option<RawUsage>), LlmError> {
        let request = match self.provider {
            // Azure authenticates with `api-key` instead of a bearer token.
            Provider::OpenAi if self.azure.is_some() => self
                .http
                .post(self.url())
                .header("api-key", self.api_key.clone()),
            Provider::OpenAi => self
                .http
                .post(self.url())
                .header("Authorization", format!("Bearer {}", self.api_key)),
            Provider::Anthropic => self
                .http
                .post(self.url())
                .header("x-api-key", self.api_key.clone())
                .header("anthropic-version", ANTHROPIC_VERSION),
        };

        let response = request
            .json(body)
            .send()
            .await
            .map_err(transport_to_llm_error)?;

        let status = response.status();
        let text = response.text().await.map_err(transport_to_llm_error)?;

        if !status.is_success() {
            return Err(LlmError::Http {
                status: Some(status.as_u16()),
                body: truncate_chars(&text, 500),
            });
        }

        match self.provider {
            Provider::OpenAi => {
                let parsed: OpenAiResponse = serde_json::from_str(&text)
                    .map_err(|e| LlmError::InvalidResponse(e.to_string()))?;
                // An empty `choices` list is a typed, non-retryable
                // InvalidResponse rather than a panic on index 0.
                let choice = parsed.choices.first().ok_or_else(|| {
                    LlmError::InvalidResponse("response contained no choices".to_string())
                })?;
                // `content: null` becomes "" rather than propagating an
                // absent value into string handling; the JSON path then
                // recovers it as `{}`.
                let content = choice.message.content.clone().unwrap_or_default();
                Ok((content, truthy_usage::<RawUsage>(parsed.usage)))
            }
            Provider::Anthropic => {
                let parsed: AnthropicResponse = serde_json::from_str(&text)
                    .map_err(|e| LlmError::InvalidResponse(e.to_string()))?;
                let content: String = parsed
                    .content
                    .iter()
                    .filter(|b| b.block_type == "text")
                    .filter_map(|b| b.text.as_deref())
                    .collect();
                let usage = truthy_usage::<AnthropicUsage>(parsed.usage).map(|u| RawUsage {
                    prompt_tokens: u.input_tokens,
                    completion_tokens: u.output_tokens,
                    completion_tokens_details: None,
                });
                Ok((content, usage))
            }
        }
    }
}

/// Map a transport-layer `reqwest::Error` to [`LlmError`]. Timeouts become
/// [`LlmError::Timeout`]; everything else becomes `Http { status: None }` so
/// the retry classifier recognises it as a transport failure.
fn transport_to_llm_error(e: reqwest::Error) -> LlmError {
    if e.is_timeout() {
        LlmError::Timeout
    } else {
        LlmError::Http {
            status: None,
            body: e.to_string(),
        }
    }
}

use crate::parsers::take_chars as truncate_chars;

// ---------------------------------------------------------------------------
// Module-level convenience wrappers
// ---------------------------------------------------------------------------

/// Call the [`Role::Index`] LLM using the process configuration. Used by the
/// summarizer and snapshot generator.
pub async fn call_llm(messages: &[Message]) -> Result<LlmResponse, LlmError> {
    LlmClient::from_config(crate::config::get_config())?
        .call_llm(messages)
        .await
}

/// Call the [`Role::Index`] LLM expecting JSON, using the process
/// configuration.
pub async fn call_llm_json(messages: &[Message]) -> Result<Value, LlmError> {
    LlmClient::from_config(crate::config::get_config())?
        .call_llm_json(messages)
        .await
}

// ---------------------------------------------------------------------------
// JSON recovery trio
// ---------------------------------------------------------------------------

use crate::utils::is_extended_whitespace as is_trim_space;

/// Remove markdown code fences from a JSON response.
///
/// Strip, drop a leading ```` ```json ```` (7 chars) else a leading
/// ```` ``` ```` (3 chars), drop a trailing ```` ``` ````, strip again.
///
/// The strips use the extended whitespace set ([`is_trim_space`]) rather than
/// `str::trim`: that set also covers U+001C..U+001F, so a stray U+001C before
/// the fence is removed and the fence is still recognised.
pub fn strip_markdown_json(text: &str) -> String {
    let text = text.trim_matches(is_trim_space);
    let text = if let Some(rest) = text.strip_prefix("```json") {
        rest
    } else if let Some(rest) = text.strip_prefix("```") {
        rest
    } else {
        text
    };
    let text = text.strip_suffix("```").unwrap_or(text);
    text.trim_matches(is_trim_space).to_string()
}

/// Extract a JSON object from surrounding text by brace-depth scanning.
///
/// The scan is deliberately **string-unaware**: a `}` inside a JSON string
/// literal closes the object, so `text {"a": "}", "b": 1}` yields `{"a": "}`,
/// which then fails to parse and falls through to `{}`. This is last-ditch
/// recovery for a model that wrapped its JSON in prose, and the behaviour is
/// pinned by tests — making it string-aware would change which malformed
/// responses succeed.
pub fn extract_json_object(text: &str) -> String {
    let Some(start) = text.find('{') else {
        return text.to_string();
    };

    let mut depth: i64 = 0;
    for (i, ch) in text[start..].char_indices() {
        if ch == '{' {
            depth += 1;
        } else if ch == '}' {
            depth -= 1;
            if depth == 0 {
                // '}' is one byte, so start + i + 1 is a char boundary.
                return text[start..start + i + 1].to_string();
            }
        }
    }

    text[start..].to_string()
}

/// Parse JSON from an LLM response with fallbacks: direct parse → strip
/// markdown fences → brace-scan extraction → `{}`.
///
/// Known limits of the parser, none of them reachable from a model answering
/// one of ekko's prompts — each simply falls through to the next recovery tier
/// and ultimately to `{}`:
/// - `NaN` / `Infinity` / `-Infinity` literals (syntax error);
/// - out-of-range floats such as `1e400` ("number out of range");
/// - lone surrogate escapes such as `"\ud800"` (Rust `String` is well-formed
///   UTF-8, so these are rejected);
/// - nesting deeper than serde_json's 128-level recursion limit (verified:
///   200 nested arrays fail here);
/// - integers wider than `i64`/`u64`, which serde_json (without
///   `arbitrary_precision`) silently widens to `f64`
///   (`123456789...` → `1.2345678901234568e28`) — the one case that parses
///   but loses precision rather than falling through to `{}`.
pub fn parse_json_response(text: &str) -> Value {
    if text.is_empty() {
        return Value::Object(Map::new());
    }

    if let Ok(value) = serde_json::from_str::<Value>(text) {
        return value;
    }

    let cleaned = strip_markdown_json(text);
    if let Ok(value) = serde_json::from_str::<Value>(&cleaned) {
        return value;
    }

    let extracted = extract_json_object(&cleaned);
    serde_json::from_str::<Value>(&extracted).unwrap_or_else(|_| Value::Object(Map::new()))
}

/// Emptiness test for a parsed JSON value, used as the re-ask gate in
/// `call_llm_json`: `{}`, `[]`, `""`, `0`, `false` and `null` all count as
/// "no answer", so the model is asked again.
use crate::models::is_truthy;

#[cfg(test)]
mod tests;
