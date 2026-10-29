//! Configuration loading.
//!
//! Precedence: env vars > `config.json` (in the ekko data dir) > defaults.
//! Unknown keys are ignored. API keys found in `config.json`
//! (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and the `AZURE_OPENAI_*` triple —
//! see [`API_KEY_ENV_KEYS`]) are kept on the [`Config`] and served via
//! [`Config::get_api_key`], where a non-empty process env var wins over the
//! file value.
//!
//! Nothing is ever written into the process environment: ekko owns its HTTP
//! clients, so the LLM layer reads keys through `get_api_key` instead.
//!
//! Env override names are `EKKO_MODEL` / `EKKO_REASONING`.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde_json::Value;

use crate::paths;

/// Default LLM model.
pub const DEFAULT_LLM_MODEL: &str = "gpt-5-mini";
/// Default reasoning effort.
pub const DEFAULT_REASONING_EFFORT: &str = "low";
/// Default max workers for parallel processing.
pub const DEFAULT_MAX_WORKERS: usize = 7;
/// Default max sessions for QA queries.
pub const DEFAULT_MAX_SESSIONS: usize = 7;

/// Env var overriding the LLM model (highest precedence).
pub const ENV_MODEL: &str = "EKKO_MODEL";
/// Env var overriding the reasoning effort (highest precedence).
pub const ENV_REASONING: &str = "EKKO_REASONING";
/// Env var overriding the completion backend (highest precedence). See
/// [`crate::llm::Backend`] for the accepted values.
pub const ENV_BACKEND: &str = "EKKO_BACKEND";
/// Env vars overriding the interactive-query backend and model
/// ([`crate::llm::Role::Query`]).
pub const ENV_QUERY_BACKEND: &str = "EKKO_QUERY_BACKEND";
pub const ENV_QUERY_MODEL: &str = "EKKO_QUERY_MODEL";
/// Default localhost port for `ekko web`.
pub const DEFAULT_WEB_PORT: u16 = 3556;
/// Default whole-call timeout for CLI completion backends, in seconds. CLI
/// runs carry the assistant's own startup cost on top of the model call, and
/// a 200K-token chunk with a long answer can take minutes.
pub const DEFAULT_CLI_TIMEOUT_SECS: u64 = 600;

/// API-key names recognized in config.json and served by
/// [`Config::get_api_key`]. ekko calls OpenAI and Anthropic only, plus the
/// Azure OpenAI triple: when
/// `AZURE_OPENAI_ENDPOINT` is set, OpenAI-family models are routed to that
/// Azure resource (see [`crate::llm`]). The endpoint and api-version ride
/// this same lookup path so they get the identical env-over-config
/// precedence.
const API_KEY_ENV_KEYS: [&str; 5] = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
];

/// Loaded configuration: the defaults merged with config.json and env
/// overrides. `database_path` / `session_paths` of `None` mean "use the
/// platform defaults from [`crate::paths`]".
///
/// `session_paths` preserves config.json insertion order — serde_json's
/// `preserve_order` feature plus a `Vec` of pairs, because a `BTreeMap` would
/// sort the CLI sources and change which parser sees a path first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub llm_model: String,
    /// True when `llm_model` came from config.json or the environment rather
    /// than [`DEFAULT_LLM_MODEL`]. A CLI backend picks its own default model
    /// when the user did not choose one (see [`crate::llm::resolve_backend`]).
    pub llm_model_explicit: bool,
    pub llm_reasoning_effort: String,
    /// Completion backend name (`"api"`, `"claude-cli"`, …), or `None` for
    /// auto-detection. Parsed by [`crate::llm::Backend::parse`].
    pub llm_backend: Option<String>,
    /// Backend for interactive queries (`ask_question`, semantic search),
    /// overriding `llm_backend` for [`crate::llm::Role::Query`]. `None`
    /// means "same as indexing".
    pub query_backend: Option<String>,
    /// Model for interactive queries; `None` means `llm_model` (when the
    /// query backend can serve it) or that backend's default.
    pub query_model: Option<String>,
    /// Whole-call timeout for CLI completion backends.
    pub cli_timeout_secs: u64,
    /// Localhost port for `ekko web`.
    pub web_port: u16,
    /// Host the web UI from inside `ekko mcp`: each MCP
    /// instance races for the port once a minute and exactly one serves.
    /// Off by default — an HTTP server inside every AI-tool launch is a
    /// choice the user should make.
    pub web_serve: bool,
    pub max_workers: usize,
    /// True when `max_workers` was set explicitly. CLI backends default to a
    /// single worker because subscription usage windows are opaque.
    pub max_workers_explicit: bool,
    pub max_sessions: usize,
    pub database_path: Option<PathBuf>,
    pub session_paths: Option<Vec<(String, Vec<PathBuf>)>>,
    /// API keys read from config.json (the [`API_KEY_ENV_KEYS`] names only),
    /// consulted by [`Config::get_api_key`] when the process env does
    /// not provide the key.
    pub api_keys: Vec<(String, String)>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            llm_model: DEFAULT_LLM_MODEL.to_string(),
            llm_model_explicit: false,
            llm_reasoning_effort: DEFAULT_REASONING_EFFORT.to_string(),
            llm_backend: None,
            query_backend: None,
            query_model: None,
            cli_timeout_secs: DEFAULT_CLI_TIMEOUT_SECS,
            web_port: DEFAULT_WEB_PORT,
            web_serve: false,
            max_workers: DEFAULT_MAX_WORKERS,
            max_workers_explicit: false,
            max_sessions: DEFAULT_MAX_SESSIONS,
            database_path: None,
            session_paths: None,
            api_keys: Vec::new(),
        }
    }
}

/// The env-var overrides, one named field each (five same-typed values in
/// positional order is a shape where a transposition passes every test).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EnvOverrides {
    pub model: Option<String>,
    pub reasoning: Option<String>,
    pub backend: Option<String>,
    pub query_backend: Option<String>,
    pub query_model: Option<String>,
}

impl EnvOverrides {
    /// Read the overrides from the process environment.
    pub fn from_env() -> Self {
        EnvOverrides {
            model: env::var(ENV_MODEL).ok(),
            reasoning: env::var(ENV_REASONING).ok(),
            backend: env::var(ENV_BACKEND).ok(),
            query_backend: env::var(ENV_QUERY_BACKEND).ok(),
            query_model: env::var(ENV_QUERY_MODEL).ok(),
        }
    }
}

impl Config {
    /// Merge recognized keys from a parsed config.json object over `self`.
    /// Unknown keys and wrongly-typed values are ignored per key, so one bad
    /// entry never costs the user the rest of their config.
    fn apply_json_object(&mut self, map: &serde_json::Map<String, Value>) {
        if let Some(v) = map.get("llm_model").and_then(Value::as_str) {
            self.llm_model = v.to_string();
            self.llm_model_explicit = true;
        }
        if let Some(v) = map.get("llm_reasoning_effort").and_then(Value::as_str) {
            self.llm_reasoning_effort = v.to_string();
        }
        // Optional-string keys share one rule: non-empty string sets, null
        // clears, anything else (including "") is ignored.
        for (key, slot) in [
            ("llm_backend", &mut self.llm_backend),
            ("query_backend", &mut self.query_backend),
            ("query_model", &mut self.query_model),
        ] {
            match map.get(key) {
                Some(Value::String(s)) if !s.is_empty() => *slot = Some(s.clone()),
                Some(Value::Null) => *slot = None,
                _ => {}
            }
        }
        if let Some(v) = map.get("web_serve").and_then(Value::as_bool) {
            self.web_serve = v;
        }
        if let Some(v) = map.get("web_port").and_then(Value::as_u64)
            && (1..=65535).contains(&v)
        {
            self.web_port = v as u16;
        }
        if let Some(v) = map.get("cli_timeout_secs").and_then(Value::as_u64) {
            self.cli_timeout_secs = v.max(1);
        }
        if let Some(v) = map.get("max_workers").and_then(Value::as_u64) {
            self.max_workers = v as usize;
            self.max_workers_explicit = true;
        }
        if let Some(v) = map.get("max_sessions").and_then(Value::as_u64) {
            self.max_sessions = v as usize;
        }
        match map.get("database_path") {
            Some(Value::String(s)) => self.database_path = Some(PathBuf::from(s)),
            Some(Value::Null) => self.database_path = None,
            _ => {}
        }
        match map.get("session_paths") {
            Some(Value::Object(sources)) => {
                // serde_json's preserve_order feature keeps config.json
                // insertion order; the CLI sources must not be sorted.
                let mut parsed: Vec<(String, Vec<PathBuf>)> = Vec::new();
                for (source, val) in sources {
                    let Some(arr) = val.as_array() else { continue };
                    // Non-string entries are dropped rather than erroring.
                    let list: Vec<PathBuf> = arr
                        .iter()
                        .filter_map(Value::as_str)
                        .map(PathBuf::from)
                        .collect();
                    parsed.push((source.clone(), list));
                }
                self.session_paths = Some(parsed);
            }
            Some(Value::Null) => self.session_paths = None,
            _ => {}
        }
        for key in API_KEY_ENV_KEYS {
            if let Some(val) = map.get(key).and_then(Value::as_str)
                && !val.is_empty()
            {
                self.api_keys.push((key.to_string(), val.to_string()));
            }
        }
    }

    /// Apply env-var overrides (highest precedence). An empty value counts as
    /// unset and is ignored.
    fn apply_env(&mut self, env: &EnvOverrides) {
        if let Some(m) = env.model.as_deref().filter(|v| !v.is_empty()) {
            self.llm_model = m.to_string();
            self.llm_model_explicit = true;
        }
        if let Some(r) = env.reasoning.as_deref().filter(|v| !v.is_empty()) {
            self.llm_reasoning_effort = r.to_string();
        }
        if let Some(b) = env.backend.as_deref().filter(|v| !v.is_empty()) {
            self.llm_backend = Some(b.to_string());
        }
        if let Some(b) = env.query_backend.as_deref().filter(|v| !v.is_empty()) {
            self.query_backend = Some(b.to_string());
        }
        if let Some(m) = env.query_model.as_deref().filter(|v| !v.is_empty()) {
            self.query_model = Some(m.to_string());
        }
    }

    /// Look up an API key (e.g. `"OPENAI_API_KEY"`): a non-empty process env
    /// var wins, otherwise the config.json value (if any). The LLM layer calls
    /// this instead of ekko mutating the process environment.
    pub fn get_api_key(&self, env_name: &str) -> Option<String> {
        if let Ok(v) = env::var(env_name)
            && !v.is_empty()
        {
            return Some(v);
        }
        self.api_keys
            .iter()
            .find(|(key, _)| key == env_name)
            .map(|(_, val)| val.clone())
    }
}

/// Load configuration from `<data_dir>/config.json` plus env overrides.
/// Uncached; this is the test-visible entry point that avoids global state.
/// Missing or malformed config.json falls back to defaults.
pub fn load_config_from_dir(data_dir: &Path) -> Config {
    let mut config = Config::default();

    let config_path = data_dir.join("config.json");
    if let Ok(text) = fs::read_to_string(&config_path)
        && let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&text)
    {
        config.apply_json_object(&map);
    }

    config.apply_env(&EnvOverrides::from_env());
    config
}

/// Load config with precedence: env vars > config.json > defaults.
///
/// If the data dir cannot be determined/created, the file layer is skipped
/// and defaults + env overrides are used.
pub fn load_config() -> Config {
    match paths::get_data_dir() {
        Ok(dir) => load_config_from_dir(&dir),
        Err(_) => {
            let mut config = Config::default();
            config.apply_env(&EnvOverrides::from_env());
            config
        }
    }
}

static CONFIG: OnceLock<Config> = OnceLock::new();

/// Get the loaded configuration: a process-wide singleton, loaded once.
pub fn get_config() -> &'static Config {
    CONFIG.get_or_init(load_config)
}





#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use crate::paths::testenv::{EnvVarGuard, env_lock};
    use tempfile::TempDir;

    fn write_config(dir: &Path, json: &str) {
        fs::write(dir.join("config.json"), json).unwrap();
    }

    #[test]
    fn defaults_are_pinned() {
        let c = Config::default();
        assert_eq!(c.llm_model, "gpt-5-mini");
        assert_eq!(c.llm_reasoning_effort, "low");
        assert_eq!(c.max_workers, 7);
        assert_eq!(c.max_sessions, 7);
        assert_eq!(c.database_path, None);
        assert_eq!(c.session_paths, None);
        assert!(c.api_keys.is_empty());
    }

    #[test]
    fn env_var_names_are_ekko_prefixed() {
        // Guard against silent renames: these exact strings are the public
        // contract, and every env override ekko reads is `EKKO_`-prefixed.
        assert_eq!(ENV_MODEL, "EKKO_MODEL");
        assert_eq!(ENV_REASONING, "EKKO_REASONING");
    }

    #[test]
    fn foreign_env_vars_are_ignored() {
        // Negative guard: env names ekko does not own must NOT affect config,
        // even plausible-looking MODEL/REASONING names under another prefix.
        let _lock = env_lock();
        let _m = EnvVarGuard::unset(ENV_MODEL);
        let _r = EnvVarGuard::unset(ENV_REASONING);
        let _sm = EnvVarGuard::set("OTHERTOOL_MODEL", "should-not-apply");
        let _sr = EnvVarGuard::set("OTHERTOOL_REASONING", "should-not-apply");
        let tmp = TempDir::new().unwrap();
        let c = load_config_from_dir(tmp.path());
        assert_eq!(c.llm_model, DEFAULT_LLM_MODEL);
        assert_eq!(c.llm_reasoning_effort, DEFAULT_REASONING_EFFORT);
    }

    #[test]
    fn missing_config_file_yields_defaults() {
        let _lock = env_lock();
        let _m = EnvVarGuard::unset(ENV_MODEL);
        let _r = EnvVarGuard::unset(ENV_REASONING);
        let tmp = TempDir::new().unwrap();
        assert_eq!(load_config_from_dir(tmp.path()), Config::default());
    }

    #[test]
    fn malformed_config_file_yields_defaults() {
        let _lock = env_lock();
        let _m = EnvVarGuard::unset(ENV_MODEL);
        let _r = EnvVarGuard::unset(ENV_REASONING);
        let tmp = TempDir::new().unwrap();
        write_config(tmp.path(), "{not json");
        assert_eq!(load_config_from_dir(tmp.path()), Config::default());
        write_config(tmp.path(), "[1, 2, 3]"); // non-object
        assert_eq!(load_config_from_dir(tmp.path()), Config::default());
    }

    #[test]
    fn file_beats_defaults_and_unknown_keys_ignored() {
        let _lock = env_lock();
        let _m = EnvVarGuard::unset(ENV_MODEL);
        let _r = EnvVarGuard::unset(ENV_REASONING);
        let tmp = TempDir::new().unwrap();
        write_config(
            tmp.path(),
            r#"{
                "llm_model": "claude-sonnet-4-5",
                "llm_reasoning_effort": "high",
                "max_workers": 3,
                "max_sessions": 12,
                "database_path": "/custom/db.sqlite",
                "session_paths": {"claude_code": ["/a", "/b"]},
                "totally_unknown_key": {"nested": true}
            }"#,
        );
        let c = load_config_from_dir(tmp.path());
        assert_eq!(c.llm_model, "claude-sonnet-4-5");
        assert_eq!(c.llm_reasoning_effort, "high");
        assert_eq!(c.max_workers, 3);
        assert_eq!(c.max_sessions, 12);
        assert_eq!(c.database_path, Some(PathBuf::from("/custom/db.sqlite")));
        let sp = c.session_paths.unwrap();
        assert_eq!(
            sp,
            vec![(
                "claude_code".to_string(),
                vec![PathBuf::from("/a"), PathBuf::from("/b")]
            )]
        );
    }

    #[test]
    fn session_paths_preserve_config_json_order() {
        // Insertion order is preserved: serde_json's preserve_order feature
        // plus a Vec<(String, _)>, where a BTreeMap would sort the sources.
        let _lock = env_lock();
        let _m = EnvVarGuard::unset(ENV_MODEL);
        let _r = EnvVarGuard::unset(ENV_REASONING);
        let tmp = TempDir::new().unwrap();
        write_config(
            tmp.path(),
            r#"{"session_paths": {
                "zeta_source": ["/z"],
                "alpha_source": ["/a"],
                "github_copilot": ["/g", 42, "/h"]
            }}"#,
        );
        let c = load_config_from_dir(tmp.path());
        let sp = c.session_paths.unwrap();
        let keys: Vec<&str> = sp.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, vec!["zeta_source", "alpha_source", "github_copilot"]);
        // Non-string entries are dropped rather than erroring.
        assert_eq!(sp[2].1, vec![PathBuf::from("/g"), PathBuf::from("/h")]);
    }

    #[test]
    fn wrongly_typed_values_are_ignored_per_key() {
        let _lock = env_lock();
        let _m = EnvVarGuard::unset(ENV_MODEL);
        let _r = EnvVarGuard::unset(ENV_REASONING);
        let tmp = TempDir::new().unwrap();
        write_config(
            tmp.path(),
            r#"{"llm_model": 42, "max_workers": "many", "max_sessions": 5}"#,
        );
        let c = load_config_from_dir(tmp.path());
        assert_eq!(c.llm_model, DEFAULT_LLM_MODEL);
        assert_eq!(c.max_workers, DEFAULT_MAX_WORKERS);
        assert_eq!(c.max_sessions, 5);
    }

    #[test]
    fn env_beats_file_beats_default() {
        let _lock = env_lock();
        let tmp = TempDir::new().unwrap();
        write_config(
            tmp.path(),
            r#"{"llm_model": "from-file", "llm_reasoning_effort": "medium"}"#,
        );

        // env set: env wins over file
        {
            let _m = EnvVarGuard::set(ENV_MODEL, "from-env");
            let _r = EnvVarGuard::set(ENV_REASONING, "high");
            let c = load_config_from_dir(tmp.path());
            assert_eq!(c.llm_model, "from-env");
            assert_eq!(c.llm_reasoning_effort, "high");
        }
        // env unset: file wins over default
        {
            let _m = EnvVarGuard::unset(ENV_MODEL);
            let _r = EnvVarGuard::unset(ENV_REASONING);
            let c = load_config_from_dir(tmp.path());
            assert_eq!(c.llm_model, "from-file");
            assert_eq!(c.llm_reasoning_effort, "medium");
        }
        // empty env value counts as unset — ignored
        {
            let _m = EnvVarGuard::set(ENV_MODEL, "");
            let _r = EnvVarGuard::unset(ENV_REASONING);
            let c = load_config_from_dir(tmp.path());
            assert_eq!(c.llm_model, "from-file");
        }
        // no file: defaults
        {
            let _m = EnvVarGuard::unset(ENV_MODEL);
            let _r = EnvVarGuard::unset(ENV_REASONING);
            let empty = TempDir::new().unwrap();
            let c = load_config_from_dir(empty.path());
            assert_eq!(c.llm_model, DEFAULT_LLM_MODEL);
        }
    }

    #[test]
    fn get_api_key_env_beats_file_beats_none() {
        // Precedence: a non-empty env var wins; otherwise the config.json
        // value; otherwise None. The process environment is never mutated —
        // the LLM layer reads keys via get_api_key instead.
        let _lock = env_lock();
        let _m = EnvVarGuard::unset(ENV_MODEL);
        let _r = EnvVarGuard::unset(ENV_REASONING);
        let tmp = TempDir::new().unwrap();
        write_config(
            tmp.path(),
            r#"{"OPENAI_API_KEY": "sk-from-file", "ANTHROPIC_API_KEY": "ant-from-file"}"#,
        );

        let _o = EnvVarGuard::unset("OPENAI_API_KEY");
        let _a = EnvVarGuard::set("ANTHROPIC_API_KEY", "ant-from-env");
        let c = load_config_from_dir(tmp.path());

        // unset in env → file value served
        assert_eq!(
            c.get_api_key("OPENAI_API_KEY").as_deref(),
            Some("sk-from-file")
        );
        // set in env → env wins
        assert_eq!(
            c.get_api_key("ANTHROPIC_API_KEY").as_deref(),
            Some("ant-from-env")
        );
        // loading must NOT have mutated the process env
        assert!(env::var_os("OPENAI_API_KEY").is_none());
        assert_eq!(env::var("ANTHROPIC_API_KEY").unwrap(), "ant-from-env");
    }

    #[test]
    fn get_api_key_empty_env_is_treated_as_unset() {
        // An env var set to '' counts as unset → the file value wins.
        let _lock = env_lock();
        let _m = EnvVarGuard::unset(ENV_MODEL);
        let _r = EnvVarGuard::unset(ENV_REASONING);
        let tmp = TempDir::new().unwrap();
        write_config(tmp.path(), r#"{"OPENAI_API_KEY": "sk-from-file"}"#);

        let _o = EnvVarGuard::set("OPENAI_API_KEY", "");
        let c = load_config_from_dir(tmp.path());
        assert_eq!(
            c.get_api_key("OPENAI_API_KEY").as_deref(),
            Some("sk-from-file")
        );
    }

    #[test]
    fn get_api_key_missing_everywhere_is_none() {
        let _lock = env_lock();
        let _m = EnvVarGuard::unset(ENV_MODEL);
        let _r = EnvVarGuard::unset(ENV_REASONING);
        let _o = EnvVarGuard::unset("OPENAI_API_KEY");
        let tmp = TempDir::new().unwrap();
        let c = load_config_from_dir(tmp.path());
        assert_eq!(c.get_api_key("OPENAI_API_KEY"), None);
        // An empty file value is ignored too.
        write_config(tmp.path(), r#"{"OPENAI_API_KEY": ""}"#);
        let c = load_config_from_dir(tmp.path());
        assert_eq!(c.get_api_key("OPENAI_API_KEY"), None);
    }

    #[test]
    fn dropped_provider_keys_are_not_stored() {
        // ekko calls OpenAI and Anthropic only, so GEMINI keys are NOT
        // recognized, and neither is the bare AZURE_API_KEY name — Azure
        // OpenAI uses the AZURE_OPENAI_* names (see the test below).
        let _lock = env_lock();
        let _m = EnvVarGuard::unset(ENV_MODEL);
        let _r = EnvVarGuard::unset(ENV_REASONING);
        let _g = EnvVarGuard::unset("GEMINI_API_KEY");
        let tmp = TempDir::new().unwrap();
        write_config(
            tmp.path(),
            r#"{"GEMINI_API_KEY": "g-key", "AZURE_API_KEY": "az"}"#,
        );

        let c = load_config_from_dir(tmp.path());
        assert!(c.api_keys.is_empty());
        assert_eq!(c.get_api_key("GEMINI_API_KEY"), None);
        assert!(env::var_os("GEMINI_API_KEY").is_none());
    }

    #[test]
    fn azure_openai_settings_are_recognized() {
        let _lock = env_lock();
        let _m = EnvVarGuard::unset(ENV_MODEL);
        let _r = EnvVarGuard::unset(ENV_REASONING);
        let _k = EnvVarGuard::unset("AZURE_OPENAI_API_KEY");
        let _e = EnvVarGuard::unset("AZURE_OPENAI_ENDPOINT");
        let _v = EnvVarGuard::unset("AZURE_OPENAI_API_VERSION");
        let tmp = TempDir::new().unwrap();
        write_config(
            tmp.path(),
            r#"{
                "AZURE_OPENAI_API_KEY": "az-key",
                "AZURE_OPENAI_ENDPOINT": "https://myres.openai.azure.com",
                "AZURE_OPENAI_API_VERSION": "preview"
            }"#,
        );

        let c = load_config_from_dir(tmp.path());
        assert_eq!(
            c.get_api_key("AZURE_OPENAI_API_KEY").as_deref(),
            Some("az-key")
        );
        assert_eq!(
            c.get_api_key("AZURE_OPENAI_ENDPOINT").as_deref(),
            Some("https://myres.openai.azure.com")
        );
        assert_eq!(
            c.get_api_key("AZURE_OPENAI_API_VERSION").as_deref(),
            Some("preview")
        );

        // Env still wins over the file, same as the other keys.
        let _k2 = EnvVarGuard::set("AZURE_OPENAI_API_KEY", "az-from-env");
        assert_eq!(
            c.get_api_key("AZURE_OPENAI_API_KEY").as_deref(),
            Some("az-from-env")
        );
    }

    #[test]
    fn null_overrides_keep_defaults() {
        let _lock = env_lock();
        let _m = EnvVarGuard::unset(ENV_MODEL);
        let _r = EnvVarGuard::unset(ENV_REASONING);
        let tmp = TempDir::new().unwrap();
        write_config(
            tmp.path(),
            r#"{"database_path": null, "session_paths": null}"#,
        );
        let c = load_config_from_dir(tmp.path());
        assert_eq!(c.database_path, None);
        assert_eq!(c.session_paths, None);
    }

    #[test]
    fn every_env_override_lands_on_its_own_field() {
        let env = EnvOverrides {
            model: Some("m-env".into()),
            reasoning: Some("r-env".into()),
            backend: Some("b-env".into()),
            query_backend: Some("qb-env".into()),
            query_model: Some("qm-env".into()),
        };
        let mut c = Config::default();
        c.apply_env(&env);
        assert_eq!(c.llm_model, "m-env");
        assert!(c.llm_model_explicit);
        assert_eq!(c.llm_reasoning_effort, "r-env");
        assert_eq!(c.llm_backend.as_deref(), Some("b-env"));
        assert_eq!(c.query_backend.as_deref(), Some("qb-env"));
        assert_eq!(c.query_model.as_deref(), Some("qm-env"));
        // Empty strings are ignored, not applied.
        let mut c = Config::default();
        c.apply_env(&EnvOverrides {
            model: Some(String::new()),
            backend: Some(String::new()),
            ..Default::default()
        });
        assert_eq!(c.llm_model, DEFAULT_LLM_MODEL);
        assert!(!c.llm_model_explicit);
        assert_eq!(c.llm_backend, None);
    }

    #[test]
    fn optional_string_config_keys_share_semantics() {
        for key in ["llm_backend", "query_backend", "query_model"] {
            let read = |c: &Config| match key {
                "llm_backend" => c.llm_backend.clone(),
                "query_backend" => c.query_backend.clone(),
                _ => c.query_model.clone(),
            };
            let mut c = Config::default();
            c.apply_json_object(json!({ key: "x" }).as_object().unwrap());
            assert_eq!(read(&c).as_deref(), Some("x"), "{key}: set");
            c.apply_json_object(json!({ key: "" }).as_object().unwrap());
            assert_eq!(read(&c).as_deref(), Some("x"), "{key}: empty ignored");
            c.apply_json_object(json!({ key: 7 }).as_object().unwrap());
            assert_eq!(read(&c).as_deref(), Some("x"), "{key}: wrong type ignored");
            c.apply_json_object(json!({ key: null }).as_object().unwrap());
            assert_eq!(read(&c), None, "{key}: null clears");
        }
    }

    #[test]
    fn backend_and_explicit_flags_round_trip() {
        let _lock = env_lock();
        let _m = EnvVarGuard::unset(ENV_MODEL);
        let _r = EnvVarGuard::unset(ENV_REASONING);
        let _b = EnvVarGuard::unset(ENV_BACKEND);
        let tmp = tempfile::tempdir().unwrap();

        // Nothing set: auto backend, defaults not "explicit".
        write_config(tmp.path(), "{}");
        let c = load_config_from_dir(tmp.path());
        assert_eq!(c.llm_backend, None);
        assert!(!c.llm_model_explicit);
        assert!(!c.max_workers_explicit);
        assert_eq!(c.cli_timeout_secs, DEFAULT_CLI_TIMEOUT_SECS);

        write_config(
            tmp.path(),
            r#"{"llm_backend": "claude-cli", "llm_model": "haiku", "max_workers": 2, "cli_timeout_secs": 0}"#,
        );
        let c = load_config_from_dir(tmp.path());
        assert_eq!(c.llm_backend.as_deref(), Some("claude-cli"));
        assert!(c.llm_model_explicit);
        assert_eq!(c.llm_model, "haiku");
        assert!(c.max_workers_explicit);
        assert_eq!(c.max_workers, 2);
        assert_eq!(c.cli_timeout_secs, 1, "clamped to at least one second");

        // Query overrides ride the same paths.
        let _qb = EnvVarGuard::unset(ENV_QUERY_BACKEND);
        let _qm = EnvVarGuard::unset(ENV_QUERY_MODEL);
        write_config(
            tmp.path(),
            r#"{"llm_backend": "claude-cli", "query_backend": "api", "query_model": "gpt-5-mini"}"#,
        );
        let c = load_config_from_dir(tmp.path());
        assert_eq!(c.query_backend.as_deref(), Some("api"));
        assert_eq!(c.query_model.as_deref(), Some("gpt-5-mini"));
        let _qb2 = EnvVarGuard::set(ENV_QUERY_BACKEND, "claude-cli");
        let _qm2 = EnvVarGuard::set(ENV_QUERY_MODEL, "haiku");
        let c = load_config_from_dir(tmp.path());
        assert_eq!(c.query_backend.as_deref(), Some("claude-cli"));
        assert_eq!(c.query_model.as_deref(), Some("haiku"));
        drop((_qb2, _qm2));

        write_config(
            tmp.path(),
            r#"{"llm_backend": "claude-cli", "llm_model": "haiku", "max_workers": 2, "cli_timeout_secs": 0}"#,
        );
        // Env overrides config.json for backend and model alike.
        let _b2 = EnvVarGuard::set(ENV_BACKEND, "api");
        let _m2 = EnvVarGuard::set(ENV_MODEL, "gpt-5-mini");
        let c = load_config_from_dir(tmp.path());
        assert_eq!(c.llm_backend.as_deref(), Some("api"));
        assert_eq!(c.llm_model, "gpt-5-mini");
        assert!(c.llm_model_explicit);
    }
}
