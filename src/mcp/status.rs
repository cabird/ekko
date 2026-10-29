//! `get_status` — the diagnostics tool.
//!
//! Reports index counts, indexer lifecycle state, LLM configuration and
//! token costs. The key order of the result is pinned by tests.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use serde_json::{Map, Value, json};

use crate::background::STARTUP_DELAY_SECONDS;
use crate::config::Config;
use crate::db::{CostBucket, CostStats, Database, DbResult};
use crate::llm::Role;
use crate::paths;

/// Where the background indexer is in its lifecycle, as `get_status`
/// reports it.
///
/// The lifespan owns the indexer handle, so the MCP server carries this
/// shared flag instead: `"not_started"` before the lifespan starts the task,
/// `"running"` while it lives, `"stopped"` after shutdown.
#[derive(Debug, Clone, Default)]
pub struct IndexerStatus(Arc<AtomicU8>);

const NOT_STARTED: u8 = 0;
const RUNNING: u8 = 1;
const STOPPED: u8 = 2;

impl IndexerStatus {
    /// A status that has not been started yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark the indexer as started.
    pub fn mark_running(&self) {
        self.0.store(RUNNING, Ordering::SeqCst);
    }

    /// Mark the indexer as shut down.
    pub fn mark_stopped(&self) {
        self.0.store(STOPPED, Ordering::SeqCst);
    }

    /// `"not_started"`, `"running"` or `"stopped"`.
    pub fn as_str(&self) -> &'static str {
        match self.0.load(Ordering::SeqCst) {
            RUNNING => "running",
            STOPPED => "stopped",
            _ => "not_started",
        }
    }

    /// Whether the indexer task has ever been started; `instance_id` is only
    /// reported once it has.
    pub fn started(&self) -> bool {
        self.0.load(Ordering::SeqCst) != NOT_STARTED
    }
}

/// The API-key environment variables `get_status` probes, in report order.
/// ekko only *calls* OpenAI and Anthropic, but the list is wider so a user
/// who configured another provider still sees it acknowledged.
const KEY_ENV_VARS: &[(&str, &str)] = &[
    ("openai", "OPENAI_API_KEY"),
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("azure", "AZURE_OPENAI_API_KEY"),
    ("google", "GEMINI_API_KEY"),
    ("cohere", "COHERE_API_KEY"),
];

/// How many trailing log lines are scanned for errors, and how many of the
/// matches are reported.
const LOG_TAIL_LINES: usize = 50;
const MAX_REPORTED_ERRORS: usize = 5;

/// Build the `get_status` result.
///
/// # Errors
///
/// Propagates database errors.
pub fn status(
    db: &Database,
    config: &Config,
    telemetry_instance_id: &str,
    indexer: &IndexerStatus,
    indexer_instance_id: &str,
) -> DbResult<Value> {
    let stats = db.get_stats()?;
    // Report the database actually in use — config.database_path may
    // override the default location (P9 review: status previously skewed
    // from the tools' DB when the override was set).
    let db_path = db.path().to_path_buf();

    let needing_summary = db.get_sessions_needing_summary(Some(1000), true)?.len();
    let needing_snapshot = db
        .get_sessions_needing_snapshot(Some(1000), true, None, false)?
        .len();

    let db_size_mb = match std::fs::metadata(&db_path) {
        Ok(meta) => json!(round2(meta.len() as f64 / (1024.0 * 1024.0))),
        Err(_) => Value::Null,
    };

    let api_keys_available: Vec<&str> = KEY_ENV_VARS
        .iter()
        .filter(|(_, env)| config.get_api_key(env).is_some_and(|k| !k.is_empty()))
        .map(|(provider, _)| *provider)
        .collect();

    let role_json = |role: Role| -> Value {
        let resolved = crate::llm::resolve_backend(config, role);
        let (available, msg) = crate::llm::backend_availability(config, &resolved);
        json!({
            "backend": resolved.backend.name(),
            "model": resolved.model,
            "reason": resolved.reason,
            "available": available,
            "cost_basis": match resolved.backend {
                crate::llm::Backend::Cli(_) => "notional (subscription CLI; not billed per call)",
                crate::llm::Backend::Api => "api",
            },
            "warning": if available { Value::Null } else { json!(msg) },
        })
    };
    let index = role_json(Role::Index);
    let query = role_json(Role::Query);
    // Web UI hosting: read the lease. A negative remaining means the holder
    // died without releasing; the next election (≤60s) will replace it.
    let web = match db.lease_info(crate::db::leases::WEB_LEASE)? {
        Some((holder, remaining)) => {
            let port = holder
                .rsplit(':')
                .next()
                .and_then(|p| p.parse::<u16>().ok())
                .unwrap_or(config.web_port);
            if remaining >= 0 {
                json!({
                    "status": "serving",
                    "url": format!("http://127.0.0.1:{port}/"),
                    "holder": holder,
                })
            } else {
                json!({ "status": "stale", "holder": holder })
            }
        }
        None => json!({
            "status": "not_running",
            "hint": if config.web_serve {
                "web_serve is on; an instance should pick it up within a minute"
            } else {
                "run `ekko web`, or set web_serve=true to co-host in the MCP server"
            },
        }),
    };
    // Top-level `model` / `missing_key_warning` keep their original meaning
    // for existing readers: the indexing configuration.
    let model = index["model"].clone();
    let missing_key_warning = index["warning"].clone();

    let log_path = paths::get_data_dir()
        .map(|d| d.join("indexer.log"))
        .unwrap_or_default();
    let recent_errors = recent_errors(&log_path);

    let costs = cost_stats_to_json(&db.get_cost_stats(Some(telemetry_instance_id))?);

    let by_source: Map<String, Value> = stats
        .by_cli
        .iter()
        .map(|(source, count)| (source.clone(), json!(count)))
        .collect();

    Ok(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "database": {
            "path": db_path.to_string_lossy(),
            "size_mb": db_size_mb,
        },
        "sessions": {
            "total": stats.total_sessions,
            "with_summary": stats.with_summary,
            "with_snapshot": stats.with_snapshot,
            "needing_summary": needing_summary,
            "needing_snapshot": needing_snapshot,
            "by_source": Value::Object(by_source),
        },
        "indexer": {
            "status": indexer.as_str(),
            "instance_id": if indexer.started() { json!(indexer_instance_id) } else { Value::Null },
            "startup_delay_seconds": STARTUP_DELAY_SECONDS,
            "recent_errors": match recent_errors {
                Some(errors) => json!(errors),
                None => Value::Null,
            },
        },
        "costs": costs,
        "web": web,
        "config": {
            "model": model,
            "index": index,
            "query": query,
            "max_workers": crate::llm::effective_max_workers(config),
            "reasoning_effort": config.llm_reasoning_effort,
            "api_keys_available": api_keys_available,
            "missing_key_warning": missing_key_warning,
        }
    }))
}

fn round2(x: f64) -> f64 {
    crate::utils::round_decimal(x, 2)
}

/// The last few `[ERROR]`/`[WARNING]` lines of the indexer log. Returns
/// `None`, not an empty list, when there are none — the field is then
/// omitted from the report entirely.
fn recent_errors(log_path: &Path) -> Option<Vec<String>> {
    let text = std::fs::read_to_string(log_path).ok()?;
    let lines: Vec<&str> = text.lines().collect();
    let tail = &lines[lines.len().saturating_sub(LOG_TAIL_LINES)..];
    let mut errors: Vec<String> = tail
        .iter()
        .filter(|line| line.contains("[ERROR]") || line.contains("[WARNING]"))
        .map(|line| line.trim().to_string())
        .collect();
    if errors.is_empty() {
        return None;
    }
    if errors.len() > MAX_REPORTED_ERRORS {
        errors.drain(..errors.len() - MAX_REPORTED_ERRORS);
    }
    Some(errors)
}

/// [`CostStats`] as the JSON object `get_status` reports: `this_instance`
/// first, and only when an instance id was supplied.
pub fn cost_stats_to_json(stats: &CostStats) -> Value {
    let mut out = Map::new();
    if let Some(bucket) = &stats.this_instance {
        out.insert("this_instance".into(), bucket_to_json(bucket));
    }
    out.insert("today".into(), bucket_to_json(&stats.today));
    out.insert("last_7_days".into(), bucket_to_json(&stats.last_7_days));
    let by_operation: Map<String, Value> = stats
        .by_operation_7d
        .iter()
        .map(|(op, cost)| {
            (
                op.clone(),
                json!({
                    "cost": cost.cost,
                    "cost_display": cost.cost_display,
                    "calls": cost.calls,
                }),
            )
        })
        .collect();
    out.insert("by_operation_7d".into(), Value::Object(by_operation));
    Value::Object(out)
}

fn bucket_to_json(bucket: &CostBucket) -> Value {
    json!({
        "cost": bucket.cost,
        "cost_display": bucket.cost_display,
        "input_tokens": bucket.input_tokens,
        "output_tokens": bucket.output_tokens,
        "thinking_tokens": bucket.thinking_tokens,
        "calls": bucket.calls,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexer_status_transitions() {
        let s = IndexerStatus::new();
        assert_eq!(s.as_str(), "not_started");
        assert!(!s.started());
        s.mark_running();
        assert_eq!(s.as_str(), "running");
        assert!(s.started());
        s.mark_stopped();
        assert_eq!(s.as_str(), "stopped");
        assert!(s.started());
    }

    #[test]
    fn recent_errors_keeps_the_last_five_matches_only() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("indexer.log");
        let mut body = String::new();
        for i in 0..8 {
            body.push_str(&format!(
                "2026-01-01 00:00:00,000 [ERROR] ekko.background: e{i}\n"
            ));
        }
        body.push_str("2026-01-01 00:00:00,000 [INFO] ekko.background: fine\n");
        std::fs::write(&path, body).unwrap();

        let errors = recent_errors(&path).unwrap();
        assert_eq!(errors.len(), 5);
        assert!(errors[0].ends_with("e3"));
        assert!(errors[4].ends_with("e7"));
    }

    #[test]
    fn recent_errors_is_none_without_matches_or_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("indexer.log");
        assert!(recent_errors(&path).is_none());
        std::fs::write(&path, "2026-01-01 [INFO] ekko.background: ok\n").unwrap();
        assert!(recent_errors(&path).is_none());
    }

    #[test]
    fn cost_stats_json_shape_is_pinned() {
        let stats = CostStats {
            this_instance: Some(CostBucket {
                cost: 0.5,
                cost_display: "$0.5000".into(),
                input_tokens: 10,
                output_tokens: 2,
                thinking_tokens: 1,
                calls: 1,
            }),
            today: CostBucket {
                cost: 0.0,
                cost_display: "$0.0000".into(),
                input_tokens: 0,
                output_tokens: 0,
                thinking_tokens: 0,
                calls: 0,
            },
            last_7_days: CostBucket {
                cost: 0.0,
                cost_display: "$0.0000".into(),
                input_tokens: 0,
                output_tokens: 0,
                thinking_tokens: 0,
                calls: 0,
            },
            by_operation_7d: vec![],
        };
        let value = cost_stats_to_json(&stats);
        let keys: Vec<&str> = value
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            ["this_instance", "today", "last_7_days", "by_operation_7d"]
        );
        let bucket_keys: Vec<&str> = value["this_instance"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            bucket_keys,
            [
                "cost",
                "cost_display",
                "input_tokens",
                "output_tokens",
                "thinking_tokens",
                "calls"
            ]
        );
        assert_eq!(value["by_operation_7d"], json!({}));
    }
}
