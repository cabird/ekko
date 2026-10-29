//! The ekko MCP server.
//!
//! [`EkkoServer`] exposes nine tools, each a thin wrapper over
//! [`crate::memory::EkkoMemory`] plus telemetry recording.
//!
//! # Client-visible result shape
//!
//! Every tool returns a [`serde_json::Value`] from the memory layer, rendered
//! two ways:
//!
//! - as a **text content block** of pretty-printed JSON (two-space indent),
//!   so a client reading `content[0].text` sees the whole payload with key
//!   order preserved (`serde_json` is built with `preserve_order`);
//! - and additionally as `structuredContent`. MCP requires that to be an
//!   object, so array results (`list_projects`, `sessions_in_directory`) are
//!   wrapped as `{"result": […]}`.
//!
//! ekko declares no `outputSchema` for any tool: the payloads are dynamic
//! maps, and a wrong schema is worse than none.
//!
//! # Errors
//!
//! A failing tool returns `Ok(CallToolResult::error(…))` carrying the
//! message, never `Err(ErrorData)`: MCP clients render protocol errors
//! opaquely and the user would lose the text. In-band `{"error": …}` objects
//! produced by the memory layer stay successful results.

use std::future::Future;

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, Implementation, ServerCapabilities, ServerInfo};
use rmcp::{ErrorData, ServerHandler, tool, tool_handler, tool_router};
use serde_json::{Map, Value, json};

use crate::config::Config;
use crate::db::{Database, DbResult, LlmUsage};
use crate::llm::{Role, check_backend_available};
use crate::memory::{AskArgs, ReadTranscriptArgs, RecentActivityArgs, SearchArgs, EkkoMemory};
use crate::telemetry::TelemetryRecorder;

pub mod descriptions;
pub mod instructions;
pub mod params;
pub mod schema;
pub mod serve;
pub mod status;

pub use serve::{init_tracing, run_stdio};
pub use status::IndexerStatus;

use params::{
    AskQuestionParams, ExportTelemetryParams, GetSessionSnapshotParams, ReadTranscriptParams,
    RecentActivityParams, SearchSessionsParams, SessionsInDirectoryParams,
};

/// The MCP server name advertised to clients.
pub const SERVER_NAME: &str = "ekko";

/// The crate version, used in the server info and the instructions banner.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The ekko MCP server.
///
/// The database and memory layer are built once in [`EkkoServer::new`]. A
/// [`Database`] handle is just a path — connections are opened per call — so
/// construction is cheap and side-effect-free beyond creating the schema.
#[derive(Clone)]
pub struct EkkoServer {
    memory: EkkoMemory,
    telemetry: TelemetryRecorder,
    config: &'static Config,
    indexer: IndexerStatus,
    tool_router: ToolRouter<Self>,
}

impl std::fmt::Debug for EkkoServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EkkoServer")
            .field("db", &self.memory.db().path())
            .field("instance_id", &self.telemetry.instance_id())
            .field("indexer", &self.indexer.as_str())
            .finish()
    }
}

impl EkkoServer {
    /// Build a server around an existing database handle.
    pub fn new(db: Database, config: &'static Config) -> Self {
        EkkoServer {
            memory: EkkoMemory::new(db.clone()),
            telemetry: TelemetryRecorder::new(db),
            config,
            indexer: IndexerStatus::new(),
            tool_router: Self::tool_router(),
        }
    }

    /// Open the configured database and build a server around it.
    ///
    /// # Errors
    ///
    /// Fails if the data directory cannot be resolved or the database cannot
    /// be opened/initialized.
    pub fn from_config(config: &'static Config) -> Result<Self, String> {
        let path = match &config.database_path {
            Some(p) => p.clone(),
            None => crate::paths::get_database_path().map_err(|e| e.to_string())?,
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let db = Database::new(path).map_err(|e| e.to_string())?;
        Ok(EkkoServer::new(db, config))
    }

    /// The shared background-indexer status flag (the lifespan flips it).
    pub fn indexer_status(&self) -> &IndexerStatus {
        &self.indexer
    }

    /// The database handle the tools read.
    pub fn db(&self) -> &Database {
        self.memory.db()
    }

    /// The service layer the tools (and the web UI) are built on.
    pub fn memory(&self) -> &EkkoMemory {
        &self.memory
    }

    /// The process configuration.
    pub fn config(&self) -> &'static Config {
        self.config
    }

    /// The telemetry recorder (its instance id stamps every event).
    pub fn telemetry(&self) -> &TelemetryRecorder {
        &self.telemetry
    }

    /// The tool descriptions and schemas this server advertises, sorted by
    /// name. Handy for tests and for `ekko check`-style diagnostics.
    pub fn tools(&self) -> Vec<rmcp::model::Tool> {
        self.tool_router.list_all()
    }
}

// =============================================================================
// Result / telemetry plumbing
// =============================================================================

/// Render a memory-layer value as a tool result (see the module docs).
fn ok(value: &Value) -> CallToolResult {
    let text = serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string());
    let structured = match value {
        Value::Object(_) => value.clone(),
        other => json!({ "result": other }),
    };
    let mut result = CallToolResult::success(vec![ContentBlock::text(text)]);
    result.structured_content = Some(structured);
    result
}

/// Render a failure as an error result whose text the user can read.
fn failed(message: impl std::fmt::Display) -> CallToolResult {
    CallToolResult::error(vec![ContentBlock::text(message.to_string())])
}

/// Pull the `usage` object out of a tool result, accepting either token
/// naming convention, and price it with the *configured* model — not
/// whatever the result claims.
pub fn extract_usage_for_telemetry(result: &Value, config: &Config) -> Option<LlmUsage> {
    let usage = result.get("usage")?;
    if usage.is_null() || usage == &json!({}) {
        return None;
    }
    let field = |a: &str, b: &str| -> i64 {
        usage
            .get(a)
            .and_then(Value::as_i64)
            .or_else(|| usage.get(b).and_then(Value::as_i64))
            .unwrap_or(0)
    };
    let input_tokens = field("input_tokens", "prompt_tokens");
    let output_tokens = field("output_tokens", "completion_tokens");
    let thinking_tokens = field("reasoning_tokens", "thinking_tokens");

    // Price with the model that actually answered: the resolved backend's
    // model, mapped from a CLI alias (`sonnet`) to a priced id where needed.
    let resolved = crate::llm::resolve_backend(config, Role::Query);
    let model = match resolved.backend {
        crate::llm::Backend::Cli(kind) => kind.pricing_model(&resolved.model),
        crate::llm::Backend::Api => resolved.model,
    };
    // A CLI backend reports its own notional cost, which the tracker folds
    // into `usage.cost`; prefer that over re-pricing when it is present.
    let cost = match usage.get("cost").and_then(Value::as_f64) {
        Some(c) if c > 0.0 => c,
        _ => crate::pricing::get_cost(
            &model,
            input_tokens.max(0) as u64,
            output_tokens.max(0) as u64,
            thinking_tokens.max(0) as u64,
        ),
    };

    Some(LlmUsage {
        model: Some(model),
        input_tokens: Some(input_tokens),
        output_tokens: Some(output_tokens),
        thinking_tokens: Some(thinking_tokens),
        cost: Some(cost),
    })
}

/// Build the telemetry argument map a tool records;
/// [`crate::telemetry::filter_safe_args`] then applies the privacy allowlist
/// on top.
fn args(pairs: impl IntoIterator<Item = (&'static str, Value)>) -> Map<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

/// Run a blocking (SQLite / filesystem) memory call off the async reactor.
async fn blocking<T, F>(f: F) -> Result<T, CallToolResult>
where
    T: Send + 'static,
    F: FnOnce() -> DbResult<T> + Send + 'static,
{
    match tokio::task::spawn_blocking(f).await {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(e)) => Err(failed(e)),
        Err(e) => Err(failed(format!("task failed: {e}"))),
    }
}

/// `await` a memory call, mapping database errors to an error result.
async fn awaited<T, F>(f: F) -> Result<T, CallToolResult>
where
    F: Future<Output = DbResult<T>>,
{
    f.await.map_err(failed)
}

// =============================================================================
// Tools
// =============================================================================

#[tool_router(router = tool_router_internal)]
impl EkkoServer {
    /// Tier 1 — recent session activity.
    #[tool(name = "recent_activity")]
    pub async fn recent_activity(
        &self,
        Parameters(p): Parameters<RecentActivityParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let memory = self.memory.clone();
        let track = self.telemetry.track("recent_activity");
        let (directory, include_subdirs, days, max_sessions) = (
            p.directory.clone(),
            p.include_subdirs,
            p.days,
            p.max_sessions,
        );

        let result = blocking(move || {
            memory.recent_activity(&RecentActivityArgs {
                directory: directory.as_deref(),
                include_subdirs,
                days,
                max_sessions: max_sessions.max(0) as usize,
            })
        })
        .await;

        let value = match result {
            Ok(value) => value,
            Err(err) => return Ok(err),
        };
        track.finish_with(
            Some(&args([
                ("include_subdirs", json!(p.include_subdirs)),
                ("days", json!(p.days)),
                ("max_sessions", json!(p.max_sessions)),
            ])),
            None,
        );
        Ok(ok(&value))
    }

    /// Tier 1 — search across all sessions.
    #[tool(name = "search_sessions")]
    pub async fn search_sessions(
        &self,
        Parameters(p): Parameters<SearchSessionsParams>,
    ) -> Result<CallToolResult, ErrorData> {
        // Semantic/hybrid are gated on LLM availability before the memory
        // layer is entered, so an unconfigured backend yields advice rather
        // than a failed search.
        let requested_mode = p.mode.clone();
        let mode = match crate::llm::gate_search_mode(self.config, &p.mode) {
            Ok(mode) => mode,
            Err(msg) => {
                return Ok(ok(&json!({
                    "error": format!("Semantic search requires LLM access. {msg}"),
                    "suggestion": "Use mode=\"keyword\" or mode=\"transcript\" instead (no LLM needed).",
                })));
            }
        };

        let track = self.telemetry.track("search_sessions");
        let result = awaited(self.memory.search_sessions(&SearchArgs {
            query: &p.query,
            directory: p.directory.as_deref(),
            include_subdirs: p.include_subdirs,
            days: p.days,
            start_date: p.start_date.as_deref(),
            end_date: p.end_date.as_deref(),
            mode: &mode,
            limit: p.limit,
        }))
        .await;

        let mut value = match result {
            Ok(value) => value,
            Err(err) => return Ok(err),
        };

        // Note if hybrid was downgraded.
        if requested_mode == "hybrid"
            && mode != requested_mode
            && let Value::Object(map) = &mut value
        {
            map.insert(
                "mode_fallback".to_string(),
                json!({
                    "requested": requested_mode,
                    "used": mode,
                    "reason": "LLM not configured, fell back to keyword search",
                }),
            );
        }

        let usage = if mode == "semantic" || mode == "hybrid" {
            extract_usage_for_telemetry(&value, self.config)
        } else {
            None
        };
        track.finish_with(
            Some(&args([
                ("mode", json!(mode)),
                ("requested_mode", json!(requested_mode)),
                ("include_subdirs", json!(p.include_subdirs)),
                ("days", json!(p.days)),
                ("limit", json!(p.limit)),
            ])),
            usage.as_ref(),
        );
        Ok(ok(&value))
    }

    /// Tier 2 — the context snapshot for a session.
    #[tool(name = "get_session_snapshot")]
    pub async fn get_session_snapshot(
        &self,
        Parameters(p): Parameters<GetSessionSnapshotParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let memory = self.memory.clone();
        let track = self.telemetry.track("get_session_snapshot");
        let session_id = p.session_id.clone();

        let value = match blocking(move || memory.get_session_snapshot(&session_id)).await {
            Ok(value) => value,
            Err(err) => return Ok(err),
        };
        track.finish();
        Ok(ok(&value))
    }

    /// Tier 3 — flexible transcript access.
    #[tool(name = "read_transcript")]
    pub async fn read_transcript(
        &self,
        Parameters(p): Parameters<ReadTranscriptParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let memory = self.memory.clone();
        let track = self.telemetry.track("read_transcript");
        let p2 = p.clone();

        let value = match blocking(move || {
            memory.read_transcript(&ReadTranscriptArgs {
                session_id: &p2.session_id,
                mode: &p2.mode,
                count: p2.count,
                start: p2.start,
                end: p2.end,
                pattern: p2.pattern.as_deref(),
                context_lines: p2.context_lines,
                max_matches: p2.max_matches,
                include_tools: p2.include_tools,
                output_path: p2.output_path.as_deref(),
            })
        })
        .await
        {
            Ok(value) => value,
            Err(err) => return Ok(err),
        };
        track.finish_with(
            Some(&args([
                ("mode", json!(p.mode)),
                ("context_lines", json!(p.context_lines)),
                ("max_matches", json!(p.max_matches)),
                ("include_tools", json!(p.include_tools)),
            ])),
            None,
        );
        Ok(ok(&value))
    }

    /// Cross-tier — smart Q&A over session history.
    #[tool(name = "ask_question")]
    pub async fn ask_question(
        &self,
        Parameters(p): Parameters<AskQuestionParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let (available, msg) = check_backend_available(self.config, Role::Query);
        if !available {
            return Ok(ok(&json!({
                "error": format!("ask_question requires LLM access. {msg}"),
                "suggestion": "Use search_sessions(mode=\"keyword\") or search_sessions(mode=\"transcript\") for text-based search (no LLM needed).",
            })));
        }

        let track = self.telemetry.track("ask_question");
        // `Box::pin` is load-bearing: the transcript fan-out inside
        // `ask_question` builds a future whose higher-ranked lifetimes rustc
        // cannot prove general enough once the tool macro wraps it. Boxing
        // erases the type and the bound is satisfied.
        let result = awaited(Box::pin(self.memory.ask_question(&AskArgs {
            question: &p.question,
            directory: p.directory.as_deref(),
            include_subdirs: p.include_subdirs,
            days: p.days,
            max_sessions: p.max_sessions.max(0) as usize,
            offset: p.offset,
            ..AskArgs::new(&p.question)
        })))
        .await;

        let value = match result {
            Ok(value) => value,
            Err(err) => return Ok(err),
        };
        let usage = extract_usage_for_telemetry(&value, self.config);
        track.finish_with(
            Some(&args([
                ("include_subdirs", json!(p.include_subdirs)),
                ("days", json!(p.days)),
                ("max_sessions", json!(p.max_sessions)),
                ("offset", json!(p.offset)),
            ])),
            usage.as_ref(),
        );
        Ok(ok(&value))
    }

    /// All projects with session counts and recent activity.
    #[tool(name = "list_projects")]
    pub async fn list_projects(&self) -> Result<CallToolResult, ErrorData> {
        let memory = self.memory.clone();
        let track = self.telemetry.track("list_projects");
        let value = match blocking(move || memory.list_projects()).await {
            Ok(value) => value,
            Err(err) => return Ok(err),
        };
        track.finish();
        Ok(ok(&value))
    }

    /// All sessions in or under a directory.
    #[tool(name = "sessions_in_directory")]
    pub async fn sessions_in_directory(
        &self,
        Parameters(p): Parameters<SessionsInDirectoryParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let memory = self.memory.clone();
        let track = self.telemetry.track("sessions_in_directory");
        let (directory, include_subdirs, days, limit) =
            (p.directory.clone(), p.include_subdirs, p.days, p.limit);

        let value = match blocking(move || {
            memory.sessions_in_directory(&directory, include_subdirs, days, limit)
        })
        .await
        {
            Ok(value) => value,
            Err(err) => return Ok(err),
        };
        track.finish_with(
            Some(&args([
                ("include_subdirs", json!(p.include_subdirs)),
                ("days", json!(p.days)),
                ("limit", json!(p.limit)),
            ])),
            None,
        );
        Ok(ok(&value))
    }

    /// Database, indexer, cost and configuration status.
    #[tool(name = "get_status")]
    pub async fn get_status(&self) -> Result<CallToolResult, ErrorData> {
        let db = self.memory.db().clone();
        let config = self.config;
        let instance_id = self.telemetry.instance_id().to_string();
        let indexer = self.indexer.clone();

        // `get_status` deliberately records no telemetry event.
        let value = match blocking(move || {
            status::status(
                &db,
                config,
                &instance_id,
                &indexer,
                crate::background::instance_id(),
            )
        })
        .await
        {
            Ok(value) => value,
            Err(err) => return Ok(err),
        };
        Ok(ok(&value))
    }

    /// Export anonymous usage telemetry to a file.
    #[tool(name = "export_telemetry")]
    pub async fn export_telemetry(
        &self,
        Parameters(p): Parameters<ExportTelemetryParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let telemetry = self.telemetry.clone();
        let filename = p.filename.clone();
        // `export_telemetry` records no telemetry event either — it would
        // land in the file it is writing.
        let exported = tokio::task::spawn_blocking(move || telemetry.export_to_file(&filename))
            .await
            .map_err(|e| ErrorData::internal_error(format!("task failed: {e}"), None))?;

        match exported {
            Ok(export) => Ok(ok(&json!({
                "file_path": export.file_path.to_string_lossy(),
                "event_count": export.event_count,
                "summary": crate::telemetry::summary_to_json(&export.summary),
            }))),
            Err(e) => Ok(failed(e)),
        }
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for EkkoServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(SERVER_NAME, VERSION))
            .with_instructions(instructions::instructions(VERSION))
    }
}

impl EkkoServer {
    /// The tool router with descriptions installed and schemas sanitized.
    ///
    /// The `#[tool_router]` macro generates the raw router; this wrapper
    /// then (a) replaces each tool's description with the LLM-facing text in
    /// [`descriptions`] and (b) rewrites the schemas that would make Claude
    /// clients drop the tool list ([`schema::sanitize`]).
    pub fn tool_router() -> ToolRouter<Self> {
        let mut router = Self::tool_router_internal();
        let names: Vec<_> = router.map.keys().cloned().collect();
        for name in names {
            let Some(route) = router.map.get_mut(&name) else {
                continue;
            };
            if let Some(text) = descriptions::description(&name) {
                route.attr.description = Some(text.into());
            }
            route.attr = schema::sanitize_tool(route.attr.clone());
        }
        router
    }
}

#[cfg(test)]
mod tests;
