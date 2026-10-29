//! Embedded browser UI.
//!
//! A dependency-free single-page app (`assets/`) compiled into the binary
//! with `include_str!`, served by Axum on localhost over the same SQLite
//! archive the MCP tools use. Serves reads plus two deliberate writes:
//! `/api/ask` runs the same Q&A path as the `ask_question` tool, and the
//! transcript route ingests a not-yet-stored session's capped events on
//! first view (see `api_transcript`).
//!
//! Design: `docs/web-ui/PORT_PLAN.md`. The JSON shapes are the MCP tools'
//! (`EkkoMemory` returns them directly) wherever a tool already exists;
//! the list and transcript routes, which have no tool counterpart in this
//! form, carry the browser's own shapes.
//!
//! Localhost only, no authentication: anyone who can reach the port reads
//! the whole archive. That matches the MCP server's trust model.

pub mod server;

use axum::Router;
use axum::extract::{Path, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::db::{DbError, DbResult, SessionRow, WebCursor, WebListFilter, WebSort};
use crate::llm::{self, Role};
use crate::mcp::EkkoServer;
use crate::mcp::status;
use crate::memory::qa::AskArgs;
use crate::memory::search::SearchArgs;
use crate::parsers;

const INDEX_HTML: &str = include_str!("assets/index.html");
const STYLE_CSS: &str = include_str!("assets/style.css");
const APP_JS: &str = include_str!("assets/app.js");

/// Default `limit` for the session list.
pub const DEFAULT_PAGE_SIZE: i64 = 50;
/// Largest page a client may ask for.
pub const MAX_PAGE_SIZE: i64 = 200;
/// Build the router. `server` is the same state the MCP tools run on.
pub fn router(server: EkkoServer) -> Router {
    Router::new()
        .route("/", get(serve_index))
        .route("/style.css", get(serve_css))
        .route("/app.js", get(serve_js))
        .route("/api/status", get(api_status))
        .route("/api/projects", get(api_projects))
        .route("/api/sessions", get(api_sessions))
        .route("/api/sessions/{id}", get(api_session))
        .route("/api/sessions/{id}/transcript", get(api_transcript))
        .route("/api/search", get(api_search))
        .route("/api/ask", post(api_ask))
        // Everything unknown gets the same JSON error shape the frontend's
        // fetch wrapper expects, instead of axum's bare-text 404.
        .fallback(async || ApiError::not_found("no such route"))
        .with_state(server)
}

// ---------------------------------------------------------------------------
// Static assets
// ---------------------------------------------------------------------------

// `no-cache` means "revalidate, don't guess": the assets change with the
// binary and carry no ETag, so an unpinned browser cache could serve a
// stale app.js across an ekko upgrade.
async fn serve_index() -> impl IntoResponse {
    ([(header::CACHE_CONTROL, "no-cache")], Html(INDEX_HTML))
}

async fn serve_css() -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, "text/css; charset=utf-8"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        STYLE_CSS,
    )
}

async fn serve_js() -> impl IntoResponse {
    (
        [
            (
                header::CONTENT_TYPE,
                "application/javascript; charset=utf-8",
            ),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        APP_JS,
    )
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// A JSON error body with a status. Every handler failure goes through
/// here so the frontend sees one shape: `{"error": "..."}`.
#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        ApiError {
            status,
            message: message.into(),
        }
    }
    fn not_found(what: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, what)
    }
    fn bad_request(what: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, what)
    }
    fn internal(e: impl std::fmt::Display) -> Self {
        tracing::error!(error = %e, "web handler failed");
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    }
}

impl From<DbError> for ApiError {
    fn from(e: DbError) -> Self {
        ApiError::internal(e)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, axum::Json(json!({ "error": self.message }))).into_response()
    }
}

/// Run synchronous database work off the async executor. ekko's `Database`
/// is connection-per-call and blocking (see PORT_PLAN §2), so every handler
/// goes through here.
async fn blocking<T, F>(f: F) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: FnOnce() -> DbResult<T> + Send + 'static,
{
    match tokio::task::spawn_blocking(f).await {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(e.into()),
        Err(e) => Err(ApiError::internal(format!("task failed: {e}"))),
    }
}

// ---------------------------------------------------------------------------
// /api/status, /api/projects
// ---------------------------------------------------------------------------

async fn api_status(State(server): State<EkkoServer>) -> Result<axum::Json<Value>, ApiError> {
    let db = server.db().clone();
    let config = server.config();
    let instance_id = server.telemetry().instance_id().to_string();
    let indexer = server.indexer_status().clone();
    let mut value = blocking(move || {
        status::status(
            &db,
            config,
            &instance_id,
            &indexer,
            crate::background::instance_id(),
        )
    })
    .await?;
    // status() already carries a `web` hosting block (lease state, URL);
    // add the UI build version into it rather than replacing it.
    if let Some(web) = value.get_mut("web").and_then(Value::as_object_mut) {
        web.insert("ui_version".into(), json!(env!("CARGO_PKG_VERSION")));
    }
    Ok(axum::Json(value))
}

/// Projects with session counts, one entry per path (the MCP tool splits
/// by CLI source; the sidebar wants a single facet list).
async fn api_projects(State(server): State<EkkoServer>) -> Result<axum::Json<Value>, ApiError> {
    let db = server.db().clone();
    let rows = blocking(move || db.get_projects()).await?;

    let mut merged: Vec<Value> = Vec::new();
    for row in rows {
        if let Some(existing) = merged
            .iter_mut()
            .find(|p| p["path"].as_str() == Some(row.project_path.as_str()))
        {
            let count = existing["count"].as_i64().unwrap_or(0) + row.session_count;
            existing["count"] = json!(count);
            continue;
        }
        let name = std::path::Path::new(&row.project_path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| row.project_path.clone());
        merged.push(json!({
            "path": row.project_path,
            "name": name,
            "count": row.session_count,
        }));
    }
    Ok(axum::Json(json!({ "projects": merged })))
}

// ---------------------------------------------------------------------------
// /api/sessions
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ListParams {
    q: Option<String>,
    project: Option<String>,
    source: Option<String>,
    days: Option<i64>,
    sort: Option<String>,
    limit: Option<i64>,
    offset: Option<i64>,
    /// Opaque keyset token from a previous response's `next_cursor`;
    /// wins over `offset`. Malformed tokens fall back to the first page.
    cursor: Option<String>,
}

async fn api_sessions(
    State(server): State<EkkoServer>,
    Query(p): Query<ListParams>,
) -> Result<axum::Json<Value>, ApiError> {
    let limit = p.limit.unwrap_or(DEFAULT_PAGE_SIZE).clamp(1, MAX_PAGE_SIZE);
    let offset = p.offset.unwrap_or(0).max(0);
    let sort = WebSort::parse(p.sort.as_deref());
    let cursor = p.cursor.as_deref().and_then(WebCursor::parse);
    let db = server.db().clone();
    let page = blocking(move || {
        let filter = WebListFilter {
            q: p.q.as_deref(),
            project: p.project.as_deref(),
            source: p.source.as_deref(),
            days: p.days,
        };
        db.web_list_sessions(&filter, sort, limit, offset, cursor.as_ref())
    })
    .await?;

    let sessions: Vec<Value> = page.sessions.iter().map(list_item).collect();
    // Facet counts share the list's filter (minus the source facet itself),
    // so the dropdown always shows what picking each source would yield.
    let source_counts: serde_json::Map<String, Value> = page
        .source_counts
        .iter()
        .map(|(source, count)| (source.clone(), json!(count)))
        .collect();
    Ok(axum::Json(json!({
        "sessions": sessions,
        "total": page.total,
        "limit": limit,
        "offset": offset,
        "sort": sort.name(),
        "next_cursor": page.next_cursor,
        "source_counts": source_counts,
    })))
}

/// Sidebar fields only (PORT_PLAN §4: the list endpoint must not
/// over-fetch). Detail comes from `/api/sessions/{id}`.
fn list_item(row: &SessionRow) -> Value {
    json!({
        "session_id": row.session_id,
        "cli_source": row.cli_source,
        "title": row.title,
        "short_desc": row.summary.as_deref().map(short_desc),
        "first_prompt": row.first_prompt.as_deref().map(|s| truncate(s, 200)),
        "project_path": row.project_path,
        "git_branch": row.git_branch,
        "created_at": row.created_at,
        "modified_at": row.modified_at,
        "message_count": row.message_count,
        "duration_ms": row.duration_ms,
        "files_touched_count": row.files_touched_count,
        "has_summary": row.summary.as_deref().is_some_and(|s| !s.is_empty()),
        "has_snapshot": row.context_snapshot.as_deref().is_some_and(|s| !s.is_empty()),
    })
}

/// First line of a summary, minus the `Outcome:` label ekko's prompt puts
/// there, capped for the sidebar.
pub fn short_desc(summary: &str) -> String {
    let first = summary
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    let stripped = first
        .strip_prefix("Outcome:")
        .map(str::trim)
        .unwrap_or(first);
    truncate(stripped, 200)
}

fn truncate(s: &str, max: usize) -> String {
    match crate::db::events::cap(s, max) {
        (out, false, _) => out,
        (mut out, true, _) => {
            out.push('…');
            out
        }
    }
}

// ---------------------------------------------------------------------------
// /api/sessions/{id}, /api/sessions/{id}/transcript
// ---------------------------------------------------------------------------

/// What a lookup found, before it is turned into a response.
enum Resolved {
    Row(Box<SessionRow>),
    NotFound,
    Ambiguous,
}

/// Resolve an id or unique prefix to its row, fetched once — handlers must
/// not re-query. Session ids are transcript file stems, so any file-name
/// charset can appear (Copilot's are not UUIDs); the check only rules out
/// strings that could never be one — the id is used solely as a SQL
/// parameter, never as a path.
async fn resolve_session(server: &EkkoServer, id: &str) -> Result<SessionRow, ApiError> {
    let plausible = !id.is_empty()
        && id.len() <= 128
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    if !plausible {
        return Err(ApiError::bad_request("invalid session id"));
    }
    let db = server.db().clone();
    let id_owned = id.to_string();
    let resolved = blocking(move || {
        if let Some(row) = db.get_session(&id_owned)? {
            return Ok(Resolved::Row(Box::new(row)));
        }
        Ok(
            match db.find_session_ids_by_prefix(&id_owned, 2)?.as_slice() {
                [only] => match db.get_session(only)? {
                    Some(row) => Resolved::Row(Box::new(row)),
                    None => Resolved::NotFound,
                },
                [] => Resolved::NotFound,
                _ => Resolved::Ambiguous,
            },
        )
    })
    .await?;
    match resolved {
        Resolved::Row(row) => Ok(*row),
        Resolved::NotFound => Err(ApiError::not_found(format!("session not found: {id}"))),
        Resolved::Ambiguous => Err(ApiError::bad_request(format!(
            "ambiguous session prefix: {id}"
        ))),
    }
}

async fn api_session(
    State(server): State<EkkoServer>,
    Path(id): Path<String>,
) -> Result<axum::Json<Value>, ApiError> {
    let row = resolve_session(&server, &id).await?;
    let memory = server.memory().clone();
    let db = server.db().clone();
    let row_for_task = row.clone();
    let (snapshot, transcript_available) = blocking(move || {
        let snapshot = memory.snapshot_payload(&row_for_task)?;
        let available = db.has_events(&row_for_task.session_id)?
            || row_for_task
                .transcript_path
                .as_deref()
                .is_some_and(|p| std::path::Path::new(p).exists());
        Ok((snapshot, available))
    })
    .await?;

    // The MCP snapshot payload (fence-stripped snapshot, metadata, files)
    // plus the row fields the tool leaves out because a model wouldn't
    // want them and a person does.
    let mut out = snapshot;
    if let Some(map) = out.as_object_mut() {
        map.insert("summary".into(), json!(row.summary));
        map.insert(
            "short_desc".into(),
            json!(row.summary.as_deref().map(short_desc)),
        );
        map.insert("first_prompt".into(), json!(row.first_prompt));
        map.insert("keywords".into(), json!(row.keywords));
        map.insert("tags".into(), json!(row.tags));
        map.insert("model".into(), json!(row.model));
        map.insert("duration_ms".into(), json!(row.duration_ms));
        map.insert("token_count".into(), json!(row.token_count));
        map.insert("event_count".into(), json!(row.event_count));
        map.insert("files_touched_count".into(), json!(row.files_touched_count));
        map.insert("summary_at".into(), json!(row.summary_at));
        map.insert("snapshot_at".into(), json!(row.snapshot_at));
        map.insert("indexed_at".into(), json!(row.indexed_at));
        // The transcript is viewable when it has been ingested (the DB is
        // the source of truth) or the file is still on disk to ingest from.
        map.insert("transcript_exists".into(), json!(transcript_available));
        // The tool's `suggestion` tells a language model to call
        // read_transcript(); a person needs different directions.
        if map.get("suggestion").is_some_and(|v| !v.is_null()) {
            map.insert(
                "suggestion".into(),
                json!(
                    "No snapshot yet — the MCP server's background indexer \
generates these for sessions over an hour old. The Transcript tab has the \
full detail meanwhile."
                ),
            );
        }
    }
    Ok(axum::Json(out))
}

/// Transcript as structured events. The database is the source of truth:
/// events are capped once at ingest (`db/events.rs`) and served verbatim
/// here, so a session stays browsable after its `.jsonl` is cleaned up.
/// A session the indexer hasn't reached yet (they get a one-hour dormancy
/// window) is ingested on first view — parse, store capped, then serve the
/// stored rows: there is exactly one rendering path.
async fn api_transcript(
    State(server): State<EkkoServer>,
    Path(id): Path<String>,
) -> Result<axum::Json<Value>, ApiError> {
    let row = resolve_session(&server, &id).await?;
    let db = server.db().clone();
    let session_id = row.session_id.clone();
    let transcript_path = row.transcript_path.clone().unwrap_or_default();
    let cli_source = crate::snapshot::effective_cli_source(&row.cli_source).to_string();

    let events = blocking(move || {
        if !db.has_events(&session_id)? {
            // Not ingested yet — do it now if the file is still there.
            let on_disk =
                !transcript_path.is_empty() && std::path::Path::new(&transcript_path).exists();
            if !on_disk {
                return Ok(None);
            }
            let Some(parser) = parsers::parser_for(&cli_source) else {
                return Ok(None);
            };
            let parsed = parser.parse_transcript(std::path::Path::new(&transcript_path));
            db.store_events(&session_id, &parsed.events)?;
        }
        db.get_events(&session_id).map(Some)
    })
    .await?;
    let Some(events) = events else {
        return Err(ApiError::not_found(
            "transcript is not in the archive and the file is no longer on disk",
        ));
    };

    Ok(axum::Json(json!({
        "session_id": row.session_id,
        "cli_source": row.cli_source,
        "event_count": events.len(),
        "events": events,
    })))
}

// ---------------------------------------------------------------------------
// /api/search, /api/ask — pass-through to the MCP-shaped service calls
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct SearchParams {
    q: String,
    mode: Option<String>,
    directory: Option<String>,
    days: Option<i64>,
    limit: Option<i64>,
}

/// Same result shape as the `search_sessions` tool. `semantic`/`hybrid`
/// need the query backend; without one `semantic` is refused and `hybrid`
/// degrades to the non-LLM tiers, exactly as the tool does.
async fn api_search(
    State(server): State<EkkoServer>,
    Query(p): Query<SearchParams>,
) -> Result<axum::Json<Value>, ApiError> {
    let mode = p.mode.unwrap_or_else(|| "keyword".to_string());
    if !matches!(
        mode.as_str(),
        "keyword" | "transcript" | "semantic" | "hybrid"
    ) {
        return Err(ApiError::bad_request(format!(
            "unknown search mode: {mode}"
        )));
    }
    let q = p.q.trim().to_string();
    if q.is_empty() {
        return Err(ApiError::bad_request("q is required"));
    }
    let effective_mode = llm::gate_search_mode(server.config(), &mode).map_err(|msg| {
        ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("Semantic search requires an LLM backend. {msg}"),
        )
    })?;
    let args = SearchArgs {
        query: &q,
        directory: p.directory.as_deref(),
        include_subdirs: true,
        days: p.days,
        start_date: None,
        end_date: None,
        mode: &effective_mode,
        limit: p.limit.unwrap_or(20).clamp(1, 100),
    };
    let mut value = server.memory().search_sessions(&args).await?;
    if let Some(map) = value.as_object_mut() {
        map.insert("requested_mode".into(), json!(mode));
        map.insert("mode".into(), json!(effective_mode));
    }
    Ok(axum::Json(value))
}

#[derive(Debug, Deserialize)]
pub struct AskBody {
    question: String,
    directory: Option<String>,
    days: Option<i64>,
    max_sessions: Option<usize>,
}

/// Same result shape as the `ask_question` tool. 503 when no query backend
/// is configured — never a success-shaped fallback (PORT_PLAN §5).
async fn api_ask(
    State(server): State<EkkoServer>,
    body: axum::body::Bytes,
) -> Result<axum::Json<Value>, ApiError> {
    // Parsed by hand so a malformed body gets the same {"error": …} shape
    // as every other failure (axum's JsonRejection is plain text).
    let body: AskBody = serde_json::from_slice(&body)
        .map_err(|e| ApiError::bad_request(format!("invalid request body: {e}")))?;
    let question = body.question.trim().to_string();
    if question.is_empty() {
        return Err(ApiError::bad_request("question is required"));
    }
    let (available, msg) = llm::check_backend_available(server.config(), Role::Query);
    if !available {
        return Err(ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("ask requires an LLM backend. {msg}"),
        ));
    }
    let mut args = AskArgs::new(&question);
    args.directory = body.directory.as_deref().filter(|d| !d.is_empty());
    args.days = body.days;
    args.max_sessions = body
        .max_sessions
        .unwrap_or(server.config().max_sessions)
        .clamp(1, 20);
    let value = server.memory().ask_question(&args).await?;
    Ok(axum::Json(value))
}

#[cfg(test)]
mod tests {
    use super::short_desc;

    #[test]
    fn short_desc_strips_outcome_label_and_caps() {
        assert_eq!(
            short_desc("Outcome: did the thing\n\nGoal: x"),
            "did the thing"
        );
        assert_eq!(
            short_desc("\n\n  plain first line  \nmore"),
            "plain first line"
        );
        let long = "x".repeat(300);
        assert_eq!(short_desc(&long).chars().count(), 201);
    }

    #[test]
    fn caps_count_chars_not_bytes() {
        // 300 four-byte scalars must not split mid-char or count as 1200.
        let s = "𝄞".repeat(300);
        let capped = short_desc(&s);
        assert_eq!(capped.chars().count(), 201);
        assert!(capped.ends_with('…'));
    }
}
