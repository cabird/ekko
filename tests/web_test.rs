//! Web route tests through `tower::ServiceExt::oneshot` — no sockets.
//!
//! The service layer under the routes reads the process-wide config
//! (`config::get_config()`), so like `mcp_test.rs` this binary pins `HOME`
//! into a temp dir and scrubs every API key / backend setting before
//! anything touches the singleton. Without that, `/api/ask` and semantic
//! search would reach the developer's real backend (and it did, once).
//! The `/api/ask` and semantic-search tests assert the 503 path.

use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::Value;
use tempfile::TempDir;
use tower::ServiceExt;

use ekko::config::Config;
use ekko::db::Database;
use ekko::mcp::EkkoServer;
use ekko::models::SessionMetadata;
use ekko::web::{DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, router};

/// Process-wide fake home, created once before any test touches the
/// environment or the config singleton.
fn sandbox() -> &'static Path {
    static SANDBOX: OnceLock<TempDir> = OnceLock::new();
    SANDBOX
        .get_or_init(|| {
            let dir = TempDir::new().expect("tempdir");
            fs::create_dir_all(dir.path().join("share/ekko")).unwrap();
            // SAFETY: every test in this binary blocks on this `OnceLock`
            // before touching the environment or the config singleton.
            unsafe {
                std::env::set_var("HOME", dir.path());
                std::env::set_var("XDG_DATA_HOME", dir.path().join("share"));
                std::env::remove_var("APPDATA");
                std::env::set_var("WIN_HOME", "/mnt/c/Users/ekko-test-no-such-user");
                // `CODEX_HOME` would otherwise point `scan_all` at a real Codex home
                // (launchers such as Orca set it) from inside the sandbox.
                std::env::remove_var("CODEX_HOME");
                std::env::remove_var("OPENAI_API_KEY");
                std::env::remove_var("ANTHROPIC_API_KEY");
                std::env::remove_var("AZURE_OPENAI_API_KEY");
                std::env::remove_var("AZURE_OPENAI_ENDPOINT");
                std::env::set_var("EKKO_BACKEND", "api");
                std::env::set_var("EKKO_QUERY_BACKEND", "api");
                std::env::set_var("EKKO_MODEL", "gpt-5-mini");
                std::env::set_var("EKKO_QUERY_MODEL", "gpt-5-mini");
            }
            dir
        })
        .path()
}

const TRANSCRIPT: &str = concat!(
    r#"{"type":"system","subtype":"init","sessionId":"aaaa1111","cwd":"/home/user/webapp","gitBranch":"main","timestamp":"2026-01-05T10:00:00Z"}"#,
    "\n",
    r#"{"type":"user","timestamp":"2026-01-05T10:00:05Z","message":{"role":"user","content":"Why is the auth redirect looping? <script>alert(1)</script>"}}"#,
    "\n",
    r#"{"type":"assistant","timestamp":"2026-01-05T10:00:09Z","message":{"role":"assistant","content":[{"type":"text","text":"Reading the auth module."},{"type":"tool_use","id":"tc_1","name":"Read","input":{"file_path":"/home/user/webapp/auth.py"}}]}}"#,
    "\n",
    r#"{"type":"assistant","timestamp":"2026-01-05T10:01:20Z","message":{"role":"assistant","content":[{"type":"text","text":"Fixed: the session cookie was never set on the redirect response."}]}}"#,
    "\n",
);

/// A keyless configuration — API backend pinned so a locally installed
/// `claude` is never auto-detected — leaked for the server's `'static`
/// bound. Must agree with the env pins above: the routes' gates read this
/// config, the service layer underneath reads the singleton.
fn config() -> &'static Config {
    static CONFIG: OnceLock<&'static Config> = OnceLock::new();
    CONFIG.get_or_init(|| {
        sandbox();
        Box::leak(Box::new(Config {
            llm_backend: Some("api".to_string()),
            query_backend: Some("api".to_string()),
            llm_model: "gpt-5-mini".to_string(),
            llm_model_explicit: true,
            query_model: Some("gpt-5-mini".to_string()),
            ..Config::default()
        }))
    })
}

struct Fixture {
    app: Router,
    _dir: TempDir,
}

fn fixture() -> Fixture {
    let config = config();
    let (available, msg) =
        ekko::llm::check_backend_available(ekko::config::get_config(), ekko::llm::Role::Query);
    assert!(
        !available,
        "test environment leaked an LLM backend ({msg}) — refusing to run routes that could call one"
    );
    let dir = TempDir::new().unwrap();
    let transcript = dir.path().join("aaaa1111.jsonl");
    std::fs::write(&transcript, TRANSCRIPT).unwrap();
    let db = Database::new(dir.path().join("sessions.db")).unwrap();

    let mut s1 = SessionMetadata::new("aaaa1111", "claude_code", transcript.to_string_lossy());
    s1.title = Some("Fix auth redirect loop".to_string());
    s1.first_prompt = Some("Why is the auth redirect looping?".to_string());
    s1.project_path = Some("/home/user/webapp".to_string());
    s1.git_branch = Some("main".to_string());
    s1.modified_at = Some(chrono::Utc::now().fixed_offset());
    s1.message_count = 3;
    db.upsert_session(&s1).unwrap();
    db.update_summary(
        "aaaa1111",
        "Fix auth redirect loop",
        "Outcome: Set the session cookie on redirects.\n\nGoal: stop the loop",
    )
    .unwrap();
    db.add_file_reference("aaaa1111", "/home/user/webapp/auth.py", "write")
        .unwrap();

    let mut s2 = SessionMetadata::new("aaaa2222", "github_copilot", "");
    s2.title = Some("Refactor DB".to_string());
    s2.project_path = Some("/home/user/backend".to_string());
    s2.modified_at = Some((chrono::Utc::now() - chrono::Duration::days(40)).fixed_offset());
    s2.message_count = 12;
    db.upsert_session(&s2).unwrap();

    let mut empty = SessionMetadata::new("bbbb0000", "claude_code", "");
    empty.project_path = Some("/home/user/webapp".to_string());
    empty.message_count = 0;
    db.upsert_session(&empty).unwrap();

    Fixture {
        app: router(EkkoServer::new(db, config)),
        _dir: dir,
    }
}

async fn get(app: &Router, uri: &str) -> (StatusCode, Value, String) {
    let resp = app
        .clone()
        .oneshot(Request::get(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = resp.status();
    let ctype = resp
        .headers()
        .get("content-type")
        .map(|v| v.to_str().unwrap_or("").to_string())
        .unwrap_or_default();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes).into_owned();
    let json = serde_json::from_str(&text).unwrap_or(Value::String(text));
    (status, json, ctype)
}

async fn post_json(app: &Router, uri: &str, body: &str) -> (StatusCode, Value) {
    let resp = app
        .clone()
        .oneshot(
            Request::post(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

#[tokio::test]
async fn static_assets_are_served_with_types() {
    let f = fixture();
    let (s, body, ct) = get(&f.app, "/").await;
    assert_eq!(s, StatusCode::OK);
    assert!(ct.starts_with("text/html"), "{ct}");
    assert!(body.as_str().unwrap().contains("<title>ekko</title>"));
    let (s, _, ct) = get(&f.app, "/style.css").await;
    assert_eq!((s, ct.starts_with("text/css")), (StatusCode::OK, true));
    let (s, _, ct) = get(&f.app, "/app.js").await;
    assert_eq!(
        (s, ct.starts_with("application/javascript")),
        (StatusCode::OK, true)
    );
    let (s, _, _) = get(&f.app, "/nope").await;
    assert_eq!(s, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn sessions_list_pages_filters_and_counts() {
    let f = fixture();
    let (s, v, _) = get(&f.app, "/api/sessions").await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["total"], 2, "empty session excluded: {v}");
    assert_eq!(v["sessions"][0]["session_id"], "aaaa1111", "newest first");
    assert_eq!(
        v["sessions"][0]["short_desc"],
        "Set the session cookie on redirects."
    );
    assert_eq!(v["sessions"][0]["has_summary"], true);
    assert_eq!(v["sessions"][0]["has_snapshot"], false);
    assert!(
        v["sessions"][0].get("summary").is_none(),
        "list must not over-fetch"
    );
    assert_eq!(v["source_counts"]["claude_code"], 1);
    assert_eq!(v["limit"], DEFAULT_PAGE_SIZE);

    let (_, v, _) = get(&f.app, "/api/sessions?limit=1&offset=1").await;
    assert_eq!(
        (v["total"].as_i64(), v["sessions"].as_array().unwrap().len()),
        (Some(2), 1)
    );
    assert_eq!(v["sessions"][0]["session_id"], "aaaa2222");

    let (_, v, _) = get(&f.app, "/api/sessions?limit=99999").await;
    assert_eq!(v["limit"], MAX_PAGE_SIZE);
    let (_, v, _) = get(&f.app, "/api/sessions?limit=0").await;
    assert_eq!(v["limit"], 1, "limit clamps up to 1");

    // Sort param is whitelisted and echoed; messages puts the big session first.
    let (_, v, _) = get(&f.app, "/api/sessions?sort=messages").await;
    assert_eq!(v["sort"], "messages");
    assert_eq!(v["sessions"][0]["session_id"], "aaaa2222", "{v}");
    let (_, v, _) = get(&f.app, "/api/sessions?sort=drop%20table").await;
    assert_eq!(v["sort"], "modified", "unknown sorts fall back");

    // Cursor pagination: follow next_cursor across both rows; the last
    // page carries no cursor, and a malformed one falls back to page one.
    let (_, v, _) = get(&f.app, "/api/sessions?limit=1").await;
    let tok = v["next_cursor"].as_str().expect("cursor on page 1").to_string();
    assert_eq!(v["sessions"][0]["session_id"], "aaaa1111");
    let (_, v, _) = get(&f.app, &format!("/api/sessions?limit=1&cursor={tok}")).await;
    assert_eq!(v["sessions"][0]["session_id"], "aaaa2222");
    assert!(v["next_cursor"].is_null(), "no dangling cursor: {v}");
    let (s, v, _) = get(&f.app, "/api/sessions?limit=1&cursor=nonsense").await;
    assert_eq!((s, v["sessions"][0]["session_id"].as_str()), (StatusCode::OK, Some("aaaa1111")));

    // Facet counts follow the active filter (minus the source facet itself):
    // scoped to the webapp project, the copilot session disappears from them.
    let (_, v, _) = get(&f.app, "/api/sessions?project=/home/user/webapp").await;
    assert_eq!(v["source_counts"]["claude_code"], 1);
    assert!(v["source_counts"].get("github_copilot").is_none(), "{v}");
    // …and picking a source keeps the other sources' counts visible.
    let (_, v, _) = get(&f.app, "/api/sessions?source=claude_code").await;
    assert_eq!(v["source_counts"]["github_copilot"], 1);

    let (_, v, _) = get(&f.app, "/api/sessions?q=redirect").await;
    assert_eq!(v["total"], 1);
    // Multi-word filter: AND of terms anywhere in the row (title has
    // "redirect", summary has "cookie") — but a quoted phrase must be
    // contiguous.
    let (_, v, _) = get(&f.app, "/api/sessions?q=cookie%20redirect").await;
    assert_eq!(v["total"], 1, "{v}");
    let (_, v, _) = get(&f.app, "/api/sessions?q=%22cookie%20redirect%22").await;
    assert_eq!(v["total"], 0, "phrase is contiguous: {v}");
    let (_, v, _) = get(&f.app, "/api/sessions?q=redirect%20nothere").await;
    assert_eq!(v["total"], 0);
    let (_, v, _) = get(&f.app, "/api/sessions?project=/home/user/backend").await;
    assert_eq!(v["sessions"][0]["session_id"], "aaaa2222");
    let (_, v, _) = get(&f.app, "/api/sessions?source=github_copilot").await;
    assert_eq!(v["total"], 1);
    let (_, v, _) = get(&f.app, "/api/sessions?days=7").await;
    assert_eq!(v["total"], 1, "40-day-old session drops out");
}

#[tokio::test]
async fn projects_merge_sources_per_path() {
    let f = fixture();
    let (s, v, _) = get(&f.app, "/api/projects").await;
    assert_eq!(s, StatusCode::OK);
    let projects = v["projects"].as_array().unwrap();
    let webapp = projects
        .iter()
        .find(|p| p["path"] == "/home/user/webapp")
        .expect("webapp project");
    assert_eq!(webapp["name"], "webapp");
    // get_projects counts the empty session too; that's the MCP tool's
    // behaviour and the facet count follows it.
    assert_eq!(webapp["count"], 2);
}

#[tokio::test]
async fn session_detail_and_prefix_resolution() {
    let f = fixture();
    let (s, v, _) = get(&f.app, "/api/sessions/aaaa1111").await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["metadata"]["title"], "Fix auth redirect loop");
    assert_eq!(
        v["summary"],
        "Outcome: Set the session cookie on redirects.\n\nGoal: stop the loop"
    );
    assert_eq!(v["has_snapshot"], false);
    assert_eq!(v["files_touched"][0]["path"], "/home/user/webapp/auth.py");
    assert_eq!(v["transcript_exists"], true);

    // Unique prefix resolves; ambiguous prefix is a 400; unknown is a 404.
    // Ids are file stems, so any [A-Za-z0-9._-] string is plausible
    // (Copilot session dirs are not UUIDs); anything else is rejected
    // before the database is consulted.
    let (s, v, _) = get(&f.app, "/api/sessions/aaaa1").await;
    assert_eq!(
        (s, v["session_id"].as_str()),
        (StatusCode::OK, Some("aaaa1111"))
    );
    let (s, v, _) = get(&f.app, "/api/sessions/aaaa").await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
    assert!(v["error"].as_str().unwrap().contains("ambiguous"));
    let (s, _, _) = get(&f.app, "/api/sessions/ffff").await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    let (s, _, _) = get(&f.app, "/api/sessions/session-state-2.v1").await;
    assert_eq!(s, StatusCode::NOT_FOUND, "file-stem ids are plausible");
    let (s, _, _) = get(&f.app, "/api/sessions/..%2F..%2Fetc").await;
    assert_ne!(s, StatusCode::OK);
    let (s, _, _) = get(&f.app, "/api/sessions/a%20b").await;
    assert_eq!(
        s,
        StatusCode::BAD_REQUEST,
        "spaces are never in a file-stem id"
    );

    // The web layer replaces the MCP tool's model-facing suggestion.
    let (_, v, _) = get(&f.app, "/api/sessions/aaaa1111").await;
    assert!(
        !v["suggestion"]
            .as_str()
            .unwrap_or("")
            .contains("read_transcript()"),
        "{v}"
    );
}

#[tokio::test]
async fn transcript_returns_structured_events() {
    let f = fixture();
    let (s, v, _) = get(&f.app, "/api/sessions/aaaa1111/transcript").await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let events = v["events"].as_array().unwrap();
    assert!(events.len() >= 4, "{v}");
    let first = &events[0];
    assert_eq!(first["type"], "message");
    assert_eq!(first["role"], "user");
    // Raw content is returned as-is (JSON-encoded); escaping is the
    // frontend's job and it must never be done twice.
    assert!(first["content"].as_str().unwrap().contains("<script>"));
    let call = events
        .iter()
        .find(|e| e["type"] == "tool_call")
        .expect("tool call");
    assert_eq!(call["tool_name"], "Read");
    assert_eq!(call["tool_input"]["file_path"], "/home/user/webapp/auth.py");
    assert_eq!(call["truncated"], false);

    // No transcript on disk → 404, not a 500.
    let (s, v, _) = get(&f.app, "/api/sessions/aaaa2222/transcript").await;
    assert_eq!(s, StatusCode::NOT_FOUND, "{v}");

    // The DB is the source of truth: after the first view ingested the
    // events, deleting the .jsonl must not take the transcript away.
    std::fs::remove_file(f._dir.path().join("aaaa1111.jsonl")).unwrap();
    let (s, v, _) = get(&f.app, "/api/sessions/aaaa1111/transcript").await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert!(v["events"].as_array().unwrap().len() >= 4);
    let (_, v, _) = get(&f.app, "/api/sessions/aaaa1111").await;
    assert_eq!(v["transcript_exists"], true, "served from the archive: {v}");

    // Prefix rules hold on this route too.
    let (s, _, _) = get(&f.app, "/api/sessions/aaaa/transcript").await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    let (s, _, _) = get(&f.app, "/api/sessions/ffff/transcript").await;
    assert_eq!(s, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn search_passes_through_and_gates_llm_modes() {
    let f = fixture();
    let (s, v, _) = get(&f.app, "/api/search?q=redirect").await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["mode"], "keyword");
    assert_eq!(v["results"][0]["session_id"], "aaaa1111");

    let (s, v, _) = get(&f.app, "/api/search?q=redirect&mode=semantic").await;
    assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{v}");
    let (s, v, _) = get(&f.app, "/api/search?q=redirect&mode=hybrid").await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(
        (v["requested_mode"].as_str(), v["mode"].as_str()),
        (Some("hybrid"), Some("keyword"))
    );
    let (s, _, _) = get(&f.app, "/api/search?q=redirect&mode=bogus").await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    let (s, _, _) = get(&f.app, "/api/search?q=%20").await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn ask_is_503_without_a_backend_and_400_without_a_question() {
    let f = fixture();
    let (s, v) = post_json(&f.app, "/api/ask", r#"{"question":"why?"}"#).await;
    assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{v}");
    assert!(v["error"].as_str().unwrap().contains("LLM backend"));
    let (s, _) = post_json(&f.app, "/api/ask", r#"{"question":"  "}"#).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn unknown_routes_and_bad_bodies_use_the_json_error_shape() {
    let f = fixture();
    let (s, v, _) = get(&f.app, "/api/nope").await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    assert!(v["error"].is_string(), "{v}");
    let (s, v) = post_json(&f.app, "/api/ask", "{not json").await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(
        v["error"]
            .as_str()
            .unwrap()
            .contains("invalid request body"),
        "{v}"
    );
}

#[tokio::test]
async fn asset_responses_ask_the_browser_to_revalidate() {
    let f = fixture();
    for uri in ["/", "/style.css", "/app.js"] {
        let resp = f
            .app
            .clone()
            .oneshot(Request::get(uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(
            resp.headers()
                .get("cache-control")
                .and_then(|v| v.to_str().ok()),
            Some("no-cache"),
            "{uri}"
        );
    }
}

/// Minimal raw-HTTP GET over std TcpStream (no client dependency): returns
/// the status line of the response, or None if nothing is listening.
fn raw_get(port: u16, path: &str) -> Option<String> {
    use std::io::{Read, Write};
    let mut sock =
        std::net::TcpStream::connect(("127.0.0.1", port)).ok()?;
    sock.set_read_timeout(Some(std::time::Duration::from_secs(2))).ok()?;
    write!(sock, "GET {path} HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\n\r\n").ok()?;
    let mut buf = String::new();
    let _ = sock.read_to_string(&mut buf);
    buf.lines().next().map(str::to_string)
}

#[tokio::test]
async fn cohosted_serving_holds_and_releases_the_lease() {
    use ekko::db::leases::WEB_LEASE;
    let config = config();
    let dir = TempDir::new().unwrap();
    let db = Database::new(dir.path().join("sessions.db")).unwrap();
    let server = EkkoServer::new(db.clone(), config);

    // Find a free port the OS way, then hand it to the election loop.
    let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = probe.local_addr().unwrap().port();
    drop(probe);

    let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
    let task = tokio::spawn(ekko::web::server::run_cohosted(server, port, stop_rx));

    // It should bind, take the lease, and answer HTTP within a few seconds.
    let mut status_line = None;
    for _ in 0..50 {
        status_line = tokio::task::spawn_blocking(move || raw_get(port, "/api/status"))
            .await
            .unwrap();
        if status_line.is_some() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    let status_line = status_line.expect("co-hosted server never answered");
    assert!(status_line.contains("200"), "{status_line}");

    let (holder, remaining) = db.lease_info(WEB_LEASE).unwrap().expect("lease held while serving");
    assert!(holder.ends_with(&format!(":{port}")), "{holder}");
    assert!(remaining > 0, "{remaining}");

    // Shutdown: server stops and the lease is released, not left to expire.
    stop_tx.send(true).unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(5), task)
        .await
        .expect("election loop did not stop")
        .unwrap();
    assert!(db.lease_info(WEB_LEASE).unwrap().is_none(), "lease must be released on shutdown");
    let gone = tokio::task::spawn_blocking(move || raw_get(port, "/")).await.unwrap();
    assert!(gone.is_none(), "listener still up after shutdown: {gone:?}");
}

#[tokio::test]
async fn status_carries_both_roles() {
    let f = fixture();
    let (s, v, _) = get(&f.app, "/api/status").await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["sessions"]["total"], 2);
    assert_eq!(v["config"]["index"]["backend"], "api");
    assert_eq!(v["config"]["query"]["backend"], "api");
    assert_eq!(v["web"]["ui_version"], env!("CARGO_PKG_VERSION"));
    // Served through oneshot, not a bound listener: no lease is held.
    assert_eq!(v["web"]["status"], "not_running", "{v}");
}
