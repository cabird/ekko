//! Tests for the LLM layer.
//!
//! The expected values for the JSON-recovery trio, the provider table and the
//! user-facing messages are pinned; update them deliberately when behavior
//! changes intentionally.
//!
//! HTTP behaviour (retry, JSON re-ask, request shape) is exercised against a
//! tiny canned-response server built on `std::net::TcpListener` — no extra
//! dev-dependencies.

use super::*;

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;

use crate::paths::testenv::{EnvVarGuard, env_lock};

// ---------------------------------------------------------------------------
// Mock HTTP server
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct RecordedRequest {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: String,
}

impl RecordedRequest {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    fn json(&self) -> Value {
        serde_json::from_str(&self.body).expect("request body is JSON")
    }
}

struct MockServer {
    addr: SocketAddr,
    requests: Arc<Mutex<Vec<RecordedRequest>>>,
}

impl MockServer {
    /// Serve exactly `responses.len()` requests, in order, then stop.
    fn start(responses: Vec<(u16, String)>) -> MockServer {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&requests);

        thread::spawn(move || {
            for (status, body) in responses {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                if let Some(req) = read_request(&mut stream) {
                    sink.lock().expect("mock lock").push(req);
                }
                let response = format!(
                    "HTTP/1.1 {status} STATUS\r\nContent-Type: application/json\r\n\
Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });

        MockServer { addr, requests }
    }

    fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    fn requests(&self) -> Vec<RecordedRequest> {
        self.requests.lock().expect("mock lock").clone()
    }
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn read_request(stream: &mut TcpStream) -> Option<RecordedRequest> {
    stream
        // Generous: the suite may run while the machine is busy compiling.
        .set_read_timeout(Some(Duration::from_secs(30)))
        .ok()?;
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 4096];

    let header_end = loop {
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            break pos;
        }
        let n = stream.read(&mut chunk).ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
    };

    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut lines = head.split("\r\n");
    let request_line = lines.next()?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();

    let mut headers = Vec::new();
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            headers.push((k.trim().to_string(), v.trim().to_string()));
        }
    }

    let content_length: usize = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(0);

    let body_start = header_end + 4;
    while buf.len() < body_start + content_length {
        let n = stream.read(&mut chunk).ok()?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let end = (body_start + content_length).min(buf.len());
    let body = String::from_utf8_lossy(&buf[body_start..end]).to_string();

    Some(RecordedRequest {
        method,
        path,
        headers,
        body,
    })
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn test_config(model: &str) -> Config {
    Config {
        llm_model: model.to_string(),
        api_keys: vec![
            ("OPENAI_API_KEY".to_string(), "sk-openai-test".to_string()),
            ("ANTHROPIC_API_KEY".to_string(), "sk-ant-test".to_string()),
        ],
        ..Config::default()
    }
}

fn client_for(server: &MockServer, model: &str) -> LlmClient {
    LlmClient::new(&test_config(model), model)
        .expect("client builds")
        .with_base_url(&server.base_url())
        // Keep retry waits invisible in tests; the schedule itself is
        // unit-tested separately via `backoff_delay`.
        .with_backoff_base(Duration::from_millis(1))
}

fn openai_body(content: &str) -> String {
    json!({
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 7}
    })
    .to_string()
}

// ---------------------------------------------------------------------------
// Provider routing
// ---------------------------------------------------------------------------

#[test]
fn provider_routing_startswith() {
    assert_eq!(get_required_provider("gpt-5-mini"), Some(Provider::OpenAi));
    assert_eq!(get_required_provider("gpt-4o"), Some(Provider::OpenAi));
    assert_eq!(get_required_provider("o1-preview"), Some(Provider::OpenAi));
    assert_eq!(get_required_provider("o3-mini"), Some(Provider::OpenAi));
    assert_eq!(get_required_provider("o4-mini"), Some(Provider::OpenAi));
    assert_eq!(
        get_required_provider("claude-opus-4-6"),
        Some(Provider::Anthropic)
    );
}

#[test]
fn provider_routing_substring_fallback() {
    // Second pass: a prefix appearing anywhere in the name.
    assert_eq!(
        get_required_provider("my-gpt-model"),
        Some(Provider::OpenAi)
    );
    assert_eq!(
        get_required_provider("openai/gpt-5-mini"),
        Some(Provider::OpenAi)
    );
    assert_eq!(
        get_required_provider("anthropic/claude-sonnet-4-5"),
        Some(Provider::Anthropic)
    );
    // There is no `azure/` row, so the substring pass finds "gpt": Azure is a
    // routing option on the OpenAI provider, not a provider of its own.
    assert_eq!(get_required_provider("azure/gpt-4"), Some(Provider::OpenAi));
}

#[test]
fn provider_routing_case_insensitive() {
    assert_eq!(get_required_provider("GPT-5-Mini"), Some(Provider::OpenAi));
    assert_eq!(
        get_required_provider("CLAUDE-OPUS-4.6"),
        Some(Provider::Anthropic)
    );
    assert_eq!(
        get_required_provider("My-GPT-Model"),
        Some(Provider::OpenAi)
    );
}

#[test]
fn provider_routing_unknown() {
    assert_eq!(get_required_provider("llama-3-70b"), None);
    assert_eq!(get_required_provider(""), None);
    // Gemini has no route: ekko cannot call it.
    assert_eq!(get_required_provider("gemini-3-pro"), None);
}

#[test]
fn provider_metadata() {
    assert_eq!(Provider::OpenAi.name(), "openai");
    assert_eq!(Provider::OpenAi.env_var(), "OPENAI_API_KEY");
    assert_eq!(Provider::Anthropic.name(), "anthropic");
    assert_eq!(Provider::Anthropic.env_var(), "ANTHROPIC_API_KEY");
}

// ---------------------------------------------------------------------------
// Availability checks
// ---------------------------------------------------------------------------

#[test]
fn check_llm_available_messages_are_pinned() {
    let _lock = env_lock();
    let _openai = EnvVarGuard::unset("OPENAI_API_KEY");
    let _anthropic = EnvVarGuard::unset("ANTHROPIC_API_KEY");

    let empty = Config::default();
    assert_eq!(
        check_llm_available(&empty, "gpt-5-mini"),
        (
            false,
            "Model 'gpt-5-mini' requires OPENAI_API_KEY to be set. Set this environment \
variable or add it to your .env file to enable LLM-powered features (summaries, \
snapshots, semantic search, Q&A)."
                .to_string()
        )
    );
    assert_eq!(
        check_llm_available(&empty, "llama-3"),
        (
            true,
            "Model 'llama-3' has unknown provider; will attempt call".to_string()
        )
    );

    let configured = test_config("gpt-5-mini");
    assert_eq!(
        check_llm_available(&configured, "gpt-5-mini"),
        (true, "API key found for openai".to_string())
    );
    assert_eq!(
        check_llm_available(&configured, "claude-opus-4-6"),
        (true, "API key found for anthropic".to_string())
    );
}

#[test]
fn require_llm_and_error_message() {
    let _lock = env_lock();
    let _openai = EnvVarGuard::unset("OPENAI_API_KEY");
    let _anthropic = EnvVarGuard::unset("ANTHROPIC_API_KEY");

    let empty = Config::default();
    let err = require_llm(&empty, "gpt-5-mini").expect_err("no key configured");
    assert_eq!(
        err.to_string(),
        "LLM not configured: model 'gpt-5-mini' requires OPENAI_API_KEY to be set. \
Add it to config.json in your data directory, or set the environment variable to \
enable LLM-powered features."
    );
    assert_eq!(err.provider, "openai");
    assert_eq!(err.env_var, "OPENAI_API_KEY");

    // An unrecognised provider is not an error here — only a missing key is.
    assert!(require_llm(&empty, "llama-3").is_ok());
    // Configured key → ok.
    assert!(require_llm(&test_config("gpt-5-mini"), "gpt-5-mini").is_ok());
}

#[test]
fn client_construction_errors() {
    let _lock = env_lock();
    let _openai = EnvVarGuard::unset("OPENAI_API_KEY");
    let _anthropic = EnvVarGuard::unset("ANTHROPIC_API_KEY");

    // Unknown provider: there is no generic routing fallback.
    let err = LlmClient::new(&Config::default(), "llama-3").expect_err("unknown provider");
    assert!(matches!(err, LlmError::UnknownProvider { .. }));
    assert!(err.to_string().contains("no known provider"));

    // Known provider, missing key.
    let err = LlmClient::new(&Config::default(), "gpt-5-mini").expect_err("missing key");
    assert!(matches!(err, LlmError::NotConfigured(_)));
}

#[test]
fn api_key_comes_from_config_when_env_unset() {
    let _lock = env_lock();
    let _openai = EnvVarGuard::unset("OPENAI_API_KEY");
    let client = LlmClient::new(&test_config("gpt-5-mini"), "gpt-5-mini").expect("client builds");
    assert_eq!(client.api_key, "sk-openai-test");
    assert_eq!(client.provider(), Provider::OpenAi);
    assert_eq!(client.model(), "gpt-5-mini");
}

#[test]
fn api_key_env_wins_over_config() {
    let _lock = env_lock();
    let _openai = EnvVarGuard::set("OPENAI_API_KEY", "sk-from-env");
    let client = LlmClient::new(&test_config("gpt-5-mini"), "gpt-5-mini").expect("client builds");
    assert_eq!(client.api_key, "sk-from-env");
}

// ---------------------------------------------------------------------------
// reasoning_effort capability
// ---------------------------------------------------------------------------

#[test]
fn supports_reasoning_effort_classifies_models() {
    for model in [
        "gpt-5",
        "gpt-5-mini",
        "gpt-5.1",
        "o1",
        "o3-mini",
        "o4-mini",
        "O3-MINI",
    ] {
        assert!(supports_reasoning_effort(model), "{model} should support");
    }
    for model in [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
        "openai-thing",
        "claude-opus-4-6",
        "",
    ] {
        assert!(!supports_reasoning_effort(model), "{model} should not");
    }
}

// ---------------------------------------------------------------------------
// Backoff schedule
// ---------------------------------------------------------------------------

#[test]
fn backoff_follows_the_expected_schedule() {
    let client = LlmClient::new(&test_config("gpt-5-mini"), "gpt-5-mini").expect("client builds");
    let transient = LlmError::Timeout;
    // min(2^attempt, 60) seconds.
    let expected = [1u64, 2, 4, 8, 16, 32, 60, 60, 60];
    for (attempt, secs) in expected.iter().enumerate() {
        assert_eq!(
            client.backoff_delay(attempt as u32, &transient),
            Duration::from_secs(*secs),
            "attempt {attempt}"
        );
    }
    // Absurd attempt counts saturate rather than overflow.
    assert_eq!(client.backoff_delay(1000, &transient), MAX_BACKOFF);
}

#[test]
fn backoff_waits_a_minute_on_rate_limit() {
    let client = LlmClient::new(&test_config("gpt-5-mini"), "gpt-5-mini").expect("client builds");
    let rate_limited = LlmError::Http {
        status: Some(429),
        body: "too_many_requests".to_string(),
    };
    // Every attempt waits the full minute — rate windows don't care which
    // retry this is.
    for attempt in [0u32, 1, 2] {
        assert_eq!(
            client.backoff_delay(attempt, &rate_limited),
            Duration::from_secs(60),
            "attempt {attempt}"
        );
    }
    // Other HTTP statuses keep the exponential schedule.
    let server_error = LlmError::Http {
        status: Some(500),
        body: "oops".to_string(),
    };
    assert_eq!(
        client.backoff_delay(0, &server_error),
        Duration::from_secs(1)
    );
    // And the factor scales with the base, so tests that shrink the base
    // shrink the rate-limit wait too.
    let fast = LlmClient::new(&test_config("gpt-5-mini"), "gpt-5-mini")
        .expect("client builds")
        .with_backoff_base(Duration::from_millis(1));
    assert_eq!(
        fast.backoff_delay(0, &rate_limited),
        Duration::from_millis(60)
    );
}

// ---------------------------------------------------------------------------
// JSON recovery trio — pinned byte-for-byte
// ---------------------------------------------------------------------------

#[test]
fn strip_markdown_json_goldens() {
    assert_eq!(strip_markdown_json("{\"a\": 1}"), "{\"a\": 1}");
    assert_eq!(
        strip_markdown_json("```json\n{\"a\": 1}\n```"),
        "{\"a\": 1}"
    );
    assert_eq!(strip_markdown_json("```\n{\"a\":1}\n```"), "{\"a\":1}");
    assert_eq!(strip_markdown_json("```json{\"a\":1}```"), "{\"a\":1}");
    assert_eq!(
        strip_markdown_json("  ```json  {\"a\": 1}  ```  "),
        "{\"a\": 1}"
    );
    assert_eq!(
        strip_markdown_json("```json\n{}\n``` trailing text"),
        "{}\n``` trailing text"
    );
    // The strip uses the extended whitespace set, which also removes
    // U+001C..U+001F where `str::trim` would not — the fence must still be
    // recognised after such a leading control character.
    assert_eq!(
        strip_markdown_json("\u{1c}```json\n{\"a\": 1}\n```"),
        "{\"a\": 1}"
    );
    assert_eq!(
        parse_json_response("\u{1c}```json\n{\"a\": 1}\n```"),
        json!({"a": 1})
    );
    assert_eq!(strip_markdown_json("no fences"), "no fences");
    assert_eq!(strip_markdown_json("```"), "");
    assert_eq!(strip_markdown_json(""), "");
}

#[test]
fn extract_json_object_goldens() {
    assert_eq!(extract_json_object("{\"a\": 1}"), "{\"a\": 1}");
    assert_eq!(
        extract_json_object("Here is: {\"a\": {\"b\": 2}} trailing"),
        "{\"a\": {\"b\": 2}}"
    );
    assert_eq!(
        extract_json_object("prefix {\"a\": {\"b\": [1,2]}} suffix"),
        "{\"a\": {\"b\": [1,2]}}"
    );
    assert_eq!(extract_json_object("{\"a\": 1} extra"), "{\"a\": 1}");
    // No opening brace → text returned unchanged.
    assert_eq!(extract_json_object("no json here"), "no json here");
    assert_eq!(extract_json_object(""), "");
    // Unbalanced → everything from the first brace.
    assert_eq!(extract_json_object("x {\"a\": 1"), "{\"a\": 1");
    // The scanner ignores string context by design, so a `}` inside a JSON
    // string closes the object.
    assert_eq!(
        extract_json_object("text {\"a\": \"}\", \"b\": 1}"),
        "{\"a\": \"}"
    );
    assert_eq!(
        extract_json_object("```json\n[{\"a\": 1}]\n```"),
        "{\"a\": 1}"
    );
    // Multi-byte text before/inside the object must not break slicing.
    assert_eq!(
        extract_json_object("日本語 {\"k\": \"ü\"} 語"),
        "{\"k\": \"ü\"}"
    );
}

#[test]
fn parse_json_response_goldens() {
    assert_eq!(parse_json_response("{\"a\": 1}"), json!({"a": 1}));
    assert_eq!(
        parse_json_response("```json\n{\"a\": 1}\n```"),
        json!({"a": 1})
    );
    assert_eq!(parse_json_response("```\n{\"a\":1}\n```"), json!({"a": 1}));
    assert_eq!(
        parse_json_response("Here is: {\"a\": {\"b\": 2}} trailing"),
        json!({"a": {"b": 2}})
    );
    assert_eq!(
        parse_json_response("prefix {\"a\": {\"b\": [1,2]}} suffix"),
        json!({"a": {"b": [1, 2]}})
    );
    assert_eq!(parse_json_response("{\"a\": 1} extra"), json!({"a": 1}));
    assert_eq!(parse_json_response("[1, 2]"), json!([1, 2]));
    assert_eq!(
        parse_json_response("```json\n[{\"a\": 1}]\n```"),
        json!([{"a": 1}])
    );
    // `null` parses successfully to Value::Null, which counts as "no answer",
    // so the caller re-asks.
    assert_eq!(parse_json_response("null"), Value::Null);
    // Failures all collapse to {}
    assert_eq!(parse_json_response(""), json!({}));
    assert_eq!(parse_json_response("blah"), json!({}));
    assert_eq!(parse_json_response("no json here"), json!({}));
    assert_eq!(parse_json_response("x {\"a\": 1"), json!({}));
    assert_eq!(parse_json_response("```"), json!({}));
    // Brace-in-string quirk: the truncated extraction fails to parse, so the
    // whole cascade falls through to {}.
    assert_eq!(
        parse_json_response("text {\"a\": \"}\", \"b\": 1}"),
        json!({})
    );
    // Trailing text after the fence: strip leaves junk, extract finds `{}`,
    // which parses to an empty object.
    assert_eq!(
        parse_json_response("```json\n{}\n``` trailing text"),
        json!({})
    );
}

#[test]
fn truthiness_follows_container_emptiness() {
    assert!(!is_truthy(&json!({})));
    assert!(!is_truthy(&json!([])));
    assert!(!is_truthy(&Value::Null));
    assert!(!is_truthy(&json!(0)));
    assert!(!is_truthy(&json!(0.0)));
    assert!(!is_truthy(&json!("")));
    assert!(!is_truthy(&json!(false)));
    assert!(is_truthy(&json!({"a": 1})));
    assert!(is_truthy(&json!([1])));
    assert!(is_truthy(&json!(1)));
    assert!(is_truthy(&json!("x")));
    assert!(is_truthy(&json!(true)));
}

// ---------------------------------------------------------------------------
// Usage tracking
// ---------------------------------------------------------------------------

#[test]
fn usage_tracker_subtracts_reasoning_tokens() {
    let tracker = UsageTracker::new();
    let usage = RawUsage {
        prompt_tokens: Some(1000),
        completion_tokens: Some(700),
        completion_tokens_details: Some(CompletionTokensDetails {
            reasoning_tokens: Some(200),
        }),
    };
    let stats = tracker.record("gpt-5-mini", &usage);
    assert_eq!(stats.calls, 1);
    assert_eq!(stats.input_tokens, 1000);
    assert_eq!(stats.output_tokens, 500); // 700 - 200
    assert_eq!(stats.reasoning_tokens, 200);
    // get_cost('gpt-5-mini', 1000, 500, 200) == 0.00165
    assert_eq!(stats.cost, 0.00165);

    // Accumulates.
    tracker.record("gpt-5-mini", &usage);
    let total = tracker.get_stats();
    assert_eq!(total.calls, 2);
    assert_eq!(total.input_tokens, 2000);
    assert_eq!(total.output_tokens, 1000);
    assert_eq!(total.reasoning_tokens, 400);

    tracker.reset();
    assert_eq!(tracker.get_stats(), UsageStats::default());
}

#[test]
fn usage_tracker_handles_missing_and_negative_fields() {
    let tracker = UsageTracker::new();
    // `record` itself always counts a call: it models a *present* usage
    // object whose fields happen to be null. The absent/empty case is gated
    // one level up in `call_llm_with`, see
    // `missing_usage_does_not_count_a_call`.
    let stats = tracker.record("gpt-5-mini", &RawUsage::default());
    assert_eq!(
        stats,
        UsageStats {
            calls: 1,
            input_tokens: 0,
            output_tokens: 0,
            reasoning_tokens: 0,
            cost: 0.0,
        }
    );

    // Every token count is clamped at 0, including `prompt_tokens`: a
    // negative count is meaningless, and the unsigned counters must not
    // underflow.
    let weird = RawUsage {
        prompt_tokens: Some(-5),
        completion_tokens: Some(10),
        completion_tokens_details: Some(CompletionTokensDetails {
            reasoning_tokens: Some(-3),
        }),
    };
    let stats = tracker.record("gpt-5-mini", &weird);
    assert_eq!(stats.input_tokens, 0);
    assert_eq!(stats.reasoning_tokens, 0);
    assert_eq!(stats.output_tokens, 10);

    // reasoning > completion → the subtraction saturates at 0.
    let inverted = RawUsage {
        prompt_tokens: Some(1),
        completion_tokens: Some(5),
        completion_tokens_details: Some(CompletionTokensDetails {
            reasoning_tokens: Some(9),
        }),
    };
    let stats = tracker.record("gpt-5-mini", &inverted);
    assert_eq!(stats.output_tokens, 0);
    assert_eq!(stats.reasoning_tokens, 9);
}

#[test]
fn usage_stats_to_json_has_the_pinned_keys() {
    let stats = UsageStats {
        calls: 2,
        input_tokens: 1000,
        output_tokens: 500,
        reasoning_tokens: 200,
        cost: 0.00165,
    };
    assert_eq!(
        stats.to_json(),
        json!({
            "calls": 2,
            "input_tokens": 1000,
            "output_tokens": 500,
            "reasoning_tokens": 200,
            "total_tokens": 1700,
            "cost": 0.00165,
            // `format!("${:.4}", 0.00165)` is '$0.0016': 0.00165 sits just
            // under the midpoint as a double.
            "cost_display": "$0.0016",
        })
    );
}

#[test]
fn usage_stats_cost_rounds_to_six_decimals() {
    // Costs are rounded to six decimal places with ties going to even. The
    // old `(v * 1e6).round() / 1e6` got the first two wrong (4e-06 / 1.4e-05):
    // half-way values are common at these magnitudes, so the rounding must be
    // done on the correctly-rounded decimal, not on the scaled binary float.
    let cost_of = |model: &str, i: u64, o: u64, t: u64| {
        UsageStats {
            calls: 1,
            input_tokens: i,
            output_tokens: o,
            reasoning_tokens: t,
            cost: crate::pricing::get_cost(model, i, o, t),
        }
        .to_json()["cost"]
            .as_f64()
            .expect("cost is a number")
    };

    // get_cost('gpt-5-nano', 70, 0, 0) == 3.5e-06 → round(...,6) == 3e-06
    assert_eq!(cost_of("gpt-5-nano", 70, 0, 0), 3e-06);
    // get_cost('gpt-5-nano', 30, 30, 0) == 1.35e-05 → 1.3e-05
    assert_eq!(cost_of("gpt-5-nano", 30, 30, 0), 1.3e-05);
    // get_cost('gpt-5-mini', 490, 490, 0) == 0.0011025 → 0.001102
    assert_eq!(cost_of("gpt-5-mini", 490, 490, 0), 0.001102);
}

#[test]
fn global_tracker_is_shared() {
    assert!(std::ptr::eq(get_usage_tracker(), get_usage_tracker()));
}

// ---------------------------------------------------------------------------
// HTTP: OpenAI request/response
// ---------------------------------------------------------------------------

#[tokio::test]
async fn openai_request_shape_and_usage() {
    let body = json!({
        "choices": [{"message": {"content": "hello"}}],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 700,
            "completion_tokens_details": {"reasoning_tokens": 200}
        }
    })
    .to_string();
    let server = MockServer::start(vec![(200, body)]);
    // The env guard is released before the first await (clippy:
    // await_holding_lock); the client has already captured the key.
    let client = {
        let _lock = env_lock();
        let _guard = EnvVarGuard::unset("OPENAI_API_KEY");
        client_for(&server, "gpt-5-mini")
    };

    let tracker = UsageTracker::new();
    let response = client
        .call_llm_with(
            &[
                Message::system("sys prompt"),
                Message::user("hi <|endoftext|> there"),
            ],
            DEFAULT_MAX_RETRIES,
            Some(&tracker),
        )
        .await
        .expect("call succeeds");

    assert_eq!(response.content(), "hello");
    assert_eq!(response.model, "gpt-5-mini");
    assert_eq!(response.call_stats.output_tokens, 500);
    assert_eq!(response.call_stats.reasoning_tokens, 200);
    assert_eq!(tracker.get_stats().calls, 1);

    let requests = server.requests();
    assert_eq!(requests.len(), 1);
    let req = &requests[0];
    assert_eq!(req.method, "POST");
    assert_eq!(req.path, "/chat/completions");
    assert_eq!(req.header("Authorization"), Some("Bearer sk-openai-test"));

    let sent = req.json();
    assert_eq!(sent["model"], "gpt-5-mini");
    assert_eq!(sent["reasoning_effort"], "low");
    // Neither temperature nor max_tokens is ever sent to OpenAI.
    assert!(sent.get("temperature").is_none());
    assert!(sent.get("max_tokens").is_none());
    // System messages stay inline for OpenAI, and special tokens are escaped.
    assert_eq!(sent["messages"][0]["role"], "system");
    assert_eq!(sent["messages"][0]["content"], "sys prompt");
    assert_eq!(sent["messages"][1]["role"], "user");
    assert_eq!(sent["messages"][1]["content"], "hi <| endoftext |> there");
}

#[tokio::test]
async fn openai_strips_reasoning_effort_for_unsupported_models() {
    let server = MockServer::start(vec![(200, openai_body("ok"))]);
    let client = client_for(&server, "gpt-4o");

    client
        .call_llm_with(&[Message::user("hi")], 1, Some(&UsageTracker::new()))
        .await
        .expect("call succeeds");

    let sent = server.requests()[0].json();
    assert_eq!(sent["model"], "gpt-4o");
    assert!(
        sent.get("reasoning_effort").is_none(),
        "reasoning_effort must be stripped, not sent"
    );
}

#[tokio::test]
async fn openai_empty_choices_is_invalid_response() {
    let server = MockServer::start(vec![(200, json!({"choices": []}).to_string())]);
    let client = client_for(&server, "gpt-5-mini");

    let err = client
        .call_llm_with(&[Message::user("hi")], 3, Some(&UsageTracker::new()))
        .await
        .expect_err("no choices");
    // An empty `choices` list is a typed, non-retryable InvalidResponse.
    assert!(matches!(err, LlmError::InvalidResponse(_)));
    // Non-retryable: only one request was made.
    assert_eq!(server.requests().len(), 1);
}

#[tokio::test]
async fn vendor_prefixed_model_is_stripped_on_the_wire() {
    // Config may name a model `openai/gpt-5-mini`, but the OpenAI API rejects
    // the prefixed form, so ekko routes on the full name and sends the bare
    // one. Capability detection must see the stripped name too.
    let server = MockServer::start(vec![(200, openai_body("ok"))]);
    let client = client_for(&server, "openai/gpt-5-mini");
    assert_eq!(client.provider(), Provider::OpenAi);
    assert_eq!(client.model(), "openai/gpt-5-mini");

    client
        .call_llm_with(&[Message::user("hi")], 1, Some(&UsageTracker::new()))
        .await
        .expect("call succeeds");

    let sent = server.requests()[0].json();
    assert_eq!(sent["model"], "gpt-5-mini");
    assert_eq!(sent["reasoning_effort"], "low");
}

#[tokio::test]
async fn missing_usage_does_not_count_a_call() {
    // No usage object means `record` is never called, so `calls` stays 0.
    // An empty or null usage object counts as absent too.
    let no_usage = json!({"choices": [{"message": {"content": "hi"}}]}).to_string();
    let empty_usage = json!({"choices": [{"message": {"content": "hi"}}], "usage": {}}).to_string();
    let null_usage =
        json!({"choices": [{"message": {"content": "hi"}}], "usage": null}).to_string();
    let server = MockServer::start(vec![(200, no_usage), (200, empty_usage), (200, null_usage)]);
    let client = client_for(&server, "gpt-5-mini");
    let tracker = UsageTracker::new();

    for _ in 0..3 {
        let response = client
            .call_llm_with(&[Message::user("hi")], 1, Some(&tracker))
            .await
            .expect("call succeeds");
        assert_eq!(response.content(), "hi");
        assert_eq!(response.call_stats, UsageStats::default());
    }
    assert_eq!(tracker.get_stats(), UsageStats::default());
}

#[tokio::test]
async fn present_but_zero_usage_counts_a_call() {
    // A usage object that is present but all-zero is still present, so
    // `calls` increments.
    let body = json!({
        "choices": [{"message": {"content": "hi"}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0}
    })
    .to_string();
    let server = MockServer::start(vec![(200, body)]);
    let client = client_for(&server, "gpt-5-mini");
    let tracker = UsageTracker::new();

    let response = client
        .call_llm_with(&[Message::user("hi")], 1, Some(&tracker))
        .await
        .expect("call succeeds");
    assert_eq!(response.call_stats.calls, 1);
    assert_eq!(tracker.get_stats().calls, 1);
    assert_eq!(tracker.get_stats().input_tokens, 0);
}

// ---------------------------------------------------------------------------
// HTTP: Anthropic request/response
// ---------------------------------------------------------------------------

#[tokio::test]
async fn anthropic_request_shape_and_usage() {
    let body = json!({
        "content": [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "part one "},
            {"type": "text", "text": "part two"}
        ],
        "usage": {"input_tokens": 120, "output_tokens": 34}
    })
    .to_string();
    let server = MockServer::start(vec![(200, body)]);
    // Env guard released before the first await (clippy: await_holding_lock).
    let client = {
        let _lock = env_lock();
        let _guard = EnvVarGuard::unset("ANTHROPIC_API_KEY");
        client_for(&server, "claude-opus-4-6")
    };

    let tracker = UsageTracker::new();
    let response = client
        .call_llm_with(
            &[
                Message::system("you are terse"),
                Message::user("q1"),
                Message::assistant("a1"),
                Message::system("also this"),
                Message::user("q2"),
            ],
            1,
            Some(&tracker),
        )
        .await
        .expect("call succeeds");

    assert_eq!(response.content(), "part one part two");
    assert_eq!(response.call_stats.input_tokens, 120);
    assert_eq!(response.call_stats.output_tokens, 34);
    assert_eq!(response.call_stats.reasoning_tokens, 0);
    // claude-opus-4-6 → version-suffix match on "claude-opus-4-6" pricing.
    assert_eq!(
        response.call_stats.cost,
        crate::pricing::get_cost("claude-opus-4-6", 120, 34, 0)
    );

    let requests = server.requests();
    let req = &requests[0];
    assert_eq!(req.path, "/messages");
    assert_eq!(req.header("x-api-key"), Some("sk-ant-test"));
    assert_eq!(req.header("anthropic-version"), Some("2023-06-01"));

    let sent = req.json();
    assert_eq!(sent["model"], "claude-opus-4-6");
    assert_eq!(sent["max_tokens"], ANTHROPIC_MAX_TOKENS);
    // System messages are hoisted out of `messages`, joined with a blank line.
    assert_eq!(sent["system"], "you are terse\n\nalso this");
    assert_eq!(sent["messages"].as_array().map(Vec::len), Some(3));
    assert_eq!(sent["messages"][0]["role"], "user");
    assert_eq!(sent["messages"][1]["role"], "assistant");
    assert_eq!(sent["messages"][2]["content"], "q2");
    // Anthropic never receives reasoning_effort.
    assert!(sent.get("reasoning_effort").is_none());
}

#[tokio::test]
async fn anthropic_without_system_message_omits_system_field() {
    let body = json!({
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 1, "output_tokens": 1}
    })
    .to_string();
    let server = MockServer::start(vec![(200, body)]);
    let client = client_for(&server, "claude-sonnet-4-5");

    client
        .call_llm_with(
            &[Message::user("hi <|im_start|>")],
            1,
            Some(&UsageTracker::new()),
        )
        .await
        .expect("call succeeds");

    let sent = server.requests()[0].json();
    assert!(sent.get("system").is_none());
    // Escaping applies on the Anthropic path too.
    assert_eq!(sent["messages"][0]["content"], "hi <| im_start |>");
}

#[tokio::test]
async fn anthropic_max_tokens_is_model_aware() {
    // claude-3-5-* caps max output at 8192; sending 32000 would be a
    // non-retryable 400 from Anthropic.
    let body = json!({
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 1, "output_tokens": 1}
    })
    .to_string();
    let server = MockServer::start(vec![(200, body)]);
    let client = client_for(&server, "claude-3-5-sonnet-20241022");

    client
        .call_llm_with(&[Message::user("hi")], 1, Some(&UsageTracker::new()))
        .await
        .expect("call succeeds");

    assert_eq!(server.requests()[0].json()["max_tokens"], 8192);
}

#[test]
fn anthropic_max_tokens_heuristic() {
    assert_eq!(anthropic_max_tokens("claude-3-5-sonnet-20241022"), 8192);
    assert_eq!(anthropic_max_tokens("claude-3-7-sonnet-latest"), 8192);
    assert_eq!(anthropic_max_tokens("claude-3-haiku-20240307"), 4096);
    assert_eq!(anthropic_max_tokens("claude-3-opus-20240229"), 4096);
    assert_eq!(anthropic_max_tokens("claude-3-sonnet-20240229"), 4096);
    assert_eq!(anthropic_max_tokens("Claude-3-Opus-20240229"), 4096);
    assert_eq!(
        anthropic_max_tokens("claude-sonnet-4-5"),
        ANTHROPIC_MAX_TOKENS
    );
    assert_eq!(
        anthropic_max_tokens("claude-opus-4-6"),
        ANTHROPIC_MAX_TOKENS
    );
}

#[test]
fn wire_model_strips_vendor_prefixes() {
    assert_eq!(wire_model("openai/gpt-5-mini"), "gpt-5-mini");
    assert_eq!(wire_model("anthropic/claude-opus-4-6"), "claude-opus-4-6");
    assert_eq!(wire_model("google/gemini-3-pro"), "gemini-3-pro");
    // Untouched otherwise (same prefix list as pricing::normalize_model_name).
    assert_eq!(wire_model("gpt-5-mini"), "gpt-5-mini");
    assert_eq!(wire_model("azure/gpt-4"), "azure/gpt-4");
    assert_eq!(wire_model("openai"), "openai");
    // Capability detection runs on the stripped name.
    assert!(supports_reasoning_effort(wire_model("openai/gpt-5-mini")));
}

// ---------------------------------------------------------------------------
// HTTP: retry policy
// ---------------------------------------------------------------------------

/// Server errors are transient, so they are retried like any other transport
/// failure. This test pins that.
#[tokio::test]
async fn retries_on_5xx_then_succeeds() {
    let server = MockServer::start(vec![
        (500, "{\"error\": \"boom\"}".to_string()),
        (503, "{\"error\": \"still boom\"}".to_string()),
        (200, openai_body("recovered")),
    ]);
    let client = client_for(&server, "gpt-5-mini");

    let response = client
        .call_llm_with(&[Message::user("hi")], 3, Some(&UsageTracker::new()))
        .await
        .expect("third attempt succeeds");
    assert_eq!(response.content(), "recovered");
    assert_eq!(server.requests().len(), 3);
}

/// Rate limiting is retryable; the wait is handled by `backoff_delay`.
#[tokio::test]
async fn retries_on_429() {
    let server = MockServer::start(vec![
        (429, "{\"error\": \"rate limited\"}".to_string()),
        (200, openai_body("ok")),
    ]);
    let client = client_for(&server, "gpt-5-mini");

    let response = client
        .call_llm_with(&[Message::user("hi")], 3, Some(&UsageTracker::new()))
        .await
        .expect("second attempt succeeds");
    assert_eq!(response.content(), "ok");
    assert_eq!(server.requests().len(), 2);
}

#[tokio::test]
async fn does_not_retry_on_4xx() {
    let server = MockServer::start(vec![
        (400, "{\"error\": \"bad request\"}".to_string()),
        (200, openai_body("never reached")),
    ]);
    let client = client_for(&server, "gpt-5-mini");

    let err = client
        .call_llm_with(&[Message::user("hi")], 3, Some(&UsageTracker::new()))
        .await
        .expect_err("400 propagates");
    match err {
        LlmError::Http { status, ref body } => {
            assert_eq!(status, Some(400));
            assert!(body.contains("bad request"));
        }
        other => panic!("unexpected error: {other}"),
    }
    assert_eq!(server.requests().len(), 1);
}

#[tokio::test]
async fn exhausts_retries_and_returns_last_error() {
    let server = MockServer::start(vec![
        (503, "a".to_string()),
        (503, "b".to_string()),
        (503, "c".to_string()),
    ]);
    let client = client_for(&server, "gpt-5-mini");

    let err = client
        .call_llm_with(&[Message::user("hi")], 3, Some(&UsageTracker::new()))
        .await
        .expect_err("all attempts fail");
    assert!(matches!(
        err,
        LlmError::Http {
            status: Some(503),
            ..
        }
    ));
    assert_eq!(server.requests().len(), 3);
}

#[tokio::test]
async fn transport_error_is_retryable() {
    // Port 1 is privileged and never bound by a test server, so this is a
    // deterministic connection-refused. (Binding an ephemeral port and
    // dropping it would race with the other tests' mock servers reusing it.)
    let client = LlmClient::new(&test_config("gpt-5-mini"), "gpt-5-mini")
        .expect("client builds")
        .with_base_url("http://127.0.0.1:1")
        .with_backoff_base(Duration::from_millis(1));

    let err = client
        .call_llm_with(&[Message::user("hi")], 2, Some(&UsageTracker::new()))
        .await
        .expect_err("connection refused");
    assert!(is_retryable(&err), "transport failures must be retryable");
    assert!(matches!(err, LlmError::Http { status: None, .. }));
}

/// A zero retry budget would mean zero attempts and nothing to report, so it
/// is rejected up front as a caller bug.
#[tokio::test]
async fn zero_max_retries_is_rejected() {
    let server = MockServer::start(vec![]);
    let client = client_for(&server, "gpt-5-mini");
    let err = client
        .call_llm_with(&[Message::user("hi")], 0, Some(&UsageTracker::new()))
        .await
        .expect_err("zero retries");
    assert!(matches!(err, LlmError::InvalidRequest(_)));
}

// ---------------------------------------------------------------------------
// HTTP: call_llm_json re-ask loop
// ---------------------------------------------------------------------------

#[tokio::test]
async fn json_retry_appends_exact_nudge_and_recovers() {
    let server = MockServer::start(vec![
        (200, openai_body("sorry, no JSON here")),
        (200, openai_body("```json\n{\"answer\": 42}\n```")),
    ]);
    let client = client_for(&server, "gpt-5-mini");

    let value = client
        .call_llm_json_with(
            &[Message::user("give me json")],
            3,
            2,
            Some(&UsageTracker::new()),
        )
        .await
        .expect("json call succeeds");
    assert_eq!(value, json!({"answer": 42}));

    let requests = server.requests();
    assert_eq!(requests.len(), 2);
    let second = requests[1].json();
    let messages = second["messages"].as_array().expect("messages array");
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0]["content"], "give me json");
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[1]["content"], "sorry, no JSON here");
    assert_eq!(messages[2]["role"], "user");
    assert_eq!(
        messages[2]["content"],
        "Please respond with valid JSON only, no markdown fences or extra text."
    );
    assert_eq!(messages[2]["content"], JSON_RETRY_NUDGE);
}

#[tokio::test]
async fn json_retries_exhausted_returns_empty_object() {
    let server = MockServer::start(vec![
        (200, openai_body("nope")),
        (200, openai_body("still nope")),
        (200, openai_body("{}")), // empty dict is falsy → also a failure
    ]);
    let client = client_for(&server, "gpt-5-mini");

    let value = client
        .call_llm_json_with(
            &[Message::user("json please")],
            3,
            2,
            Some(&UsageTracker::new()),
        )
        .await
        .expect("returns {} rather than erroring");
    assert_eq!(value, json!({}));
    assert_eq!(server.requests().len(), 3);
}

#[tokio::test]
async fn json_call_propagates_api_errors() {
    let server = MockServer::start(vec![(401, "{\"error\": \"bad key\"}".to_string())]);
    let client = client_for(&server, "gpt-5-mini");

    let err = client
        .call_llm_json_with(&[Message::user("json")], 3, 2, Some(&UsageTracker::new()))
        .await
        .expect_err("401 propagates");
    assert!(matches!(
        err,
        LlmError::Http {
            status: Some(401),
            ..
        }
    ));
    assert_eq!(server.requests().len(), 1);
}

#[tokio::test]
async fn json_first_response_parses_without_retry() {
    let server = MockServer::start(vec![(200, openai_body("{\"ok\": true}"))]);
    let client = client_for(&server, "gpt-5-mini");

    let value = client
        .call_llm_json_with(&[Message::user("json")], 3, 2, Some(&UsageTracker::new()))
        .await
        .expect("parses");
    assert_eq!(value, json!({"ok": true}));
    assert_eq!(server.requests().len(), 1);
}

// ---------------------------------------------------------------------------
// Message preparation
// ---------------------------------------------------------------------------

#[test]
fn prepare_messages_escapes_every_content() {
    let prepared = prepare_messages(&[
        Message::system("<|endoftext|>"),
        Message::user("a <|fim_prefix|> b"),
        Message::assistant("plain"),
    ]);
    assert_eq!(prepared[0].content, "<| endoftext |>");
    assert_eq!(prepared[1].content, "a <| fim_prefix |> b");
    assert_eq!(prepared[2].content, "plain");
    // Roles are untouched.
    assert_eq!(prepared[0].role, "system");
}

#[test]
fn message_serializes_to_role_and_content() {
    let value = serde_json::to_value(Message::user("hi")).expect("serializes");
    assert_eq!(value, json!({"role": "user", "content": "hi"}));
}

// ---------------------------------------------------------------------------
// Azure OpenAI routing
// ---------------------------------------------------------------------------

/// A config that routes OpenAI-family models to an Azure endpoint. No
/// `OPENAI_API_KEY` on purpose: Azure mode must use the Azure key.
fn azure_config(model: &str, endpoint: &str, api_version: Option<&str>) -> Config {
    let mut api_keys = vec![
        (AZURE_KEY_VAR.to_string(), "az-test-key".to_string()),
        (AZURE_ENDPOINT_VAR.to_string(), endpoint.to_string()),
        ("ANTHROPIC_API_KEY".to_string(), "sk-ant-test".to_string()),
    ];
    if let Some(v) = api_version {
        api_keys.push((AZURE_API_VERSION_VAR.to_string(), v.to_string()));
    }
    Config {
        llm_model: model.to_string(),
        api_keys,
        ..Config::default()
    }
}

/// Guards that blank out any Azure/OpenAI keys leaking in from the real
/// process env, so `get_api_key` resolves from the test config alone.
fn azure_env_guards() -> [EnvVarGuard; 4] {
    [
        EnvVarGuard::unset("OPENAI_API_KEY"),
        EnvVarGuard::unset(AZURE_KEY_VAR),
        EnvVarGuard::unset(AZURE_ENDPOINT_VAR),
        EnvVarGuard::unset(AZURE_API_VERSION_VAR),
    ]
}

#[tokio::test]
async fn azure_request_uses_v1_path_and_api_key_header() {
    let server = MockServer::start(vec![(200, openai_body("hi from azure"))]);
    // Trailing slash on purpose: it must be trimmed, not doubled.
    let endpoint = format!("{}/", server.base_url());
    let client = {
        let _lock = env_lock();
        let _guards = azure_env_guards();
        LlmClient::new(&azure_config("gpt-5-mini", &endpoint, None), "gpt-5-mini")
            .expect("client builds")
            .with_backoff_base(Duration::from_millis(1))
    };

    let response = client
        .call_llm_with(&[Message::user("hi")], 1, Some(&UsageTracker::new()))
        .await
        .expect("call succeeds");
    assert_eq!(response.content(), "hi from azure");

    let requests = server.requests();
    assert_eq!(requests.len(), 1);
    let req = &requests[0];
    assert_eq!(req.path, "/openai/v1/chat/completions");
    assert_eq!(req.header("api-key"), Some("az-test-key"));
    assert_eq!(
        req.header("Authorization"),
        None,
        "no bearer token to Azure"
    );
    // Same body schema as OpenAI: deployment name rides the model field.
    assert_eq!(req.json()["model"], "gpt-5-mini");
}

#[test]
fn azure_url_appends_api_version_when_configured() {
    let _lock = env_lock();
    let _guards = azure_env_guards();

    let config = azure_config(
        "gpt-5-mini",
        "https://myres.openai.azure.com",
        Some("preview"),
    );
    let client = LlmClient::new(&config, "gpt-5-mini").expect("client builds");
    assert_eq!(
        client.url(),
        "https://myres.openai.azure.com/openai/v1/chat/completions?api-version=preview"
    );

    // And without a version, no query string at all.
    let config = azure_config("gpt-5-mini", "https://myres.openai.azure.com", None);
    let client = LlmClient::new(&config, "gpt-5-mini").expect("client builds");
    assert_eq!(
        client.url(),
        "https://myres.openai.azure.com/openai/v1/chat/completions"
    );
}

#[test]
fn azure_endpoint_does_not_touch_anthropic_models() {
    let _lock = env_lock();
    let _guards = azure_env_guards();
    let _ant = EnvVarGuard::unset("ANTHROPIC_API_KEY");

    let config = azure_config("claude-opus-4-6", "https://myres.openai.azure.com", None);
    let client = LlmClient::new(&config, "claude-opus-4-6").expect("client builds");
    assert_eq!(client.url(), format!("{ANTHROPIC_BASE_URL}/messages"));
    assert!(require_llm(&config, "claude-opus-4-6").is_ok());
}

#[test]
fn azure_endpoint_requires_the_azure_key() {
    let _lock = env_lock();
    let _guards = azure_env_guards();

    // Endpoint set, OPENAI_API_KEY set — but no Azure key: Azure routing is
    // active, so the OpenAI key must NOT satisfy the check.
    let config = Config {
        api_keys: vec![
            (
                AZURE_ENDPOINT_VAR.to_string(),
                "https://myres.openai.azure.com".to_string(),
            ),
            ("OPENAI_API_KEY".to_string(), "sk-openai-test".to_string()),
        ],
        ..Config::default()
    };

    let err = LlmClient::new(&config, "gpt-5-mini").expect_err("must not build");
    assert!(err.to_string().contains(AZURE_KEY_VAR), "got: {err}");

    let (available, msg) = check_llm_available(&config, "gpt-5-mini");
    assert!(!available);
    assert!(msg.contains(AZURE_KEY_VAR), "got: {msg}");

    let err = require_llm(&config, "gpt-5-mini").expect_err("must not pass");
    assert_eq!(err.env_var, AZURE_KEY_VAR);
}

#[test]
fn with_base_url_switches_off_azure_routing() {
    let _lock = env_lock();
    let _guards = azure_env_guards();

    let config = azure_config("gpt-5-mini", "https://myres.openai.azure.com", None);
    let client = LlmClient::new(&config, "gpt-5-mini")
        .expect("client builds")
        .with_base_url("http://127.0.0.1:9/");
    assert_eq!(client.url(), "http://127.0.0.1:9/chat/completions");
}

// ── backend resolution ────────────────────────────────────────────────

/// Point PATH at a directory with a fake `claude` executable (or at an
/// empty directory) so resolution is independent of the host machine.
fn fake_path(with_claude: bool) -> (tempfile::TempDir, EnvVarGuard) {
    let dir = tempfile::tempdir().expect("tempdir");
    if with_claude {
        let bin = dir.path().join("claude");
        std::fs::write(&bin, "#!/bin/sh\nexit 0\n").expect("write");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        }
    }
    let guard = EnvVarGuard::set("PATH", dir.path().to_str().expect("utf8 path"));
    (dir, guard)
}

fn keyless_guards() -> [EnvVarGuard; 4] {
    [
        EnvVarGuard::unset("OPENAI_API_KEY"),
        EnvVarGuard::unset("ANTHROPIC_API_KEY"),
        EnvVarGuard::unset(AZURE_KEY_VAR),
        EnvVarGuard::unset(AZURE_ENDPOINT_VAR),
    ]
}

#[test]
fn backend_parse_accepts_aliases() {
    assert_eq!(Backend::parse("api"), Some(Backend::Api));
    assert_eq!(Backend::parse("OpenAI"), Some(Backend::Api));
    assert_eq!(Backend::parse("azure"), Some(Backend::Api));
    assert_eq!(
        Backend::parse("claude-cli"),
        Some(Backend::Cli(CliKind::Claude))
    );
    assert_eq!(
        Backend::parse(" claude_cli "),
        Some(Backend::Cli(CliKind::Claude))
    );
    assert_eq!(Backend::parse("codex-cli"), None);
}

#[test]
fn resolve_prefers_api_key_when_present() {
    let _lock = env_lock();
    let _keys = keyless_guards();
    let (_dir, _path) = fake_path(true);

    let resolved = resolve_backend(&test_config("gpt-5-mini"), Role::Index);
    assert_eq!(resolved.backend, Backend::Api);
    assert_eq!(resolved.model, "gpt-5-mini");
    assert!(
        resolved.reason.contains("API key found"),
        "{}",
        resolved.reason
    );
}

#[test]
fn resolve_falls_back_to_cli_without_key() {
    let _lock = env_lock();
    let _keys = keyless_guards();
    let (_dir, _path) = fake_path(true);

    // Default model (gpt-5-mini, not explicit) → CLI default model.
    let resolved = resolve_backend(&Config::default(), Role::Index);
    assert_eq!(resolved.backend, Backend::Cli(CliKind::Claude));
    assert_eq!(resolved.model, CliKind::Claude.default_model());
    assert!(
        resolved.reason.contains("no API key"),
        "{}",
        resolved.reason
    );

    // Explicit OpenAI-family model the CLI cannot serve → CLI default.
    let config = Config {
        llm_model: "gpt-5-mini".to_string(),
        llm_model_explicit: true,
        ..Config::default()
    };
    assert_eq!(resolve_backend(&config, Role::Index).model, "sonnet");

    // Explicit Claude model → kept.
    let config = Config {
        llm_model: "haiku".to_string(),
        llm_model_explicit: true,
        ..Config::default()
    };
    assert_eq!(resolve_backend(&config, Role::Index).model, "haiku");
}

#[test]
fn resolve_without_key_or_cli_stays_api() {
    let _lock = env_lock();
    let _keys = keyless_guards();
    let (_dir, _path) = fake_path(false);

    let resolved = resolve_backend(&Config::default(), Role::Index);
    assert_eq!(resolved.backend, Backend::Api);
    assert!(
        resolved.reason.contains("no supported CLI"),
        "{}",
        resolved.reason
    );
    let (available, _) = check_backend_available(&Config::default(), Role::Index);
    assert!(!available);
}

#[test]
fn resolve_explicit_backend_wins() {
    let _lock = env_lock();
    let _keys = keyless_guards();
    let (_dir, _path) = fake_path(true);

    // Explicit CLI even though an API key exists.
    let config = Config {
        llm_backend: Some("claude-cli".to_string()),
        ..test_config("gpt-5-mini")
    };
    let resolved = resolve_backend(&config, Role::Index);
    assert_eq!(resolved.backend, Backend::Cli(CliKind::Claude));
    assert_eq!(resolved.model, "sonnet");
    assert!(resolved.reason.contains("explicit"));

    // Explicit CLI with an explicit Claude model: taken verbatim.
    let config = Config {
        llm_backend: Some("claude-cli".to_string()),
        llm_model: "opus".to_string(),
        llm_model_explicit: true,
        ..Config::default()
    };
    assert_eq!(resolve_backend(&config, Role::Index).model, "opus");

    // Explicit CLI with a leftover OpenAI model: CLI default, and the reason
    // explains the substitution.
    let config = Config {
        llm_backend: Some("claude-cli".to_string()),
        llm_model: "gpt-5-mini".to_string(),
        llm_model_explicit: true,
        ..Config::default()
    };
    let resolved = resolve_backend(&config, Role::Index);
    assert_eq!(resolved.model, "sonnet");
    assert!(
        resolved.reason.contains("cannot serve"),
        "{}",
        resolved.reason
    );

    // Explicit api with no key: stays api so the key guidance shows.
    let config = Config {
        llm_backend: Some("api".to_string()),
        ..Config::default()
    };
    let resolved = resolve_backend(&config, Role::Index);
    assert_eq!(resolved.backend, Backend::Api);
    assert!(!check_backend_available(&config, Role::Index).0);

    // Unrecognised value: auto-detect, but say so.
    let config = Config {
        llm_backend: Some("codex-cli".to_string()),
        ..Config::default()
    };
    let resolved = resolve_backend(&config, Role::Index);
    assert_eq!(resolved.backend, Backend::Cli(CliKind::Claude));
    assert!(
        resolved.reason.contains("not recognised"),
        "{}",
        resolved.reason
    );
}

#[test]
fn cli_backend_defaults_to_one_worker() {
    let _lock = env_lock();
    let _keys = keyless_guards();
    let (_dir, _path) = fake_path(true);

    assert_eq!(effective_max_workers(&Config::default()), 1);
    let config = Config {
        max_workers: 4,
        max_workers_explicit: true,
        ..Config::default()
    };
    assert_eq!(effective_max_workers(&config), 4);
    assert_eq!(
        effective_max_workers(&test_config("gpt-5-mini")),
        crate::config::DEFAULT_MAX_WORKERS
    );
}

#[test]
fn cli_client_builds_and_reports_backend() {
    let _lock = env_lock();
    let _keys = keyless_guards();
    let (_dir, _path) = fake_path(true);

    let client = LlmClient::from_config(&Config::default()).expect("cli client builds");
    assert_eq!(client.backend(), Backend::Cli(CliKind::Claude));
    assert_eq!(client.model(), "sonnet");
    assert!(!format!("{client:?}").contains("sk-"));

    let (_dir2, _path2) = fake_path(false);
    let err =
        LlmClient::new_cli(&Config::default(), CliKind::Claude, "sonnet").expect_err("no binary");
    assert!(matches!(err, LlmError::NotConfigured(_)), "{err:?}");
}

#[test]
fn tracker_uses_reported_cost_when_given() {
    let tracker = UsageTracker::new();
    let usage = RawUsage {
        prompt_tokens: Some(10),
        completion_tokens: Some(5),
        completion_tokens_details: None,
    };
    let stats = tracker.record_with_cost("claude-sonnet-4-5", &usage, Some(0.1234567));
    assert_eq!(stats.cost, 0.123457);
    let stats = tracker.record_with_cost("gpt-5-mini", &usage, None);
    assert_eq!(stats.cost, get_cost("gpt-5-mini", 10, 5, 0));
}

#[test]
fn query_role_follows_index_unless_overridden() {
    let _lock = env_lock();
    let _keys = keyless_guards();
    let (_dir, _path) = fake_path(true);

    // No query settings: identical to index, reason says so.
    let config = Config {
        llm_backend: Some("claude-cli".to_string()),
        ..Config::default()
    };
    let index = resolve_backend(&config, Role::Index);
    let query = resolve_backend(&config, Role::Query);
    assert_eq!(
        (query.backend, query.model.as_str()),
        (index.backend, index.model.as_str())
    );
    assert!(
        query.reason.ends_with("(same as index)"),
        "{}",
        query.reason
    );

    // query_backend=api with a key: API for queries, CLI for indexing, and
    // the *inherited* CLI-alias llm_model is replaced by the API default.
    let config = Config {
        llm_backend: Some("claude-cli".to_string()),
        llm_model: "sonnet".to_string(),
        llm_model_explicit: true,
        query_backend: Some("api".to_string()),
        ..test_config("sonnet")
    };
    assert_eq!(
        resolve_backend(&config, Role::Index).backend,
        Backend::Cli(CliKind::Claude)
    );
    let query = resolve_backend(&config, Role::Query);
    assert_eq!(query.backend, Backend::Api);
    assert_eq!(query.model, crate::config::DEFAULT_LLM_MODEL);
    assert!(query.reason.contains("no API provider"), "{}", query.reason);
    assert!(check_backend_available(&config, Role::Query).0);

    // query_model alone: same backend, different model.
    let config = Config {
        llm_backend: Some("claude-cli".to_string()),
        query_model: Some("haiku".to_string()),
        ..Config::default()
    };
    let query = resolve_backend(&config, Role::Query);
    assert_eq!(query.backend, Backend::Cli(CliKind::Claude));
    assert_eq!(query.model, "haiku");
    assert_eq!(resolve_backend(&config, Role::Index).model, "sonnet");

    // A query_model chosen for the role is kept verbatim even when ekko
    // cannot route it — the call fails honestly instead of silently using
    // another model.
    let config = Config {
        query_backend: Some("api".to_string()),
        query_model: Some("ekko-test-no-such-provider".to_string()),
        ..test_config("gpt-5-mini")
    };
    assert_eq!(
        resolve_backend(&config, Role::Query).model,
        "ekko-test-no-such-provider"
    );
    assert!(matches!(
        LlmClient::for_role(&config, Role::Query),
        Err(LlmError::UnknownProvider { .. })
    ));

    // query_backend + query_model together.
    let config = Config {
        llm_backend: Some("claude-cli".to_string()),
        query_backend: Some("api".to_string()),
        query_model: Some("gpt-5-mini".to_string()),
        ..test_config("sonnet")
    };
    let query = resolve_backend(&config, Role::Query);
    assert_eq!(
        (query.backend, query.model.as_str()),
        (Backend::Api, "gpt-5-mini")
    );

    // Unrecognised query_backend: index resolution, flagged.
    let config = Config {
        llm_backend: Some("claude-cli".to_string()),
        query_backend: Some("gemini-cli".to_string()),
        ..Config::default()
    };
    let query = resolve_backend(&config, Role::Query);
    assert_eq!(query.backend, Backend::Cli(CliKind::Claude));
    assert!(query.reason.contains("not recognised"), "{}", query.reason);

    // Workers follow the index role only.
    let config = Config {
        llm_backend: Some("api".to_string()),
        query_backend: Some("claude-cli".to_string()),
        ..test_config("gpt-5-mini")
    };
    assert_eq!(
        effective_max_workers(&config),
        crate::config::DEFAULT_MAX_WORKERS
    );
    assert_eq!(
        LlmClient::for_role(&config, Role::Query)
            .expect("builds")
            .backend(),
        Backend::Cli(CliKind::Claude)
    );
    assert_eq!(
        LlmClient::for_role(&config, Role::Index)
            .expect("builds")
            .backend(),
        Backend::Api
    );
}

// ── refactor guards: reason ladder, availability-of-resolution, mode gate ──

#[test]
fn resolution_reasons_are_stable_across_the_precedence_ladder() {
    let _lock = env_lock();
    let _keys = keyless_guards();
    let (_dir, _path) = fake_path(true);
    let cli_path = std::env::var("PATH").unwrap();
    let claude = format!("{cli_path}/claude");

    // Index: explicit / unrecognised / absent.
    let c = Config { llm_backend: Some("claude-cli".into()), ..Config::default() };
    assert_eq!(resolve_backend(&c, Role::Index).reason, "llm_backend=claude-cli (explicit)");
    let c = Config { llm_backend: Some("bogus".into()), ..Config::default() };
    assert_eq!(
        resolve_backend(&c, Role::Index).reason,
        format!("llm_backend 'bogus' not recognised (expected one of api, claude-cli); no API key configured; found claude at {claude}")
    );
    let c = Config::default();
    assert_eq!(
        resolve_backend(&c, Role::Index).reason,
        format!("no API key configured; found claude at {claude}")
    );
    assert_eq!(
        resolve_backend(&test_config("gpt-5-mini"), Role::Index).reason,
        "API key found for model 'gpt-5-mini'"
    );

    // Query: explicit / unrecognised / absent (+ query_model only).
    let c = Config { query_backend: Some("api".into()), ..test_config("gpt-5-mini") };
    assert_eq!(resolve_backend(&c, Role::Query).reason, "query_backend=api (explicit)");
    let c = Config { query_backend: Some("nope".into()), ..test_config("gpt-5-mini") };
    assert_eq!(
        resolve_backend(&c, Role::Query).reason,
        "query_backend 'nope' not recognised (expected one of api, claude-cli); API key found for model 'gpt-5-mini'"
    );
    assert_eq!(
        resolve_backend(&test_config("gpt-5-mini"), Role::Query).reason,
        "API key found for model 'gpt-5-mini' (same as index)"
    );
    let c = Config { query_model: Some("gpt-4o".into()), ..test_config("gpt-5-mini") };
    assert_eq!(
        resolve_backend(&c, Role::Query).reason,
        "API key found for model 'gpt-5-mini' (query_model set)"
    );
}

#[test]
fn availability_describes_the_passed_resolution() {
    let _lock = env_lock();
    let _keys = keyless_guards();
    let (_dir, _path) = fake_path(false);

    // This config auto-resolves to Api (no CLI on PATH); hand it a CLI
    // resolution and the answer must be about the CLI, not the key.
    let config = test_config("gpt-5-mini");
    let cli = ResolvedBackend {
        backend: Backend::Cli(CliKind::Claude),
        model: "sonnet".into(),
        reason: String::new(),
    };
    let (ok, msg) = backend_availability(&config, &cli);
    assert!(!ok);
    assert!(msg.contains("claude-cli") && msg.contains("PATH"), "{msg}");
    let api = ResolvedBackend {
        backend: Backend::Api,
        model: "gpt-5-mini".into(),
        reason: String::new(),
    };
    let (ok, msg) = backend_availability(&config, &api);
    assert!(ok, "{msg}");
    // The wrapper agrees with itself.
    let resolved = resolve_backend(&config, Role::Index);
    assert_eq!(
        check_backend_available(&config, Role::Index),
        backend_availability(&config, &resolved)
    );
}

#[test]
fn gate_search_mode_refuses_semantic_and_downgrades_hybrid() {
    let _lock = env_lock();
    let _keys = keyless_guards();
    let (_dir, _path) = fake_path(false);
    let no_llm = Config { llm_backend: Some("api".into()), query_backend: Some("api".into()), ..Config::default() };
    assert_eq!(gate_search_mode(&no_llm, "keyword").as_deref(), Ok("keyword"));
    assert_eq!(gate_search_mode(&no_llm, "transcript").as_deref(), Ok("transcript"));
    assert_eq!(gate_search_mode(&no_llm, "hybrid").as_deref(), Ok("keyword"));
    let err = gate_search_mode(&no_llm, "semantic").expect_err("refused");
    assert!(err.contains("OPENAI_API_KEY"), "{err}");

    let with_llm = test_config("gpt-5-mini");
    assert_eq!(gate_search_mode(&with_llm, "semantic").as_deref(), Ok("semantic"));
    assert_eq!(gate_search_mode(&with_llm, "hybrid").as_deref(), Ok("hybrid"));
}
