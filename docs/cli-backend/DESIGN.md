# CLI completion backends (design)

Status: implemented 2026-08-23 (`src/llm/cli.rs`, `Backend` in `src/llm.rs`).
Model choice settled by the haiku-vs-sonnet experiment (see `EXPERIMENT.md`).
Deviations from the draft are marked **Implemented:** below.

## Problem

ekko's summaries, snapshots, `ask_question`, and semantic search all go
through `LlmClient`, which only speaks HTTP to OpenAI / Azure / Anthropic
with an API key. Most users of ekko already pay for a coding-assistant
subscription (Claude Code, Codex, Gemini CLI, Copilot) and have no API key.
Those CLIs all support a non-interactive single-prompt mode, which is
exactly the shape ekko needs: every call is one user message, no system
prompt, no tools, no multi-turn.

## Decisions

- **Transport: stdin → stdout.** Not "read foo.txt, write bar.txt". A file
  round-trip turns a completion into an agentic task (tools on, multiple
  turns, permission handling, temp-file cleanup) and Claude Code's Read tool
  truncates large files, so a big chunk would be summarized partially and
  silently. Prompts up to the 200K-token chunk (~800 KB) exceed `ARG_MAX`
  as an argument but are fine on a pipe.
- **Claude CLI first**, behind an abstraction so Codex / Gemini / Copilot are
  each "verify flags, add a variant". Non-Claude backends are out of scope
  for the first cut.
- **Explicit backend selection, with a fallback when unset.** Config key
  `llm_backend`. When absent: if an API key is configured use HTTP as today;
  otherwise use the first CLI found on `PATH` (claude). `ekko check` and
  `get_status` always report which backend resolved and why.
- **Cheapest model that is reliable, configurable.** Summarization is a
  high-volume background job. The experiment (`EXPERIMENT.md`) showed haiku
  is not reliable for JSON summaries of sessions ≥100K tokens, so the
  claude-cli default is `sonnet`; HTTP backends keep `gpt-5-mini`.
  `llm_model` overrides in all cases.
- **Same `CHUNK_THRESHOLD` (200K tokens)** for CLI and HTTP backends.

## Architecture

```
summarize.rs / snapshot.rs / search.rs / check.rs
        │  call_llm / call_llm_json           (unchanged)
        ▼
LlmClient            retry loop, backoff, usage tracker, JSON extraction
        │  Completer::complete(&[Message]) -> (text, Option<RawUsage>)
        ├── HttpCompleter    existing build_body + send_once (OpenAI/Azure/Anthropic)
        └── CliCompleter     spawn process, prompt on stdin, parse stdout
              └── ClaudeCli  (later: CodexCli, GeminiCli, CopilotCli)
```

`Provider` stays as the HTTP-only enum. A new `Backend` enum selects the
completer:

```rust
pub enum Backend { Api, Cli(CliKind) }   // Api: provider still follows the model name
pub enum CliKind { Claude }              // Codex, Gemini, Copilot later
```

**Implemented:** `LlmClient::new(config, model)` stays the HTTP constructor;
`LlmClient::new_cli(config, kind, model)` is the CLI one, and
`from_config` picks via `resolve_backend(config)`. Rather than a trait
object, `LlmClient` holds `cli: Option<CliBackend>` and dispatches in the
retry loop — the retry/backoff/tracker/JSON code is shared unchanged.

### Config

```json
{
  "llm_backend": "claude-cli",          // api | claude-cli | (absent = auto); env EKKO_BACKEND
  "llm_model": "sonnet",                // optional for CLI backends; backend default otherwise
  "llm_reasoning_effort": "low",
  "cli_timeout_secs": 600,
  "max_workers": 1                      // see rate limits below
}
```

**Implemented:** an explicit `llm_model` that is an OpenAI-family name
(typically left over from an API setup) is replaced by the CLI default
when a CLI backend is in use, and the resolution reason says so — a CLI
can only serve its own vendor's models.

Backend defaults: `claude-cli` → model `sonnet` (see `EXPERIMENT.md`),
`openai` → `gpt-5-mini` (today's default), `anthropic` →
`claude-haiku-4-5`.

### Claude CLI invocation (verified against Claude Code 2.1.241)

```
claude -p --no-session-persistence --tools "" --output-format json \
       --model <model> [--effort <level>]       < prompt
```

- `--no-session-persistence` — the run is not written to
  `~/.claude/projects/…`, so ekko never indexes its own summarization runs.
  This removes the recursion hazard entirely; no indexer exclusion needed.
- `--tools ""` — no Read/Bash/etc.; the model can only answer.
- `--output-format json` — single JSON object with `result`,
  `is_error`, `usage.{input_tokens,output_tokens,cache_*}`,
  `total_cost_usd`, `duration_ms`.
- `--bare` is **not** used: it skips keychain reads and the run fails with
  "Not logged in".
- cwd: a dedicated empty directory under the ekko data dir, so project
  `CLAUDE.md` / settings are not discovered.
- `--effort` maps from `llm_reasoning_effort` when the value is one of
  low/medium/high; otherwise omitted.
- **Implemented: no `--json-schema`.** A loose `{"type":"object"}` schema
  made the CLI return wrapper objects (`{"input": …}`, a schema echo) in 3
  of 4 runs, and ekko's callers do not pass their exact schemas down.
  Prompts already demand JSON and `call_llm_json_with` re-asks on a parse
  failure; the experiment found sonnet reliable that way. Per-caller
  schemas are the route to supporting smaller models on long sessions.

### Error mapping

| Observation                                  | `LlmError`            | Retry |
|----------------------------------------------|-----------------------|-------|
| binary not on PATH                           | NotConfigured         | no    |
| `is_error: true`, result contains "Not logged in" | NotConfigured    | no    |
| result mentions usage/rate limit             | RateLimited           | yes, 60s flat (existing 429 path) |
| non-zero exit, other                         | Http{status:None,body:stderr} | yes (transient schedule) |
| timeout                                      | Timeout               | yes   |
| exit 0, unparseable JSON envelope            | InvalidResponse       | no — propagates (is_retryable excludes it) |

### Usage and cost

`--output-format json` reports real token counts and a notional
`total_cost_usd`, so the existing `UsageTracker` works unchanged. Status
output should label CLI-backend cost as "notional (subscription)" rather
than implying a bill.

### Concurrency and latency

- Process startup + Claude Code's own system prompt add ~15–25 s per call
  regardless of prompt size (measured). Fine for background indexing;
  noticeable for `ask_question`. `get_status` should surface the backend
  so the slowness is explainable.
- Subscription limits are per rolling window and opaque. Default
  `max_workers` to 1 for CLI backends (2 at most); keep today's value for
  HTTP.

### Availability check

`check_llm_available` for a CLI backend = binary resolvable on PATH.
`ekko check` keeps its "Reply with exactly: OK" live probe, which also
catches not-logged-in.

### Roles (added after first use)

One backend for everything made `ask_question` and semantic search pay the
CLI's 15–30 s startup on every interactive call. `llm::Role` splits the
configuration: `Index` (summaries, snapshots, the indexer's gate and worker
count) reads `llm_backend`/`llm_model`; `Query` (`ask_question`,
semantic/hybrid search, query telemetry) reads `query_backend`/`query_model`
and falls back to the index settings. `LlmClient::for_role` is the
constructor; the free `call_llm`/`call_llm_json` are index-only. `ekko check`
and `get_status` report both roles and probe each when they differ.

## Out of scope for v1

Codex / Gemini / Copilot variants; `--json-schema` enforcement; streaming.
