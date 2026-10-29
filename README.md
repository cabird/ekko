# ekko

A single-binary MCP server that gives AI coding assistants access to your past coding
sessions. It reads the transcript directories that Claude Code, VS Code Copilot,
GitHub Copilot CLI and OpenAI Codex already write, indexes them into SQLite, and exposes
search, browse and Q&A tools over stdio.

It ships as a single static binary with no runtime to install: transcripts are read in
place, everything it derives lives in one SQLite file under its own data directory, and
the whole thing runs locally.

## Tools

Nine tools, marked `[FAST]` (a SQLite query — instant) or `[LLM]` (needs a completion
backend — an API key, or a logged-in Claude Code CLI; see [Configuration](#configuration)).

| Tool | What it does |
|---|---|
| `recent_activity` | `[FAST]` Recent sessions with titles, summaries and key files |
| `search_sessions` | `[FAST]`/`[LLM]` Search sessions — `keyword`, `transcript`, `semantic` or `hybrid` mode |
| `get_session_snapshot` | `[FAST]` Decisions, commands, architecture and files for one session |
| `read_transcript` | `[FAST]` Full transcript access — `info`, `head`, `tail`, `lines`, `grep`, `full`, `save` |
| `ask_question` | `[LLM]` Natural-language question answered from summaries, falling back to transcripts |
| `list_projects` | `[FAST]` Projects with session counts and last-activity times |
| `sessions_in_directory` | `[FAST]` Sessions recorded in (or under) a directory |
| `get_status` | `[FAST]` Index counts, indexer state, LLM config, token costs |
| `export_telemetry` | `[FAST]` Dump the telemetry table as JSONL under `<data dir>/exports/` |

Indexing is not exposed as a tool — it runs in the background and as `ekko index`. The
browser UI is `ekko web` (below).

## Build

```bash
cargo build --release       # target/release/ekko
cargo test                  # ~500 tests, no network, no API keys needed
```

Rust 1.85+ (edition 2024). SQLite is bundled; TLS is rustls, so there is no OpenSSL
dependency.

## Subcommands

| Command | What it does |
|---|---|
| `ekko mcp` | Run the MCP server over stdio (the default when no subcommand is given) |
| `ekko web` | Serve the browser UI on `127.0.0.1:3556` (`--port`, `--open`) — see [Web UI](#web-ui) |
| `ekko check` | Health checks — platform, API keys / CLI backend, a live test call per backend, prompts, database, session discovery |
| `ekko index` | Index everything now, in parallel, then exit |

The server is started by whichever AI client you configure it in; you don't run it by hand.

### `ekko index`

The running server indexes in the background, but deliberately slowly: it waits 60s after
launch, then does at most 10 sessions per 5-minute cycle with one LLM call at a time, so a
normal session stays cheap and quiet. `ekko index` does the same work in the foreground and
in parallel, then exits — use it for a fresh database, after a long gap, or any time you'd
rather not wait hours for the background loop to catch up.

```bash
ekko index                 # scan, FTS, keywords, summaries, snapshots
ekko index --workers 16    # more LLM calls in flight (default: config max_workers, 7)
ekko index --no-llm        # scan + FTS + keywords only; no LLM backend needed, no cost
ekko index --force         # redo summaries/snapshots that are already up to date
```

| Flag | Effect |
|---|---|
| `--workers N` | Sessions summarized/snapshotted concurrently. Default: `max_workers` from `config.json` (7). This is the only throttle — it caps LLM concurrency. |
| `--no-llm` | Stop after the local passes (scan, transcript FTS, keyword extraction). |
| `--force` | Re-summarize and re-snapshot sessions whose summaries are already current. Bypasses staleness, not the stability window or the claims. |

It runs the *same* pipeline the background indexer runs, one session at a time through the
same claim → parse → LLM → write → release path, so it is safe to run while a server is up:
each session is claimed in SQLite before work starts and the two processes never duplicate
work. If a run is interrupted (Ctrl-C), the sessions that were in flight stay claimed for up
to 10 minutes; an immediate re-run reports them as "claimed elsewhere" and skips them until
the claims expire on their own. Sessions whose transcript changed within the last hour are skipped either way — they
are probably still being written.

The scan and the keyword pass re-read every transcript on every run (that is how a change
to keyword extraction reaches old sessions), so even a no-op run takes a few minutes on a
large archive; the LLM passes, which are the expensive part, only touch sessions that
actually need work unless you pass `--force`.

Progress lines go to stdout (one per stage and per finished session), errors to stderr, and
the usual log lines to `indexer.log` in the data directory. The closing line reports wall
time and total LLM calls, tokens and cost.

## Web UI

```
ekko web            # http://127.0.0.1:3556/
ekko web --open     # …and open it
ekko web --port 8080
```

A dependency-free single-page app compiled into the binary, over the same archive the
MCP server uses. Two ways to run it:

- **Standalone**: `ekko web`, as above.
- **Co-hosted**: set `"web_serve": true` in `config.json` and every `ekko mcp` (one per
  Claude Code session) races for the port once a minute; exactly one serves, the rest
  stay quiet, and if the host exits another takes over within a minute. Coordination is a
  SQLite lease (`leases` table) with a heartbeat, taken only *after* binding the port so a
  failed bind can never strand a lease. `get_status` reports `web.status` / `web.url`, so
  an assistant can tell you where the UI is.

Neither mode runs indexing passes (the one write: first view of a not-yet-ingested
transcript stores its capped events), so "Status" shows whatever the background indexer
has done. What it offers:

- Session list, 50 per page, grouped by date (Today / Yesterday / This week …), with
  project / source / time filters, sort (modified, created, duration, messages) and a
  word-AND filter that highlights matches (every word must appear somewhere in the
  session, in any order; `"quotes"` require a contiguous phrase); the segmented mode
  control switches to
  `keyword`, `transcript`, `semantic` or `hybrid` — the same search the
  `search_sessions` tool runs (LLM-backed modes run on Enter, not per keystroke).
- Per session: the summary, the context snapshot (rendered Markdown), the transcript as
  role-coded messages with tool calls and (optionally) tool output, and files touched
  (copy path / open in VS Code).
  `#s=<id or unique prefix>` deep-links a session; filters live in the URL too, so a
  filtered view is bookmarkable.
- Transcripts are served from the database, not the `.jsonl`: the indexer stores each
  session's structured events (capped at ingest — 10K chars of message text, 2K of tool
  input, 4K of tool output per event, with truncation flagged), so a session stays fully
  browsable after the original transcript file is cleaned up. A session the indexer
  hasn't reached yet is ingested on first view. The caps keep the whole store small
  (measured: a 121 MB transcript archive stores as ~7 MB of events).
- **Ask** runs the `ask_question` tool's Q&A over your sessions on the *query* backend;
  it is a 503, not a fake answer, when no backend is configured.
- Light and dark themes (system default, toggle in the top bar), keyboard: `/` search,
  `j`/`k` move, `1`–`4` tabs, `a` ask, `?` help.

Localhost only, no authentication — anyone who can reach the port can read the whole
archive, the same trust model as the MCP server. `web_port` in `config.json` changes
the default port.

## MCP client config

Claude Code:

```bash
claude mcp add --transport stdio --scope user ekko -- /path/to/ekko mcp
```

or, in `~/.claude.json` (user scope) / `.mcp.json` (project scope):

```json
{
  "mcpServers": {
    "ekko": {
      "type": "stdio",
      "command": "/path/to/target/release/ekko",
      "args": ["mcp"]
    }
  }
}
```

VS Code Copilot uses the same shape under `"servers"` in `.vscode/mcp.json`; GitHub
Copilot CLI uses `"mcpServers"` in `~/.copilot/mcp-config.json`.

## Configuration

Settings live in `config.json` in the data directory. Keeping keys there avoids putting
them in MCP client config files that may be shared or checked in. A commented starting
point is [`config.example.json`](config.example.json); the minimal API setup is:

```json
{
  "OPENAI_API_KEY": "sk-...",
  "llm_model": "gpt-5-mini",
  "llm_reasoning_effort": "low"
}
```

and the no-API-key setup (Claude Code subscription) is just:

```json
{
  "llm_backend": "claude-cli"
}
```

| Key | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | — |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key (used when the endpoint is set) | — |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint, e.g. `https://myres.openai.azure.com`. When set, OpenAI-family models (`gpt*`, `o*`) are sent to this resource via its `/openai/v1/chat/completions` surface; the model name must match an Azure *deployment* name | — |
| `AZURE_OPENAI_API_VERSION` | Optional `api-version` query parameter for the Azure v1 surface (e.g. `preview`); usually not needed | — |
| `llm_backend` | How completions are obtained: `api` (HTTP with an API key) or `claude-cli` (drive Claude Code's `claude -p` on a subscription — see below). Unset = auto-detect | auto |
| `llm_model` | Model for summaries and snapshots (and for queries unless overridden) | `gpt-5-mini` (API) / `sonnet` (claude-cli) |
| `query_backend` | Backend for interactive use — `ask_question`, semantic/hybrid search — when it should differ from indexing (e.g. fast API for queries, subscription CLI for the background indexer) | same as `llm_backend` |
| `query_model` | Model for interactive use | `llm_model` if the query backend can serve it, else that backend's default |
| `llm_reasoning_effort` | Reasoning effort, dropped for models that don't accept it | `low` |
| `cli_timeout_secs` | Whole-call timeout for CLI backends | `600` |
| `web_port` | Localhost port for the web UI | `3556` |
| `web_serve` | Host the web UI from inside `ekko mcp` (lease-elected, one instance at a time) | `false` |
| `max_workers` | Concurrent LLM calls for `ekko index`. An explicit value applies to every backend — setting it (e.g. `5`) lifts the cautious claude-cli default of 1 | `7` (API) / `1` (claude-cli) |
| `database_path` | Override the SQLite file location | `<data dir>/sessions.db` |
| `session_paths` | Override transcript discovery, per source | platform defaults |

Environment variables win over `config.json`:

| Variable | Effect |
|---|---|
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | API keys (ekko calls these two providers) |
| `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION` | Route OpenAI-family models through an Azure OpenAI resource |
| `EKKO_BACKEND` | Override `llm_backend` |
| `EKKO_QUERY_BACKEND`, `EKKO_QUERY_MODEL` | Override `query_backend` / `query_model` |
| `EKKO_MODEL` | Override `llm_model` |
| `EKKO_REASONING` | Override `llm_reasoning_effort` |
| `EKKO_LOG` | Log level for stderr logging (e.g. `debug`) |
| `EKKO_LOG_FILE` | Relocate the indexer log away from `<data dir>/indexer.log` |
| `EKKO_DEBUG` | `1` to keep debug lines in the indexer log |
| `XDG_DATA_HOME` | Relocate the whole data directory (Linux/WSL) |

A `.env` file in the working directory is loaded at startup. Without any API key or CLI
backend the `[FAST]` tools all work; only summaries, snapshots, semantic search and Q&A
are disabled.

### No API key? Use your Claude Code subscription

If `claude` is on your `PATH` and you are logged in, ekko can get its completions by
running `claude -p` instead of calling an API. Auto-detection picks this whenever no API
key is configured; set `"llm_backend": "claude-cli"` to force it. Each call runs

```
claude -p --no-session-persistence --tools "" --output-format json --model sonnet
```

with the prompt on stdin, from an empty working directory under the data dir, so the
run sees none of your project instructions and leaves no transcript behind for ekko to
re-index. `llm_model` accepts Claude Code's aliases (`sonnet`, `haiku`, `opus`) or a
full model id; the default is `sonnet` because `haiku` misreads very long sessions
(`docs/cli-backend/EXPERIMENT.md`). Expect 15–60 s per call — Claude Code's own startup
dominates — and one call at a time unless you raise `max_workers`. Reported costs are
notional: subscription runs are not billed per call. `ekko check` and `get_status`
show which backend resolved and why.

Parallel CLI calls work: concurrent `claude -p` processes are independent, and bulk
indexing speeds up near-linearly with `--workers N` or an explicit `"max_workers"` in
`config.json` (measured: 4 sessions in 75 s at 4 workers vs ~45 s each serially, no
auth or rate-limit errors). The 1-worker default is deliberate caution — each call
consumes your subscription's usage window, and parallel runs consume it that much
faster — not a hard limit.

Because every CLI call carries Claude Code's startup cost, a common split is the CLI
for the background indexer and an API for the interactive tools:

```json
{
  "llm_backend": "claude-cli",
  "llm_model": "sonnet",
  "query_backend": "api",
  "query_model": "gpt-5-mini",
  "OPENAI_API_KEY": "sk-..."
}
```

## What gets indexed

Only interactive sessions. Claude Code stamps each transcript with how it was
started — `cli` (terminal), `claude-desktop`, or `sdk-cli` for `claude -p` one-shots
and Agent SDK programs — and ekko skips `sdk-cli` transcripts at scan time: they are
scripts, cron jobs and CI runs, not a person's work. A previously indexed one-shot is
removed on the next scan. Codex threads are treated the same way: `codex exec` one-shots
(source `exec`) and the per-subagent rollouts a thread spawns (source `subagent`) are
skipped, so only the thread you typed into is indexed. Transcripts without the field
(older versions, other CLIs) are treated as interactive. `ekko index` reports the skipped
count on its `scan:` line.

Codex injects environment context, plugin lists and `AGENTS.md` as user-role messages;
ekko keeps only what you typed (Codex's `content_item_kinds` marks it) for first prompts,
message counts and conversation text. Its `exec` tool call is a JavaScript program, so
ekko lifts the shell command out of it for tool timelines, and reads the file names out
of `apply_patch` envelopes for the files-touched list.

## Data directory

| Platform | Location |
|---|---|
| Linux / WSL | `~/.local/share/ekko/` |
| Windows | `%LOCALAPPDATA%\ekko\` |
| macOS | `~/Library/Application Support/ekko/` |

It holds `sessions.db`, `config.json`, `indexer.log` and `exports/`.

Transcripts themselves are only ever read, from the CLIs' own directories
(`~/.claude/projects`, `~/.copilot/session-state`, VS Code's `workspaceStorage`,
`~/.codex/sessions` and `~/.codex/archived_sessions`, plus the same two under
`$CODEX_HOME` when it points somewhere else). On WSL the Windows-side copies under
`/mnt/c/Users/<you>/` are discovered too.

A background indexer starts 60 seconds after the server does, then rescans every 5
minutes, picking up sessions idle for at least an hour. Run `ekko index` to do the same
work immediately and in parallel.

## Architecture

| Module | Purpose |
|---|---|
| `main.rs` | clap CLI — `mcp` (default), `check` and `index` |
| `mcp/` | rmcp stdio server: tool schemas, descriptions, instructions, `get_status` |
| `memory/` | The tool implementations — activity, search, transcript access, Q&A |
| `db/` | SQLite schema, FTS5 search, indexer claims, telemetry storage, stats |
| `parsers/` | Transcript parsers per source (Claude Code, Copilot CLI, VS Code Copilot) |
| `indexer.rs` | Scan, parse and upsert sessions; keyword and file-path extraction |
| `background.rs` | The background scan loop, with claim-based single-writer coordination |
| `fastindex.rs` | `ekko index` — the same per-session pipeline, driven in parallel to completion |
| `search.rs` | Keyword → transcript → semantic cascade and result reranking |
| `summarize.rs`, `snapshot.rs` | LLM summary and snapshot generation, with map-reduce for long transcripts |
| `llm.rs` | OpenAI and Anthropic clients, retry/backoff, JSON-response recovery, usage tracking |
| `prompts.rs` | minijinja rendering of the embedded `prompts/*.md` templates |
| `models.rs` | Events, session metadata, conversation and tool-call rendering |
| `config.rs`, `paths.rs` | Config loading, platform paths, transcript discovery, export sandbox |
| `tokens.rs`, `pricing.rs`, `telemetry.rs` | tiktoken counting, cost table, per-tool telemetry |
| `check.rs` | The `ekko check` health checks |
