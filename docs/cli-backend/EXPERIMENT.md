# Experiment: haiku vs sonnet via `claude -p` for ekko summarization

Date: 2026-08-23. Claude Code 2.1.241. Prompts are ekko's real
`summarize_prompt` / `snapshot_prompt` output for five real sessions
(generated with `cargo run --example dump_prompt <transcript.jsonl> <out_dir>`).
Invocation: `claude -p --no-session-persistence --tools "" --output-format json --model <m> < prompt`.
Cost is Claude Code's notional `total_cost_usd` (subscription runs are not billed per call).
Raw outputs were kept out of the repo (they contain client material).

## Results

| session  | prompt tokens | kind      | haiku s / $   | sonnet s / $  | haiku quality | sonnet quality |
|----------|--------------:|-----------|---------------|---------------|---------------|----------------|
| ce7a56bc |           567 | summarize | 22 / 0.06     | 15 / 0.14     | good          | good |
| ce7a56bc |           567 | snapshot  | 26 / 0.02     | 27 / 0.15     | good          | good |
| 72c7a446 |         3,721 | summarize | 48 / 0.04     | 20 / 0.06     | good          | good |
| 72c7a446 |         3,721 | snapshot  | 36 / 0.03     | 32 / 0.07     | good          | good |
| f41ddbfe |        21,590 | summarize | 26 / 0.06     | 20 / 0.16     | good, accurate vs known ground truth | good, accurate |
| f41ddbfe |        21,590 | snapshot  | 45 / 0.07     | 42 / 0.18     | good          | good |
| 3113f35a |       100,042 | summarize | 21 / 0.24     | 37 / 0.67     | **FAIL: invented own schema** (`session_summary.project…`) → ekko would store null title/summary | good |
| 3113f35a |       100,042 | snapshot  | 48 / 0.25     | 59 / 0.68     | good          | good |
| f99b952e |       137,637 | summarize | 16 / 0.33     | 29 / 0.89     | **FAIL: continued the conversation** (drafted the email the last turn asked for) | good |
| f99b952e |       137,637 | snapshot  | 78 / 0.35     | 53 / 0.91     | good          | good |

Follow-up with `--json-schema` (haiku only, the two failures):

| session  | result |
|----------|--------|
| 3113f35a | fixed — valid, complete summary |
| f99b952e | shape fixed, content wrong — summarized only the last ~10% of the session; the hours of work before it are missing |

## Observations

- Latency is dominated by Claude Code startup + its own system prompt:
  15–30 s even for a 600-token prompt. Prompt size matters less than
  output length. Fine for background indexing; too slow to feel snappy for
  `ask_question`.
- Every run succeeded (no rate-limit or auth errors) with 4 concurrent
  processes.
- haiku emits ```json fences; sonnet usually emits bare JSON. ekko's
  `parse_json_response` handles both.
- haiku's failures are specific to **long prompts (≥100K tokens) with a
  JSON-output instruction**: the instruction at the top of the prompt loses
  to the conversation content. Free-form markdown snapshots were fine at
  every size, and JSON summaries were fine at ≤22K tokens.
- `--json-schema` guarantees shape, not coverage: at 138K haiku still
  exhibited recency bias.
- Notional cost: sonnet ≈ 2.5–3× haiku. Irrelevant under a subscription
  except as a proxy for usage-window consumption.
- `--bare` cannot be used: it skips keychain reads and the run fails with
  "Not logged in".

## Recommendation

- **Default `claude-cli` model: `sonnet`.** It was correct on every run,
  including the two where haiku failed, at comparable latency. The cheaper
  tier is only safe below ~50–80K prompt tokens, and the sessions that most
  need a good summary are the long ones.
- Keep `llm_model` configurable; document `haiku` as a valid choice for
  users who mostly have short sessions or want to conserve their usage
  window.
- ~~Use `--json-schema` for `summarize`~~ — **withdrawn in implementation**:
  a loose `{"type":"object"}` schema made the CLI return wrapper objects
  (`{"input": …}`, a schema echo) in 3 of 4 follow-up runs, and ekko's
  callers don't pass exact schemas down. No schema is passed; the prompt +
  JSON re-ask loop is what ships (see DESIGN.md and src/llm/cli.rs).
  Per-caller schemas remain the route to smaller models on long sessions.
- Possible refinement (not now): size-tiered model choice — haiku under
  ~64K tokens, sonnet above — would recover most of the haiku savings
  without the failure mode. Worth revisiting only if usage-window pressure
  shows up in practice.
