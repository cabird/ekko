# ekko — instructions for AI coding agents

This is the one file every agent reads (Codex, Copilot, Cursor, Gemini CLI read it
directly; `CLAUDE.md` points Claude Code here). Keep repo-wide rules in this file only.

ekko is a single-binary Rust MCP server that indexes AI coding-session transcripts
(Claude Code, GitHub Copilot CLI, VS Code Copilot, OpenAI Codex) into SQLite. See
`README.md` for what it does.

## Versioning — required on every behavior change

The version in `Cargo.toml` is the single source of truth, and it must identify what is
installed. The binary the owner's MCP clients run is `~/.cargo/bin/ekko`, installed with
`cargo install --path .`; `ekko --version` prints the `Cargo.toml` version. Semantic
versioning, starting at 1.2.1 (2026-09-07).

Bump the version whenever the installed binary would behave differently:

| Bump | When |
|---|---|
| patch (`x.y.Z`) | bug fixes, parser corrections, UI polish, doc-only changes need no bump |
| minor (`x.Y.0`) | new features: a new transcript source, a new MCP tool or CLI command, new config keys |
| major (`X.0.0`) | breaking changes: SQLite schema or data-dir layout that older binaries cannot read, removed or renamed tools/config keys, changed MCP output shapes |

The procedure, in order:

1. Edit `version` in `Cargo.toml`. `cargo build` refreshes `Cargo.lock`; commit both.
2. Commit the change and the bump together (or the bump as the last commit of the series).
3. Tag that commit: `git tag -a vX.Y.Z -m "vX.Y.Z"` and push tags with the branch.
4. Only then `cargo install --path .`, and confirm with `ekko --version`.

Rules:

- Never install from an uncommitted tree. Before 1.2.1 this happened and left a binary
  that no commit described.
- Never reuse or lower a version. If a bump was forgotten, bump now; do not rewrite history.
- A bump with no code change is fine; a code change that alters behavior with no bump is a
  defect — call it out in review.
- If you are unsure which level applies, choose the higher one.

## Working in this repo

- Tests: `cargo test --no-fail-fast`. Three failures are pre-existing on macOS and not yours
  to fix in passing: `utils::tests::path_under_basic`, `utils::tests::paths_equal_basic`,
  `export_telemetry_writes_into_the_export_sandbox` (path-canonicalization assumptions).
- Every parser and indexer change must be tried against real transcripts before it lands:
  `cargo run --example dump_transcript -- <cli_source> <path>` prints what would be indexed.
  Never run `ekko index` or anything that writes to the real database
  (`~/Library/Application Support/ekko`) from a test or an experiment.
- Test sandboxes pin `HOME`, `XDG_DATA_HOME`, `APPDATA`, `WIN_HOME` and unset `CODEX_HOME`;
  a new environment-driven path must be pinned there too or real data leaks into tests.
- Every parser documents the format assumptions it relies on, and any deliberate
  deviation from what a transcript format nominally allows, in its module doc or in a
  `DIVERGENCE` comment at the site. Keep that up to date when you change one.
