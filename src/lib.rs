//! ekko — a single-binary MCP server and web UI over your past AI coding sessions.
//!
//! It parses the transcripts Claude Code, GitHub Copilot CLI, VS Code Copilot and
//! OpenAI Codex already write, indexes them into SQLite (full text, keywords,
//! LLM summaries and snapshots), and serves search, browse and Q&A over stdio MCP
//! or a local web UI.

pub mod background;
pub mod check;
pub mod config;
pub mod db;
pub mod fastindex;
pub mod indexer;
pub mod llm;
pub mod mcp;
pub mod memory;
pub mod models;
pub mod parsers;
pub mod paths;
pub mod pricing;
pub mod prompts;
pub mod search;
pub mod signals;
pub mod snapshot;
pub mod summarize;
pub mod telemetry;
pub mod tokens;
pub mod utils;
pub mod web;
