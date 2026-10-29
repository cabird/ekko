//! The server instructions string handed to clients on `initialize`.
//!
//! It opens with a version banner, then tells the client when this archive
//! is worth consulting (historical context) and when it is not (current file
//! contents, git history), how directory scoping works, and which tools are
//! `[FAST]` versus `[LLM]`.
//!
//! The tool-tier lists must name only tools ekko actually exposes — a client
//! told about a tool that does not exist will try to call it. The tests
//! below pin both the banner and that constraint.

/// The instructions string, with `{version}` substituted.
pub fn instructions(version: &str) -> String {
    format!(
        "ekko v{version} - Archive of past coding sessions with AI assistants (Claude Code, VS Code Copilot, GitHub Copilot).

## WHEN TO USE THIS

Use ekko when you need HISTORICAL context:
- \"What was I working on?\" / \"Where did we leave off?\"
- \"Why did we choose X over Y?\" (decision rationale)
- \"How do we deploy this?\" / \"What was that command?\"
- \"We hit this error before - what fixed it?\"
- Recovering code/commands that weren't committed
- \"What did we try that didn't work?\" (negative knowledge)

## WHEN NOT TO USE THIS

Do NOT use for CURRENT state:
- Reading current file contents → use file tools (Read, Glob)
- Git history → use git commands
- Current project structure → use file tools

## AUTOMATIC SCOPING

When working in a directory, use the `directory` parameter to scope queries.
Sessions from that project will be prioritized.

## TOOL SPEED

Tools are marked [FAST] or [LLM] in their descriptions:
- **[FAST]** - Database queries, instant results
- **[LLM]** - Uses AI for reasoning/synthesis, takes 2-5 seconds

Use [FAST] tools for direct lookups (session lists, transcript reading).
Use [LLM] tools when you need reasoning, synthesis, or natural language answers.
[LLM] tools need a completion backend: an API key (e.g. OPENAI_API_KEY) or a
logged-in Claude Code CLI (`llm_backend: claude-cli`, slower per call). If neither
is configured, they return an error with setup instructions. [FAST] tools always work.

## TOOL TIERS

**Tier 1 - Quick Context (summaries):**
- `recent_activity` - Recent sessions, key files touched
- `search_sessions` - Find sessions by keyword or concept

**Tier 2 - Medium Detail (snapshots):**
- `get_session_snapshot` - Decisions, commands, architecture

**Tier 3 - Full Detail (transcripts):**
- `read_transcript` - Flexible transcript access: info, head, tail, lines, grep, full, save

**Cross-Tier:**
- `ask_question` - Answers from summaries if possible, falls back to transcripts
"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banner_carries_the_version_and_the_summary_sentence() {
        let text = instructions("9.9.9");
        assert!(text.starts_with(
            "ekko v9.9.9 - Archive of past coding sessions with AI assistants \
             (Claude Code, VS Code Copilot, GitHub Copilot).\n"
        ));
    }

    #[test]
    fn dropped_tools_and_sections_are_absent() {
        let text = instructions("0.1.0");
        for gone in [
            "AGENT-ASSISTED INDEXING",
            "index_sessions",
            "submit_index_result",
            "find_sessions_by_file",
            "ask_session",
            "ask_sessions",
            "get_file_history",
        ] {
            assert!(!text.contains(gone), "instructions still mention {gone}");
        }
    }

    #[test]
    fn kept_sections_survive_verbatim() {
        let text = instructions("0.1.0");
        for kept in [
            "## WHEN TO USE THIS",
            "## WHEN NOT TO USE THIS",
            "## AUTOMATIC SCOPING",
            "## TOOL SPEED",
            "## TOOL TIERS",
            "- \"What was I working on?\" / \"Where did we leave off?\"",
            "- Reading current file contents → use file tools (Read, Glob)",
            "[LLM] tools need a completion backend: an API key (e.g. OPENAI_API_KEY) or a",
            "**Cross-Tier:**",
            "- `ask_question` - Answers from summaries if possible, falls back to transcripts",
        ] {
            assert!(text.contains(kept), "instructions lost {kept:?}");
        }
        assert!(text.ends_with("falls back to transcripts\n"));
    }
}
