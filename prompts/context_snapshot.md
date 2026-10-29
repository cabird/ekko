# Context Snapshot Prompt

You are generating a "Context Snapshot" for a NEW AI assistant that will work in this codebase after reading a prior session transcript.

This is NOT a task-completion plan. Do NOT propose new features or next steps. Focus on transferable knowledge and orientation - helping someone understand the codebase context from this session.

## Input Context

**Project:** {{project_path}}
**CLI Source:** {{cli_source}}
**Git Branch:** {{git_branch}}
**Session Date:** {{session_date}}

## Your Job

1. Extract and normalize the durable context a new assistant needs: repo structure, key modules, APIs, data models, commands, environments, constraints, and decisions made.
2. Separate what is explicitly stated vs inferred vs uncertain using tags: [Explicit], [Inferred], [Uncertain]
3. Capture "where we left off": what changed, what was attempted, and what remains unclear—without turning it into a todo list.
4. Produce a context snapshot that can be pasted into another AI tool as context.

## Output Rules

- Use the EXACT output format below
- Be concise but comprehensive; target 400-700 tokens
- If you cite code, include file paths and short excerpts only (no huge blocks)
- If information is missing, write it under "Unknowns" rather than guessing
- Use consistent names for components (choose the most common name from the transcript)
- If the session mainly *recalled or discussed prior work* (e.g. via a session-history tool or by reading old transcripts), the title and "What This Session Covered" must say so, and that work must be attributed to the earlier session ("recalled that a prior session merged the v4 redesign"), not to this one. Do not invent file paths the recall did not explicitly name.

## Output Format

```markdown
# Context Snapshot: {{brief_title}}

## Quick Start
- **Project:** [path]
- **Key files:** [3-5 most important files]
- **Main focus:** [one sentence]
- **Biggest gotcha:** [if any]

## What This Session Covered
[2-3 bullets summarizing the session focus]

## Codebase Landmarks
[Key files/modules with their purpose]
- `path/to/file` — purpose — key interfaces/exports

## Architecture & Data Flow
[How things connect, A → B → C style]

## Decisions & Rationale
[Important choices made and why]
- Decision — Why — [Explicit/Inferred]

## Gotchas & Sharp Edges
[Things that caused issues or would trip someone up]

## Current State
- What was changed/implemented
- What was attempted but incomplete
- What remains unverified

## Unknowns & Questions
[What info is missing or uncertain - prioritized]

## Files to Read First
[Ordered list of files to read to understand this area]
1. `path/to/file` — why read this
2. ...
```

## Transcript

<transcript>
{{transcript}}
</transcript>

Generate the context snapshot now.
