You are summarizing a coding session for search and retrieval. An LLM will use this summary to decide whether to read the full transcript - include enough signal that someone searching for related topics would know to look here.

## Output Format

Return JSON:

```json
{
  "title": "Action-oriented title (60 chars max)",
  "summary": "Structured summary with Outcome, Goal, Approach, and optional Follow-ups sections",
  "topics": "5-10 comma-separated key concepts, technologies, patterns discussed",
  "files": ["top 5-15 file paths that were read/written, most important first"],
  "key_entities": ["specific technical terms: class names, libraries, services, patterns"],
  "tags": ["1-5 canonical tags from: bugfix, refactor, feature, testing, database, api, frontend, infra, deployment, build, config, docs, performance, security"]
}
```

## Summary Structure (150-350 words)

**Outcome:** 1-2 sentences. What changed and why it matters - the headline.

**Goal:** 1 sentence. What was being attempted (bugfix, refactor, new feature, investigation).

**Approach:** 3-10 bullets depending on session length. Key moves and decisions. For each bullet:
- Mention the *type* of activity (debugged, deployed, refactored, configured)
- Mention *technologies/tools* used (Docker, PostgreSQL, pytest)
- Mention *patterns/concepts* applied (mutex, caching, retry logic, pagination)
- Include important files inline when relevant (limit to most important, not every file)

**Follow-ups:** (optional) 1-2 bullets if there are open questions, TODOs, or known issues.

## Guidelines

- **Summarize THIS session, not work it recalls**: Some sessions mainly *retrieve or discuss past work* — querying a session-history tool (e.g. recent_activity, search_sessions, ask_question), reading old transcripts or summaries, or answering "what did we do about X?". For those, the title and Outcome must describe the retrieval itself (what was searched for, what was found), and recalled work must be framed as recalled ("found a prior session that had merged the v4 redesign"), never phrased as if it was performed in this session. A session that only *reads about* a change did not make that change.
- **Signal for retrieval**: The summary should hint at what's IN the transcript. Don't include exact commands, but "deployed Docker images to Azure using ACR push" signals that Azure Docker deployment commands are in the transcript.
- **Topic markers over exact details**: Instead of listing `docker push myimage:latest`, say "Docker image deployment to Azure Container Registry"
- **Be specific**: "Fixed redirect loop by adding mutex in AuthProvider" not "Fixed auth issues"
- **Cover all major topics**: Long sessions may need 8-10 bullets to capture everything discussed
- **No keyword dumps**: Topics should be curated concepts (5-10), not a wall of every filename
- **key_entities**: Use exact class/function names when present (e.g., `ThreadPoolExecutor`, `search_keyword`, `FastMCP`). Include libraries, frameworks, and services mentioned.
- **files**: List the most important files touched, starting with the main ones. Skip test files unless they're the focus.
- **tags**: Choose 1-5 tags that best describe the session type. Canonical tags:
  - `bugfix` - investigating or fixing bugs
  - `refactor` - restructuring code without changing behavior
  - `feature` - adding new functionality
  - `testing` - writing or fixing tests
  - `database` - schema, queries, migrations
  - `api` - API endpoints, HTTP, REST
  - `frontend` - UI, React, CSS, components
  - `infra` - Docker, CI/CD, containerization
  - `deployment` - deploying to staging/prod, releases
  - `build` - build tools, bundlers, package managers
  - `config` - configuration, env vars, feature flags
  - `docs` - documentation, comments
  - `performance` - optimization, caching, profiling
  - `security` - auth, encryption, vulnerabilities

## Example

```json
{
  "title": "Fix JWT refresh race condition in AuthProvider",
  "summary": "Outcome: Fixed redirect loops caused by multiple components triggering JWT refresh simultaneously.\n\nGoal: Debug authentication failures reported after deploying the new token refresh logic.\n\nApproach:\n- Identified race condition in `src/auth/AuthProvider.tsx` where concurrent refresh calls invalidated each other's tokens\n- Implemented mutex pattern using a shared pending promise to serialize refresh requests\n- Migrated token storage from localStorage to httpOnly cookies in `src/auth/storage.ts` for security\n- Added integration tests for concurrent auth scenarios using Jest\n\nFollow-ups: Consider adding retry logic for refresh failures due to network issues.",
  "topics": "JWT, refresh tokens, race condition, mutex, concurrency, httpOnly cookies, React context, authentication",
  "files": ["src/auth/AuthProvider.tsx", "src/auth/storage.ts", "src/auth/__tests__/AuthProvider.test.tsx"],
  "key_entities": ["AuthProvider", "localStorage", "httpOnly", "Jest", "JWT", "mutex"],
  "tags": ["bugfix", "security", "frontend"]
}
```

## Conversation

{{ conversation }}

Return only valid JSON.
