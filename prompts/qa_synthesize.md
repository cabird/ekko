You are synthesizing answers from multiple coding sessions into a single coherent response. Each session below provided information relevant to the user's question.

## Question

"{{ question }}"

## Important Context

Software systems evolve over time. These sessions may span days, weeks, or months of development. When you see conflicting information:

- **More recent sessions generally reflect the current state** - code gets refactored, bugs get fixed, approaches change
- **Separate stable facts from time-sensitive state** - some things (architecture, core concepts) are durable; others (specific implementations, workarounds) may have changed
- **Note when things changed** - if an older session shows one approach and a newer session shows a different one, that's valuable context

## Session Answers

{% for answer in answers %}
---
### Session {{ loop.index }}: {{ answer.project }} ({{ answer.date }})
{% if answer.title %}- **Title:** {{ answer.title }}{% endif %}
{% if answer.summary %}- **Summary:** {{ answer.summary }}{% endif %}

**Answer from this session:**
{{ answer.answer }}

{% endfor %}

---

## Your Response

Synthesize these into one coherent answer. Structure your response as:

1. **Current answer** - What's the best answer based on the most recent/authoritative information?
2. **Evolution** (if relevant) - Did the approach or implementation change over time? When?
3. **Conflicts** (if any) - Are there unresolved contradictions? What's uncertain?

Reference sessions by project and date when attributing information (e.g., "As of the payments-api session on 2025-02-10...").
