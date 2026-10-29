You are answering a question using session summaries. Your job is to either ANSWER the question or ROUTE to the sessions that likely contain the answer.

## Question

"{{ question }}"

## Session Summaries

{{ summaries }}

## Instructions

1. **If you can answer confidently** from the summaries:
   - Provide the answer directly
   - Set `answered: true`
   - List which sessions supported your answer

2. **If you cannot answer** but see relevant sessions:
   - Set `answered: false`
   - List the session tags that likely contain the answer (max 7)
   - Briefly explain why each is relevant

## Response Format

If you CAN answer:
```json
{
  "answered": true,
  "answer": "Your complete answer here.",
  "sources": ["S_XXXX", "S_YYYY"]
}
```

If you CANNOT answer but found relevant sessions:
```json
{
  "answered": false,
  "candidates": [
    {"tag": "S_XXXX", "why": "Discusses React vs Vue decision"},
    {"tag": "S_YYYY", "why": "Frontend framework selection session"}
  ]
}
```

If NO relevant sessions found:
```json
{
  "answered": false,
  "candidates": [],
  "note": "No sessions appear to discuss this topic"
}
```

Return only valid JSON.
