You are extracting an answer from a coding session transcript. This transcript was selected as potentially relevant to the user's question - your job is to find the answer if it's here, or indicate if it's not.

## Question

"{{ question }}"

## Session Context

- **Project:** {{ project }}
- **Date:** {{ date }}
{% if summary %}- **Summary:** {{ summary }}{% endif %}

## Important Guidelines

- **Only answer from what's in this transcript.** Do not use general knowledge or assumptions.
- If the transcript discusses the topic but doesn't contain a clear answer, extract what's relevant.
- If the transcript doesn't help answer the question at all, respond with exactly: `NOT_RELEVANT`

## What Makes a Good Answer

- Be specific and grounded in what the transcript says
- Include brief evidence: quote or paraphrase the relevant parts
- Mention file names, commands, or code snippets if they're part of the answer
- If the transcript shows something was tried but didn't work, that's useful context too

## Conversation

{{ conversation }}

---

Provide your answer below. Include brief supporting evidence from the transcript (quotes or paraphrases). If this conversation doesn't help answer the question, respond with exactly: `NOT_RELEVANT`
