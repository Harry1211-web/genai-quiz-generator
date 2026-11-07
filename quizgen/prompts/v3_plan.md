<task>plan</task>
Read the document and list at most {k} of the most important KEY POINTS that can be tested with a question.
Each point must be different and the points should cover the whole document. For each point give a
supporting quote copied EXACTLY from the document (one sentence or bullet) and a suggested cognitive level.
Use these levels: {levels}.

Cognitive levels:
- remember: recall a definition or fact
- understand: explain, compare, paraphrase
- apply: use the knowledge in a new situation

<document title="{title}">
{text}
</document>

Return JSON that matches this schema:
{schema}
