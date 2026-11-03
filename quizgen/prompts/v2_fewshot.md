<task>generate</task>
Write <n>{n}</n> multiple-choice questions from the document below. Cognitive levels: {levels}.

Cognitive levels:
- remember: recall a definition or fact
- understand: explain, compare, paraphrase
- apply: use the knowledge in a new situation

Examples of GOOD questions (from a different document, for style reference only):
{examples}

<document title="{title}">
{text}
</document>

Return JSON that matches this schema:
{schema}
