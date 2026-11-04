<task>verify</task>
Answer the multiple-choice question using ONLY the document excerpt.
The options may paraphrase the excerpt (different words, same meaning): pick the option that the
excerpt supports best, even if its wording is not identical to the document.
Return answer_index = -1 only if NO option is supported by the excerpt, or if two or more options are
equally supported.

<document>
{source}
</document>

<question>{question_json}</question>

Return JSON: {{"answer_index": <-1..3>, "reason": "<one sentence>"}}
