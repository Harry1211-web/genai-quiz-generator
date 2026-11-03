<task>verify</task>
Answer the multiple-choice question using ONLY the document excerpt. If the excerpt does not contain
enough information to pick exactly one option, return answer_index = -1.

<document>
{source}
</document>

<question>{question_json}</question>

Return JSON: {{"answer_index": <-1..3>, "reason": "<one sentence>"}}
