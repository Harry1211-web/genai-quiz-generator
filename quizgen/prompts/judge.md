<task>judge</task>
You grade multiple-choice questions. Score each rubric criterion 0 or 1:
- factual: the question and the answer marked correct are factually correct according to the document
- answerable: it can be answered from the document alone (no outside knowledge needed)
- single_correct: exactly one option is correct
- distractors: all 3 distractors are wrong but plausible (not absurd, not giveaways)
- level_match: matches the requested level "{level}"
(remember = recall; understand = explain/compare; apply = use in a new situation)

<document>
{source}
</document>

<question>{question_json}</question>

Return JSON: {{"factual":0|1,"answerable":0|1,"single_correct":0|1,"distractors":0|1,"level_match":0|1,"comment":"<short>"}}
