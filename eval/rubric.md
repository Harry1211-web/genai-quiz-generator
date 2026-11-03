# Human rating guide (rubric v1)

Each question is scored on 5 criteria, **0 or 1** each. Rate in `reports/eval/labeling_sheet.csv`.
Do not open `key.csv` while rating (blind rating, so no configuration is favoured).
Read `source_quote` and, if needed, open the original document at the right section (`section`).

| Criterion | 1 point when | 0 points when |
|---|---|---|
| `factual` | The question and the answer marked correct are **correct** according to the document | The marked answer is actually wrong, or the question contains false information |
| `answerable` | Someone who read the section carefully can answer it without outside knowledge | It requires knowledge not present in the document |
| `single_correct` | Exactly one option is correct | Two or more options could be considered correct, or none is |
| `distractors` | All 3 distractors are wrong but **plausible** to a learner who has not mastered the material | A distractor is absurd, a giveaway, grammatically inconsistent with the question, or the correct answer is easy to spot (much longer, different style) |
| `level_match` | Matches the requested level (`level`) | e.g. labelled `apply` but only asks to recall a definition |

Levels: **remember** = recall a definition or fact · **understand** = explain, compare, say why ·
**apply** = use the knowledge in a new situation that is not stated verbatim in the document.

## Borderline examples (add your own as you meet hard cases)

- "Default port of HTTPS?" with distractors `80, 22, 8080` -> distractors = 1 (all real ports, easy to confuse).
- Same question with distractors `cat, 3.14, blue` -> distractors = 0.
- "Why is UDP suitable for voice calls?" labelled `remember` -> level_match = 0 (this is understand).
- Correct answer "UDP does not guarantee ordering", distractor "UDP has no mechanism to keep packets in order"
  -> single_correct = 0.

## Procedure

1. Rate a trial batch of 20 questions, add the hard cases to the examples above, then continue.
2. Rate at least 150 questions over several sessions (at most 50 per session to avoid fatigue).
3. Ask a second person to rate 30 questions independently -> human-human kappa (`python -m eval.kappa`).
   If human-human kappa is low, the rubric is unclear: fix the rubric before trusting any other number.
