# Sprint 2 summary - evaluation

All numbers come from `reports/eval` (evaluation set) and `reports/dev_check_v2` (development set). The course
documents and the generated questions are not published (copyrighted slide quotes); this file contains only
aggregate numbers.

## Setup

| | |
|---|---|
| Evaluation set | 2 Data Structures slide decks + 1 LLM-security handbook: 51 sections after removing cover, agenda, summary and administrative slides |
| Development set | 4 Cloud Computing chapters (used to tune the prompts and the extraction; never used for the final numbers) |
| Generator | gpt-6-luna (model 1) |
| Verifier | qwen3.5:4b via Ollama, thinking off (model 2) |
| Judge | gpt-6-luna, 5-criterion 0/1 rubric (`eval/rubric.md`) - evaluation only, not part of the product |
| Strategies | v1 zero-shot (rules only) · v2 few-shot (+ 5 examples from other subjects, + level definitions) · v3 plan -> generate (v2 + a planning call that picks key points, quotes and levels) |
| Size | 3 questions per section: 153 questions per strategy, 459 in total |

## Results (evaluation set, judge scores)

| Strategy | Accepted by verification | Factual | Single correct | Plausible distractors | Level match | $ / 100 questions | Minutes (gen + verify + judge) |
|---|---|---|---|---|---|---|---|
| v1 zero-shot | 148 / 153 (96.7 %) | 0.97 | 0.99 | 0.72 | 0.81 | 0.040 | 44 |
| **v2 few-shot** | 149 / 153 (97.4 %) | 1.00 | 0.99 | 0.67 | 0.85 | **0.039** | 40 |
| v3 plan -> generate | 152 / 153 (99.3 %) | 0.99 | 0.99 | 0.68 | 0.88 | 0.065 | 54 |

`answerable` is 0.99-1.00 for every strategy. Scores are the share of accepted questions passing each criterion.

**How large must a difference be to count?** With ~150 questions, the 95 % interval of a share near 0.7 is about
±0.07. The same prompts were run twice with only a small extraction change in between, and the `distractors`
score of v1 moved from 0.82 to 0.72. Only differences larger than about 0.1, or seen in every run, are treated
as real.

- **Distractors** (0.67-0.72): no real difference between strategies.
- **Level match**: v1 is weaker on *understand* questions (0.61 vs 0.76-0.84). v2 and v3 are close (see below).
- **Cost**: v3 costs ~1.7x because it calls the model twice per section.

## Finding 1 - one missing definition cost v3 twenty points

v3's `level_match` was 0.65 and 0.67 in two full runs, against 0.82 and 0.85 for v2. Reading the prompts showed
that v2 defined *remember / understand / apply* but v3 did not. After adding the same definitions to v3
(hash `84c59b5d` -> `1378a2d8`), v3 reached **0.875**. The gap was larger than the run-to-run noise in both runs.

**But v3 now scores higher partly because it chooses easier levels.** v1 and v2 follow the requested mix
exactly (51 / 51 / 51); the planning step of v3 picks its own levels and avoids *apply*:

| Level | v1: n, level match | v2: n, level match | v3: n, level match |
|---|---|---|---|
| remember | 50, 1.00 | 51, 1.00 | 45, 1.00 |
| understand | 51, 0.61 | 51, 0.76 | **86**, 0.84 |
| apply | 47, 0.83 | 47, 0.77 | **21**, 0.76 |

With the same level mix, v3 and v2 are about equal (~0.87 vs ~0.84). A metric that a strategy can influence by
choosing what it is measured on has to be read per level.

## Finding 2 - checking the judge without human ratings

No human rating was done (one person, ~150 questions = several sessions). The judge was checked instead with a
**known-defect test** (`eval.judge_check plant`): questions the judge had rated 5/5 were broken on purpose and
graded again; unchanged control questions measure false alarms and consistency.

| Defect (criterion that must become 0) | gpt-6-luna | qwen3.5:4b, thinking off | qwen3.5:4b, thinking on |
|---|---|---|---|
| Answer key moved to a distractor (`factual`) | **30 / 30** | 4 / 15 | 8 / 8 |
| A distractor replaced by the correct option (`single_correct`) | **30 / 30** | 7 / 15 | - |
| A distractor replaced by an absurd phrase (`distractors`) | **30 / 30** | 1 / 15 | - |
| Question from another document (`answerable`) | **30 / 30** | 14 / 15 | - |
| "remember" question labelled "apply" (`level_match`) | **30 / 30** | 11 / 15 | - |
| Control: unchanged question still rated 5/5 | 29 / 30 | 13 / 15 | - |
| Cost / time | $0.028 for 180 gradings | free, ~10 s per grading | free, **~20 min per grading** |

- gpt-6-luna detects every planted defect and is consistent with itself (one false alarm in 30). This is the
  evidence behind trusting its scores above.
- The 4B local model without thinking is **not usable as a judge**, although it is a good verifier (choosing one
  option). Thinking fixes the defects it was tested on, but needs 4,000-8,000 reasoning tokens, a 16k context
  window and ~20 minutes per grading on the author's GPU: the run was stopped after 9 gradings.
- The second-judge comparison (`eval.judge_check cross`) was therefore not run: agreement with a judge that
  misses most planted defects would say nothing about the first judge.
- **Not measured:** whether the judge agrees with a human teacher on subtle cases (a slightly weak distractor,
  an *understand* question written as *remember*). The planted defects are obvious ones.

## Finding 3 - thinking mode depends on the task

| Task (qwen3.5:4b) | Thinking off | Thinking on |
|---|---|---|
| Verify (pick the option the excerpt supports) | 13/13 same verdicts, 6.5-7.6 s | 13/13 same verdicts, 106-111 s (~15x) |
| Judge (5 criteria incl. distractor quality) | misses most planted defects | detects them, ~20 min per grading, needs a larger context |

With Ollama's default context, thinking filled the window and the JSON answer was cut off (3 x 4,096 tokens for
one grading): the run failed silently until the context was raised. The code now warns when an answer is cut.

## Development set - which model should generate?

`reports/dev_check_v2`, 2 sections per chapter, each model verified by the other:

| Generator | Accepted | Seconds per question (generation) | $ / 100 questions |
|---|---|---|---|
| gpt-6-luna | 71 / 72 | ~5-10 | 0.025-0.057 |
| qwen3.5:4b (local) | 68 / 72 | ~22-37 | 0.007 (verification by gpt-6-luna only) |

Reading the rejected questions: the 4B model broke prompt rules that gpt-6-luna followed (NOT-questions, merged
quotes, two correct options). Its context window was not the cause (prompt token counts match gpt-6-luna's).
gpt-6-luna was kept as the generator; only that configuration was run on the evaluation set.

## Decision

| Component | Choice | Why |
|---|---|---|
| Strategy | **v2 few-shot** | same quality as v3 within the noise, follows the requested level mix, ~40 % cheaper; v1 is weaker on *understand* questions |
| Generator | gpt-6-luna | fewer rule violations, ~4x faster than the local 4B model |
| Verifier | qwen3.5:4b, thinking off | free, good at picking one option; thinking adds ~15x time for the same verdicts |
| Judge (evaluation only) | gpt-6-luna | passes the known-defect test |

## Cost of Sprint 2

| Run | Cost |
|---|---|
| Full evaluation run (459 questions, generation + verification + judge) | $0.22 (~2.3 h) |
| Known-defect test, gpt-6-luna judge | $0.03 |
| Development checks, trial run, the first full run and the v3 run before the fix (kept in `reports/eval/archive/`) | ~$0.4 |

## Limitations

- No human rating: agreement between the judge and a teacher is not measured, and the judge (gpt-6-luna) grades
  questions written by the same model (possible self-preference, not measured).
- One generator configuration on the evaluation set; the 4B generator was compared on the development set only.
- Evaluation set of 3 documents in English; Vietnamese was tested on the development set only.
- Text in images and code screenshots is not read (slides with only a picture produce no question).

## Problems found by reading outputs (details in `docs/DECISIONS.md`)

- The quote check rejected good questions (coarse fuzzy search, quoted lists) - fixed, D2.
- A Vietnamese PDF had no space characters ("vềhạtầng") - text rebuilt from character positions, D3.
- Cover, agenda, summary and closing slides produced questions about the lecture itself - dropped, D4 and D8.
- CSV files looked garbled in Excel (UTF-8 without BOM) - D9.
- v3 had no level definitions - D10.
