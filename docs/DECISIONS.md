# Design decisions

Questions to answer in your own words before interviews. Keep the answers short and back them with your numbers.

1. Why does this project not use RAG? When would it be needed?
2. Why is `source_quote` mandatory? How is it checked, and how was the 0.85 fuzzy-match threshold chosen?
3. Why should the verifier be a different model from the generator?
4. Why are the answer options shuffled after generation?
5. How are documents split into sections, and why ~300-1,500 tokens? (ideally: results of the chunk-size experiment)
6. Which prompting strategy won, by how much, on which criteria? Is the difference meaningful given the sample size?
7. On which criteria does the judge agree with humans, and on which does it not? Why?
8. Small local model vs large hosted model: quality, cost, latency, privacy.
9. Did LoRA help the small model catch up? Where did the training data come from, and how was leakage into the
   evaluation set prevented?
10. Uploaded documents can contain prompt injection - how is it mitigated?
11. What would you change to serve 1,000 lecturers?

## Decision log (what was changed, and the evidence that triggered it)

| # | Finding | Evidence | Change |
|---|---|---|---|
| D1 | Thinking mode of the local verifier is not worth it | 13/13 identical verdicts, ~15x slower (106-111 s vs 6.5-7.6 s per question) | Ollama models run with `think=false` by default (`LLM_THINK_<n>` to change) |
| D2 | The quote check rejected good questions | Dev check: 3 of 7 rejected questions had real quotes (items joined as `"...", "..."`; a slide typo fixed by the model, ratio 0.95 missed by the coarse scan) | Fine scan around the best positions; `", "` splits quoted items; `--recheck-quotes` re-verifies only those questions |
| D3 | Vietnamese PDF text had no spaces ("vềhạtầng") | The PDF draws a gap between words but contains no space character | Text rebuilt from character positions; a space is inserted for gaps > 0.2 x font size |
| D4 | Cover, agenda and objectives slides produced questions about the course outline | The agenda of one deck says "Single-tasking" while the slides say "Single-tenant": one generated question used the agenda term | Front-matter slides are dropped before chunking (slide decks only) |
| D5 | NOT / EXCEPT questions cannot be grounded | 3 of 7 rejected dev-check questions were "Which is NOT listed..." | Prompt rule: positive questions only |
| D6 | A question had no correct option | "Customer responsibility in SaaS" with four provider responsibilities | Prompt rule: the explanation covers every option and the model must check each one |
| D7 | Resumed runs could mix prompt or extraction versions | Prompt and extraction changed during Sprint 2 | `gen_prompt` and `sections` hashes are saved in the run metadata; a mismatch stops the run |
| D8 | Closing slides produced off-topic questions | "Which approach does the summary suggest?" generated from a "Summary / End of file" slide | Summary, practice exercises, references, links and closing slides (Thanks, Q&A, End of file) are dropped; `--update-sections` regenerates only the sections that changed |
| D9 | Text looked garbled in Excel and some slides had odd characters | The CSV was UTF-8 without a BOM (Excel read it as Windows-1252); slides contained symbol-font glyphs, zero-width spaces and soft line breaks | CSVs are written with a BOM and read with or without it (comma or semicolon); odd characters are cleaned at extraction |
| D10 | v3 had no definition of the cognitive levels (v2 had one), so "plan first" was compared unfairly | Judge `level_match`: v3 0.65 and 0.67 in two full runs vs v2 0.82 and 0.85; the v1 `distractors` score moved 0.82 -> 0.72 between the same two runs, so only differences larger than ~0.1 are trusted | The v3 plan and generate prompts now carry the same level definitions as v2 (v1 has none, by design: it is the zero-shot baseline); v3 regenerated, the previous v3 kept in `reports/eval/archive/` |
| D11 | Human rating of 150 questions is too slow for a single person | Estimated 3-4 sessions; 3 of the 5 criteria are at the ceiling (>= 0.97), where a human sample says little | The judge is checked without humans: known-defect test (`eval.judge_check plant`) and a second judge (`eval.judge_check cross`); agreement with humans is reported as not measured |
| D12 | The local 4B model cannot be the judge, so the second-judge comparison was not run | Known-defect test: qwen3.5:4b without thinking caught 1/15 absurd distractors and 4/15 wrong answer keys (gpt-6-luna: 30/30 each); a run with thinking on was too slow on the author's GPU and was stopped (not reported) | gpt-6-luna stays the judge; qwen3.5:4b stays the verifier (thinking off); a stronger second judge is future work |
| D13 | Strategy for the product | Evaluation set, judge scores: distractors 0.72 / 0.67 / 0.68 (within noise), level match 0.81 / 0.85 / 0.88 but v3 picks fewer *apply* questions and costs 1.7x; v1 is weak on *understand* (0.61) | v2 few-shot |
