# GenAI Quiz Generator

Generates multiple-choice questions (4 options, answer, explanation and a **verbatim source quote**) from
lecture slides and PDFs, verifies them automatically before they reach the user, and **measures quality with
numbers**: 2 models x 3 prompting strategies, blind human ratings on a 5-criterion rubric, and LLM-as-judge
agreement with humans (Cohen's kappa). Supports English and Vietnamese lectures.

> Status: Sprint 1 (development run) is done - see below. The evaluation-set table is filled after Sprint 2.

![demo](reports/demo.gif)

## Results (evaluation set, Sprint 2)

| Model x Prompt | Correct & grounded (%) | Plausible distractors (%) | Level match (%) | Auto-rejected (%) | Format errors | $/100 questions | s/lecture |
|---|---|---|---|---|---|---|---|
| gpt-6-luna x zero-shot | | | | | | | |
| gpt-6-luna x few-shot | | | | | | | |
| gpt-6-luna x plan -> generate | | | | | | | |
| qwen3.5:4b x zero-shot | | | | | | | |
| qwen3.5:4b x few-shot | | | | | | | |
| qwen3.5:4b x plan -> generate | | | | | | | |

Judge-vs-human kappa per criterion: see `reports/quality_report.md`.

**Data split.** Development set (prompt tuning, few-shot examples, LoRA data): 5 chapters of a university Cloud
Computing course. Evaluation set (used only for final numbers): 2 Data Structures slide decks and one LLM-security
handbook. Course materials are copyrighted and are not included in this repository.

## Sprint 1 results (development run)

Chapter 4 (PaaS) of the development set: 13 sections, 3 questions per section.
Generator: gpt-6-luna (only the default temperature 1 is accepted). Verifier: qwen3.5:4b via Ollama, thinking on.

**Generation**

| Strategy | Questions | LLM calls | Tokens in | Tokens out | Cost (USD) | USD / 100 questions | Format errors |
|---|---|---|---|---|---|---|---|
| v1 zero-shot | 39 | 13 | 11,050 | 11,109 | 0.0067 | 0.017 | 0 |
| v2 few-shot | 39 | 13 | 21,047 | 10,481 | 0.0073 | 0.019 | 0 |
| v3 plan -> generate | 38 | 26 | n/a | n/a | n/a | n/a | 0 |

v3 produced 38 questions because the planning step returned only 2 key points for one section. Its token usage was
not recorded (the run was resumed from a checkpoint).

**Verification with the original rules**

| Strategy | Accepted | Rejected (label) | s / question |
|---|---|---|---|
| v1 zero-shot | 38 / 39 | 1 (`answer_leak`) | 115 |
| v2 few-shot | 38 / 39 | 1 (`unanswerable`) | 130 |
| v3 plan -> generate | 37 / 38 | 1 (`quote_not_found`) | 118 |

**Manual review of the 3 rejected questions: all 3 were false rejections**

| Strategy | Label | Set by | Why it was wrong | Fix |
|---|---|---|---|---|
| v1 | `answer_leak` | length rule | The correct option was much longer, but "Vendor lock-in" was still a strong distractor | `answer_leak` is now a warning shown to the reviewer, not a rejection |
| v2 | `unanswerable` | Qwen verifier | The correct option paraphrased the slide; the 4B model did not accept the paraphrase | The verifier prompt now accepts paraphrases (still returns -1 if several options fit) |
| v3 | `quote_not_found` | quote check | The quote combined two bullets from different places; each bullet exists in the slides | Quotes are checked sentence by sentence; the verifier sees the context of every sentence |

**After the fixes (same questions re-verified with `--reverify`, no new generation cost)**

| Strategy | Accepted | Rejected | Warnings | Rejected before, accepted now | Accepted before, rejected now | s / question |
|---|---|---|---|---|---|---|
| v1 zero-shot | 39 / 39 | 0 | 1 (`answer_leak`) | 1 | 0 | 141 |
| v2 few-shot | 39 / 39 | 0 | 0 | 1 | 0 | 137 |
| v3 plan -> generate | 38 / 38 | 0 | 0 | 1 | 0 | 162 |

The new quote rule was also replayed offline on all 116 questions: it changed exactly one result (the v3 case).

**Takeaways**

- With a strong generator, the verification step mostly produced false rejections (3 of 116, all wrong).
  A verifier has to be measured too: it can throw away good questions.
- The opposite error (wrong questions that the verifier accepts) is checked by manual review of accepted
  questions below and by blind human rating in Sprint 2.
- Verification with a 4B local model in thinking mode costs no money but takes about 2 minutes per question.

**Manual check of accepted questions**

| Strategy | Questions read | Wrong but accepted | Weak (e.g. obvious distractors) | Notes |
|---|---|---|---|---|
| v1 zero-shot | TBD | TBD | TBD | |
| v2 few-shot | TBD | TBD | TBD | |
| v3 plan -> generate | TBD | TBD | TBD | |

## Known limitations

- gpt-6-luna only accepts the default temperature (1): generation is not deterministic and the v3 planning step
  cannot run at temperature 0.
- With thinking mode on, qwen3.5:4b needs about 2 minutes per verification on a local machine.
- Slides whose content is only an image are not covered: chapter 2 of the development set has 16 of 39 such slides.
- Sprint 1 used a single chapter (116 questions): differences between strategies are trends, not significant results.
- Few-shot examples come from the development set only, never from the evaluation set.

## Architecture

```
PDF / PPTX -> extract (PyMuPDF / python-pptx, Unicode NFC)
           -> strip headers, footers and page numbers -> split by heading or slide, merge short parts (~300-1,500 tokens)
           -> [v1 zero-shot | v2 few-shot | v3 plan key points -> generate]   (JSON + Pydantic, self-repair on errors)
           -> shuffle options
           -> verify: quote exists in the document? answer leak? a second model re-answers from the excerpt only
           -> dedupe (embeddings) -> review/edit in Streamlit -> export JSON / CSV / GIFT (Moodle)
```

| Module | Purpose |
|---|---|
| `quizgen/extract.py` | Text extraction, boilerplate removal, section splitting/merging |
| `quizgen/schema.py` | Pydantic schemas for questions, key points, verifier and judge outputs |
| `quizgen/llm.py` | One client for any OpenAI-compatible endpoint (OpenAI, Ollama, ...), retries, cost tracking, budget cap, fake LLM for tests |
| `quizgen/generate.py` + `prompts/` | Three prompting strategies, option shuffling |
| `quizgen/verify.py` | Source-quote check, answer-leak check, answer-back verification |
| `quizgen/dedupe.py` | Near-duplicate removal |
| `quizgen/export.py` | JSON, CSV, GIFT |
| `eval/` | Experiment runner, rubric, LLM-as-judge, Cohen's kappa, quality report, SFT data builder |
| `app/`, `api/` | Streamlit review UI, FastAPI `POST /generate` |
| `notebooks/lora_finetune_colab.py` | Optional LoRA fine-tuning of a small model |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                                   # LLM_MODEL=fake by default: zero-cost dry run
pytest -q
python -m quizgen.pipeline data/samples/transport_layer_tcp_udp.md --n 2 --strategy v3
streamlit run app/streamlit_app.py
uvicorn api.main:app --reload                          # POST /generate
docker compose up --build                              # API on :8000, UI on :8501
```

To use real models, edit `.env` (`LLM_*` generator, `VERIFY_*` verifier, `JUDGE_*` judge). Any OpenAI-compatible
endpoint works, including a local Ollama server (`http://localhost:11434/v1`). Recommended setup: a hosted API model
generates and judges, a local model verifies (free), and roles are swapped when the small model is the generator so
no configuration verifies its own output.

## Running the experiment

```bash
cp eval/configs.example.json eval/configs.json         # 2 generators, each with the other model as verifier
python -m eval.run_eval data/private/eval_set --n 3 --max-sections 1   # cheap trial run first
python -m eval.run_eval data/private/eval_set --n 3
# rate reports/eval/labeling_sheet.csv by hand following eval/rubric.md (blind: model/strategy hidden)
python -m eval.report reports/eval
```

Both `quizgen.pipeline` and `eval.run_eval` are crash-safe: generation is saved after every section,
verification and judging after every question. After a crash, a timeout or Ctrl+C, run the same command
again - finished work is skipped, so nothing is paid for twice. Ratings already typed into
`labeling_sheet.csv` are kept when the sheet is rebuilt (a backup is written first).

## Progress

**Sprint 1 - controlled generation (development set)**

- [x] Real models: gpt-6-luna generates, qwen3.5:4b verifies
- [x] Few-shot examples written by the author from the development set
- [x] Development set (5 Cloud Computing chapters) and evaluation set (2 Data Structures decks + LLM-security handbook) fixed
- [x] v1, v2 and v3 run on chapter 4
- [x] Manual review of rejected questions; 3 verification rules fixed and re-verified
- [x] Crash-safe runs: progress saved after every step, `--regenerate` and `--reverify` without paying twice
- [x] Run outputs (verbatim slide quotes) removed from git history and ignored
- [ ] `pytest -q` passes locally (43 tests)
- [ ] Pushed to GitHub, CI green
- [ ] Qwen thinking mode turned off for verification, seconds per question re-measured
- [ ] 20-30 accepted questions read by hand (table above)
- [ ] `scripts/check_sections.py` run on the other 4 development chapters and the 3 evaluation files
- [ ] `quiz.gift.txt` checked (imported into Moodle if available)

**Sprint 2 - evaluation**

- [ ] Trial run on the evaluation set (`--max-sections 1`)
- [ ] Full run: 2 generators x 3 strategies, each verified by the other model, judged by the same judge
- [ ] At least 150 questions rated blind by hand; 30 rated by a second person (human-human kappa)
- [ ] Quality table and judge-human kappa (`python -m eval.report`)
- [ ] Streamlit demo GIF, README results table filled

## Design decisions

See [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Data and licenses

Code: MIT. The sample lectures in `data/samples/` were written by the author (CC BY 4.0). Copyrighted course
materials used for evaluation are kept in `data/private/`, which is git-ignored.
