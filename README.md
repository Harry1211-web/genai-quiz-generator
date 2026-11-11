# GenAI Quiz Generator

Generates multiple-choice questions (4 options, answer, explanation and a **verbatim source quote**) from
lecture slides and PDFs, verifies them automatically before they reach the user, and **measures quality with
numbers**: 3 prompting strategies scored by an LLM-as-judge on a 5-criterion rubric, with the judge itself
checked by a known-defect test. Supports English and Vietnamese lectures.

> Status: evaluation (Sprint 2) and packaging (Sprint 3: UI, API, Docker) are done - results below, details in
> [`reports/summary.md`](reports/summary.md).

## Results (evaluation set, Sprint 2)

Generator gpt-6-luna, verifier qwen3.5:4b (local, thinking off), judge gpt-6-luna. 153 questions per strategy.

| Strategy | Accepted by verification | Factual | Single correct | Plausible distractors | Level match | $ / 100 questions |
|---|---|---|---|---|---|---|
| v1 zero-shot | 96.7 % | 0.97 | 0.99 | 0.72 | 0.81 | 0.040 |
| **v2 few-shot (chosen)** | 97.4 % | 1.00 | 0.99 | 0.67 | 0.85 | 0.039 |
| v3 plan -> generate | 99.3 % | 0.99 | 0.99 | 0.68 | 0.88 | 0.065 |

Differences under ~0.1 are within run-to-run noise (measured by running the same prompts twice). v3's higher
level match comes partly from choosing fewer *apply* questions; per level, v2 and v3 are equal and v3 costs 1.7x.

**Judge check (known defects planted in questions it had rated 5/5):**

| Judge | Wrong answer key | Two correct options | Absurd distractor | Question from another document | Wrong level label | Control kept 5/5 |
|---|---|---|---|---|---|---|
| gpt-6-luna | 30/30 | 30/30 | 30/30 | 30/30 | 30/30 | 29/30 |
| qwen3.5:4b, thinking off | 4/15 | 7/15 | 1/15 | 14/15 | 11/15 | 13/15 |

No human rating was done: agreement with a teacher on subtle cases is not measured.

**Data split.** Development set (prompt and extraction tuning): 4 chapters of a university Cloud
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

**Wrong questions that the verifier accepts** are measured in Sprint 2 instead: the judge grades every accepted
question, and the judge itself is checked with planted defects (see Results).

## Known limitations

- gpt-6-luna only accepts the default temperature (1): generation is not deterministic and the v3 planning step
  cannot run at temperature 0.
- With thinking mode on, qwen3.5:4b needs about 2 minutes per verification on a local machine.
- Slides whose content is only an image are not covered: chapter 2 of the development set has 16 of 39 such slides.
- Sprint 1 used a single chapter (116 questions): differences between strategies are trends, not significant results.
- Few-shot examples come from other subjects (networking, databases, operating systems), never from the
  development or evaluation documents.
- No human rating in Sprint 2: the judge is checked with planted defects only, and it grades questions written
  by the same model (self-preference not measured).
- The 4B local model is a good verifier but not a usable judge without thinking; a judge run with thinking was too
  slow on the author's GPU and is not reported.

## Architecture

```
PDF / PPTX -> extract (PyMuPDF / python-pptx, Unicode NFC)
           -> strip headers, footers, page numbers, cover / agenda / summary / closing slides
           -> split by heading or slide, merge short parts (~300-1,500 tokens)
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
| `eval/` | Experiment runner, rubric, LLM-as-judge, Cohen's kappa, quality report, judge check |
| `app/`, `api/` | Streamlit review UI, FastAPI `POST /generate` |
| `notebooks/pipeline_walkthrough.ipynb` | Step-by-step visual walkthrough: pages -> sections -> generator -> verifier -> judge, cost per step, charts of a finished evaluation run |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                                   # without LLM_MODEL_<n>: fake model, zero-cost dry run
pytest -q
python -m quizgen.pipeline data/samples/transport_layer_tcp_udp.md --n 2 --strategy v2
streamlit run app/streamlit_app.py
jupyter lab notebooks/pipeline_walkthrough.ipynb       # visual walkthrough (FakeLLM by default: free)
uvicorn api.main:app --reload                          # POST /generate
docker compose up --build                              # API on :8000, UI on :8501
```

**Docker.** The containers read the same `.env` (passed with `env_file`, never copied into the image).
`localhost` in a base URL is rewritten to `host.docker.internal`, so an Ollama server running on the host is
used without changing `.env`; `.dockerignore` keeps `.env`, `data/private/` and `reports/` out of the build.

`.env` declares each model once, by number: `LLM_MODEL_<n>`, `LLM_BASE_URL_<n>`, `LLM_API_KEY_<n>`,
`LLM_PRICE_IN_<n>`, `LLM_PRICE_OUT_<n>`. Roles are not stored there: by default model 1 generates and judges and
model 2 verifies (`DEFAULT_ROLES` in `quizgen/llm.py`); switch them with `--generator 2 --verifier 1`, or per
configuration in `eval/configs.json`. Any OpenAI-compatible endpoint works; a base URL on port 11434 is treated as
a local Ollama server, called through its native API with thinking off. In the experiment the two models swap roles,
so no configuration verifies its own output.

## Running the experiment

```bash
cp eval/configs.example.json eval/configs.json         # 2 generators, each with the other model as verifier
python -m eval.run_eval data/private/eval_set --n 3 --max-sections 1   # cheap trial run first
python -m eval.run_eval data/private/eval_set --n 3
# check the LLM-as-judge without human ratings (results in reports/eval/judge_check/judge_check.md)
python -m eval.judge_check plant reports/eval --docs data/private/eval_set --judge 1 --n 30   # known defects
python -m eval.judge_check cross reports/eval --docs data/private/eval_set --judge 2          # second judge
# optional: rate reports/eval/labeling_sheet.csv by hand (eval/rubric.md), then
python -m eval.report reports/eval
```

**How the judge is checked.** No human rating was done (too slow for one person). Instead the judge is
checked in two automatic ways: *known defects* - questions it rated 5/5 are broken on purpose (wrong answer
key, two correct options, an absurd distractor, a question from another document, a wrong level label) and
it must notice each defect, while unchanged control questions measure its false alarms and its
consistency; *second judge* - another model grades the same questions and the agreement is measured. This
shows whether the judge detects errors and how stable it is, but not whether it agrees with a human
teacher: that remains a limitation.

Both `quizgen.pipeline` and `eval.run_eval` are crash-safe: generation is saved after every section,
verification and judging after every question. After a crash, a timeout or Ctrl+C, run the same command
again - finished work is skipped, so nothing is paid for twice. Ratings already typed into
`labeling_sheet.csv` are kept when the sheet is rebuilt (a backup is written first).

## Design decisions

See [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Data and licenses

Code: MIT. The sample lectures in `data/samples/` were written by the author (CC BY 4.0). Copyrighted course
materials used for evaluation are kept in `data/private/`, which is git-ignored.
