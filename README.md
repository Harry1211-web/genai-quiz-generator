# GenAI Quiz Generator

Generates multiple-choice questions (4 options, answer, explanation and a **verbatim source quote**) from
lecture slides and PDFs, verifies them automatically before they reach the user, and **measures quality with
numbers**: 3 prompting strategies scored by an LLM-as-judge on a 5-criterion rubric, with the judge itself
checked by a known-defect test. Supports English and Vietnamese lectures.

> Status: done in 3 sprints - controlled generation, evaluation, packaging (UI, API, Docker). Results below,
> sprint by sprint in [Project timeline](#project-timeline), details in [`reports/summary.md`](reports/summary.md).

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

**Data split.** Development set (prompt and extraction tuning): 4 chapters of a university Cloud
Computing course. Evaluation set (used only for final numbers): 2 Data Structures slide decks and one LLM-security
handbook. Course materials are copyrighted and are not included in this repository.

## Project timeline

### Sprint 1 - controlled generation (development run)

Chapter 4 (PaaS) of the development set: 13 sections, 3 questions per section, 116 questions.
Generator gpt-6-luna, verifier qwen3.5:4b via Ollama.

| Strategy | Questions | Tokens in / out | Cost (USD) | USD / 100 questions | Format errors |
|---|---|---|---|---|---|
| v1 zero-shot | 39 | 11,050 / 11,109 | 0.0067 | 0.017 | 0 |
| v2 few-shot | 39 | 21,047 / 10,481 | 0.0073 | 0.019 | 0 |
| v3 plan -> generate | 38 | not recorded (resumed run) | - | - | 0 |

**Verification.** The first rules rejected 3 of 116 questions. Reading them showed that all 3 were good
questions (false rejections), so the rules were fixed and the same questions re-verified with `--reverify`
(no new generation cost): 116 / 116 accepted.

| Strategy | Rejected by | Why it was a false rejection | Fix |
|---|---|---|---|
| v1 | length rule (`answer_leak`) | the long correct option still had a strong distractor | `answer_leak` became a warning, not a rejection |
| v2 | Qwen verifier (`unanswerable`) | the correct option paraphrased the slide | the verifier prompt accepts paraphrases |
| v3 | quote check (`quote_not_found`) | the quote joined two real bullets | quotes are checked sentence by sentence |

**Also built in Sprint 1:** crash-safe runs (saved after every section and question, resumed without paying
twice), `--regenerate` / `--reverify`, cost and token tracking, course outputs kept out of git.

### Sprint 2 - evaluation

1. **Choosing the generator** on the development set (`reports/dev_check_v2`, each model verified by the other):
   gpt-6-luna 71 / 72 accepted, ~7 s per question; qwen3.5:4b 68 / 72, ~28 s per question and it broke prompt
   rules (NOT-questions, merged quotes). gpt-6-luna kept as the generator, qwen3.5:4b as the verifier.
2. **Fixes found by reading the outputs** (details in [`docs/DECISIONS.md`](docs/DECISIONS.md)):
   coarse fuzzy search rejecting real quotes; a Vietnamese PDF without space characters; cover, agenda and
   summary slides producing off-topic questions; NOT-questions; a question with no correct option; CSV files
   garbled in Excel; prompt and extraction versions now hashed so results are never mixed.
3. **Full run** on the evaluation set: 3 strategies x 153 questions, judged by gpt-6-luna (table at the top,
   $0.22, ~2.3 h). The same prompts run twice moved scores by up to 0.1, so only larger differences count.
4. **v3 was missing the level definitions** that v1 and v2 had: its level match was 0.65-0.67 in two runs and
   0.875 after the fix. Per level, v3 also chooses fewer *apply* questions (21 vs 47):

   | Level | v1: n, level match | v2: n, level match | v3: n, level match |
   |---|---|---|---|
   | remember | 50, 1.00 | 51, 1.00 | 45, 1.00 |
   | understand | 51, 0.61 | 51, 0.76 | 86, 0.84 |
   | apply | 47, 0.83 | 47, 0.77 | 21, 0.76 |

5. **Judge check** with planted defects (table at the top): gpt-6-luna detects 150 / 150 and keeps 29 / 30
   unchanged questions at 5/5; qwen3.5:4b without thinking is not a usable judge.
6. **Decision:** v2 few-shot (same quality as v3 within the noise, follows the requested level mix, ~40 %
   cheaper), gpt-6-luna generator, qwen3.5:4b verifier with thinking off (same verdicts, ~15x faster).

### Sprint 3 - packaging

- Streamlit review UI updated to the Sprint 2 pipeline (cost metric fixed, v2 by default, model names and the
  verifier's reason shown for rejected questions); API defaults to v2.
- Docker: the same `.env` works with and without Docker (`localhost` is rewritten to `host.docker.internal`, so the
  Ollama server on the host is used), health checks, `.dockerignore` keeps secrets and course documents out.
- Visual walkthrough notebook (`notebooks/pipeline_walkthrough.ipynb`): prompt, raw answer and cost of every step.
- 73 tests pass on Python 3.11 (CI and Docker) and 3.13; one-off migration code and unused tools removed.

## Challenges and how they were handled

| Challenge | How it was handled |
|---|---|
| Rating 150 questions by hand is too slow for one person | The LLM judge is checked automatically with planted defects (150 / 150 detected, 29 / 30 consistent) |
| Scores move between runs | The same prompts were run twice to measure the noise; only differences above ~0.1 are trusted |
| A metric can look better than it is (v3 level match) | Scores are read per level, which showed that v3 avoids *apply* questions |
| The verifier can reject good questions | Rejected questions are read by hand; three rules were fixed; `--reverify` / `--recheck-quotes` re-check without paying for generation |
| A small local model breaks prompt rules and cannot judge | Roles follow ability: gpt-6-luna generates and judges, qwen3.5:4b only verifies |
| Thinking mode is slow and can overflow the context | Thinking off for verification (same verdicts); `LLM_NUM_CTX_<n>` and a warning when an answer is cut off |
| gpt-6-luna only accepts the default temperature | Answers are validated against a schema and repaired; option order is shuffled with a fixed seed |
| Slides whose content is only an image have no text | Such slides are skipped instead of producing empty or invented questions |
| Few-shot examples could leak the test material | Examples are written from other subjects (networking, databases, operating systems) |
| Judge and generator are the same model | The judge is validated with planted defects; a stronger second judge or a small human sample is the next step |

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
EXCL="Acknowledgement|Policies for students|COURSE HANDOUT|EXAM DOMAINS"    # administrative slides
python -m eval.run_eval data/private/eval_set --configs eval/configs.luna.json --n 3 --max-sections 1 \
    --exclude-sections "$EXCL" --out reports/eval_trial                       # cheap trial run first
python -m eval.run_eval data/private/eval_set --configs eval/configs.luna.json --n 3 \
    --exclude-sections "$EXCL" --out reports/eval
# eval/configs.example.json also has the reverse configuration (qwen3.5:4b generates, gpt-6-luna verifies)
# check the LLM-as-judge without human ratings (results in reports/eval/judge_check/judge_check.md)
python -m eval.judge_check plant reports/eval --docs data/private/eval_set --judge 1 --n 30   # known defects
python -m eval.judge_check cross reports/eval --docs data/private/eval_set --judge 2          # second judge (optional)
# optional: rate reports/eval/labeling_sheet.csv by hand (eval/rubric.md), then
python -m eval.report reports/eval
```

**How the judge is checked.** No human rating was done (too slow for one person). Instead the judge is
checked in two automatic ways: *known defects* - questions it rated 5/5 are broken on purpose (wrong answer
key, two correct options, an absurd distractor, a question from another document, a wrong level label) and
it must notice each defect, while unchanged control questions measure its false alarms and its
consistency; *second judge* (optional) - another model grades the same questions and the agreement is
measured. A small human sample is the natural next step to compare the judge with a teacher on subtle cases.

Both `quizgen.pipeline` and `eval.run_eval` are crash-safe: generation is saved after every section,
verification and judging after every question. After a crash, a timeout or Ctrl+C, run the same command
again - finished work is skipped, so nothing is paid for twice. Ratings already typed into
`labeling_sheet.csv` are kept when the sheet is rebuilt (a backup is written first).

## Design decisions

See [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Data and licenses

Code: MIT. The sample lectures in `data/samples/` were written by the author (CC BY 4.0). Copyrighted course
materials used for evaluation are kept in `data/private/`, which is git-ignored.
