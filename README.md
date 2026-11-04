# GenAI Quiz Generator

Generates multiple-choice questions (4 options, answer, explanation and a **verbatim source quote**) from
lecture slides and PDFs, verifies them automatically before they reach the user, and **measures quality with
numbers**: 2 models x 3 prompting strategies, blind human ratings on a 5-criterion rubric, and LLM-as-judge
agreement with humans (Cohen's kappa). Supports English and Vietnamese lectures.

> ⚠️ Starter version: the result tables below are empty - they are filled in after running the real experiments.

![demo](reports/demo.gif)

## Results

| Model x Prompt | Correct & grounded (%) | Plausible distractors (%) | Level match (%) | Auto-rejected (%) | Format errors | $/100 questions | s/lecture |
|---|---|---|---|---|---|---|---|
| Large x zero-shot | | | | | | | |
| Large x few-shot | | | | | | | |
| Large x plan -> generate | | | | | | | |
| Small x zero-shot | | | | | | | |
| Small x few-shot | | | | | | | |
| Small x plan -> generate | | | | | | | |

Judge-vs-human kappa per criterion: see `reports/quality_report.md`.

**Data split.** Development set (prompt tuning, few-shot examples, LoRA data): 5 chapters of a university Cloud
Computing course. Evaluation set (used only for final numbers): 2 Data Structures slide decks and one LLM-security
handbook. Course materials are copyrighted and are not included in this repository.

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

## Design decisions

See [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Data and licenses

Code: MIT. The sample lectures in `data/samples/` were written by the author (CC BY 4.0). Copyrighted course
materials used for evaluation are kept in `data/private/`, which is git-ignored.
