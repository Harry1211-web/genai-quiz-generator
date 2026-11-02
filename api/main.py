"""API: uvicorn api.main:app --reload  → POST /generate (multipart file)"""
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from quizgen.extract import load_sections
from quizgen.llm import BudgetExceeded, from_env
from quizgen.pipeline import run

app = FastAPI(title="GenAI Quiz Generator")
MAX_BYTES = 10 * 1024 * 1024


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
async def generate(file: UploadFile = File(...), strategy: str = Query("v3", pattern="^v[123]$"),
                   n: int = Query(3, ge=1, le=5), max_sections: int = Query(6, ge=1, le=20),
                   verify: bool = True):
    suffix = Path(file.filename or "x.pdf").suffix.lower()
    if suffix not in {".pdf", ".pptx", ".txt", ".md"}:
        raise HTTPException(415, "Chỉ nhận PDF, PPTX, TXT, MD")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File quá lớn (tối đa 10 MB)")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
    try:
        sections = load_sections(f.name)
        qs, stats = run(sections, from_env("LLM"), from_env("VERIFY") if verify else None,
                        n, strategy, max_sections)
    except BudgetExceeded as e:
        raise HTTPException(429, str(e)) from e
    finally:
        Path(f.name).unlink(missing_ok=True)
    return {"stats": stats, "questions": [q.to_dict() for q in qs]}
