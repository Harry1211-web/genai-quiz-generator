FROM python:3.11-slim
WORKDIR /app
# QUIZGEN_IN_DOCKER: "localhost" in .env (e.g. Ollama on the host) is rewritten to host.docker.internal
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1 QUIZGEN_IN_DOCKER=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY quizgen quizgen
COPY api api
COPY app app
COPY data/samples data/samples
RUN useradd -m appuser
USER appuser
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
