# API-only image: serves committed artifacts, no training, no database.
# Target: Hugging Face Spaces (Docker SDK, port 7860) or any container host.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FFAI_ARTIFACTS_DIR=/app/artifacts

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY requirements-api.txt pyproject.toml README.md LICENSE ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY ffai ./ffai
RUN pip install --no-cache-dir --no-deps .

COPY artifacts ./artifacts

USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health')"
CMD ["uvicorn", "ffai.serve.app:app", "--host", "0.0.0.0", "--port", "7860"]
