# ── cloud_run_ingest/Dockerfile ─────────────────────────────────────────────
# Standalone ingestion service for Google Cloud Run. Build context must be
# the REPO ROOT (not this directory) so it can reuse app/core, app/db, and
# the relevant app/services modules directly:
#
#   docker build -f cloud_run_ingest/Dockerfile -t thotqen-ingest .
#
# Kept deliberately separate from the main Dockerfile — this image has none
# of the LLM provider SDKs, LangGraph, or the reranker the chat path needs,
# only what chunking + embedding + DB writes actually require.
# ─────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /code

COPY cloud_run_ingest/requirements.txt .
# CPU-only torch wheel — the default PyPI wheel bundles CUDA libs that are
# large and useless on Cloud Run's CPU-only instances.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Reuse the existing chunking/embedding/DB-write modules directly — single
# source of truth, not a reimplementation.
COPY app/db/database.py app/db/__init__.py app/db/
COPY app/core/config.py app/core/
COPY app/services/chunking.py app/services/document_service.py app/services/embeddings.py app/services/vectorstore.py app/services/

COPY cloud_run_ingest/main.py .

# Cloud Run injects $PORT (defaults to 8080) and expects the container to
# listen on it — never hardcode the port.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
