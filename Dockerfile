# ── Dockerfile ────────────────────────────────────────────────────────────────
# Production image for Hugging Face Spaces (UID 1000, port 7860)
# Multi-process: supervisord manages FastAPI (uvicorn) + Celery worker.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Python runtime optimisations ──────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── System dependencies ───────────────────────────────────────────────────────
# libpq-dev + gcc  → psycopg / psycopg2 build
# supervisor       → installed via pip, but needs system Python path
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev gcc python3-dev postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (required by HF Spaces) ─────────────────────────────────────
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /code

# ── Python dependencies ────────────────────────────────────────────────────────
# Installed in a separate layer so Docker caches them even if source changes.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source + supervisord config ────────────────────────────────────
COPY --chown=user . .
RUN chmod +x /code/entrypoint.sh

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Entrypoint ────────────────────────────────────────────────────────────────
# entrypoint.sh picks supervisord.api.conf or supervisord.worker.conf based on
# the SPACE_ROLE variable — same image, deployed to two separate HF Spaces so
# FastAPI and Celery aren't competing for RAM in one small container.
CMD ["/code/entrypoint.sh"]
