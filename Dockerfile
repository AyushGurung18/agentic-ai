# Dockerfile – Production image for Hugging Face Spaces (UID 1000, port 7860)
# ---------------------------------------------------------------
# Base image – lightweight Python 3.11 slim
FROM python:3.11-slim

# ---- Python runtime optimizations -------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---- System dependencies (pgvector needs libpq-dev) -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev gcc python3-dev postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# ---- Create non‑root user required by Hugging Face Spaces ------------------------
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# ---- Working directory ------------------------------------------------------------
WORKDIR /code

# ---- Install Python dependencies – leverage pip cache layer -----------------------
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy application source ------------------------------------------------------
COPY --chown=user . .

# ---- Expose the default HF port --------------------------------------------------
EXPOSE 7860

# ---- Entrypoint – run FastAPI via uvicorn on 0.0.0.0:7860 ---------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
