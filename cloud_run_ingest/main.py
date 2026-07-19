"""
cloud_run_ingest/main.py
─────────────────────────
Standalone ingestion service — the heavy half of what used to be
app.services.rag.process_document(), moved out of the HF Space entirely.

Reuses the existing chunking/embedding/DB-write modules directly (single
source of truth — nothing here is a reimplementation, just a different
front door onto the same app/services code) so behavior stays identical
to what process_document() already did. Deployed on Cloud Run instead of
the HF Space specifically because it needs torch + sentence-transformers
loaded, and Cloud Run's scale-to-zero means that cost is only paid while
an ingestion job is actually running, not sitting idle 24/7 the way the
old Celery worker did.

Called by app/worker/tasks.py's lightweight bridge task via HTTP POST.
"""

import logging
import os

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from app.core.config import DEV_USER_ID, DEV_USER_EMAIL
from app.db.database import get_conn
from app.services.chunking import hierarchical_chunk_text
from app.services.document_service import ensure_user, ingest_document
from app.services.vectorstore import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cloud_run_ingest")

INTERNAL_API_SECRET = os.environ.get("INTERNAL_API_SECRET", "")

app = FastAPI(title="thotqen-ingest")
vectorstore = VectorStore()


class IngestRequest(BaseModel):
    job_id: str
    text: str
    filename: str
    user_id: str | None = None
    session_id: str | None = None
    metadata_tags: dict[str, str] | None = None


def _update_job(job_id: str, **fields) -> None:
    set_clauses = ", ".join(f"{k} = %s" for k in fields)
    values = list(fields.values()) + [job_id]
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE document_jobs SET {set_clauses} WHERE id = %s::uuid",
                values,
            )
        conn.commit()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest, x_internal_secret: str = Header(default="")):
    if not INTERNAL_API_SECRET or x_internal_secret != INTERNAL_API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    user_id = req.user_id or DEV_USER_ID
    logger.info("[ingest] job_id=%s filename=%s user=%s", req.job_id, req.filename, user_id)

    _update_job(req.job_id, status="processing", progress_percent=0)

    def on_progress(percent: int):
        _update_job(req.job_id, progress_percent=percent)

    try:
        ensure_user(user_id, email=DEV_USER_EMAIL)

        hierarchical_chunks = hierarchical_chunk_text(req.text)
        flat_children = [c for p in hierarchical_chunks for c in p["children"]]
        logger.info("[ingest] %d parents, %d children", len(hierarchical_chunks), len(flat_children))

        tags = req.metadata_tags or {}
        if req.session_id:
            tags["session_id"] = req.session_id

        document_id, chunk_metadatas = ingest_document(
            text=req.text,
            hierarchical_chunks=hierarchical_chunks,
            filename=req.filename,
            user_id=user_id,
            metadata_tags=tags,
            progress_callback=on_progress,
        )

        vectorstore.add(flat_children, metadata=chunk_metadatas)

        _update_job(req.job_id, status="done", progress_percent=100, document_id=document_id)
        logger.info("[ingest] done job_id=%s document_id=%s", req.job_id, document_id)
        return {"job_id": req.job_id, "document_id": document_id}

    except Exception as exc:
        error_msg = str(exc)[:500]
        _update_job(req.job_id, status="failed", error=error_msg)
        logger.exception("[ingest] failed job_id=%s", req.job_id)
        raise HTTPException(status_code=500, detail=error_msg)
