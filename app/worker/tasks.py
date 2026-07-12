"""
app/worker/tasks.py
────────────────────
Celery tasks for background document processing.

The only task here is `ingest_document_task`, which:
  1. Marks the job as "processing" in document_jobs
  2. Calls process_document() (chunk → embed → PGVector insert)
  3. Marks the job as "done" (or "failed" on exception)

Status transitions:
    pending → processing → done
                        ↘ failed (on any exception)
"""

import logging
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from app.worker.celery_app import celery_app
from app.db.database import get_conn
from app.services.rag import process_document

logger = logging.getLogger("worker.tasks")


# ── DB helpers ────────────────────────────────────────────────────────────────

def _update_job(job_id: str, **fields) -> None:
    """Update a document_jobs row with arbitrary fields."""
    set_clauses = ", ".join(f"{k} = %s" for k in fields)
    values = list(fields.values()) + [job_id]
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE document_jobs SET {set_clauses} WHERE id = %s::uuid",
                values,
            )
        conn.commit()


# ── Task ──────────────────────────────────────────────────────────────────────

class BaseTaskWithRetry(Task):
    """Base task that logs exceptions before they propagate."""
    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error("[Task:%s] Failed: %s", task_id, exc, exc_info=einfo)
        super().on_failure(exc, task_id, args, kwargs, einfo)


@celery_app.task(
    bind=True,
    base=BaseTaskWithRetry,
    name="thotqen.ingest_document",
    max_retries=2,
    default_retry_delay=30,   # wait 30s between retries
)
def ingest_document_task(
    self,
    job_id: str,
    text: str,
    filename: str,
    user_id: str,
    session_id: str | None = None,
    metadata_tags: dict | None = None,
) -> dict:
    """
    Background task: chunk + embed a document and store in PGVector.

    Parameters
    ----------
    job_id       : UUID of the document_jobs row (for status updates)
    text         : Extracted plain text from the PDF
    filename     : Original PDF filename
    user_id      : Owning user UUID string
    session_id   : Optional session to attach the document to
    metadata_tags: Optional extra metadata (e.g. {"r2_url": "..."})

    Returns
    -------
    dict with document_id on success
    """
    logger.info("[ingest_document_task] job_id=%s filename=%s user=%s", job_id, filename, user_id)

    # ── Mark as processing ─────────────────────────────────────────────────────
    _update_job(job_id, status="processing", progress_percent=0)
    
    def on_progress(percent: int):
        _update_job(job_id, progress_percent=percent)

    try:
        document_id = process_document(
            text=text,
            filename=filename,
            user_id=user_id,
            session_id=session_id,
            metadata_tags=metadata_tags or {},
            progress_callback=on_progress,
        )

        # ── Mark as done ───────────────────────────────────────────────────────
        _update_job(job_id, status="done", progress_percent=100, document_id=document_id)
        logger.info("[ingest_document_task] ✅ Done job_id=%s document_id=%s", job_id, document_id)
        return {"job_id": job_id, "document_id": document_id}

    except SoftTimeLimitExceeded:
        # Graceful timeout — mark failed so the client knows
        _update_job(job_id, status="failed", error="Task timed out after 9 minutes.")
        logger.error("[ingest_document_task] ⏱ Soft time limit exceeded for job_id=%s", job_id)
        raise  # Let Celery handle it

    except Exception as exc:
        error_msg = str(exc)[:500]  # truncate to avoid huge DB values
        _update_job(job_id, status="failed", error=error_msg)
        logger.error("[ingest_document_task] ❌ job_id=%s error=%s", job_id, error_msg)

        # Retry on transient errors (DB blips, embedding model loading issues)
        # Don't retry if it's a ValueError (bad input — won't succeed on retry)
        if not isinstance(exc, (ValueError, TypeError)):
            raise self.retry(exc=exc)
        raise
