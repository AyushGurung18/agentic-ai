"""
app/worker/tasks.py
────────────────────
Celery tasks for background document processing.

`ingest_document_task` is a lightweight bridge, not the actual worker: it
relays the job to the Cloud Run ingestion service (cloud_run_ingest/) over
HTTP instead of doing the chunk/embed/store work itself. This is
deliberate — this process no longer imports anything that touches
torch/sentence-transformers, so it stays small and never OOMs the HF
Space, and the heavy compute only runs (and only costs anything) while
Cloud Run is actually processing a job, not sitting idle 24/7.

Status transitions (now mostly owned by cloud_run_ingest, which has the
fullest picture of actual progress):
    pending → processing → done
                        ↘ failed (on any exception, including this
                          bridge failing to reach Cloud Run at all)
"""

import logging
import os

import httpx
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from app.worker.celery_app import celery_app
from app.db.database import get_conn

logger = logging.getLogger("worker.tasks")

INGEST_SERVICE_URL = os.environ.get("INGEST_SERVICE_URL", "")
INTERNAL_API_SECRET = os.environ.get("INTERNAL_API_SECRET", "")


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
    Relay a document to the Cloud Run ingestion service and wait for it to
    finish. Cloud Run does the actual chunk/embed/store work and owns the
    "processing"/progress/"done" updates on document_jobs from there —
    this task's own status writes just cover the window before Cloud Run
    picks it up, and failures if it can't be reached at all.

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
    logger.info(
        "[ingest_document_task] job_id=%s filename=%s user=%s -> forwarding to Cloud Run",
        job_id, filename, user_id,
    )

    if not INGEST_SERVICE_URL:
        _update_job(job_id, status="failed", error="INGEST_SERVICE_URL is not configured.")
        raise RuntimeError("INGEST_SERVICE_URL is not configured.")

    _update_job(job_id, status="processing", progress_percent=0)

    try:
        resp = httpx.post(
            f"{INGEST_SERVICE_URL}/ingest",
            json={
                "job_id": job_id,
                "text": text,
                "filename": filename,
                "user_id": user_id,
                "session_id": session_id,
                "metadata_tags": metadata_tags or {},
            },
            headers={"X-Internal-Secret": INTERNAL_API_SECRET},
            # Generous — matches Cloud Run cold start (torch + model load)
            # plus the same ~10min ceiling the old in-process task used.
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
        logger.info(
            "[ingest_document_task] ✅ Done job_id=%s document_id=%s",
            job_id, data.get("document_id"),
        )
        return data

    except SoftTimeLimitExceeded:
        # Graceful timeout — mark failed so the client knows
        _update_job(job_id, status="failed", error="Task timed out after 9 minutes.")
        logger.error("[ingest_document_task] ⏱ Soft time limit exceeded for job_id=%s", job_id)
        raise  # Let Celery handle it

    except httpx.HTTPStatusError as exc:
        # Cloud Run already updated document_jobs itself for errors that
        # happen mid-ingestion — this covers the ones it couldn't (e.g. the
        # auth check rejecting the request before any work started).
        error_msg = f"Ingest service returned {exc.response.status_code}: {exc.response.text[:300]}"
        _update_job(job_id, status="failed", error=error_msg)
        logger.error("[ingest_document_task] ❌ job_id=%s error=%s", job_id, error_msg)
        raise self.retry(exc=exc)

    except Exception as exc:
        error_msg = str(exc)[:500]  # truncate to avoid huge DB values
        _update_job(job_id, status="failed", error=error_msg)
        logger.error("[ingest_document_task] ❌ job_id=%s error=%s", job_id, error_msg)
        raise self.retry(exc=exc)
