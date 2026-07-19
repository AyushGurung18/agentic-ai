"""
app/worker/celery_app.py
────────────────────────
Celery application factory wired to CloudAMQP (RabbitMQ).

Broker  : CLOUDAMQP_URL  (amqps://... from CloudAMQP dashboard)
Backend : rpc://          (stateless — job status is tracked in Supabase,
                           not in the Celery result backend, so rpc is fine)

Worker start command:
    celery -A app.worker.celery_app worker --loglevel=info --concurrency=1

Configuration notes:
  • acks_late=True          → message only ACK'd after task succeeds/fails,
                              so the job is never silently lost on worker crash
  • prefetch_multiplier=1   → worker only grabs 1 task at a time; important for
                              long-running PDF jobs that consume significant RAM
  • task_time_limit=600     → hard kill after 10 min (handles runaway embedding)
  • task_soft_time_limit=540→ raises SoftTimeLimitExceeded 60s before hard kill
                              so the task can clean up and mark itself failed
"""

import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

# ── Broker URL ────────────────────────────────────────────────────────────────
# CloudAMQP provides amqps:// (TLS) — Celery + kombu handle it natively.
CLOUDAMQP_URL = os.environ.get("CLOUDAMQP_URL", "")
if not CLOUDAMQP_URL:
    import warnings
    warnings.warn(
        "CLOUDAMQP_URL is not set. The Celery worker will fail to connect. "
        "Add amqps://user:pass@host/vhost to your .env file.",
        RuntimeWarning,
        stacklevel=2,
    )

# ── Celery app ────────────────────────────────────────────────────────────────
# Falls back to local RabbitMQ if URL is unset — keeps the module importable
# in the API process even when the worker isn't configured yet.
_broker = CLOUDAMQP_URL or "amqp://guest:guest@localhost//"
celery_app = Celery(
    "thotqen",
    broker=_broker,
    backend="rpc://",           # lightweight; real status lives in document_jobs
    include=["app.worker.tasks"],
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # ── CloudAMQP Free-Tier connection budget (hard cap: 20 connections) ────────
    # 1 pool slot for publisher (API process) + 1 for worker = 2 total.
    # Without this, Celery opens a pool of connections per process and
    # blows past the 20-connection limit within minutes.
    broker_pool_limit=1,

    # These three flags stop the worker from opening extra control-plane
    # connections for gossip (peer discovery), mingle (state sync), and
    # heartbeat (liveness pings). On CloudAMQP Free, every connection counts.
    worker_gossip=False,
    worker_mingle=False,
    # heartbeat is disabled via CLI --without-heartbeat in supervisord.conf

    # Reliability
    task_acks_late=True,
    worker_prefetch_multiplier=1,

    # Recycle the worker process every few tasks so memory fragmentation
    # from repeated PDF/embedding runs doesn't creep up over a long-lived
    # container uptime on a small, memory-constrained host.
    worker_max_tasks_per_child=5,

    # Timeouts — 100-page PDFs take ~2-4 min to embed on CPU
    task_time_limit=600,        # hard kill at 10 min
    task_soft_time_limit=540,   # graceful at 9 min

    # Retry policy for transient broker blips
    broker_connection_retry_on_startup=True,
    broker_transport_options={
        "visibility_timeout": 3600,  # 1 hr — keeps msg invisible while processing
        "confirm_publish": True,     # publisher confirms for durability
    },

    # Timezone
    timezone="UTC",
    enable_utc=True,
)
