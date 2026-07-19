#!/bin/sh
# Picks which supervisord config to run based on the SPACE_ROLE variable
# (set per-Space in HF Space settings — a plain Variable, not a secret):
#   SPACE_ROLE=api     (default, unset) → FastAPI only
#   SPACE_ROLE=worker                   → Celery worker only
#
# Both Spaces deploy from the exact same image/repo — this is the only
# thing that differs between them, so there's nothing to keep in sync by
# hand between two separate Dockerfiles.
set -e

if [ "$SPACE_ROLE" = "worker" ]; then
    exec supervisord -c /code/supervisord.worker.conf
else
    exec supervisord -c /code/supervisord.api.conf
fi
