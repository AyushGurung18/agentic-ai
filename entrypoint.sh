#!/bin/sh
# Picks which supervisord config to run based on the SPACE_ROLE variable:
#   unset/anything else → FastAPI + Celery together (supervisord.conf)
#   SPACE_ROLE=worker    → Celery worker only (supervisord.worker.conf)
#
# HF's free tier only allows one non-static Space per account (a second
# Docker Space needs a PRO subscription), so the combined config is what
# actually runs on aayos/thotqen. The worker-only path is kept dormant for
# if a genuinely free second host (e.g. a separate VM) ever comes up.
set -e

if [ "$SPACE_ROLE" = "worker" ]; then
    exec supervisord -c /code/supervisord.worker.conf
else
    exec supervisord -c /code/supervisord.conf
fi
