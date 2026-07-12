-- =============================================================================
-- supabase/migrations/20260712_document_jobs.sql
-- Job tracking table for async PDF ingestion via Celery + RabbitMQ.
--
-- Flow:
--   POST /upload   → INSERT document_jobs (status=pending) → enqueue Celery task
--   Celery worker  → UPDATE status=processing → run ingest → UPDATE status=done
--   GET /upload/status/{job_id} → SELECT from this table
-- =============================================================================

CREATE TABLE IF NOT EXISTS public.document_jobs (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID        REFERENCES public.users(id) ON DELETE CASCADE,
    filename    TEXT        NOT NULL,
    r2_url      TEXT,                   -- Cloudflare R2 URL of the uploaded PDF
    status      TEXT        NOT NULL
                            CHECK (status IN ('pending', 'processing', 'done', 'failed'))
                            DEFAULT 'pending',
    document_id UUID,                   -- set when status = 'done'
    error       TEXT,                   -- set when status = 'failed'
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Index for fast per-user job lookups
CREATE INDEX IF NOT EXISTS idx_document_jobs_user_id
    ON public.document_jobs (user_id, created_at DESC);

-- Auto-update updated_at on every change
CREATE OR REPLACE FUNCTION public.set_document_jobs_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_document_jobs_updated_at ON public.document_jobs;
CREATE TRIGGER trg_document_jobs_updated_at
    BEFORE UPDATE ON public.document_jobs
    FOR EACH ROW EXECUTE FUNCTION public.set_document_jobs_updated_at();
