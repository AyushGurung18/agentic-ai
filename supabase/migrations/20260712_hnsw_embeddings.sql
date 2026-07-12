-- =============================================================================
-- supabase/migrations/20260712_hnsw_embeddings.sql
-- Add HNSW indexes for fast ANN search on all embedding tables.
--
-- Why HNSW over IVFFlat?
--   • No training phase — works immediately even with few rows
--   • Better recall at same ef_search budget
--   • Safe for growing datasets (no need to rebuild like IVFFlat requires)
--
-- Index params:
--   m               = 16   (number of bi-directional links per node; 16 is a
--                            good balance of speed vs recall for RAG workloads)
--   ef_construction = 64   (higher → better recall during build, slower build)
--
-- At query time, set ef_search ≥ 40 for high recall:
--   SET hnsw.ef_search = 100;
-- =============================================================================

-- Ensure pgvector is available
CREATE EXTENSION IF NOT EXISTS vector;

-- ── 1. embeddings table (our normalized schema) ───────────────────────────────
-- Drop any existing default index first (IVFFlat or btree on embedding)
DROP INDEX IF EXISTS public.idx_embeddings_embedding;

CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
    ON public.embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ── 2. langchain_pg_embedding table (managed by langchain-postgres / PGVector) ─
-- This is the table PGVector uses for similarity_search() calls.
-- Created automatically by langchain-postgres on first use — the index may not
-- exist yet on a fresh install, so we guard with IF NOT EXISTS.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name   = 'langchain_pg_embedding'
    ) THEN
        -- Drop old ivfflat index if present
        DROP INDEX IF EXISTS public.langchain_pg_embedding_embedding_idx;

        -- Create HNSW
        EXECUTE $sql$
            CREATE INDEX IF NOT EXISTS idx_lc_embedding_hnsw
                ON public.langchain_pg_embedding
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
        $sql$;
    END IF;
END;
$$;

-- ── 3. Recommended: set ef_search per session for high-recall RAG ─────────────
-- Run this in your application query context:
--   SET hnsw.ef_search = 100;
-- Or set it globally:
--   ALTER DATABASE postgres SET hnsw.ef_search = 100;
-- (Uncomment the line below to apply globally — requires superuser)
-- ALTER DATABASE postgres SET hnsw.ef_search = 100;
