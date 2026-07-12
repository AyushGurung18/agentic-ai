-- ============================================================
-- thotqen — full normalized schema (Supabase Optimized)
-- Run once; safe to re-run (all statements are IF NOT EXISTS)
-- ============================================================

-- 1. Extensions
-- Supabase enables these in the 'extensions' schema sometimes, 
-- but 'public' is fine for most cases.
CREATE EXTENSION IF NOT EXISTS "pgcrypto";   -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS "vector";     -- pgvector

-- 2. Users
CREATE TABLE IF NOT EXISTS users (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    email       TEXT        UNIQUE NOT NULL,
    name        TEXT,
    created_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 3. Sessions (conversations)
CREATE TABLE IF NOT EXISTS sessions (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID        REFERENCES users(id) ON DELETE CASCADE,
    title       TEXT,
    created_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 4. Messages
CREATE TABLE IF NOT EXISTS messages (
    id          BIGSERIAL   PRIMARY KEY,
    session_id  UUID        REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT        CHECK (role IN ('user', 'assistant')) NOT NULL,
    content     TEXT        NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);

-- 5. Documents
CREATE TABLE IF NOT EXISTS documents (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID        REFERENCES users(id) ON DELETE CASCADE,
    filename    TEXT,
    content     TEXT,         -- raw extracted text
    created_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 6. Document Chunks
CREATE TABLE IF NOT EXISTS document_chunks (
    id            UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id   UUID    REFERENCES documents(id) ON DELETE CASCADE,
    parent_chunk_id UUID  REFERENCES document_chunks(id) ON DELETE CASCADE,
    chunk_index   INT     NOT NULL,
    content       TEXT    NOT NULL,
    fts           TSVECTOR,
    created_at    TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_fts ON document_chunks USING GIN(fts);

CREATE OR REPLACE FUNCTION update_document_chunks_fts()
RETURNS TRIGGER AS $$
BEGIN
    NEW.fts = to_tsvector('english', NEW.content);
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_document_chunks_fts ON document_chunks;
CREATE TRIGGER trg_document_chunks_fts
BEFORE INSERT OR UPDATE ON document_chunks
FOR EACH ROW EXECUTE FUNCTION update_document_chunks_fts();


-- 7. Embeddings
-- Using 384-dim for all-MiniLM-L6-v2 (sentence-transformers)
CREATE TABLE IF NOT EXISTS embeddings (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id    UUID        REFERENCES document_chunks(id) ON DELETE CASCADE,
    embedding   VECTOR(384) NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- HNSW index for fast ANN cosine-similarity search (replaces sequential scan)
-- m=16, ef_construction=64 is a well-balanced default for RAG workloads.
-- Set hnsw.ef_search=100 at query time for high-recall retrieval.
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
    ON embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 8. Document Metadata
CREATE TABLE IF NOT EXISTS document_metadata (
    id          UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID    REFERENCES documents(id) ON DELETE CASCADE,
    key         TEXT    NOT NULL,
    value       TEXT
);

CREATE INDEX IF NOT EXISTS idx_document_metadata_document_id ON document_metadata(document_id);

-- 9. Document Jobs (async processing status for Celery worker)
CREATE TABLE IF NOT EXISTS document_jobs (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID        REFERENCES users(id) ON DELETE CASCADE,
    filename    TEXT        NOT NULL,
    r2_url      TEXT,
    status      TEXT        NOT NULL
                            CHECK (status IN ('pending', 'processing', 'done', 'failed'))
                            DEFAULT 'pending',
    progress_percent INT    DEFAULT 0,
    document_id UUID,
    error       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_document_jobs_user_id
    ON document_jobs (user_id, created_at DESC);

-- 10. Hybrid Search Function (Vector + BM25 with Reciprocal Rank Fusion)
CREATE OR REPLACE FUNCTION hybrid_search(
    query_text TEXT,
    query_embedding VECTOR(384),
    match_count INT DEFAULT 10,
    filter_user_id UUID DEFAULT NULL,
    rrf_k INT DEFAULT 60
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    parent_chunk_id UUID,
    content TEXT,
    score FLOAT
) AS $$
WITH dense_search AS (
    SELECT
        c.id AS chunk_id,
        c.document_id,
        c.parent_chunk_id,
        c.content,
        ROW_NUMBER() OVER(ORDER BY e.embedding <=> query_embedding) AS rank
    FROM document_chunks c
    JOIN embeddings e ON c.id = e.chunk_id
    JOIN documents d ON c.document_id = d.id
    WHERE (filter_user_id IS NULL OR d.user_id = filter_user_id)
    ORDER BY e.embedding <=> query_embedding
    LIMIT match_count * 2
),
keyword_search AS (
    SELECT
        c.id AS chunk_id,
        c.document_id,
        c.parent_chunk_id,
        c.content,
        ROW_NUMBER() OVER(ORDER BY ts_rank(c.fts, websearch_to_tsquery('english', query_text)) DESC) AS rank
    FROM document_chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE (filter_user_id IS NULL OR d.user_id = filter_user_id)
      AND c.fts @@ websearch_to_tsquery('english', query_text)
    ORDER BY ts_rank(c.fts, websearch_to_tsquery('english', query_text)) DESC
    LIMIT match_count * 2
)
SELECT
    COALESCE(d.chunk_id, k.chunk_id) AS chunk_id,
    COALESCE(d.document_id, k.document_id) AS document_id,
    COALESCE(d.parent_chunk_id, k.parent_chunk_id) AS parent_chunk_id,
    COALESCE(d.content, k.content) AS content,
    COALESCE(1.0 / (rrf_k + d.rank), 0.0) + COALESCE(1.0 / (rrf_k + k.rank), 0.0) AS score
FROM dense_search d
FULL OUTER JOIN keyword_search k ON d.chunk_id = k.chunk_id
ORDER BY score DESC
LIMIT match_count;
$$ LANGUAGE sql STABLE;
