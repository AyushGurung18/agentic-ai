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

-- Per-quote verification results (verified/unverified, document, page,
-- char range) for an assistant message — e.g. {"quotes": [...]}. Without
-- this, reopening a past conversation had no way to know which quotes were
-- originally verified (that metadata only ever existed in the live SSE
-- stream, never persisted), so every quote in a reloaded chat silently
-- rendered as "unverified" regardless of its real, original status.
ALTER TABLE messages ADD COLUMN IF NOT EXISTS metadata JSONB;

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);

-- 5. Documents
CREATE TABLE IF NOT EXISTS documents (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID        REFERENCES users(id) ON DELETE CASCADE,
    filename    TEXT,
    content     TEXT,         -- raw extracted text
    created_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Library / viewer metadata — additive, safe to re-run.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS doc_type    TEXT DEFAULT 'pdf';   -- 'pdf' | 'docx'
ALTER TABLE documents ADD COLUMN IF NOT EXISTS status      TEXT DEFAULT 'ready'; -- 'ready' | 'empty' | 'failed'
ALTER TABLE documents ADD COLUMN IF NOT EXISTS page_count  INT;                  -- PDF only; NULL for docx
ALTER TABLE documents ADD COLUMN IF NOT EXISTS char_count  INT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS r2_url      TEXT;
-- The session this document was originally uploaded into — "open" from the
-- library jumps straight back into this chat so history isn't lost, rather
-- than starting a new blank thread every time.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS home_session_id UUID REFERENCES sessions(id) ON DELETE SET NULL;

-- Per-page character offsets into documents.content, used to turn a verified
-- quote's character range into "which page is this on" for citation
-- highlighting. PDF only — DOCX has no fixed pagination, so it gets no rows
-- here and the viewer highlights directly against the flowing HTML instead.
CREATE TABLE IF NOT EXISTS document_pages (
    id          UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID    REFERENCES documents(id) ON DELETE CASCADE,
    page_number INT     NOT NULL,
    char_start  INT     NOT NULL,
    char_end    INT     NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_document_pages_document_id ON document_pages(document_id);

-- Which document(s) a chat session is grounded in. A normal single-doc chat
-- has exactly one row; a multi-document question has several.
CREATE TABLE IF NOT EXISTS session_documents (
    session_id  UUID REFERENCES sessions(id) ON DELETE CASCADE,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    PRIMARY KEY (session_id, document_id)
);

CREATE INDEX IF NOT EXISTS idx_session_documents_document_id ON session_documents(document_id);

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

-- Backfill session_documents for documents uploaded before this table
-- existed, whose session association only ever lived as a
-- document_metadata(key='session_id') tag. Without this, a document/session
-- pair from before this migration has zero session_documents rows, so
-- retrieval for that session silently scopes to nothing post-upgrade.
-- Re-running this is harmless: already-populated pairs hit ON CONFLICT.
-- Guards the value against looking like a real UUID first, in its own CTE —
-- this runs on every app startup (init_db()), so a single malformed legacy
-- tag must never throw and break schema bootstrap for every user. Filtering
-- in a separate CTE (rather than an AND'd WHERE clause) means the ::uuid
-- cast below only ever runs against already-validated rows regardless of
-- how Postgres orders predicate evaluation.
WITH valid_tags AS MATERIALIZED (
    -- MATERIALIZED (not just a plain CTE, which Postgres 12+ may inline and
    -- reorder against the join below) forces this filter to run as its own
    -- step, so the cast in the JOIN can never see an unvalidated value.
    SELECT document_id, value AS session_id_text
    FROM   document_metadata
    WHERE  key = 'session_id'
      AND  value ~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
)
INSERT INTO session_documents (session_id, document_id)
SELECT vt.session_id_text::uuid, vt.document_id
FROM   valid_tags vt
JOIN   sessions s ON s.id = vt.session_id_text::uuid
ON CONFLICT DO NOTHING;

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
--
-- filter_document_ids scopes retrieval to an explicit set of documents (the
-- ones attached to the current chat session) rather than every document the
-- user has ever uploaded. Without this, a question asked in one document's
-- chat could retrieve — and "verify" quotes against — a completely
-- unrelated document the same user happened to upload earlier.
DROP FUNCTION IF EXISTS hybrid_search(TEXT, VECTOR(384), INT, UUID, INT);
DROP FUNCTION IF EXISTS hybrid_search(TEXT, VECTOR(384), INT, UUID, UUID[], INT);

CREATE OR REPLACE FUNCTION hybrid_search(
    query_text TEXT,
    query_embedding VECTOR(384),
    match_count INT DEFAULT 10,
    filter_user_id UUID DEFAULT NULL,
    filter_document_ids UUID[] DEFAULT NULL,
    rrf_k INT DEFAULT 60
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    parent_chunk_id UUID,
    content TEXT,
    filename TEXT,
    score FLOAT
) AS $$
WITH dense_search AS (
    SELECT
        c.id AS chunk_id,
        c.document_id,
        c.parent_chunk_id,
        c.content,
        d.filename,
        ROW_NUMBER() OVER(ORDER BY e.embedding <=> query_embedding) AS rank
    FROM document_chunks c
    JOIN embeddings e ON c.id = e.chunk_id
    JOIN documents d ON c.document_id = d.id
    WHERE (filter_user_id IS NULL OR d.user_id = filter_user_id)
      AND (filter_document_ids IS NULL OR d.id = ANY(filter_document_ids))
    ORDER BY e.embedding <=> query_embedding
    LIMIT match_count * 2
),
keyword_search AS (
    SELECT
        c.id AS chunk_id,
        c.document_id,
        c.parent_chunk_id,
        c.content,
        d.filename,
        ROW_NUMBER() OVER(ORDER BY ts_rank(c.fts, websearch_to_tsquery('english', query_text)) DESC) AS rank
    FROM document_chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE (filter_user_id IS NULL OR d.user_id = filter_user_id)
      AND (filter_document_ids IS NULL OR d.id = ANY(filter_document_ids))
      AND c.fts @@ websearch_to_tsquery('english', query_text)
    ORDER BY ts_rank(c.fts, websearch_to_tsquery('english', query_text)) DESC
    LIMIT match_count * 2
)
SELECT
    COALESCE(d.chunk_id, k.chunk_id) AS chunk_id,
    COALESCE(d.document_id, k.document_id) AS document_id,
    COALESCE(d.parent_chunk_id, k.parent_chunk_id) AS parent_chunk_id,
    COALESCE(d.content, k.content) AS content,
    COALESCE(d.filename, k.filename) AS filename,
    COALESCE(1.0 / (rrf_k + d.rank), 0.0) + COALESCE(1.0 / (rrf_k + k.rank), 0.0) AS score
FROM dense_search d
FULL OUTER JOIN keyword_search k ON d.chunk_id = k.chunk_id
ORDER BY score DESC
LIMIT match_count;
$$ LANGUAGE sql STABLE;
