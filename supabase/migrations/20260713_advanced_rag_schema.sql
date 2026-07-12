-- Migration: Advanced RAG Schema (Parent-Child, Hybrid Search, Progress)

-- 1. Add progress tracking to document_jobs
ALTER TABLE document_jobs
ADD COLUMN IF NOT EXISTS progress_percent INT DEFAULT 0;

-- 2. Add Parent-Child relationship and Full Text Search to document_chunks
ALTER TABLE document_chunks
ADD COLUMN IF NOT EXISTS parent_chunk_id UUID REFERENCES document_chunks(id) ON DELETE CASCADE,
ADD COLUMN IF NOT EXISTS fts tsvector;

-- 3. Auto-update fts column based on content
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

-- Create index for faster FTS
CREATE INDEX IF NOT EXISTS idx_document_chunks_fts ON document_chunks USING GIN(fts);

-- 4. Hybrid Search Function (Vector + BM25 with Reciprocal Rank Fusion)
-- This function takes a query embedding and a text query, searches both spaces,
-- and combines them using RRF formula: 1 / (k + rank)
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
