-- supabase/migrations/20240901_add_semantic_cache.sql
-- Create semantic_responses_cache table for cached Q&A + vector embedding
CREATE EXTENSION IF NOT EXISTS vector;  -- ensure pgvector extension is available

CREATE TABLE IF NOT EXISTS public.semantic_responses_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    raw_question TEXT NOT NULL,
    cached_answer TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Create an HNSW index for fast ANN search on the embedding column
CREATE INDEX IF NOT EXISTS idx_semantic_cache_embedding
    ON public.semantic_responses_cache USING hnsw (embedding vector_cosine_ops);

-- Function to find the most similar cached answer (cosine similarity)
CREATE OR REPLACE FUNCTION public.match_semantic_cache(
    query_embedding VECTOR(384),
    similarity_threshold FLOAT
) RETURNS TABLE (
    cached_answer TEXT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        cached_answer,
        1 - (query_embedding <=> embedding) AS similarity   -- cosine distance -> similarity
    FROM public.semantic_responses_cache
    WHERE (query_embedding <=> embedding) <= (1 - similarity_threshold)
    ORDER BY similarity DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql STABLE;

-- Optional: grant SELECT/INSERT/UPDATE to the Supabase anon/service role if needed
-- ALTER ROLE service_role GRANT SELECT, INSERT, UPDATE ON public.semantic_responses_cache TO service_role;
