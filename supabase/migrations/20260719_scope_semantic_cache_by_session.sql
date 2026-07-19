-- supabase/migrations/20260719_scope_semantic_cache_by_session.sql
-- The semantic cache had zero scoping — lookups matched purely on question
-- text similarity, across every user, every session, every document. A
-- generic question like "summarize" cached once (with no useful document
-- context) would then incorrectly answer "summarize" for every other
-- document, in every other session, for every user, forever. Scoping by
-- session_id so a cache hit only ever replays an answer that was actually
-- generated within the same conversation/document context.

ALTER TABLE public.semantic_responses_cache
    ADD COLUMN IF NOT EXISTS session_id UUID;

CREATE INDEX IF NOT EXISTS idx_semantic_cache_session
    ON public.semantic_responses_cache (session_id);

-- Existing rows have no session context recorded — a NULL session_id must
-- not become a wildcard that matches every future lookup, so these are
-- unrecoverable and get dropped rather than guessed at.
DELETE FROM public.semantic_responses_cache WHERE session_id IS NULL;

ALTER TABLE public.semantic_responses_cache
    ALTER COLUMN session_id SET NOT NULL;
