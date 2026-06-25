-- Semantic memory via pgvector: store an embedding per memory_unit so retrieval
-- can find the most *relevant* past memory (by meaning), not just the most recent.
--
-- Every block is independently fault-tolerant. If this Postgres doesn't have the
-- pgvector extension available, each block logs a NOTICE and the migration still
-- succeeds — memory simply falls back to recency/SimpleMem and worker startup is
-- never broken. (To use semantic recall, run a pgvector-enabled Postgres, e.g.
-- the Railway `pgvector` template / the pgvector/pgvector image.)
--
-- Dimension 1536 matches OpenAI text-embedding-3-small. Changing the embedding
-- model to a different dimension requires a new migration that alters the column.

DO $$
BEGIN
  CREATE EXTENSION IF NOT EXISTS vector;
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'pgvector extension unavailable; semantic memory will fall back: %', SQLERRM;
END $$;

DO $$
BEGIN
  ALTER TABLE memory_units ADD COLUMN IF NOT EXISTS embedding vector(1536);
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'embedding column skipped (pgvector type missing): %', SQLERRM;
END $$;

-- HNSW index for fast cosine-similarity search (pgvector >= 0.5). If unsupported,
-- search still works via a sequential scan — just slower.
DO $$
BEGIN
  CREATE INDEX IF NOT EXISTS memory_units_embedding_idx
    ON memory_units USING hnsw (embedding vector_cosine_ops);
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'embedding index skipped (hnsw unsupported?): %', SQLERRM;
END $$;
