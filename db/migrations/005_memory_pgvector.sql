-- Optional pgvector support for durable semantic memory retrieval.
-- This migration must not fail worker startup on Postgres instances where the
-- vector extension is unavailable. All extension/type-dependent DDL is dynamic
-- and guarded inside DO blocks.
DO $$
BEGIN
  BEGIN
    CREATE EXTENSION IF NOT EXISTS vector;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pgvector extension unavailable; memory vector columns/indexes skipped: %', SQLERRM;
  END;
END $$;

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
    EXECUTE 'ALTER TABLE memory_units ADD COLUMN IF NOT EXISTS embedding vector(1536)';
    EXECUTE 'ALTER TABLE memory_units ADD COLUMN IF NOT EXISTS embedding_model text';
    EXECUTE 'ALTER TABLE memory_units ADD COLUMN IF NOT EXISTS embedding_created_at timestamptz';
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_memory_units_embedding_hnsw ON memory_units USING hnsw (embedding vector_cosine_ops)';
  ELSE
    RAISE NOTICE 'pgvector extension not installed; memory_units embedding columns not created';
  END IF;
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'pgvector memory migration skipped after guarded error: %', SQLERRM;
END $$;
