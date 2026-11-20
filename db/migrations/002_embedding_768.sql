-- Switches embeddings from 384 dimensions (paraphrase-multilingual-MiniLM-L12-v2)
-- to 768 (google/embeddinggemma-300m). Old vectors can't be converted, so they are
-- cleared; search returns nothing useful until the re-seed fills them again:
--   docker compose exec -T db psql -U store -d store < db/migrations/002_embedding_768.sql
--   uv run store-agent-seed
BEGIN;
DROP INDEX IF EXISTS products_embedding_idx;
ALTER TABLE products ALTER COLUMN embedding TYPE vector(768) USING NULL;
CREATE INDEX products_embedding_idx ON products USING hnsw (embedding vector_cosine_ops);
COMMIT;
