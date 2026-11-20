-- Adds diet flags to an existing database (new ones get them from schema.sql).
-- Run once, then re-seed to fill them:
--   docker compose exec -T db psql -U store -d store < db/migrations/001_product_diet.sql
--   uv run store-agent-seed
ALTER TABLE products
    ADD COLUMN IF NOT EXISTS diet text[] NOT NULL DEFAULT '{}'
    CHECK (diet <@ ARRAY['vegan', 'vegetarian', 'dairy-free']);
