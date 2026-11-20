CREATE EXTENSION IF NOT EXISTS vector;

-- Product catalog. `embedding` holds a 768-dim sentence embedding of
-- name + category + description, used for semantic search.
CREATE TABLE products (
    id          bigserial PRIMARY KEY,
    sku         text NOT NULL UNIQUE,
    name        text NOT NULL,
    category    text NOT NULL,
    description text NOT NULL,
    price       numeric(10, 2) NOT NULL CHECK (price > 0),
    stock       integer NOT NULL CHECK (stock >= 0),
    -- Diet flags the agent can filter on. Embeddings can't express "no meat",
    -- so exclusion is a filter, not part of the search text.
    diet        text[] NOT NULL DEFAULT '{}' CHECK (diet <@ ARRAY['vegan', 'vegetarian', 'dairy-free']),
    embedding   vector(768),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX products_embedding_idx ON products USING hnsw (embedding vector_cosine_ops);

-- The agent can only *propose* a price change. A human approves or rejects it
-- through the approvals API; only then does the price in `products` change.
CREATE TABLE price_change_requests (
    id            bigserial PRIMARY KEY,
    sku           text NOT NULL REFERENCES products (sku),
    old_price     numeric(10, 2) NOT NULL,
    new_price     numeric(10, 2) NOT NULL CHECK (new_price > 0),
    reason        text NOT NULL,
    risk          text NOT NULL CHECK (risk IN ('low', 'high')),
    status        text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
    requested_by  text NOT NULL,
    decided_by    text,
    decision_note text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    decided_at    timestamptz,
    CHECK ((status = 'pending') = (decided_at IS NULL))
);

-- At most one open request per product, enforced by the database rather than
-- by a check-then-insert in application code (which would race).
CREATE UNIQUE INDEX one_pending_request_per_sku
    ON price_change_requests (sku) WHERE status = 'pending';

-- Every tool call the agent makes and every human decision, in order.
-- Each row stores the hash of the previous row, so editing or deleting a row
-- in the middle breaks the chain and `verify` reports where.
CREATE TABLE audit_log (
    id          bigserial PRIMARY KEY,
    ts          timestamptz NOT NULL,
    actor       text NOT NULL,
    tool        text NOT NULL,
    args        jsonb NOT NULL,
    outcome     text NOT NULL CHECK (outcome IN ('ok', 'rejected', 'error')),
    result      jsonb,
    error       text,
    duration_ms integer NOT NULL,
    prev_hash   text NOT NULL,
    hash        text NOT NULL UNIQUE
);
