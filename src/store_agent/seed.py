"""Load data/products.json into Postgres with embeddings.

Safe to re-run: existing products get new names, descriptions, diet flags and embeddings,
but keep their live price and stock.
"""

import argparse
import json
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from .config import get_settings
from .embeddings import make_embedder, to_pgvector

DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data" / "products.json"


def embedding_text(p: dict) -> str:
    # Tags (diet, dishes, Greek name) are embedded for search but not stored as a column.
    text = f"{p['name']}. {p['category']}. {p['description']}"
    return f"{text} {p['tags']}" if p.get("tags") else text


def seed(conn: psycopg.Connection, products: list[dict], embedder) -> int:
    vectors = embedder.embed([embedding_text(p) for p in products])
    with conn.transaction():
        for p, v in zip(products, vectors, strict=True):
            conn.execute(
                """
                INSERT INTO products (sku, name, category, description, price, stock, diet, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (sku) DO UPDATE SET
                    name = EXCLUDED.name, category = EXCLUDED.category,
                    description = EXCLUDED.description, diet = EXCLUDED.diet, embedding = EXCLUDED.embedding
                -- price and stock only come from the file for new products. Existing prices
                -- change through approval with an audit row, and a re-seed must not undo that.
                -- updated_at is left alone too: it records the last price change.
                """,
                (
                    p["sku"], p["name"], p["category"], p["description"], p["price"], p["stock"],
                    p.get("diet", []), to_pgvector(v),
                ),
            )
    return len(products)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--reset", action="store_true", help="also clear requests and the audit log")
    args = parser.parse_args()

    settings = get_settings()
    products = json.loads(args.data.read_text())
    with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
        if args.reset:
            conn.execute("TRUNCATE audit_log, price_change_requests RESTART IDENTITY")
        n = seed(conn, products, make_embedder(settings.embedder))
    print(f"seeded {n} products")


if __name__ == "__main__":
    main()
