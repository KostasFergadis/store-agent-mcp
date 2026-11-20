import json

from store_agent.seed import DEFAULT_DATA, seed


def test_reseed_keeps_live_price_and_stock(conn, embedder):
    # Prices change only through approval (with an audit row). Re-seeding to pick up
    # new descriptions, tags or diet flags must not quietly reset them.
    conn.execute("UPDATE products SET price = 10.90, stock = 3 WHERE sku = 'PAN-001'")
    products = json.loads(DEFAULT_DATA.read_text())
    products[0] = {**products[0], "description": "updated description"}
    products.append({**products[1], "sku": "NEW-001", "price": 2.50, "stock": 7})

    seed(conn, products, embedder)

    oil = conn.execute("SELECT price, stock FROM products WHERE sku = 'PAN-001'").fetchone()
    assert (float(oil["price"]), oil["stock"]) == (10.90, 3)
    first = conn.execute("SELECT description FROM products WHERE sku = %s", (products[0]["sku"],)).fetchone()
    assert first["description"] == "updated description"
    new = conn.execute("SELECT price, stock FROM products WHERE sku = 'NEW-001'").fetchone()
    assert (float(new["price"]), new["stock"]) == (2.50, 7)
