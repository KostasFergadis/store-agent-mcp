import pytest

from store_agent import tools


def test_search_finds_relevant_product(conn, embedder):
    results = tools.search_products(conn, embedder, "feta cheese for greek salad", limit=3)
    assert results[0]["sku"] == "DAI-002"
    assert 0 < results[0]["similarity"] <= 1


def test_search_filters(conn, embedder):
    results = tools.search_products(conn, embedder, "orange juice", limit=20, in_stock_only=True, max_price=2.0)
    assert all(r["stock"] > 0 and r["price"] <= 2.0 for r in results)
    assert "BEV-003" not in {r["sku"] for r in results}  # out of stock


def test_search_rejects_empty_query(conn, embedder):
    with pytest.raises(tools.ToolError):
        tools.search_products(conn, embedder, "   ")


def test_get_product_unknown(conn):
    with pytest.raises(tools.NotFound):
        tools.get_product(conn, "NOPE-1")


def test_request_does_not_change_price(conn):
    req = tools.request_price_change(conn, "DAI-001", 3.99, "Competitor sells it for 3.99", "agent")
    assert req["status"] == "pending" and req["risk"] == "low"
    assert tools.get_product(conn, "DAI-001")["price"] == 4.29


def test_large_change_is_high_risk(conn):
    req = tools.request_price_change(conn, "DAI-001", 1.99, "Clearance before expiry date", "agent")
    assert req["risk"] == "high"


@pytest.mark.parametrize("price", [0, -1, "abc", float("nan"), float("inf"), 1e12])
def test_invalid_prices_rejected(conn, price):
    with pytest.raises(tools.ToolError):
        tools.request_price_change(conn, "DAI-001", price, "a valid enough reason", "agent")


def test_reason_required(conn):
    with pytest.raises(tools.ToolError):
        tools.request_price_change(conn, "DAI-001", 3.99, "cheaper", "agent")


def test_only_one_pending_request_per_product(conn):
    tools.request_price_change(conn, "DAI-001", 3.99, "Competitor sells it for 3.99", "agent")
    with pytest.raises(tools.Conflict, match="already has a pending request"):
        tools.request_price_change(conn, "DAI-001", 3.89, "Even cheaper competitor", "agent")


def test_approve_applies_price(conn):
    req = tools.request_price_change(conn, "DAI-001", 3.99, "Competitor sells it for 3.99", "agent")
    done = tools.decide_request(conn, req["id"], True, "maria", "ok")
    assert done["status"] == "approved" and done["decided_by"] == "maria"
    assert tools.get_product(conn, "DAI-001")["price"] == 3.99


def test_reject_keeps_price(conn):
    req = tools.request_price_change(conn, "DAI-001", 3.99, "Competitor sells it for 3.99", "agent")
    tools.decide_request(conn, req["id"], False, "maria", "no")
    assert tools.get_product(conn, "DAI-001")["price"] == 4.29


def test_cannot_decide_twice(conn):
    req = tools.request_price_change(conn, "DAI-001", 3.99, "Competitor sells it for 3.99", "agent")
    tools.decide_request(conn, req["id"], True, "maria")
    with pytest.raises(tools.Conflict, match="already approved"):
        tools.decide_request(conn, req["id"], False, "maria")


def test_requester_cannot_approve_own_request(conn):
    req = tools.request_price_change(conn, "DAI-001", 3.99, "Competitor sells it for 3.99", "agent")
    with pytest.raises(tools.Conflict):
        tools.decide_request(conn, req["id"], True, "agent")


def test_stale_request_cannot_be_approved(conn):
    req = tools.request_price_change(conn, "DAI-001", 3.99, "Competitor sells it for 3.99", "agent")
    conn.execute("UPDATE products SET price = 4.49 WHERE sku = 'DAI-001'")  # changed elsewhere
    with pytest.raises(tools.Conflict, match="now costs"):
        tools.decide_request(conn, req["id"], True, "maria")
    assert tools.get_product(conn, "DAI-001")["price"] == 4.49
