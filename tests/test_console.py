from fastapi.testclient import TestClient

from store_agent.approvals_api import create_app


def test_console_page_is_served(settings, conn):
    with TestClient(create_app(settings)) as api:
        r = api.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Price approvals" in r.text
    assert r.headers["x-frame-options"] == "DENY"


def test_products_needs_key_and_lists_catalog(settings, conn):
    with TestClient(create_app(settings)) as api:
        assert api.get("/products").status_code == 401
        r = api.get("/products", headers={"X-API-Key": "test-key"})
    assert r.status_code == 200
    skus = {p["sku"] for p in r.json()}
    assert {"DAI-002", "PAN-001"} <= skus
    assert "embedding" not in r.json()[0]
