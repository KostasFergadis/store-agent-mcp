import asyncio

import pytest
from fastapi.testclient import TestClient
from mcp.server.mcpserver.exceptions import ToolError

from store_agent import audit
from store_agent.approvals_api import create_app
from store_agent.mcp_server import build_server


def call(server, name, args):
    """Call a tool through the real MCP layer and return its structured result."""
    result = asyncio.run(server.call_tool(name, args))
    data = result.structured_content
    if isinstance(data, dict) and set(data) == {"result"}:  # SDK wraps non-object returns
        data = data["result"]
    return data


def test_every_tool_call_is_audited(settings, pool, embedder, conn):
    server = build_server(settings, pool, embedder)
    call(server, "search_products", {"query": "olive oil"})
    with pytest.raises(ToolError, match="NOPE-1"):  # the message reaches the model
        call(server, "get_product", {"sku": "NOPE-1"})

    rows = audit.recent(conn)
    assert [r["tool"] for r in rows] == ["get_product", "search_products"]
    assert rows[0]["outcome"] == "rejected" and "NOPE-1" in rows[0]["error"]
    assert rows[1]["outcome"] == "ok" and rows[1]["actor"] == "test-agent"
    assert rows[1]["args"]["query"] == "olive oil"


def test_mcp_search_accepts_diet_and_rejects_unknown(settings, pool, embedder):
    server = build_server(settings, pool, embedder)
    rows = call(server, "search_products", {"query": "dinner", "limit": 20, "diet": ["vegetarian"]})
    assert rows and not any(r["category"] == "Meat & Fish" for r in rows)
    with pytest.raises(ToolError):
        call(server, "search_products", {"query": "dinner", "diet": ["keto"]})


def test_mcp_exposes_no_way_to_approve(settings, pool, embedder):
    server = build_server(settings, pool, embedder)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert "request_price_change" in names
    assert not any("approve" in n or "decide" in n or "set_price" in n for n in names)


def test_chain_verifies_and_detects_tampering(settings, pool, embedder, conn):
    server = build_server(settings, pool, embedder)
    for q in ["yogurt", "bread", "coffee"]:
        call(server, "search_products", {"query": q})
    assert audit.verify(conn) == {"valid": True, "checked": 3, "first_bad_id": None}

    conn.execute("UPDATE audit_log SET args = '{\"query\": \"something else\"}' WHERE id = 2")
    assert audit.verify(conn) == {"valid": False, "checked": 1, "first_bad_id": 2}


def test_chain_detects_deleted_row(settings, pool, embedder, conn):
    server = build_server(settings, pool, embedder)
    for q in ["yogurt", "bread", "coffee"]:
        call(server, "search_products", {"query": q})
    conn.execute("DELETE FROM audit_log WHERE id = 2")
    assert audit.verify(conn)["first_bad_id"] == 3


def test_full_loop_agent_requests_human_approves(settings, pool, embedder, conn):
    server = build_server(settings, pool, embedder)
    result = call(server, "request_price_change",
                  {"sku": "PAN-001", "new_price": 10.90, "reason": "Two competitors are at 10.90 this week"})
    assert result["status"] == "pending"
    req_id = result["id"]

    with TestClient(create_app(settings)) as api:
        h = {"X-API-Key": "test-key"}
        assert api.get("/requests", headers={"X-API-Key": "wrong"}).status_code == 401
        assert api.get("/requests").status_code == 401
        pending = api.get("/requests", headers=h).json()
        assert [r["id"] for r in pending] == [req_id]

        r = api.post(f"/requests/{req_id}/approve", headers=h, json={"approver": "maria", "note": "fine"})
        assert r.status_code == 200 and r.json()["status"] == "approved"
        assert api.post(f"/requests/{req_id}/approve", headers=h, json={"approver": "maria"}).status_code == 409

        log = api.get("/audit", headers=h).json()
        assert [(e["actor"], e["tool"], e["outcome"]) for e in log] == [
            ("maria", "approve_request", "rejected"),
            ("maria", "approve_request", "ok"),
            ("test-agent", "request_price_change", "ok"),
        ]
        assert api.get("/audit/verify", headers=h).json()["valid"] is True

    status = call(server, "get_price_change_request", {"request_id": req_id})
    assert status["status"] == "approved" and status["decided_by"] == "maria"
    assert call(server, "get_product", {"sku": "PAN-001"})["price"] == 10.90


def test_parallel_first_searches_load_model_once(settings, pool, monkeypatch):
    import threading
    import time

    from store_agent import mcp_server
    from store_agent.embeddings import FakeEmbedder

    loads = []

    def slow_factory(kind):
        loads.append(threading.get_ident())
        time.sleep(0.3)  # like loading a real model
        return FakeEmbedder()

    monkeypatch.setattr(mcp_server, "make_embedder", slow_factory)
    server = build_server(settings, pool)  # no embedder given: loads lazily

    async def many():
        await asyncio.gather(*(server.call_tool("search_products", {"query": q})
                               for q in ["feta", "bread", "coffee", "honey", "oats"]))

    asyncio.run(many())
    assert len(loads) == 1


def test_decision_rolls_back_if_audit_fails(settings, pool, embedder, conn, monkeypatch):
    from store_agent import approvals_api, tools

    req = tools.request_price_change(conn, "PAN-001", 10.90, "Two competitors are at 10.90", "agent")
    conn.commit()

    def broken_append(*a, **kw):
        raise RuntimeError("audit storage down")

    monkeypatch.setattr(approvals_api.audit, "append", broken_append)
    with TestClient(create_app(settings), raise_server_exceptions=False) as api:
        r = api.post(f"/requests/{req['id']}/approve", headers={"X-API-Key": "test-key"}, json={"approver": "maria"})
    assert r.status_code == 500
    conn.rollback()
    assert tools.get_request(conn, req["id"])["status"] == "pending"
    assert tools.get_product(conn, "PAN-001")["price"] == 11.90  # price unchanged
