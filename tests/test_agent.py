"""The agent loop, tested with a scripted model instead of a real LLM.

The last test starts the real MCP server over stdio, so it also checks that the
agent's tool calls reach the database and the audit log.
"""

import asyncio

from store_agent import audit
from store_agent.agent import run_agent, run_with_store


def scripted(*replies):
    """A fake model that returns the given replies in order and records what it was sent."""
    seen = []

    async def chat(messages, tools):
        seen.append([dict(m) for m in messages])
        return replies[len(seen) - 1]

    chat.seen = seen
    return chat


def tool_call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def answer(text):
    return {"role": "assistant", "content": text}


def test_runs_the_tool_and_returns_the_answer():
    chat = scripted(tool_call("search_products", query="feta"), answer("Feta is €5.49."))
    calls = []

    async def call_tool(name, args):
        calls.append((name, args))
        return '[{"sku": "DAI-002", "price": 5.49}]'

    result = asyncio.run(run_agent("price of feta?", chat, call_tool, tools=[]))

    assert result == "Feta is €5.49."
    assert calls == [("search_products", {"query": "feta"})]
    # The model's second turn sees the tool result.
    assert chat.seen[1][-1] == {"role": "tool", "tool_name": "search_products",
                                "content": '[{"sku": "DAI-002", "price": 5.49}]'}


def test_stops_at_the_step_limit():
    chat = scripted(*[tool_call("search_products", query="x")] * 10)

    async def call_tool(name, args):
        return "[]"

    result = asyncio.run(run_agent("loop forever", chat, call_tool, tools=[], max_steps=3))

    assert result == "Stopped after 3 steps without a final answer."
    assert len(chat.seen) == 3


def test_tool_errors_go_back_to_the_model():
    chat = scripted(tool_call("get_product", sku="NOPE"), answer("That product doesn't exist."))

    async def call_tool(name, args):
        return "ERROR: no product with sku NOPE"

    result = asyncio.run(run_agent("show NOPE", chat, call_tool, tools=[]))

    assert result == "That product doesn't exist."
    assert chat.seen[1][-1]["content"] == "ERROR: no product with sku NOPE"


def test_real_mcp_server_runs_tools_and_audits_them(database, conn):
    chat = scripted(
        tool_call("search_products", query="feta cheese", limit=3),
        tool_call("request_price_change", sku="DAI-002", new_price=4.99, reason="Summer promo"),
        answer("Requested €4.99 for feta. A person has to approve it."),
    )

    result = asyncio.run(run_with_store(
        "Propose a promo price for feta", chat, agent_id="test-local-agent",
        extra_env={"DATABASE_URL": database, "EMBEDDER": "fake"},
    ))

    assert result.startswith("Requested €4.99")
    assert '"status": "pending"' in chat.seen[2][-1]["content"]  # the request was made, not applied
    price = conn.execute("SELECT price FROM products WHERE sku = 'DAI-002'").fetchone()["price"]
    assert float(price) == 5.49
    rows = audit.recent(conn)
    assert [r["tool"] for r in rows] == ["request_price_change", "search_products"]
    assert {r["actor"] for r in rows} == {"test-local-agent"}
