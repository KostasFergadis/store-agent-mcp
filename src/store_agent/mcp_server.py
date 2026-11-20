"""MCP server: the tools an AI agent (Claude Code, Claude Desktop, ...) can call.

Every call goes through `audited`, which writes one audit_log row per call
whether it succeeds, is refused, or crashes. The agent can read the catalog
and *propose* price changes; it has no tool that changes a price directly.
"""

import argparse
import functools
import logging
import sys
import threading
import time
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError as MCPToolError
from mcp_types import ToolAnnotations
from psycopg_pool import ConnectionPool

from . import audit, tools
from .config import Settings, get_settings
from .db import make_pool
from .embeddings import Embedder, make_embedder

log = logging.getLogger("store_agent")

INSTRUCTIONS = """\
Tools for a grocery retailer's product catalog.
- Use search_products to find products by meaning, not exact names.
- You cannot change prices. request_price_change creates a request that a
  human approves or rejects; tell the user it is pending, and use
  get_price_change_request later to see the decision.
- Every call you make is recorded in an audit log.
"""


def build_server(settings: Settings, pool: ConnectionPool, embedder: Embedder | None = None) -> MCPServer:
    mcp = MCPServer("store-agent", instructions=INSTRUCTIONS)
    _embedder = embedder
    _embedder_lock = threading.Lock()

    def get_embedder() -> Embedder:
        # Loading the model takes a second or two, so do it on first search, once.
        # Agents call tools in parallel; without the lock each concurrent first
        # search loaded its own copy (seen in the end-to-end run).
        nonlocal _embedder
        if _embedder is None:
            with _embedder_lock:
                if _embedder is None:
                    _embedder = make_embedder(settings.embedder)
        return _embedder

    def audited(fn):
        @functools.wraps(fn)
        def wrapper(**kwargs):
            start = time.perf_counter()
            outcome, result, error = "ok", None, None
            try:
                result = fn(**kwargs)
                return result
            except tools.ToolError as e:
                # Re-raise as the SDK's ToolError so the model sees the message and can correct itself.
                outcome, error = "rejected", str(e)
                raise MCPToolError(str(e)) from e
            except Exception as e:
                outcome, error = "error", f"{type(e).__name__}: {e}"
                log.exception("tool %s failed", fn.__name__)
                raise
            finally:
                ms = int((time.perf_counter() - start) * 1000)
                try:
                    with pool.connection() as conn:
                        audit.append(
                            conn,
                            actor=settings.agent_id,
                            tool=fn.__name__,
                            args=kwargs,
                            outcome=outcome,
                            result=result,
                            error=error,
                            duration_ms=ms,
                        )
                except Exception:
                    # Fail loudly in the logs; the agent still gets its answer.
                    log.exception("could not write audit entry for %s", fn.__name__)

        return wrapper

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
    @audited
    def search_products(
        query: str,
        limit: int = 5,
        max_price: float | None = None,
        in_stock_only: bool = False,
        diet: list[Literal["vegan", "vegetarian", "dairy-free"]] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search over the catalog, e.g. "something for a greek salad" or "cheap breakfast".

        Set `diet` whenever the user rules something out (vegan, vegetarian, dairy-free):
        meaning-based search alone cannot exclude meat or dairy. Products must match every
        diet given.

        Returns sku, name, category, price (EUR), stock and a similarity score (0-1).
        """
        with pool.connection() as conn:
            return tools.search_products(conn, get_embedder(), query, limit, max_price, in_stock_only, diet)

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
    @audited
    def get_product(sku: str) -> dict[str, Any]:
        """Full details for one product, including its description."""
        with pool.connection() as conn:
            return tools.get_product(conn, sku)

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False))
    @audited
    def request_price_change(sku: str, new_price: float, reason: str) -> dict[str, Any]:
        """Propose a new price for a product. Does NOT change the price.

        A human reviews the request with your reason and approves or rejects it.
        Changes of more than 30% are flagged as high risk for the approver.
        Only one pending request per product is allowed.
        """
        with pool.connection() as conn:
            return tools.request_price_change(
                conn, sku, new_price, reason, settings.agent_id, settings.high_risk_threshold
            )

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
    @audited
    def get_price_change_request(request_id: int) -> dict[str, Any]:
        """Check the status of a price change request: pending, approved or rejected (with the approver's note)."""
        with pool.connection() as conn:
            return tools.get_request(conn, request_id)

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
    @audited
    def list_price_change_requests(
        status: Literal["pending", "approved", "rejected"] | None = None,
    ) -> list[dict[str, Any]]:
        """List recent price change requests, optionally filtered by status (pending, approved, rejected)."""
        with pool.connection() as conn:
            return tools.list_requests(conn, status)

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="Store agent MCP server")
    parser.add_argument("--http", action="store_true", help="serve over streamable HTTP instead of stdio")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    # stdout is the MCP channel in stdio mode, so all logging goes to stderr.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s")

    settings = get_settings()
    pool = make_pool(settings.database_url)
    mcp = build_server(settings, pool)
    if args.http:
        mcp.run(transport="streamable-http", port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
