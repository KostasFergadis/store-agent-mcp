"""Business logic shared by the MCP server (agent side) and the approvals API (human side).

Functions here take an open connection and know nothing about MCP or HTTP, so
they can be tested directly against Postgres.
"""

from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from psycopg import Connection, errors, sql

from .embeddings import Embedder, to_pgvector

MAX_PRICE = Decimal(100000)
DIETS = ("vegan", "vegetarian", "dairy-free")


class ToolError(Exception):
    """A request the system refuses: bad input or a rule it would break. Shown to the agent as-is."""


class NotFound(ToolError):
    pass


class Conflict(ToolError):
    pass


def _plain(row: dict | None) -> dict | None:
    if row is None:
        return None
    def conv(v):
        if isinstance(v, Decimal):
            return float(v)
        if isinstance(v, datetime):
            return v.isoformat()
        return v

    return {k: conv(v) for k, v in row.items()}


def _money(value) -> Decimal:
    try:
        d = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except InvalidOperation:
        raise ToolError(f"invalid price {value!r}") from None
    if not d.is_finite() or d <= 0:
        raise ToolError("price must be a positive amount")
    if d >= MAX_PRICE:
        raise ToolError(f"price must be below {MAX_PRICE}")
    return d


# --- agent-facing reads -------------------------------------------------------


def search_products(
    conn: Connection,
    embedder: Embedder,
    query: str,
    limit: int = 5,
    max_price: float | None = None,
    in_stock_only: bool = False,
    diet: list[str] | None = None,
) -> list[dict]:
    if not query.strip():
        raise ToolError("query must not be empty")
    diet = list(diet or [])
    if unknown := [d for d in diet if d not in DIETS]:
        raise ToolError(f"unknown diet {', '.join(unknown)}; use any of {', '.join(DIETS)}")
    limit = max(1, min(int(limit), 20))
    vec = to_pgvector(embedder.embed_query(query))
    rows = conn.execute(
        """
        SELECT sku, name, category, price, stock,
               round((1 - (embedding <=> %(v)s::vector))::numeric, 3) AS similarity
        FROM products
        WHERE (%(max_price)s::numeric IS NULL OR price <= %(max_price)s::numeric)
          AND (NOT %(in_stock)s OR stock > 0)
          AND diet @> %(diet)s::text[]
        ORDER BY embedding <=> %(v)s::vector
        LIMIT %(limit)s
        """,
        {"v": vec, "max_price": max_price, "in_stock": in_stock_only, "diet": diet, "limit": limit},
    ).fetchall()
    return [_plain(r) for r in rows]


def get_product(conn: Connection, sku: str) -> dict:
    row = conn.execute(
        "SELECT sku, name, category, description, price, stock, diet, updated_at FROM products WHERE sku = %s",
        (sku,),
    ).fetchone()
    if row is None:
        raise NotFound(f"no product with sku {sku!r}")
    return _plain(row)


def list_products(conn: Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT sku, name, category, price, stock, updated_at FROM products ORDER BY category, name"
    ).fetchall()
    return [_plain(r) for r in rows]


# --- price changes: the agent proposes, a human decides ----------------------

_REQUEST_COLS = (
    "id, sku, old_price, new_price, reason, risk, status, requested_by, "
    "decided_by, decision_note, created_at, decided_at"
)
_COLS = sql.SQL(_REQUEST_COLS)


def request_price_change(
    conn: Connection,
    sku: str,
    new_price: float,
    reason: str,
    requested_by: str,
    high_risk_threshold: float = 0.30,
) -> dict:
    price = _money(new_price)
    reason = reason.strip()
    if len(reason) < 10:
        raise ToolError("give a reason of at least 10 characters; the approver will read it")

    with conn.transaction():
        product = conn.execute("SELECT price FROM products WHERE sku = %s", (sku,)).fetchone()
        if product is None:
            raise NotFound(f"no product with sku {sku!r}")
        old = product["price"]
        if price == old:
            raise ToolError(f"{sku} already costs {old}")
        change = abs(price - old) / old
        risk = "high" if change > Decimal(str(high_risk_threshold)) else "low"
        try:
            with conn.transaction():  # savepoint, so a violation doesn't poison the outer tx
                row = conn.execute(
                    sql.SQL("""
                    INSERT INTO price_change_requests (sku, old_price, new_price, reason, risk, requested_by)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING {cols}
                    """).format(cols=_COLS),
                    (sku, old, price, reason, risk, requested_by),
                ).fetchone()
        except errors.UniqueViolation:
            pending = conn.execute(
                "SELECT id FROM price_change_requests WHERE sku = %s AND status = 'pending'", (sku,)
            ).fetchone()
            raise Conflict(
                f"{sku} already has a pending request (#{pending['id'] if pending else '?'}); "
                "wait for a decision before proposing another"
            ) from None

    out = _plain(row)
    out["change_pct"] = round(float(change) * 100, 1)
    out["message"] = "Pending human approval. The price has NOT changed yet."
    return out


def get_request(conn: Connection, request_id: int) -> dict:
    row = conn.execute(
        sql.SQL("SELECT {cols} FROM price_change_requests WHERE id = %s").format(cols=_COLS),
        (request_id,),
    ).fetchone()
    if row is None:
        raise NotFound(f"no price change request #{request_id}")
    return _plain(row)


def list_requests(conn: Connection, status: str | None = None, limit: int = 50) -> list[dict]:
    if status is not None and status not in ("pending", "approved", "rejected"):
        raise ToolError("status must be pending, approved or rejected")
    rows = conn.execute(
        sql.SQL("""
        SELECT {cols} FROM price_change_requests
        WHERE %(status)s::text IS NULL OR status = %(status)s::text
        ORDER BY id DESC LIMIT %(limit)s
        """).format(cols=_COLS),
        {"status": status, "limit": max(1, min(int(limit), 200))},
    ).fetchall()
    return [_plain(r) for r in rows]


def decide_request(
    conn: Connection, request_id: int, approve: bool, decided_by: str, note: str | None = None
) -> dict:
    """Human-only. Not exposed as an MCP tool, so the agent can never approve its own request."""
    decided_by = decided_by.strip()
    if not decided_by:
        raise ToolError("decided_by is required")

    with conn.transaction():
        req = conn.execute(
            "SELECT id, sku, old_price, new_price, status, requested_by "
            "FROM price_change_requests WHERE id = %s FOR UPDATE",
            (request_id,),
        ).fetchone()
        if req is None:
            raise NotFound(f"no price change request #{request_id}")
        if req["status"] != "pending":
            raise Conflict(f"request #{request_id} is already {req['status']}")
        if decided_by == req["requested_by"]:
            raise Conflict("a request can't be decided by the one who made it")

        if approve:
            current = conn.execute(
                "SELECT price FROM products WHERE sku = %s FOR UPDATE", (req["sku"],)
            ).fetchone()
            if current["price"] != req["old_price"]:
                raise Conflict(
                    f"{req['sku']} now costs {current['price']}, not {req['old_price']} as when this was "
                    "requested; reject it and let the agent propose again"
                )
            conn.execute(
                "UPDATE products SET price = %s, updated_at = now() WHERE sku = %s",
                (req["new_price"], req["sku"]),
            )

        row = conn.execute(
            sql.SQL("""
            UPDATE price_change_requests
            SET status = %s, decided_by = %s, decision_note = %s, decided_at = now()
            WHERE id = %s
            RETURNING {cols}
            """).format(cols=_COLS),
            ("approved" if approve else "rejected", decided_by, note, request_id),
        ).fetchone()
    return _plain(row)
