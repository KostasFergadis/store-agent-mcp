"""Approvals API: the human side of the loop.

A person reviews the agent's pending price change requests and approves or
rejects them. Decisions go into the same audit log as the agent's tool calls,
so one log shows who asked for what and who allowed it.

Auth is a single shared API key to keep the demo small. A real deployment
would identify each approver through SSO rather than trust the `approver`
field in the request body.
"""

import secrets
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from psycopg import Connection
from pydantic import BaseModel, Field

from . import audit, tools
from .config import Settings, get_settings
from .db import make_pool


class Decision(BaseModel):
    approver: str = Field(min_length=1, examples=["maria.k"])
    note: str | None = Field(default=None, examples=["Matches the competitor price, OK."])


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pool = make_pool(settings.database_url)
        yield
        app.state.pool.close()

    app = FastAPI(
        title="Store Agent: Approvals",
        description="Human-in-the-loop review of price changes proposed by the AI agent.",
        lifespan=lifespan,
    )

    def get_conn(request: Request):
        with request.app.state.pool.connection() as c:
            yield c

    Conn = Annotated[Connection, Depends(get_conn)]

    def require_key(x_api_key: Annotated[str | None, Header()] = None):
        if x_api_key is None or not secrets.compare_digest(x_api_key, settings.approver_api_key):
            raise HTTPException(401, "missing or invalid API key")

    auth = [Depends(require_key)]

    def http_error(e: tools.ToolError) -> HTTPException:
        status = 404 if isinstance(e, tools.NotFound) else 409 if isinstance(e, tools.Conflict) else 400
        return HTTPException(status, str(e))

    console = files(__package__).joinpath("static/console.html").read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def approver_console():
        # The page itself is public; every call it makes still needs the API key.
        # Other sites may not frame it, so a decision can't be clickjacked.
        return HTMLResponse(console, headers={
            "Content-Security-Policy": "frame-ancestors 'none'",
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "no-referrer",
        })

    @app.get("/health")
    def health(c: Conn):
        c.execute("SELECT 1")
        return {"ok": True}

    @app.get("/requests", dependencies=auth)
    def list_requests(c: Conn, status: Literal["pending", "approved", "rejected"] | None = "pending"):
        return tools.list_requests(c, status)

    @app.get("/products", dependencies=auth)
    def list_products(c: Conn):
        return tools.list_products(c)

    def decide(request_id: int, body: Decision, approve: bool, c) -> dict:
        tool = "approve_request" if approve else "reject_request"
        args = {"request_id": request_id, "note": body.note}
        try:
            # Decision and its audit entry commit together or not at all:
            # a price can never change without a matching audit row.
            with c.transaction():
                result = tools.decide_request(c, request_id, approve, body.approver, body.note)
                audit.append(c, actor=body.approver, tool=tool, args=args, outcome="ok", result=result)
        except tools.ToolError as e:
            audit.append(c, actor=body.approver, tool=tool, args=args, outcome="rejected", error=str(e))
            raise http_error(e) from None
        return result

    @app.post("/requests/{request_id}/approve", dependencies=auth)
    def approve(request_id: int, body: Decision, c: Conn):
        return decide(request_id, body, True, c)

    @app.post("/requests/{request_id}/reject", dependencies=auth)
    def reject(request_id: int, body: Decision, c: Conn):
        return decide(request_id, body, False, c)

    @app.get("/audit", dependencies=auth)
    def audit_log(c: Conn, limit: int = 50):
        return audit.recent(c, max(1, min(limit, 500)))

    @app.get("/audit/verify", dependencies=auth)
    def audit_verify(c: Conn):
        return audit.verify(c)

    return app
