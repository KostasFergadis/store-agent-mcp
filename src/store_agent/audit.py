"""Append-only, hash-chained audit log.

Each entry's hash covers its own content plus the previous entry's hash. If
someone edits or deletes a row in the middle, `verify` finds the first row
where the chain no longer adds up.

Limits: someone with write access could rewrite the whole chain from the
tampered row onwards. Closing that gap means periodically copying the latest
hash somewhere they can't write to (another system, a signed export).
"""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any

from psycopg import Connection
from psycopg.types.json import Jsonb

GENESIS = "0" * 64
# Serialises appends so two writers can't both chain onto the same previous row.
_LOCK_KEY = 815_001
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def to_json_safe(value: Any) -> Any:
    """Round-trip through JSON so what we hash is exactly what jsonb gives back."""
    return json.loads(json.dumps(value, default=str))


def _entry(ts: datetime, actor, tool, args, outcome, result, error, duration_ms) -> dict:
    return {
        "ts_us": (ts - _EPOCH) // timedelta(microseconds=1),
        "actor": actor,
        "tool": tool,
        "args": args,
        "outcome": outcome,
        "result": result,
        "error": error,
        "duration_ms": duration_ms,
    }


def compute_hash(prev_hash: str, entry: dict) -> str:
    canonical = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((prev_hash + canonical).encode()).hexdigest()


def append(
    conn: Connection,
    *,
    actor: str,
    tool: str,
    args: dict,
    outcome: str,
    result: Any = None,
    error: str | None = None,
    duration_ms: int = 0,
) -> str:
    ts = datetime.now(UTC)
    args = to_json_safe(args)
    result = to_json_safe(result)
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_KEY,))
        row = conn.execute("SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
        prev = row["hash"] if row else GENESIS
        h = compute_hash(prev, _entry(ts, actor, tool, args, outcome, result, error, duration_ms))
        conn.execute(
            """
            INSERT INTO audit_log (ts, actor, tool, args, outcome, result, error, duration_ms, prev_hash, hash)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (ts, actor, tool, Jsonb(args), outcome, Jsonb(result), error, duration_ms, prev, h),
        )
    return h


def recent(conn: Connection, limit: int = 50) -> list[dict]:
    rows = conn.execute(
        "SELECT id, ts, actor, tool, args, outcome, result, error, duration_ms, hash "
        "FROM audit_log ORDER BY id DESC LIMIT %s",
        (limit,),
    ).fetchall()
    return to_json_safe(rows)


def verify(conn: Connection) -> dict:
    prev = GENESIS
    checked = 0
    with conn.cursor(name="audit_verify") as cur:  # server-side cursor: constant memory
        cur.execute(
            "SELECT id, ts, actor, tool, args, outcome, result, error, duration_ms, prev_hash, hash "
            "FROM audit_log ORDER BY id"
        )
        for r in cur:
            entry = _entry(
                r["ts"], r["actor"], r["tool"], r["args"], r["outcome"], r["result"], r["error"], r["duration_ms"]
            )
            expected = compute_hash(prev, entry)
            if r["prev_hash"] != prev or r["hash"] != expected:
                return {"valid": False, "checked": checked, "first_bad_id": r["id"]}
            prev = r["hash"]
            checked += 1
    return {"valid": True, "checked": checked, "first_bad_id": None}
