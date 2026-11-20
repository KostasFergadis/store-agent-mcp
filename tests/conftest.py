"""Tests run against a real Postgres + pgvector (the docker compose `db` service).

A separate `store_test` database is recreated from db/schema.sql once per run
and emptied before each test.
"""

import json
import os
from dataclasses import replace
from pathlib import Path

import psycopg
import pytest
from psycopg.rows import dict_row

from store_agent.config import get_settings
from store_agent.db import make_pool
from store_agent.embeddings import FakeEmbedder
from store_agent.seed import seed

ROOT = Path(__file__).resolve().parents[1]
ADMIN_URL = os.environ.get("TEST_ADMIN_DATABASE_URL", "postgresql://store:store@localhost:5544/postgres")
TEST_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://store:store@localhost:5544/store_test")


@pytest.fixture(scope="session")
def database():
    with psycopg.connect(ADMIN_URL, autocommit=True) as admin:
        admin.execute("DROP DATABASE IF EXISTS store_test WITH (FORCE)")
        admin.execute("CREATE DATABASE store_test")
    with psycopg.connect(TEST_URL, autocommit=True) as conn:
        conn.execute((ROOT / "db" / "schema.sql").read_text())
    return TEST_URL


@pytest.fixture
def settings(database):
    return replace(get_settings(), database_url=database, embedder="fake", agent_id="test-agent",
                   approver_api_key="test-key")


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def conn(database, embedder):
    with psycopg.connect(database, row_factory=dict_row) as c:
        c.execute("TRUNCATE audit_log, price_change_requests, products RESTART IDENTITY CASCADE")
        products = json.loads((ROOT / "data" / "products.json").read_text())
        seed(c, products, embedder)
        c.commit()
        yield c


@pytest.fixture
def pool(conn, database):  # depends on conn so each test starts from fresh data
    p = make_pool(database)
    yield p
    p.close()
