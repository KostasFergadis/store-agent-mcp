import psycopg

from store_agent.db import make_pool


def test_pool_replaces_connection_killed_by_server(database):
    # Restarting Postgres (e.g. `docker compose up --build`) drops every idle
    # connection. The next call must get a fresh one, not the dead one.
    pool = make_pool(database)
    try:
        pool.wait()
        with pool.connection() as c:
            pid = c.execute("SELECT pg_backend_pid() AS pid").fetchone()["pid"]

        with psycopg.connect(database, autocommit=True) as admin:
            admin.execute("SELECT pg_terminate_backend(%s)", (pid,))

        with pool.connection() as c:
            assert c.execute("SELECT 1 AS ok").fetchone()["ok"] == 1
    finally:
        pool.close()
