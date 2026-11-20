from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


def make_pool(database_url: str) -> ConnectionPool:
    return ConnectionPool(
        database_url,
        min_size=1,
        max_size=5,
        kwargs={"row_factory": dict_row},
        # Test each connection before handing it out, so one dropped by a
        # database restart is replaced instead of failing the next call.
        check=ConnectionPool.check_connection,
        open=True,
    )
