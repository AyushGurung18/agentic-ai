"""
app/db/database.py
──────────────────
Thin psycopg connection pool + schema bootstrap.

Usage:
    from app.db.database import get_conn, init_db

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
"""

import os
import pathlib
import psycopg
from psycopg_pool import ConnectionPool
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# ── Connection string ─────────────────────────────────────────────────────────
# Strip the SQLAlchemy prefix so plain psycopg can use it.
_RAW_URL: str = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://ayush:password@localhost:5432/octo",
).replace("postgresql+psycopg://", "postgresql://")

# ── Pool (created lazily on first call so import order doesn't matter) ────────
_pool: ConnectionPool | None = None

def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=_RAW_URL,
            min_size=1,
            max_size=10,
            open=True,
            kwargs={"connect_timeout": 10},
        )
    return _pool


def get_conn() -> psycopg.Connection:
    """Return a connection from the pool (use as context manager)."""
    return _get_pool().connection()


# ── Schema bootstrap ──────────────────────────────────────────────────────────
_SQL_PATH = pathlib.Path(__file__).parent / "init.sql"

def init_db() -> None:
    """
    Run init.sql against the DB.
    All statements are IF NOT EXISTS — safe to call every startup.
    """
    try:
        sql = _SQL_PATH.read_text()
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
        print("✅ Database schema initialised on Supabase")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
