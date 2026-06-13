"""
PostgreSQL Database Connection Module
"""
import os
from pathlib import Path
from contextlib import contextmanager
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables from a local .env file when present. In container
# deployments (Railway, GitHub Actions) the vars are injected directly into the
# environment, so the file is optional — never raise if it is missing.
env_path = Path(__file__).resolve().parents[3] / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Connection config is resolved in priority order:
#   1. DATABASE_URL  — a single libpq connection string. This is what Neon,
#      Railway, Supabase, and Vercel all hand you, and it carries sslmode itself
#      (e.g. ...?sslmode=require). Used verbatim when present.
#   2. DB_* vars     — the local-dev fallback (docker-compose Postgres on 5433).
#      An optional DB_SSLMODE lets local point at a managed DB without a URL.
#
# _POOL_ARGS / _POOL_KWARGS are spread into ThreadedConnectionPool so either
# form works through the same code path.
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

if DATABASE_URL:
    _POOL_ARGS: tuple = (DATABASE_URL,)
    # Keepalives prevent Neon from silently dropping idle connections.
    # After 60 s idle the OS probes every 10 s; 5 missed probes = discard.
    _POOL_KWARGS: dict = {
        "keepalives": 1,
        "keepalives_idle": 60,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    }
    DB_CONFIG: dict = {}  # unused in URL mode; defined so helpers can reference it
else:
    REQUIRED_ENV_VARS = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        raise EnvironmentError(
            "Set DATABASE_URL, or provide all of: "
            f"{', '.join(missing)} (missing)."
        )
    DB_CONFIG = {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "database": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }
    # Optional: require SSL when pointing the DB_* path at a managed Postgres.
    _sslmode = os.getenv("DB_SSLMODE", "").strip()
    if _sslmode:
        DB_CONFIG["sslmode"] = _sslmode
    _POOL_ARGS = ()
    _POOL_KWARGS = DB_CONFIG

# Connection pool (initialized lazily)
_connection_pool = None


def get_connection_pool(minconn=5, maxconn=20):
    """Get or create the connection pool.

    Defaults are sized for the FastAPI workload: a warm pool of 5 connections
    eliminates first-request latency, and 20 max prevents stalls when the UI
    fires several /api requests in parallel (dashboard does this).
    """
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = pool.ThreadedConnectionPool(
            minconn, maxconn, *_POOL_ARGS, **_POOL_KWARGS
        )
    return _connection_pool


@contextmanager
def get_db_connection():
    """Context manager for database connections.

    Creates a fresh connection per call and closes it on exit. This is the
    correct pattern for Neon (serverless Postgres) — Neon is designed for
    short-lived connections and aggressively closes idle pool connections,
    making persistent pools unreliable without their own PgBouncer endpoint.
    """
    if DATABASE_URL:
        conn = psycopg2.connect(
            DATABASE_URL,
            keepalives=1,
            keepalives_idle=60,
            keepalives_interval=10,
            keepalives_count=5,
        )
    else:
        conn = psycopg2.connect(
            **{
                **DB_CONFIG,
                "keepalives": 1,
                "keepalives_idle": 60,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            }
        )
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


@contextmanager
def get_db_cursor(commit=True):
    """
    Context manager for database cursor with automatic commit.
    Returns results as dictionaries.
    
    Usage:
        with get_db_cursor() as cur:
            cur.execute("SELECT * FROM FighterStats WHERE Name = %s", ("Jon Jones",))
            fighter = cur.fetchone()
    """
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            yield cursor
            if commit:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()


def close_all_connections():
    """Close all connections in the pool."""
    global _connection_pool
    if _connection_pool:
        _connection_pool.closeall()
        _connection_pool = None


# Simple query helpers
def execute_query(query, params=None, fetch_one=False):
    """Execute a SELECT query and return results."""
    with get_db_cursor(commit=False) as cur:
        cur.execute(query, params)
        return cur.fetchone() if fetch_one else cur.fetchall()


def execute_write(query, params=None):
    """Execute an INSERT/UPDATE/DELETE query."""
    with get_db_cursor(commit=True) as cur:
        cur.execute(query, params)
        return cur.rowcount
