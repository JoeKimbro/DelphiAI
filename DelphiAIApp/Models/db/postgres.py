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

# Load environment variables from .env file
# Looks for .env in project root (3 levels up from this file)
env_path = Path(__file__).resolve().parents[3] / ".env"
if not env_path.exists():
    raise FileNotFoundError(f".env file not found at {env_path}")
load_dotenv(env_path)

# Required environment variables
REQUIRED_ENV_VARS = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

# Database configuration - requires .env file
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

# Connection pool (initialized lazily)
_connection_pool = None


def get_connection_pool(minconn=1, maxconn=10):
    """Get or create the connection pool."""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = pool.ThreadedConnectionPool(
            minconn, maxconn, **DB_CONFIG
        )
    return _connection_pool


@contextmanager
def get_db_connection():
    """
    Context manager for database connections.
    Automatically returns connection to pool when done.
    
    Usage:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM FighterStats")
                results = cur.fetchall()
    """
    conn = None
    try:
        conn = get_connection_pool().getconn()
        yield conn
    finally:
        if conn:
            get_connection_pool().putconn(conn)


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
