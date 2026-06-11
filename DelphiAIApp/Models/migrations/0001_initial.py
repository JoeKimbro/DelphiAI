"""yoyo baseline: the full current schema (schemas.sql + schemas_auth.sql).

The pre-yoyo deltas in db/migrations/001-006 are already folded into these two
files, so this single baseline reproduces a complete, current database on a
fresh server (Neon/Railway). Executed as a Python step (not raw .sql) so
psycopg2 runs each file's statements in one shot — no statement-splitting.

Forward rule: never edit this file or schemas.sql for new changes. Add a new
numbered migration (0002_*.py / 0002_*.sql) instead.
"""
from pathlib import Path

from yoyo import step

_DB = Path(__file__).resolve().parents[1] / "db"


def apply_step(conn):
    cur = conn.cursor()
    for fname in ("schemas.sql", "schemas_auth.sql"):
        cur.execute((_DB / fname).read_text(encoding="utf-8"))


steps = [step(apply_step)]
