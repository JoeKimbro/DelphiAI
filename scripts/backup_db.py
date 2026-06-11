"""Logical Postgres backup via pg_dump (custom format).

Resolves the connection from DATABASE_URL (Neon/Railway) or DB_* env vars
(local docker-compose). Writes a timestamped .dump to BACKUP_DIR (default
./backups) and prunes dumps older than BACKUP_RETENTION_DAYS.

Usage:
    python -m scripts.backup_db
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


def backup_filename(db_name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{db_name}-{stamp}.dump"


def _dsn_from_env() -> tuple[str, str]:
    """Return (dsn, db_name) from DATABASE_URL or DB_* vars."""
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        from urllib.parse import urlparse
        name = urlparse(url).path.lstrip("/") or "database"
        return url, name
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5433")
    name = os.getenv("DB_NAME", "delphi_db")
    user = os.getenv("DB_USER", "")
    pw = os.getenv("DB_PASSWORD", "")
    dsn = f"postgresql://{user}:{pw}@{host}:{port}/{name}"
    return dsn, name


def build_pg_dump_cmd(dsn: str, out_path: str) -> list[str]:
    # -Fc = custom (compressed, restorable with pg_restore); -f = output file.
    return ["pg_dump", "-Fc", "-f", out_path, dsn]


def prune_old(backup_dir: Path, retention_days: int) -> int:
    cutoff = time.time() - retention_days * 86400
    removed = 0
    for f in backup_dir.glob("*.dump"):
        if f.stat().st_mtime < cutoff:
            f.unlink()
            removed += 1
    return removed


def main() -> int:
    backup_dir = Path(os.getenv("BACKUP_DIR", "backups"))
    backup_dir.mkdir(parents=True, exist_ok=True)
    retention = int(os.getenv("BACKUP_RETENTION_DAYS", "14"))

    dsn, db_name = _dsn_from_env()
    out_path = backup_dir / backup_filename(db_name)
    cmd = build_pg_dump_cmd(dsn, str(out_path))

    print(f"[backup] writing {out_path} ...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[backup] FAILED: {result.stderr}", file=sys.stderr)
        return result.returncode
    print(f"[backup] OK ({out_path.stat().st_size} bytes)", flush=True)

    pruned = prune_old(backup_dir, retention)
    if pruned:
        print(f"[backup] pruned {pruned} dump(s) older than {retention}d", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
