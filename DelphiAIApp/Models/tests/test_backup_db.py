import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.backup_db import (
    build_pg_dump_cmd,
    backup_filename,
    split_dsn_password,
    redact,
)


def test_build_cmd_uses_database_url_without_password():
    cmd = build_pg_dump_cmd("postgres://u:secret@host/db", "/tmp/out.sql.gz")
    assert cmd[0] == "pg_dump"
    assert "-Fc" in cmd
    # The password must NOT appear in argv (it goes via PGPASSWORD instead).
    assert not any("secret" in part for part in cmd)
    assert "postgres://u@host/db" in cmd


def test_split_dsn_password_strips_secret():
    sanitized, pw = split_dsn_password("postgres://u:secret@host:5432/db?sslmode=require")
    assert pw == "secret"
    assert "secret" not in sanitized
    assert sanitized == "postgres://u@host:5432/db?sslmode=require"


def test_split_dsn_password_handles_no_password():
    sanitized, pw = split_dsn_password("postgres://host/db")
    assert pw is None
    assert sanitized == "postgres://host/db"


def test_redact_masks_credentials():
    assert redact("connection to postgresql://u:secret@host failed") == (
        "connection to postgresql://u:***@host failed"
    )


def test_backup_filename_is_timestamped():
    name = backup_filename("delphi")
    assert name.startswith("delphi-")
    assert name.endswith(".dump")
