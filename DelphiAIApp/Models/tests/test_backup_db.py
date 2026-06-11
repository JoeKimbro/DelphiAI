import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.backup_db import build_pg_dump_cmd, backup_filename


def test_build_cmd_uses_database_url():
    cmd = build_pg_dump_cmd("postgres://u:p@host/db", "/tmp/out.sql.gz")
    assert cmd[0] == "pg_dump"
    assert "postgres://u:p@host/db" in cmd
    assert "-Fc" in cmd


def test_backup_filename_is_timestamped():
    name = backup_filename("delphi")
    assert name.startswith("delphi-")
    assert name.endswith(".dump")
