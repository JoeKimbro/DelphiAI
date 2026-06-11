"""Unit tests for the boot-time model fetcher's pure helpers."""
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.fetch_model import resolve_target, should_fetch


def test_should_fetch_when_missing(tmp_path):
    assert should_fetch(tmp_path / "nope.pkl", 168) is True


def test_no_fetch_when_fresh(tmp_path):
    f = tmp_path / "m.pkl"
    f.write_bytes(b"x")
    assert should_fetch(f, 168) is False


def test_fetch_when_stale(tmp_path):
    f = tmp_path / "m.pkl"
    f.write_bytes(b"x")
    old = time.time() - 200 * 3600  # ~8.3 days old
    os.utime(f, (old, old))
    assert should_fetch(f, 168) is True


def test_resolve_target_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DELPHI_MODEL_PATH", str(tmp_path / "x.pkl"))
    assert resolve_target() == tmp_path / "x.pkl"
