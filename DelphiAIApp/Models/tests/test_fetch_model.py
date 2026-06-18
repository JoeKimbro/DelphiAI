"""Unit tests for the boot-time model fetcher's pure helpers + integrity check."""
import hashlib
import os
import sys
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts import fetch_model
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


# ---------------------------------------------------------------------------
# Download integrity verification (no real network)
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=1):
        yield self._payload


@pytest.fixture
def fake_download(monkeypatch):
    """Patch requests.get so download() streams a known payload."""
    payload = b"pretend-model-bytes"

    def _get(url, **kwargs):
        # Guard the SSRF-relevant default while we're here.
        assert kwargs.get("allow_redirects") is False
        return _FakeResponse(payload)

    monkeypatch.setattr(fetch_model.requests, "get", _get)
    return payload


def test_download_writes_sidecar_with_digest(tmp_path, fake_download, monkeypatch):
    monkeypatch.delenv("MODEL_SHA256", raising=False)
    monkeypatch.delenv("DELPHI_MODEL_SHA256", raising=False)
    target = tmp_path / "model_latest.pkl"

    digest = fetch_model.download("https://r2.example/model", target)

    assert target.read_bytes() == fake_download
    assert digest == hashlib.sha256(fake_download).hexdigest()
    assert Path(str(target) + ".sha256").read_text().strip() == digest


def test_download_matching_pin_succeeds(tmp_path, fake_download, monkeypatch):
    monkeypatch.setenv("MODEL_SHA256", hashlib.sha256(fake_download).hexdigest())
    target = tmp_path / "model_latest.pkl"

    digest = fetch_model.download("https://r2.example/model", target)

    assert target.exists()
    assert digest == hashlib.sha256(fake_download).hexdigest()


def test_download_mismatched_pin_raises_and_cleans_up(tmp_path, fake_download, monkeypatch):
    monkeypatch.setenv("MODEL_SHA256", "deadbeef" * 8)
    target = tmp_path / "model_latest.pkl"

    with pytest.raises(ValueError, match="integrity check failed"):
        fetch_model.download("https://r2.example/model", target)

    # Neither the installed file nor the temp part should survive a bad download.
    assert not target.exists()
    assert not target.with_suffix(target.suffix + ".part").exists()
