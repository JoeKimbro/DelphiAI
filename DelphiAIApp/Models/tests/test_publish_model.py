"""Unit tests for the R2 publisher's credential/kwarg builder (no network)."""
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.publish_model import build_client_kwargs


def test_build_client_kwargs_from_env(monkeypatch):
    monkeypatch.setenv("R2_ENDPOINT_URL", "https://acct.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "id")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("R2_BUCKET", "models")
    kw = build_client_kwargs()
    assert kw["service_name"] == "s3"
    assert kw["endpoint_url"].endswith("cloudflarestorage.com")
    assert kw["region_name"] == "auto"


def test_build_client_kwargs_missing_raises(monkeypatch):
    for k in ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(EnvironmentError):
        build_client_kwargs()
