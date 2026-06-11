"""Security header + body-cap + prod-key tests via FastAPI TestClient."""
import importlib
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DELPHI_DISABLE_RATELIMIT", "1")
    monkeypatch.delenv("DELPHI_API_KEY", raising=False)
    monkeypatch.delenv("DELPHI_ENV", raising=False)
    import DelphiAIApp.security as sec
    importlib.reload(sec)
    import DelphiAIApp.main as main
    importlib.reload(main)
    return TestClient(main.app)


def test_security_headers_present(client):
    r = client.get("/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert "strict-transport-security" in r.headers
    assert "referrer-policy" in r.headers


def test_api_csp_is_locked_down(client):
    r = client.get("/health")
    assert "default-src 'none'" in r.headers["content-security-policy"]


def test_production_requires_api_key(monkeypatch):
    monkeypatch.setenv("DELPHI_ENV", "production")
    monkeypatch.delenv("DELPHI_API_KEY", raising=False)
    import DelphiAIApp.security as sec
    with pytest.raises(RuntimeError):
        importlib.reload(sec)
    # Reset module state for other tests.
    monkeypatch.delenv("DELPHI_ENV", raising=False)
    importlib.reload(sec)
