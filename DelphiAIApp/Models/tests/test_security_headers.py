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


def test_oversized_body_rejected(client):
    big = "x" * (300 * 1024)  # 300 KB > 256 KB cap
    r = client.post(
        "/api/results/update",
        data=big,
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 413


@pytest.fixture
def admin_client(monkeypatch):
    """Client with both a read key and a separate admin key configured."""
    monkeypatch.setenv("DELPHI_DISABLE_RATELIMIT", "1")
    monkeypatch.setenv("DELPHI_API_KEY", "read-key")
    monkeypatch.setenv("DELPHI_ADMIN_API_KEY", "admin-key")
    monkeypatch.delenv("DELPHI_ENV", raising=False)
    import DelphiAIApp.security as sec
    importlib.reload(sec)
    import DelphiAIApp.main as main
    importlib.reload(main)
    yield TestClient(main.app)
    # Reset module state for other tests.
    monkeypatch.delenv("DELPHI_API_KEY", raising=False)
    monkeypatch.delenv("DELPHI_ADMIN_API_KEY", raising=False)
    importlib.reload(sec)


def test_admin_endpoint_rejects_read_key_only(admin_client):
    # Valid read key but no admin key -> 401 on an admin (expensive) route.
    r = admin_client.post(
        "/api/results/update",
        json={"event_name": "UFC 300"},
        headers={"X-API-Key": "read-key"},
    )
    assert r.status_code == 401
    assert "admin key" in r.json()["detail"].lower()


def test_admin_endpoint_accepts_admin_key(admin_client):
    # With both keys present, the admin check passes (any non-auth failure
    # downstream is fine — we only assert it's NOT a 401 from the admin gate).
    r = admin_client.post(
        "/api/results/update",
        json={"event_name": "UFC 300"},
        headers={"X-API-Key": "read-key", "X-Admin-Key": "admin-key"},
    )
    assert r.status_code != 401


def test_read_endpoint_unaffected_by_admin_gate(admin_client):
    # Non-admin routes need only the read key, not the admin key.
    r = admin_client.get(
        "/api/performance/summary",
        headers={"X-API-Key": "read-key"},
    )
    assert r.status_code != 401
