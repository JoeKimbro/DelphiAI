"""Tests for the SSRF guard (no real network / DNS).

Run from DelphiAIApp/Models:
    python -m pytest ml/tests/test_net_guard.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml import net_guard
from ml.net_guard import assert_allowed_url, safe_get, DisallowedURLError


def _patch_resolve(monkeypatch, ip):
    """Force every hostname to resolve to `ip` (skips real DNS)."""
    monkeypatch.setattr(
        net_guard.socket,
        "getaddrinfo",
        lambda host, *a, **k: [(0, 0, 0, "", (ip, 0))],
    )


def test_rejects_non_http_scheme():
    for url in ("file:///etc/passwd", "ftp://ufc.com/x", "gopher://ufc.com"):
        with pytest.raises(DisallowedURLError):
            assert_allowed_url(url)


def test_rejects_host_not_on_allowlist():
    # Cloud metadata endpoint — blocked at the allowlist step.
    with pytest.raises(DisallowedURLError):
        assert_allowed_url("http://169.254.169.254/latest/meta-data/")
    with pytest.raises(DisallowedURLError):
        assert_allowed_url("https://evil.example.com/athlete/x")


def test_rejects_allowlisted_host_resolving_private(monkeypatch):
    # An allowlisted name that (via DNS rebinding) points at a private IP.
    _patch_resolve(monkeypatch, "127.0.0.1")
    with pytest.raises(DisallowedURLError):
        assert_allowed_url("http://www.ufc.com/athlete/x")


def test_allows_public_allowlisted_host(monkeypatch):
    _patch_resolve(monkeypatch, "93.184.216.34")  # public address
    assert_allowed_url("https://www.ufc.com/athlete/x")  # must not raise


def test_safe_get_blocks_redirect_to_internal(monkeypatch):
    _patch_resolve(monkeypatch, "93.184.216.34")  # initial host is public

    class _Redirect:
        is_redirect = True
        status_code = 302
        headers = {"location": "http://169.254.169.254/latest/"}

    monkeypatch.setattr(net_guard.requests, "get", lambda *a, **k: _Redirect())
    with pytest.raises(DisallowedURLError):
        safe_get("https://www.ufcstats.com/statistics/fighters")


def test_safe_get_returns_non_redirect_response(monkeypatch):
    _patch_resolve(monkeypatch, "93.184.216.34")

    class _Ok:
        is_redirect = False
        status_code = 200
        headers = {}

    monkeypatch.setattr(net_guard.requests, "get", lambda *a, **k: _Ok())
    resp = safe_get("https://www.ufc.com/athlete/jon-jones")
    assert resp.status_code == 200
