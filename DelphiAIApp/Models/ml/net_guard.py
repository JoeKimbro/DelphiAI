"""SSRF guard for server-side outbound fetches to external data sources.

The prediction request path fetches URLs that are not always fixed: a fighter
detail-page URL is pulled as an <a href> off a scraped listing page, and event
URLs come from scraped index pages. A hostile or compromised upstream could
therefore hand us a URL pointing at an internal address (cloud metadata,
localhost, RFC-1918), and `requests` follows redirects by default — so a benign
allowlisted URL could 302 somewhere internal.

Route those fetches through `safe_get()` (or at least `assert_allowed_url()`),
which restricts:

  - the scheme to http/https,
  - the host to a fixed allowlist of known data-source domains,
  - the resolved IP(s) to public address space (blocks DNS-rebinding to
    private/loopback/link-local/reserved ranges, incl. 169.254.169.254), and
  - redirects: each hop is re-validated against the same rules.
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import requests

# Domains the predictor is allowed to fetch from. Keep this tight — adding a
# host here widens the SSRF surface.
ALLOWED_HOSTS = frozenset(
    {
        "ufc.com",
        "www.ufc.com",
        "ufcstats.com",
        "www.ufcstats.com",
        "bestfightodds.com",
        "www.bestfightodds.com",
    }
)

DEFAULT_TIMEOUT = 15
MAX_REDIRECTS = 3


class DisallowedURLError(ValueError):
    """Raised when a URL fails the SSRF allowlist / public-IP checks."""


def _host_is_public(host: str) -> bool:
    """True only if every address `host` resolves to is public/global.

    Resolving here (rather than letting requests do it) is what defeats a
    hostname that points at an internal IP — we reject before connecting.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local  # 169.254.0.0/16 — cloud metadata
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    return True


def assert_allowed_url(url: str, allowed_hosts: frozenset[str] = ALLOWED_HOSTS) -> None:
    """Raise DisallowedURLError unless `url` is safe to fetch server-side."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise DisallowedURLError(f"scheme not allowed: {parsed.scheme!r} ({url!r})")
    host = (parsed.hostname or "").lower()
    if host not in allowed_hosts:
        raise DisallowedURLError(f"host not on allowlist: {host!r} ({url!r})")
    if not _host_is_public(host):
        raise DisallowedURLError(f"host resolves to a non-public address: {host!r}")


def safe_get(
    url: str,
    *,
    headers: dict | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    allowed_hosts: frozenset[str] = ALLOWED_HOSTS,
    max_redirects: int = MAX_REDIRECTS,
) -> requests.Response:
    """`requests.get` with SSRF guards and per-hop redirect re-validation.

    Redirects are handled manually so each Location is re-checked against the
    allowlist + public-IP rules (this keeps legitimate http→https / trailing-
    slash redirects working while blocking a redirect to an internal host).
    """
    current = url
    for _ in range(max_redirects + 1):
        assert_allowed_url(current, allowed_hosts)
        resp = requests.get(
            current, headers=headers, timeout=timeout, allow_redirects=False
        )
        location = resp.headers.get("location")
        if resp.is_redirect and location:
            current = urljoin(current, location)
            continue
        return resp
    raise DisallowedURLError(f"too many redirects (> {max_redirects}) starting at {url!r}")
