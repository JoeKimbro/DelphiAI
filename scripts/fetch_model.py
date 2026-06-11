"""Boot-time model fetch from object storage (Cloudflare R2 / any HTTPS URL).

If MODEL_URL is set and the local model file is missing or stale, download it to
the path MLPredictor loads (DELPHI_MODEL_PATH, else the bundled artifacts
location). Uses a plain HTTPS GET (requests) so the API container needs no S3
SDK; publishing (which needs credentials) lives in publish_model.py.

Usage:
    python scripts/fetch_model.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests

DEFAULT_TARGET = (
    Path(__file__).resolve().parents[1]
    / "DelphiAIApp" / "Models" / "ml" / "artifacts" / "model_latest.pkl"
)
MAX_AGE_HOURS = float(os.getenv("MODEL_MAX_AGE_HOURS", "168"))  # 7 days


def resolve_target() -> Path:
    """Where the model should land — DELPHI_MODEL_PATH or the bundled location."""
    env_path = os.getenv("DELPHI_MODEL_PATH", "").strip()
    return Path(env_path) if env_path else DEFAULT_TARGET


def should_fetch(target: Path, max_age_hours: float) -> bool:
    """True if the target is missing or older than max_age_hours."""
    if not target.exists():
        return True
    age_hours = (time.time() - target.stat().st_mtime) / 3600
    return age_hours > max_age_hours


def download(url: str, target: Path) -> None:
    """Stream the model to a temp file, then atomically rename into place."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.replace(target)


def main() -> int:
    url = os.getenv("MODEL_URL", "").strip()
    if not url:
        print("[model] MODEL_URL unset — using bundled/local model.", flush=True)
        return 0
    target = resolve_target()
    if not should_fetch(target, MAX_AGE_HOURS):
        print(f"[model] {target} is fresh — skipping download.", flush=True)
        return 0
    print(f"[model] downloading {url} -> {target} ...", flush=True)
    try:
        download(url, target)
    except Exception as e:  # noqa: BLE001 — boot must not crash on a transient pull
        print(f"[model] download failed: {e}", file=sys.stderr)
        return 1
    print(f"[model] OK ({target.stat().st_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
