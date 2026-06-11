"""Upload the freshly retrained model to Cloudflare R2 (S3-compatible).

Run locally AFTER retraining. Needs R2 credentials (which NEVER enter the
container): R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET.
boto3 is imported lazily so the API image stays free of the S3 SDK.

Usage:
    pip install -r requirements-dev.txt   # one-time, gets boto3
    python scripts/publish_model.py [path/to/model.pkl]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_MODEL = (
    Path(__file__).resolve().parents[1]
    / "DelphiAIApp" / "Models" / "ml" / "artifacts" / "model_latest.pkl"
)
OBJECT_KEY = os.getenv("R2_MODEL_KEY", "model_latest.pkl")
_REQUIRED = ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")


def build_client_kwargs() -> dict:
    """boto3 S3-client kwargs for R2 from env; raises if any var is missing."""
    missing = [k for k in _REQUIRED if not os.getenv(k)]
    if missing:
        raise EnvironmentError(f"Missing R2 env vars: {', '.join(missing)}")
    return {
        "service_name": "s3",
        "endpoint_url": os.environ["R2_ENDPOINT_URL"],
        "aws_access_key_id": os.environ["R2_ACCESS_KEY_ID"],
        "aws_secret_access_key": os.environ["R2_SECRET_ACCESS_KEY"],
        "region_name": "auto",
    }


def main() -> int:
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL
    if not model_path.exists():
        print(f"[publish] no model at {model_path}", file=sys.stderr)
        return 1
    import boto3  # lazy: only the publisher needs the SDK

    client = boto3.client(**build_client_kwargs())
    bucket = os.environ["R2_BUCKET"]
    print(f"[publish] uploading {model_path} -> r2://{bucket}/{OBJECT_KEY} ...", flush=True)
    client.upload_file(str(model_path), bucket, OBJECT_KEY)
    print("[publish] OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
