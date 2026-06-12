#!/usr/bin/env sh
set -e
# Pull the model from R2 if MODEL_URL is set; a failure here must not stop boot
# (e.g. local runs with no MODEL_URL — fetch_model exits 0 and is a no-op).
python scripts/fetch_model.py || echo "[entrypoint] model fetch skipped/failed; continuing"
exec uvicorn DelphiAIApp.main:app --host 0.0.0.0 --port "${PORT:-8000}"
