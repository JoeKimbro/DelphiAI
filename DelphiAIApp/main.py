"""
DelphiAI FastAPI entry point.

Run from project root:
    uvicorn DelphiAIApp.main:app --reload --port 8000

The Models/ml/ pipeline imports expect to be invoked from DelphiAIApp/Models/,
so this app inserts that directory onto sys.path before importing Controllers.
"""
import os
import sys
from pathlib import Path

_MODELS_DIR = Path(__file__).resolve().parent / "Models"
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from DelphiAIApp.Controllers import health, events, fighters, performance, results
from DelphiAIApp.Models.db.postgres import get_connection_pool, close_all_connections
from DelphiAIApp.security import (
    DOCS_ENABLED,
    SecurityMiddleware,
    warn_if_unconfigured,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Pre-warm the DB pool so the first request doesn't pay connection cost.
    get_connection_pool()
    warn_if_unconfigured()
    yield
    close_all_connections()


# CORS_ORIGINS: comma-separated list. Defaults to the local dev frontend.
# In production, set e.g. DELPHI_CORS_ORIGINS="https://delphi.yourdomain.com"
_cors_origins = [
    o.strip()
    for o in os.environ.get("DELPHI_CORS_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]


app = FastAPI(
    title="DelphiAI",
    description="UFC fight prediction API (XGBoost + ELO blended model).",
    version="1.0.0",
    lifespan=lifespan,
    # Hide the interactive docs unless explicitly enabled — they leak the
    # full route map to drive-by scanners.
    docs_url="/docs" if DOCS_ENABLED else None,
    redoc_url="/redoc" if DOCS_ENABLED else None,
    openapi_url="/openapi.json" if DOCS_ENABLED else None,
)

# Order matters: SecurityMiddleware must run BEFORE CORS for OPTIONS handling
# to work cleanly. Starlette runs middleware in reverse-registration order, so
# add Security last (executes first).
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SecurityMiddleware)

app.include_router(health.router)
app.include_router(events.router, prefix="/api/events", tags=["events"])
app.include_router(fighters.router, prefix="/api/fighters", tags=["fighters"])
app.include_router(performance.router, prefix="/api/performance", tags=["performance"])
app.include_router(results.router, prefix="/api/results", tags=["results"])
