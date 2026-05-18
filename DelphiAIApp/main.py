"""
DelphiAI FastAPI entry point.

Run from project root:
    uvicorn DelphiAIApp.main:app --reload --port 8000

The Models/ml/ pipeline imports expect to be invoked from DelphiAIApp/Models/,
so this app inserts that directory onto sys.path before importing Controllers.
"""
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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Pre-warm the DB pool so the first request doesn't pay connection cost.
    get_connection_pool()
    yield
    close_all_connections()


app = FastAPI(
    title="DelphiAI",
    description="UFC fight prediction API (XGBoost + ELO blended model).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(events.router, prefix="/api/events", tags=["events"])
app.include_router(fighters.router, prefix="/api/fighters", tags=["fighters"])
app.include_router(performance.router, prefix="/api/performance", tags=["performance"])
app.include_router(results.router, prefix="/api/results", tags=["results"])
