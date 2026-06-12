import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from DelphiAIApp.Services.auto_resolve import auto_resolve_pending
from DelphiAIApp.Services.event_service import resolve_event_results

router = APIRouter()
logger = logging.getLogger(__name__)


class UpdateRequest(BaseModel):
    event_name: str
    force: bool = False


@router.post("/update")
def update_results(payload: UpdateRequest):
    try:
        summary = resolve_event_results(payload.event_name, force=payload.force)
    except Exception as e:
        logger.exception("Failed to update results for '%s': %s", payload.event_name, e)
        raise HTTPException(status_code=502, detail="Failed to update results.")
    if not summary:
        raise HTTPException(status_code=404, detail=f"No tracked predictions for '{payload.event_name}'.")
    return summary


@router.post("/auto-resolve")
def auto_resolve(force_ttl: bool = False):
    """
    Manually trigger the auto-resolver that scans for past-but-unresolved
    events and resolves them. Runs synchronously and returns the per-event
    summary. The same job runs automatically as a background task after
    every `/api/performance/summary` request (TTL-guarded).
    """
    try:
        return auto_resolve_pending(force_ttl=force_ttl)
    except Exception as e:
        logger.exception("Auto-resolve failed: %s", e)
        raise HTTPException(status_code=502, detail="Auto-resolve failed.")
