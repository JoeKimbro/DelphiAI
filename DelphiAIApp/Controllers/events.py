import logging

from fastapi import APIRouter, HTTPException

from DelphiAIApp.Services.event_service import (
    list_upcoming_events,
    get_event_predictions,
    get_event_predictions_cached,
    list_past_events,
    get_past_event_details,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/upcoming")
def upcoming_events():
    try:
        return {"events": list_upcoming_events()}
    except Exception as e:
        logger.exception("Failed to fetch events: %s", e)
        raise HTTPException(status_code=502, detail="Failed to fetch events.")


@router.get("/past")
def past_events(limit: int = 4):
    # Cap to keep accidental clients from pulling the entire archive.
    limit = max(1, min(limit, 50))
    try:
        return {"events": list_past_events(limit=limit)}
    except Exception as e:
        logger.exception("Failed to fetch past events: %s", e)
        raise HTTPException(status_code=502, detail="Failed to fetch past events.")


@router.get("/past/{event_id}")
def past_event_detail(event_id: str):
    try:
        data = get_past_event_details(event_id)
    except Exception as e:
        logger.exception("Failed to fetch past event '%s': %s", event_id, e)
        raise HTTPException(status_code=502, detail="Failed to fetch past event.")
    if not data:
        raise HTTPException(status_code=404, detail=f"No resolved predictions for event '{event_id}'.")
    return data


@router.get("/{event_id}/predictions")
def event_predictions(event_id: str, refresh: bool = False, cached_only: bool = False):
    try:
        if cached_only:
            data = get_event_predictions_cached(event_id)
        else:
            data = get_event_predictions(event_id, force_recompute=refresh)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No predictions found for event '{event_id}'.")
    except Exception as e:
        logger.exception("Prediction pipeline failed for '%s': %s", event_id, e)
        raise HTTPException(status_code=502, detail="Prediction pipeline failed.")
    if not data:
        raise HTTPException(status_code=404, detail=f"No predictions for event '{event_id}'.")
    return data
