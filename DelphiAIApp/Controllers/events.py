from fastapi import APIRouter, HTTPException

from DelphiAIApp.Services.event_service import (
    list_upcoming_events,
    get_event_predictions,
    get_event_predictions_cached,
    list_past_events,
    get_past_event_details,
)

router = APIRouter()


@router.get("/upcoming")
def upcoming_events():
    try:
        return {"events": list_upcoming_events()}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch events: {e}")


@router.get("/past")
def past_events(limit: int = 4):
    # Cap to keep accidental clients from pulling the entire archive.
    limit = max(1, min(limit, 50))
    try:
        return {"events": list_past_events(limit=limit)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch past events: {e}")


@router.get("/past/{event_id}")
def past_event_detail(event_id: str):
    try:
        data = get_past_event_details(event_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch past event: {e}")
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
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Prediction pipeline failed: {e}")
    if not data:
        raise HTTPException(status_code=404, detail=f"No predictions for event '{event_id}'.")
    return data
