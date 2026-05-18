from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from DelphiAIApp.Services.event_service import resolve_event_results

router = APIRouter()


class UpdateRequest(BaseModel):
    event_name: str
    force: bool = False


@router.post("/update")
def update_results(payload: UpdateRequest):
    try:
        summary = resolve_event_results(payload.event_name, force=payload.force)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to update results: {e}")
    if not summary:
        raise HTTPException(status_code=404, detail=f"No tracked predictions for '{payload.event_name}'.")
    return summary
