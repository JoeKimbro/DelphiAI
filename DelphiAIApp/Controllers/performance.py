import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException

from DelphiAIApp.Services.auto_resolve import auto_resolve_pending
from DelphiAIApp.Services.performance_service import get_performance_summary

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/summary")
def performance_summary(
    background_tasks: BackgroundTasks,
    prediction_type: str = "live",
    year: int | None = None,
):
    # Kick off a background resolution pass for any past-but-unresolved
    # events so the next dashboard refresh picks up freshly-completed cards
    # without the user running update_results by hand. TTL-guarded inside.
    background_tasks.add_task(auto_resolve_pending)
    try:
        return get_performance_summary(prediction_type=prediction_type, year=year)
    except Exception as e:
        logger.exception("Failed to compute performance summary: %s", e)
        raise HTTPException(status_code=502, detail="Failed to compute performance summary.")
