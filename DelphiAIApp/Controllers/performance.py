from fastapi import APIRouter, HTTPException

from DelphiAIApp.Services.performance_service import get_performance_summary

router = APIRouter()


@router.get("/summary")
def performance_summary(prediction_type: str = "live"):
    try:
        return get_performance_summary(prediction_type=prediction_type)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to compute performance summary: {e}")
