from fastapi import APIRouter, HTTPException

from DelphiAIApp.Services.fighter_service import get_fighter_profile

router = APIRouter()


@router.get("/{slug}")
def fighter_profile(slug: str):
    profile = get_fighter_profile(slug)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Fighter '{slug}' not found.")
    return profile
