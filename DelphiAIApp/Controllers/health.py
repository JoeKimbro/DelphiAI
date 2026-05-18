from fastapi import APIRouter

from DelphiAIApp.Models.db.postgres import execute_query

router = APIRouter()


@router.get("/health")
def health():
    try:
        execute_query("SELECT 1", fetch_one=True)
        db_ok = True
    except Exception:
        db_ok = False
    return {"status": "ok" if db_ok else "degraded", "database": db_ok}
