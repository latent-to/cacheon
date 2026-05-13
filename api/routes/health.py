"""GET /api/health -- lightweight liveness check."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get(
    "/api/health",
    tags=["Overview"],
    summary="Liveness check",
    description="Returns 200 if the API is up. Use for uptime monitors.",
)
def health():
    return JSONResponse(
        content={"status": "ok"},
        headers={"Cache-Control": "no-cache"},
    )
