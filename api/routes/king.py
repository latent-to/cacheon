"""GET /api/king -- current king record.
GET /api/king/history -- dethronement timeline."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.config import STATE_DIR
from api.helpers.state_reader import safe_json_load, safe_jsonl_load, sanitize_floats

router = APIRouter()


@router.get(
    "/api/king",
    tags=["King"],
    summary="Current king",
    description="Full record of the reigning champion: UID, score, image, per-prompt stats.",
)
def get_king():
    state = safe_json_load(STATE_DIR / "state.json", {})
    king = state.get("king")
    if king is None:
        return JSONResponse(
            content={"king": None, "message": "No king yet"},
            headers={"Cache-Control": "public, max-age=30"},
        )
    return JSONResponse(
        content=sanitize_floats({"king": king}),
        headers={"Cache-Control": "public, max-age=30"},
    )


@router.get(
    "/api/king/history",
    tags=["King"],
    summary="Dethronement history",
    description="Chronological list of king changes. Each entry shows the new king, the dethroned king, and the margin.",
)
def get_king_history():
    entries = safe_jsonl_load(STATE_DIR / "king-history.jsonl")
    return JSONResponse(
        content=sanitize_floats({"history": entries, "total": len(entries)}),
        headers={"Cache-Control": "public, max-age=30"},
    )
