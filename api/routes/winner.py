"""GET /api/winner -- current winner record.
GET /api/winner/history -- overtake timeline."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.config import STATE_DIR
from api.helpers.state_reader import safe_json_load, safe_jsonl_load, sanitize_floats

router = APIRouter()


@router.get(
    "/api/winner",
    tags=["Winner"],
    summary="Current winner",
    description="Full record of the reigning champion: UID, score, image, per-prompt stats.",
)
def get_winner():
    state = safe_json_load(STATE_DIR / "state.json", {})
    winner = state.get("winner") or state.get("king")
    if winner is None:
        return JSONResponse(
            content={"winner": None, "message": "No winner yet"},
            headers={"Cache-Control": "public, max-age=30"},
        )
    return JSONResponse(
        content=sanitize_floats({"winner": winner}),
        headers={"Cache-Control": "public, max-age=30"},
    )


@router.get(
    "/api/winner/history",
    tags=["Winner"],
    summary="Overtake history",
    description="Chronological list of winner changes. Each entry shows the new winner, the previous winner, and the margin.",
)
def get_winner_history():
    entries = safe_jsonl_load(STATE_DIR / "winner-history.jsonl")
    if not entries:
        entries = safe_jsonl_load(STATE_DIR / "king-history.jsonl")
    return JSONResponse(
        content=sanitize_floats({"history": entries, "total": len(entries)}),
        headers={"Cache-Control": "public, max-age=30"},
    )
