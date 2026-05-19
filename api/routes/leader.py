"""GET /api/leader -- current leader record.
GET /api/leader/history -- overtake timeline."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.config import STATE_DIR
from api.helpers.state_reader import safe_json_load, safe_jsonl_load, sanitize_floats

router = APIRouter()


@router.get(
    "/api/leader",
    tags=["Leader"],
    summary="Current leader",
    description="Full record of the reigning champion: UID, score, image, per-prompt stats.",
)
def get_leader():
    state = safe_json_load(STATE_DIR / "state.json", {})
    winner = state.get("winner") or state.get("king")
    if winner is None:
        return JSONResponse(
            content={"leader": None, "runner_up": None, "message": "No leader yet"},
            headers={"Cache-Control": "public, max-age=30"},
        )
    if "crowned_at_block" in winner and "won_at_block" not in winner:
        winner = {**winner, "won_at_block": winner["crowned_at_block"]}
    runner_up = state.get("runner_up")
    return JSONResponse(
        content=sanitize_floats({"leader": winner, "runner_up": runner_up}),
        headers={"Cache-Control": "public, max-age=30"},
    )


@router.get(
    "/api/leader/history",
    tags=["Leader"],
    summary="Overtake history",
    description="Chronological list of leader changes. Each entry shows the new leader, the previous leader, and the margin.",
)
def get_leader_history():
    entries = safe_jsonl_load(STATE_DIR / "winner-history.jsonl")
    if not entries:
        entries = safe_jsonl_load(STATE_DIR / "king-history.jsonl")
    normalized = [_normalize_history_entry(e) for e in entries]
    return JSONResponse(
        content=sanitize_floats({"history": normalized, "total": len(normalized)}),
        headers={"Cache-Control": "public, max-age=30"},
    )


def _normalize_history_entry(e: dict) -> dict:
    """Translate on-disk winner/king field names to leader field names."""
    out: dict = {
        "ts": e.get("ts"),
        "block": e.get("block"),
        "new_leader_uid": (
            e.get("new_winner_uid")
            if e.get("new_winner_uid") is not None
            else e.get("new_king_uid")
        ),
        "new_leader_hotkey": e.get("new_winner_hotkey") or e.get("new_king_hotkey"),
        "new_leader_score": (
            e.get("new_winner_score")
            if e.get("new_winner_score") is not None
            else e.get("new_king_score")
        ),
        "new_leader_image": e.get("new_winner_image") or e.get("new_king_image"),
        "new_leader_digest": e.get("new_winner_digest") or e.get("new_king_digest"),
        "overtake_threshold": (
            e.get("overtake_threshold")
            if e.get("overtake_threshold") is not None
            else e.get("dethrone_threshold")
        ),
    }
    prev_uid = (
        e.get("prev_winner_uid")
        if e.get("prev_winner_uid") is not None
        else e.get("prev_king_uid")
    )
    if prev_uid is not None:
        out["prev_leader_uid"] = prev_uid
        out["prev_leader_hotkey"] = e.get("prev_winner_hotkey") or e.get(
            "prev_king_hotkey"
        )
        prev_score = (
            e.get("prev_winner_score")
            if e.get("prev_winner_score") is not None
            else e.get("prev_king_score")
        )
        out["prev_leader_score"] = prev_score
    return out
