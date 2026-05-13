"""GET /api/status -- detailed validator overview."""

import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.config import STATE_DIR
from api.helpers.state_reader import safe_json_load, sanitize_floats

router = APIRouter()


@router.get(
    "/api/status",
    tags=["Overview"],
    summary="Validator status overview",
    description=(
        "Current king, evaluation counts, time since last eval, "
        "and chain scan / weight-set block numbers."
    ),
)
def status():
    state = safe_json_load(STATE_DIR / "state.json", {})
    evals = state.get("evaluations") or {}
    king = state.get("king")

    n_dq = sum(1 for e in evals.values() if e.get("disqualified"))
    n_active = sum(1 for e in evals.values() if not e.get("disqualified"))

    last_eval_ts = max((e.get("evaluated_at") or 0 for e in evals.values()), default=0)
    last_eval_age_min = (
        round((time.time() - last_eval_ts) / 60, 1) if last_eval_ts else None
    )

    return JSONResponse(
        content=sanitize_floats(
            {
                "king_uid": king.get("uid") if king else None,
                "king_score": king.get("score") if king else None,
                "king_image": king.get("image") if king else None,
                "n_evaluated": len(evals),
                "n_active": n_active,
                "n_disqualified": n_dq,
                "last_eval_ts": last_eval_ts or None,
                "last_eval_age_min": last_eval_age_min,
                "last_scan_block": state.get("last_scan_block"),
                "last_weights_set_block": state.get("last_weights_set_block"),
            }
        ),
        headers={"Cache-Control": "public, max-age=30"},
    )
