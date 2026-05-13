"""GET /api/rounds -- eval rounds grouped by evaluation_block.
GET /api/eval-job -- current pending eval job."""

from collections import defaultdict

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.config import STATE_DIR
from api.helpers.state_reader import safe_json_load, sanitize_floats

router = APIRouter()


@router.get(
    "/api/rounds",
    tags=["Rounds"],
    summary="Evaluation rounds",
    description=(
        "Evaluations grouped by round (evaluation_block). Each round shows "
        "the block, timestamp, and the list of challengers with their outcomes."
    ),
)
def list_rounds():
    state = safe_json_load(STATE_DIR / "state.json", {})
    evals = (state.get("evaluations") or {}).values()

    by_block: dict[int, list[dict]] = defaultdict(list)
    ts_by_block: dict[int, float] = {}
    for e in evals:
        block = e.get("evaluation_block", 0)
        by_block[block].append(
            {
                "uid": e.get("uid"),
                "hotkey": e.get("hotkey"),
                "image": e.get("image"),
                "score": e.get("score"),
                "disqualified": e.get("disqualified"),
                "disqualify_reason": e.get("disqualify_reason"),
            }
        )
        ts = e.get("evaluated_at") or 0
        if ts > ts_by_block.get(block, 0):
            ts_by_block[block] = ts

    rounds = []
    for block in sorted(by_block, reverse=True):
        challengers = by_block[block]
        latest_ts = ts_by_block.get(block)
        rounds.append(
            {
                "evaluation_block": block,
                "evaluated_at": latest_ts or None,
                "n_challengers": len(challengers),
                "challengers": challengers,
            }
        )

    return JSONResponse(
        content=sanitize_floats({"rounds": rounds, "total": len(rounds)}),
        headers={"Cache-Control": "public, max-age=30"},
    )


@router.get(
    "/api/eval-job",
    tags=["Rounds"],
    summary="Current pending eval job",
    description="The eval_job.json file that the CPU validator wrote for the next GPU eval round. Shows queued challengers.",
)
def get_eval_job():
    job = safe_json_load(STATE_DIR / "eval_job.json")
    if job is None:
        return JSONResponse(
            content={"eval_job": None, "message": "No pending eval job"},
            headers={"Cache-Control": "public, max-age=30"},
        )
    return JSONResponse(
        content=sanitize_floats({"eval_job": job}),
        headers={"Cache-Control": "public, max-age=30"},
    )
