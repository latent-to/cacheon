"""GET /api/evaluations -- all eval records.
GET /api/evaluations/{uid} -- eval history for a specific UID."""

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from api.config import STATE_DIR
from api.helpers.state_reader import safe_json_load, sanitize_floats

router = APIRouter()


def _load_evals() -> list[dict]:
    state = safe_json_load(STATE_DIR / "state.json", {})
    evals = list((state.get("evaluations") or {}).values())
    evals.sort(key=lambda e: e.get("evaluated_at") or 0, reverse=True)
    return evals


@router.get(
    "/api/evaluations",
    tags=["Evaluations"],
    summary="All evaluation records",
    description=(
        "Every completed evaluation, newest first. "
        "Filter with ?status=dq for disqualified only, ?status=active for passing only."
    ),
)
def list_evaluations(
    status: str | None = Query(
        None,
        description="Filter: 'dq' for disqualified, 'active' for non-disqualified",
    ),
):
    evals = _load_evals()
    if status == "dq":
        evals = [e for e in evals if e.get("disqualified")]
    elif status == "active":
        evals = [e for e in evals if not e.get("disqualified")]
    return JSONResponse(
        content=sanitize_floats({"evaluations": evals, "total": len(evals)}),
        headers={"Cache-Control": "public, max-age=30"},
    )


@router.get(
    "/api/evaluations/{uid}",
    tags=["Evaluations"],
    summary="Evaluations for a specific UID",
    description="All evaluation records for a given UID, newest first.",
)
def get_evaluations_by_uid(uid: int):
    evals = [e for e in _load_evals() if e.get("uid") == uid]
    if not evals:
        return JSONResponse(
            status_code=404,
            content={"detail": f"No evaluations found for UID {uid}"},
        )
    return JSONResponse(
        content=sanitize_floats(
            {"uid": uid, "evaluations": evals, "total": len(evals)}
        ),
        headers={"Cache-Control": "public, max-age=30"},
    )
