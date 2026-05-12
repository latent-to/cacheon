"""GET /api/container-logs -- list available container logs.
GET /api/container-log/{label} -- raw container log text."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse

from api.config import STATE_DIR

router = APIRouter()

LOG_DIR = STATE_DIR / "container_logs"


@router.get(
    "/api/container-logs",
    tags=["Logs"],
    summary="List container logs",
    description="Available container log labels. Use a label with /api/container-log/{label} to fetch the raw text.",
)
def list_container_logs():
    if not LOG_DIR.is_dir():
        return JSONResponse(content={"logs": [], "total": 0})

    logs = sorted(
        [
            {
                "label": f.stem,
                "filename": f.name,
                "size_bytes": f.stat().st_size,
            }
            for f in LOG_DIR.iterdir()
            if f.is_file() and f.suffix == ".log"
        ],
        key=lambda x: x["filename"],
        reverse=True,
    )
    return JSONResponse(
        content={"logs": logs, "total": len(logs)},
        headers={"Cache-Control": "public, max-age=60"},
    )


@router.get(
    "/api/container-log/{label}",
    tags=["Logs"],
    summary="Raw container log",
    description=(
        "Full Docker stdout/stderr from a miner or baseline container. "
        "Returns plain text. Label format: uid{N}_{hotkey8}_{block} or baseline_{hash}."
    ),
)
def get_container_log(label: str):
    safe_label = Path(label).name
    log_path = LOG_DIR / f"{safe_label}.log"

    if not log_path.is_file():
        return JSONResponse(
            status_code=404,
            content={"detail": f"Container log '{safe_label}' not found"},
        )

    text = log_path.read_text(encoding="utf-8", errors="replace")
    return PlainTextResponse(
        content=text,
        headers={"Cache-Control": "public, max-age=300"},
    )
