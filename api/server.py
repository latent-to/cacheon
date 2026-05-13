"""Cacheon monitoring API. Read-only surface over on-disk validator state."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.config import ALLOWED_ORIGINS
from api.routes.health import router as health_router
from api.routes.status import router as status_router
from api.routes.king import router as king_router
from api.routes.evaluations import router as evaluations_router
from api.routes.logs import router as logs_router
from api.routes.rounds import router as rounds_router

TRUSTED_PROXIES = {"127.0.0.1", "::1"}


def _client_ip(request: Request) -> str:
    """Use X-Forwarded-For when the direct peer is the local reverse proxy."""
    peer = request.client.host if request.client else "127.0.0.1"
    if peer in TRUSTED_PROXIES:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[-1].strip()
    return peer


limiter = Limiter(key_func=_client_ip, default_limits=["60/minute"])

app = FastAPI(
    title="Cacheon Monitoring API",
    description=(
        "Read-only status surface for the Cacheon subnet (SN14). "
        "Serves evaluation results, king status, container logs, and round history "
        "from the validator's on-disk state files."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Overview", "description": "Health check and validator status"},
        {"name": "King", "description": "Current king and dethronement history"},
        {
            "name": "Evaluations",
            "description": "Completed evaluation records and per-UID history",
        },
        {"name": "Logs", "description": "Raw Docker container logs from eval runs"},
        {"name": "Rounds", "description": "Eval rounds and pending eval jobs"},
    ],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(status_router)
app.include_router(king_router)
app.include_router(evaluations_router)
app.include_router(logs_router)
app.include_router(rounds_router)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")
