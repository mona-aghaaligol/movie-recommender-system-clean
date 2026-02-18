from __future__ import annotations

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    summary="Health Check",
    description=(
        "Liveness probe.\n\n"
        "Indicates whether the application process is alive.\n"
        "If this endpoint fails, the container should be restarted."
    ),
)
def health_check():
    return {"status": "ok"}


@router.get(
    "/ready",
    summary="Readiness Check",
    description=(
        "Readiness probe.\n\n"
        "Indicates whether the application is ready to receive traffic.\n"
        "Returns 503 until startup bootstrap completes successfully."
    ),
    responses={
        200: {"description": "Service is ready", "content": {"application/json": {"example": {"status": "ready"}}}},
        503: {"description": "Service is warming up", "content": {"application/json": {"example": {"status": "not_ready"}}}},
    },
)
def readiness_check(request: Request):
    is_ready = bool(getattr(request.app.state, "is_ready", False))
    if not is_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready"},
        )
    return {"status": "ready"}
