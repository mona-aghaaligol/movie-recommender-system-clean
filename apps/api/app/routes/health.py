from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health", tags=["health"])
def health_check():
    """
    Liveness probe.

    Indicates whether the application process is alive.
    If this endpoint fails, the container should be restarted.
    """
    return {"status": "ok"}


@router.get("/ready", tags=["health"])
def readiness_check():
    """
    Readiness probe.

    Indicates whether the application is ready to receive traffic.
    In future steps, this will verify:
      - Database connectivity
      - Recommender service initialization
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "ready"},
    )

