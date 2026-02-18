from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.recommender.bootstrap import bootstrap_service
from src.recommender.logging_utils import configure_logger

from .error_codes import ErrorCode
from .errors import get_request_id, make_error
from .logging_setup import configure_app_logging
from .routes.health import router as health_router
from .routes.recommendations import router as recommendations_router


configure_app_logging()
logger = configure_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


async def _bootstrap_in_background(app: FastAPI) -> None:
    """
    Run heavy bootstrap outside of startup so the server can accept requests immediately.
    Marks app.state.is_ready=True only after the recommender service is initialized.
    """
    try:
        logger.info(
            "Background bootstrap started",
            extra={"event": "bootstrap.bg_start"},
        )

        # bootstrap_service() is synchronous and may be heavy (Mongo + pandas)
        service = await asyncio.to_thread(bootstrap_service)

        app.state.recommender_service = service
        app.state.is_ready = True

        logger.info(
            "Background bootstrap finished; application marked as ready",
            extra={"event": "bootstrap.bg_ready"},
        )
    except Exception:
        app.state.is_ready = False
        logger.exception(
            "Background bootstrap failed; application will remain not ready",
            extra={"event": "bootstrap.bg_failed"},
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler (FAANG-grade):
    - Do NOT block startup on heavy dependencies.
    - Start server quickly, then bootstrap heavy dependencies in background.
    """
    app.state.is_ready = False
    app.state.recommender_service = None

    # Start background bootstrap task
    task = asyncio.create_task(_bootstrap_in_background(app))
    app.state.bootstrap_task = task

    yield

    # Optional: graceful shutdown for the background task
    task = getattr(app.state, "bootstrap_task", None)
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            logger.info(
                "Background bootstrap task cancelled on shutdown",
                extra={"event": "bootstrap.bg_cancelled"},
            )


app = FastAPI(
    title="Movie Recommender API",
    version="0.1.0",
    description="Production-oriented movie recommendation service",
    lifespan=lifespan,
)

# Ensure readiness state exists even in unusual contexts
if not hasattr(app.state, "is_ready"):
    app.state.is_ready = False


# Option 1 (FAANG-grade): do not emit "request completed" logs for health checks
HEALTHCHECK_PATHS = {"/v1/health", "/v1/ready"}

@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """
    Attach a request id to every request and emit structured request lifecycle logs.
    """
    request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
    request.state.request_id = request_id

    start = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000.0, 2)
        status_code = getattr(response, "status_code", None)
        path = request.url.path

        # Option 1: skip access-style logs for health checks to avoid log noise
        if path not in HEALTHCHECK_PATHS:
            logger.info(
                "Request completed",
                extra={
                    "event": "request.completed",
                    "request_id": request_id,
                    "method": request.method,
                    "path": path,
                    "status_code": status_code,
                    "duration_ms": duration_ms,
                },
            )

        if response is not None:
            response.headers[REQUEST_ID_HEADER] = request_id


@app.middleware("http")
async def readiness_gate_middleware(request: Request, call_next):
    """
    FAANG-grade global readiness gate:
    - Before app is ready, allow ONLY /v1/health and /v1/ready.
    - All other routes return 503 with a stable error payload.
    This prevents "Empty reply" / unpredictable responses during warm-up.
    """
    path = request.url.path

    # Allow liveness/readiness endpoints even when not ready
    if path in ("/v1/health", "/v1/ready"):
        return await call_next(request)

    is_ready = bool(getattr(request.app.state, "is_ready", False))
    if not is_ready:
        request_id = (
            request.headers.get(REQUEST_ID_HEADER)
            or getattr(request.state, "request_id", None)
            or uuid.uuid4().hex
        )

        payload = {
            "error": {
                "code": "SERVICE_UNAVAILABLE",
                "message": "Service is warming up. Please retry shortly.",
                "request_id": request_id,
            }
        }

        return JSONResponse(
            status_code=503,
            content=payload,
            headers={REQUEST_ID_HEADER: request_id},
        )

    return await call_next(request)


def _map_http_status_to_error_code(status_code: int) -> ErrorCode:
    """
    Map HTTP status codes to stable API error codes.
    """
    mapping: dict[int, ErrorCode] = {
        400: ErrorCode.BAD_REQUEST,
        404: ErrorCode.NOT_FOUND,
        409: ErrorCode.CONFLICT,
        429: ErrorCode.RATE_LIMITED,
    }
    return mapping.get(status_code, ErrorCode.HTTP_ERROR)


def _error_response(
    *,
    status_code: int,
    code: ErrorCode,
    message: str,
    request_id: str,
    details: Optional[Any] = None,
) -> JSONResponse:
    """
    Build a canonical JSON error response.
    """
    payload = make_error(
        code=code.value,
        message=message,
        request_id=request_id,
        details=details,
    ).model_dump()

    return JSONResponse(
        status_code=status_code,
        content=payload,
        headers={REQUEST_ID_HEADER: request_id},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    request_id = get_request_id(request)

    logger.warning(
        "Request validation failed",
        extra={
            "event": "request.validation_error",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "code": ErrorCode.VALIDATION_ERROR.value,
        },
    )

    return _error_response(
        status_code=422,
        code=ErrorCode.VALIDATION_ERROR,
        message="Request validation failed",
        request_id=request_id,
        details={"errors": exc.errors()},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = get_request_id(request)
    code = _map_http_status_to_error_code(exc.status_code)

    if isinstance(exc.detail, str):
        message = exc.detail
        details = None
    elif isinstance(exc.detail, dict):
        message = "Request failed"
        details = exc.detail
    else:
        message = "Request failed"
        details = None

    logger.info(
        "HTTP exception raised",
        extra={
            "event": "request.http_exception",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": exc.status_code,
            "code": code.value,
        },
    )

    return _error_response(
        status_code=exc.status_code,
        code=code,
        message=message,
        request_id=request_id,
        details=details,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = get_request_id(request)

    logger.exception(
        "Unhandled exception",
        extra={
            "event": "error.unhandled_exception",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "code": ErrorCode.INTERNAL_ERROR.value,
        },
    )

    return _error_response(
        status_code=500,
        code=ErrorCode.INTERNAL_ERROR,
        message="Internal Server Error",
        request_id=request_id,
        details=None,
    )


app.include_router(health_router, prefix="/v1")
app.include_router(recommendations_router, prefix="/v1")


