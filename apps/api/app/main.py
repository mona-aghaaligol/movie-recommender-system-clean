from __future__ import annotations

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler:
    - Warm up heavy dependencies at startup
    - (Optional) cleanup at shutdown
    """
    app.state.recommender_service = bootstrap_service()
    yield
    # Cleanup hooks can be added here if needed


app = FastAPI(
    title="Movie Recommender API",
    version="0.1.0",
    description="Production-oriented movie recommendation service",
    lifespan=lifespan,
)


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

        logger.info(
            "Request completed",
            extra={
                "event": "request.completed",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": duration_ms,
            },
        )

        if response is not None:
            response.headers[REQUEST_ID_HEADER] = request_id


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


