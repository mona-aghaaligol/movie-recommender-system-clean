from __future__ import annotations

import logging
import sys

from src.recommender.logging_utils import JsonFormatter


class DropHealthcheckAccessLogs(logging.Filter):
    """
    Filter that removes uvicorn access logs for health/readiness endpoints.

    This reduces noise in production log streams while keeping
    access logs for real business endpoints.
    """
    _paths = ("/v1/health", "/v1/ready")

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(path in message for path in self._paths)


def configure_app_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger to ensure consistent JSON logs across the application.

    This enforces a single StreamHandler with JsonFormatter so that all modules
    (including data loaders) produce uniform structured logs.
    """
    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root.handlers = [handler]

    # FAANG-grade: keep access logs, but drop noise from health endpoints
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.addFilter(DropHealthcheckAccessLogs())
