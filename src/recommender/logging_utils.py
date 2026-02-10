from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Iterable


class JsonFormatter(logging.Formatter):
    """
    Structured JSON formatter designed for production logging.

    Produces machine-readable logs that are easy to index in
    Datadog, Splunk, CloudWatch, and ELK.
    """

    _extra_keys: Iterable[str] = (
        # Observability / HTTP request context
        "event",
        "request_id",
        "method",
        "path",
        "status_code",
        "duration_ms",
        "code",
        # Optional domain/app context
        "user_id",
        "shape",
        "step",
        "exception_type",
        # Useful for certain infra logs
        "attempt",
        "collection",
        "rows",
        "columns",
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key in self._extra_keys:
            if hasattr(record, key):
                payload[key] = getattr(record, key)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def configure_logger(name: str = "app", level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger with JSON formatting.

    This function enforces a consistent handler/formatter setup to avoid
    mixed log formats across modules.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Enforce a single StreamHandler with JsonFormatter for consistency.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    logger.handlers = [handler]
    logger.propagate = False
    return logger
