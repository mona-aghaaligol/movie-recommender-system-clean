from __future__ import annotations

import logging
import sys

from src.recommender.logging_utils import JsonFormatter


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

