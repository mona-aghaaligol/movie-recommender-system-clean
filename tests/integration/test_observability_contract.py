from __future__ import annotations

import logging

from fastapi.testclient import TestClient

from apps.api.app.main import REQUEST_ID_HEADER, app


def _attach_caplog_to_logger(caplog, logger_name: str) -> logging.Logger:
    """
    Our production loggers use propagate=False.
    caplog captures LogRecords, but only if its handler is attached to the logger.
    """
    logger = logging.getLogger(logger_name)
    logger.addHandler(caplog.handler)
    return logger


def _detach_caplog_from_logger(caplog, logger: logging.Logger) -> None:
    logger.removeHandler(caplog.handler)


def _records_with_event(caplog, event_name: str):
    return [r for r in caplog.records if getattr(r, "event", None) == event_name]


def test_request_id_header_is_present_and_matches_log_request_id(caplog):
    caplog.set_level(logging.INFO)

    # Attach caplog handler to our non-propagating app logger
    app_logger = _attach_caplog_to_logger(caplog, "apps.api.app.main")
    try:
        with TestClient(app) as client:
            resp = client.get("/v1/health")

        assert resp.status_code == 200
        assert REQUEST_ID_HEADER in resp.headers

        rid = resp.headers[REQUEST_ID_HEADER]
        assert isinstance(rid, str) and rid.strip()

        completed_records = _records_with_event(caplog, "request.completed")
        matching = [r for r in completed_records if getattr(r, "request_id", None) == rid]

        assert matching, "Expected a request.completed log with matching request_id"
    finally:
        _detach_caplog_from_logger(caplog, app_logger)
