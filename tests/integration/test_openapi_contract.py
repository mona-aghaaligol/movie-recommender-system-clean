from __future__ import annotations

from typing import Any, Dict

from apps.api.app.error_codes import ErrorCode
from apps.api.app.main import app


def _get(responses: Dict[str, Any], status_code: int) -> Dict[str, Any]:
    key = str(status_code)
    assert key in responses, f"OpenAPI missing response for {status_code}"
    return responses[key]


def test_openapi_contains_error_responses_for_recommendations() -> None:
    spec = app.openapi()

    path_item = spec["paths"]["/v1/recommendations/{user_id}"]
    op = path_item["get"]

    responses = op["responses"]

    for status_code in (400, 404, 422, 500):
        _get(responses, status_code)


def test_openapi_error_examples_shape_and_codes() -> None:
    spec = app.openapi()

    op = spec["paths"]["/v1/recommendations/{user_id}"]["get"]
    responses = op["responses"]

    allowed = {e.value for e in ErrorCode}

    for status_code in (400, 404, 422, 500):
        r = _get(responses, status_code)

        content = r.get("content", {})
        assert "application/json" in content, f"{status_code} missing application/json"

        example = content["application/json"].get("example")
        assert example is not None, f"{status_code} missing example"

        # Validate error envelope shape
        assert "error" in example, f"{status_code} example missing 'error'"
        err = example["error"]

        assert "code" in err, f"{status_code} example missing error.code"
        assert "message" in err, f"{status_code} example missing error.message"
        assert "request_id" in err, f"{status_code} example missing error.request_id"

        # Validate code is part of official ErrorCode enum
        assert err["code"] in allowed, (
            f"{status_code} invalid error.code: {err['code']}"
        )

