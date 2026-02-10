from fastapi.testclient import TestClient

from apps.api.app.main import app


client = TestClient(app)


def test_error_request_id_matches_header():
    """
    Ensure that the request id returned in the error body
    matches the X-Request-ID response header.
    """
    response = client.get("/v1/recommendations/999999?limit=10")
    assert response.status_code == 400

    header_request_id = (
        response.headers.get("x-request-id")
        or response.headers.get("X-Request-ID")
    )
    assert header_request_id is not None

    body = response.json()
    assert body["error"]["request_id"] == header_request_id


def test_success_has_request_id_header():
    """
    Ensure that successful responses always include X-Request-ID.
    """
    response = client.get("/v1/recommendations/1?limit=10")
    assert response.status_code == 200

    header_request_id = (
        response.headers.get("x-request-id")
        or response.headers.get("X-Request-ID")
    )
    assert header_request_id is not None

