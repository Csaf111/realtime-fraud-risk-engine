"""Boundary tests for client-provided correlation IDs."""

import re

from fraud_engine.request_ids import safe_request_id


def test_valid_correlation_ids_preserved():
    for candidate in ("abc-123", "trace_1.2:3", "A" * 128):
        assert safe_request_id(candidate) == candidate


def test_invalid_correlation_ids_replaced_with_uuid():
    for candidate in (None, "", "a" * 129, " leading", "a/b", "a\nb", "é", "a,b"):
        result = safe_request_id(candidate)
        assert result != candidate
        assert re.fullmatch(r"[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}", result)


def test_api_does_not_reflect_untrusted_request_id(client):
    for invalid in ("a" * 129, "a,b", "é"):
        response = client.get("/health", headers={"x-request-id": invalid})
        assert response.status_code == 200
        assert response.headers["x-request-id"] != invalid
        assert len(response.headers["x-request-id"]) == 36


def test_api_preserves_valid_request_id(client):
    response = client.get("/health", headers={"x-request-id": "trace-2026:11"})
    assert response.headers["x-request-id"] == "trace-2026:11"
