from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fraud_engine.api import create_app
from fraud_engine.config import Settings
from fraud_engine.limits import (
    DEFAULT_MAX_BODY_BYTES,
    BodySizeLimitMiddleware,
    BodyTooLarge,
)

from .conftest import make_txn

LIMIT = 4096


@pytest.fixture
def small_client():
    with TestClient(create_app(Settings(max_body_bytes=LIMIT))) as c:
        yield c


def _batch_body(n: int) -> bytes:
    txns = [make_txn(transaction_id=f"t{i}") for i in range(n)]
    return json.dumps({"transactions": txns}).encode()


def test_body_within_limit_is_scored(small_client):
    r = small_client.post("/v1/score", json=make_txn(transaction_id="ok"))
    assert r.status_code == 200
    assert r.json()["transaction_id"] == "ok"


def test_declared_oversize_body_rejected_with_413(small_client):
    body = _batch_body(40)
    assert len(body) > LIMIT
    r = small_client.post(
        "/v1/score/batch",
        content=body,
        headers={"content-type": "application/json", "x-request-id": "big-1"},
    )
    assert r.status_code == 413
    assert r.json() == {"detail": f"Request body too large: max {LIMIT} bytes"}
    # The size limit sits inside the request-id middleware.
    assert r.headers["x-request-id"] == "big-1"


def test_rejected_body_is_not_scored(small_client):
    small_client.post(
        "/v1/score/batch", content=_batch_body(40), headers={"content-type": "application/json"}
    )
    assert "fre_scored_total 0" in small_client.get("/metrics").text


def test_streamed_oversize_body_rejected_with_413(small_client):
    body = _batch_body(40)

    def chunks():
        for i in range(0, len(body), 1000):
            yield body[i : i + 1000]

    # A generator body is sent with chunked transfer encoding: no Content-Length.
    r = small_client.post(
        "/v1/score/batch", content=chunks(), headers={"content-type": "application/json"}
    )
    assert r.status_code == 413
    assert "too large" in r.json()["detail"]


def test_streamed_body_within_limit_is_scored(small_client):
    body = json.dumps(make_txn(transaction_id="s1")).encode()
    r = small_client.post(
        "/v1/score",
        content=iter([body[:50], body[50:]]),
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 200
    assert r.json()["transaction_id"] == "s1"


@pytest.mark.parametrize("value", ["abc", "-1", "1.5", "", "+10"])
def test_malformed_content_length_rejected_with_400(value: str):
    status, body = _call_raw([(b"content-length", value.encode())])
    assert status == 400
    assert body == {"detail": "Invalid Content-Length header"}


def test_conflicting_duplicate_content_length_rejected_with_400():
    status, _ = _call_raw([(b"content-length", b"10"), (b"content-length", b"20")])
    assert status == 400


def test_agreeing_duplicate_content_length_allowed():
    status, _ = _call_raw([(b"content-length", b"2"), (b"content-length", b"2")], body=b"{}")
    assert status == 200


def test_default_limit_fits_a_full_default_batch():
    settings = Settings()
    assert settings.max_body_bytes == DEFAULT_MAX_BODY_BYTES
    assert len(_batch_body(settings.max_batch_size)) < settings.max_body_bytes


def test_default_settings_accept_full_batch():
    with TestClient(create_app(Settings())) as c:
        r = c.post(
            "/v1/score/batch",
            content=_batch_body(1000),
            headers={"content-type": "application/json"},
        )
    assert r.status_code == 200
    assert r.json()["count"] == 1000


def test_settings_validation_and_env(monkeypatch):
    with pytest.raises(ValueError, match="max_body_bytes"):
        Settings(max_body_bytes=100)
    with pytest.raises(ValueError, match="max_batch_size"):
        Settings(max_batch_size=0)
    monkeypatch.setenv("FRE_MAX_BODY_BYTES", "65536")
    assert Settings.from_env().max_body_bytes == 65536


def test_middleware_rejects_tiny_limit():
    async def app(scope, receive, send):  # pragma: no cover - never called
        pass

    with pytest.raises(ValueError):
        BodySizeLimitMiddleware(app, max_body_bytes=10)


# --- raw ASGI checks -------------------------------------------------------------


async def _echo_app(scope: dict[str, Any], receive, send) -> None:
    """Reads the whole body outside FastAPI, then replies 200."""
    while True:
        message = await receive()
        if not message.get("more_body"):
            break
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"{}"})


def _run(
    app,
    headers: list[tuple[bytes, bytes]],
    chunks: list[bytes],
    limit: int = 1024,
) -> tuple[int, list[dict[str, Any]]]:
    sent: list[dict[str, Any]] = []
    pending = list(chunks)

    async def receive() -> dict[str, Any]:
        body = pending.pop(0) if pending else b""
        return {"type": "http.request", "body": body, "more_body": bool(pending)}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/", "headers": headers}
    asyncio.run(BodySizeLimitMiddleware(app, max_body_bytes=limit)(scope, receive, send))
    return sent[0]["status"], sent


def _call_raw(headers: list[tuple[bytes, bytes]], body: bytes = b"") -> tuple[int, Any]:
    status, sent = _run(_echo_app, headers, [body])
    return status, json.loads(sent[-1]["body"])


def test_streamed_overflow_outside_fastapi_gets_413():
    status, sent = _run(_echo_app, [], [b"x" * 600, b"x" * 600])
    assert status == 413
    assert json.loads(sent[-1]["body"])["detail"].startswith("Request body too large")


def test_overflow_after_response_started_propagates():
    async def eager_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        while (await receive()).get("more_body"):
            pass

    with pytest.raises(BodyTooLarge):
        _run(eager_app, [], [b"x" * 600, b"x" * 600])


def test_non_http_scopes_pass_through():
    seen: list[str] = []

    async def app(scope, receive, send):
        seen.append(scope["type"])

    async def noop_receive():  # pragma: no cover - never called
        return {}

    async def noop_send(message):  # pragma: no cover - never called
        pass

    mw = BodySizeLimitMiddleware(app)
    asyncio.run(mw({"type": "lifespan"}, noop_receive, noop_send))
    assert seen == ["lifespan"]
