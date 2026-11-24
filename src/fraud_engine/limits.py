"""Request body size limit, enforced before any JSON is parsed.

``FRE_MAX_BATCH_SIZE`` caps how many transactions a batch may hold, but only *after*
FastAPI has read and parsed the whole body, so a client could still make the server
buffer and decode an arbitrarily large payload. :class:`BodySizeLimitMiddleware` is a
pure ASGI middleware that stops that early:

* a ``Content-Length`` above the limit is rejected with ``413`` without reading a byte;
* a malformed ``Content-Length`` is rejected with ``400``;
* bodies without a length (chunked transfer) are counted as they stream in and the
  request fails with ``413`` as soon as the running total passes the limit, so at most
  one chunk beyond the limit is ever held in memory.

The limit applies to every HTTP request, so the default must comfortably fit a full
batch of ``FRE_MAX_BATCH_SIZE`` transactions (roughly 300 bytes each as JSON).
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from starlette.exceptions import HTTPException

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

DEFAULT_MAX_BODY_BYTES = 2 * 1024 * 1024  # 2 MiB: a 1000-transaction batch is ~0.3 MiB
MIN_MAX_BODY_BYTES = 1024


class BodyTooLarge(HTTPException):
    """Raised from ``receive`` when a streamed body passes the limit.

    It subclasses ``HTTPException`` so FastAPI's body parsing re-raises it unchanged
    (instead of wrapping it as a generic 400) and the exception middleware renders a 413.
    """

    def __init__(self, limit: int) -> None:
        super().__init__(status_code=413, detail=_too_large_detail(limit))


def _too_large_detail(limit: int) -> str:
    return f"Request body too large: max {limit} bytes"


async def _send_json(send: Send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"connection", b"close"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _content_length(scope: Scope) -> int | str | None:
    """The declared body length, ``None`` if absent, or ``"invalid"``."""
    values = [v for k, v in scope.get("headers", []) if k.lower() == b"content-length"]
    if not values:
        return None
    # Duplicate headers must agree (RFC 9110 §8.6); anything else is ambiguous framing.
    if len(set(values)) != 1:
        return "invalid"
    raw = values[0].strip()
    if not raw.isdigit():  # rejects signs, spaces, decimals and the empty string
        return "invalid"
    return int(raw)


class BodySizeLimitMiddleware:
    """Reject HTTP requests whose body exceeds ``max_body_bytes``."""

    def __init__(self, app: ASGIApp, max_body_bytes: int = DEFAULT_MAX_BODY_BYTES) -> None:
        if max_body_bytes < MIN_MAX_BODY_BYTES:
            raise ValueError(f"max_body_bytes must be >= {MIN_MAX_BODY_BYTES}")
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit = self.max_body_bytes
        declared = _content_length(scope)
        if declared == "invalid":
            await _send_json(send, 400, "Invalid Content-Length header")
            return
        if isinstance(declared, int) and declared > limit:
            await _send_json(send, 413, _too_large_detail(limit))
            return

        received = 0
        started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise BodyTooLarge(limit)
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except BodyTooLarge:
            # Only reached if something read the body outside FastAPI's handler
            # (e.g. another middleware); FastAPI itself renders the 413.
            if started:
                raise
            await _send_json(send, 413, _too_large_detail(limit))
