"""Optional API-key authentication for the ``/v1/*`` endpoints, enforced before body parsing.

Two scopes keep the two kinds of caller apart:

* ``score``: ``POST /v1/score``, ``POST /v1/score/batch``, ``GET /v1/rules`` and any other
  ``/v1/*`` path (deny by default, so a new endpoint is never accidentally public). Keys come
  from ``FRE_API_KEYS``.
* ``labels``: ``POST /v1/labels``. Keys come from ``FRE_LABEL_API_KEYS``; when that is unset
  the ``score`` keys are accepted instead. Giving the labelling job its own keys means a
  payment client that can score cannot also post fraud labels, which would otherwise let it
  poison the ``fre_label_*`` metrics and the retraining feedback log.

A scope with no keys configured is open, so local development keeps working unchanged.
``/health``, ``/ready``, ``/metrics`` and the OpenAPI docs are never authenticated: probes and
scrapers stay simple and are expected to be protected by the network.

Keys are sent as ``Authorization: Bearer <key>`` or ``x-api-key: <key>``. The middleware is
pure ASGI and runs before FastAPI reads the request, so an unauthenticated client cannot make
the server buffer or parse a body, and gets no ``422`` that would reveal the schema. Keys are
stored only as SHA-256 digests and compared in constant time against every configured key.
They never appear in logs, responses, ``repr(Settings)`` or metrics; failures are counted in
``fre_auth_failures_total{scope, reason}`` (bounded labels).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
from collections import Counter
from collections.abc import Iterable
from enum import StrEnum

from .limits import ASGIApp, Receive, Scope, Send

MIN_KEY_LENGTH = 16
MAX_KEY_LENGTH = 256
PROTECTED_PREFIX = "/v1/"
_REALM = b'Bearer realm="fraud-engine"'


class AuthScope(StrEnum):
    SCORE = "score"
    LABELS = "labels"


class FailureReason(StrEnum):
    MISSING = "missing"
    INVALID = "invalid"


def parse_keys(raw: str | None) -> tuple[str, ...]:
    """Split a comma-separated key list, dropping blanks and surrounding whitespace."""
    if not raw:
        return ()
    return tuple(k.strip() for k in raw.split(",") if k.strip())


def validate_keys(name: str, keys: Iterable[str]) -> None:
    """Reject keys that are too short/long to be safe or that would not survive a header."""
    for key in keys:
        if not MIN_KEY_LENGTH <= len(key) <= MAX_KEY_LENGTH:
            raise ValueError(
                f"{name}: every key must be {MIN_KEY_LENGTH}-{MAX_KEY_LENGTH} characters"
            )
        if not key.isascii() or not key.isprintable() or any(c in key for c in " ,\t"):
            raise ValueError(f"{name}: keys must be printable ASCII without spaces or commas")


def scope_for_path(path: str) -> AuthScope | None:
    """The scope a request path needs, or ``None`` if the path is never authenticated."""
    if not path.startswith(PROTECTED_PREFIX):
        return None
    if path.rstrip("/") == "/v1/labels":
        return AuthScope.LABELS
    return AuthScope.SCORE


def _digest(key: str) -> bytes:
    return hashlib.sha256(key.encode()).digest()


class ApiKeyAuth:
    """Holds the key digests per scope and counts authentication failures."""

    def __init__(self, score_keys: Iterable[str] = (), label_keys: Iterable[str] = ()) -> None:
        score = [_digest(k) for k in score_keys]
        labels = [_digest(k) for k in label_keys] or score
        self._keys: dict[AuthScope, tuple[bytes, ...]] = {
            AuthScope.SCORE: tuple(score),
            AuthScope.LABELS: tuple(labels),
        }
        self._lock = threading.Lock()
        self.failures: Counter[tuple[str, str]] = Counter()

    def enabled(self, scope: AuthScope) -> bool:
        return bool(self._keys[scope])

    @property
    def any_enabled(self) -> bool:
        return any(self.enabled(s) for s in AuthScope)

    @property
    def open_scopes(self) -> list[str]:
        return [s.value for s in AuthScope if not self.enabled(s)]

    def check(self, scope: AuthScope, presented: str | None) -> FailureReason | None:
        """``None`` if the request may proceed, otherwise why it is rejected."""
        allowed = self._keys[scope]
        if not allowed:
            return None
        if presented is None:
            return self._fail(scope, FailureReason.MISSING)
        if len(presented) > MAX_KEY_LENGTH:
            return self._fail(scope, FailureReason.INVALID)
        candidate = _digest(presented)
        match = False
        for key in allowed:  # no early exit: timing doesn't reveal which key matched
            match |= hmac.compare_digest(candidate, key)
        return None if match else self._fail(scope, FailureReason.INVALID)

    def _fail(self, scope: AuthScope, reason: FailureReason) -> FailureReason:
        with self._lock:
            self.failures[(scope.value, reason.value)] += 1
        return reason

    def render(self) -> list[str]:
        with self._lock:
            lines = [
                "# HELP fre_auth_failures_total Rejected /v1 requests by scope and reason.",
                "# TYPE fre_auth_failures_total counter",
            ]
            for s in AuthScope:
                if not self.enabled(s):
                    continue
                for r in FailureReason:
                    n = self.failures[(s.value, r.value)]
                    lines.append(f'fre_auth_failures_total{{scope="{s}",reason="{r}"}} {n}')
            return lines


def presented_key(scope: Scope) -> str | None:
    """The key from ``x-api-key`` or ``Authorization: Bearer``.

    Returns ``""`` (treated as invalid) when the headers are ambiguous: repeated, or both set
    to different values, so a proxy and the app can't disagree about which key was used.
    """
    api_keys: list[bytes] = []
    bearers: list[bytes] = []
    for k, v in scope.get("headers", []):
        name = k.lower()
        if name == b"x-api-key":
            api_keys.append(v.strip())
        elif name == b"authorization":
            scheme, _, token = v.strip().partition(b" ")
            bearers.append(token.strip() if scheme.lower() == b"bearer" else b"")
    found = api_keys + bearers
    if not found:
        return None
    if len(api_keys) > 1 or len(bearers) > 1 or len(set(found)) != 1:
        return ""
    try:
        return found[0].decode("ascii")
    except UnicodeDecodeError:
        return ""


class ApiKeyMiddleware:
    """Reject ``/v1/*`` requests without a valid key for their scope with ``401``."""

    def __init__(self, app: ASGIApp, auth: ApiKeyAuth) -> None:
        self.app = app
        self.auth = auth

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        needed = scope_for_path(scope.get("path", "")) if scope["type"] == "http" else None
        if needed is None:
            await self.app(scope, receive, send)
            return
        reason = self.auth.check(needed, presented_key(scope))
        if reason is None:
            await self.app(scope, receive, send)
            return
        detail = "Missing API key" if reason is FailureReason.MISSING else "Invalid API key"
        body = json.dumps({"detail": detail}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"www-authenticate", _REALM),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
