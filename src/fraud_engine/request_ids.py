"""Request correlation identifiers are untrusted input from HTTP clients.

Only short, printable ASCII tokens are echoed in response headers. Invalid or
oversized identifiers are replaced with a server-generated UUID.
"""

from __future__ import annotations

import re
import uuid

_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z", re.ASCII)


def safe_request_id(candidate: str | None) -> str:
    """Return a safe correlation ID without reflecting arbitrary header content."""
    if candidate is not None and _REQUEST_ID.fullmatch(candidate):
        return candidate
    return str(uuid.uuid4())
