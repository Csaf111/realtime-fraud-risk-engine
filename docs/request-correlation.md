# Request correlation ID boundary

Clients may supply an `x-request-id` header to correlate their requests with upstream traces. This value is untrusted input.

The API preserves identifiers only when they are 1–128 ASCII characters, begin with an alphanumeric character, and otherwise contain alphanumerics, dots, underscores, colons or hyphens. Missing, malformed or oversized identifiers are replaced by a generated UUIDv4. The accepted value is echoed in the response header.

This limits response-header reflection, oversized correlation labels and ambiguous delimiter/control characters. The API does **not** log the identifier by default, so this change is not a substitute for structured trace propagation, authentication, rate limiting, or a privacy-safe audit trail.

## Validation

Run `pytest tests/test_request_ids.py tests/test_api.py` and `ruff check src tests`. Regression tests cover valid identifiers, invalid Unicode/delimiters, oversized values and HTTP responses. Review and verify GitHub Actions before merging.
