"""API-key authentication: scopes, header handling, pre-parse rejection and secrecy."""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from fraud_engine.api import create_app
from fraud_engine.auth import (
    ApiKeyAuth,
    AuthScope,
    FailureReason,
    parse_keys,
    presented_key,
    scope_for_path,
)
from fraud_engine.config import Settings

from .conftest import make_txn

SCORE_KEY = "score-key-0123456789abcdef"
SCORE_KEY_2 = "score-key-rotated-0123456789"
LABEL_KEY = "label-key-0123456789abcdef"


def _client(**kw) -> TestClient:
    return TestClient(create_app(Settings(**kw)))


@pytest.fixture
def secured():
    with _client(api_keys=(SCORE_KEY, SCORE_KEY_2), label_api_keys=(LABEL_KEY,)) as c:
        yield c


def _bearer(key: str) -> dict[str, str]:
    return {"authorization": f"Bearer {key}"}


# --- open by default ---------------------------------------------------------------------


def test_no_keys_configured_leaves_api_open(client):
    assert client.post("/v1/score", json=make_txn()).status_code == 200
    assert "fre_auth_failures_total" not in client.get("/metrics").text


# --- scoring scope -----------------------------------------------------------------------


def test_missing_key_is_401_with_challenge_and_request_id(secured):
    resp = secured.post("/v1/score", json=make_txn())
    assert resp.status_code == 401
    assert resp.json() == {"detail": "Missing API key"}
    assert resp.headers["www-authenticate"].startswith("Bearer")
    assert resp.headers["x-request-id"]


@pytest.mark.parametrize("key", [SCORE_KEY, SCORE_KEY_2])
def test_any_configured_key_works_via_bearer_or_header(secured, key):
    assert secured.post("/v1/score", json=make_txn(), headers=_bearer(key)).status_code == 200
    assert secured.post("/v1/score", json=make_txn(), headers={"x-api-key": key}).status_code == 200


def test_bearer_scheme_is_case_insensitive(secured):
    resp = secured.post(
        "/v1/score", json=make_txn(), headers={"authorization": f"bearer {SCORE_KEY}"}
    )
    assert resp.status_code == 200


@pytest.mark.parametrize(
    "headers",
    [
        {"x-api-key": "wrong-key-0123456789abcdef"},
        {"authorization": f"Basic {SCORE_KEY}"},
        {"x-api-key": SCORE_KEY + "x"},
        {"x-api-key": "k" * 1000},
    ],
)
def test_wrong_key_is_401_invalid(secured, headers):
    resp = secured.post("/v1/score", json=make_txn(), headers=headers)
    assert resp.status_code == 401
    assert resp.json() == {"detail": "Invalid API key"}


def test_all_v1_paths_are_protected_including_batch_rules_and_unknown(secured):
    batch = {"transactions": [make_txn()]}
    assert secured.post("/v1/score/batch", json=batch).status_code == 401
    assert secured.get("/v1/rules").status_code == 401
    assert secured.get("/v1/not-a-route").status_code == 401  # deny by default
    assert (
        secured.post("/v1/score/batch", json=batch, headers=_bearer(SCORE_KEY)).json()["count"] == 1
    )
    assert secured.get("/v1/rules", headers=_bearer(SCORE_KEY)).status_code == 200


def test_ops_endpoints_stay_open(secured):
    for path in ("/health", "/ready", "/metrics", "/openapi.json"):
        assert secured.get(path).status_code == 200, path


def test_rejected_before_body_is_parsed(secured):
    # Malformed JSON would be a 422 if it reached FastAPI: 401 proves it never did, so an
    # unauthenticated caller learns nothing about the schema and costs no parsing.
    resp = secured.post(
        "/v1/score", content=b"{not json", headers={"content-type": "application/json"}
    )
    assert resp.status_code == 401


def test_rejected_request_does_not_touch_state_or_counters(secured):
    for _ in range(3):
        secured.post("/v1/score", json=make_txn(transaction_id="t1"))
    text = secured.get("/metrics").text
    assert 'fre_requests_total{endpoint="score"}' not in text
    assert "fre_scored_total 0" in text
    # The first authenticated attempt is a fresh score, not an idempotent replay.
    resp = secured.post("/v1/score", json=make_txn(transaction_id="t1"), headers=_bearer(SCORE_KEY))
    assert resp.status_code == 200
    assert "x-idempotent-replay" not in resp.headers


# --- labels scope ------------------------------------------------------------------------


def test_label_scope_is_separate_from_scoring(secured):
    labels = {"labels": [{"transaction_id": "t1", "is_fraud": True}]}
    # A scoring client cannot post labels (that would poison metrics and the feedback log)...
    assert secured.post("/v1/labels", json=labels, headers=_bearer(SCORE_KEY)).status_code == 401
    # ...and the labelling job cannot score.
    assert secured.post("/v1/score", json=make_txn(), headers=_bearer(LABEL_KEY)).status_code == 401
    assert secured.post("/v1/labels", json=labels, headers=_bearer(LABEL_KEY)).status_code == 200


def test_label_scope_falls_back_to_score_keys():
    with _client(api_keys=(SCORE_KEY,)) as c:
        labels = {"labels": [{"transaction_id": "t1", "is_fraud": False}]}
        assert c.post("/v1/labels", json=labels).status_code == 401
        assert c.post("/v1/labels", json=labels, headers=_bearer(SCORE_KEY)).status_code == 200


def test_only_label_keys_leaves_scoring_open():
    with _client(label_api_keys=(LABEL_KEY,)) as c:
        assert c.post("/v1/score", json=make_txn()).status_code == 200
        labels = {"labels": [{"transaction_id": "t1", "is_fraud": False}]}
        assert c.post("/v1/labels", json=labels).status_code == 401
        text = c.get("/metrics").text
        assert 'scope="labels",reason="missing"} 1' in text
        assert 'scope="score"' not in text  # open scope: no failure series


# --- header ambiguity --------------------------------------------------------------------


def test_conflicting_headers_are_rejected(secured):
    headers = {"x-api-key": SCORE_KEY, **_bearer(SCORE_KEY_2)}
    assert secured.post("/v1/score", json=make_txn(), headers=headers).status_code == 401


def test_matching_headers_are_accepted(secured):
    headers = {"x-api-key": SCORE_KEY, **_bearer(SCORE_KEY)}
    assert secured.post("/v1/score", json=make_txn(), headers=headers).status_code == 200


def test_presented_key_edge_cases():
    def scope(*headers: tuple[bytes, bytes]):
        return {"headers": list(headers)}

    assert presented_key(scope()) is None
    assert presented_key(scope((b"X-Api-Key", b" abc "))) == "abc"
    assert presented_key(scope((b"x-api-key", b"a"), (b"x-api-key", b"a"))) == ""
    assert presented_key(scope((b"authorization", b"Bearer"))) == ""
    assert presented_key(scope((b"x-api-key", "é".encode()))) == ""


# --- metrics and secrecy -----------------------------------------------------------------


def test_failures_are_counted_by_scope_and_reason(secured):
    secured.post("/v1/score", json=make_txn())
    secured.post("/v1/score", json=make_txn(), headers={"x-api-key": "nope-nope-nope-nope"})
    secured.post("/v1/labels", json={"labels": []})
    text = secured.get("/metrics").text
    assert 'fre_auth_failures_total{scope="score",reason="missing"} 1' in text
    assert 'fre_auth_failures_total{scope="score",reason="invalid"} 1' in text
    assert 'fre_auth_failures_total{scope="labels",reason="missing"} 1' in text
    for key in (SCORE_KEY, SCORE_KEY_2, LABEL_KEY):
        assert key not in text


def test_keys_never_in_settings_repr():
    s = Settings(api_keys=(SCORE_KEY,), label_api_keys=(LABEL_KEY,))
    assert SCORE_KEY not in repr(s)
    assert LABEL_KEY not in repr(s)


def test_open_scope_warns_in_production(caplog):
    with (
        caplog.at_level(logging.WARNING, logger="fraud_engine.api"),
        _client(environment="production", label_api_keys=(LABEL_KEY,)),
    ):
        pass
    assert "authentication is OFF for scope(s) score" in caplog.text
    assert LABEL_KEY not in caplog.text


def test_no_warning_when_fully_secured(caplog):
    with (
        caplog.at_level(logging.WARNING, logger="fraud_engine.api"),
        _client(environment="production", api_keys=(SCORE_KEY,)),
    ):
        pass
    assert "authentication is OFF" not in caplog.text


# --- configuration -----------------------------------------------------------------------


def test_parse_keys():
    assert parse_keys(None) == ()
    assert parse_keys("") == ()
    assert parse_keys(f" {SCORE_KEY} ,, {SCORE_KEY_2},") == (SCORE_KEY, SCORE_KEY_2)


def test_from_env_reads_keys(monkeypatch):
    monkeypatch.setenv("FRE_API_KEYS", f"{SCORE_KEY},{SCORE_KEY_2}")
    monkeypatch.setenv("FRE_LABEL_API_KEYS", LABEL_KEY)
    s = Settings.from_env()
    assert s.api_keys == (SCORE_KEY, SCORE_KEY_2)
    assert s.label_api_keys == (LABEL_KEY,)


@pytest.mark.parametrize(
    "bad",
    ["short", "x" * 257, "has space in the key!!", "tab\tin-the-key-12345", "ünïcode-key-123456"],
)
def test_weak_or_malformed_keys_rejected(bad):
    with pytest.raises(ValueError):
        Settings(api_keys=(bad,))
    with pytest.raises(ValueError):
        Settings(label_api_keys=(bad,))


def test_scope_for_path():
    assert scope_for_path("/health") is None
    assert scope_for_path("/metrics") is None
    assert scope_for_path("/v1") is None
    assert scope_for_path("/v1/score") is AuthScope.SCORE
    assert scope_for_path("/v1/labels") is AuthScope.LABELS
    assert scope_for_path("/v1/labels/") is AuthScope.LABELS


def test_check_unit():
    auth = ApiKeyAuth([SCORE_KEY])
    assert auth.check(AuthScope.SCORE, SCORE_KEY) is None
    assert auth.check(AuthScope.SCORE, None) is FailureReason.MISSING
    assert auth.check(AuthScope.SCORE, "") is FailureReason.INVALID
    assert auth.check(AuthScope.LABELS, "x" * 300) is FailureReason.INVALID
    assert auth.any_enabled
    assert auth.open_scopes == []
    assert ApiKeyAuth().open_scopes == ["score", "labels"]
