"""FastAPI application.

Run locally:  uvicorn fraud_engine.api:app --reload
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, Protocol, runtime_checkable

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from . import __version__
from .auth import ApiKeyAuth, ApiKeyMiddleware
from .config import Settings
from .drift import DriftMonitor, ScoreReference
from .feature_drift import FeatureDriftMonitor, FeatureReference
from .feedback import FeedbackLog
from .idempotency import (
    IdempotencyBackend,
    IdempotencyConflictError,
    LookupStatus,
    build_idempotency_store,
    fingerprint,
)
from .limits import BodySizeLimitMiddleware
from .metrics import Metrics
from .outcomes import LabelStatus, OutcomeTracker
from .schemas import (
    BatchRequest,
    BatchResponse,
    LabelBatch,
    LabelBatchResponse,
    RiskResult,
    Transaction,
)
from .scoring import RiskEngine, ScoredTransaction
from .state import VelocityStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("fraud_engine.api")


def _observe(m: Metrics, scored: ScoredTransaction, served: bool = True) -> None:
    result, shadow = scored.result, scored.shadow
    if served and m.outcomes is not None:
        m.outcomes.record_decision(
            result.transaction_id,
            result.risk_score,
            result.decision.value,
            features=scored.features,
            step=scored.step,
            model_version=result.model_version,
        )
    m.observe(
        result.decision.value,
        result.latency_ms,
        risk_score=result.risk_score,
        rule_codes=[h.code for h in result.rule_hits],
    )
    if shadow is not None:
        m.observe_shadow(
            result.decision.value,
            result.risk_score,
            None if shadow.decision is None else shadow.decision.value,
            shadow.risk_score,
            shadow.latency_ms,
        )


def _score_served(
    eng: RiskEngine, m: Metrics, cache: IdempotencyBackend | None, txn: Transaction
) -> tuple[RiskResult, bool]:
    """Score a state-updating transaction once per ``transaction_id``.

    Returns ``(result, replayed)``. A replay returns the original result and touches neither
    velocity state nor metrics. Raises ``HTTPException`` 409 on an id reused with a different
    payload and 503 if a concurrent duplicate is still being scored.
    """
    txn_id = txn.transaction_id
    if cache is None or not txn_id:
        scored = eng.score_detailed(txn)
        _observe(m, scored)
        return scored.result, False
    try:
        lookup = cache.claim(txn_id, fingerprint(txn.model_dump()))
    except IdempotencyConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={"retry-after": "1"}) from exc
    if lookup.status is LookupStatus.REPLAY:
        assert lookup.result is not None
        return lookup.result, True
    try:
        scored = eng.score_detailed(txn)
    except BaseException:
        cache.release(txn_id)
        raise
    cache.complete(txn_id, scored.result)
    _observe(m, scored)
    return scored.result, False


def _check_batch_ids(transactions: list[Transaction]) -> None:
    """Reject a batch that reuses one ``transaction_id`` for different payloads, before any
    transaction in it is scored (identical repeats are fine: they replay)."""
    seen: dict[str, str] = {}
    for t in transactions:
        if not t.transaction_id:
            continue
        fp = fingerprint(t.model_dump())
        if seen.setdefault(t.transaction_id, fp) != fp:
            raise HTTPException(
                status_code=409,
                detail=f"transaction_id {t.transaction_id!r} appears in the batch with "
                "different payloads",
            )


@runtime_checkable
class _Pingable(Protocol):
    def ping(self) -> None: ...


def _check_dependencies(app: FastAPI) -> dict[str, str]:
    """Ping every shared backend (Redis velocity / idempotency store) the replica needs.

    In-process backends have nothing to ping and are omitted. A failed ping is reported as
    ``"error: <ExceptionType>"`` only: the exception text (which can carry hostnames or
    credentials from the Redis URL) goes to the log, never to the probe response.
    """
    backends: dict[str, Any] = {
        "velocity_store": app.state.engine.store,
        "idempotency_store": app.state.idempotency,
    }
    checks: dict[str, str] = {}
    for name, backend in backends.items():
        if not isinstance(backend, _Pingable):
            continue
        try:
            backend.ping()
        except Exception as exc:
            log.warning("Readiness check failed for %s: %r", name, exc)
            checks[name] = f"error: {type(exc).__name__}"
        else:
            checks[name] = "ok"
    return checks


def _drift_monitor(settings: Settings) -> DriftMonitor | None:
    if not settings.score_reference_path:
        return None
    reference = ScoreReference.load(settings.score_reference_path)
    return DriftMonitor(reference, settings.drift_window, settings.drift_min_samples)


def _feature_drift_monitor(settings: Settings) -> FeatureDriftMonitor | None:
    if not settings.feature_reference_path:
        return None
    reference = FeatureReference.load(settings.feature_reference_path)
    return FeatureDriftMonitor(reference, settings.drift_window, settings.drift_min_samples)


def create_app(settings: Settings | None = None, engine: RiskEngine | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    auth = ApiKeyAuth(settings.api_keys, settings.label_api_keys)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        feature_drift = _feature_drift_monitor(settings)
        app.state.engine = engine or RiskEngine(settings)
        if feature_drift is not None:
            app.state.engine.feature_observer = feature_drift.observe
        store = app.state.engine.store
        app.state.idempotency = build_idempotency_store(settings)
        app.state.metrics = Metrics(
            drift=_drift_monitor(settings),
            feature_drift=feature_drift,
            outcomes=OutcomeTracker(
                settings.label_pending_capacity,
                settings.label_window,
                feedback=(
                    FeedbackLog(settings.feedback_log_path) if settings.feedback_log_path else None
                ),
            ),
            velocity=store if isinstance(store, VelocityStore) else None,
            idempotency=app.state.idempotency,
            auth=auth,
        )
        if auth.open_scopes and settings.environment in ("staging", "production"):
            log.warning(
                "API-key authentication is OFF for scope(s) %s in %s: set FRE_API_KEYS "
                "(and FRE_LABEL_API_KEYS) or enforce authentication at the gateway",
                ",".join(auth.open_scopes),
                settings.environment,
            )
        log.info(
            "Engine ready: mode=%s version=%s env=%s",
            app.state.engine.mode,
            app.state.engine.model_version,
            settings.environment,
        )
        yield

    app = FastAPI(
        title="Realtime Fraud Risk Engine",
        version=__version__,
        description="Explainable real-time fraud risk scoring for PaySim-style transactions.",
        lifespan=lifespan,
    )

    # Added before the request-id middleware so it sits *inside* it: 413/400 responses
    # from the size limit still carry an x-request-id.
    app.add_middleware(BodySizeLimitMiddleware, max_body_bytes=settings.max_body_bytes)
    # Outside the size limit (an unauthenticated client gets 401 before any body handling)
    # and inside the request-id middleware (the 401 still carries an x-request-id).
    app.add_middleware(ApiKeyMiddleware, auth=auth)

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        response: Response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, str]:
        """Liveness: the process is up."""
        return {"status": "ok", "version": __version__}

    @app.get(
        "/ready",
        tags=["ops"],
        responses={503: {"description": "A shared backend (e.g. Redis) is unreachable"}},
    )
    def ready(request: Request) -> Any:
        """Readiness: the engine (and model, if configured) is loaded and every shared
        backend answers a ping. Returns ``503`` with ``status: not_ready`` otherwise, so a
        load balancer stops routing to a replica that would fail every scoring request."""
        eng: RiskEngine = request.app.state.engine
        checks = _check_dependencies(request.app)
        healthy = all(v == "ok" for v in checks.values())
        body: dict[str, Any] = {
            "status": "ready" if healthy else "not_ready",
            "mode": eng.mode,
            "model_version": eng.model_version,
            "environment": settings.environment,
            "drift_reference": "loaded" if request.app.state.metrics.drift else "none",
            "feature_drift_reference": (
                "loaded" if request.app.state.metrics.feature_drift else "none"
            ),
        }
        if checks:
            body["dependencies"] = checks
        if eng.shadow_model_version is not None:
            body["shadow_model_version"] = eng.shadow_model_version
        if not healthy:
            return JSONResponse(status_code=503, content=body)
        return body

    @app.post("/v1/score", response_model=RiskResult, tags=["scoring"])
    def score(txn: Transaction, request: Request, response: Response) -> RiskResult:
        """Score one transaction and return the decision with its reasons.

        Idempotent per ``transaction_id``: a retry with the same payload returns the original
        result (header ``x-idempotent-replay: true``) without re-recording velocity; the same
        id with a different payload is rejected with ``409``.
        """
        eng: RiskEngine = request.app.state.engine
        m: Metrics = request.app.state.metrics
        m.request("score")
        result, replayed = _score_served(eng, m, request.app.state.idempotency, txn)
        if replayed:
            response.headers["x-idempotent-replay"] = "true"
        return result

    @app.post("/v1/score/batch", response_model=BatchResponse, tags=["scoring"])
    def score_batch(batch: BatchRequest, request: Request) -> BatchResponse:
        """Score transactions in the order given (order matters for velocity features)."""
        if len(batch.transactions) > settings.max_batch_size:
            raise HTTPException(
                status_code=413,
                detail=f"Batch too large: max {settings.max_batch_size} transactions",
            )
        eng: RiskEngine = request.app.state.engine
        m: Metrics = request.app.state.metrics
        m.request("score_batch")
        cache: IdempotencyBackend | None = request.app.state.idempotency
        if batch.update_state and cache is not None:
            _check_batch_ids(batch.transactions)
        results = []
        for t in batch.transactions:
            if batch.update_state:
                # Same idempotency as /v1/score: a repeated id (within this batch or from an
                # earlier request) gets its original result and is not recorded again.
                result, _ = _score_served(eng, m, cache, t)
                results.append(result)
                continue
            # What-if scoring changes no state, so it bypasses the idempotency cache, and it
            # is not a served decision: don't join labels.
            scored = eng.score_detailed(t, update_state=False)
            _observe(m, scored, served=False)
            results.append(scored.result)
        return BatchResponse(results=results, count=len(results))

    @app.post("/v1/labels", response_model=LabelBatchResponse, tags=["monitoring"])
    def post_labels(batch: LabelBatch, request: Request) -> LabelBatchResponse:
        """Report delayed fraud outcomes for previously scored transactions.

        Labels feed only the ``fre_label_*`` performance metrics and, when
        ``FRE_FEEDBACK_LOG_PATH`` is set, the offline retraining log. They never change
        features, rules, the model or velocity state, so they cannot leak into later scores.
        """
        if len(batch.labels) > settings.max_batch_size:
            raise HTTPException(
                status_code=413,
                detail=f"Batch too large: max {settings.max_batch_size} labels",
            )
        m: Metrics = request.app.state.metrics
        m.request("labels")
        tracker = m.outcomes
        if tracker is None or not tracker.enabled:
            raise HTTPException(status_code=404, detail="Label tracking is disabled")
        counts = {s: 0 for s in LabelStatus}
        for label in batch.labels:
            counts[tracker.record_label(label.transaction_id, label.is_fraud)] += 1
        return LabelBatchResponse(
            matched=counts[LabelStatus.MATCHED],
            unknown=counts[LabelStatus.UNKNOWN],
            duplicate=counts[LabelStatus.DUPLICATE],
        )

    @app.get("/v1/rules", tags=["explainability"])
    def list_rules(request: Request) -> list[dict[str, object]]:
        """The full rulebook, so every reason code in a response can be looked up."""
        eng: RiskEngine = request.app.state.engine
        return [
            {"code": r.code, "description": r.description, "points": r.points} for r in eng.rules
        ]

    @app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
    def metrics(request: Request) -> str:
        return request.app.state.metrics.render()

    return app


app = create_app()
