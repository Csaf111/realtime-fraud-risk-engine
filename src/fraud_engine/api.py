"""FastAPI application.

Run locally:  uvicorn fraud_engine.api:app --reload
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse

from . import __version__
from .config import Settings
from .drift import DriftMonitor, ScoreReference
from .metrics import Metrics
from .schemas import BatchRequest, BatchResponse, RiskResult, Transaction
from .scoring import RiskEngine, ShadowResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("fraud_engine.api")


def _observe(m: Metrics, result: RiskResult, shadow: ShadowResult | None = None) -> None:
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


def _drift_monitor(settings: Settings) -> DriftMonitor | None:
    if not settings.score_reference_path:
        return None
    reference = ScoreReference.load(settings.score_reference_path)
    return DriftMonitor(reference, settings.drift_window, settings.drift_min_samples)


def create_app(settings: Settings | None = None, engine: RiskEngine | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine or RiskEngine(settings)
        app.state.metrics = Metrics(drift=_drift_monitor(settings))
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

    @app.get("/ready", tags=["ops"])
    def ready(request: Request) -> dict[str, str]:
        """Readiness: the engine (and model, if configured) is loaded."""
        eng: RiskEngine = request.app.state.engine
        body = {
            "status": "ready",
            "mode": eng.mode,
            "model_version": eng.model_version,
            "environment": settings.environment,
            "drift_reference": "loaded" if request.app.state.metrics.drift else "none",
        }
        if eng.shadow_model_version is not None:
            body["shadow_model_version"] = eng.shadow_model_version
        return body

    @app.post("/v1/score", response_model=RiskResult, tags=["scoring"])
    def score(txn: Transaction, request: Request) -> RiskResult:
        """Score one transaction and return the decision with its reasons."""
        eng: RiskEngine = request.app.state.engine
        m: Metrics = request.app.state.metrics
        m.request("score")
        result, shadow = eng.score_with_shadow(txn)
        _observe(m, result, shadow)
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
        results = []
        for t in batch.transactions:
            r, shadow = eng.score_with_shadow(t, update_state=batch.update_state)
            _observe(m, r, shadow)
            results.append(r)
        return BatchResponse(results=results, count=len(results))

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
