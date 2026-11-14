"""The risk engine: features -> rules (+ optional model) -> score, decision, reasons.

Optionally a *shadow* model is scored next to the primary path. It receives exactly the
same point-in-time features and rules score (computed once, before the event is recorded
in the velocity store), so a shadow comparison has online/offline parity by construction.
The shadow result never reaches the API response, the decision or the velocity state.

An optional ``feature_observer`` receives the same point-in-time feature dict (for example
a :class:`~fraud_engine.feature_drift.FeatureDriftMonitor`). It is observe-only: it cannot
change the score, and an exception in it is logged and swallowed.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from . import __version__
from .artifacts import ScoringModel, load_model
from .config import Settings
from .features import compute_features
from .rules import DEFAULT_RULES, Rule, RuleContext, evaluate_rules
from .schemas import Decision, FeatureContribution, RiskResult, RuleHit, Transaction
from .state import VelocityBackend, build_velocity_store
from .thresholds import blend_risk

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShadowResult:
    """What the shadow model *would* have decided. Internal: only aggregated into metrics.

    ``risk_score`` and ``decision`` are ``None`` when the shadow model raised; the error is
    logged and swallowed so a broken candidate can never fail a live request.
    """

    model_version: str
    primary_decision: Decision
    risk_score: float | None
    decision: Decision | None
    latency_ms: float

    @property
    def failed(self) -> bool:
        return self.decision is None

    @property
    def agrees(self) -> bool:
        return self.decision == self.primary_decision


class RiskEngine:
    def __init__(
        self,
        settings: Settings | None = None,
        model: ScoringModel | None = None,
        store: VelocityBackend | None = None,
        rules: tuple[Rule, ...] = DEFAULT_RULES,
        shadow_model: ScoringModel | None = None,
        feature_observer: Callable[[Mapping[str, float]], None] | None = None,
    ) -> None:
        self.feature_observer = feature_observer
        self.settings = settings or Settings()
        self.store = store if store is not None else build_velocity_store(self.settings)
        self.rules = rules
        self.rule_ctx = RuleContext(high_amount_threshold=self.settings.high_amount_threshold)
        self.model = model
        if self.model is None and self.settings.model_path:
            self.model = load_model(self.settings.model_path)
            log.info("Loaded model %s from %s", self.model.version, self.settings.model_path)
        self.shadow_model = shadow_model
        if self.shadow_model is None and self.settings.shadow_model_path:
            # Fail fast on a bad artifact at startup (caught by readiness during a rollout)
            # rather than silently running without the comparison the operator asked for.
            self.shadow_model = load_model(self.settings.shadow_model_path)
            log.info(
                "Loaded shadow model %s from %s",
                self.shadow_model.version,
                self.settings.shadow_model_path,
            )

    @property
    def mode(self) -> str:
        return "rules+model" if self.model else "rules-only"

    @property
    def model_version(self) -> str:
        base = f"rules-v{__version__}"
        return f"{base}+{self.model.kind}-{self.model.version}" if self.model else base

    @property
    def shadow_model_version(self) -> str | None:
        if self.shadow_model is None:
            return None
        return f"rules-v{__version__}+{self.shadow_model.kind}-{self.shadow_model.version}"

    def decide(self, score: float) -> Decision:
        if score >= self.settings.block_threshold:
            return Decision.BLOCK
        if score >= self.settings.review_threshold:
            return Decision.REVIEW
        return Decision.APPROVE

    def score(self, txn: Transaction | Mapping[str, Any], update_state: bool = True) -> RiskResult:
        return self.score_with_shadow(txn, update_state=update_state)[0]

    def score_with_shadow(
        self, txn: Transaction | Mapping[str, Any], update_state: bool = True
    ) -> tuple[RiskResult, ShadowResult | None]:
        """Score a transaction; also return the shadow model's verdict when one is loaded."""
        start = time.perf_counter()
        data: Mapping[str, Any] = txn.model_dump() if isinstance(txn, Transaction) else txn

        # 1. Point-in-time state BEFORE this event (no self-leakage).
        velocity = self.store.snapshot(data["nameOrig"], data["nameDest"], int(data["step"]))
        features = compute_features(data, velocity)
        if self.feature_observer is not None:
            self._observe_features(features)

        # 2. Rules.
        rules_score, hits = evaluate_rules(data, velocity, self.rule_ctx, self.rules)

        # 3. Optional model + blend.
        model_score: float | None = None
        top: list[FeatureContribution] = []
        if self.model is not None:
            model_score, contribs = self.model.explain(features)
            top = [FeatureContribution(feature=f, value=v, contribution=c) for f, v, c in contribs]
            risk = blend_risk(model_score, rules_score, self.settings.model_weight)
        else:
            risk = round(min(max(rules_score, 0.0), 1.0), 6)

        reasons = [f"{r.code}: {r.description}" for r in hits]
        reasons += [
            f"MODEL: {c.feature} raised risk (+{c.contribution:.2f} log-odds)" for c in top[:3]
        ]

        # 4. Record AFTER scoring so future events see this one.
        if update_state:
            self.store.record(
                data["nameOrig"], data["nameDest"], int(data["step"]), float(data["amount"])
            )

        result = RiskResult(
            transaction_id=data.get("transaction_id"),
            risk_score=risk,
            decision=self.decide(risk),
            rules_score=round(rules_score, 6),
            model_score=None if model_score is None else round(model_score, 6),
            reasons=reasons,
            rule_hits=[
                RuleHit(code=r.code, description=r.description, points=r.points) for r in hits
            ],
            top_features=top,
            model_version=self.model_version,
            latency_ms=round((time.perf_counter() - start) * 1000, 3),
        )
        shadow = None
        if self.shadow_model is not None:
            shadow = self._score_shadow(self.shadow_model, features, rules_score, result.decision)
        return result, shadow

    def _observe_features(self, features: Mapping[str, float]) -> None:
        assert self.feature_observer is not None
        try:
            self.feature_observer(features)
        except Exception:  # monitoring must never fail a live request
            log.exception("Feature observer failed; scoring unaffected")

    def _score_shadow(
        self,
        shadow_model: ScoringModel,
        features: Mapping[str, float],
        rules_score: float,
        primary_decision: Decision,
    ) -> ShadowResult:
        start = time.perf_counter()
        version = f"rules-v{__version__}+{shadow_model.kind}-{shadow_model.version}"
        try:
            p = shadow_model.predict_proba(features)
            if not math.isfinite(p):
                raise ValueError(f"shadow model returned non-finite probability {p!r}")
            risk: float | None = blend_risk(p, rules_score, self.settings.model_weight)
        except Exception:  # a shadow candidate must never fail a live request
            log.exception("Shadow model %s failed; primary result unaffected", version)
            risk = None
        return ShadowResult(
            model_version=version,
            primary_decision=primary_decision,
            risk_score=risk,
            decision=None if risk is None else self.decide(risk),
            latency_ms=round((time.perf_counter() - start) * 1000, 3),
        )
