"""The risk engine: features -> rules (+ optional model) -> score, decision, reasons."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

from . import __version__
from .config import Settings
from .features import compute_features
from .model import LogisticBaseline
from .rules import DEFAULT_RULES, Rule, RuleContext, evaluate_rules
from .schemas import Decision, FeatureContribution, RiskResult, RuleHit, Transaction
from .state import VelocityStore

log = logging.getLogger(__name__)


class RiskEngine:
    def __init__(
        self,
        settings: Settings | None = None,
        model: LogisticBaseline | None = None,
        store: VelocityStore | None = None,
        rules: tuple[Rule, ...] = DEFAULT_RULES,
    ) -> None:
        self.settings = settings or Settings()
        self.store = store or VelocityStore(self.settings.velocity_window_steps)
        self.rules = rules
        self.rule_ctx = RuleContext(high_amount_threshold=self.settings.high_amount_threshold)
        self.model = model
        if self.model is None and self.settings.model_path:
            self.model = LogisticBaseline.load(self.settings.model_path)
            log.info("Loaded model %s from %s", self.model.version, self.settings.model_path)

    @property
    def mode(self) -> str:
        return "rules+model" if self.model else "rules-only"

    @property
    def model_version(self) -> str:
        base = f"rules-v{__version__}"
        return f"{base}+lr-{self.model.version}" if self.model else base

    def decide(self, score: float) -> Decision:
        if score >= self.settings.block_threshold:
            return Decision.BLOCK
        if score >= self.settings.review_threshold:
            return Decision.REVIEW
        return Decision.APPROVE

    def score(self, txn: Transaction | Mapping[str, Any], update_state: bool = True) -> RiskResult:
        start = time.perf_counter()
        data: Mapping[str, Any] = txn.model_dump() if isinstance(txn, Transaction) else txn

        # 1. Point-in-time state BEFORE this event (no self-leakage).
        velocity = self.store.snapshot(data["nameOrig"], data["nameDest"], int(data["step"]))
        features = compute_features(data, velocity)

        # 2. Rules.
        rules_score, hits = evaluate_rules(data, velocity, self.rule_ctx, self.rules)

        # 3. Optional model + blend.
        model_score: float | None = None
        top: list[FeatureContribution] = []
        if self.model is not None:
            model_score, contribs = self.model.explain(features)
            top = [FeatureContribution(feature=f, value=v, contribution=c) for f, v, c in contribs]
            w = self.settings.model_weight
            risk = w * model_score + (1 - w) * rules_score
        else:
            risk = rules_score
        risk = round(min(max(risk, 0.0), 1.0), 6)

        reasons = [f"{r.code}: {r.description}" for r in hits]
        reasons += [
            f"MODEL: {c.feature} raised risk (+{c.contribution:.2f} log-odds)" for c in top[:3]
        ]

        # 4. Record AFTER scoring so future events see this one.
        if update_state:
            self.store.record(
                data["nameOrig"], data["nameDest"], int(data["step"]), float(data["amount"])
            )

        return RiskResult(
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
