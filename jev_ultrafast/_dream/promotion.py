"""_dream.promotion — extracted from jev_ultrafast.dream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .policy import ReplayMetrics
    from .replay import ReplayResult

__all__ = [
    'PromotionDecision',
    'PromotionGate',
]


@dataclass(frozen=True)
class PromotionDecision:
    approved: bool
    reason: str
    baseline: ReplayMetrics
    candidate: ReplayMetrics
    validation: ReplayMetrics | None = None
    holdout: ReplayMetrics | None = None


class PromotionGate:
    """Fail-closed replay promotion gate.

    A candidate must improve the training objective and avoid regressions in
    verified success, risk, coverage, and normalized objective on the
    validation split. The holdout check runs in a second phase only:
    ``DreamImprover`` calls ``assess`` once more — with holdout evidence — on
    the single already-selected candidate, so holdout can veto a promotion
    but never influence which candidate was selected. Replay approval remains
    provisional; a bound live-canary evidence bundle is still required before
    activation.
    """

    def __init__(
        self,
        *,
        min_score_gain_per_world: float = 0.0,
        min_coverage: float = 0.75,
        max_success_regression: float = 0.0,
        max_risk_regression: int = 0,
        max_objective_regression_per_world: float = 0.0,
    ):
        self.min_score_gain_per_world = min_score_gain_per_world
        self.min_coverage = min_coverage
        self.max_success_regression = max_success_regression
        self.max_risk_regression = max_risk_regression
        self.max_objective_regression_per_world = max_objective_regression_per_world

    def _split_failure(self, label: str, baseline: ReplayMetrics, candidate: ReplayMetrics, *, require_gain=False):
        if candidate.worlds == 0:
            return f"no {label} replay worlds"
        if candidate.coverage < self.min_coverage:
            return f"insufficient {label} coverage"
        if candidate.success_rate + self.max_success_regression < baseline.success_rate:
            return f"{label} success regression"
        if candidate.risk_events > baseline.risk_events + self.max_risk_regression:
            return f"{label} risk regression"
        delta = candidate.score_per_world - baseline.score_per_world
        if require_gain and delta <= self.min_score_gain_per_world:
            return f"candidate did not improve {label} replay objective"
        if not require_gain and delta < -self.max_objective_regression_per_world:
            return f"{label} objective regression"
        return None

    def assess(
        self,
        baseline: ReplayResult,
        candidate: ReplayResult,
        *,
        validation_baseline: ReplayResult | None = None,
        validation_candidate: ReplayResult | None = None,
        holdout_baseline: ReplayResult | None = None,
        holdout_candidate: ReplayResult | None = None,
    ) -> PromotionDecision:
        b, c = baseline.metrics, candidate.metrics
        reason = self._split_failure("training", b, c, require_gain=True)
        if reason:
            return PromotionDecision(False, reason, b, c)

        validation_metrics = holdout_metrics = None
        for label, vb, vc in (
            ("validation", validation_baseline, validation_candidate),
            ("holdout", holdout_baseline, holdout_candidate),
        ):
            if (vb is None) != (vc is None):
                return PromotionDecision(False, f"incomplete {label} evidence", b, c)
            if vb and vc:
                split_reason = self._split_failure(label, vb.metrics, vc.metrics)
                if split_reason:
                    return PromotionDecision(
                        False,
                        split_reason,
                        b,
                        c,
                        vc.metrics if label == "validation" else None,
                        vc.metrics if label == "holdout" else None,
                    )
                if label == "validation":
                    validation_metrics = vc.metrics
                else:
                    holdout_metrics = vc.metrics
        return PromotionDecision(
            True,
            "replay gates passed; bound live canary still required",
            b,
            c,
            validation_metrics,
            holdout_metrics,
        )
