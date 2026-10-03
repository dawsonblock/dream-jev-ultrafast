"""_dream.health — extracted from jev_ultrafast.dream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .canary import CanaryMetrics

__all__ = [
    'HealthDecision',
    'HealthGate',
]


@dataclass(frozen=True)
class HealthDecision:
    healthy: bool
    reason: str
    reference: CanaryMetrics
    observed: CanaryMetrics
    sufficient: bool = True


class HealthGate:
    """Detect post-promotion drift from hash-verified active-policy traces."""

    def __init__(self, *, min_tasks: int = 20, max_success_regression: float = 0.10, max_extra_risk_rate: float = 0.0):
        self.min_tasks = min_tasks
        self.max_success_regression = max_success_regression
        self.max_extra_risk_rate = max_extra_risk_rate

    def assess(self, reference: CanaryMetrics, observed: CanaryMetrics) -> HealthDecision:
        if observed.tasks < self.min_tasks:
            return HealthDecision(
                True, "insufficient recent tasks for drift decision", reference, observed, sufficient=False
            )
        if reference.tasks and observed.success_rate + self.max_success_regression < reference.success_rate:
            return HealthDecision(False, "active policy verified-success drift", reference, observed)
        if observed.risk_rate > reference.risk_rate + self.max_extra_risk_rate:
            return HealthDecision(False, "active policy risk-rate drift", reference, observed)
        return HealthDecision(True, "active policy health gates passed", reference, observed)
