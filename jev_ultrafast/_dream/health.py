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
    """Detect post-promotion drift from hash-verified active-policy traces.

    The monitors cover the same dimensions the promotion gate measured —
    success, failure rate, risk, latency, actions, tokens — plus the
    *abandoned-run* matrix (Phase 21): runs the active policy started and
    never finished, and the candidate-attributable crashes among them.
    Attrition is checked *before* task sufficiency: a policy that dies on
    every start yields zero tasks, and "insufficient data" must never be
    the verdict on a death spiral.
    """

    def __init__(
        self,
        *,
        min_tasks: int = 20,
        max_success_regression: float = 0.10,
        max_extra_risk_rate: float = 0.0,
        max_failure_regression: float = 0.15,
        max_latency_regression_ratio: float = 0.5,
        max_action_regression_ratio: float = 0.5,
        max_token_regression_ratio: float = 0.5,
        min_abandoned_runs: int = 5,
        max_abandoned_fraction: float = 0.5,
        min_crash_runs: int = 3,
        max_crash_fraction: float = 0.2,
    ):
        self.min_tasks = min_tasks
        self.max_success_regression = max_success_regression
        self.max_extra_risk_rate = max_extra_risk_rate
        self.max_failure_regression = max_failure_regression
        self.max_latency_regression_ratio = max_latency_regression_ratio
        self.max_action_regression_ratio = max_action_regression_ratio
        self.max_token_regression_ratio = max_token_regression_ratio
        # Both a count floor and a fraction must trip: a handful of flaky
        # aborts is noise, a sustained abandoned majority is a policy
        # failing to survive its own runs.
        self.min_abandoned_runs = min_abandoned_runs
        self.max_abandoned_fraction = max_abandoned_fraction
        self.min_crash_runs = min_crash_runs
        self.max_crash_fraction = max_crash_fraction

    def assess(self, reference: CanaryMetrics, observed: CanaryMetrics) -> HealthDecision:
        # Attrition first: a crash loop produces few or no measured tasks,
        # so these checks must run before the sufficiency floor.
        if (
            observed.crash_runs >= self.min_crash_runs
            and observed.crash_fraction >= self.max_crash_fraction
        ):
            return HealthDecision(
                False,
                "active policy crash signature "
                f"({observed.crash_runs} attributable aborts in window)",
                reference,
                observed,
            )
        if (
            observed.abandoned_runs >= self.min_abandoned_runs
            and observed.abandoned_fraction >= self.max_abandoned_fraction
        ):
            return HealthDecision(
                False,
                "active policy abandoned-run drift "
                f"({observed.abandoned_runs} of "
                f"{observed.tasks + observed.abandoned_runs} started runs lost)",
                reference,
                observed,
            )
        if observed.tasks < self.min_tasks:
            return HealthDecision(
                True, "insufficient recent tasks for drift decision", reference, observed, sufficient=False
            )
        if observed.verified_successes == 0:
            # Every measured task failed — a collapse signature, distinct
            # from a rate regression.
            return HealthDecision(False, "active policy has no verified successes", reference, observed)
        if reference.tasks and observed.success_rate + self.max_success_regression < reference.success_rate:
            return HealthDecision(False, "active policy verified-success drift", reference, observed)
        if reference.tasks and observed.failure_rate > reference.failure_rate + self.max_failure_regression:
            return HealthDecision(False, "active policy failure-rate drift", reference, observed)
        if observed.risk_rate > reference.risk_rate + self.max_extra_risk_rate:
            return HealthDecision(False, "active policy risk-rate drift", reference, observed)
        if (
            reference.avg_latency_ms > 0
            and observed.avg_latency_ms > reference.avg_latency_ms * (1 + self.max_latency_regression_ratio)
        ):
            return HealthDecision(False, "active policy latency drift", reference, observed)
        if (
            reference.avg_actions > 0
            and observed.avg_actions > reference.avg_actions * (1 + self.max_action_regression_ratio)
        ):
            return HealthDecision(False, "active policy action-count drift", reference, observed)
        if (
            reference.avg_tokens > 0
            and observed.avg_tokens > reference.avg_tokens * (1 + self.max_token_regression_ratio)
        ):
            return HealthDecision(False, "active policy token drift", reference, observed)
        return HealthDecision(True, "active policy health gates passed", reference, observed)
