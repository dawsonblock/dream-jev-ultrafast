"""_learning.scheduler — extracted from jev_ultrafast.dreamlearn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .causal import CounterfactualTrials

if TYPE_CHECKING:
    from typing import Iterable

__all__ = [
    'ExperimentScheduler',
]


@dataclass(frozen=True)
class ExperimentScheduler:
    """Which unresolved hypothesis is worth paying a real browser experiment for.

    A hypothesis whose randomized evidence is already ``beneficial`` or
    ``harmful`` is *settled* — no information left to buy — and is excluded.
    For the rest, the score combines:

    - **practical importance** — the utility-weighted expected delta (or the
      prior's predicted delta when no estimate exists yet), floored at
      ``min_importance`` so statistically-positive-but-irrelevant differences
      do not consume the trial budget;
    - **remaining uncertainty** — the width of the α-spent approximate
      interval, discounted as support accumulates (``width / (1 + support)``);
    - **evidence coverage** — a mild penalty for families already sampled
      heavily, so the budget spreads across contexts instead of over-fitting
      one family.

    The scheduler ranks hypotheses; it never executes anything and never
    fabricates a browser outcome. Selection order is deterministic
    (score, then expected delta, then a stable key) so replays are
    reproducible.
    """

    budget: int = 24
    min_importance: float = 0.02
    coverage_scale: float = 16.0

    def rank(
        self,
        hypotheses: Iterable[dict],
        *,
        estimates: CounterfactualTrials | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        hypotheses = list(hypotheses)
        family_counts: dict[str, int] = {}
        for hypothesis in hypotheses:
            family = self._family(hypothesis)
            family_counts[family] = family_counts.get(family, 0) + 1
        ranked = []
        for hypothesis in hypotheses:
            estimate = hypothesis.get("estimate")
            if estimate is None and estimates is not None:
                historical = hypothesis.get("historical") or {}
                proposal = hypothesis.get("proposal") or {}
                estimate = estimates.resolve(
                    task_family=self._family(hypothesis) or None,
                    site=hypothesis.get("site"),
                    model_kind=str(
                        historical.get("kind")
                        or hypothesis.get("model_choice_kind")
                        or hypothesis.get("model_kind")
                        or "unknown"
                    ),
                    model_overlap=historical.get(
                        "goal_overlap",
                        hypothesis.get("model_choice_overlap", hypothesis.get("model_overlap")),
                    ),
                    model_effect=historical.get(
                        "effect",
                        hypothesis.get("model_choice_effect", hypothesis.get("model_effect")),
                    ),
                    model_role=historical.get("role", hypothesis.get("model_choice_role")),
                    model_rank=historical.get("offered_rank"),
                    proposal_kind=str(
                        proposal.get("kind")
                        or hypothesis.get("proposal_kind")
                        or "unknown"
                    ),
                    proposal_overlap=proposal.get(
                        "goal_overlap", hypothesis.get("proposal_overlap")
                    ),
                    proposal_effect=proposal.get("effect", hypothesis.get("proposal_effect")),
                    proposal_role=proposal.get("role", hypothesis.get("proposal_role")),
                    proposal_rank=hypothesis.get("proposal_offered_rank"),
                    phase=hypothesis.get("step"),
                )
            status = (estimate or {}).get("effect_status")
            if status in {"beneficial", "harmful"}:
                continue
            if estimate is not None and "candidate" in estimate and "control" in estimate:
                sequential = estimate.get("delta_cs")
                width = (sequential[1] - sequential[0]) if sequential else 1.0
                support = min(
                    estimate["candidate"]["ess"], estimate["control"]["ess"]
                ) / max(1e-9, CounterfactualTrials.MIN_ESS)
                support_fraction = min(1.0, max(0.0, support))
                importance_basis = estimate.get("utility_delta")
                if importance_basis is None:
                    importance_basis = estimate.get("delta")
            else:
                # No contrast (or no evidence at all): maximum uncertainty,
                # importance from the prior's own expected delta.
                width = 1.0
                support_fraction = 0.0
                importance_basis = None
            expected_delta = hypothesis.get("expected_delta")
            if importance_basis is None:
                importance_basis = expected_delta
            importance = max(abs(float(importance_basis or 0.0)), self.min_importance)
            family = self._family(hypothesis)
            coverage = family_counts.get(family, 0)
            score = (
                importance
                * (width / (1.0 + support_fraction))
                / (1.0 + coverage / max(1e-9, self.coverage_scale))
            )
            ranked.append({
                **hypothesis,
                "estimate": estimate,
                "scheduler": {
                    "score": score,
                    "importance": importance,
                    "uncertainty_width": width,
                    "support_fraction": support_fraction,
                    "effect_status": status,
                    "family_coverage": coverage,
                },
            })
        ranked.sort(
            key=lambda item: (
                -item["scheduler"]["score"],
                -(float(item.get("expected_delta") or 0.0)),
                str(item.get("world", "")),
                int(item.get("step", 0) or 0),
            )
        )
        return ranked[: (self.budget if limit is None else max(0, int(limit)))]

    @staticmethod
    def _family(hypothesis: dict) -> str:
        return str(
            hypothesis.get("family_key") or hypothesis.get("task_family") or ""
        ).strip().lower()
