"""_dream.policy — extracted from jev_ultrafast.dream."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace

from ..privacy import action_goal_overlap
from .common import _stable_hash

__all__ = [
    'ExplorationPolicy',
    'ObjectiveWeights',
    'ReplayMetrics',
    'mutate_policies',
]


@dataclass(frozen=True)
class ExplorationPolicy:
    """Bounded policy surface that DREAM-Jev is allowed to optimize.

    These knobs influence candidate allocation and search patience only. They do
    not bypass action validation, approval requirements, completion verification,
    or browser execution guards.
    """

    name: str = "baseline"
    version: int = 1
    goal_overlap_weight: float = 10.0
    overlap_exponent: float = 1.0
    order_penalty: float = 0.00001
    click_bonus: float = 0.5
    fill_bonus: float = 1.5
    select_bonus: float = 1.0
    model_action_limit: int = 250
    duplicate_node_cap: int = 250
    min_goal_overlap: int = 0
    click_quota: int = 150
    fill_quota: int = 45
    select_quota: int = 55
    max_actions: int = 60
    no_progress_window: int = 3

    def __post_init__(self):
        if not self.name or self.version < 1:
            raise ValueError("Exploration policy needs a name and positive version")
        for value in (
            self.goal_overlap_weight,
            self.overlap_exponent,
            self.order_penalty,
            self.click_bonus,
            self.fill_bonus,
            self.select_bonus,
        ):
            if not math.isfinite(value):
                raise ValueError("Exploration policy weights must be finite")
        if not 0.25 <= self.overlap_exponent <= 2.0:
            raise ValueError("overlap_exponent must be between 0.25 and 2.0")
        if not 16 <= self.model_action_limit <= 250:
            raise ValueError("model_action_limit must be between 16 and 250")
        if not 1 <= self.duplicate_node_cap <= 250:
            raise ValueError("duplicate_node_cap must be between 1 and 250")
        if not 0 <= self.min_goal_overlap <= 5:
            raise ValueError("min_goal_overlap must be between 0 and 5")
        for quota in (self.click_quota, self.fill_quota, self.select_quota):
            if not 1 <= quota <= 250:
                raise ValueError("candidate quotas must be between 1 and 250")
        if not 1 <= self.max_actions <= 120:
            raise ValueError("max_actions must be between 1 and 120")
        if not 2 <= self.no_progress_window <= 10:
            raise ValueError("no_progress_window must be between 2 and 10")

    def candidate_score(self, action: dict, goal_tokens: set[str], order: int, overlap=None) -> float:
        if overlap is None:
            overlap = action_goal_overlap(action, goal_tokens)
        bonus = {
            "fill": self.fill_bonus,
            "select": self.select_bonus,
            "click": self.click_bonus,
        }.get(action.get("kind"), 0.0)
        return (overlap ** self.overlap_exponent) * self.goal_overlap_weight + bonus - order * self.order_penalty

    def quota_for(self, kind: str) -> int:
        return {
            "click": self.click_quota,
            "fill": self.fill_quota,
            "select": self.select_quota,
        }.get(kind, self.model_action_limit)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "ExplorationPolicy":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown exploration policy keys: {sorted(unknown)}")
        return cls(**payload)

    _BEHAVIOR_FIELDS = (
        "goal_overlap_weight",
        "overlap_exponent",
        "order_penalty",
        "click_bonus",
        "fill_bonus",
        "select_bonus",
        "model_action_limit",
        "duplicate_node_cap",
        "min_goal_overlap",
        "click_quota",
        "fill_quota",
        "select_quota",
        "max_actions",
        "no_progress_window",
    )

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return _stable_hash(encoded)

    @property
    def behavior_digest(self) -> str:
        """Identity-free digest of only the parameters that change runtime behavior.

        ``digest`` binds name/version lineage; ``behavior_digest`` deduplicates
        candidates that differ in identity but not in what the agent would do.
        """
        material = {field: getattr(self, field) for field in self._BEHAVIOR_FIELDS}
        return _stable_hash(json.dumps(material, sort_keys=True, separators=(",", ":")))


@dataclass(frozen=True)
class ObjectiveWeights:
    success: float = 100.0
    verified: float = 20.0
    page_progress: float = 0.5
    latency_seconds: float = 0.25
    action: float = 0.15
    model_call: float = 0.10
    thousand_tokens: float = 0.02
    offered_candidate: float = 0.002
    stale_or_failure: float = 2.0
    risk_event: float = 5.0
    coverage_miss: float = 4.0


@dataclass(frozen=True)
class ReplayMetrics:
    score: float
    worlds: int
    successes: int
    verified_successes: int
    actions: int
    model_calls: int
    tokens: int
    latency_ms: int
    offered_candidates: int
    page_progress: int
    stale_or_failures: int
    risk_events: int
    coverage_misses: int
    coverage: float
    estimated_tokens: int = 0
    estimated_latency_ms: int = 0
    estimated_score: float | None = None

    @property
    def success_rate(self) -> float:
        return self.successes / self.worlds if self.worlds else 0.0

    @property
    def score_per_world(self) -> float:
        return self.score / self.worlds if self.worlds else 0.0


def mutate_policies(base: ExplorationPolicy) -> list[ExplorationPolicy]:
    """Deterministic, bidirectional, bounded candidate generator for offline dreaming.

    Every knob is perturbed in both directions within the ``ExplorationPolicy``
    envelopes, so the search can expand as well as contract the candidate space.
    The generator changes only whitelisted exploration knobs. It never emits
    code, and candidates identical in behavior to the base are dropped by
    ``behavior_digest`` rather than kept as distinct digests.
    """
    specs = [
        {"goal_overlap_weight": max(0.1, base.goal_overlap_weight * factor)}
        for factor in (0.25, 0.5, 0.75, 1.25, 1.5, 2.0)
    ]
    specs += [
        {field: max(-2.0, min(10.0, getattr(base, field) + delta))}
        for field in ("click_bonus", "fill_bonus", "select_bonus")
        for delta in (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0)
    ]
    specs += [
        {field: max(1, min(250, getattr(base, field) + delta))}
        for field in ("click_quota", "fill_quota", "select_quota")
        for delta in (-25, -10, 10, 25)
    ]
    specs += [
        {"model_action_limit": max(16, min(250, base.model_action_limit + delta))}
        for delta in (-32, -16, 16, 32)
    ]
    specs += [
        {"max_actions": max(4, min(120, base.max_actions + delta))}
        for delta in (-16, -8, 8, 16)
    ]
    specs += [
        {"no_progress_window": max(2, min(10, base.no_progress_window + delta))}
        for delta in (-1, 1)
    ]
    specs += [
        {"order_penalty": max(0.0, base.order_penalty * factor)}
        for factor in (0.2, 5.0)
    ]
    specs += [
        {"overlap_exponent": max(0.25, min(2.0, base.overlap_exponent * factor))}
        for factor in (0.7, 1.4)
    ]
    specs += [
        {"duplicate_node_cap": max(1, min(250, base.duplicate_node_cap + delta))}
        for delta in (-100, -50, -25, 25, 100)
    ]
    specs += [
        {"min_goal_overlap": max(0, min(5, base.min_goal_overlap + delta))}
        for delta in (-1, 1)
    ]
    candidates = []
    seen = {base.behavior_digest}
    for index, changes in enumerate(specs, 1):
        candidate = replace(base, name=f"{base.name}-dream-{index}", version=base.version + 1, **changes)
        if candidate.behavior_digest in seen:
            continue
        seen.add(candidate.behavior_digest)
        candidates.append(candidate)
    return candidates
