"""_dream.policy — extracted from jev_ultrafast.dream."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import ClassVar

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

    # The declarative policy DSL (Phase 18): the entire vocabulary a
    # candidate may write. Each field declares its JSON type and, where
    # bounded, its inclusive envelope — the same numbers this class has
    # always enforced, declared once so validation, serialization,
    # introspection, and mutation cannot drift apart. ``type`` is ``str``,
    # ``int``, or ``number`` (int-or-float); ``bool`` is never an integer
    # here — a flag must not silently satisfy a numeric envelope.
    _FIELD_SPECS: ClassVar[dict] = {
        "name": {"type": "str", "nonempty": True},
        "version": {"type": "int", "min": 1},
        "goal_overlap_weight": {"type": "number"},
        "overlap_exponent": {"type": "number", "min": 0.25, "max": 2.0},
        "order_penalty": {"type": "number"},
        "click_bonus": {"type": "number"},
        "fill_bonus": {"type": "number"},
        "select_bonus": {"type": "number"},
        "model_action_limit": {"type": "int", "min": 16, "max": 250},
        "duplicate_node_cap": {"type": "int", "min": 1, "max": 250},
        "min_goal_overlap": {"type": "int", "min": 0, "max": 5},
        "click_quota": {"type": "int", "min": 1, "max": 250},
        "fill_quota": {"type": "int", "min": 1, "max": 250},
        "select_quota": {"type": "int", "min": 1, "max": 250},
        "max_actions": {"type": "int", "min": 1, "max": 120},
        "no_progress_window": {"type": "int", "min": 2, "max": 10},
    }

    @classmethod
    def schema(cls) -> dict:
        """The declared write surface: field → type/envelope spec.

        This *is* the DSL — the complete vocabulary recursive improvement
        may write. Anything outside it fails closed at ``from_dict``."""
        return {k: dict(v) for k, v in cls._FIELD_SPECS.items()}

    def __post_init__(self):
        for name, spec in self._FIELD_SPECS.items():
            value = getattr(self, name)
            kind = spec["type"]
            if kind == "str":
                if not isinstance(value, str):
                    raise ValueError(f"{name} must be a string")
                if spec.get("nonempty") and not value:
                    raise ValueError(
                        "Exploration policy needs a name and positive version"
                    )
            elif kind == "int":
                # bool is an int subclass — never silently accept a flag
                # where the DSL declares an integer envelope.
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError(f"{name} must be an integer")
            else:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError(f"{name} must be a number")
                if not math.isfinite(value):
                    raise ValueError("Exploration policy weights must be finite")
            lo, hi = spec.get("min"), spec.get("max")
            if lo is not None and hi is not None:
                if not lo <= value <= hi:
                    raise ValueError(f"{name} must be between {lo} and {hi}")
            elif lo is not None and value < lo:
                raise ValueError(f"{name} must be at least {lo}")
            elif hi is not None and value > hi:
                raise ValueError(f"{name} must be at most {hi}")

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
        if not isinstance(payload, dict):
            raise ValueError("Exploration policy payload must be a dict")
        allowed = set(cls._FIELD_SPECS)
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown exploration policy keys: {sorted(unknown)}")
        return cls(**payload)

    # Behavior fields are the DSL minus identity — derived from the spec
    # so the two can never drift.
    _BEHAVIOR_FIELDS: ClassVar[tuple] = tuple(
        k for k in _FIELD_SPECS if k not in ("name", "version")
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
    # Phase 13 verdict tally — how each world ended, counted once per world
    # under ``REPLAY_VERDICTS`` (see replay.py). A coverage miss is truncated
    # evidence, never a measured failure; the tally keeps those distinct.
    verdicts: dict = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.successes / self.worlds if self.worlds else 0.0

    @property
    def score_per_world(self) -> float:
        return self.score / self.worlds if self.worlds else 0.0


def mutate_policies(base: ExplorationPolicy) -> list[ExplorationPolicy]:
    """Deterministic, bidirectional, bounded candidate generator for offline dreaming.

    Two layers of moves: scalar perturbations walk each knob in both
    directions inside its declared envelope, and structural moves (Phase
    19) change the candidate's *shape* — envelope-boundary probes, joint
    moves of coupled knob groups (patience, kind budgets, selectivity),
    ablations resetting one knob to the class default, and knock-outs
    zeroing each kind prior. The generator writes only the declared DSL
    surface, never code, and candidates identical in behavior to the base
    are dropped by ``behavior_digest`` rather than kept as distinct
    digests.
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

    # ------------------------------------------------------------------
    # Structural mutations (Phase 19): candidates whose *shape* differs,
    # not just one coordinate's value. Scalar perturbations explore along
    # single axes; these probe envelope boundaries, move coupled knob
    # groups jointly, ablate one knob back to the class default, and
    # knock out each kind prior entirely. Every move stays inside the
    # declared DSL envelope, and construction still validates — a
    # structural candidate is exactly as bounded as a scalar one.
    # ------------------------------------------------------------------
    defaults = ExplorationPolicy()

    # Envelope-boundary probes: each bounded numeric field pushed to its
    # declared min and max alone — the corners of the write surface.
    # Identity fields (name, version) are never mutation targets.
    for field_name, spec in ExplorationPolicy._FIELD_SPECS.items():
        if field_name in ("name", "version"):
            continue
        if spec["type"] not in ("int", "number"):
            continue
        for bound in (spec.get("min"), spec.get("max")):
            if bound is not None:
                specs.append({field_name: bound})

    # Joint moves: coupled knob groups stepped together in both
    # directions. A group moved jointly is a different *policy shape* —
    # e.g. "patient" scales the whole patience envelope at once, which no
    # sequence of single-axis perturbations expresses in one candidate.
    _GROUPS = (
        # Patience: how long the search may run.
        ("model_action_limit", "duplicate_node_cap", "max_actions",
         "no_progress_window"),
        # Kind budgets.
        ("click_quota", "fill_quota", "select_quota"),
        # Selectivity: how strongly goal relevance gates the catalogue.
        ("min_goal_overlap", "goal_overlap_weight", "overlap_exponent"),
    )
    for group in _GROUPS:
        for direction in (-1, 1):
            changes = {}
            for field_name in group:
                spec = ExplorationPolicy._FIELD_SPECS[field_name]
                lo, hi = spec.get("min"), spec.get("max")
                current = getattr(base, field_name)
                if spec["type"] == "int":
                    # Integer envelopes step a quarter-span in the move's
                    # direction — a joint move must visibly move every
                    # member, never round a contraction up to +1.
                    span = (
                        hi - lo
                        if hi is not None and lo is not None
                        else 1
                    )
                    moved = current + direction * max(1, round(span * 0.25))
                else:
                    moved = current * (1.5 if direction > 0 else 0.75)
                if lo is not None:
                    moved = max(lo, moved)
                if hi is not None:
                    moved = min(hi, moved)
                changes[field_name] = moved
            specs.append(changes)

    # Ablations: each behavior field reset to the class default alone —
    # "what if this knob went back to baseline" — and knock-outs zeroing
    # each kind prior entirely (0 is inside every declared envelope).
    for field_name in ExplorationPolicy._BEHAVIOR_FIELDS:
        specs.append({field_name: getattr(defaults, field_name)})
    for field_name in ("click_bonus", "fill_bonus", "select_bonus"):
        specs.append({field_name: 0.0})

    candidates = []
    seen = {base.behavior_digest}
    for index, changes in enumerate(specs, 1):
        candidate = replace(base, name=f"{base.name}-dream-{index}", version=base.version + 1, **changes)
        if candidate.behavior_digest in seen:
            continue
        seen.add(candidate.behavior_digest)
        candidates.append(candidate)
    return candidates
