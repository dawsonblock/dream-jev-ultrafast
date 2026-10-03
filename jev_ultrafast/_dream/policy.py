"""_dream.policy — extracted from jev_ultrafast.dream."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import ClassVar

from ..privacy import action_goal_overlap
from .common import _stable_hash
from .policylang import (
    BOOL_FEATURES,
    NUMERIC_FEATURES,
    normalize_rules,
    program_adjustment,
    program_stop,
)

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
    # The verified policy program (Phase 20): the structural surface.
    # ``rules`` is a tuple of ``{"when": <bool expr>, "add": <num expr>}``
    # ASTs over the fixed ``policylang`` vocabulary, and ``stop_when`` an
    # optional boolean expression over run features. The knobs are a
    # parameter vector; the program is a mutable *policy computation* —
    # validated, serializable, digestible, replayable, and interpreted by
    # a fixed trusted interpreter (``_dream/policylang.py``). Empty is the
    # historical default: a program-free policy keeps its digest exactly.
    rules: tuple = ()
    stop_when: dict | None = None

    # The declarative policy DSL (Phase 18): the entire vocabulary a
    # candidate may write. Each field declares its JSON type and, where
    # bounded, its inclusive envelope — the same numbers this class has
    # always enforced, declared once so validation, serialization,
    # introspection, and mutation cannot drift apart. ``type`` is ``str``,
    # ``int``, or ``number`` (int-or-float); ``bool`` is never an integer
    # here — a flag must not silently satisfy a numeric envelope.
    # ``program-rules``/``program-expr`` fields carry verified policylang
    # ASTs, validated by ``policylang.normalize_rules`` at construction.
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
        "rules": {"type": "program-rules"},
        "stop_when": {"type": "program-expr"},
    }

    @classmethod
    def schema(cls) -> dict:
        """The declared write surface: field → type/envelope spec.

        This *is* the DSL — the complete vocabulary recursive improvement
        may write. Anything outside it fails closed at ``from_dict``."""
        return {k: dict(v) for k, v in cls._FIELD_SPECS.items()}

    def __post_init__(self):
        # The program validates first: construction is the verification
        # boundary — a policy object that exists at all carries a proven
        # AST. Normalized (canonical) forms are stored back so digest and
        # serialization never depend on the writer's key order.
        rules, stop = normalize_rules(
            getattr(self, "rules") or (), getattr(self, "stop_when")
        )
        object.__setattr__(self, "rules", rules)
        object.__setattr__(self, "stop_when", stop)
        for name, spec in self._FIELD_SPECS.items():
            kind = spec["type"]
            if kind.startswith("program"):
                continue
            value = getattr(self, name)
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

    def candidate_score(self, action: dict, goal_tokens: set[str], order: int, overlap=None, context=None) -> float:
        if overlap is None:
            overlap = action_goal_overlap(action, goal_tokens)
        bonus = {
            "fill": self.fill_bonus,
            "select": self.select_bonus,
            "click": self.click_bonus,
        }.get(action.get("kind"), 0.0)
        base = (overlap ** self.overlap_exponent) * self.goal_overlap_weight + bonus - order * self.order_penalty
        if not self.rules:
            return base
        return base + program_adjustment(
            self.rules, self._score_features(action, order, overlap, context)
        )

    @staticmethod
    def _score_features(action: dict, order: int, overlap, context) -> dict:
        """The candidate/run feature record the score rules evaluate over.

        Run features (``no_progress``, ``steps_taken``) describe the
        trajectory *before* this decision — uniform across candidates at a
        step, and zero when the caller supplies no context.
        """
        kind = action.get("kind")
        ctx = context or {}
        return {
            "overlap": float(overlap),
            "offered_rank": float(order),
            "is_click": 1.0 if kind == "click" else 0.0,
            "is_fill": 1.0 if kind == "fill" else 0.0,
            "is_select": 1.0 if kind == "select" else 0.0,
            "is_upload": 1.0 if kind == "upload" else 0.0,
            "is_control": 0.0
            if kind in {"click", "fill", "select", "upload"}
            else 1.0,
            "has_node": 1.0 if action.get("node") is not None else 0.0,
            "no_progress": float(ctx.get("no_progress", 0.0) or 0.0),
            "steps_taken": float(ctx.get("steps_taken", 0.0) or 0.0),
        }

    def should_stop(self, features: dict) -> bool:
        """The program's extra stopping authority — additive only.

        The base ``no_progress_window`` check applies independently and can
        never be weakened: ``stop_when`` may block a run earlier, never
        later.
        """
        return program_stop(self.stop_when, features or {})

    def quota_for(self, kind: str) -> int:
        return {
            "click": self.click_quota,
            "fill": self.fill_quota,
            "select": self.select_quota,
        }.get(kind, self.model_action_limit)

    def to_dict(self) -> dict:
        data = asdict(self)
        # Program fields serialize only when non-default — a program-free
        # policy keeps its historical digest byte-for-byte, so every stored
        # run_started record written before the AST existed still verifies.
        if not self.rules:
            data.pop("rules", None)
        if self.stop_when is None:
            data.pop("stop_when", None)
        return data

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
    # so the two can never drift. Program fields join the digest material
    # conditionally (see ``behavior_digest``) so a program-free policy
    # keeps its historical behavior digest.
    _BEHAVIOR_FIELDS: ClassVar[tuple] = tuple(
        k for k in _FIELD_SPECS if k not in ("name", "version", "rules", "stop_when")
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
        if self.rules:
            material["rules"] = [dict(rule) for rule in self.rules]
        if self.stop_when is not None:
            material["stop_when"] = self.stop_when
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


def _mutate_first_const(node, factor: float):
    """Scale the first ``const`` leaf inside a policylang AST by ``factor``.

    Depth-first over the fixed node vocabulary; returns the original
    object when no constant exists so callers can detect a no-op. The
    result is *rewritten*, not revalidated — ``ExplorationPolicy``
    construction validates it like every other candidate.
    """
    (op, arg), = node.items()
    if op == "const":
        return {"const": float(arg) * factor}
    children = []
    if op in {"cmp", "clamp", "if"}:
        children = [c for c in arg if isinstance(c, dict)]
    elif op == "pow":
        children = [arg[0]]
    elif op in {"neg", "abs", "not"}:
        children = [arg]
    elif isinstance(arg, list):
        children = [c for c in arg if isinstance(c, dict)]
    for child in children:
        new = _mutate_first_const(child, factor)
        if new is not child:
            if op in {"neg", "abs", "not"}:
                return {op: new}
            out = [
                new if c is child else c
                for c in arg
            ]
            return {op: out}
    return node


# ---------------------------------------------------------------------------
# Grammar-aware program search.
#
# The operators below are the structural-search layer the language was built
# for: one-position rewrites of the verified AST — operator swaps inside a
# signature family, relation and operand moves on comparisons, feature and
# constant leaves exchanged within their type, unary wrappers added or
# stripped, composite nodes pruned to a child, and growth wraps that add a
# grammar context around an existing subtree. Nothing here inserts a seed
# template: a mutant is the base program with one node position rewritten,
# and the constructor's validation boundary drops any rewrite that would
# break depth, budget, or type rules — the search can never produce a
# program the verifier would not also accept.
# ---------------------------------------------------------------------------

_ARITH_2_OPS = ("add", "sub", "mul", "min", "max")
_BOOL_2_OPS = ("and", "or")
_UNARY_NUM_OPS = ("neg", "abs")
_CMP_RELS = ("<", "<=", "==", ">=", ">")
# Search breadth, not exhaustiveness: one generation emits at most this
# many grammar-level mutants, interleaved across every expression position
# so no single rule consumes the whole budget.
_PROGRAM_MUTANT_CAP = 48
_POW_EXPONENT_RANGE = (0.25, 2.0)


def _expr_positions(node, path=()):
    """Yield ``(path, node)`` for every dict subtree, pre-order.

    ``path`` is the child-index route ``_expr_replace`` follows. The
    ``pow`` exponent position is excluded — it must stay a const leaf, so
    it is perturbed by the ``pow`` node's own mutants rather than opened
    to subtree replacement.
    """
    yield path, node
    (op, arg), = node.items()
    if op in {"neg", "abs", "not"}:
        children = ((0, arg),) if isinstance(arg, dict) else ()
    elif op == "pow":
        children = ((0, arg[0]),) if isinstance(arg[0], dict) else ()
    elif isinstance(arg, list):
        children = tuple(
            (index, child)
            for index, child in enumerate(arg)
            if isinstance(child, dict)
        )
    else:
        children = ()
    for index, child in children:
        yield from _expr_positions(child, path + (index,))


def _expr_replace(node, path, new_sub):
    """A copy of ``node`` with the subtree at ``path`` swapped for ``new_sub``."""
    if not path:
        return new_sub
    (op, arg), = node.items()
    index = path[0]
    if op in {"neg", "abs", "not"}:
        return {op: _expr_replace(arg, path[1:], new_sub)}
    out = [
        _expr_replace(child, path[1:], new_sub)
        if isinstance(child, dict) and j == index
        else child
        for j, child in enumerate(arg)
    ]
    return {op: out}


def _node_type(node) -> str:
    (op, arg), = node.items()
    if op == "feature":
        return "num" if arg in NUMERIC_FEATURES else "bool"
    if op == "bool" or op in _BOOL_2_OPS or op in {"cmp", "not"}:
        return "bool"
    return "num"


def _node_mutants(node) -> list:
    """Every same-typed one-node rewrite — the grammar-aware operator set."""
    (op, arg), = node.items()
    out = []
    if op == "const":
        for factor in (0.5, 1.5, -1.0):
            value = float(arg) * factor
            if math.isfinite(value) and abs(value) <= 1.0e6:
                out.append({"const": value})
    elif op == "feature":
        pool = NUMERIC_FEATURES if arg in NUMERIC_FEATURES else BOOL_FEATURES
        out += [{"feature": name} for name in pool if name != arg]
    elif op == "bool":
        out.append({"bool": not arg})
    elif op in _ARITH_2_OPS or op in _BOOL_2_OPS:
        family = _ARITH_2_OPS if op in _ARITH_2_OPS else _BOOL_2_OPS
        out += [
            {other: [arg[0], arg[1]]} for other in family if other != op
        ]
        out += list(arg)  # prune to an operand (same static type)
    elif op in _UNARY_NUM_OPS:
        other = "abs" if op == "neg" else "neg"
        out += [{other: arg}, arg]  # swap the wrapper, or drop it
    elif op == "not":
        out.append(arg)  # strip the negation
    elif op == "cmp":
        out += [
            {"cmp": [rel, arg[1], arg[2]]}
            for rel in _CMP_RELS
            if rel != arg[0]
        ]
        out.append({"cmp": [arg[0], arg[2], arg[1]]})  # operand order
    elif op == "if":
        out.append({"if": [arg[0], arg[2], arg[1]]})  # branch swap
        out += [arg[1], arg[2]]  # collapse to a branch
    elif op == "clamp":
        out.append(arg[0])  # drop the bounds
    elif op == "pow":
        out.append(arg[0])  # drop the exponent
        for factor in (0.5, 1.5):
            exponent = float(arg[1].get("const", 1.0)) * factor
            if _POW_EXPONENT_RANGE[0] <= exponent <= _POW_EXPONENT_RANGE[1]:
                out.append({"pow": [arg[0], {"const": exponent}]})
    # Growth wraps: a grammar context placed around the node itself.
    if _node_type(node) == "num":
        out += [
            {"abs": node},
            {"neg": node},
            {"mul": [node, {"const": 2.0}]},
            {"add": [node, {"const": -1.0}]},
            {"min": [node, {"const": 0.0}]},
        ]
    else:
        out.append({"not": node})
    return out


def _expr_mutants(expr):
    """Yield every one-position rewrite of an expression, pre-order."""
    seen = set()
    for path, node in _expr_positions(expr):
        for new_sub in _node_mutants(node):
            if new_sub == node:
                continue
            candidate = _expr_replace(expr, path, new_sub)
            key = json.dumps(candidate, sort_keys=True)
            if key not in seen:
                seen.add(key)
                yield candidate


def _program_mutants(base: ExplorationPolicy, cap: int = _PROGRAM_MUTANT_CAP) -> list[dict]:
    """Grammar-aware mutants of the policy program as constructor specs.

    Mutants from every rule's ``when`` and ``add`` are interleaved
    round-robin so the first ``cap`` candidates cover positions across the
    whole program rather than exhausting the first rule's moves.
    ``stop_when`` mutants follow the same rules.
    """
    generators = []
    for index, rule in enumerate(base.rules):
        generators.append((index, "when", iter(_expr_mutants(rule["when"]))))
        generators.append((index, "add", iter(_expr_mutants(rule["add"]))))
    specs: list[dict] = []
    while len(specs) < cap and generators:
        for entry in list(generators):
            index, field_name, gen = entry
            try:
                mutant = next(gen)
            except StopIteration:
                generators.remove(entry)
                continue
            specs.append({
                "rules": tuple(
                    {**r, field_name: mutant} if j == index else r
                    for j, r in enumerate(base.rules)
                )
            })
            if len(specs) >= cap:
                break
    if base.stop_when is not None and len(specs) < cap:
        for mutant in _expr_mutants(base.stop_when):
            specs.append({"stop_when": mutant})
            if len(specs) >= cap:
                break
    return specs


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

    # ------------------------------------------------------------------
    # Program mutations (Phase 20): real structural moves on the verified
    # policy AST — introduce a rule, remove a rule, perturb a constant
    # *inside* an expression tree, change the stopping expression. These
    # change what the policy computes; no scalar perturbation of the knob
    # vector expresses "add a conditional rule" or "drop this feature
    # interaction". Every candidate is written through the declared DSL
    # surface and validated by the same constructor boundary.
    # ------------------------------------------------------------------
    _SEED_RULES = (
        # Gate: a fill with no goal overlap loses its kind prior.
        {"when": {"and": [{"feature": "is_fill"},
                          {"cmp": ["==", {"feature": "overlap"}, {"const": 0}]}]},
         "add": {"const": -1.0}},
        # Deep-catalogue actions are suspect.
        {"when": {"cmp": [">=", {"feature": "offered_rank"}, {"const": 10}]},
         "add": {"const": -1.0}},
        # Recovery bias: while progress has stalled, prefer clicks.
        {"when": {"and": [{"feature": "is_click"},
                          {"cmp": [">=", {"feature": "no_progress"}, {"const": 2}]}]},
         "add": {"const": 1.0}},
        # Interaction: fills earn extra credit per unit of goal overlap.
        {"when": {"feature": "is_fill"},
         "add": {"mul": [{"feature": "overlap"}, {"const": 0.5}]}},
    )
    _SEED_STOP = {"cmp": [">=", {"feature": "no_progress"}, {"const": 2}]}

    if not base.rules and base.stop_when is None:
        for rule in _SEED_RULES:
            specs.append({"rules": (rule,)})
        specs.append({"rules": _SEED_RULES[:2]})
        specs.append({"stop_when": _SEED_STOP})
        # Generated single-feature seeds — grammar-level probes, not
        # hand-picked templates: each boolean flag gates a small penalty
        # and each numeric feature thresholds at a small constant, so an
        # empty program gains material the subtree mutations can then
        # combine and reshape.
        for name in BOOL_FEATURES:
            specs.append({"rules": (
                {"when": {"feature": name}, "add": {"const": -0.5}},
            )})
        for name in NUMERIC_FEATURES:
            specs.append({"rules": (
                {"when": {"cmp": [">=", {"feature": name}, {"const": 2}]},
                 "add": {"const": -0.5}},
            )})
    else:
        for index in range(len(base.rules)):
            # Structural removal: drop rule ``index`` outright.
            specs.append({
                "rules": tuple(
                    rule for j, rule in enumerate(base.rules) if j != index
                )
            })
            # Structural edit: scale the first constant inside this rule's
            # adjustment expression — an AST node rewrite, not a knob move.
            for factor in (0.5, 1.5):
                specs.append({
                    "rules": tuple(
                        {**rule, "add": _mutate_first_const(rule["add"], factor)}
                        if j == index else rule
                        for j, rule in enumerate(base.rules)
                    )
                })
        # Growth: a new rule joins the existing program.
        specs.append({"rules": tuple(base.rules) + (_SEED_RULES[0],)})
        # Stopping-expression moves: introduce one, or remove the existing.
        specs.append({
            "stop_when": _SEED_STOP if base.stop_when is None else None
        })
        # Grammar-aware search: subtree/operator-level rewrites at every
        # expression position — the bounded language the verifier checks,
        # explored rather than templated.
        specs += _program_mutants(base)

    candidates = []
    seen = {base.behavior_digest}
    for index, changes in enumerate(specs, 1):
        try:
            candidate = replace(base, name=f"{base.name}-dream-{index}", version=base.version + 1, **changes)
        except ValueError:
            # A rewrite that would break depth, budget, or type rules dies
            # at the same validation boundary every candidate crosses.
            continue
        if candidate.behavior_digest in seen:
            continue
        seen.add(candidate.behavior_digest)
        candidates.append(candidate)
    return candidates
