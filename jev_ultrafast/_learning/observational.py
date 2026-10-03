"""_learning.observational — extracted from jev_ultrafast.dreamlearn."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import TYPE_CHECKING

from jev_ultrafast._dream.policy import ExplorationPolicy
from jev_ultrafast._dream.replay import ReplaySimulator

from .censoring import _run_outcome
from .common import _stable_hash
from .signatures import overlap_bucket, rank_bucket

if TYPE_CHECKING:
    from typing import Iterable

__all__ = [
    'ChoiceModel',
    'CostModel',
    'OutcomeModel',
    '_fit_cells',
    '_fit_line',
    '_offered_count',
    '_selected_offered_rank',
    '_transitions',
]


def _offered_count(event: dict) -> int | None:
    offered = event.get("offered_count")
    if offered is not None:
        return max(0, int(offered))
    candidates = event.get("candidates")
    return len(candidates) if candidates is not None else None


def _fit_line(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least-squares slope/intercept with residual stddev. Slope is clamped
    non-negative: fewer offered candidates cannot cost more in this domain."""
    mx, my = fmean(xs), fmean(ys)
    var = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var if var else 0.0
    slope = max(0.0, slope)
    intercept = max(0.0, my - slope * mx)
    residual = math.sqrt(fmean([(y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)]))
    return intercept, slope, residual


@dataclass(frozen=True)
class CostModel:
    """Linear cost scaling in offered candidates, fit on recorded transitions.

    ``predict`` returns absolute estimates; callers anchor them to the recorded
    transition values (estimate deltas, not absolute reality).

    Reliability requires *identifiability*, not just volume: eight samples at
    one offered count carry zero information about the per-candidate slope, so
    the design must contain at least ``MIN_DISTINCT_OFFERED`` distinct offered
    counts spanning ``MIN_OFFERED_SPAN``.
    """

    samples: int = 0
    token_intercept: float = 0.0
    token_per_candidate: float = 0.0
    token_residual: float = 0.0
    latency_intercept_ms: float = 0.0
    latency_per_candidate_ms: float = 0.0
    latency_residual_ms: float = 0.0
    offered_min: int = 0
    offered_max: int = 0
    distinct_offered: int = 0
    version: str = "jev-cost/2"

    MIN_SAMPLES = 8
    MIN_DISTINCT_OFFERED = 2
    MIN_OFFERED_SPAN = 4

    @classmethod
    def fit(cls, events: Iterable[dict]) -> "CostModel":
        xs, tokens, latencies = [], [], []
        for event in events:
            if event.get("event") != "transition":
                continue
            offered = _offered_count(event)
            if offered is None:
                continue
            xs.append(offered)
            tokens.append(max(0, float(event.get("tokens", 0))))
            latencies.append(max(0, float(event.get("latency_ms", 0))))
        if not xs:
            return cls()
        ti, ts, tr = _fit_line(xs, tokens)
        li, ls, lr = _fit_line(xs, latencies)
        return cls(
            samples=len(xs),
            token_intercept=ti,
            token_per_candidate=ts,
            token_residual=tr,
            latency_intercept_ms=li,
            latency_per_candidate_ms=ls,
            latency_residual_ms=lr,
            offered_min=min(xs),
            offered_max=max(xs),
            distinct_offered=len(set(xs)),
        )

    @property
    def reliable(self) -> bool:
        return (
            self.samples >= self.MIN_SAMPLES
            and self.distinct_offered >= self.MIN_DISTINCT_OFFERED
            and self.offered_max - self.offered_min >= self.MIN_OFFERED_SPAN
        )

    def _extrapolation(self, offered: int) -> float:
        if not self.samples or self.offered_min <= offered <= self.offered_max:
            return 1.0
        span = max(1, self.offered_max - self.offered_min)
        distance = max(self.offered_min - offered, offered - self.offered_max)
        return 1.0 + distance / span

    def predict(self, offered_count: int) -> dict:
        x = max(0, int(offered_count))
        extrapolation = self._extrapolation(x)
        return {
            "tokens": max(0.0, self.token_intercept + self.token_per_candidate * x),
            "latency_ms": max(0.0, self.latency_intercept_ms + self.latency_per_candidate_ms * x),
            "token_uncertainty": self.token_residual * extrapolation,
            "latency_uncertainty_ms": self.latency_residual_ms * extrapolation,
            "extrapolated": extrapolation > 1.0,
        }

    @property
    def digest(self) -> str:
        return _stable_hash(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "CostModel":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown cost model keys: {sorted(unknown)}")
        return cls(**payload)


def _transitions(events: Iterable[dict]):
    """Yield ``(transition_event, run_policy, terminal, verified_done, outcome, family)``.

    Run context supplies the policy needed to re-derive the offered catalogue
    (the rank coordinate the models train on) and the run's outcome class —
    ``success``/``failure``/``censored`` — which applies to *every* step of
    the trajectory. Transitions without a run_id keep the historical
    treatment: no outcome was recorded for them, so they carry ``legacy`` and
    continue to label as non-positive observations rather than disappearing
    from models trained on older stores.
    """
    events = list(events)
    runs: dict[str, list[dict]] = defaultdict(list)
    loose: list[dict] = []
    for event in events:
        if event.get("run_id"):
            runs[event["run_id"]].append(event)
        else:
            loose.append(event)
    for run_events in runs.values():
        run_events.sort(key=lambda e: (e.get("sequence", 0), e.get("recorded_at_ms", 0)))
        start = next((e for e in run_events if e.get("event") == "run_started"), None)
        final = next((e for e in reversed(run_events) if e.get("event") == "run_finished"), None)
        policy = None
        if start is not None and start.get("policy") is not None:
            try:
                policy = ExplorationPolicy.from_dict(start["policy"])
            except (ValueError, TypeError):
                policy = None
        # The run's final independently verified outcome — one label for the
        # whole trajectory. Every transition in a verified run shares it, so
        # intermediate page movement can never launder a failed trajectory.
        outcome = _run_outcome(final)
        run_verified = outcome == "success"
        family = str(start.get("task_family") or "").strip().lower() if start else ""
        transitions = [e for e in run_events if e.get("event") == "transition"]
        for index, event in enumerate(transitions):
            terminal = index == len(transitions) - 1 and final is not None
            verified_done = bool(terminal and run_verified)
            yield event, policy, terminal, verified_done, outcome, family
    for event in loose:
        if event.get("event") == "transition":
            yield event, None, False, False, "legacy", ""


def _selected_offered_rank(event: dict, policy: ExplorationPolicy | None) -> int | None:
    """Rank of the recorded action in the catalogue the model was offered.

    New traces record ``selected_offered_rank`` explicitly. For older evidence
    the honest resolution is: an absent offered catalogue means the observed
    catalogue *was* the offer, and a recorded policy lets the offered catalogue
    be re-derived exactly as replay does. Anything else is ``None`` (the
    "unknown" bucket) — an observed rank is never silently reinterpreted as an
    offered rank.
    """
    recorded = event.get("selected_offered_rank")
    if recorded is not None:
        rank = int(recorded)
        return rank if rank >= 0 else None
    candidates = event.get("candidates") or ()
    selected_id = (event.get("selected") or {}).get("id")
    if event.get("catalog") != "observed" and event.get("offered_digest") is None:
        # "recorded" catalogues: no separate offered set exists, so the
        # observed position IS the offered rank.
        return next((i for i, c in enumerate(candidates) if c.get("id") == selected_id), None)
    if policy is not None:
        offered = ReplaySimulator._retained_candidates(policy, candidates)
        return next((i for i, c in enumerate(offered) if c.get("id") == selected_id), None)
    return None


def _fit_cells(events: Iterable[dict], label) -> tuple[dict, dict, dict, int, int]:
    """Count outcomes per ``(kind, overlap_bucket, offered_rank_bucket)`` cell.

    ``label(event, terminal, verified_done, outcome)`` decides what counts as
    a positive observation — or returns ``None`` to censor the sample entirely
    (an unmeasured outcome is missing data, not a negative example). The rank
    coordinate is always the offered-catalogue rank so training and
    counterfactual replay share one basis.

    Also returns ``family_counts``: the same (kind, overlap, rank) counts
    stratified by the run's ``task_family``, so prediction can prefer the
    family-specific stratum before backing off to the pooled cell —
    unrelated tasks no longer share one tiny posterior.
    """
    cell_counts: dict[tuple[str, str, str], list[int]] = {}
    family_counts: dict[tuple[str, str, str, str], list[int]] = {}
    kind_counts: dict[str, list[int]] = {}
    positive_total = 0
    samples = 0
    for event, policy, terminal, verified_done, outcome, family in _transitions(events):
        selected = event.get("selected") or {}
        kind = str(selected.get("kind") or "unknown")
        selected_id = selected.get("id")
        candidates = event.get("candidates") or ()
        overlap = next(
            (int(c.get("goal_overlap", 0)) for c in candidates if c.get("id") == selected_id),
            0,
        )
        rank = _selected_offered_rank(event, policy)
        positive = label(event, terminal, verified_done, outcome)
        if positive is None:
            continue
        positive = int(bool(positive))
        cell = (kind, overlap_bucket(overlap), rank_bucket(rank))
        cell_counts.setdefault(cell, [0, 0])
        cell_counts[cell][0] += 1
        cell_counts[cell][1] += positive
        if family:
            fcell = (family,) + cell
            family_counts.setdefault(fcell, [0, 0])
            family_counts[fcell][0] += 1
            family_counts[fcell][1] += positive
        kind_counts.setdefault(kind, [0, 0])
        kind_counts[kind][0] += 1
        kind_counts[kind][1] += positive
        positive_total += positive
        samples += 1
    return cell_counts, family_counts, kind_counts, positive_total, samples


@dataclass(frozen=True)
class OutcomeModel:
    """Coarse transition-effect prior over (kind, overlap, offered-rank) cells.

    Learns P(page_changed) for the action actually taken online, keyed by the
    action's rank in the catalogue the recorded policy offered. Beta(1,1)
    smoothing keeps sparse cells honest; ``predict`` reports the supporting
    sample count so callers can see when a cell is a guess.
    """

    samples: int = 0
    cells: tuple[tuple[str, str, str, int, int], ...] = ()
    kind_totals: tuple[tuple[str, int, int], ...] = ()
    global_changed: int = 0
    version: str = "jev-outcome/2"

    @classmethod
    def fit(cls, events: Iterable[dict]) -> "OutcomeModel":
        cell_counts, _family_counts, kind_counts, changed_total, samples = _fit_cells(
            events,
            # ``page_changed`` is factual transition telemetry, not an outcome
            # judgment — a step that moved the page did so whether or not the
            # run later reached a measured outcome, so censored runs still
            # count here.
            lambda event, terminal, verified_done, outcome: bool(event.get("page_changed")),
        )
        return cls(
            samples=samples,
            cells=tuple(
                sorted((*cell, n, changed) for cell, (n, changed) in cell_counts.items())
            ),
            kind_totals=tuple(
                sorted((kind, n, changed) for kind, (n, changed) in kind_counts.items())
            ),
            global_changed=changed_total,
        )

    def predict(self, *, kind: str, goal_overlap: int = 0, rank: int | None = None) -> dict:
        cell_key = (str(kind), overlap_bucket(goal_overlap), rank_bucket(rank))
        for k, o, r, n, changed in self.cells:
            if (k, o, r) == cell_key:
                return {
                    "p_page_changed": (changed + 1) / (n + 2),
                    "n": n,
                    "level": "cell",
                }
        for k, n, changed in self.kind_totals:
            if k == cell_key[0]:
                return {
                    "p_page_changed": (changed + 1) / (n + 2),
                    "n": n,
                    "level": "kind",
                }
        return {
            "p_page_changed": (self.global_changed + 1) / (self.samples + 2),
            "n": self.samples,
            "level": "global",
        }

    @property
    def digest(self) -> str:
        return _stable_hash(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "OutcomeModel":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown outcome model keys: {sorted(unknown)}")
        cells = tuple(tuple(c) for c in payload.get("cells", ()))
        kind_totals = tuple(tuple(k) for k in payload.get("kind_totals", ()))
        return cls(
            samples=payload.get("samples", 0),
            cells=cells,
            kind_totals=kind_totals,
            global_changed=payload.get("global_changed", 0),
            version=payload.get("version", "jev-outcome/1"),
        )


@dataclass(frozen=True)
class ChoiceModel:
    """Uncertainty-aware counterfactual choice prior (Level 3, advisory).

    Beta-smoothed ``P(progress)`` over ``(kind, overlap, offered-rank)`` cells,
    with three deliberate properties:

    - The training coordinate is the action's rank in the catalogue the
      recorded policy actually *offered* — replay queries the model under the
      candidate policy's recomputed offered rank, so both sides share the
      offered-rank basis instead of mixing in the pre-policy observed rank.
    - The learned target is the run's final independently verified outcome:
      every action on a trajectory that ended verifier-confirmed ``done`` is
      a positive association, and every action on a failed or unverified run
      is a negative — *unless the run's outcome was never measured*. A run
      that ends ``aborted`` (operator closed the session, browser crashed,
      recording interrupted) or ``claimed_done`` without a verifier is
      censored out of the fit entirely: interruption is not evidence the
      trajectory was failing. This is deliberate *trajectory* credit —
      correlational, not per-action causality — and it restores the hardened
      v0.6.2 labeling: intermediate ``page_changed`` movement must never let a
      run the verifier rejected teach the prior that its actions were good.
      Per-action causal credit is the job of ``CounterfactualTrials``, which
      randomizes.
    - Every prediction carries the posterior standard deviation and an
      explicit ``confident`` flag; on sparse cells the model abstains rather
      than returning a smoothed guess.

    This is still a correlational prior over *selected* historical actions,
    not a causal success model: outcomes for rejected candidates are not in
    the data, and recorded propensities are the beginning of defensible
    off-policy estimation, not the end of it.

    ``choose`` proposes which offered action the prior would prefer using an
    upper-confidence score (mean + ``EXPLORE_BONUS`` * posterior std). A
    counterfactual proposal is annotation metadata only: it is never an
    outcome, never evidence, and is never consulted by a qualification gate.
    Only real executions may qualify anything.
    """

    samples: int = 0
    cells: tuple[tuple[str, str, str, int, int], ...] = ()
    # ``(task_family, kind, overlap_bucket, rank_bucket, n, positives)`` — the
    # family-specific stratum consulted before the pooled cell, so a prior
    # learned on "book a flight" stops contaminating "pay an invoice".
    family_cells: tuple[tuple[str, str, str, str, int, int], ...] = ()
    kind_totals: tuple[tuple[str, int, int], ...] = ()
    global_positive: int = 0
    version: str = "jev-choice/5"

    MIN_CONFIDENT = 4
    EXPLORE_BONUS = 0.5

    @classmethod
    def fit(cls, events: Iterable[dict]) -> "ChoiceModel":
        # Trajectory-success association (v0.6.2 semantics): the label is the
        # run's verified final outcome for *every* step, not the step's page
        # change. Censored runs (aborted/interrupted/unverifiable) drop out
        # instead of fitting as negatives; transitions without run context
        # keep the legacy non-positive label.
        cell_counts, family_counts, kind_counts, positive_total, samples = _fit_cells(
            events,
            lambda event, terminal, verified_done, outcome: (
                # An experiment-tagged transition was chosen by the trial
                # scheduler, not by the policy under study — fitting it as an
                # ordinary choice would mix randomized evidence into the
                # correlational prior. It belongs to CounterfactualTrials. The
                # same holds for an active causal-policy override: that action
                # was the causal prior's choice, not the recorded policy's.
                None
                if outcome == "censored"
                or event.get("experiment")
                or event.get("causal_override")
                else outcome == "success"
            ),
        )
        return cls(
            samples=samples,
            cells=tuple(
                sorted((*cell, n, changed) for cell, (n, changed) in cell_counts.items())
            ),
            family_cells=tuple(
                sorted((*cell, n, changed) for cell, (n, changed) in family_counts.items())
            ),
            kind_totals=tuple(
                sorted((kind, n, changed) for kind, (n, changed) in kind_counts.items())
            ),
            global_positive=positive_total,
        )

    @staticmethod
    def _posterior(n: int, changed: int) -> tuple[float, float]:
        """Beta(changed+1, n-changed+1) posterior mean and stddev."""
        mean = (changed + 1) / (n + 2)
        return mean, math.sqrt(mean * (1 - mean) / (n + 3))

    def predict(
        self,
        *,
        kind: str,
        goal_overlap: int = 0,
        rank: int | None = None,
        task_family: str | None = None,
    ) -> dict:
        cell_key = (str(kind), overlap_bucket(goal_overlap), rank_bucket(rank))
        family = str(task_family or "").strip().lower()
        if family:
            # Hierarchical backoff: the task-family stratum wins when it has
            # enough support of its own, otherwise the pooled cell answers.
            for f, k, o, r, n, changed in self.family_cells:
                if (f, k, o, r) == (family,) + cell_key and n >= self.MIN_CONFIDENT:
                    p, std = self._posterior(n, changed)
                    return {
                        "p_progress": p,
                        "n": n,
                        "uncertainty": std,
                        "level": "family",
                        "confident": True,
                        # Provenance: trajectory association over the
                        # observational trace — never causal evidence.
                        "source": "observational",
                    }
        for k, o, r, n, changed in self.cells:
            if (k, o, r) == cell_key:
                p, std = self._posterior(n, changed)
                return {
                    "p_progress": p,
                    "n": n,
                    "uncertainty": std,
                    "level": "cell",
                    "confident": n >= self.MIN_CONFIDENT,
                    "source": "observational",
                }
        for k, n, changed in self.kind_totals:
            if k == cell_key[0]:
                p, std = self._posterior(n, changed)
                return {
                    "p_progress": p,
                    "n": n,
                    "uncertainty": std,
                    "level": "kind",
                    "confident": n >= self.MIN_CONFIDENT,
                    "source": "observational",
                }
        p, std = self._posterior(self.samples, self.global_positive)
        return {
            "p_progress": p,
            "n": self.samples,
            "uncertainty": std,
            "level": "global",
            # Global support says nothing about this action type in this
            # context, so a global fallback is always an abstention.
            "confident": False,
            "source": "observational",
        }

    def choose(self, candidates: Iterable[dict], *, task_family: str | None = None) -> dict | None:
        """The offered candidate this prior prefers under uncertainty.

        Score = posterior mean + ``EXPLORE_BONUS`` * posterior std, so sparse
        but promising actions are still proposed (optimism) while the
        ``confident`` flag records how much evidence backs the proposal.
        ``rank`` is the candidate's position in the offered catalogue — the
        same basis as the recorded ``selected_offered_rank``.
        """
        best = None
        for rank, candidate in enumerate(candidates):
            prediction = self.predict(
                kind=str(candidate.get("kind") or "unknown"),
                goal_overlap=int(candidate.get("goal_overlap", 0) or 0),
                rank=rank,
                task_family=task_family,
            )
            score = prediction["p_progress"] + self.EXPLORE_BONUS * prediction["uncertainty"]
            if best is None or score > best[0]:
                best = (score, candidate, prediction)
        if best is None:
            return None
        return {
            "id": best[1].get("id"),
            "kind": best[1].get("kind"),
            "p_progress": best[2]["p_progress"],
            "uncertainty": best[2]["uncertainty"],
            "confident": best[2]["confident"],
            "score": best[0],
            "source": "observational",
        }

    @property
    def digest(self) -> str:
        return _stable_hash(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "ChoiceModel":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown choice model keys: {sorted(unknown)}")
        return cls(
            samples=payload.get("samples", 0),
            cells=tuple(tuple(c) for c in payload.get("cells", ())),
            family_cells=tuple(tuple(f) for f in payload.get("family_cells", ())),
            kind_totals=tuple(tuple(k) for k in payload.get("kind_totals", ())),
            global_positive=payload.get("global_positive", 0),
            version=payload.get("version", "jev-choice/1"),
        )
