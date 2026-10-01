"""Learned auxiliary models for DREAM-Jev hypothesis prioritization.

These are Level-1/Level-2 learners in the DREAM stack:

- Level 0 (authority): empirical replay of recorded trajectories plus bound
  live canaries. Only Level-0 evidence can stage or promote a policy.
- Level 1 (``CostModel``): estimates how token and latency cost scale with the
  offered candidate count, anchored to each recorded transition's real values.
  It reorders which replay-qualified candidate is selected for staging; it can
  never make a failing candidate pass a gate.
- Level 2 (``OutcomeModel``): a bucketed, Beta-smoothed estimate of coarse
  transition effects (did the page change?). It annotates candidates with prior
  expectations for experiment design; it is never consulted by a gate.
- Level 3 (``ChoiceModel``): an uncertainty-aware counterfactual choice prior
  trained on trajectory success — the run's final independently verified
  outcome labels every step of that run — over the action's *offered* rank.
  It predicts
  under the candidate policy's *own* recomputed offered rank and proposes
  which offered action it would prefer (posterior mean plus an exploration
  bonus on posterior stddev), abstaining via an explicit ``confident`` flag on
  sparse cells. A proposal is annotation metadata only — never an outcome,
  never evidence, never a gate input. A counterfactual "Candidate 17 would
  have succeeded" can only be tested by actually running Candidate 17; real
  executions qualify, priors merely propose.
- Level 4 (``CounterfactualTrials``): intention-to-treat effect estimates over
  real randomized arm assignments recorded before the authority plane. The
  unit of analysis is the assignment, not the executed step — denied,
  rejected, and stale trials count toward their arm — while runs that never
  produced a measured outcome are censored rather than counted as failures.

No model may fabricate a model choice or a browser outcome, and none is part
of the trusted evidence path.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Iterable

from .dream import ExplorationPolicy, ReplaySimulator


def _stable_hash(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


_OVERLAP_BUCKETS = ("0", "1", "2", "3+")
_RANK_BUCKETS = ("0-4", "5-19", "20-99", "100+")


def _bucket(value: int, edges: tuple[int, ...], labels: tuple[str, ...]) -> str:
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def overlap_bucket(overlap: int) -> str:
    return _bucket(max(0, int(overlap)), (1, 2, 3), _OVERLAP_BUCKETS)


def rank_bucket(rank: int | None) -> str:
    if rank is None or rank < 0:
        return "unknown"
    return _bucket(rank, (5, 20, 100), _RANK_BUCKETS)


def _run_outcome(final: dict | None) -> str:
    """Terminal outcome class of a recorded run.

    ``success`` — verifier-confirmed done. ``failure`` — a measured terminal
    non-success: blocked, denied, budget exhaustion, or a done-claim the
    verifier did not confirm. ``censored`` — no outcome was measured at all:
    the run was aborted, the recording was interrupted before
    ``run_finished``, or it ended ``claimed_done`` with no verifier ever
    checking. Censored is *missing data*, not a negative example — an
    operator closing a session or a browser crash says nothing about whether
    the trajectory was working.
    """
    if final is None:
        return "censored"
    status = str(final.get("status") or "")
    if status == "done" and final.get("verified"):
        return "success"
    if status in {"aborted", "claimed_done"}:
        return "censored"
    return "failure"


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
                # correlational prior. It belongs to CounterfactualTrials.
                None
                if outcome == "censored" or event.get("experiment")
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


def _wilson_interval(p: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval on a (possibly IPW-weighted) proportion.

    ``n`` is the effective sample size — the count of independent
    observations the weight mass is worth — so wide intervals on thin support
    are intrinsic to the estimate, not a reporting choice.
    """
    if n <= 0:
        return 0.0, 1.0
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _overlap_of(meta_key: str, cand_id_key: str, meta: dict, transition: dict | None) -> str:
    """Overlap bucket for a context member, from the assignment record first.

    Assignment meta carries the pre-treatment overlap directly; for records
    that lack it, resolve the candidate's recorded ``goal_overlap`` from the
    executed transition's catalogue by id.
    """
    raw = meta.get(meta_key)
    if raw is not None:
        try:
            return overlap_bucket(int(raw))
        except (TypeError, ValueError):
            pass
    if transition is not None:
        cand = next(
            (
                c
                for c in transition.get("candidates") or ()
                if c.get("id") == meta.get(cand_id_key)
            ),
            None,
        )
        if cand is not None:
            try:
                return overlap_bucket(int(cand.get("goal_overlap", 0) or 0))
            except (TypeError, ValueError):
                pass
    return "unknown"


def _trial_context(meta: dict, transition: dict | None) -> tuple[str, str, str, str, str]:
    """The cell a trial belongs to: ``(scope, mc_kind, mc_ov, p_kind, p_ov)``.

    ``scope`` is the task stratum — ``f:<task_family>``, ``s:<site>`` when no
    family was recorded, or ``*`` pooled — so estimates from unrelated tasks
    stop sharing one tiny posterior. The remaining coordinates key on the
    *pre-treatment divergence*: the model choice's kind/overlap and the
    proposal's kind/overlap. Keying on the executed action would put the two
    arms of one assignment in different cells whenever the proposal's overlap
    differs from the model choice's — exactly when the prior diverged for a
    reason. Assignment events carry the context directly; the transition's
    candidate list is only a fallback for older records that lack it.
    """
    family = str(meta.get("task_family") or "").strip().lower()
    site = str(meta.get("site") or "").strip().lower()
    scope = f"f:{family}" if family else (f"s:{site}" if site else "*")
    kind = str(meta.get("model_choice_kind") or "unknown")
    if kind == "unknown" and transition is not None:
        kind = str((transition.get("selected") or {}).get("kind") or "unknown")
    p_kind = str(meta.get("proposal_kind") or "unknown")
    if p_kind == "unknown" and transition is not None:
        p_cand = next(
            (
                c
                for c in transition.get("candidates") or ()
                if c.get("id") == meta.get("proposal_id")
            ),
            None,
        )
        if p_cand is not None:
            p_kind = str(p_cand.get("kind") or "unknown")
    return (
        scope,
        kind,
        _overlap_of("model_choice_overlap", "model_choice_id", meta, transition),
        p_kind,
        _overlap_of("proposal_overlap", "proposal_id", meta, transition),
    )


@dataclass(frozen=True)
class CounterfactualTrials:
    """Intention-to-treat estimates from randomized arm *assignments*.

    The unit of analysis is the ``experiment_assigned`` event — recorded at
    randomization, before the authority plane — not the executed transition.
    Assigning the candidate arm is the intervention; whatever the authority
    plane then does with it (execute, approval-pause, operator reject, policy
    deny) is part of the treatment being measured. An assignment that never
    produced a transition still enters its arm's estimate, so post-
    randomization selection cannot silently drop the losing half of a trial.

    Endpoint semantics per assignment:

    - ``success`` — the run finished verifier-confirmed ``done``.
    - ``failure`` — the run reached a measured terminal failure, including a
      policy denial or an operator rejection that ended the run. Under ITT
      that is a legitimate outcome of *assigning* the arm inside the real
      authority system, not censoring.
    - ``censored`` — the run produced no measured outcome at all (``aborted``,
      interrupted recording, unverifiable claim). Censored assignments are
      counted and reported but excluded from the endpoint estimate: an
      unrelated infrastructure interruption is not evidence about the arm.
      A trial whose execution is truly informative-but-blocked shows up as a
      ``failure``, so the estimand is "effect of being assigned this arm",
      not "effect of the arm executing" (which would need a compliance-aware
      design this deliberately does not pretend to be).

    ``p_page_changed`` remains secondary telemetry over *executed* trials —
    page movement is only defined for assignments that reached the browser —
    and is never a treatment effect.

    Context cells key on the model choice's ``(kind, overlap_bucket)``, the
    pre-randomization tension point, so both arms of one divergence share a
    cell. A legacy transition carrying an ``experiment`` block but no
    ``experiment_assigned`` event (pre-0.11 pools) is the only assignment
    evidence available for that run and is analyzed as such; pools with
    assignment events never take that path.

    Estimates annotate reports only; a promising candidate arm still has to
    qualify through the replay + bound-canary path like everything else.
    """

    # (scope, mc_kind, mc_ov, p_kind, p_ov, arm, assigned, analyzed, executed,
    #  censored, w_success, w_sum, w_sq, w_page, w_exec_sum)
    cells: tuple[
        tuple[str, str, str, str, str, str, int, int, int, int, float, float, float, float, float],
        ...,
    ] = ()
    version: str = "jev-trials/4"

    MIN_ESS = 8.0

    @classmethod
    def fit(cls, events: Iterable[dict]) -> "CounterfactualTrials":
        events = list(events)
        runs: dict[str, list[dict]] = defaultdict(list)
        loose: list[dict] = []
        for event in events:
            if event.get("run_id"):
                runs[event["run_id"]].append(event)
            else:
                loose.append(event)
        acc: dict[tuple[str, str], list[float]] = {}

        def record(meta: dict, outcome: str, page_changed, transition):
            """Fold one arm assignment into its context cell.

            ``page_changed`` is the executed step's telemetry (None when the
            trial never reached the browser); ``outcome`` is the run's class —
            censored assignments are tallied but carry no endpoint weight.
            """
            arm = str(meta.get("arm") or "")
            try:
                propensity = float(meta.get("assignment_probability"))
            except (TypeError, ValueError):
                return
            if arm not in {"candidate", "control"} or not 0.0 < propensity <= 1.0:
                return
            cell = acc.setdefault(
                (*_trial_context(meta, transition), arm),
                [0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
            cell[0] += 1  # assigned
            weight = 1.0 / propensity
            if page_changed is not None:
                cell[2] += 1  # executed
                cell[7] += float(bool(page_changed)) * weight
                cell[8] += weight
            if outcome == "censored":
                cell[3] += 1
                return
            cell[1] += 1  # analyzed (ITT denominator)
            cell[4] += float(outcome == "success") * weight
            cell[5] += weight
            cell[6] += weight * weight

        for run_id, run_events in runs.items():
            run_events.sort(key=lambda e: (e.get("sequence", 0), e.get("recorded_at_ms", 0)))
            final = next(
                (e for e in reversed(run_events) if e.get("event") == "run_finished"), None
            )
            outcome = _run_outcome(final)
            assignments = [
                e
                for e in run_events
                if e.get("event") == "experiment_assigned" and isinstance(e.get("experiment"), dict)
            ]
            trial_steps = [
                e
                for e in run_events
                if e.get("event") == "transition" and isinstance(e.get("experiment"), dict)
            ]
            if assignments:
                for assigned_event in assignments:
                    meta = assigned_event["experiment"]
                    step = cls._execution_step(meta, trial_steps)
                    record(
                        meta,
                        outcome,
                        step.get("page_changed") if step is not None else None,
                        step,
                    )
            else:
                # Pre-assignment evidence (pools older than jev-ultrafast-
                # tcb/0.11): an experiment-tagged transition is the only
                # assignment record that exists — analyzed as an assignment
                # observed at execution time.
                for step in trial_steps:
                    record(step["experiment"], outcome, step.get("page_changed"), step)
        for event in loose:
            # No run means no measurable outcome — the assignment is real but
            # censored by construction.
            if event.get("event") == "experiment_assigned" and isinstance(
                event.get("experiment"), dict
            ):
                record(event["experiment"], "censored", None, None)
            elif event.get("event") == "transition" and isinstance(event.get("experiment"), dict):
                record(event["experiment"], "censored", event.get("page_changed"), event)
        return cls(
            cells=tuple(
                sorted(
                    (*key[:5], key[5], *counts)
                    for key, counts in acc.items()
                )
            )
        )

    @staticmethod
    def _execution_step(meta: dict, trial_steps: list[dict]) -> dict | None:
        """The transition that executed this assignment, if one exists.

        At most one trial is assigned per run so at most one step matches;
        ``experiment_id``/arm/proposal identity are used when present, falling
        back to the run's single tagged transition.
        """
        if not trial_steps:
            return None
        experiment_id = meta.get("experiment_id")
        if experiment_id is not None:
            match = next(
                (
                    t
                    for t in trial_steps
                    if (t.get("experiment") or {}).get("experiment_id") == experiment_id
                ),
                None,
            )
            if match is not None:
                return match
        proposal_id = meta.get("proposal_id")
        match = next(
            (
                t
                for t in trial_steps
                if (t.get("experiment") or {}).get("proposal_id") == proposal_id
                and (t.get("experiment") or {}).get("model_choice_id")
                == meta.get("model_choice_id")
            ),
            None,
        )
        return match if match is not None else (trial_steps[0] if len(trial_steps) == 1 else None)

    def estimate(self, context: str | None = None) -> dict:
        """Self-normalized IPW intention-to-treat estimates per context/arm.

        ``p_success`` is the weighted share of non-censored assignments whose
        run ended verifier-confirmed done — the effect of *being assigned*
        the arm. ``trials`` is the analyzed (non-censored) count; ``assigned``
        includes censored ones. ``ess`` is the effective sample size of the
        analyzed pool — the count of independent observations the weights are
        worth — and ``p_success_ci``/``delta_ci`` are Wilson/Newcombe-Wilson
        intervals on that effective support, so thin evidence cannot
        masquerade as a confident estimate.
        """
        arms: dict[str, dict] = {}
        grouped: dict[tuple, dict[str, list]] = {}
        for scope, mk, mov, pk, pov, arm, *counts in self.cells:
            ctx = (scope, mk, mov, pk, pov)
            if context is not None and "|".join(ctx) != context:
                continue
            bucket = grouped.setdefault(ctx, {})
            slot = bucket.setdefault(arm, [0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0])
            for i, v in enumerate(counts):
                slot[i] += v
        for ctx, bucket in grouped.items():
            arms["|".join(ctx)] = self._contrast(bucket)
        return arms

    @staticmethod
    def _arm_entry(counts: list) -> dict:
        """Per-arm report from sufficient statistics
        ``[assigned, analyzed, executed, censored, ws, wsum, wsq, wpage, wexec]``."""
        assigned, analyzed, executed, censored, ws, wsum, wsq, wpage, wexec = counts
        ess = (wsum * wsum / wsq) if wsq else 0.0
        p_success = (ws / wsum) if wsum else 0.0
        return {
            # Primary endpoint: verified run success under ITT.
            # ``p_page_changed`` is executed-trial telemetry — never a
            # treatment effect.
            "p_success": p_success,
            "p_success_ci": list(_wilson_interval(p_success, ess)),
            "p_page_changed": (wpage / wexec) if wexec else None,
            "assigned": assigned,
            "trials": analyzed,
            "executed": executed,
            "censored": censored,
            "ess": ess,
            # Epsilon slack: IPW weights make an exactly-at-threshold
            # effective size land a hair below it in floating point.
            "reliable": ess >= CounterfactualTrials.MIN_ESS - 1e-9,
        }

    @classmethod
    def _contrast(cls, bucket: dict[str, list]) -> dict:
        """Per-arm estimates plus the candidate−control contrast."""
        entry = {arm: cls._arm_entry(counts) for arm, counts in bucket.items()}
        if "candidate" in entry and "control" in entry:
            candidate, control = entry["candidate"], entry["control"]
            delta = candidate["p_success"] - control["p_success"]
            entry["delta"] = delta
            if candidate["ess"] and control["ess"]:
                # Newcombe-Wilson interval on the difference, built from
                # the per-arm Wilson bounds. The naive normal-approx SE
                # collapses to zero when either arm sits at the 0/1
                # boundary — exactly the extreme evidence where a
                # degenerate interval would claim false precision.
                pc, pk = candidate["p_success"], control["p_success"]
                lc, uc = candidate["p_success_ci"]
                lk, uk = control["p_success_ci"]
                entry["delta_ci"] = [
                    delta - math.sqrt((pc - lc) ** 2 + (uk - pk) ** 2),
                    delta + math.sqrt((uc - pc) ** 2 + (pk - lk) ** 2),
                ]
            else:
                entry["delta_ci"] = None
            # The contrast is only as trustworthy as its weaker arm.
            entry["delta_reliable"] = candidate["reliable"] and control["reliable"]
        return entry

    def resolve(
        self,
        *,
        task_family: str | None = None,
        site: str | None = None,
        model_kind: str = "unknown",
        model_overlap=None,
        proposal_kind: str = "unknown",
        proposal_overlap=None,
    ) -> dict | None:
        """The best-supported effect estimate for one divergence hypothesis.

        Cells keep each assignment under its own scope, so strata merge by
        summing sufficient statistics — the pooled estimate stays honestly
        IPW-weighted. Resolution walks ``f:<family>`` → ``s:<site>`` →
        pooled-over-all-scopes and returns the first stratum whose contrast
        meets the reliability floor; a thin specific stratum does not shadow
        a reliable broader one (a 2-assignment family cell cannot hide a
        50-assignment pooled refutation). When no stratum is reliable the
        most specific estimate that exists is returned — still flagged
        ``delta_reliable: false`` — so reporting sees the best-supported data
        rather than nothing. ``level`` names which stratum answered.
        """
        try:
            mov = overlap_bucket(int(model_overlap)) if model_overlap is not None else "unknown"
        except (TypeError, ValueError):
            mov = "unknown"
        try:
            pov = overlap_bucket(int(proposal_overlap)) if proposal_overlap is not None else "unknown"
        except (TypeError, ValueError):
            pov = "unknown"
        sig = (str(model_kind or "unknown"), mov, str(proposal_kind or "unknown"), pov)
        scopes: list[tuple[str, str | None]] = []
        family = str(task_family or "").strip().lower()
        host = str(site or "").strip().lower()
        if family:
            scopes.append(("family", f"f:{family}"))
        if host:
            scopes.append(("site", f"s:{host}"))
        scopes.append(("pooled", None))
        fallback = None
        for level, scope in scopes:
            bucket: dict[str, list] = {}
            for cell in self.cells:
                if (scope is None or cell[0] == scope) and tuple(cell[1:5]) == sig:
                    slot = bucket.setdefault(cell[5], [0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    for i, v in enumerate(cell[6:]):
                        slot[i] += v
            if not bucket:
                continue
            contrast = {**self._contrast(bucket), "level": level}
            if contrast.get("delta_reliable"):
                return contrast
            if fallback is None:
                fallback = contrast
        return fallback

    @property
    def digest(self) -> str:
        return _stable_hash(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "CounterfactualTrials":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown counterfactual-trials keys: {sorted(unknown)}")
        cells = tuple(tuple(c) for c in payload.get("cells", ()))
        if any(len(c) != 15 for c in cells):
            raise ValueError("Unsupported counterfactual-trials cell arity")
        return cls(
            cells=cells,
            version=payload.get("version", "jev-trials/4"),
        )


@dataclass(frozen=True)
class TrialChoiceModel:
    """A decision prior built ONLY from randomized arm assignments.

    ``ChoiceModel`` learns what the recorded policy's own choices tended to
    lead to — correlational trajectory credit over observational traces.
    This model is the causal channel: it consumes the randomized
    ``experiment_assigned``/``run_finished`` evidence through
    ``CounterfactualTrials`` and reports the intention-to-treat effect of
    *assigning* a divergence — "when the model picks A-type here, assigning
    B-type changed verified success by delta" — with scope backoff so a
    task-family stratum wins when it has its own support and unrelated tasks
    stop pooling into one tiny posterior.

    Same advisory contract as the other learned layers: proposals are
    hypotheses, never evidence, never gate input. A candidate with a
    reliable non-positive delta is *refuted* — the proposal layer uses that
    to stop re-stamping an experiment the randomized evidence already
    settled, which is what makes the loop self-improving rather than
    self-repeating.
    """

    trials: CounterfactualTrials = CounterfactualTrials()
    version: str = "jev-causal/1"

    @classmethod
    def fit(cls, events: Iterable[dict]) -> "TrialChoiceModel":
        return cls(trials=CounterfactualTrials.fit(events))

    def _estimate(
        self,
        *,
        proposal_kind: str,
        proposal_overlap=None,
        model_kind: str = "unknown",
        model_overlap=None,
        task_family: str | None = None,
        site: str | None = None,
    ) -> dict | None:
        return self.trials.resolve(
            task_family=task_family,
            site=site,
            model_kind=model_kind,
            model_overlap=model_overlap,
            proposal_kind=proposal_kind,
            proposal_overlap=proposal_overlap,
        )

    def choose(
        self,
        candidates: Iterable[dict],
        *,
        model_choice: dict | None = None,
        task_family: str | None = None,
        site: str | None = None,
    ) -> dict | None:
        """The offered candidate with the best reliable measured effect.

        ``model_choice`` supplies the divergence premise — the action the
        decision model actually picked — because a trial only measures
        "assigning B when the model would have picked A". With no premise or
        no reliable positive delta for any offered candidate, abstain: a
        causal prior without evidence proposes nothing.
        """
        if not model_choice:
            return None
        mc_kind = str(model_choice.get("kind") or "unknown")
        mc_overlap = model_choice.get("goal_overlap")
        best = None
        for candidate in candidates:
            if candidate.get("id") == model_choice.get("id"):
                continue
            estimate = self._estimate(
                proposal_kind=str(candidate.get("kind") or "unknown"),
                proposal_overlap=candidate.get("goal_overlap"),
                model_kind=mc_kind,
                model_overlap=mc_overlap,
                task_family=task_family,
                site=site,
            )
            if (
                estimate is None
                or not estimate.get("delta_reliable")
                or estimate["delta"] <= 0
            ):
                continue
            # Among reliable positive deltas, prefer the strongest point
            # estimate — uncertainty is already gated by delta_reliable.
            if best is None or estimate["delta"] > best[0]:
                best = (estimate["delta"], candidate, estimate)
        if best is None:
            return None
        delta, candidate, estimate = best
        control_p = estimate["control"]["p_success"]
        ci = estimate.get("delta_ci") or [delta, delta]
        return {
            "id": candidate.get("id"),
            "kind": candidate.get("kind"),
            # The candidate's implied success probability — control-arm rate
            # plus the measured effect — so downstream ``expected_delta``
            # arithmetic stays probability-shaped.
            "p_progress": min(max(control_p + delta, 0.0), 1.0),
            "uncertainty": (ci[1] - ci[0]) / 2.0,
            "confident": True,
            "expected_delta": delta,
            "trial_level": estimate["level"],
            "delta_ci": ci,
        }

    def refuted(
        self,
        *,
        kind: str,
        goal_overlap=None,
        model_kind: str = "unknown",
        model_overlap=None,
        task_family: str | None = None,
        site: str | None = None,
    ) -> bool:
        """True when randomized evidence already settled this divergence as
        non-positive at a reliable stratum — the hypothesis was tested and
        failed, so the proposal layer should not stamp it again."""
        estimate = self._estimate(
            proposal_kind=kind,
            proposal_overlap=goal_overlap,
            model_kind=model_kind,
            model_overlap=model_overlap,
            task_family=task_family,
            site=site,
        )
        return bool(
            estimate is not None
            and estimate.get("delta_reliable")
            and estimate["delta"] <= 0
        )

    @property
    def digest(self) -> str:
        return _stable_hash(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "TrialChoiceModel":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown trial-choice model keys: {sorted(unknown)}")
        return cls(
            trials=CounterfactualTrials.from_dict(payload.get("trials") or {}),
            version=payload.get("version", "jev-causal/1"),
        )
