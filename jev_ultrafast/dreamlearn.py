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
  produced a measured outcome are censored (with a structured termination
  reason) rather than counted as failures. Support and effect certainty are
  separate: ``support_sufficient`` is the weighted-sample floor,
  ``effect_status`` is the sign actually established by a sequentially valid
  interval, and differential censoring blocks establishment outright.
- Level 5 (``TrialChoiceModel``): the causal decision prior. It proposes only
  divergences whose randomized evidence *established* a beneficial effect,
  treats only established-harmful divergences as refuted, and keeps its
  provenance (``source: "randomized"``) separate from the observational
  ``ChoiceModel`` forever. ``ExperimentScheduler`` ranks unresolved
  hypotheses by expected information gain per trial cost, and
  ``CausalChoicePolicy`` combines the channels under operator-gated modes
  (shadow → canary → active) — always as prioritization, never as authority.

No model may fabricate a model choice or a browser outcome, and none is part
of the trusted evidence path.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from statistics import NormalDist, fmean
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


# Structured censoring causes. A bare ``aborted`` status cannot distinguish an
# operator closing the session (unrelated to treatment) from a candidate arm
# that hangs or crashes the browser (potentially *caused* by treatment) — and
# differential censoring between arms is exactly the pattern that can make a
# dangerous candidate look excellent on the surviving subset. The reason
# taxonomy keeps those cases separable in evidence; the censor-rate imbalance
# gate in ``CounterfactualTrials`` refuses to establish an effect while the
# arms are censored at sharply different rates, because the surviving subset is
# then a biased sample of the arm, not a smaller unbiased one.
TERMINATION_REASONS = (
    "operator_cancel",
    "browser_crash",
    "agent_exception",
    "network_failure",
    "timeout",
    "recorder_shutdown",
    "unverified_claim",
    "unknown_abort",
)
_REASON_INDEX = {reason: index for index, reason in enumerate(TERMINATION_REASONS)}


def _termination_reason(final: dict | None) -> str:
    """Why a censored run produced no measured outcome.

    New evidence records the reason on ``run_finished`` (``operator_cancel``
    for a session closed mid-run, the exception class for a crash/timeout/
    network failure, ``recorder_shutdown`` when the store itself failed).
    Older evidence without a reason is ``unknown_abort``; a run with no
    ``run_finished`` at all is ``unknown_abort`` too — the recording was
    interrupted, and pretending to know why would be invention.
    """
    if final is None:
        return "unknown_abort"
    status = str(final.get("status") or "")
    if status == "claimed_done":
        return "unverified_claim"
    reason = str(final.get("reason") or "").strip().lower()
    return reason if reason in _REASON_INDEX else "unknown_abort"


_PHASE_BUCKETS = ("0-2", "3-9", "10+", "unknown")


def phase_bucket(step) -> str:
    """Workflow-phase bucket of a step index — early/mid/late/unknown."""
    if step is None:
        return "unknown"
    try:
        value = max(0, int(step))
    except (TypeError, ValueError):
        return "unknown"
    return _bucket(value, (3, 10), _PHASE_BUCKETS)


def _coord(value) -> str:
    """A trial-context coordinate, safe for the ``|``-joined context key.

    Page metadata is attacker-shaped input: a role like ``"a|b"`` would let two
    distinct context tuples collide under ``"|".join`` — silently merging two
    cells into one estimate key. The join delimiter can never survive into a
    stored coordinate.
    """
    text = str(value).replace("|", " ").strip()
    return text if text else "unknown"


def _role_bucket(role) -> str:
    return _coord(str(role or "").lower())


def _effect_bucket(effect) -> str:
    """Effect class of an action, as the deterministic classifier names it."""
    if effect is None:
        return "unknown"
    return _coord(str(getattr(effect, "value", effect)).lower())


# The bounded treatment signature: which *kind* of intervention this arm was,
# in which semantic class, on which element role, at which overlap/rank/phase.
# Coarse by design — every coordinate is derivable at assignment time and
# auditable — but rich enough that a measured delta is not applied to an
# arbitrary never-tested action ID. Resolution backs off over these
# coordinates hierarchically (see ``CounterfactualTrials.resolve``).
_SIGNATURE_FIELDS = (
    "m_kind", "m_effect", "m_role", "m_ov", "m_rank", "m_phase",
    "p_kind", "p_effect", "p_role", "p_ov", "p_rank", "p_phase",
)


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


def _rank_of(meta: dict, key: str) -> str:
    raw = meta.get(key)
    if raw is None:
        return "unknown"
    try:
        return rank_bucket(int(raw))
    except (TypeError, ValueError):
        return "unknown"


def _trial_context(meta: dict, transition: dict | None) -> tuple[str, ...]:
    """The cell a trial belongs to — 14 coordinates, no information collapsed.

    ``(task_family, site, m_kind, m_effect, m_role, m_ov, m_rank, m_phase,
    p_kind, p_effect, p_role, p_ov, p_rank, p_phase)``.

    Task family and site are *separate* coordinates: a trial carrying both must
    stay indexable under either (the v0.8.3 ``scope`` collapsed them into one
    mutually exclusive key, so a family-carrying trial lost its site stratum
    and resolution could never fall back to it). Resolution walks
    family+site → site → family → pooled over these stored cells.

    The remaining coordinates are the bounded *treatment signature* of the
    pre-treatment divergence: the model choice's kind/effect-class/role/
    overlap/offered-rank/phase and the proposal's same six. Keying on the
    divergence (not the executed action) keeps both arms of one assignment in
    one cell. Assignment events carry every coordinate directly; the
    transition's selected action and candidate list are fallbacks for older
    records, and anything still unknown stays ``"unknown"`` — a wildcard
    dimension during resolution, never a fabricated value.
    """
    family = str(meta.get("task_family") or "").replace("|", " ").strip().lower()
    site = str(meta.get("site") or "").replace("|", " ").strip().lower()
    selected = (transition.get("selected") or {}) if transition is not None else {}
    m_kind = str(meta.get("model_choice_kind") or "unknown").replace("|", " ")
    if m_kind == "unknown":
        m_kind = str(selected.get("kind") or "unknown").replace("|", " ")
    p_kind = str(meta.get("proposal_kind") or "unknown").replace("|", " ")
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
            p_kind = str(p_cand.get("kind") or "unknown").replace("|", " ")
    return (
        family,
        site,
        m_kind,
        _effect_bucket(meta.get("model_choice_effect", selected.get("effect"))),
        _role_bucket(meta.get("model_choice_role")),
        _overlap_of("model_choice_overlap", "model_choice_id", meta, transition),
        _rank_of(meta, "model_choice_offered_rank"),
        phase_bucket(meta.get("phase")),
        p_kind,
        _effect_bucket(meta.get("proposal_effect")),
        _role_bucket(meta.get("proposal_role")),
        _overlap_of("proposal_overlap", "proposal_id", meta, transition),
        _rank_of(meta, "proposal_offered_rank"),
        phase_bucket(meta.get("phase")),
    )


@dataclass(frozen=True)
class UtilityWeights:
    """Multi-objective utility weights for *prioritization* — never a gate.

    Per analyzed assignment the utility is ``success·p - λ_failures·(1-p) -
    λ_latency·latency_ms - λ_tokens·tokens - λ_approvals·approvals -
    λ_censoring·censor_rate``. Statistical significance alone is not a reason
    to prefer an arm: a +0.5% success improvement that doubles latency should
    not automatically count as an improvement.

    Hard safety and authority constraints stay *outside* this function. No
    weight combination can trade away an approval, an effect classification,
    or a browser guard — the utility only decides which hypotheses are worth
    more real experiments.
    """

    success: float = 1.0
    failures: float = 0.25
    latency_ms: float = 1.0 / 60000.0
    tokens: float = 1.0 / 10000.0
    approvals: float = 0.05
    censoring: float = 0.10

    def arm_utility(self, arm: dict) -> float:
        p = float(arm.get("p_success") or 0.0)
        return (
            self.success * p
            - self.failures * (1.0 - p)
            - self.latency_ms * float(arm.get("avg_latency_ms") or 0.0)
            - self.tokens * float(arm.get("avg_tokens") or 0.0)
            - self.approvals * float(arm.get("avg_approvals") or 0.0)
            - self.censoring * float(arm.get("censor_rate") or 0.0)
        )


# --- trial cell layout (jev-trials/5) --------------------------------------
# Key (15): family, site, m_kind, m_effect, m_role, m_ov, m_rank, m_phase,
#           p_kind, p_effect, p_role, p_ov, p_rank, p_phase, arm
# Stats (9): assigned, analyzed, executed, censored,
#            w_success, w_sum, w_sq, w_page, w_exec
# Costs (3): w_latency, w_tokens, w_approvals
# Reasons (8): one counter per TERMINATION_REASONS entry
_TRIAL_KEY_LEN = 15
_TRIAL_STATS = 9
_TRIAL_COSTS = 3
_COST_BASE = _TRIAL_KEY_LEN + _TRIAL_STATS
_REASON_BASE = _COST_BASE + _TRIAL_COSTS
TRIAL_CELL_LEN = _REASON_BASE + len(TERMINATION_REASONS)
# The sufficient-statistics tail stored per cell (everything after the key).
TRIAL_COUNT_LEN = TRIAL_CELL_LEN - _TRIAL_KEY_LEN
_COUNT_REASON_BASE = _REASON_BASE - _TRIAL_KEY_LEN

_SIGNATURE_INDEX = {field: 2 + index for index, field in enumerate(_SIGNATURE_FIELDS)}
_SIGNATURE_DROP_FIELDS = (
    ("phase", ("m_phase", "p_phase")),
    ("rank", ("m_rank", "p_rank")),
    ("role", ("m_role", "p_role")),
    ("effect", ("m_effect", "p_effect")),
    ("overlap", ("m_ov", "p_ov")),
)
# ``kind`` is the signature floor: it is never dropped. A click's measured
# effect must not answer a fill proposal just because every other coordinate
# was too thin — operation kind is always known and semantically meaningful,
# so evidence never generalizes across it.

# Summable alpha-spending constant: sum over integer looks of
# SPEND·n^-1.5 equals 1 (ζ(3/2) normalizes the sequence).
_ZETA_1_5 = 2.6123753486854882
_SEQUENTIAL_SPEND = 1.0 / _ZETA_1_5


def _sequential_alpha(looks: float, *, base: float) -> float:
    """Alpha spent at look ``n``: a summable n^-3/2 spending sequence.

    Establishment is checked whenever a contrast is recomputed, so the
    per-look error budget must be summable: with ``α_n = base·n^-1.5/ζ(3/2)``
    the union bound over every integer look gives an anytime-valid guarantee —
    the probability that a single fixed hypothesis is *ever* established at
    the wrong sign is at most ``base``, no matter how often it is peeked at.
    Multiplicity across simultaneously tracked hypotheses is not corrected by
    this bound (documented limitation); the conservative direction is always
    ``unresolved``.
    """
    n = max(1.0, float(looks))
    return min(0.5, float(base) * _SEQUENTIAL_SPEND / (n ** 1.5))


def _z_for_alpha(alpha: float) -> float:
    return NormalDist().inv_cdf(1.0 - float(alpha) / 2.0)


def _newcombe(candidate: dict, control: dict, *, z: float = 1.96) -> list[float]:
    """Newcombe-Wilson interval on the difference of two arm proportions.

    Built from the per-arm Wilson bounds at ``z``. The naive normal-approx SE
    collapses to zero when either arm sits at the 0/1 boundary — exactly the
    extreme evidence where a degenerate interval would claim false precision.
    """
    pc, pk = candidate["p_success"], control["p_success"]
    delta = pc - pk
    lc, uc = _wilson_interval(pc, candidate["ess"], z)
    lk, uk = _wilson_interval(pk, control["ess"], z)
    return [
        delta - math.sqrt((pc - lc) ** 2 + (uk - pk) ** 2),
        delta + math.sqrt((uc - pc) ** 2 + (pk - lk) ** 2),
    ]


def _signature_masks(known: list[str]) -> list[tuple[str, tuple[str, ...]]]:
    """Progressive backoff over the *known* signature coordinates.

    Each mask names the coordinates still enforced; the rest are wildcards
    merged by summation. The order is most specific first, dropping phase,
    rank, role, effect class, and overlap — a bounded, auditable hierarchy
    that lets thin signature cells borrow support from coarser ones without
    ever mixing unrelated strata or inventing values. ``kind`` is the floor:
    it is never dropped, so click evidence can never answer a fill proposal.
    """
    enforced = list(known)
    masks = [("full", tuple(enforced))]
    dropped: list[str] = []
    for name, fields in _SIGNATURE_DROP_FIELDS:
        dropped.append(name)
        enforced = [f for f in enforced if f not in fields]
        label = "minus_" + "_".join(dropped)
        if masks[-1][1] != tuple(enforced):
            masks.append((label, tuple(enforced)))
    return masks


def _migrate_v4_trial_cells(cells) -> tuple[tuple, ...]:
    """v4 cells → v5: split the collapsed scope, widen the signature.

    v4 stored one mutually exclusive scope (``f:<family>`` / ``s:<site>`` /
    ``*``); splitting it restores the lost dimension, so family-carrying
    trials regain their site stratum. Signature coordinates v4 never recorded
    (effect class, role, offered rank, phase) migrate as ``"unknown"`` —
    wildcards, never invented values — and censored counts migrate under
    ``unknown_abort`` because v4 evidence did not record why a run stopped.

    Values are carried over verbatim: the four counters are ints, but the
    weighted sums are floats and *fractional* whenever IPW weights differ
    (propensity 1/3 → weight 3.0 against others) — casting them would
    silently truncate the evidence.
    """
    migrated = []
    for cell in cells:
        scope, mk, mov, pk, pov, arm, *counts = cell
        if scope.startswith("f:"):
            family, site = scope[2:], ""
        elif scope.startswith("s:"):
            family, site = "", scope[2:]
        else:
            family, site = "", ""
        stats = list(counts[:9])
        key = (
            family, site,
            str(mk), "unknown", "unknown", str(mov), "unknown", "unknown",
            str(pk), "unknown", "unknown", str(pov), "unknown", "unknown",
            str(arm),
        )
        reasons = [0] * len(TERMINATION_REASONS)
        reasons[_REASON_INDEX["unknown_abort"]] = stats[3]
        migrated.append((*key, *stats, 0.0, 0.0, 0.0, *reasons))
    return tuple(sorted(migrated))


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

    Two *different* questions are kept separate, because v0.8.3 conflated
    them and the conflation is a real defect:

    - ``support_sufficient`` — do both arms have enough weighted observations
      (``MIN_ESS``) to say anything at all?
    - ``effect_status`` — has the data actually established the sign of the
      treatment effect? ``beneficial`` requires the *sequentially valid*
      interval to lie entirely above ``min_effect``; ``harmful`` requires it
      to lie entirely below ``-min_effect``; a confidence interval that
      crosses zero leaves the hypothesis ``unresolved`` and schedules more
      evidence. "Eight observations exist" is not "we know the intervention
      helps".

    ``delta_ci`` is the fixed-sample Newcombe-Wilson interval for reporting;
    ``sequential_delta_ci`` is the anytime-valid interval establishment uses
    (see ``_sequential_alpha``), so repeatedly peeking after every batch
    cannot inflate the error rate of a single hypothesis.

    Censoring is a first-class causal concern, not bookkeeping: per-arm
    ``censor_rate`` and the ``terminations`` reason breakdown are reported,
    and ``censoring_imbalance`` refuses to establish an effect when either
    arm is censored at an extreme rate or the arms are censored at sharply
    different rates — eight surviving successes must not hide forty
    candidate-arm crashes. The taxonomy separates an operator closing the
    session (unrelated to treatment) from a crash/timeout that the treatment
    itself may have caused, so differential censoring stays visible.

    ``p_page_changed`` remains secondary telemetry over *executed* trials —
    page movement is only defined for assignments that reached the browser —
    and is never a treatment effect.

    Context cells keep task family and site as separate coordinates, plus the
    bounded treatment signature of the pre-treatment divergence, so both arms
    of one divergence share a cell and resolution can walk
    family+site → site → family → pooled *and* back off over signature
    specificity without ever destroying evidence during ingestion. A legacy
    transition carrying an ``experiment`` block but no ``experiment_assigned``
    event (pre-0.11 pools) is the only assignment evidence available for that
    run and is analyzed as such; pools with assignment events never take that
    path.

    Estimates annotate reports only; a promising candidate arm still has to
    qualify through the replay + bound-canary path like everything else.
    """

    cells: tuple[tuple, ...] = ()
    # Assignment records deduplicated during the fit: the same
    # ``(run, experiment_id, arm)`` assignment written twice is one unit of
    # analysis, and counting it twice would double-weight that arm. Reported
    # so corruption stays visible instead of silently biasing an estimate.
    duplicates: int = 0
    version: str = "jev-trials/5"

    MIN_ESS = 8.0
    MIN_EFFECT = 0.0
    SEQUENTIAL_ALPHA = 0.05
    CENSOR_RATE_MAX = 0.5
    CENSOR_RATE_GAP = 0.25

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
        acc: dict[tuple, list] = {}
        duplicates = 0

        def record(meta, outcome, reason, page_changed, transition, run_costs):
            """Fold one arm assignment into its context cell.

            ``page_changed`` is the executed step's telemetry (None when the
            trial never reached the browser); ``outcome`` is the run's class —
            censored assignments are tallied with their termination reason but
            carry no endpoint weight. ``run_costs`` are the run's latency,
            tokens, and approval count, used only for the multi-objective
            utility annotation.
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
                [0] * TRIAL_COUNT_LEN,
            )
            weight = 1.0 / propensity
            cell[0] += 1  # assigned
            if page_changed is not None:
                cell[2] += 1  # executed
                cell[7] += float(bool(page_changed)) * weight
                cell[8] += weight
            if outcome == "censored":
                cell[3] += 1
                cell[_COUNT_REASON_BASE + _REASON_INDEX[reason]] += 1
                return
            cell[1] += 1  # analyzed (ITT denominator)
            cell[4] += float(outcome == "success") * weight
            cell[5] += weight
            cell[6] += weight * weight
            latency, tokens, approvals = run_costs
            cell[9] += latency * weight
            cell[10] += tokens * weight
            cell[11] += approvals * weight

        for run_id, run_events in runs.items():
            run_events.sort(key=lambda e: (e.get("sequence", 0), e.get("recorded_at_ms", 0)))
            final = next(
                (e for e in reversed(run_events) if e.get("event") == "run_finished"), None
            )
            outcome = _run_outcome(final)
            reason = _termination_reason(final)
            run_costs = (
                sum(max(0, int(e.get("latency_ms", 0))) for e in run_events if e.get("event") == "transition"),
                sum(max(0, int(e.get("tokens", 0))) for e in run_events if e.get("event") == "transition"),
                sum(1 for e in run_events if e.get("event") == "approval_required"),
            )
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
            seen: set = set()
            if assignments:
                for assigned_event in assignments:
                    meta = assigned_event["experiment"]
                    identity = (
                        meta.get("experiment_id"),
                        meta.get("proposal_id"),
                        meta.get("model_choice_id"),
                        meta.get("arm"),
                    )
                    if identity in seen:
                        # A duplicated assignment record is one unit of
                        # analysis; counting it twice would double-weight the
                        # arm. The duplicate count is reported, not hidden.
                        duplicates += 1
                        continue
                    seen.add(identity)
                    step = cls._execution_step(meta, trial_steps)
                    record(
                        meta,
                        outcome,
                        reason,
                        step.get("page_changed") if step is not None else None,
                        step,
                        run_costs,
                    )
            else:
                # Pre-assignment evidence (pools older than jev-ultrafast-
                # tcb/0.11): an experiment-tagged transition is the only
                # assignment record that exists — analyzed as an assignment
                # observed at execution time.
                for step in trial_steps:
                    record(step["experiment"], outcome, reason, step.get("page_changed"), step, run_costs)
        loose_seen: set = set()
        for event in loose:
            # No run means no measurable outcome — the assignment is real but
            # censored by construction, with no recorded cause. Dedup applies
            # here too: a loose record replayed twice is still one unit.
            if event.get("event") == "experiment_assigned" and isinstance(
                event.get("experiment"), dict
            ):
                meta = event["experiment"]
                identity = (
                    meta.get("experiment_id"),
                    meta.get("proposal_id"),
                    meta.get("model_choice_id"),
                    meta.get("arm"),
                )
                if identity in loose_seen:
                    duplicates += 1
                    continue
                loose_seen.add(identity)
                record(meta, "censored", "unknown_abort", None, None, (0, 0, 0))
            elif event.get("event") == "transition" and isinstance(event.get("experiment"), dict):
                record(
                    event["experiment"], "censored", "unknown_abort",
                    event.get("page_changed"), event, (0, 0, 0),
                )
        return cls(
            cells=tuple(sorted(key + tuple(counts) for key, counts in acc.items())),
            duplicates=duplicates,
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
        if proposal_id is not None:
            model_choice_id = meta.get("model_choice_id")
            match = next(
                (
                    t
                    for t in trial_steps
                    if (t.get("experiment") or {}).get("proposal_id") == proposal_id
                    and (
                        model_choice_id is None
                        or (t.get("experiment") or {}).get("model_choice_id")
                        == model_choice_id
                    )
                ),
                None,
            )
            if match is not None:
                return match
        # A meta with no disambiguating ids must not match a step by
        # ``None == None`` — that would attribute a different experiment's
        # execution to this assignment. Only a single tagged step is safe.
        return trial_steps[0] if len(trial_steps) == 1 else None

    def estimate(self, context: str | None = None) -> dict:
        """Self-normalized IPW intention-to-treat estimates per context/arm.

        ``p_success`` is the weighted share of non-censored assignments whose
        run ended verifier-confirmed done — the effect of *being assigned*
        the arm. ``trials`` is the analyzed (non-censored) count; ``assigned``
        includes censored ones, reported alongside ``censor_rate`` and the
        per-arm ``terminations`` reason breakdown. ``ess`` is the effective
        sample size of the analyzed pool — the count of independent
        observations the weights are worth.

        Each context carries two *different* questions: ``support_sufficient``
        (enough weighted observations to say anything) and ``effect_status``
        (``beneficial`` / ``harmful`` / ``unresolved`` / ``insufficient_data``)
        — a cell with enough samples whose interval crosses zero stays
        unresolved rather than becoming a recommendation or a refutation.
        """
        arms: dict[str, dict] = {}
        grouped: dict[tuple, dict[str, list]] = {}
        for cell in self.cells:
            ctx = tuple(cell[:_TRIAL_KEY_LEN - 1])
            if context is not None and "|".join(ctx) != context:
                continue
            bucket = grouped.setdefault(ctx, {})
            arm = cell[_TRIAL_KEY_LEN - 1]
            slot = bucket.setdefault(arm, [0] * TRIAL_COUNT_LEN)
            for i, v in enumerate(cell[_TRIAL_KEY_LEN:]):
                slot[i] += v
        for ctx, bucket in grouped.items():
            arms["|".join(ctx)] = self._contrast(bucket)
        return arms

    @staticmethod
    def _arm_entry(counts: list) -> dict:
        """Per-arm report from the sufficient statistics (see cell layout)."""
        assigned, analyzed, executed, censored, ws, wsum, wsq, wpage, wexec = counts[:9]
        w_latency, w_tokens, w_approvals = counts[9:12]
        terminations = {
            reason: int(counts[12 + index])
            for index, reason in enumerate(TERMINATION_REASONS)
            if counts[12 + index]
        }
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
            "censor_rate": (censored / assigned) if assigned else 0.0,
            "terminations": terminations,
            "ess": ess,
            # Epsilon slack: IPW weights make an exactly-at-threshold
            # effective size land a hair below it in floating point.
            "support_sufficient": ess >= CounterfactualTrials.MIN_ESS - 1e-9,
            "avg_latency_ms": (w_latency / wsum) if wsum else 0.0,
            "avg_tokens": (w_tokens / wsum) if wsum else 0.0,
            "avg_approvals": (w_approvals / wsum) if wsum else 0.0,
        }

    @classmethod
    def _contrast(cls, bucket: dict[str, list], *, min_effect=None, weights=None) -> dict:
        """Per-arm estimates plus the candidate−control contrast.

        Separates *support* from *effect certainty*: ``support_sufficient`` is
        the weighted-sample floor, ``effect_status`` is the sign actually
        established by the sequentially valid interval. A censoring-rate
        imbalance blocks establishment outright — the surviving subset of an
        arm censored far more than the other is a biased sample, not a
        smaller unbiased one.
        """
        entry = {arm: cls._arm_entry(counts) for arm, counts in bucket.items()}
        if "candidate" not in entry or "control" not in entry:
            return entry
        candidate, control = entry["candidate"], entry["control"]
        delta = candidate["p_success"] - control["p_success"]
        entry["delta"] = delta
        if candidate["ess"] and control["ess"]:
            entry["delta_ci"] = _newcombe(candidate, control)
            # Establishment uses the anytime-valid interval: the fixed-sample
            # interval is for reporting, not for deciding that an experiment
            # is settled after being peeked at every batch.
            looks = max(1, candidate["trials"] + control["trials"])
            alpha = _sequential_alpha(looks, base=cls.SEQUENTIAL_ALPHA)
            entry["sequential_alpha"] = alpha
            entry["sequential_delta_ci"] = _newcombe(candidate, control, z=_z_for_alpha(alpha))
        else:
            entry["delta_ci"] = None
            entry["sequential_alpha"] = None
            entry["sequential_delta_ci"] = None
        support = bool(candidate["support_sufficient"] and control["support_sufficient"])
        entry["support_sufficient"] = support
        imbalance = (
            max(candidate["censor_rate"], control["censor_rate"]) > cls.CENSOR_RATE_MAX
            or abs(candidate["censor_rate"] - control["censor_rate"]) > cls.CENSOR_RATE_GAP
        )
        entry["censoring_imbalance"] = imbalance
        threshold = cls.MIN_EFFECT if min_effect is None else max(0.0, float(min_effect))
        entry["min_effect"] = threshold
        sequential = entry["sequential_delta_ci"]
        if not support or sequential is None:
            entry["effect_status"] = "insufficient_data"
            entry["unresolved_reason"] = None
        elif imbalance:
            entry["effect_status"] = "unresolved"
            entry["unresolved_reason"] = "censoring_imbalance"
        elif sequential[0] > threshold:
            entry["effect_status"] = "beneficial"
            entry["unresolved_reason"] = None
        elif sequential[1] < -threshold:
            entry["effect_status"] = "harmful"
            entry["unresolved_reason"] = None
        else:
            entry["effect_status"] = "unresolved"
            entry["unresolved_reason"] = "ci_crosses_zero"
        weights = weights or UtilityWeights()
        entry["utility"] = {
            arm: weights.arm_utility(entry[arm]) for arm in ("candidate", "control")
        }
        entry["utility_delta"] = entry["utility"]["candidate"] - entry["utility"]["control"]
        return entry

    @staticmethod
    def _stratum_matches(cell, level: str, family: str, host: str) -> bool:
        if level == "family+site":
            return cell[0] == family and cell[1] == host
        if level == "site":
            return cell[1] == host
        if level == "family":
            return cell[0] == family
        return True

    def resolve(
        self,
        *,
        task_family: str | None = None,
        site: str | None = None,
        model_kind: str = "unknown",
        model_overlap=None,
        proposal_kind: str = "unknown",
        proposal_overlap=None,
        model_effect=None,
        proposal_effect=None,
        model_role=None,
        proposal_role=None,
        model_rank=None,
        proposal_rank=None,
        phase=None,
        min_effect=None,
        weights=None,
    ) -> dict | None:
        """The best-supported effect estimate for one divergence hypothesis.

        Two hierarchies are walked, outermost first:

        1. **Strata** — ``family+site`` → ``site`` → ``family`` → pooled. Task
           family and site are separate stored coordinates, so a
           family-carrying trial keeps its site stratum and resolution can
           genuinely fall back through it (the v0.8.3 collapsed scope made
           that path unreachable). The first stratum with a
           ``support_sufficient`` contrast answers; a *thin* specific stratum
           cannot shadow a reliable broader one, but a supported specific
           stratum does answer even when its effect is still unresolved —
           "we don't know yet in this context" is the honest answer there,
           and pooling unrelated tasks must not override it.
        2. **Treatment signature** — within a stratum, the full signature is
           tried first, then coordinates are dropped in order (phase, rank,
           role, effect class, overlap) until a mask has both arms with
           support. ``kind`` is never dropped: it is the signature floor, so
           evidence about one operation kind can never answer for another.
           Coordinates the caller does not know are wildcards from
           the start; unknown *stored* values only match once their coordinate
           is dropped.

        When no stratum supports a contrast, the most specific estimate that
        exists is returned, flagged ``effect_status: insufficient_data``, so
        reporting sees the best-supported data rather than nothing.
        ``level`` and ``signature_level`` name which mask answered.
        """
        def _overlap(value):
            try:
                return overlap_bucket(int(value)) if value is not None else "unknown"
            except (TypeError, ValueError):
                return "unknown"

        def _rank(value):
            try:
                return rank_bucket(int(value)) if value is not None else "unknown"
            except (TypeError, ValueError):
                return "unknown"

        query = {
            "m_kind": str(model_kind or "unknown").replace("|", " ") or "unknown",
            "m_effect": _effect_bucket(model_effect),
            "m_role": _role_bucket(model_role),
            "m_ov": _overlap(model_overlap),
            "m_rank": _rank(model_rank),
            "m_phase": phase_bucket(phase),
            "p_kind": str(proposal_kind or "unknown").replace("|", " ") or "unknown",
            "p_effect": _effect_bucket(proposal_effect),
            "p_role": _role_bucket(proposal_role),
            "p_ov": _overlap(proposal_overlap),
            "p_rank": _rank(proposal_rank),
            "p_phase": phase_bucket(phase),
        }
        known = [field for field in _SIGNATURE_FIELDS if query[field] != "unknown"]
        masks = _signature_masks(known)
        family = str(task_family or "").replace("|", " ").strip().lower()
        host = str(site or "").replace("|", " ").strip().lower()
        levels = []
        if family and host:
            levels.append("family+site")
        if host:
            levels.append("site")
        if family:
            levels.append("family")
        levels.append("pooled")
        fallback = None
        for level in levels:
            for signature_level, enforced in masks:
                bucket: dict[str, list] = {}
                for cell in self.cells:
                    if not self._stratum_matches(cell, level, family, host):
                        continue
                    if any(cell[_SIGNATURE_INDEX[field]] != query[field] for field in enforced):
                        continue
                    arm = cell[_TRIAL_KEY_LEN - 1]
                    slot = bucket.setdefault(arm, [0] * TRIAL_COUNT_LEN)
                    for i, v in enumerate(cell[_TRIAL_KEY_LEN:]):
                        slot[i] += v
                if not bucket:
                    continue
                if "candidate" not in bucket or "control" not in bucket:
                    # A bucket holding only one arm cannot answer a *contrast*
                    # question — "is assigning B better than A" needs both
                    # sides. The arm stays visible in ``estimate()`` reporting;
                    # resolution simply keeps walking (and may return None).
                    continue
                contrast = {
                    **self._contrast(bucket, min_effect=min_effect, weights=weights),
                    "level": level,
                    "signature_level": signature_level,
                }
                if contrast.get("support_sufficient"):
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
        version = str(payload.get("version", "jev-trials/5"))
        if cells and len(cells[0]) == 15:
            # v4 evidence migrates in place; the collapsed scope is split and
            # the widened coordinates land as "unknown" wildcards.
            cells = _migrate_v4_trial_cells(cells)
            version = "jev-trials/5"
        if any(len(c) != TRIAL_CELL_LEN for c in cells):
            raise ValueError("Unsupported counterfactual-trials cell arity")
        # Coordinates are join-delimiter-free by construction at fit time;
        # normalize stored cells too so a hostile or legacy value can never
        # collide two distinct contexts under estimate()'s "|".join key.
        cells = tuple(
            tuple(str(v).replace("|", " ") for v in cell[:_TRIAL_KEY_LEN]) + cell[_TRIAL_KEY_LEN:]
            for cell in cells
        )
        return cls(
            cells=cells,
            duplicates=int(payload.get("duplicates", 0) or 0),
            version=version,
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
    B-type changed verified success by delta" — with hierarchical backoff
    over task family, site, and the bounded treatment signature, so a
    task-family stratum wins when it has its own support and unrelated tasks
    stop pooling into one tiny posterior.

    Two facts are kept separate, because conflating them was the v0.8.3
    defect: *having enough samples* (``support_sufficient``) is not *having
    established the effect* (``effect_status``). ``choose`` proposes only a
    ``beneficial`` divergence — the sequentially valid interval lies entirely
    above the practical threshold — and ``refuted`` is true only for a
    ``harmful`` one. A supported cell whose interval still crosses zero is
    ``unresolved``: it schedules more evidence, and it must never permanently
    suppress a potentially beneficial hypothesis or prematurely adopt one.

    What the estimate generalizes over is a *treatment signature* — kind,
    effect class, role, overlap, offered rank, workflow phase — not an action
    ID: "an action of B's coarse class, in this context, improved verified
    success" is the honest reading, and the signature coordinates are what
    resolution backoff and the ranking below operate on.

    Every prediction carries its provenance (``source: "randomized"``);
    observational and causal evidence are never silently mixed. Same advisory
    contract as the other learned layers: proposals are hypotheses, never
    evidence, never gate input, and anything this model prefers still has to
    pass the full authority plane and bound live canaries.
    """

    trials: CounterfactualTrials = CounterfactualTrials()
    # Per-task-family minimum worthwhile effect (absolute success delta).
    # Empty → ``CounterfactualTrials.MIN_EFFECT``. Statistical significance
    # alone is not enough: a family may demand a practically meaningful
    # improvement before a divergence counts as beneficial there.
    minimum_effects: tuple[tuple[str, float], ...] = ()
    version: str = "jev-causal/2"

    @classmethod
    def fit(cls, events: Iterable[dict]) -> "TrialChoiceModel":
        return cls(trials=CounterfactualTrials.fit(events))

    def min_effect_for(self, task_family: str | None = None) -> float:
        family = str(task_family or "").strip().lower()
        if family:
            for key, value in self.minimum_effects:
                if str(key).strip().lower() == family:
                    return max(0.0, float(value))
        return CounterfactualTrials.MIN_EFFECT

    def _estimate(
        self,
        *,
        proposal_kind: str,
        proposal_overlap=None,
        proposal_effect=None,
        proposal_role=None,
        proposal_rank=None,
        model_kind: str = "unknown",
        model_overlap=None,
        model_effect=None,
        model_role=None,
        model_rank=None,
        phase=None,
        task_family: str | None = None,
        site: str | None = None,
        min_effect: float | None = None,
    ) -> dict | None:
        return self.trials.resolve(
            task_family=task_family,
            site=site,
            model_kind=model_kind,
            model_overlap=model_overlap,
            model_effect=model_effect,
            model_role=model_role,
            model_rank=model_rank,
            proposal_kind=proposal_kind,
            proposal_overlap=proposal_overlap,
            proposal_effect=proposal_effect,
            proposal_role=proposal_role,
            proposal_rank=proposal_rank,
            phase=phase,
            min_effect=(self.min_effect_for(task_family) if min_effect is None else min_effect),
        )

    def choose(
        self,
        candidates: Iterable[dict],
        *,
        model_choice: dict | None = None,
        task_family: str | None = None,
        site: str | None = None,
        phase=None,
    ) -> dict | None:
        """The offered candidate whose arm *established* a beneficial effect.

        ``model_choice`` supplies the divergence premise — the action the
        decision model actually picked — because a trial only measures
        "assigning B when the model would have picked A". With no premise, or
        no candidate whose randomized evidence is ``beneficial``, abstain: a
        causal prior without settled evidence proposes nothing. A supported
        but ``unresolved`` cell is *not* a proposal — it is a hypothesis that
        needs more evidence — and a ``harmful`` one is refuted, never
        re-proposed.
        """
        if not model_choice:
            return None
        candidates = list(candidates)
        mc_kind = str(model_choice.get("kind") or "unknown")
        mc_overlap = model_choice.get("goal_overlap")
        mc_effect = model_choice.get("effect")
        mc_role = model_choice.get("role")
        mc_rank = next(
            (i for i, c in enumerate(candidates) if c.get("id") == model_choice.get("id")),
            None,
        )
        best = None
        for p_rank, candidate in enumerate(candidates):
            if candidate.get("id") == model_choice.get("id"):
                continue
            estimate = self._estimate(
                proposal_kind=str(candidate.get("kind") or "unknown"),
                proposal_overlap=candidate.get("goal_overlap"),
                proposal_effect=candidate.get("effect"),
                proposal_role=candidate.get("role"),
                proposal_rank=p_rank,
                model_kind=mc_kind,
                model_overlap=mc_overlap,
                model_effect=mc_effect,
                model_role=mc_role,
                model_rank=mc_rank,
                phase=phase,
                task_family=task_family,
                site=site,
            )
            if estimate is None or estimate.get("effect_status") != "beneficial":
                continue
            # Among established beneficial effects, prefer the strongest
            # point estimate — establishment is already gated by the
            # sequentially valid interval and the practical threshold.
            if best is None or estimate["delta"] > best[0]:
                best = (estimate["delta"], candidate, estimate)
        if best is None:
            return None
        delta, candidate, estimate = best
        control_p = estimate["control"]["p_success"] if "control" in estimate else 0.0
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
            "signature_level": estimate.get("signature_level"),
            "effect_status": estimate["effect_status"],
            "support_sufficient": estimate.get("support_sufficient"),
            "sequential_delta_ci": estimate.get("sequential_delta_ci"),
            "delta_ci": ci,
            # Provenance: this prediction comes only from randomized evidence.
            "source": "randomized",
        }

    def predict(
        self,
        *,
        kind: str,
        goal_overlap=0,
        rank=None,
        task_family: str | None = None,
        site: str | None = None,
        model_kind: str = "unknown",
        model_overlap=None,
        model_effect=None,
        model_role=None,
        model_rank=None,
        proposal_effect=None,
        proposal_role=None,
        phase=None,
    ) -> dict:
        """The candidate-side implied success probability for one action.

        Resolution follows the same hierarchy ``choose`` uses; the estimate
        is the intention-to-treat effect of *assigning* an action of this
        treatment signature in this context. Returns ``p_progress: None``
        when no evidence exists — never a fabricated baseline. ``source`` is
        always ``"randomized"``: this model never reads the observational
        trace.
        """
        estimate = self._estimate(
            proposal_kind=str(kind or "unknown"),
            proposal_overlap=goal_overlap,
            proposal_effect=proposal_effect,
            proposal_role=proposal_role,
            proposal_rank=rank,
            model_kind=model_kind,
            model_overlap=model_overlap,
            model_effect=model_effect,
            model_role=model_role,
            model_rank=model_rank,
            phase=phase,
            task_family=task_family,
            site=site,
        )
        if estimate is None:
            return {
                "p_progress": None,
                "n": 0,
                "level": None,
                "signature_level": None,
                "effect_status": "insufficient_data",
                "support_sufficient": False,
                "source": "randomized",
            }
        delta = estimate.get("delta")
        control = estimate.get("control")
        if delta is None or control is None:
            # Defensive: a resolved estimate always carries both arms, but a
            # one-armed bucket must never crash a caller — it simply predicts
            # nothing.
            return {
                "p_progress": None,
                "n": int((estimate.get("candidate") or {}).get("trials", 0)),
                "level": estimate.get("level"),
                "signature_level": estimate.get("signature_level"),
                "effect_status": estimate.get("effect_status") or "insufficient_data",
                "support_sufficient": bool(estimate.get("support_sufficient")),
                "source": "randomized",
            }
        control_p = control["p_success"]
        return {
            "p_progress": min(max(control_p + delta, 0.0), 1.0),
            "delta": delta,
            "n": estimate["control"]["trials"] + estimate["candidate"]["trials"],
            "level": estimate.get("level"),
            "signature_level": estimate.get("signature_level"),
            "effect_status": estimate.get("effect_status"),
            "support_sufficient": estimate.get("support_sufficient"),
            "source": "randomized",
        }

    def rank(
        self,
        candidates: Iterable[dict],
        *,
        model_choice: dict,
        task_family: str | None = None,
        site: str | None = None,
        phase=None,
    ) -> list[dict]:
        """Bounded causal ranking of offered candidates, with provenance.

        Established-beneficial divergences rank first, unresolved ones next
        (more evidence needed), and harmful-established ones are omitted
        entirely. This is the candidate-prioritization channel — the
        ``CausalChoicePolicy`` consumes it — never an execution authority:
        whatever it prefers still passes effect classification, payload
        review, approvals, and the browser guards.
        """
        candidates = list(candidates)
        mc_kind = str(model_choice.get("kind") or "unknown")
        mc_overlap = model_choice.get("goal_overlap")
        mc_effect = model_choice.get("effect")
        mc_role = model_choice.get("role")
        mc_rank = next(
            (i for i, c in enumerate(candidates) if c.get("id") == model_choice.get("id")),
            None,
        )
        tiers = {"beneficial": 0, "unresolved": 1, "insufficient_data": 2}
        entries = []
        for p_rank, candidate in enumerate(candidates):
            if candidate.get("id") == model_choice.get("id"):
                continue
            estimate = self._estimate(
                proposal_kind=str(candidate.get("kind") or "unknown"),
                proposal_overlap=candidate.get("goal_overlap"),
                proposal_effect=candidate.get("effect"),
                proposal_role=candidate.get("role"),
                proposal_rank=p_rank,
                model_kind=mc_kind,
                model_overlap=mc_overlap,
                model_effect=mc_effect,
                model_role=mc_role,
                model_rank=mc_rank,
                phase=phase,
                task_family=task_family,
                site=site,
            )
            if estimate is None:
                continue
            status = str(estimate.get("effect_status") or "insufficient_data")
            if status == "harmful":
                continue
            entries.append({
                "id": candidate.get("id"),
                "kind": candidate.get("kind"),
                "expected_delta": estimate.get("delta"),
                "utility_delta": estimate.get("utility_delta"),
                "effect_status": status,
                "support_sufficient": estimate.get("support_sufficient"),
                "trial_level": estimate.get("level"),
                "signature_level": estimate.get("signature_level"),
                "source": "randomized",
                "_tier": tiers.get(status, 3),
            })
        entries.sort(key=lambda e: (e["_tier"], -(e["expected_delta"] or 0.0), str(e["id"])))
        for entry in entries:
            entry.pop("_tier", None)
        return entries

    def refuted(
        self,
        *,
        kind: str,
        goal_overlap=None,
        model_kind: str = "unknown",
        model_overlap=None,
        model_effect=None,
        model_role=None,
        model_rank=None,
        proposal_effect=None,
        proposal_role=None,
        proposal_rank=None,
        phase=None,
        task_family: str | None = None,
        site: str | None = None,
    ) -> bool:
        """True only when randomized evidence *established* this divergence as
        harmful at a supported stratum — the hypothesis was tested and failed,
        so the proposal layer should not stamp it again.

        An inconclusive interval must never suppress a potentially beneficial
        hypothesis: ``unresolved`` is not ``refuted``.
        """
        estimate = self._estimate(
            proposal_kind=kind,
            proposal_overlap=goal_overlap,
            proposal_effect=proposal_effect,
            proposal_role=proposal_role,
            proposal_rank=proposal_rank,
            model_kind=model_kind,
            model_overlap=model_overlap,
            model_effect=model_effect,
            model_role=model_role,
            model_rank=model_rank,
            phase=phase,
            task_family=task_family,
            site=site,
        )
        return bool(estimate is not None and estimate.get("effect_status") == "harmful")

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
        minimum_effects = payload.get("minimum_effects") or ()
        return cls(
            trials=CounterfactualTrials.from_dict(payload.get("trials") or {}),
            minimum_effects=tuple((str(k), float(v)) for k, v in minimum_effects),
            version=payload.get("version", "jev-causal/1"),
        )


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
    - **remaining uncertainty** — the width of the sequentially valid
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
                    model_role=historical.get("role"),
                    model_rank=historical.get("offered_rank"),
                    proposal_kind=str(
                        proposal.get("kind")
                        or hypothesis.get("proposal_kind")
                        or "unknown"
                    ),
                    proposal_overlap=proposal.get(
                        "goal_overlap", hypothesis.get("proposal_overlap")
                    ),
                    proposal_role=proposal.get("role"),
                    proposal_rank=hypothesis.get("proposal_offered_rank"),
                    phase=hypothesis.get("step"),
                )
            status = (estimate or {}).get("effect_status")
            if status in {"beneficial", "harmful"}:
                continue
            if estimate is not None and "candidate" in estimate and "control" in estimate:
                sequential = estimate.get("sequential_delta_ci")
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


@dataclass(frozen=True)
class CausalChoicePolicy:
    """Bounded combination of the base decision score with learned priors.

    Modes are *operator-gated* — the policy never promotes its own mode, and
    a mode change is a configuration decision, not a learned one:

    - ``shadow`` — produce the ranking as an annotation only; nothing it
      prefers can influence execution.
    - ``canary`` — its top proposal may be used as the *experimental*
      candidate under the existing randomized assignment (recorded
      propensity, real authority plane, at most one trial per run).
    - ``active`` — a *beneficial*, randomized-established proposal may be
      used as the executed choice, still bounded to the offered catalogue and
      still passing effect classification, payload review, approvals, and the
      browser guards.

    Every entry carries provenance: ``randomized`` (supported randomized
    evidence at a specific stratum), ``pooled_randomized`` (randomized
    evidence that had to back off to the pooled stratum), or
    ``observational`` (trajectory association only). The two evidence
    channels are never silently mixed — a combined score reports which
    channel each component came from.
    """

    mode: str = "shadow"
    trial_model: TrialChoiceModel | None = None
    choice_model: ChoiceModel | None = None
    w_causal: float = 1.0
    w_observational: float = 0.5
    min_causal_delta: float = 0.0

    MODES = ("shadow", "canary", "active")

    def __post_init__(self):
        if self.mode not in self.MODES:
            raise ValueError(f"CausalChoicePolicy mode must be one of {self.MODES}")

    def rank(
        self,
        candidates: Iterable[dict],
        *,
        model_choice: dict,
        task_family: str | None = None,
        site: str | None = None,
        phase=None,
    ) -> dict:
        candidates = list(candidates)
        causal = {}
        if self.trial_model is not None:
            causal = {
                entry["id"]: entry
                for entry in self.trial_model.rank(
                    candidates,
                    model_choice=model_choice,
                    task_family=task_family,
                    site=site,
                    phase=phase,
                )
            }
        entries = []
        for index, candidate in enumerate(candidates):
            if candidate.get("id") == model_choice.get("id"):
                continue
            causal_entry = causal.get(candidate.get("id"))
            observational = None
            if self.choice_model is not None:
                observational = self.choice_model.predict(
                    kind=str(candidate.get("kind") or "unknown"),
                    goal_overlap=int(candidate.get("goal_overlap", 0) or 0),
                    rank=index,
                    task_family=task_family,
                )
            supported = bool(causal_entry and causal_entry.get("support_sufficient"))
            if supported:
                source = (
                    "pooled_randomized"
                    if causal_entry.get("trial_level") == "pooled"
                    else "randomized"
                )
            elif causal_entry is not None:
                source = "randomized"
            elif observational is not None and observational.get("confident"):
                source = "observational"
            else:
                source = "observational"
            score = 0.0
            if causal_entry is not None:
                score += self.w_causal * float(causal_entry.get("expected_delta") or 0.0)
            if observational is not None and observational.get("p_progress") is not None:
                score += self.w_observational * (float(observational["p_progress"]) - 0.5)
            entries.append({
                "id": candidate.get("id"),
                "kind": candidate.get("kind"),
                "score": score,
                "source": source,
                "causal": causal_entry,
                "observational": observational,
                "executable": bool(
                    self.mode == "active"
                    and causal_entry is not None
                    and causal_entry.get("effect_status") == "beneficial"
                    and float(causal_entry.get("expected_delta") or 0.0) > self.min_causal_delta
                ),
            })
        tiers = {"beneficial": 0, "unresolved": 1, "insufficient_data": 2}
        entries.sort(
            key=lambda entry: (
                tiers.get((entry["causal"] or {}).get("effect_status"), 3),
                -entry["score"],
                str(entry["id"]),
            )
        )
        return {
            "mode": self.mode,
            "shadow": self.mode == "shadow",
            "proposals": entries,
            "abstain": not entries,
        }

    def proposal(
        self,
        candidates: Iterable[dict],
        *,
        model_choice: dict,
        task_family: str | None = None,
        site: str | None = None,
        phase=None,
    ) -> dict | None:
        """The top entry when this mode may act on it, else ``None``.

        ``canary`` returns the top entry for the randomized-assignment path;
        ``active`` returns it only when it is a randomized-established
        beneficial proposal; ``shadow`` always returns ``None`` — shadow mode
        observes, it does not act.
        """
        ranking = self.rank(
            candidates, model_choice=model_choice, task_family=task_family,
            site=site, phase=phase,
        )
        if not ranking["proposals"] or self.mode == "shadow":
            return None
        top = ranking["proposals"][0]
        if self.mode == "active" and not top["executable"]:
            return None
        causal = top.get("causal") or {}
        observational = top.get("observational") or {}
        p_progress = observational.get("p_progress")
        if causal.get("expected_delta") is not None and causal.get("support_sufficient"):
            p_progress = min(
                max(0.5 + float(causal["expected_delta"]), 0.0), 1.0
            )
        return {
            "id": top["id"],
            "kind": top["kind"],
            "p_progress": p_progress,
            "uncertainty": observational.get("uncertainty"),
            "confident": bool(causal.get("support_sufficient") or observational.get("confident")),
            "expected_delta": causal.get("expected_delta"),
            "effect_status": causal.get("effect_status"),
            "trial_level": causal.get("trial_level"),
            "signature_level": causal.get("signature_level"),
            "source": top["source"],
            "mode": self.mode,
            "executable": top["executable"],
        }
