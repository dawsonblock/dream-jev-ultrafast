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
    """Yield ``(transition_event, run_policy, terminal, verified_done, outcome)``.

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
        transitions = [e for e in run_events if e.get("event") == "transition"]
        for index, event in enumerate(transitions):
            terminal = index == len(transitions) - 1 and final is not None
            verified_done = bool(terminal and run_verified)
            yield event, policy, terminal, verified_done, outcome
    for event in loose:
        if event.get("event") == "transition":
            yield event, None, False, False, "legacy"


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


def _fit_cells(events: Iterable[dict], label) -> tuple[dict, dict, int, int]:
    """Count outcomes per ``(kind, overlap_bucket, offered_rank_bucket)`` cell.

    ``label(event, terminal, verified_done, outcome)`` decides what counts as
    a positive observation — or returns ``None`` to censor the sample entirely
    (an unmeasured outcome is missing data, not a negative example). The rank
    coordinate is always the offered-catalogue rank so training and
    counterfactual replay share one basis.
    """
    cell_counts: dict[tuple[str, str, str], list[int]] = {}
    kind_counts: dict[str, list[int]] = {}
    positive_total = 0
    samples = 0
    for event, policy, terminal, verified_done, outcome in _transitions(events):
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
        kind_counts.setdefault(kind, [0, 0])
        kind_counts[kind][0] += 1
        kind_counts[kind][1] += positive
        positive_total += positive
        samples += 1
    return cell_counts, kind_counts, positive_total, samples


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
        cell_counts, kind_counts, changed_total, samples = _fit_cells(
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
    kind_totals: tuple[tuple[str, int, int], ...] = ()
    global_positive: int = 0
    version: str = "jev-choice/4"

    MIN_CONFIDENT = 4
    EXPLORE_BONUS = 0.5

    @classmethod
    def fit(cls, events: Iterable[dict]) -> "ChoiceModel":
        # Trajectory-success association (v0.6.2 semantics): the label is the
        # run's verified final outcome for *every* step, not the step's page
        # change. Censored runs (aborted/interrupted/unverifiable) drop out
        # instead of fitting as negatives; transitions without run context
        # keep the legacy non-positive label.
        cell_counts, kind_counts, positive_total, samples = _fit_cells(
            events,
            lambda event, terminal, verified_done, outcome: (
                None if outcome == "censored" else outcome == "success"
            ),
        )
        return cls(
            samples=samples,
            cells=tuple(
                sorted((*cell, n, changed) for cell, (n, changed) in cell_counts.items())
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

    def predict(self, *, kind: str, goal_overlap: int = 0, rank: int | None = None) -> dict:
        cell_key = (str(kind), overlap_bucket(goal_overlap), rank_bucket(rank))
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

    def choose(self, candidates: Iterable[dict]) -> dict | None:
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


def _trial_context(meta: dict, transition: dict | None) -> str:
    """The cell a trial belongs to: the *model choice's* features.

    Keying on the executed action would put the two arms of one assignment in
    different cells whenever the proposal's overlap differs from the model
    choice's — exactly when the prior diverged for a reason. Assignment
    events carry the pre-treatment context directly; the transition's
    candidate list is only a fallback for older records that lack it.
    """
    kind = str(meta.get("model_choice_kind") or "unknown")
    if kind == "unknown" and transition is not None:
        kind = str((transition.get("selected") or {}).get("kind") or "unknown")
    overlap_raw = meta.get("model_choice_overlap")
    if overlap_raw is not None:
        try:
            return f"{kind}|{overlap_bucket(int(overlap_raw))}"
        except (TypeError, ValueError):
            pass
    if transition is not None:
        model_choice = next(
            (
                c
                for c in transition.get("candidates") or ()
                if c.get("id") == meta.get("model_choice_id")
            ),
            None,
        )
        if model_choice is not None:
            return f"{kind}|{overlap_bucket(int(model_choice.get('goal_overlap', 0) or 0))}"
    return f"{kind}|unknown"


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

    # (context, arm, assigned, analyzed, executed, censored,
    #  w_success, w_sum, w_sq, w_page, w_exec_sum)
    cells: tuple[tuple[str, str, int, int, int, int, float, float, float, float, float], ...] = ()
    version: str = "jev-trials/3"

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
                (_trial_context(meta, transition), arm),
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
                    (context, arm, *counts)
                    for (context, arm), counts in acc.items()
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
        worth — and ``p_success_ci``/``delta_ci`` are Wilson/normal intervals
        on that effective support, so thin evidence cannot masquerade as a
        confident estimate.
        """
        arms: dict[str, dict] = {}
        for ctx, arm, assigned, analyzed, executed, censored, ws, wsum, wsq, wpage, wexec in self.cells:
            if context is not None and ctx != context:
                continue
            entry = arms.setdefault(ctx, {})
            ess = (wsum * wsum / wsq) if wsq else 0.0
            p_success = (ws / wsum) if wsum else 0.0
            entry[arm] = {
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
                "reliable": ess >= self.MIN_ESS - 1e-9,
            }
        for entry in arms.values():
            if "candidate" in entry and "control" in entry:
                candidate, control = entry["candidate"], entry["control"]
                delta = candidate["p_success"] - control["p_success"]
                entry["delta"] = delta
                if candidate["ess"] and control["ess"]:
                    pc, pk = candidate["p_success"], control["p_success"]
                    se = math.sqrt(
                        pc * (1.0 - pc) / candidate["ess"]
                        + pk * (1.0 - pk) / control["ess"]
                    )
                    entry["delta_ci"] = [delta - 1.96 * se, delta + 1.96 * se]
                else:
                    entry["delta_ci"] = None
                # The contrast is only as trustworthy as its weaker arm.
                entry["delta_reliable"] = candidate["reliable"] and control["reliable"]
        return arms

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
        if any(len(c) != 11 for c in cells):
            raise ValueError("Unsupported counterfactual-trials cell arity")
        return cls(
            cells=cells,
            version=payload.get("version", "jev-trials/3"),
        )
