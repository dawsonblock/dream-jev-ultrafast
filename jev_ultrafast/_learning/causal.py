"""_learning.causal — extracted from jev_ultrafast.dreamlearn."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from .censoring import _run_outcome
from .common import _stable_hash
from .confidence import (
    _arm_confidence_bounds,
    _arm_ws,
    _arm_wsq,
    _arm_wsum,
    _newcombe,
    _sequential_alpha,
    _wilson_interval,
)
from .signatures import (
    _REASON_INDEX,
    _SIGNATURE_FIELDS,
    _SIGNATURE_INDEX,
    TERMINATION_REASONS,
    _effect_bucket,
    _role_bucket,
    _signature_masks,
    _termination_reason,
    _trial_context,
    overlap_bucket,
    phase_bucket,
    rank_bucket,
)

if TYPE_CHECKING:
    from typing import Iterable

__all__ = [
    'CounterfactualTrials',
    'OUTCOME_VECTOR_SCHEMA',
    'TRIAL_CELL_LEN',
    'TRIAL_COUNT_LEN',
    'TrialChoiceModel',
    'UtilityWeights',
    '_COST_BASE',
    '_COUNT_REASON_BASE',
    '_REASON_BASE',
    '_TRIAL_COSTS',
    '_TRIAL_KEY_LEN',
    '_TRIAL_STATS',
    '_V4_CELL_LEN',
    '_V5_CELL_LEN',
    '_V6_CELL_LEN',
    '_candidate_probability',
    '_migrate_v4_trial_cells',
    '_migrate_v5_trial_cells',
    '_migrate_v6_trial_cells',
]


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


_TRIAL_KEY_LEN = 15


_TRIAL_STATS = 10


_TRIAL_COSTS = 3


_COST_BASE = _TRIAL_KEY_LEN + _TRIAL_STATS


_REASON_BASE = _COST_BASE + _TRIAL_COSTS


_COUNT_REASON_BASE = _REASON_BASE - _TRIAL_KEY_LEN


# jev-trials/8 outcome-vector block: appended *after* the reason tail so every
# v7 counter index is unchanged. Six weighted counters per analyzed
# assignment — measured mass (how much of the analyzed pool actually carried
# outcome fields), verified transitions, steps used, recoveries, authority
# touches, and guard failures.
_COUNT_OUTCOME_BASE = _COUNT_REASON_BASE + len(TERMINATION_REASONS)


_OUTCOME_STATS = 6


TRIAL_COUNT_LEN = _COUNT_OUTCOME_BASE + _OUTCOME_STATS


TRIAL_CELL_LEN = _TRIAL_KEY_LEN + TRIAL_COUNT_LEN


_V4_CELL_LEN = 15


_V5_CELL_LEN = 35


_V6_CELL_LEN = 36


_V7_CELL_LEN = 37


OUTCOME_VECTOR_SCHEMA = "jev-outcome-vector/1"


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


def _migrate_v5_trial_cells(cells) -> tuple[tuple, ...]:
    """v5 cells → v6: the reasons tail gained ``indeterminate_execution``.

    v5 stored eight reason counters ending
    ``[…, recorder_shutdown, unverified_claim, unknown_abort]``; v6 inserts
    ``indeterminate_execution`` between ``recorder_shutdown`` and
    ``unverified_claim``. Recorded verdicts carry over verbatim — the new
    slot starts at zero because v5 could never record the state. Detected by
    cell arity, not the version label.
    """
    insert_at = _V6_CELL_LEN - len(TERMINATION_REASONS) + _REASON_INDEX["indeterminate_execution"]
    return tuple(
        tuple(cell[:insert_at]) + (0,) + tuple(cell[insert_at:])
        for cell in cells
    )


def _migrate_v6_trial_cells(cells) -> tuple[tuple, ...]:
    """v6 cells → v7: the stats block gained ``w_censored``.

    v7 records the IPW weight mass of censored assignments so Manski-style
    worst/best-case endpoint bounds can be computed at estimate time. A v6
    cell cannot recover that weight — it migrates as ``0.0`` (never an
    invented value) and ``_arm_entry`` imputes a conservative fallback from
    the analyzed-pool mean weight, flagging the bound as imputed.
    """
    insert_at = _TRIAL_KEY_LEN + 9
    return tuple(
        tuple(cell[:insert_at]) + (0.0,) + tuple(cell[insert_at:])
        for cell in cells
    )


def _migrate_v7_trial_cells(cells) -> tuple[tuple, ...]:
    """v7 cells → v8: the outcome-vector block appended after the reason tail.

    v8 records per-assignment weighted counters for the structured
    ``jev-outcome-vector/1`` secondary and safety endpoints. A v7 cell never
    measured them — every migrated slot is ``0.0`` (never an invented
    value), and because the first counter is the *measured* mass itself, a
    migrated arm reports ``coverage: 0`` and ``None``-valued secondary and
    safety rates rather than confident zeros.
    """
    return tuple(tuple(cell) + (0.0,) * _OUTCOME_STATS for cell in cells)


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
      treatment effect? ``beneficial`` requires the confidence sequence to
      lie entirely above ``min_effect`` *and* the censoring bounds to confirm
      the margin could not be erased by the missing outcomes; ``harmful`` is
      symmetric; a sequence that crosses the threshold leaves the hypothesis
      ``unresolved`` and schedules more evidence. "Eight observations exist"
      is not "we know the intervention helps".

    ``delta_ci`` is the fixed-sample Newcombe-Wilson interval for reporting;
    ``delta_cs`` is the interval establishment uses — a **confidence
    sequence** for the candidate−control difference, valid at every look
    simultaneously. It is built by spending ``SEQUENTIAL_ALPHA`` over all
    integer looks (``_sequential_alpha``) on *exact* per-look concentration
    bounds: the Chernoff/KL bound on a Bernoulli mean when the IPW weights
    are uniform, Hoeffding's weighted bound when they are not. Unlike the
    previous α-spent Newcombe approximation, the composition is formally
    anytime-valid, and ``SEQUENTIAL_ALPHA`` is Bonferroni-split across the
    simultaneously tracked hypothesis family (``hypothesis_count``).

    Censoring is a first-class causal concern, not bookkeeping. Three
    mechanisms handle it:

    - per-arm ``censor_rate`` and the ``terminations`` reason breakdown are
      reported, and ``censoring_imbalance`` refuses to establish when either
      arm is censored at an extreme rate or the arms are censored at sharply
      different rates;
    - ``p_success_bounds`` per arm and ``delta_bounds`` for the contrast are
      Manski worst/best-case bounds: every censored unit is re-counted as a
      success (upper) or a failure (lower) — no ignorable-censoring
      assumption at all;
    - establishment requires the bounds to *agree* with the sequence: if
      ``delta_cs`` excludes the threshold but ``delta_bounds`` still crosses
      it, the missing outcomes could account for the apparent effect and the
      hypothesis stays ``unresolved`` (``censoring_bounds_cross_threshold``).

    The taxonomy separates an operator closing the session (unrelated to
    treatment) from a crash/timeout that the treatment itself may have
    caused, so differential censoring stays visible.

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
    version: str = "jev-trials/8"

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
            tokens, and approval count, plus the run-level outcome-vector
            signals (steps used, guard-failure markers, authority touches),
            used only for the multi-objective utility annotation and the
            structured secondary/safety report.
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
                # The censored units' weight mass is kept so estimate-time
                # worst/best-case endpoint bounds can be computed — censored
                # outcomes are missing data, but their *mass* is not.
                cell[9] += weight
                cell[_COUNT_REASON_BASE + _REASON_INDEX[reason]] += 1
                return
            cell[1] += 1  # analyzed (ITT denominator)
            cell[4] += float(outcome == "success") * weight
            cell[5] += weight
            cell[6] += weight * weight
            latency, tokens, approvals, steps, guard_failures, authority_touches = run_costs
            cell[10] += latency * weight
            cell[11] += tokens * weight
            cell[12] += approvals * weight
            # jev-outcome-vector/1 counters (appended block; indices stable
            # across the v7→v8 migration because they sit past the reason
            # tail). ``measured`` records that this analyzed run actually
            # carried outcome fields — a migrated v7 arm reports zero mass
            # here and surfaces ``coverage: 0`` instead of fabricated rates.
            cell[_COUNT_OUTCOME_BASE] += weight
            # Verified state transition under ITT: the assigned step executed
            # and the page moved. An assigned-but-unexecuted arm contributes
            # 0 — its trajectory is the treatment being measured.
            cell[_COUNT_OUTCOME_BASE + 1] += float(bool(page_changed)) * weight
            cell[_COUNT_OUTCOME_BASE + 2] += steps * weight
            # Recovery: the run recorded guard-failure markers yet still
            # finished verifier-confirmed done.
            cell[_COUNT_OUTCOME_BASE + 3] += (
                float(bool(guard_failures and outcome == "success")) * weight
            )
            cell[_COUNT_OUTCOME_BASE + 4] += authority_touches * weight
            cell[_COUNT_OUTCOME_BASE + 5] += guard_failures * weight

        for run_id, run_events in runs.items():
            run_events.sort(key=lambda e: (e.get("sequence", 0), e.get("recorded_at_ms", 0)))
            final = next(
                (e for e in reversed(run_events) if e.get("event") == "run_finished"), None
            )
            outcome = _run_outcome(final)
            reason = _termination_reason(final)
            transitions_all = [e for e in run_events if e.get("event") == "transition"]
            run_costs = (
                sum(max(0, int(e.get("latency_ms", 0))) for e in transitions_all),
                sum(max(0, int(e.get("tokens", 0))) for e in transitions_all),
                sum(1 for e in run_events if e.get("event") == "approval_required"),
                # Outcome-vector run signals: steps used, transitions carrying
                # guard-failure markers (``stale_or_failure``), and transitions
                # that consumed authority-plane approvals (``risk_events``).
                len(transitions_all),
                sum(
                    1 for e in transitions_all
                    if int(e.get("stale_or_failure") or 0) > 0
                ),
                sum(
                    1 for e in transitions_all
                    if int(e.get("risk_events") or 0) > 0
                ),
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
                record(meta, "censored", "unknown_abort", None, None, (0, 0, 0, 0, 0, 0))
            elif event.get("event") == "transition" and isinstance(event.get("experiment"), dict):
                record(
                    event["experiment"], "censored", "unknown_abort",
                    event.get("page_changed"), event, (0, 0, 0, 0, 0, 0),
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
        # Multiplicity is over every tracked hypothesis, not just the ones
        # this report happens to print — a filtered view does not shrink the
        # family that is actually being watched.
        hypothesis_count = len({tuple(cell[:_TRIAL_KEY_LEN - 1]) for cell in self.cells})
        for ctx, bucket in grouped.items():
            arms["|".join(ctx)] = self._contrast(bucket, hypothesis_count=hypothesis_count)
        return arms

    @staticmethod
    def _arm_entry(counts: list) -> dict:
        """Per-arm report from the sufficient statistics (see cell layout)."""
        (
            assigned, analyzed, executed, censored,
            ws, wsum, wsq, wpage, wexec, wcensored,
        ) = counts[:10]
        w_latency, w_tokens, w_approvals = counts[10:13]
        terminations = {
            reason: int(counts[13 + index])
            for index, reason in enumerate(TERMINATION_REASONS)
            if counts[13 + index]
        }
        (
            w_outcome_measured, w_vtrans, w_steps,
            w_recovery, w_authority, w_guard,
        ) = counts[_COUNT_OUTCOME_BASE:_COUNT_OUTCOME_BASE + _OUTCOME_STATS]
        ess = (wsum * wsum / wsq) if wsq else 0.0
        p_success = (ws / wsum) if wsum else 0.0
        # Cells migrated from jev-trials/6 never recorded the censored mass's
        # weight. Impute it conservatively — each censored unit counts at
        # least one full observation, plus the analyzed pool's mean weight
        # when that is heavier — and say so on the report.
        wcensored_imputed = censored > 0 and wcensored <= 0.0
        wcensored_eff = wcensored
        if wcensored_imputed:
            mean_weight = (wsum / analyzed) if analyzed else 0.0
            wcensored_eff = censored * max(1.0, mean_weight)
        total = wsum + wcensored_eff
        # Outcome vector: structured secondary/cost/safety endpoints under the
        # same ITT weighting as the primary. ``coverage`` is the weighted
        # share of the analyzed pool whose runs carried the fields at all —
        # migrated v7 cells report 0 and every secondary/safety rate is None
        # rather than a fabricated zero.
        outcome_measured = bool(w_outcome_measured)

        def _rate(x):
            return (x / wsum) if wsum else None

        bounds_lo = ws / total if total else 0.0
        bounds_hi = (ws + wcensored_eff) / total if total else 1.0

        def _binary(x):
            p = _rate(x)
            return {
                "p": p if outcome_measured else None,
                "ci": (
                    list(_wilson_interval(p, ess))
                    if outcome_measured and p is not None else None
                ),
            }

        outcome_vector = {
            "schema": OUTCOME_VECTOR_SCHEMA,
            "coverage": (w_outcome_measured / wsum) if wsum else 0.0,
            # The primary endpoint is restated here for readers — promotion
            # authority still runs through the top-level p_success path and
            # its confidence sequence; this block cannot outrank it.
            "primary": {
                "verified_success": p_success,
                "ci": list(_wilson_interval(p_success, ess)),
                "bounds": [bounds_lo, bounds_hi],
            },
            "secondary": {
                "verified_transition": _binary(w_vtrans),
                "recovery": _binary(w_recovery),
                "steps": {
                    "mean": _rate(w_steps) if outcome_measured else None,
                },
            },
            "cost": {
                "latency_ms": (w_latency / wsum) if wsum else 0.0,
                "tokens": (w_tokens / wsum) if wsum else 0.0,
                "approvals": (w_approvals / wsum) if wsum else 0.0,
            },
            # Safety measurements are constraints, not utility: a positive
            # mass here can only ever veto or flag, never be bought off by
            # speed, cost, or success elsewhere.
            "safety": {
                "authority_touches": (
                    _rate(w_authority) if outcome_measured else None
                ),
                "guard_failures": (
                    _rate(w_guard) if outcome_measured else None
                ),
                "indeterminate_rate": (
                    (int(counts[_COUNT_REASON_BASE + _REASON_INDEX["indeterminate_execution"]])
                     / assigned) if assigned else None
                ),
            },
        }
        return {
            # Primary endpoint: verified run success under ITT.
            # ``p_page_changed`` is executed-trial telemetry — never a
            # treatment effect.
            "p_success": p_success,
            "p_success_ci": list(_wilson_interval(p_success, ess)),
            # Manski bounds on the endpoint under arbitrary censored
            # outcomes: every censored unit could have succeeded (best case)
            # or failed (worst case). No ignorable-censoring assumption — the
            # honest range is exactly what the missing outcomes could do.
            "p_success_bounds": [ws / total if total else 0.0,
                                 (ws + wcensored_eff) / total if total else 1.0],
            "w_censored": wcensored_eff,
            "w_censored_imputed": wcensored_imputed,
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
            "outcome_vector": outcome_vector,
        }

    @classmethod
    def _contrast(
        cls,
        bucket: dict[str, list],
        *,
        min_effect=None,
        weights=None,
        hypothesis_count: int = 1,
    ) -> dict:
        """Per-arm estimates plus the candidate−control contrast.

        Separates *support* from *effect certainty*: ``support_sufficient`` is
        the weighted-sample floor, ``effect_status`` is the sign actually
        established by the confidence sequence. A censoring-rate imbalance
        blocks establishment outright — the surviving subset of an arm
        censored far more than the other is a biased sample, not a smaller
        unbiased one — and the Manski ``delta_bounds`` veto catches the
        subtler case where balanced censoring could still flip the sign.

        ``hypothesis_count`` is the number of simultaneously tracked
        divergences; the family's ``SEQUENTIAL_ALPHA`` is Bonferroni-split
        across them before the per-look spending is applied, so no fleet of
        parallel hypotheses can inflate the false-establishment rate.
        """
        entry = {arm: cls._arm_entry(counts) for arm, counts in bucket.items()}
        if "candidate" not in entry or "control" not in entry:
            return entry
        candidate, control = entry["candidate"], entry["control"]
        delta = candidate["p_success"] - control["p_success"]
        entry["delta"] = delta
        # Worst/best-case difference under arbitrary censored outcomes —
        # the honest range the unobserved endpoints could force.
        entry["delta_bounds"] = [
            candidate["p_success_bounds"][0] - control["p_success_bounds"][1],
            candidate["p_success_bounds"][1] - control["p_success_bounds"][0],
        ]
        k = max(1, int(hypothesis_count))
        entry["hypothesis_count"] = k
        # The family's error budget is split over the concurrent hypotheses
        # (Bonferroni); each hypothesis then spends its share over looks.
        family_alpha = cls.SEQUENTIAL_ALPHA / k
        if candidate["ess"] and control["ess"]:
            entry["delta_ci"] = _newcombe(candidate, control)
            # Establishment uses the confidence sequence: the fixed-sample
            # interval is for reporting, not for deciding that an experiment
            # is settled after being peeked at every batch.
            looks = max(1, candidate["trials"] + control["trials"])
            alpha = _sequential_alpha(looks, base=family_alpha)
            entry["sequential_alpha"] = alpha
            # Per-look contrast budget is alpha; each arm gets half, and each
            # arm's two-sided bound splits that half across its two tails —
            # per-tail probability e^{-tau} = alpha/4.
            tau = math.log(4.0 / alpha) if alpha > 0 else math.inf
            cand_lo, cand_hi = _arm_confidence_bounds(
                candidate["trials"], _arm_wsum(bucket["candidate"]),
                _arm_wsq(bucket["candidate"]), _arm_ws(bucket["candidate"]), tau,
            )
            ctrl_lo, ctrl_hi = _arm_confidence_bounds(
                control["trials"], _arm_wsum(bucket["control"]),
                _arm_wsq(bucket["control"]), _arm_ws(bucket["control"]), tau,
            )
            entry["delta_cs"] = [cand_lo - ctrl_hi, cand_hi - ctrl_lo]
        else:
            entry["delta_ci"] = None
            entry["sequential_alpha"] = None
            entry["delta_cs"] = None
        support = bool(candidate["support_sufficient"] and control["support_sufficient"])
        entry["support_sufficient"] = support
        imbalance = (
            max(candidate["censor_rate"], control["censor_rate"]) > cls.CENSOR_RATE_MAX
            or abs(candidate["censor_rate"] - control["censor_rate"]) > cls.CENSOR_RATE_GAP
        )
        entry["censoring_imbalance"] = imbalance
        threshold = cls.MIN_EFFECT if min_effect is None else max(0.0, float(min_effect))
        entry["min_effect"] = threshold
        cs = entry["delta_cs"]
        bounds = entry["delta_bounds"]
        if not support or cs is None:
            entry["effect_status"] = "insufficient_data"
            entry["unresolved_reason"] = None
        elif imbalance:
            entry["effect_status"] = "unresolved"
            entry["unresolved_reason"] = "censoring_imbalance"
        elif cs[0] > threshold:
            # The sequence says the effect is positive — but only if the
            # censored outcomes cannot plausibly erase the whole margin is
            # that finding allowed to stand. A bound crossing the threshold
            # means informative censoring could account for the result.
            if bounds[0] > threshold:
                entry["effect_status"] = "beneficial"
                entry["unresolved_reason"] = None
            else:
                entry["effect_status"] = "unresolved"
                entry["unresolved_reason"] = "censoring_bounds_cross_threshold"
        elif cs[1] < -threshold:
            if bounds[1] < -threshold:
                entry["effect_status"] = "harmful"
                entry["unresolved_reason"] = None
            else:
                entry["effect_status"] = "unresolved"
                entry["unresolved_reason"] = "censoring_bounds_cross_threshold"
        else:
            entry["effect_status"] = "unresolved"
            entry["unresolved_reason"] = "ci_crosses_zero"
        weights = weights or UtilityWeights()
        entry["utility"] = {
            arm: weights.arm_utility(entry[arm]) for arm in ("candidate", "control")
        }
        entry["utility_delta"] = entry["utility"]["candidate"] - entry["utility"]["control"]
        # Secondary endpoints: fixed-sample contrasts for diagnosis and
        # prioritization. ``establishment`` is pinned False — secondary
        # deltas deliberately spend none of the family's sequential alpha;
        # the primary endpoint's confidence sequence is the only path that
        # can declare an effect established.
        entry["secondary_delta"] = {
            name: cls._secondary_delta(candidate, control, name)
            for name in ("verified_transition", "recovery")
        }
        c_steps = candidate["outcome_vector"]["secondary"]["steps"]
        k_steps = control["outcome_vector"]["secondary"]["steps"]
        entry["secondary_delta"]["steps"] = {
            "delta": (
                c_steps["mean"] - k_steps["mean"]
                if c_steps["mean"] is not None and k_steps["mean"] is not None
                else None
            ),
            "ci": None,
            "establishment": False,
        }
        # Safety is a constraint block: a measured excess on any safety
        # endpoint over the control arm is a regression flag — a veto that
        # can only ever close the active path, never a quantity the utility
        # annotation or a success delta can compensate for.
        cand_safety = candidate["outcome_vector"]["safety"]
        ctrl_safety = control["outcome_vector"]["safety"]
        both_measured = bool(
            candidate["outcome_vector"]["coverage"] > 0
            and control["outcome_vector"]["coverage"] > 0
        )
        regressions = [
            name
            for name in ("authority_touches", "guard_failures", "indeterminate_rate")
            if cand_safety.get(name) is not None
            and ctrl_safety.get(name) is not None
            and cand_safety[name] > ctrl_safety[name] + 1e-9
        ]
        entry["safety"] = {
            "regression": bool(regressions) if both_measured else None,
            "regressions": regressions,
        }
        return entry

    @staticmethod
    def _secondary_delta(candidate: dict, control: dict, key: str) -> dict:
        """Fixed-sample Newcombe contrast on a secondary binary endpoint.

        Diagnostic only — ``establishment`` stays False because secondary
        endpoints spend none of the family's sequential alpha. A missing
        (unmeasured) arm yields ``None`` fields, never a fabricated zero
        delta.
        """
        cs = candidate["outcome_vector"]["secondary"][key]
        ks = control["outcome_vector"]["secondary"][key]
        if cs["p"] is None or ks["p"] is None:
            return {"delta": None, "ci": None, "establishment": False}
        ci = _newcombe(
            {"p_success": cs["p"], "ess": candidate["ess"]},
            {"p_success": ks["p"], "ess": control["ess"]},
        )
        return {"delta": cs["p"] - ks["p"], "ci": list(ci), "establishment": False}

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
           role, overlap) until a mask has both arms with support. ``kind``
           and ``effect`` are never dropped: they are the signature floor, so
           evidence about one operation kind or one authority class can never
           answer for another. Coordinates the caller does not know are
           wildcards from the start; unknown *stored* values only match once
           their coordinate is dropped (or, for the floor coordinates, only
           match a caller that is itself unknown).

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
                    **self._contrast(
                        bucket,
                        min_effect=min_effect,
                        weights=weights,
                        hypothesis_count=len(
                            {tuple(cell[:_TRIAL_KEY_LEN - 1]) for cell in self.cells}
                        ),
                    ),
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
        version = str(payload.get("version", cls.version))
        migrated = False
        if cells and len(cells[0]) == _V4_CELL_LEN:
            # v4 evidence migrates in place; the collapsed scope is split and
            # the widened coordinates land as "unknown" wildcards.
            cells = _migrate_v6_trial_cells(_migrate_v4_trial_cells(cells))
            migrated = True
        elif cells and len(cells[0]) == _V5_CELL_LEN:
            # v5 cells are one reason counter short of v6; splice in the new
            # indeterminate_execution slot rather than rejecting the store.
            cells = _migrate_v6_trial_cells(_migrate_v5_trial_cells(cells))
            migrated = True
        elif cells and len(cells[0]) == _V6_CELL_LEN:
            # v6 cells lack the censored-weight statistic; splice a zero so
            # the bound computation can apply its imputed fallback.
            cells = _migrate_v6_trial_cells(cells)
            migrated = True
        if cells and len(cells[0]) == _V7_CELL_LEN:
            # v7 cells lack the outcome-vector counters; append zeros so
            # secondary/safety endpoints report as unmeasured, never as zero.
            cells = _migrate_v7_trial_cells(cells)
            migrated = True
        if migrated:
            version = cls.version
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


def _candidate_probability(estimate: dict | None) -> tuple[float | None, float | None]:
    """``(control_p, p_progress)`` implied by a resolved contrast.

    The causal candidate probability is the *measured* control-arm success
    rate plus the measured effect, clamped to [0, 1]. It is derived here, at
    the estimation layer, so no downstream consumer ever reconstructs it
    from a fabricated baseline such as ``0.5 + delta``. A contrast lacking a
    valid finite control probability yields ``p_progress = None`` — causal
    support without a control arm is not a probability.
    """
    if not isinstance(estimate, dict):
        return None, None
    control = estimate.get("control")
    delta = estimate.get("delta")
    try:
        control_p = (
            float(control.get("p_success")) if isinstance(control, dict) else None
        )
        delta_f = float(delta) if delta is not None else None
    except (TypeError, ValueError):
        return None, None
    if control_p is None or not math.isfinite(control_p):
        return None, None
    if delta_f is None or not math.isfinite(delta_f):
        return control_p, None
    return control_p, min(max(control_p + delta_f, 0.0), 1.0)


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
    ``beneficial`` divergence — the confidence sequence lies entirely above
    the practical threshold and the censoring bounds agree — and ``refuted``
    is true only for a ``harmful`` one. A supported cell whose interval still crosses zero is
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
            # confidence sequence, the censoring bounds, and the threshold.
            if best is None or estimate["delta"] > best[0]:
                best = (estimate["delta"], candidate, estimate)
        if best is None:
            return None
        delta, candidate, estimate = best
        control_p, p_progress = _candidate_probability(estimate)
        ci = estimate.get("delta_ci") or [delta, delta]
        return {
            "id": candidate.get("id"),
            "kind": candidate.get("kind"),
            # The candidate's implied success probability — measured
            # control-arm rate plus the measured effect — so downstream
            # ``expected_delta`` arithmetic stays probability-shaped.
            "control_p": control_p,
            "p_progress": p_progress,
            "uncertainty": (ci[1] - ci[0]) / 2.0,
            "confident": True,
            "expected_delta": delta,
            "trial_level": estimate["level"],
            "signature_level": estimate.get("signature_level"),
            "effect_status": estimate["effect_status"],
            "support_sufficient": estimate.get("support_sufficient"),
            "delta_cs": estimate.get("delta_cs"),
            "delta_bounds": estimate.get("delta_bounds"),
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
                "control_p": None,
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
                "control_p": None,
                "n": int((estimate.get("candidate") or {}).get("trials", 0)),
                "level": estimate.get("level"),
                "signature_level": estimate.get("signature_level"),
                "effect_status": estimate.get("effect_status") or "insufficient_data",
                "support_sufficient": bool(estimate.get("support_sufficient")),
                "source": "randomized",
            }
        control_p, p_progress = _candidate_probability(estimate)
        return {
            "p_progress": p_progress,
            "control_p": control_p,
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
            control_p, p_progress = _candidate_probability(estimate)
            ci = estimate.get("delta_ci")
            entries.append({
                "id": candidate.get("id"),
                "kind": candidate.get("kind"),
                "expected_delta": estimate.get("delta"),
                "control_p": control_p,
                # Propagated causal probability: measured control rate plus
                # measured delta, derived at the estimation layer. None when
                # the contrast lacks a valid finite control probability —
                # consumers fail closed on absent, never on assumed 0.5.
                "p_progress": p_progress,
                "uncertainty": (
                    (float(ci[1]) - float(ci[0])) / 2.0
                    if isinstance(ci, (list, tuple)) and len(ci) == 2
                    else None
                ),
                "delta_ci": list(ci) if isinstance(ci, (list, tuple)) else None,
                "utility_delta": estimate.get("utility_delta"),
                "effect_status": status,
                "support_sufficient": estimate.get("support_sufficient"),
                "trial_level": estimate.get("level"),
                "signature_level": estimate.get("signature_level"),
                # Structured outcome provenance: secondary deltas are
                # diagnostic only (``establishment`` pinned False at the
                # contrast layer), and ``safety_regression`` is a constraint
                # flag — it can close the active path, never open it.
                "safety_regression": (estimate.get("safety") or {}).get("regression"),
                "secondary_delta": estimate.get("secondary_delta"),
                "outcome_vector": (estimate.get("candidate") or {}).get("outcome_vector"),
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
