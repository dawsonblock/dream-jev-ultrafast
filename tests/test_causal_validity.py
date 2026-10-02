"""Adversarial suite for the v0.9.0 causal-validity guarantees.

Every test here encodes a failure mode an audit called out, or a property the
causal layer must never lose: "enough samples" is not "effect established",
differential censoring cannot hide behind surviving successes, evidence is
indexed under the context it was measured in, plans are bound and — when
signed — authenticated, duplicate assignments count once, and the learned
layers never mix observational and randomized provenance.

No paid APIs; everything is synthetic evidence.
"""

import json

from jev_ultrafast.dream import (
    DreamImprover,
    ExplorationPolicy,
    PromotionGate,
    ReplayWorld,
    experiment_plan_authority,
    experiment_plan_digest,
    experiment_plan_signature,
)
from jev_ultrafast.dreamlearn import (
    CausalChoicePolicy,
    ChoiceModel,
    CounterfactualTrials,
    ExperimentScheduler,
    TrialChoiceModel,
    UtilityWeights,
)
from jev_ultrafast.signing import EvidenceSigner

# --------------------------------------------------------------- builders


def _trial_meta(run_id, arm, *, propensity=0.5, kind="click", model_kind="click",
                overlap=1, model_overlap=0, task_family=None, site=None,
                proposal_effect=None, model_effect=None, phase=None):
    meta = {
        "experiment_id": f"x{run_id}",
        "arm": arm,
        "assignment_probability": propensity,
        "proposal_id": f"p{run_id}",
        "proposal_kind": kind,
        "proposal_overlap": overlap,
        "proposal_offered_rank": 0,
        "model_choice_id": f"m{run_id}",
        "model_choice_kind": model_kind,
        "model_choice_overlap": model_overlap,
        "model_choice_offered_rank": 1,
        "task_key": "t",
    }
    if task_family is not None:
        meta["task_family"] = task_family
    if site is not None:
        meta["site"] = site
    if proposal_effect is not None:
        meta["proposal_effect"] = proposal_effect
    if model_effect is not None:
        meta["model_choice_effect"] = model_effect
    if phase is not None:
        meta["phase"] = phase
    return meta


def _trial_run(run_id, arm, *, success=True, propensity=0.5, kind="click",
               model_kind="click", overlap=1, model_overlap=0, status=None,
               verified=None, executed=True, task_family=None, site=None,
               reason=None, proposal_effect=None, model_effect=None,
               latency_ms=0, tokens=0, approvals=0, duplicate=False):
    """One randomized trial as a complete run (see test_learned's builder)."""
    meta = _trial_meta(
        run_id, arm, propensity=propensity, kind=kind, model_kind=model_kind,
        overlap=overlap, model_overlap=model_overlap,
        task_family=task_family, site=site,
        proposal_effect=proposal_effect, model_effect=model_effect,
    )
    transitions = []
    if executed:
        transitions.append({
            "event": "transition",
            "state": "S",
            "next_state": "D",
            "selected": {"id": meta["proposal_id"] if arm == "candidate" else meta["model_choice_id"],
                         "kind": kind},
            "candidates": [
                {"id": meta["proposal_id"], "kind": kind, "label": "Go", "goal_overlap": overlap},
                {"id": meta["model_choice_id"], "kind": model_kind, "label": "M",
                 "goal_overlap": model_overlap},
            ],
            "page_changed": True,
            "latency_ms": latency_ms,
            "model_calls": 1,
            "tokens": tokens,
            "stale_or_failure": 0,
            "risk_events": 0,
            "experiment": meta,
        })
    finished = {"event": "run_finished", "run_id": run_id, "task_key": "t",
                "status": status or ("done" if success else "blocked"),
                "verified": success if verified is None else verified}
    if reason is not None:
        finished["reason"] = reason
    assignment = {"event": "experiment_assigned", "run_id": run_id, "task_key": "t",
                  "experiment": meta}
    events = [
        {"event": "run_started", "run_id": run_id, "task_key": "t", "goal": "g",
         "policy": ExplorationPolicy().to_dict(), "policy_digest": ExplorationPolicy().digest},
        assignment,
        *[{**t, "run_id": run_id, "task_key": "t"} for t in transitions],
        finished,
    ]
    if approvals:
        events = events[:2] + [
            {"event": "approval_required", "run_id": run_id, "task_key": "t"}
        ] * int(approvals) + events[2:]
    if duplicate:
        events.insert(2, dict(assignment))
    return events


def _cell(events, *, family="", site=""):
    """The single trial contrast in a fit — convenience for one-cell suites."""
    trials = CounterfactualTrials.fit(events)
    for key, entry in trials.estimate().items():
        if key.startswith(f"{family}|{site}|"):
            return entry
    raise AssertionError(f"no cell for family={family!r} site={site!r}")


# ------------------------------------------- support vs. effect certainty


def test_ci_crossing_zero_is_unresolved_not_beneficial():
    """The audit's repro: 5/8 vs 4/8 has enough samples but a delta CI that
    clearly crosses zero. It must be *supported but unresolved* — not a
    confident proposal."""
    events = []
    for i in range(8):
        events += _trial_run(f"c{i}", "candidate", success=i < 5)
        events += _trial_run(f"k{i}", "control", success=i < 4)
    arms = _cell(events)
    assert abs(arms["delta"] - 0.125) < 1e-9
    assert arms["support_sufficient"] is True
    assert arms["effect_status"] == "unresolved"
    assert arms["unresolved_reason"] == "ci_crosses_zero"
    assert arms["alpha_spent_delta_ci"][0] <= 0 <= arms["alpha_spent_delta_ci"][1]
    model = TrialChoiceModel.fit(events)
    assert model.choose(
        [{"id": "m", "kind": "click", "goal_overlap": 0},
         {"id": "p", "kind": "click", "goal_overlap": 1}],
        model_choice={"id": "m", "kind": "click", "goal_overlap": 0},
    ) is None
    assert model.refuted(
        kind="click", goal_overlap=1, model_kind="click", model_overlap=0) is False


def test_inconclusive_negative_delta_is_not_refuted():
    """The reverse case is worse: an inconclusive experiment must not
    permanently suppress a potentially beneficial hypothesis."""
    events = []
    for i in range(8):
        events += _trial_run(f"c{i}", "candidate", success=i < 4)
        events += _trial_run(f"k{i}", "control", success=i < 5)
    arms = _cell(events)
    assert arms["delta"] < 0
    assert arms["support_sufficient"] is True
    assert arms["effect_status"] == "unresolved"
    model = TrialChoiceModel.fit(events)
    assert model.refuted(
        kind="click", goal_overlap=1, model_kind="click", model_overlap=0) is False


def test_established_signs_still_establish():
    """The conservative direction must not become vacuity: strong, fully
    separated arms do establish in both directions."""
    positive = []
    negative = []
    for i in range(8):
        positive += _trial_run(f"c{i}", "candidate", success=True)
        positive += _trial_run(f"k{i}", "control", success=False)
        negative += _trial_run(f"d{i}", "candidate", success=False)
        negative += _trial_run(f"n{i}", "control", success=True)
    assert _cell(positive)["effect_status"] == "beneficial"
    assert _cell(negative)["effect_status"] == "harmful"
    assert TrialChoiceModel.fit(positive).refuted(
        kind="click", goal_overlap=1, model_kind="click", model_overlap=0) is False
    assert TrialChoiceModel.fit(negative).refuted(
        kind="click", goal_overlap=1, model_kind="click", model_overlap=0) is True


def test_sequential_establishment_is_stricter_than_fixed_sample():
    """Optional stopping: a fixed-sample interval that excludes zero is not
    enough. Establishment uses the α-spent approximate interval, so data that
    merely looks favorable at one peek stays unresolved."""
    events = []
    for i in range(8):
        events += _trial_run(f"c{i}", "candidate", success=i < 7)
        events += _trial_run(f"k{i}", "control", success=i < 3)
    arms = _cell(events)
    assert arms["delta_ci"][0] > 0  # a single fixed-sample look excludes zero
    assert arms["effect_status"] == "unresolved"  # the sequential rule does not
    assert arms["alpha_spent_delta_ci"][0] < 0
    assert arms["alpha_spent_delta_ci"][0] <= arms["delta_ci"][0]
    assert arms["sequential_alpha"] < CounterfactualTrials.SEQUENTIAL_ALPHA


# ------------------------------------------------------- differential censoring


def test_differential_censoring_blocks_establishment():
    """The audit's repro: a candidate arm censored 83% of the time must not
    look excellent on its eight surviving successes."""
    events = []
    for i in range(8):
        events += _trial_run(f"c{i}", "candidate", success=True)
    for i in range(40):
        events += _trial_run(f"a{i}", "candidate", status="aborted", verified=False,
                             reason="timeout")
    for i in range(8):
        events += _trial_run(f"k{i}", "control", success=False)
    arms = _cell(events)
    candidate = arms["candidate"]
    assert candidate["assigned"] == 48
    assert candidate["trials"] == 8
    assert candidate["censored"] == 40
    assert candidate["p_success"] > 0.99
    assert arms["support_sufficient"] is True
    assert arms["censoring_imbalance"] is True
    assert arms["effect_status"] == "unresolved"
    assert arms["unresolved_reason"] == "censoring_imbalance"
    # The surviving subset must not be proposed as an established winner.
    model = TrialChoiceModel.fit(events)
    assert model.choose(
        [{"id": "m", "kind": "click", "goal_overlap": 0},
         {"id": "p", "kind": "click", "goal_overlap": 1}],
        model_choice={"id": "m", "kind": "click", "goal_overlap": 0},
    ) is None


def test_candidate_induced_timeout_is_distinguishable_from_operator_cancel():
    """The reason taxonomy separates an operator cancel from a crash/timeout
    the treatment may have caused, and the imbalance gate refuses to
    establish under extreme candidate-side censoring."""
    events = []
    for i in range(8):
        events += _trial_run(f"c{i}", "candidate", success=True)
    for i in range(10):
        events += _trial_run(f"t{i}", "candidate", status="aborted", verified=False,
                             reason="timeout")
    for i in range(8):
        events += _trial_run(f"k{i}", "control", success=False)
    arms = _cell(events)
    assert arms["candidate"]["terminations"] == {"timeout": 10}
    assert arms["control"]["terminations"] == {}
    assert abs(arms["candidate"]["censor_rate"] - 10 / 18) < 1e-9
    assert arms["effect_status"] == "unresolved"


def test_balanced_operator_cancels_are_censored_not_failures():
    """Operator cancels are unrelated to treatment: when both arms carry the
    same small censor rate, they are excluded from the endpoint and the
    effect can still establish."""
    events = []
    for i in range(8):
        events += _trial_run(f"c{i}", "candidate", success=True)
        events += _trial_run(f"k{i}", "control", success=False)
    for i in range(2):
        events += _trial_run(f"ca{i}", "candidate", status="aborted", verified=False,
                             reason="operator_cancel")
        events += _trial_run(f"ka{i}", "control", status="aborted", verified=False,
                             reason="operator_cancel")
    arms = _cell(events)
    assert arms["candidate"]["censored"] == 2
    assert arms["control"]["censored"] == 2
    assert abs(arms["candidate"]["censor_rate"] - 0.2) < 1e-9
    assert arms["censoring_imbalance"] is False
    assert arms["effect_status"] == "beneficial"


# ------------------------------------------------- practical-effect thresholds


def test_minimum_effect_requires_practical_significance():
    """Statistically established is not automatically worthwhile: a family
    may demand a minimum absolute improvement."""
    events = []
    for i in range(8):
        events += _trial_run(f"c{i}", "candidate", success=True)
        events += _trial_run(f"k{i}", "control", success=False)
    resolved = CounterfactualTrials.fit(events).resolve(
        model_kind="click", model_overlap=0, proposal_kind="click", proposal_overlap=1)
    assert resolved["effect_status"] == "beneficial"
    gated = CounterfactualTrials.fit(events).resolve(
        model_kind="click", model_overlap=0, proposal_kind="click", proposal_overlap=1,
        min_effect=0.9)
    assert gated["effect_status"] == "unresolved"
    model = TrialChoiceModel(
        trials=CounterfactualTrials.fit(events),
        minimum_effects=(("flights", 0.9),),
    )
    assert model.choose(
        [{"id": "p", "kind": "click", "goal_overlap": 1}],
        model_choice={"id": "m", "kind": "click", "goal_overlap": 0},
        task_family="flights",
    ) is None
    assert model.choose(
        [{"id": "p", "kind": "click", "goal_overlap": 1}],
        model_choice={"id": "m", "kind": "click", "goal_overlap": 0},
    ) is not None


# ---------------------------------------------- treatment-signature backoff


def test_treatment_signature_backoff_reports_its_level():
    """Evidence recorded for one effect class must not answer a different
    class at full specificity: resolution drops coordinates hierarchically
    and names the mask that answered, so a class-level delta is never
    presented as if it were measured on the queried treatment."""
    events = []
    for i in range(8):
        events += _trial_run(f"c{i}", "candidate", success=True,
                             proposal_effect="navigate", model_effect="form_edit")
        events += _trial_run(f"k{i}", "control", success=False,
                             proposal_effect="navigate", model_effect="form_edit")
    trials = CounterfactualTrials.fit(events)
    exact = trials.resolve(
        model_kind="click", model_overlap=0, model_effect="form_edit",
        proposal_kind="click", proposal_overlap=1, proposal_effect="navigate")
    assert exact["signature_level"] == "full"
    assert exact["effect_status"] == "beneficial"
    # A *different* effect class never matches: effect is a signature floor —
    # it is not dropped under backoff, so evidence about "navigate" proposals
    # cannot answer a "form_edit" divergence merely because both were clicks.
    backoff = trials.resolve(
        model_kind="click", model_overlap=0, model_effect="form_edit",
        proposal_kind="click", proposal_overlap=1, proposal_effect="form_edit")
    assert backoff is None


def test_never_randomized_action_class_is_flagged_not_assumed():
    """A never-randomized action inherits a *class-level* estimate with
    provenance, and the policy layer refuses to execute it deterministically
    unless the estimate is established."""
    events = []
    for i in range(8):
        events += _trial_run(f"c{i}", "candidate", success=True, proposal_effect="navigate")
        events += _trial_run(f"k{i}", "control", success=False, proposal_effect="navigate")
    model = TrialChoiceModel.fit(events)
    candidates = [
        {"id": "m", "kind": "click", "goal_overlap": 0},
        # Same treatment *class* (kind + effect) as the measured evidence —
        # role/rank/phase may back off, but a different effect class could
        # never inherit this estimate.
        {"id": "never-randomized-before", "kind": "click", "goal_overlap": 1,
         "effect": "navigate"},
    ]
    ranked = model.rank(candidates, model_choice={"id": "m", "kind": "click", "goal_overlap": 0})
    assert ranked and ranked[0]["id"] == "never-randomized-before"
    assert ranked[0]["source"] == "randomized"
    # The action ID was never randomized, but its full treatment signature is
    # exactly the measured class — kind + effect is the identity the estimate
    # generalizes over, so this answers at "full" with randomized provenance.
    assert ranked[0]["signature_level"] == "full"
    assert ranked[0]["effect_status"] == "beneficial"

    shadow = CausalChoicePolicy(mode="shadow", trial_model=model)
    assert shadow.proposal(
        candidates, model_choice={"id": "m", "kind": "click", "goal_overlap": 0}
    ) is None  # shadow observes, never acts
    active = CausalChoicePolicy(mode="active", trial_model=model)
    assert active.proposal(
        candidates, model_choice={"id": "m", "kind": "click", "goal_overlap": 0}
    ) is not None
    # An *unresolved* class-level estimate blocks deterministic execution.
    thin = TrialChoiceModel.fit(_trial_run("a", "candidate") + _trial_run("b", "control"))
    assert CausalChoicePolicy(mode="active", trial_model=thin).proposal(
        candidates, model_choice={"id": "m", "kind": "click", "goal_overlap": 0}
    ) is None


# ------------------------------------------------------ duplicates / integrity


def test_duplicate_assignment_records_count_once():
    """The same assignment written twice is one unit of analysis — counting
    it twice would double-weight that arm. The duplicate is reported."""
    events = _trial_run("dup", "candidate", success=True, duplicate=True)
    trials = CounterfactualTrials.fit(events)
    arm = trials.estimate()[next(iter(trials.estimate()))]["candidate"]
    assert arm["assigned"] == 1
    assert trials.duplicates == 1


def test_resolve_never_answers_a_contrast_from_one_arm():
    """A bucket holding only one arm cannot answer "is assigning B better
    than A" — resolution keeps walking and returns None, while the raw arm
    stays visible in ``estimate()`` reporting. Regression: a one-armed
    resolution used to crash ``predict`` and the scheduler with KeyError."""
    events = []
    for i in range(8):
        events += _trial_run(f"c{i}", "candidate", success=True)
    trials = CounterfactualTrials.fit(events)
    assert trials.resolve(
        model_kind="click", model_overlap=0, proposal_kind="click", proposal_overlap=1
    ) is None
    arm = trials.estimate()[next(iter(trials.estimate()))]["candidate"]
    assert arm["assigned"] == 8  # the arm itself is still reported
    model = TrialChoiceModel(trials=trials)
    prediction = model.predict(kind="click", goal_overlap=1)
    assert prediction["p_progress"] is None
    assert model.choose(
        [{"id": "p", "kind": "click", "goal_overlap": 1}],
        model_choice={"id": "m", "kind": "click", "goal_overlap": 0},
    ) is None
    ranked = ExperimentScheduler(budget=5).rank([{
        "world": "w", "step": 0, "expected_delta": 0.5,
        "proposal": {"id": "p", "kind": "click", "goal_overlap": 1},
        "historical": {"kind": "click", "goal_overlap": 0}}], estimates=trials)
    assert [r["world"] for r in ranked] == ["w"]
    assert ranked[0]["scheduler"]["support_fraction"] == 0.0


def test_v4_migration_preserves_fractional_weights():
    """v4 cells migrate verbatim: IPW weighted sums are fractional whenever
    propensities differ, and truncating them would silently corrupt the
    estimate (regression: ``int()`` on migration)."""
    v4 = ("*", "click", "0", "click", "1", "candidate",
          2, 2, 2, 0, 1.0, 2.3333333333333335, 3.0, 2.0, 2.3333333333333335)
    trials = CounterfactualTrials.from_dict({"version": "jev-trials/4", "cells": [v4]})
    entry = trials.estimate()[next(iter(trials.estimate()))]["candidate"]
    assert abs(entry["p_success"] - 1.0 / 2.3333333333333335) < 1e-12
    assert abs(entry["ess"] - 2.3333333333333335 ** 2 / 3.0) < 1e-12


# ------------------------------------------------------ serialization / v4


def test_v4_cells_migrate_to_split_coordinates():
    """v4 evidence (collapsed scope, 15 fields) migrates in place: the scope
    is split back into family/site, censored counts land under
    ``unknown_abort`` (v4 did not record why), and the widened signature
    coordinates are wildcards — never invented values."""
    v4 = (
        "f:flights", "click", "0", "click", "1", "candidate",
        10, 8, 8, 2, 6.0, 8.0, 8.0, 8.0, 8.0,
    )
    trials = CounterfactualTrials.from_dict({"version": "jev-trials/4", "cells": [v4]})
    assert trials.version == "jev-trials/6"
    cell = trials.cells[0]
    assert cell[0] == "flights" and cell[1] == ""
    assert cell[3] == "unknown" and cell[6] == "unknown"  # effect/rank wildcards
    assert cell[-1] == 2  # unknown_abort count survived
    entry = trials.estimate()[next(iter(trials.estimate()))]["candidate"]
    assert entry["assigned"] == 10 and entry["trials"] == 8 and entry["censored"] == 2


def test_v5_cells_migrate_with_zeroed_indeterminate_counter():
    """v5 cells (8 reason counters) migrate to the v6 layout: the new
    ``indeterminate_execution`` slot splices in as zero and every recorded
    verdict — including the unverified_claim counter that used to sit at the
    same tail position — keeps its count."""
    from jev_ultrafast.dreamlearn import TERMINATION_REASONS, TRIAL_CELL_LEN

    reasons_v5 = [0, 1, 0, 0, 0, 0, 3, 2]  # v5 tail order, ending unverified/unknown
    key = ("flights", "example.test", "click", "navigate", "button", "0", "0-4", "0-2",
           "click", "navigate", "link", "1", "0-4", "0-2", "candidate")
    stats = [10, 8, 8, 2, 6.0, 8.0, 8.0, 8.0, 8.0]
    v5 = (*key, *stats, 1.0, 2.0, 0.5, *reasons_v5)
    assert len(v5) == TRIAL_CELL_LEN - 1
    trials = CounterfactualTrials.from_dict({"version": "jev-trials/5", "cells": [v5]})
    assert trials.version == "jev-trials/6"
    cell = trials.cells[0]
    assert len(cell) == TRIAL_CELL_LEN
    tail = cell[-len(TERMINATION_REASONS):]
    names = list(TERMINATION_REASONS)
    assert tail[names.index("indeterminate_execution")] == 0
    assert tail[names.index("unverified_claim")] == 3
    assert tail[names.index("unknown_abort")] == 2
    assert tail[names.index("browser_crash")] == 1
    entry = trials.estimate()[next(iter(trials.estimate()))]["candidate"]
    assert entry["terminations"]["unverified_claim"] == 3


def test_censoring_reason_taxonomy_roundtrips():
    """Every structured reason survives the recorder → fit path unchanged,
    and an unrecognized or missing reason degrades to ``unknown_abort``."""
    from jev_ultrafast.dreamlearn import TERMINATION_REASONS

    for reason in TERMINATION_REASONS:
        events = _trial_run("a", "candidate", status="aborted", verified=False,
                            reason=reason)
        arm = _cell(events)["candidate"]
        assert arm["terminations"] == {reason: 1}, reason
    invented = _trial_run("b", "candidate", status="aborted", verified=False,
                          reason="cosmic_ray")
    assert _cell(invented)["candidate"]["terminations"] == {"unknown_abort": 1}


def test_censoring_gate_boundaries_are_strict():
    """The imbalance gate triggers only beyond its thresholds: a small
    balanced censor rate is tolerated, an extreme or sharply imbalanced one
    is not."""
    def build(candidate_censored, candidate_analyzed):
        events = []
        for i in range(candidate_analyzed):
            events += _trial_run(f"c{i}", "candidate", success=True)
        for i in range(candidate_censored):
            events += _trial_run(f"a{i}", "candidate", status="aborted", verified=False,
                                 reason="operator_cancel")
        for i in range(8):
            events += _trial_run(f"k{i}", "control", success=False)
        return _cell(events)

    extreme = build(8, 8)  # 8/16 = 0.5 censored against 0.0 — gap 0.5 > 0.25
    assert extreme["censoring_imbalance"] is True
    assert extreme["effect_status"] == "unresolved"

    balanced = build(2, 8)  # 2/10 = 0.2, gap 0.2 — tolerated
    assert balanced["censoring_imbalance"] is False
    assert balanced["effect_status"] == "beneficial"


def test_minimum_effect_boundary_is_strict():
    """Establishment requires the interval to lie entirely *above* the
    threshold — a CI touching it is not established."""
    events = []
    for i in range(8):
        events += _trial_run(f"c{i}", "candidate", success=True)
        events += _trial_run(f"k{i}", "control", success=False)
    trials = CounterfactualTrials.fit(events)
    resolved = trials.resolve(
        model_kind="click", model_overlap=0, proposal_kind="click", proposal_overlap=1)
    lower = resolved["alpha_spent_delta_ci"][0]
    assert lower > 0
    at_boundary = trials.resolve(
        model_kind="click", model_overlap=0, proposal_kind="click", proposal_overlap=1,
        min_effect=lower)
    assert at_boundary["effect_status"] == "unresolved"
    below = trials.resolve(
        model_kind="click", model_overlap=0, proposal_kind="click", proposal_overlap=1,
        min_effect=max(0.0, lower - 0.01))
    assert below["effect_status"] == "beneficial"


def test_sequential_spending_sequence_is_summable():
    """The alpha-spending sequence must sum to at most the nominal level over
    every possible look — that is what makes repeated peeking safe for a
    single hypothesis."""
    from jev_ultrafast.dreamlearn import _sequential_alpha

    partial = sum(_sequential_alpha(n, base=0.05) for n in range(1, 100000))
    # The finite partial sum is already below the nominal level, and the
    # tail beyond it is bounded by 0.05·2/√N (integral bound) — so the whole
    # sequence sums to at most alpha.
    tail_bound = 0.05 * (2 / (99999 ** 0.5))
    assert partial <= 0.05
    assert partial + tail_bound <= 0.05 * 1.01  # the tail is genuinely small
    assert partial > 0.049  # and the spending is not wastefully conservative
    assert _sequential_alpha(8, base=0.05) > _sequential_alpha(64, base=0.05)


def test_scheduler_spreads_coverage_across_families():
    """The coverage penalty keeps the trial budget from over-fitting one
    family: an identical hypothesis in a heavily-sampled family ranks below
    the same hypothesis in a fresh family."""
    events = []
    for i in range(8):
        events += _trial_run(f"c{i}", "candidate", success=True)
        events += _trial_run(f"k{i}", "control", success=False)
    trials = CounterfactualTrials.fit(events)

    def hypothesis(world, family):
        return {
            "world": world, "step": 0, "expected_delta": 0.4, "family_key": family,
            "proposal": {"id": "q", "kind": "fill", "goal_overlap": 1},
            "historical": {"kind": "click", "goal_overlap": 0},
        }

    ranked = ExperimentScheduler(budget=10).rank(
        [hypothesis("crowded1", "flights"), hypothesis("crowded2", "flights"),
         hypothesis("fresh", "banking")],
        estimates=trials,
    )
    assert ranked[0]["world"] == "fresh"


def test_trial_model_ignores_observational_transitions():
    """The causal channel reads only randomized assignments: an
    experiment-free store yields no causal evidence at all."""
    events = []
    for i in range(8):
        events.append({"event": "run_started", "run_id": f"o{i}", "task_key": "t",
                       "goal": "g", "policy": ExplorationPolicy().to_dict(),
                       "policy_digest": ExplorationPolicy().digest})
        events.append({"event": "transition", "run_id": f"o{i}", "task_key": "t",
                       "state": "S", "next_state": "D",
                       "selected": {"id": "m", "kind": "click"},
                       "candidates": [{"id": "m", "kind": "click", "goal_overlap": 1}],
                       "page_changed": True, "latency_ms": 1, "model_calls": 1,
                       "tokens": 1, "stale_or_failure": 0, "risk_events": 0})
        events.append({"event": "run_finished", "run_id": f"o{i}", "task_key": "t",
                       "status": "done", "verified": True})
    model = TrialChoiceModel.fit(events)
    assert model.trials.cells == ()
    assert model.choose(
        [{"id": "p", "kind": "click", "goal_overlap": 1}],
        model_choice={"id": "m", "kind": "click", "goal_overlap": 0},
    ) is None
    assert model.predict(kind="click", goal_overlap=1)["p_progress"] is None


# ------------------------------------------------------------ multi-objective


def test_utility_weights_penalize_cost_and_censoring():
    """Success alone is not utility: latency, tokens, approvals, and censoring
    all reduce it — and safety constraints are outside the function."""
    events = []
    for i in range(8):
        events += _trial_run(f"c{i}", "candidate", success=True, latency_ms=10, tokens=10)
        events += _trial_run(f"k{i}", "control", success=False, latency_ms=10, tokens=10)
    cheap = _cell(events)
    slow_events = []
    for i in range(8):
        slow_events += _trial_run(f"c{i}", "candidate", success=True, latency_ms=10000,
                                  tokens=10000, approvals=2)
        slow_events += _trial_run(f"k{i}", "control", success=False, latency_ms=10, tokens=10)
    costly = _cell(slow_events)
    weights = UtilityWeights()
    assert weights.arm_utility(costly["candidate"]) < weights.arm_utility(cheap["candidate"])
    assert cheap["utility_delta"] > costly["utility_delta"]
    assert costly["utility"]["candidate"] < costly["utility"]["control"] + 0.5


def test_scheduler_drops_settled_hypotheses_and_respects_budget():
    """The scheduler only buys information that still exists: settled
    hypotheses are excluded, unresolved ones are ranked, and the budget caps
    the batch deterministically."""
    events = []
    for i in range(8):
        events += _trial_run(f"c{i}", "candidate", success=True)
        events += _trial_run(f"k{i}", "control", success=False)
    trials = CounterfactualTrials.fit(events)
    settled = {
        "world": "w0", "step": 0, "expected_delta": 0.9,
        "proposal": {"id": "p", "kind": "click", "goal_overlap": 1},
        "historical": {"kind": "click", "goal_overlap": 0},
    }
    unresolved = {
        "world": "w1", "step": 1, "expected_delta": 0.4,
        # A *fill* proposal has no measured evidence at all — the kind
        # coordinate is a floor, so click evidence cannot answer it.
        "proposal": {"id": "q", "kind": "fill", "goal_overlap": 2},
        "historical": {"kind": "click", "goal_overlap": 0},
    }
    ranked = ExperimentScheduler(budget=1).rank([settled, unresolved], estimates=trials)
    assert len(ranked) == 1
    assert ranked[0]["world"] == "w1"
    # The surviving hypothesis has no evidence yet (no estimate at all) —
    # exactly the uncertainty the scheduler exists to buy down.
    assert ranked[0]["scheduler"]["effect_status"] is None
    assert ranked[0]["scheduler"]["support_fraction"] == 0.0


# ---------------------------------------------------------------- provenance


def test_provenance_never_mixes_channels():
    """Observational and randomized predictions carry their channel; a
    randomized estimate that had to back off to the pooled stratum says
    ``pooled_randomized``."""
    events = []
    for i in range(8):
        events += _trial_run(f"c{i}", "candidate", success=True)
        events += _trial_run(f"k{i}", "control", success=False)
    trials = CounterfactualTrials.fit(events)
    causal = TrialChoiceModel(trials=trials)
    assert causal.predict(kind="click", goal_overlap=1)["source"] == "randomized"
    proposal = causal.choose(
        [{"id": "p", "kind": "click", "goal_overlap": 1}],
        model_choice={"id": "m", "kind": "click", "goal_overlap": 0},
    )
    assert proposal["source"] == "randomized"
    observational = ChoiceModel.fit([]).predict(kind="click", goal_overlap=1)
    assert observational["source"] == "observational"
    policy = CausalChoicePolicy(mode="shadow", trial_model=causal)
    ranking = policy.rank(
        [{"id": "m", "kind": "click", "goal_overlap": 0},
         {"id": "p", "kind": "click", "goal_overlap": 1}],
        model_choice={"id": "m", "kind": "click", "goal_overlap": 0},
        task_family="flights",
    )
    # The evidence is pooled (no family stratum exists), and the provenance
    # says exactly that.
    assert ranking["proposals"][0]["source"] == "pooled_randomized"
    assert ranking["shadow"] is True


# --------------------------------------------------------- plan signatures


def test_signed_plan_roundtrip_and_tamper_detection():
    """SHA-256 gives integrity, Ed25519 gives provenance. A tampered plan
    fails the digest; a tampered signature fails verification; an untrusted
    key is not authority."""
    signer = EvidenceSigner(bytes(range(32)))
    other = EvidenceSigner(bytes(range(1, 33)))
    plan = {
        "schema": "jev-experiment-plan/1",
        "task_key": "t",
        "family_key": "f",
        "state": "S",
        "proposal": {"id": "p", "kind": "click"},
    }
    plan["digest"] = experiment_plan_digest(plan)
    plan["authority"] = experiment_plan_signature(plan, signer)
    trusted = experiment_plan_authority(plan, {signer.key_id})
    assert trusted == {"present": True, "signature_valid": True, "trusted": True}
    untrusted = experiment_plan_authority(plan, {other.key_id})
    assert untrusted["signature_valid"] is True and untrusted["trusted"] is False
    # The digest excludes the authority block, so signing did not change it.
    assert plan["digest"] == experiment_plan_digest(plan)
    tampered = dict(plan)
    tampered["proposal"] = {"id": "evil", "kind": "click"}
    assert tampered["digest"] != experiment_plan_digest(tampered)
    forged = dict(plan)
    forged["authority"] = {**plan["authority"], "signature": "00" * 64}
    assert experiment_plan_authority(forged, {signer.key_id})["signature_valid"] is False
    unsigned = {k: v for k, v in plan.items() if k != "authority"}
    assert experiment_plan_authority(unsigned, {signer.key_id})["present"] is False


def test_improve_stamps_signed_plans_with_the_scheduler(tmp_path):
    """End to end: improve() stamps scheduler-ranked proposals, and with a
    signer configured every plan carries a verifiable authority block."""
    store = tmp_path / "events.jsonl"
    worlds = _divergent_worlds(store)
    events = [json.loads(line) for line in store.read_text().splitlines() if line.strip()]
    choice_model = ChoiceModel.fit(events)
    signer = EvidenceSigner(bytes(range(32)))
    report = DreamImprover(gate=PromotionGate(min_coverage=0.0)).improve(
        worlds,
        ExplorationPolicy(),
        choice_model=choice_model,
        scheduler=ExperimentScheduler(),
        plan_signer=signer,
    )
    assert report.experiment_proposals
    for plan in report.experiment_proposals:
        assert plan["digest"] == experiment_plan_digest(plan)
        authority = experiment_plan_authority(plan, {signer.key_id})
        assert authority["trusted"] is True
        # The scheduler's bookkeeping never leaks into the immutable artifact.
        assert "estimate" not in plan and "scheduler" not in plan


def _divergent_worlds(store_path):
    """Runs where the recorded policy picked the low-overlap sibling while a
    goal-relevant action sat in the same offered catalogue (see
    test_learned's builder for the full rationale)."""
    events = []
    policy = ExplorationPolicy().to_dict()

    def run(run_id, selected, positive, status):
        events.append({"event": "run_started", "run_id": run_id, "task_key": f"t{run_id}",
                       "policy": policy, "policy_digest": ExplorationPolicy().digest})
        events.append({
            "event": "transition", "run_id": run_id, "task_key": f"t{run_id}",
            "state": "S", "next_state": "S2",
            "selected": {"id": selected, "kind": "click"},
            "candidates": [
                {"id": "low", "kind": "click", "label": "Low", "goal_overlap": 0},
                {"id": "high", "kind": "click", "label": "Goal", "goal_overlap": 3},
            ],
            "page_changed": positive, "latency_ms": 5, "model_calls": 1, "tokens": 1,
            "stale_or_failure": 0, "risk_events": 0,
        })
        events.append({"event": "run_finished", "run_id": run_id, "task_key": f"t{run_id}",
                       "status": status, "verified": status == "done"})

    for i in range(4):
        run(f"good{i}", "high", True, "done")
    for i in range(6):
        run(f"bad{i}", "low", False, "blocked")
    store_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return ReplayWorld.from_events(events)


def test_execution_step_never_matches_on_absent_ids():
    """A sparse assignment meta must not claim an unrelated step.

    ``proposal_id=None`` matching a step that lacks a proposal id is a
    ``None == None`` coincidence, not evidence — misattribution would fold a
    different experiment's execution (and its treatment-signature context)
    into this arm's cell.
    """
    step = {
        "event": "transition",
        "experiment": {"experiment_id": "other", "proposal_id": "pB", "model_choice_id": "mB"},
    }
    tagged = {"event": "transition", "experiment": {"experiment_id": "tagged"}}
    meta = {"experiment_id": "self", "proposal_id": None, "model_choice_id": None}
    assert CounterfactualTrials._execution_step(meta, [step, tagged]) is None
    # Identity still resolves when the meta is informative.
    assert (
        CounterfactualTrials._execution_step(
            {"experiment_id": "nope", "proposal_id": "pB", "model_choice_id": "mB"},
            [step, tagged],
        )
        is step
    )
    # ...and a run with exactly one tagged step still resolves by singleton.
    assert (
        CounterfactualTrials._execution_step(
            {"experiment_id": "self", "proposal_id": None}, [tagged]
        )
        is tagged
    )


def test_loose_assignment_records_deduplicate():
    """Runless ``experiment_assigned`` events deduplicate like run-scoped ones.

    A loose record is censored-by-construction evidence, but two copies of the
    same assignment are still one unit of analysis — without the dedupe the
    arm's censor counts inflate silently.
    """
    meta = {
        "experiment_id": "e1",
        "proposal_id": "p",
        "model_choice_id": "m",
        "arm": "candidate",
        "assignment_probability": 0.5,
        "proposal_kind": "click",
        "model_choice_kind": "click",
    }
    trials = CounterfactualTrials.fit(
        [{"event": "experiment_assigned", "experiment": meta} for _ in range(3)]
    )
    estimate = trials.estimate()
    arm = next(iter(estimate.values()))["candidate"]
    assert arm["assigned"] == 1 and arm["censored"] == 1
    assert trials.duplicates == 2


def test_context_coordinates_never_carry_the_join_delimiter():
    """Page-controlled metadata cannot collide two contexts under ``|``.join.

    A role like ``"a|b"`` is attacker-shaped input: left raw, two distinct
    signature tuples would merge into one estimate key and one cell would
    silently overwrite the other's evidence in reporting.
    """
    def assignment(run_id, arm, m_role, p_role):
        return {
            "event": "experiment_assigned",
            "run_id": run_id,
            "experiment": {
                "experiment_id": f"e{run_id}",
                "arm": arm,
                "assignment_probability": 0.5,
                "model_choice_kind": "click",
                "model_choice_role": m_role,
                "proposal_id": "p",
                "proposal_kind": "click",
                "proposal_role": p_role,
            },
        }

    events = [
        assignment("a1", "candidate", "a|b", "c"),
        assignment("a2", "control", "a|b", "c"),
        assignment("b1", "candidate", "a", "b|c"),
        assignment("b2", "control", "a", "b|c"),
    ]
    trials = CounterfactualTrials.fit(events)
    # Two distinct cells, two distinct estimate keys — the adversarial values
    # sanitized so neither silently overwrote the other's evidence.
    assert len(trials.estimate()) == 2
    for key in trials.estimate():
        assert len(key.split("|")) == 14  # exactly the 14 coordinates joined
    # Stored cells are normalized on load as well: a legacy or hostile cell
    # cannot smuggle the delimiter back in.
    loaded = CounterfactualTrials.from_dict(
        {
            "cells": [
                ["f|x", "", "click", "unknown", "a|b", "unknown", "unknown",
                 "unknown", "click", "unknown", "c", "unknown", "unknown",
                 "unknown", "candidate"] + [1] * 20
            ]
        }
    )
    assert all("|" not in str(v) for v in loaded.cells[0][:15])


def test_causal_override_steps_stay_out_of_observational_fit():
    """An active-mode override is the causal prior's choice, not the policy's.

    The step is tagged ``causal_override`` on the transition — it must not
    teach the observational prior that the recorded policy chose it, exactly
    like an experiment-tagged step.
    """
    events = []
    for i in range(10):
        run = f"o{i}"
        events += [
            {"event": "run_started", "run_id": run, "task_key": "t", "goal": "g",
             "task_family": "f", "policy": ExplorationPolicy().to_dict()},
            {"event": "transition", "run_id": run, "task_key": "t",
             "selected": {"id": "pX", "kind": "fill", "label": "P"},
             "candidates": [{"id": "pX", "kind": "fill", "goal_overlap": 2}],
             "page_changed": True,
             "causal_override": {"id": "pX", "mode": "active"}},
            {"event": "run_finished", "run_id": run, "task_key": "t",
             "status": "done", "verified": True},
        ]
    model = ChoiceModel.fit(events)
    assert model.samples == 0
    assert model.predict(kind="fill", goal_overlap=2, rank=0, task_family="f")["n"] == 0


def test_signature_role_rank_and_phase_are_queryable():
    """The richer signature coordinates are live, not just stored.

    A query that supplies matching role values resolves at ``full``; one whose
    roles differ must visibly back off through ``minus_phase_rank_role`` —
    while still answering from the pooled evidence underneath.
    """
    events = []
    for i in range(12):
        meta = _trial_meta(
            f"s{i}", "candidate", task_family="f",
        )
        meta["model_choice_role"] = "button"
        meta["proposal_role"] = "button"
        events += [
            {"event": "run_started", "run_id": f"s{i}", "task_key": "t", "goal": "g",
             "task_family": "f"},
            {"event": "experiment_assigned", "run_id": f"s{i}", "task_key": "t",
             "experiment": meta},
            {"event": "run_finished", "run_id": f"s{i}", "task_key": "t",
             "status": "done", "verified": True},
        ]
    for i in range(12):
        meta = _trial_meta(f"c{i}", "control", task_family="f")
        meta["model_choice_role"] = "button"
        meta["proposal_role"] = "button"
        events += [
            {"event": "run_started", "run_id": f"c{i}", "task_key": "t", "goal": "g",
             "task_family": "f"},
            {"event": "experiment_assigned", "run_id": f"c{i}", "task_key": "t",
             "experiment": meta},
            {"event": "run_finished", "run_id": f"c{i}", "task_key": "t",
             "status": "blocked", "verified": False},
        ]
    trials = CounterfactualTrials.fit(events)
    matched = trials.resolve(
        task_family="f", model_role="button", proposal_role="button"
    )
    assert matched["signature_level"] == "full"
    assert matched["effect_status"] == "beneficial"
    divergent = trials.resolve(
        task_family="f", model_role="link", proposal_role="link"
    )
    assert divergent is not None
    assert divergent["signature_level"] == "minus_phase_rank_role"
    # The backoff still answers — a mismatched role generalizes over the cell,
    # it does not erase the evidence.
    assert divergent["effect_status"] == "beneficial"
    # And the TrialChoiceModel query path exercises the same coordinates.
    model = TrialChoiceModel(trials=trials)
    estimate = model._estimate(
        proposal_kind="click", proposal_role="link", model_kind="click",
        model_role="link", task_family="f",
    )
    assert estimate["signature_level"] == "minus_phase_rank_role"
