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

import pytest

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
               latency_ms=0, tokens=0, approvals=0, duplicate=False,
               page_changed=True, stale_or_failure=0, risk_events=0,
               extra_transitions=0, extra_risk_events=0, extra_stale=0):
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
            "page_changed": page_changed,
            "latency_ms": latency_ms,
            "model_calls": 1,
            "tokens": tokens,
            "stale_or_failure": stale_or_failure,
            "risk_events": risk_events,
            "experiment": meta,
        })
        # Untagged extra transitions extend the run's real trajectory —
        # they count toward steps/cost/authority markers without being the
        # assigned step itself.
        for _ in range(int(extra_transitions)):
            transitions.append({
                "event": "transition",
                "state": "S",
                "next_state": "D",
                "selected": {"id": f"aux{run_id}", "kind": kind},
                "candidates": [],
                "page_changed": False,
                "latency_ms": 0,
                "model_calls": 1,
                "tokens": 0,
                "stale_or_failure": extra_stale,
                "risk_events": extra_risk_events,
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
    assert arms["delta_cs"][0] <= 0 <= arms["delta_cs"][1]
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
    separated arms do establish in both directions — the anytime-valid
    sequence is stricter than the old approximate interval, so this needs
    more than the old eight per arm."""
    positive = []
    negative = []
    for i in range(24):
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
    enough. Establishment uses the confidence sequence, so data that
    merely looks favorable at one peek stays unresolved."""
    events = []
    for i in range(8):
        events += _trial_run(f"c{i}", "candidate", success=i < 7)
        events += _trial_run(f"k{i}", "control", success=i < 3)
    arms = _cell(events)
    assert arms["delta_ci"][0] > 0  # a single fixed-sample look excludes zero
    assert arms["effect_status"] == "unresolved"  # the sequence does not
    assert arms["delta_cs"][0] < 0
    assert arms["delta_cs"][0] <= arms["delta_ci"][0]
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
    effect can still establish — provided the worst-case bounds confirm the
    censored outcomes could not erase the margin."""
    events = []
    for i in range(24):
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
    assert abs(arms["candidate"]["censor_rate"] - 2 / 26) < 1e-9
    assert arms["censoring_imbalance"] is False
    # The bounds still contain a positive margin: even if every missing
    # outcome broke against the candidate, the effect survives.
    assert arms["delta_bounds"][0] > 0
    assert arms["effect_status"] == "beneficial"


def test_balanced_heavy_censoring_vetoes_a_sign_the_bounds_cannot_confirm():
    """The subtler censoring failure the rate gates cannot see: perfectly
    balanced censoring, but enough of it that worst-case bounds cross zero.
    A candidate arm that is all surviving successes must not establish when
    half its assignments are missing."""
    events = []
    for i in range(24):
        events += _trial_run(f"c{i}", "candidate", success=True)
        events += _trial_run(f"k{i}", "control", success=False)
    for i in range(24):
        events += _trial_run(f"ca{i}", "candidate", status="aborted", verified=False,
                             reason="operator_cancel")
        events += _trial_run(f"ka{i}", "control", status="aborted", verified=False,
                             reason="operator_cancel")
    arms = _cell(events)
    # Rate gates pass: 50% censoring is not *over* the max, and the gap is 0.
    assert arms["censoring_imbalance"] is False
    assert arms["support_sufficient"] is True
    # The point estimate looks perfect; the worst-case bound does not —
    # every censored control could have succeeded and every censored
    # candidate could have failed, flipping the sign.
    assert arms["delta_bounds"][0] <= 0 <= arms["delta_bounds"][1]
    assert arms["effect_status"] == "unresolved"
    assert arms["unresolved_reason"] == "censoring_bounds_cross_threshold"


# ------------------------------------------------- practical-effect thresholds


def test_minimum_effect_requires_practical_significance():
    """Statistically established is not automatically worthwhile: a family
    may demand a minimum absolute improvement."""
    events = []
    for i in range(24):
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
    for i in range(24):
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
    """A never-randomized exact action inherits only *class-level* evidence.

    The estimate still resolves — shadow and canary see the class effect —
    but ``generalization_level`` reports ``same_context_class`` /
    ``pooled_class``, never ``exact``, and the active gate refuses a
    deterministic override: class-level exchangeability is a canary
    hypothesis, not deployment authority. Only a completed contextual
    canary (recorded in ``confirmed_action_keys``) or genuinely exact
    randomized evidence opens the active path.
    """
    from jev_ultrafast.dreamlearn import action_key

    events = []
    for i in range(24):
        events += _trial_run(f"c{i}", "candidate", success=True, proposal_effect="navigate")
        events += _trial_run(f"k{i}", "control", success=False, proposal_effect="navigate")
    model = TrialChoiceModel.fit(events)
    candidates = [
        {"id": "m", "kind": "click", "goal_overlap": 0, "label": "M"},
        # Same treatment *class* (kind + effect) as the measured evidence,
        # but a different semantic identity — this exact action was never
        # the randomized arm.
        {"id": "never-randomized-before", "kind": "click", "goal_overlap": 1,
         "effect": "navigate", "label": "Checkout"},
    ]
    model_choice = {"id": "m", "kind": "click", "goal_overlap": 0}
    ranked = model.rank(candidates, model_choice=model_choice)
    assert ranked and ranked[0]["id"] == "never-randomized-before"
    assert ranked[0]["source"] == "randomized"
    # The class estimate still generalizes — the signature machinery is
    # unchanged — but provenance now names it class-level, never exact.
    assert ranked[0]["effect_status"] == "beneficial"
    assert ranked[0]["generalization_level"] in {
        "same_context_class", "pooled_class"}
    assert ranked[0]["exact_action_randomized"] is False

    shadow = CausalChoicePolicy(mode="shadow", trial_model=model)
    assert shadow.proposal(candidates, model_choice=model_choice) is None
    # Pooled class evidence annotates and nominates canaries — it cannot
    # drive a deterministic override in a context it never measured.
    active = CausalChoicePolicy(mode="active", trial_model=model)
    pooled_entry = active.rank(candidates, model_choice=model_choice)["proposals"][0]
    assert pooled_entry["execution_blocker"] == "pooled_only"
    assert active.proposal(candidates, model_choice=model_choice) is None
    # Even at a context-specific stratum, class-level evidence alone is not
    # active authority for an action that was never itself randomized.
    family_events = []
    for i in range(24):
        family_events += _trial_run(
            f"fc{i}", "candidate", success=True, proposal_effect="navigate",
            task_family="f")
        family_events += _trial_run(
            f"fk{i}", "control", success=False, proposal_effect="navigate",
            task_family="f")
    family_model = TrialChoiceModel.fit(family_events)
    scoped = CausalChoicePolicy(mode="active", trial_model=family_model)
    class_entry = scoped.rank(
        candidates, model_choice=model_choice, task_family="f",
    )["proposals"][0]
    assert class_entry["causal"]["trial_level"] == "family"
    assert class_entry["causal"]["generalization_level"] == "same_context_class"
    assert class_entry["execution_blocker"] == "unverified_generalization"
    assert scoped.proposal(
        candidates, model_choice=model_choice, task_family="f",
    ) is None
    # A completed contextual canary — recorded by the operator as the action
    # key it confirmed — is the explicit bridge from class evidence to
    # active authority for this exact action.
    novel_key = action_key(kind="click", effect="navigate", label="Checkout")
    confirmed = CausalChoicePolicy(
        mode="active", trial_model=family_model,
        confirmed_action_keys=(novel_key,),
    )
    confirmed_entry = confirmed.rank(
        candidates, model_choice=model_choice, task_family="f",
    )["proposals"][0]
    assert confirmed_entry["executable"] is True
    assert confirmed.proposal(
        candidates, model_choice=model_choice, task_family="f",
    ) is not None
    # And genuinely exact evidence — the same semantic identity that was
    # randomized — needs no confirmation at all.
    exact_candidates = [
        candidates[0],
        {"id": "p2", "kind": "click", "goal_overlap": 1,
         "effect": "navigate", "label": "Go"},
    ]
    exact_entry = scoped.rank(
        exact_candidates, model_choice=model_choice, task_family="f",
    )["proposals"][0]
    assert exact_entry["causal"]["generalization_level"] == "exact"
    assert exact_entry["causal"]["exact_action_randomized"] is True
    assert exact_entry["executable"] is True
    # An *unresolved* class-level estimate blocks deterministic execution.
    thin = TrialChoiceModel.fit(_trial_run("a", "candidate") + _trial_run("b", "control"))
    assert CausalChoicePolicy(mode="active", trial_model=thin).proposal(
        candidates, model_choice=model_choice
    ) is None


def test_generalization_level_tracks_site_and_semantic_identity():
    """Site-level class evidence is ``same_context_class`` at the site
    stratum; a materially changed label is a different action entirely —
    neither becomes ``exact`` without that identity being randomized."""
    events = []
    for i in range(24):
        events += _trial_run(f"c{i}", "candidate", success=True,
                             proposal_effect="navigate", site="h")
        events += _trial_run(f"k{i}", "control", success=False,
                             proposal_effect="navigate", site="h")
    model = TrialChoiceModel.fit(events)
    model_choice = {"id": "m", "kind": "click", "goal_overlap": 0, "label": "M"}
    policy = CausalChoicePolicy(mode="active", trial_model=model)

    # The randomized action itself — exact at the site stratum.
    exact = policy.rank(
        [{"id": "m", "kind": "click", "goal_overlap": 0, "label": "M"},
         {"id": "p", "kind": "click", "goal_overlap": 1,
          "effect": "navigate", "label": "Go"}],
        model_choice=model_choice, site="h",
    )["proposals"][0]
    assert exact["causal"]["trial_level"] == "site"
    assert exact["causal"]["generalization_level"] == "exact"
    assert exact["executable"] is True

    # Same class, different site evidence: at family scope the site cells
    # still answer (family backs off to pooled when no family coordinate
    # exists) — the action key was randomized *somewhere*, so the pooled
    # stratum reports cross-context class evidence, never pooled-fresh.
    other_site = policy.rank(
        [{"id": "m", "kind": "click", "goal_overlap": 0, "label": "M"},
         {"id": "p", "kind": "click", "goal_overlap": 1,
          "effect": "navigate", "label": "Go"}],
        model_choice=model_choice, site="other",
    )["proposals"][0]
    assert other_site["causal"]["trial_level"] == "pooled"
    assert other_site["causal"]["generalization_level"] == "cross_context_class"
    # ``exact_action_randomized`` mirrors ``exact`` exactly — deployable-
    # precision evidence only. The broader fact lives under its own name.
    assert other_site["causal"]["exact_action_randomized"] is False
    assert other_site["causal"]["action_randomized_anywhere"] is True
    assert other_site["execution_blocker"] == "pooled_only"

    # Materially changed semantics — same id shape, different label — is a
    # different action: class-level evidence only, blocked for active.
    relabeled = policy.rank(
        [{"id": "m", "kind": "click", "goal_overlap": 0, "label": "M"},
         {"id": "p", "kind": "click", "goal_overlap": 1,
          "effect": "navigate", "label": "Proceed to payment"}],
        model_choice=model_choice, site="h",
    )["proposals"][0]
    assert relabeled["causal"]["generalization_level"] == "same_context_class"
    assert relabeled["execution_blocker"] == "unverified_generalization"
    assert relabeled["executable"] is False

    # A different element role is a different action identity too — a
    # randomized "Go" *link* never proves a "Go" *button*.
    different_role = policy.rank(
        [{"id": "m", "kind": "click", "goal_overlap": 0, "label": "M"},
         {"id": "p", "kind": "click", "goal_overlap": 1, "role": "link",
          "effect": "navigate", "label": "Go"}],
        model_choice=model_choice, site="h",
    )["proposals"][0]
    assert different_role["causal"]["generalization_level"] == "same_context_class"
    assert different_role["execution_blocker"] == "unverified_generalization"


def test_unkeyed_query_reports_none_and_fails_closed():
    """A proposal with no provable action identity reports
    ``generalization_level: "none"`` — distinguishable from provably
    class-only evidence — and the active gate refuses it."""
    model = TrialChoiceModel.fit(_scoped_events())
    policy = CausalChoicePolicy(mode="active", trial_model=model)
    unlabeled = [
        {"id": "m", "kind": "click", "goal_overlap": 0},
        {"id": "p", "kind": "click", "goal_overlap": 1},
    ]
    entry = policy.rank(
        unlabeled, model_choice={"id": "m", "kind": "click", "goal_overlap": 0},
        task_family="f", site="h",
    )["proposals"][0]
    assert entry["causal"]["generalization_level"] == "none"
    assert entry["causal"]["exact_action_randomized"] is False
    assert entry["execution_blocker"] == "unverified_generalization"
    assert entry["executable"] is False


def test_contextual_canary_bridge_is_evidence_derived():
    """``action_randomized_in_context`` is the in-model canary record: an
    action that was itself randomized inside the answering context — even
    under a different enforced signature — bridges class-level evidence to
    active authority, with no operator-supplied confirmation needed."""
    from jev_ultrafast.dreamlearn import action_key

    def _assigned(run_id, arm, *, key_label, p_overlap, success):
        meta = _trial_meta(run_id, arm, task_family="f", site="h")
        meta["proposal_effect"] = "navigate"
        meta["model_choice_offered_rank"] = 0
        meta["proposal_offered_rank"] = 1
        meta["proposal_overlap"] = p_overlap
        meta["proposal_action_key"] = action_key(
            kind="click", effect="navigate", label=key_label)
        return [
            {"event": "run_started", "run_id": run_id, "task_key": "t",
             "goal": "g", "task_family": "f"},
            {"event": "experiment_assigned", "run_id": run_id, "task_key": "t",
             "experiment": meta},
            {"event": "run_finished", "run_id": run_id, "task_key": "t",
             "status": "done" if success else "blocked", "verified": success},
        ]

    events = []
    # "Ship" was randomized at overlap 1 — the class evidence the strict
    # mask will answer from for an overlap-1 query.
    for i in range(24):
        events += _assigned(f"sc{i}", "candidate", key_label="Ship",
                            p_overlap=1, success=True)
        events += _assigned(f"sk{i}", "control", key_label="Ship",
                            p_overlap=1, success=False)
    # "Go" was itself randomized in this context — a real contextual canary —
    # but under overlap 0, outside the enforced signature that answers.
    for i in range(6):
        events += _assigned(f"gc{i}", "candidate", key_label="Go",
                            p_overlap=0, success=True)
        events += _assigned(f"gk{i}", "control", key_label="Go",
                            p_overlap=0, success=False)
    model = TrialChoiceModel.fit(events)
    policy = CausalChoicePolicy(mode="active", trial_model=model)
    model_choice = {"id": "m", "kind": "click", "goal_overlap": 0,
                    "label": "M"}
    candidates = [
        {"id": "m", "kind": "click", "goal_overlap": 0, "label": "M"},
        {"id": "go", "kind": "click", "goal_overlap": 1,
         "effect": "navigate", "label": "Go"},
    ]
    entry = policy.rank(
        candidates, model_choice=model_choice,
        task_family="f", site="h")["proposals"][0]
    assert entry["causal"]["generalization_level"] == "same_context_class"
    assert entry["causal"]["exact_action_randomized"] is False
    assert entry["causal"]["action_randomized_in_context"] is True
    assert entry["execution_blocker"] is None
    assert entry["executable"] is True
    # An action that was never randomized in this context still fails —
    # confirmation is derived, never assumed.
    novel = [
        candidates[0],
        {"id": "n", "kind": "click", "goal_overlap": 1,
         "effect": "navigate", "label": "Never tried"},
    ]
    blocked = policy.rank(
        novel, model_choice=model_choice,
        task_family="f", site="h")["proposals"][0]
    assert blocked["causal"]["action_randomized_in_context"] is False
    assert blocked["execution_blocker"] == "unverified_generalization"


def test_generalization_level_survives_serialization():
    """``action_index`` round-trips through to_dict/from_dict — a stored
    model answers exactness questions exactly as the fitted one did."""
    events = _trials("candidate", 24, 24, "c", task_family="f") + _trials(
        "control", 24, 0, "k", task_family="f")
    fitted = TrialChoiceModel.fit(events)
    clone = TrialChoiceModel.from_dict(fitted.to_dict())
    entry = clone.rank(
        _CANDIDATES, model_choice=_MODEL_CHOICE, task_family="f")[0]
    assert entry["generalization_level"] == "exact"
    assert entry["exact_action_randomized"] is True
    # Older payloads without the index degrade to class-level, never exact.
    payload = fitted.to_dict()
    payload["trials"]["action_index"] = ()
    degraded = TrialChoiceModel.from_dict(payload)
    old = degraded.rank(
        _CANDIDATES, model_choice=_MODEL_CHOICE, task_family="f")[0]
    assert old["generalization_level"] == "same_context_class"
    assert old["exact_action_randomized"] is False


# ------------------------------------------------------ duplicates / integrity


def test_duplicate_assignment_records_count_once():
    """The same assignment written twice is one unit of analysis — counting
    it twice would double-weight that arm. The duplicate is reported."""
    events = _trial_run("dup", "candidate", success=True, duplicate=True)
    trials = CounterfactualTrials.fit(events)
    arm = trials.estimate()[next(iter(trials.estimate()))]["candidate"]
    assert arm["assigned"] == 1
    assert trials.duplicates == 1


def test_two_distinct_assignments_invalidate_the_experimental_unit():
    """One randomized assignment per run — enforced, not trusted.

    Two *different* assignment decisions inside one run cannot each be
    credited with the run's single outcome; the whole unit is quarantined,
    counted under ``invalid_units``, and contributes nothing to any arm.
    """
    meta_a = _trial_meta("u", "candidate")
    meta_b = _trial_meta("u", "control")
    meta_b["experiment_id"] = "xu-other"  # a *different* assignment decision
    events = [
        {"event": "run_started", "run_id": "u", "task_key": "t", "goal": "g"},
        {"event": "experiment_assigned", "run_id": "u", "task_key": "t",
         "experiment": meta_a},
        {"event": "experiment_assigned", "run_id": "u", "task_key": "t",
         "experiment": meta_b},
        {"event": "run_finished", "run_id": "u", "task_key": "t",
         "status": "done", "verified": True},
    ]
    trials = CounterfactualTrials.fit(events)
    assert trials.invalid_units == 1
    assert not trials.estimate()  # no arm may absorb the spliced unit
    # Three distinct assignments are rejected the same way.
    meta_c = _trial_meta("u", "candidate")
    meta_c["experiment_id"] = "xu-third"
    events.insert(4, {"event": "experiment_assigned", "run_id": "u",
                      "task_key": "t", "experiment": meta_c})
    trials3 = CounterfactualTrials.fit(events)
    assert trials3.invalid_units == 1
    assert not trials3.estimate()


def test_identical_duplicate_assignments_still_form_one_unit():
    """Deduplication is by assignment *identity*: the same decision written
    twice is one unit (``duplicates``), not an invalid experiment."""
    meta = _trial_meta("u", "candidate")
    events = [
        {"event": "run_started", "run_id": "u", "task_key": "t", "goal": "g"},
        {"event": "experiment_assigned", "run_id": "u", "task_key": "t",
         "experiment": meta},
        {"event": "experiment_assigned", "run_id": "u", "task_key": "t",
         "experiment": dict(meta)},
        {"event": "run_finished", "run_id": "u", "task_key": "t",
         "status": "done", "verified": True},
    ]
    trials = CounterfactualTrials.fit(events)
    assert trials.invalid_units == 0
    assert trials.duplicates == 1
    arm = trials.estimate()[next(iter(trials.estimate()))]["candidate"]
    assert arm["assigned"] == 1


def test_assignment_after_terminal_is_not_a_trial_of_that_run():
    """An assignment ordered after the run's terminal event cannot have
    produced the outcome it would be credited with — the unit is invalid."""
    meta = _trial_meta("u", "candidate")
    events = [
        {"event": "run_started", "run_id": "u", "task_key": "t", "goal": "g",
         "sequence": 0},
        {"event": "run_finished", "run_id": "u", "task_key": "t",
         "status": "done", "verified": True, "sequence": 1},
        {"event": "experiment_assigned", "run_id": "u", "task_key": "t",
         "experiment": meta, "sequence": 2},
    ]
    trials = CounterfactualTrials.fit(events)
    assert trials.invalid_units == 1
    assert not trials.estimate()


def test_malformed_assignment_records_are_rejected_and_counted():
    """A record that fails the unit gate — arm outside {candidate, control}
    or propensity outside (0, 1] — is rejected before it can shape a cell."""
    good = _trial_run("g", "candidate", success=True) + _trial_run(
        "h", "control", success=False)
    bad_meta = _trial_meta("b", "candidate", propensity=0.0)
    events = good + [
        {"event": "run_started", "run_id": "b", "task_key": "t", "goal": "g"},
        {"event": "experiment_assigned", "run_id": "b", "task_key": "t",
         "experiment": bad_meta},
        {"event": "run_finished", "run_id": "b", "task_key": "t",
         "status": "done", "verified": True},
    ]
    trials = CounterfactualTrials.fit(events)
    assert trials.rejected == 1
    # The valid runs still analyze; the malformed one contributed nothing.
    total = sum(
        cell[arm]["assigned"]
        for cell in trials.estimate().values()
        for arm in ("candidate", "control")
        if isinstance(cell.get(arm), dict)
    )
    assert total == 2


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
    assert trials.version == "jev-trials/9"
    cell = trials.cells[0]
    assert cell[0] == "flights" and cell[1] == ""
    assert cell[3] == "unknown" and cell[6] == "unknown"  # effect/rank wildcards
    # v8 appends the outcome-vector block after the reason tail — the last
    # reason counter sits at -1-_OUTCOME_STATS, so read the tail explicitly.
    from jev_ultrafast.dreamlearn import _REASON_BASE, TERMINATION_REASONS
    reasons = cell[_REASON_BASE:_REASON_BASE + len(TERMINATION_REASONS)]
    assert reasons[-1] == 2  # unknown_abort count survived
    entry = trials.estimate()[next(iter(trials.estimate()))]["candidate"]
    assert entry["assigned"] == 10 and entry["trials"] == 8 and entry["censored"] == 2


def test_v5_cells_migrate_with_zeroed_indeterminate_counter():
    """v5 cells (8 reason counters) migrate through v6 to the v7 layout: the
    ``indeterminate_execution`` slot splices in as zero, the censored-weight
    statistic starts empty for imputation, and every recorded verdict —
    including the unverified_claim counter that used to sit at the same tail
    position — keeps its count."""
    from jev_ultrafast.dreamlearn import (
        _REASON_BASE,
        TERMINATION_REASONS,
        TRIAL_CELL_LEN,
    )

    reasons_v5 = [0, 1, 0, 0, 0, 0, 3, 2]  # v5 tail order, ending unverified/unknown
    key = ("flights", "example.test", "click", "navigate", "button", "0", "0-4", "0-2",
           "click", "navigate", "link", "1", "0-4", "0-2", "candidate")
    stats_v5 = [10, 8, 8, 2, 6.0, 8.0, 8.0, 8.0, 8.0]  # v5/v6 stats: no w_censored
    v5 = (*key, *stats_v5, 1.0, 2.0, 0.5, *reasons_v5)
    assert len(v5) == 35
    trials = CounterfactualTrials.from_dict({"version": "jev-trials/5", "cells": [v5]})
    assert trials.version == "jev-trials/9"
    cell = trials.cells[0]
    assert len(cell) == TRIAL_CELL_LEN
    # The v8 outcome-vector block is appended after the reason tail; the
    # reasons themselves keep their v7 positions.
    tail = cell[_REASON_BASE:_REASON_BASE + len(TERMINATION_REASONS)]
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
    def build(candidate_censored, analyzed):
        events = []
        for i in range(analyzed):
            events += _trial_run(f"c{i}", "candidate", success=True)
        for i in range(candidate_censored):
            events += _trial_run(f"a{i}", "candidate", status="aborted", verified=False,
                                 reason="operator_cancel")
        for i in range(analyzed):
            events += _trial_run(f"k{i}", "control", success=False)
        return _cell(events)

    extreme = build(8, 8)  # 8/16 = 0.5 censored against 0.0 — gap 0.5 > 0.25
    assert extreme["censoring_imbalance"] is True
    assert extreme["effect_status"] == "unresolved"

    balanced = build(2, 24)  # 2/26 ≈ 0.077, gap ≈ 0.077 — tolerated
    assert balanced["censoring_imbalance"] is False
    assert balanced["effect_status"] == "beneficial"


def test_minimum_effect_boundary_is_strict():
    """Establishment requires the sequence to lie entirely *above* the
    threshold — a bound touching it is not established."""
    events = []
    for i in range(24):
        events += _trial_run(f"c{i}", "candidate", success=True)
        events += _trial_run(f"k{i}", "control", success=False)
    trials = CounterfactualTrials.fit(events)
    resolved = trials.resolve(
        model_kind="click", model_overlap=0, proposal_kind="click", proposal_overlap=1)
    lower = resolved["delta_cs"][0]
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
    for i in range(24):
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
    for i in range(24):
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
    for i in range(24):
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
    for i in range(24):
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
    for i in range(24):
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


# --------------------------------------- causal probability propagation
#
# The candidate-side success probability is control_p + delta, derived at the
# estimation layer — never 0.5 + delta. These tests pin the regression where
# CausalChoicePolicy.proposal() fabricated a 0.5 control rate.


def _trials(arm, count, successes, prefix, **kw):
    return [
        event
        for i in range(count)
        for event in _trial_run(f"{prefix}{i}", arm, success=(i < successes), **kw)
    ]


_CANDIDATES = [
    {"id": "m", "kind": "click", "goal_overlap": 0, "label": "M"},
    # The label matches what _trial_run records in its transition catalogue —
    # exact-action evidence (Phase 3) is only provable when the offered
    # action's identity equals a randomized one.
    {"id": "p", "kind": "click", "goal_overlap": 1, "label": "Go"},
]
_MODEL_CHOICE = {"id": "m", "kind": "click", "goal_overlap": 0}


def _causal_entry(model, candidates=_CANDIDATES, model_choice=_MODEL_CHOICE, **kw):
    ranked = model.rank(candidates, model_choice=model_choice, **kw)
    assert len(ranked) == 1
    return ranked[0]


def test_causal_p_progress_is_control_rate_plus_delta_not_half_plus_delta():
    """control_p = 0.80, delta = +0.10 → p_progress = 0.90.

    The fabricated-0.5 formula would report 0.60 — silently inverting the
    ordering against any candidate whose true control rate exceeds 0.5."""
    events = _trials("candidate", 20, 18, "c") + _trials("control", 20, 16, "k")
    model = TrialChoiceModel.fit(events)
    entry = _causal_entry(model)
    assert entry["control_p"] == pytest.approx(0.80)
    assert entry["expected_delta"] == pytest.approx(0.10)
    assert entry["p_progress"] == pytest.approx(0.90)
    # The policy consumes the propagated value, not a reconstruction.
    policy = CausalChoicePolicy(mode="canary", trial_model=model)
    proposal = policy.proposal(_CANDIDATES, model_choice=_MODEL_CHOICE)
    assert proposal["p_progress"] == pytest.approx(0.90)
    assert proposal["causal_p_progress"] == pytest.approx(0.90)
    assert proposal["control_p"] == pytest.approx(0.80)


def test_causal_p_progress_low_control_rate():
    """control_p = 0.10, delta = +0.20 → p_progress = 0.30.

    Under the defect this returned 0.70 — a weak candidate inflated past the
    honest value by the assumed 0.5 baseline."""
    events = _trials("candidate", 20, 6, "c") + _trials("control", 20, 2, "k")
    model = TrialChoiceModel.fit(events)
    entry = _causal_entry(model)
    assert entry["control_p"] == pytest.approx(0.10)
    assert entry["expected_delta"] == pytest.approx(0.20)
    assert entry["p_progress"] == pytest.approx(0.30)
    policy = CausalChoicePolicy(mode="canary", trial_model=model)
    proposal = policy.proposal(_CANDIDATES, model_choice=_MODEL_CHOICE)
    assert proposal["p_progress"] == pytest.approx(0.30)


class _StaticTrials:
    """Stand-in for CounterfactualTrials returning a fixed contrast.

    Real Bernoulli evidence can never produce |control + delta| outside
    [0, 1] — the clamp is a defensive bound exercised here through the real
    ``resolve`` consumer (``predict``) rather than by fabricating cells."""

    def __init__(self, estimate):
        self._estimate = estimate

    def resolve(self, **kwargs):
        return self._estimate


def test_causal_p_progress_clamps_high():
    """control_p = 0.95, delta = +0.20 → 1.0 after the [0, 1] clamp."""
    model = TrialChoiceModel(trials=_StaticTrials({
        "control": {"p_success": 0.95, "trials": 20},
        "candidate": {"p_success": 1.0, "trials": 20},
        "delta": 0.20,
        "effect_status": "beneficial",
        "support_sufficient": True,
        "level": "pooled",
    }))
    prediction = model.predict(kind="click", goal_overlap=1)
    assert prediction["p_progress"] == pytest.approx(1.0)
    assert prediction["control_p"] == pytest.approx(0.95)
    entry = _causal_entry(model)
    assert entry["p_progress"] == pytest.approx(1.0)


def test_causal_p_progress_clamps_low():
    """control_p = 0.05, delta = −0.20 → 0.0 after the [0, 1] clamp."""
    model = TrialChoiceModel(trials=_StaticTrials({
        "control": {"p_success": 0.05, "trials": 20},
        "candidate": {"p_success": 0.0, "trials": 20},
        "delta": -0.20,
        "effect_status": "unresolved",
        "support_sufficient": True,
        "level": "pooled",
    }))
    prediction = model.predict(kind="click", goal_overlap=1)
    assert prediction["p_progress"] == pytest.approx(0.0)
    entry = _causal_entry(model)
    assert entry["p_progress"] == pytest.approx(0.0)


def test_causal_support_without_control_arm_fabricates_no_probability():
    """A contrast that claims support but has no valid finite control
    probability yields p_progress = None — never 0.5 + delta, and the
    observational estimate is not silently substituted as if it were the
    causal one."""
    # Estimation layer: control arm missing or non-finite → no probability.
    missing = TrialChoiceModel(trials=_StaticTrials({
        "candidate": {"p_success": 0.9, "trials": 20},
        "delta": 0.3,
        "effect_status": "beneficial",
        "support_sufficient": True,
        "level": "pooled",
    }))
    prediction = missing.predict(kind="click", goal_overlap=1)
    assert prediction["p_progress"] is None
    assert prediction["control_p"] is None
    nonfinite = TrialChoiceModel(trials=_StaticTrials({
        "control": {"p_success": float("nan"), "trials": 20},
        "candidate": {"p_success": 0.9, "trials": 20},
        "delta": 0.3,
        "effect_status": "beneficial",
        "support_sufficient": True,
        "level": "pooled",
    }))
    assert nonfinite.predict(kind="click", goal_overlap=1)["p_progress"] is None

    # Policy layer: a supported causal entry with no propagated probability
    # reports None — the observational channel's own value stays separate.
    class _StubTrialModel:
        def rank(self, candidates, **kwargs):
            return [{
                "id": "p", "kind": "click",
                "expected_delta": 0.3, "control_p": None, "p_progress": None,
                "uncertainty": None, "delta_ci": None,
                "effect_status": "beneficial", "support_sufficient": True,
                "trial_level": "pooled", "signature_level": "full",
                "source": "randomized",
            }]

    class _StubChoiceModel:
        def predict(self, **kwargs):
            return {"p_progress": 0.9, "uncertainty": 0.05, "confident": True,
                    "source": "observational"}

    policy = CausalChoicePolicy(
        mode="canary", trial_model=_StubTrialModel(), choice_model=_StubChoiceModel()
    )
    proposal = policy.proposal(_CANDIDATES, model_choice=_MODEL_CHOICE)
    assert proposal["p_progress"] is None
    assert proposal["causal_p_progress"] is None
    assert proposal["observational_p_progress"] == pytest.approx(0.9)


def _observational_run(run_id, *, verified=True):
    """A non-experimental run selecting the rank-1 click — feeds only the
    observational ChoiceModel (experiment-tagged transitions are excluded
    from its fit by construction)."""
    return [
        {"event": "run_started", "run_id": run_id, "task_key": "t", "goal": "g",
         "policy": ExplorationPolicy().to_dict(),
         "policy_digest": ExplorationPolicy().digest},
        {"event": "transition", "run_id": run_id, "task_key": "t",
         "state": "S", "next_state": "D",
         "selected": {"id": "p", "kind": "click", "goal_overlap": 1},
         "candidates": [
             {"id": "m", "kind": "click", "label": "M", "goal_overlap": 0},
             {"id": "p", "kind": "click", "label": "P", "goal_overlap": 1},
         ],
         "page_changed": True, "latency_ms": 0, "model_calls": 1,
         "tokens": 0, "stale_or_failure": 0, "risk_events": 0},
        {"event": "run_finished", "run_id": run_id, "task_key": "t",
         "status": "done", "verified": verified},
    ]


def test_observational_and_causal_probabilities_keep_separate_provenance():
    """The two channels disagree here — observational ≈ 0.91, causal = 0.30 —
    and both stay on the proposal under their own names. The supported
    randomized value governs ``p_progress``; the observational value is never
    averaged in or relabeled."""
    causal_events = _trials("candidate", 20, 6, "c") + _trials("control", 20, 2, "k")
    obs_events = [e for i in range(9) for e in _observational_run(f"o{i}")]
    trial_model = TrialChoiceModel.fit(causal_events)
    choice_model = ChoiceModel.fit(obs_events)
    policy = CausalChoicePolicy(
        mode="canary", trial_model=trial_model, choice_model=choice_model
    )
    proposal = policy.proposal(_CANDIDATES, model_choice=_MODEL_CHOICE)
    # 10/11 ≈ 0.909 observational posterior vs 0.30 causal — far enough apart
    # that any mixing between the channels would be visible.
    assert proposal["observational_p_progress"] == pytest.approx(10 / 11)
    assert proposal["causal_p_progress"] == pytest.approx(0.30)
    assert proposal["p_progress"] == pytest.approx(0.30)
    assert proposal["source"] == "pooled_randomized"
    assert proposal["expected_delta"] == pytest.approx(0.20)


def test_active_ranking_follows_causal_not_observational_signal():
    """Two candidates where the channels deliberately disagree: the
    observational prior loves ``weak`` and shrugs at ``strong``, but the
    randomized evidence says ``strong`` is the larger established effect.
    Active ordering must deploy on causal terms — the observational score
    annotates the entry, it cannot reorder it."""
    class _DisagreeTrials:
        def rank(self, candidates, **kwargs):
            return [
                {"id": "strong", "kind": "click", "expected_delta": 0.25,
                 "control_p": 0.5, "p_progress": 0.75, "uncertainty": 0.1,
                 "delta_ci": [0.1, 0.4], "utility_delta": 0.2,
                 "effect_status": "beneficial", "support_sufficient": True,
                 "trial_level": "family", "signature_level": "full",
                 "generalization_level": "exact", "action_key": "k-strong",
                 "source": "randomized"},
                {"id": "weak", "kind": "click", "expected_delta": 0.05,
                 "control_p": 0.5, "p_progress": 0.55, "uncertainty": 0.1,
                 "delta_ci": [0.0, 0.1], "utility_delta": 0.04,
                 "effect_status": "beneficial", "support_sufficient": True,
                 "trial_level": "family", "signature_level": "full",
                 "generalization_level": "exact", "action_key": "k-weak",
                 "source": "randomized"},
            ]

    class _DisagreeChoice:
        def predict(self, *, rank=None, **kwargs):
            # Observational posterior favors the causally-weaker candidate
            # (offered at rank 1) over the causally-stronger one (rank 2).
            if rank == 1:
                return {"p_progress": 0.95, "uncertainty": 0.02,
                        "confident": True, "source": "observational"}
            return {"p_progress": 0.51, "uncertainty": 0.4,
                    "confident": False, "source": "observational"}

    candidates = [
        {"id": "m", "kind": "click", "goal_overlap": 0, "label": "M"},
        {"id": "weak", "kind": "click", "goal_overlap": 1, "label": "W"},
        {"id": "strong", "kind": "click", "goal_overlap": 1, "label": "S"},
    ]
    model_choice = {"id": "m", "kind": "click", "goal_overlap": 0}
    policy = CausalChoicePolicy(
        mode="active", trial_model=_DisagreeTrials(), choice_model=_DisagreeChoice()
    )
    proposals = policy.rank(
        candidates, model_choice=model_choice, task_family="f")["proposals"]
    # Combined shadow score would favor ``weak`` (0.05 + 0.5*(0.95-0.5)
    # beats 0.25 + 0.5*(0.51-0.5)); active ordering ignores it.
    assert proposals[0]["id"] == "strong"
    assert proposals[0]["deployment_score"] == pytest.approx(0.25)
    assert proposals[0]["observational_score"] < proposals[1]["observational_score"]
    # Separate score fields are populated — one overloaded ``score`` is no
    # longer the only ranking signal downstream consumers can see.
    for entry in proposals:
        assert {"causal_score", "observational_score",
                "experiment_priority_score", "deployment_score"} <= set(entry)
    assert policy.proposal(
        candidates, model_choice=model_choice, task_family="f")["id"] == "strong"


def test_provenance_dimensions_stay_separate_from_support():
    """An *unsupported* pooled contrast is still randomized evidence at the
    pooled level: ``evidence_origin`` and ``trial_level`` describe where it
    came from, ``support_status`` describes whether it sufficed — a thin
    pooled estimate is never relabeled as if it were context-specific."""
    thin = _trials("candidate", 3, 3, "c") + _trials("control", 3, 0, "k")
    model = TrialChoiceModel.fit(thin)
    policy = CausalChoicePolicy(mode="shadow", trial_model=model)
    entry = policy.rank(_CANDIDATES, model_choice=_MODEL_CHOICE)["proposals"][0]
    assert entry["causal"]["trial_level"] == "pooled"
    assert entry["causal"]["support_sufficient"] is False
    assert entry["source"] == "pooled_randomized"
    assert entry["evidence_origin"] == "randomized"
    assert entry["trial_level"] == "pooled"
    assert entry["support_status"] == "insufficient"
    assert entry["generalization_level"] in {"cross_context_class", "pooled_class"}
    # And a context-level supported entry reports the structured form too.
    strong = TrialChoiceModel.fit(_scoped_events())
    active = CausalChoicePolicy(mode="active", trial_model=strong)
    ok = active.rank(
        _CANDIDATES, model_choice=_MODEL_CHOICE,
        task_family="f", site="h")["proposals"][0]
    assert ok["evidence_origin"] == "randomized"
    assert ok["trial_level"] == "family+site"
    assert ok["support_status"] == "sufficient"
    assert ok["generalization_level"] == "exact"


# ----------------------------------------- active-mode evidence strata
#
# Active execution requires context-specific randomized evidence. Pooled
# randomized evidence is a hypothesis/nomination channel — it can annotate
# (shadow) and can nominate a randomized canary, but it can never drive a
# deterministic substitution in a context it did not measure.


def _scoped_events(cand_success=True, ctrl_success=False, n=24, *, family="f",
                   site="h"):
    return (
        _trials("candidate", n, n if cand_success else 0, "c",
                task_family=family, site=site)
        + _trials("control", n, n if ctrl_success else 0, "k",
                  task_family=family, site=site)
    )


def _active_entry(policy, **query):
    return policy.rank(_CANDIDATES, model_choice=_MODEL_CHOICE, **query)["proposals"][0]


def test_active_executes_on_family_site_evidence():
    """beneficial at the strictest stratum → executable."""
    model = TrialChoiceModel.fit(_scoped_events())
    policy = CausalChoicePolicy(mode="active", trial_model=model)
    entry = _active_entry(policy, task_family="f", site="h")
    assert entry["causal"]["trial_level"] == "family+site"
    assert entry["executable"] is True
    assert entry["execution_blocker"] is None
    assert policy.proposal(
        _CANDIDATES, model_choice=_MODEL_CHOICE, task_family="f", site="h"
    ) is not None


def test_active_executes_on_family_and_site_strata():
    """beneficial at ``family`` and at ``site`` → executable under the
    default active-evidence policy."""
    family_events = _trials("candidate", 24, 24, "c", task_family="f") + _trials(
        "control", 24, 0, "k", task_family="f"
    )
    model = TrialChoiceModel.fit(family_events)
    policy = CausalChoicePolicy(mode="active", trial_model=model)
    entry = _active_entry(policy, task_family="f")
    assert entry["causal"]["trial_level"] == "family"
    assert entry["executable"] is True

    site_events = _trials("candidate", 24, 24, "c", site="h") + _trials(
        "control", 24, 0, "k", site="h"
    )
    site_model = TrialChoiceModel.fit(site_events)
    site_policy = CausalChoicePolicy(mode="active", trial_model=site_model)
    site_entry = _active_entry(site_policy, site="h")
    assert site_entry["causal"]["trial_level"] == "site"
    assert site_entry["executable"] is True


def test_active_never_executes_on_pooled_evidence():
    """beneficial at ``pooled`` → not executable; the blocker names why.
    Pooled evidence generalizes over all tasks — it may hypothesize, never
    authorize a deterministic override in an unmeasured context."""
    events = _trials("candidate", 24, 24, "c") + _trials("control", 24, 0, "k")
    model = TrialChoiceModel.fit(events)
    policy = CausalChoicePolicy(mode="active", trial_model=model)
    entry = _active_entry(policy)
    assert entry["causal"]["trial_level"] == "pooled"
    assert entry["causal"]["effect_status"] == "beneficial"
    assert entry["executable"] is False
    assert entry["execution_blocker"] == "pooled_only"
    # proposal() honors the gate.
    assert policy.proposal(_CANDIDATES, model_choice=_MODEL_CHOICE) is None


def test_shadow_and_canary_still_see_pooled_evidence():
    """Pooled evidence keeps its advisory roles: it ranks in shadow and may
    nominate a randomized canary — execution authority is the only thing
    withheld."""
    events = _trials("candidate", 24, 24, "c") + _trials("control", 24, 0, "k")
    model = TrialChoiceModel.fit(events)

    shadow = CausalChoicePolicy(mode="shadow", trial_model=model)
    ranking = shadow.rank(_CANDIDATES, model_choice=_MODEL_CHOICE)
    assert ranking["proposals"][0]["causal"]["trial_level"] == "pooled"
    assert ranking["proposals"][0]["source"] == "pooled_randomized"
    assert ranking["proposals"][0]["executable"] is False
    assert shadow.proposal(_CANDIDATES, model_choice=_MODEL_CHOICE) is None

    canary = CausalChoicePolicy(mode="canary", trial_model=model)
    proposal = canary.proposal(_CANDIDATES, model_choice=_MODEL_CHOICE)
    assert proposal is not None and proposal["id"] == "p"
    assert proposal["executable"] is False
    assert proposal["execution_blocker"] == "pooled_only"


def test_active_unresolved_stratum_specific_evidence_blocks():
    """A supported but unresolved estimate at family+site cannot execute —
    'we measured this context and it is not established' is not authority."""
    events = (
        _trials("candidate", 20, 12, "c", task_family="f", site="h")
        + _trials("control", 20, 10, "k", task_family="f", site="h")
    )
    model = TrialChoiceModel.fit(events)
    policy = CausalChoicePolicy(mode="active", trial_model=model)
    entry = _active_entry(policy, task_family="f", site="h")
    assert entry["causal"]["trial_level"] == "family+site"
    assert entry["causal"]["effect_status"] == "unresolved"
    assert entry["executable"] is False
    assert entry["execution_blocker"] == "unresolved"


def test_active_omits_harmful_and_blocks_insufficient():
    """Established-harmful divergences are flagged and refused — never
    executable in active, never nominated as a canary; thin evidence cannot
    execute either."""
    harmful = _trials("candidate", 24, 0, "c", task_family="f") + _trials(
        "control", 24, 24, "k", task_family="f"
    )
    model = TrialChoiceModel.fit(harmful)
    policy = CausalChoicePolicy(mode="active", trial_model=model)
    ranking = policy.rank(_CANDIDATES, model_choice=_MODEL_CHOICE, task_family="f")
    entry = ranking["proposals"][0]
    assert entry["causal"] is None
    assert entry["execution_blocker"] == "harmful"
    assert entry["executable"] is False
    assert policy.proposal(_CANDIDATES, model_choice=_MODEL_CHOICE,
                           task_family="f") is None
    # Canary mode refuses it too: a refuted divergence must not consume a
    # real randomized trial.
    canary = CausalChoicePolicy(mode="canary", trial_model=model)
    assert canary.proposal(_CANDIDATES, model_choice=_MODEL_CHOICE,
                           task_family="f") is None

    thin = _trials("candidate", 4, 4, "c", task_family="f") + _trials(
        "control", 4, 0, "k", task_family="f"
    )
    thin_model = TrialChoiceModel.fit(thin)
    thin_policy = CausalChoicePolicy(mode="active", trial_model=thin_model)
    entry = _active_entry(thin_policy, task_family="f")
    assert entry["executable"] is False
    assert entry["execution_blocker"] == "insufficient_support"


def test_active_trial_levels_reject_unknown_strings():
    """Configuration cannot smuggle in an unrecognized evidence level — an
    unknown name fails closed at construction."""
    with pytest.raises(ValueError):
        CausalChoicePolicy(mode="active", active_trial_levels=("made_up",))
    with pytest.raises(ValueError):
        CausalChoicePolicy(mode="active",
                           active_trial_levels=("family", "inferred"))
    # An explicit operator may still widen the list — modes are configuration
    # — but only with real level names.
    CausalChoicePolicy(mode="active", active_trial_levels=("family", "pooled"))


# ==================================================== hierarchical outcomes
#
# jev-outcome-vector/1: the primary endpoint stays verified run success under
# ITT. Secondary endpoints carry diagnostics and vetoes; safety endpoints are
# constraints, never utility.


def test_outcome_vector_schema_and_coverage():
    """Every arm report carries the versioned outcome vector. Fresh v8 cells
    report coverage 1 (the fields were actually recorded); the primary block
    restates — never replaces — the top-level endpoint."""
    entry = _cell(_trials("candidate", 10, 8, "c") + _trials("control", 10, 4, "k"))
    for arm in ("candidate", "control"):
        ov = entry[arm]["outcome_vector"]
        assert ov["schema"] == "jev-outcome-vector/1"
        assert ov["coverage"] == 1.0
        assert ov["primary"]["verified_success"] == entry[arm]["p_success"]
        assert set(ov["secondary"]) == {"verified_transition", "recovery", "steps"}
        assert set(ov["cost"]) == {"latency_ms", "tokens", "approvals"}
        assert set(ov["safety"]) == {
            "authority_touches", "guard_failures", "indeterminate_rate"}
    assert entry["safety"]["regression"] is False
    assert set(entry["secondary_delta"]) == {
        "verified_transition", "recovery", "steps"}


def test_local_progress_with_unrelated_failure_is_not_promoted():
    """Scenario: the assigned step reliably moved the page (verified
    transition) but the runs all failed for unrelated reasons. The primary
    endpoint must dominate — the divergence is refuted, not promoted — while
    the secondary channel still shows the mechanism firing."""
    # Every candidate run: step executed and changed the page, run blocked.
    events = (
        _trials("candidate", 20, 0, "c", task_family="f", page_changed=True)
        + _trials("control", 20, 0, "k", task_family="f", page_changed=False)
    )
    entry = _cell(events, family="f")
    assert entry["candidate"]["outcome_vector"]["secondary"]["verified_transition"]["p"] == 1.0
    assert entry["control"]["outcome_vector"]["secondary"]["verified_transition"]["p"] == 0.0
    assert entry["secondary_delta"]["verified_transition"]["delta"] == 1.0
    # Both arms scored 0 on the primary endpoint — no effect is established,
    # and nothing about the secondary progress rescues it.
    assert entry["effect_status"] == "unresolved"
    model = TrialChoiceModel.fit(events)
    assert model.rank(_CANDIDATES, model_choice=_MODEL_CHOICE, task_family="f")


def test_inert_step_with_accidental_success_stays_separate():
    """Scenario: the assigned step never changed the page but the runs
    succeeded anyway (the success came from elsewhere in the trajectory).
    The secondary transition channel records 0 — mechanism and outcome are
    reported independently."""
    events = (
        _trials("candidate", 24, 24, "c", task_family="f", page_changed=False)
        + _trials("control", 24, 0, "k", task_family="f", page_changed=False)
    )
    entry = _cell(events, family="f")
    assert entry["candidate"]["outcome_vector"]["secondary"]["verified_transition"]["p"] == 0.0
    assert entry["secondary_delta"]["verified_transition"]["delta"] == 0.0
    assert entry["effect_status"] == "beneficial"  # primary still governs


def test_latency_gain_cannot_compensate_success_regression():
    """Scenario: the candidate is much faster but fails more. The cost block
    reports the speed gain *and* the primary endpoint stays harmful — a
    report reader sees both, and the prior never proposes the arm."""
    events = (
        _trials("candidate", 24, 0, "c", task_family="f", latency_ms=10)
        + _trials("control", 24, 24, "k", task_family="f", latency_ms=500)
    )
    entry = _cell(events, family="f")
    assert entry["effect_status"] == "harmful"
    cand = entry["candidate"]["outcome_vector"]
    ctrl = entry["control"]["outcome_vector"]
    assert cand["cost"]["latency_ms"] < ctrl["cost"]["latency_ms"]
    model = TrialChoiceModel.fit(events)
    assert model.rank(_CANDIDATES, model_choice=_MODEL_CHOICE,
                      task_family="f") == []


def test_success_gain_with_authority_burden_is_a_safety_regression():
    """Scenario: the candidate wins on the primary endpoint but every
    execution consumed an authority-plane approval. Safety is a constraint:
    the regression is flagged on the contrast, active execution is blocked
    with ``safety_regression``, and canary refuses to nominate the arm —
    success cannot buy off an authority regression."""
    events = (
        _trials("candidate", 24, 24, "c", task_family="f", risk_events=1)
        + _trials("control", 24, 0, "k", task_family="f", risk_events=0)
    )
    entry = _cell(events, family="f")
    assert entry["effect_status"] == "beneficial"
    assert entry["safety"]["regression"] is True
    assert "authority_touches" in entry["safety"]["regressions"]
    model = TrialChoiceModel.fit(events)
    ranked = model.rank(_CANDIDATES, model_choice=_MODEL_CHOICE, task_family="f")
    assert ranked[0]["safety_regression"] is True
    active = CausalChoicePolicy(mode="active", trial_model=model)
    ent = _active_entry(active, task_family="f")
    assert ent["execution_blocker"] == "safety_regression"
    assert ent["executable"] is False
    assert active.proposal(_CANDIDATES, model_choice=_MODEL_CHOICE,
                           task_family="f") is None
    canary = CausalChoicePolicy(mode="canary", trial_model=model)
    assert canary.proposal(_CANDIDATES, model_choice=_MODEL_CHOICE,
                           task_family="f") is None


def test_unmeasured_arms_report_null_not_zero():
    """Scenario: cells migrated from jev-trials/7 never recorded outcome
    fields. Secondary and safety endpoints must report ``None`` — a migrated
    zero is absence of evidence, not a measured zero — and the safety
    verdict is unknown, not clean."""
    from jev_ultrafast.dreamlearn import TERMINATION_REASONS
    key = ("f", "s", "click", "navigate", "button", "0", "0-4", "0-2",
           "click", "navigate", "link", "1", "0-4", "0-2", "candidate")
    stats = [10, 10, 10, 0, 9.0, 10.0, 10.0, 10.0, 10.0, 0.0]
    costs = [30.0, 40.0, 2.0]
    reasons = [0] * len(TERMINATION_REASONS)
    v7 = (*key, *stats, *costs, *reasons)
    assert len(v7) == 37
    ctrl = (*key[:-1], "control")
    v7k = (*ctrl, *stats, *costs, *reasons)
    trials = CounterfactualTrials.from_dict(
        {"version": "jev-trials/7", "cells": [v7, v7k]})
    assert trials.version == "jev-trials/9"
    entry = next(iter(trials.estimate().values()))
    for arm in ("candidate", "control"):
        ov = entry[arm]["outcome_vector"]
        assert ov["coverage"] == 0.0
        assert ov["secondary"]["verified_transition"]["p"] is None
        assert ov["secondary"]["verified_transition"]["ci"] is None
        assert ov["secondary"]["steps"]["mean"] is None
        assert ov["safety"]["authority_touches"] is None
        assert ov["safety"]["guard_failures"] is None
    assert entry["secondary_delta"]["verified_transition"]["delta"] is None
    assert entry["safety"]["regression"] is None  # unknown, not clean


def test_differential_censoring_excludes_secondary_mass_too():
    """Scenario: candidate arm heavily censored by crash terminations. The
    censored mass counts toward no endpoint — secondary counters come from
    analyzed runs only — and the imbalance veto still holds."""
    events = _trials("candidate", 16, 16, "c", task_family="f", page_changed=True)
    for i in range(16):
        events += _trial_run(f"cx{i}", "candidate", status="aborted",
                             verified=False, executed=False,
                             reason="browser_crash", task_family="f")
    events += _trials("control", 16, 4, "k", task_family="f",
                      page_changed=False)
    for i in range(2):
        events += _trial_run(f"kx{i}", "control", status="aborted",
                             verified=False, executed=False,
                             reason="operator_cancel", task_family="f")
    entry = _cell(events, family="f")
    assert entry["censoring_imbalance"] is True
    assert entry["effect_status"] == "unresolved"
    assert entry["unresolved_reason"] == "censoring_imbalance"
    # The 16 analyzed candidate runs still carry their measured transitions.
    assert entry["candidate"]["outcome_vector"]["coverage"] == 1.0


def test_secondary_delta_never_establishes():
    """Scenario: a secondary endpoint's fixed-sample CI excludes zero under
    optional-stopping-shaped data — that must not touch the primary
    establishment machinery. ``establishment`` is pinned False and
    ``effect_status`` is decided only by the primary confidence sequence."""
    events = (
        _trials("candidate", 12, 6, "c", task_family="f", page_changed=True)
        + _trials("control", 12, 6, "k", task_family="f", page_changed=False)
    )
    entry = _cell(events, family="f")
    vt = entry["secondary_delta"]["verified_transition"]
    assert vt["ci"][0] > 0.0  # the secondary contrast IS lopsided
    assert vt["establishment"] is False
    assert entry["secondary_delta"]["recovery"]["establishment"] is False
    assert entry["secondary_delta"]["steps"]["establishment"] is False
    # And the primary endpoint, dead even, stays unresolved — secondary
    # certainty cannot leak into it.
    assert entry["effect_status"] == "unresolved"


def test_recovery_endpoint_counts_stale_marker_then_success():
    """A run that recorded guard-failure markers yet still verified done is
    a measured recovery — distinct from a clean success."""
    events = (
        _trials("candidate", 16, 12, "c", task_family="f",
                stale_or_failure=1)
        + _trials("control", 16, 8, "k", task_family="f")
    )
    entry = _cell(events, family="f")
    cand = entry["candidate"]["outcome_vector"]
    assert cand["secondary"]["recovery"]["p"] == 0.75
    assert cand["safety"]["guard_failures"] == 1.0
    # Control had no guard failures and no recoveries.
    assert entry["control"]["outcome_vector"]["safety"]["guard_failures"] == 0.0
    # A guard-failure burden on the candidate is a safety regression —
    # the candidate recovered often, but recovering from a self-inflicted
    # guard failure is not a free lunch.
    assert "guard_failures" in entry["safety"]["regressions"]
    assert entry["safety"]["regression"] is True


def test_v8_roundtrip_preserves_outcome_counters():
    """fit → to_dict → from_dict must round-trip the appended outcome block."""
    events = _trials("candidate", 10, 8, "c", task_family="f",
                     stale_or_failure=1, risk_events=1, extra_transitions=2)
    trials = CounterfactualTrials.fit(events)
    clone = CounterfactualTrials.from_dict(trials.to_dict())
    assert clone.version == "jev-trials/9"
    assert clone.cells == trials.cells
    assert clone.estimate() == trials.estimate()
    ov = next(iter(clone.estimate().values()))["candidate"]["outcome_vector"]
    assert ov["secondary"]["steps"]["mean"] == 3.0
    # One authority touch and one guard failure per run — only on the
    # assigned step (extra transitions stay clean).
    assert ov["safety"]["authority_touches"] == 1.0
    assert ov["safety"]["guard_failures"] == 1.0
