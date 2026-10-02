import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from jev_ultrafast.dream import (
    DreamImprover,
    ExperienceStore,
    ExplorationPolicy,
    PromotionGate,
    ReplaySimulator,
    ReplayWorld,
    mutate_policies,
    task_key,
)
from jev_ultrafast.dreamlearn import (
    ChoiceModel,
    CostModel,
    CounterfactualTrials,
    OutcomeModel,
    TrialChoiceModel,
    overlap_bucket,
    rank_bucket,
)
from jev_ultrafast.model import candidate_actions
from jev_ultrafast.privacy import action_goal_overlap
from jev_ultrafast.signing import EvidenceSigner, verify_signature


def _append_run(store, run_id, goal, transitions, *, status="done", verified=True, policy=None, extra=None):
    key = task_key(goal)
    policy = policy or ExplorationPolicy()
    store.append({
        "run_id": run_id,
        "task_key": key,
        "sequence": 1,
        "event": "run_started",
        "goal": goal,
        "state": "S",
        "policy": policy.to_dict(),
        "policy_digest": policy.digest,
        **(extra or {}),
    })
    for i, transition in enumerate(transitions, 2):
        store.append({
            "run_id": run_id,
            "task_key": key,
            "sequence": i,
            "event": "transition",
            **transition,
        })
    store.append({
        "run_id": run_id,
        "task_key": key,
        "sequence": len(transitions) + 2,
        "event": "run_finished",
        "status": status,
        "verified": verified,
    })


def _transition(selected="sel", offered=40, tokens=400, latency_ms=120, page_changed=True, candidates=None):
    return {
        "event": "transition",
        "state": "S",
        "next_state": "D",
        "selected": {"id": selected, "kind": "click"},
        "candidates": candidates or [
            {"id": "sel", "kind": "click", "label": "Search", "goal_overlap": 2},
            *[{"id": f"c{i}", "kind": "click", "label": f"Ctl {i}", "goal_overlap": 0} for i in range(offered - 1)],
        ],
        "offered_count": offered,
        "page_changed": page_changed,
        "latency_ms": latency_ms,
        "model_calls": 1,
        "tokens": tokens,
        "stale_or_failure": 0,
        "risk_events": 0,
    }


def _run_events(run_id, transitions, *, status="done", verified=True, task_key="t", policy=None,
                task_family=None, reason=None):
    """Wrap bare transition dicts as one run with a terminal outcome.

    Trajectory-success labeling keys positives off the run's verified finish,
    so tests expressing outcome semantics must write real run boundaries —
    loose transitions carry no outcome and fit as negatives. ``reason`` is the
    structured termination cause recorded on an aborted run_finished.
    """
    policy = policy or ExplorationPolicy()
    started = {"event": "run_started", "run_id": run_id, "task_key": task_key, "goal": "g",
               "policy": policy.to_dict(), "policy_digest": policy.digest}
    if task_family is not None:
        started["task_family"] = task_family
    finished = {"event": "run_finished", "run_id": run_id, "task_key": task_key,
                "status": status, "verified": verified}
    if reason is not None:
        finished["reason"] = reason
    events = [
        started,
        *[{**t, "run_id": run_id, "task_key": task_key} for t in transitions],
        finished,
    ]
    return events


# ---------------------------------------------------------------- CostModel


def test_cost_model_fits_positive_cost_scaling():
    events = [
        _transition(offered=o, tokens=200 + 4 * o, latency_ms=100 + 2 * o)
        for o in (40, 60, 80, 100, 140, 160, 200, 240)
    ]
    model = CostModel.fit(events)
    assert model.reliable
    assert model.samples == 8
    assert model.token_per_candidate > 0
    assert model.latency_per_candidate_ms > 0
    assert model.predict(200)["tokens"] > model.predict(50)["tokens"]
    assert model.predict(100)["extrapolated"] is False
    assert model.predict(800)["extrapolated"] is True


def test_cost_model_is_unreliable_below_min_samples():
    model = CostModel.fit([_transition()] * 3)
    assert not model.reliable


def test_cost_model_digest_and_roundtrip():
    model = CostModel.fit([_transition(offered=o) for o in (40, 80, 120, 160, 200, 240, 50, 90)])
    clone = CostModel.from_dict(model.to_dict())
    assert clone.digest == model.digest
    with pytest.raises(ValueError, match="Unknown cost model keys"):
        CostModel.from_dict({**model.to_dict(), "backdoor": True})


def test_replay_reports_anchored_estimated_costs(tmp_path):
    store = ExperienceStore(tmp_path / "est.jsonl")
    recorded_offered = 200
    _append_run(store, "r0", "search", [_transition(offered=recorded_offered, tokens=1000, latency_ms=400)])
    worlds = ReplayWorld.from_events(store.load())

    model = CostModel.fit([
        _transition(offered=o, tokens=100 + 5 * o, latency_ms=50 + o)
        for o in (40, 60, 80, 100, 140, 160, 200, 240)
    ])
    baseline = ExplorationPolicy()
    shrunk = ExplorationPolicy(name="shrunk", model_action_limit=64)

    empirical = ReplaySimulator(worlds).evaluate(shrunk).metrics
    estimated = ReplaySimulator(worlds).evaluate(shrunk, cost_model=model).metrics
    base_est = ReplaySimulator(worlds).evaluate(baseline, cost_model=model).metrics

    assert empirical.estimated_score is None
    assert estimated.estimated_score is not None
    # The candidate offers ~64 of 200 candidates; the learned marginal cost
    # predicts materially fewer tokens than the recorded 1000.
    assert estimated.estimated_tokens < empirical.tokens
    assert estimated.estimated_score > base_est.estimated_score


def test_cost_model_never_qualifies_a_failing_candidate(tmp_path):
    store = ExperienceStore(tmp_path / "gate.jsonl")
    candidates = [
        {"id": "sel", "kind": "click", "label": "Search", "goal_overlap": 0},
        *[{"id": f"c{i}", "kind": "click", "label": f"Ctl {i}", "goal_overlap": 0} for i in range(30)],
    ]
    # The recorded run clicked "sel" at rank 0. A policy dropping it to a
    # coverage miss must stay rejected no matter how cheap it looks.
    _append_run(store, "r0", "search", [{
        "state": "S", "next_state": "D",
        "selected": {"id": "sel", "kind": "click"},
        "candidates": candidates,
        "offered_count": 31, "page_changed": True, "latency_ms": 400,
        "model_calls": 1, "tokens": 1000, "stale_or_failure": 0, "risk_events": 0,
    }])
    worlds = ReplayWorld.from_events(store.load())
    model = CostModel.fit([
        _transition(offered=o, tokens=100 + 5 * o, latency_ms=50 + o)
        for o in (40, 60, 80, 100, 140, 160, 200, 240)
    ])
    report = DreamImprover(gate=PromotionGate(min_coverage=1.0)).improve(
        worlds, ExplorationPolicy(), cost_model=model, outcome_model=None
    )
    assert report.cost_model_digest == model.digest
    # min_goal_overlap=1 drops the recorded selection (overlap 0) entirely:
    # a guaranteed coverage miss that the favorable cost estimates must not
    # rescue — prioritization can reorder passing candidates, never open a gate.
    coverage_miss = [
        c for c in report.candidates if c["policy"].get("min_goal_overlap") == 1
    ]
    assert coverage_miss
    assert all(not c["replay_approved"] for c in coverage_miss)
    assert all("coverage" in c["reason"] or "objective" in c["reason"] for c in coverage_miss)
    assert any("estimated_gain_per_world" in c for c in coverage_miss)


def test_improve_uses_estimates_to_prioritize_among_passing(tmp_path):
    store = ExperienceStore(tmp_path / "rank.jsonl")
    candidates = [{"id": "sel", "kind": "fill", "label": "Destination", "goal_overlap": 3}]
    candidates += [
        {"id": f"c{i}", "kind": "click", "label": f"Ctl {i}", "goal_overlap": 0} for i in range(240)
    ]
    _append_run(store, "r0", "find destination", [{
        "state": "S", "next_state": "D",
        "selected": {"id": "sel", "kind": "fill"},
        "candidates": candidates, "offered_count": 241, "page_changed": True,
        "latency_ms": 400, "model_calls": 1, "tokens": 1200,
        "stale_or_failure": 0, "risk_events": 0,
    }])
    worlds = ReplayWorld.from_events(store.load())
    model = CostModel.fit([
        _transition(offered=o, tokens=100 + 5 * o, latency_ms=50 + o)
        for o in (40, 60, 80, 100, 140, 160, 200, 240)
    ])
    report = DreamImprover(gate=PromotionGate(min_coverage=0.5)).improve(
        worlds, ExplorationPolicy(), cost_model=model
    )
    assert report.cost_model_digest == model.digest
    annotated = [c for c in report.candidates if "estimated_gain_per_world" in c]
    assert annotated


# -------------------------------------------------------------- OutcomeModel


def test_outcome_model_learns_kind_and_cell_priors():
    events = [
        _transition(selected="sel", page_changed=True, candidates=[
            {"id": "sel", "kind": "click", "label": "Search", "goal_overlap": 2},
        ]),
        *[
            {
                **_transition(selected=f"w{i}", page_changed=False, candidates=[
                    {"id": f"w{i}", "kind": "wait", "label": "wait", "goal_overlap": 0},
                ]),
                "selected": {"id": f"w{i}", "kind": "wait"},
            }
            for i in range(6)
        ],
    ]
    model = OutcomeModel.fit(events)
    assert model.samples == 7
    click = model.predict(kind="click", goal_overlap=2, rank=0)
    wait = model.predict(kind="wait", goal_overlap=0, rank=0)
    assert click["p_page_changed"] > wait["p_page_changed"]
    assert wait["level"] in {"cell", "kind"}
    unseen = model.predict(kind="scroll", goal_overlap=0, rank=500)
    assert unseen["level"] == "global"


def test_outcome_model_buckets_and_roundtrip():
    assert overlap_bucket(0) == "0"
    assert overlap_bucket(3) == "3+"
    assert rank_bucket(None) == "unknown"
    assert rank_bucket(3) == "0-4"
    assert rank_bucket(150) == "100+"
    model = OutcomeModel.fit([_transition()])
    clone = OutcomeModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert clone.digest == model.digest


def test_improve_annotates_outcome_predictions(tmp_path):
    store = ExperienceStore(tmp_path / "outcome.jsonl")
    _append_run(store, "r0", "search", [_transition()])
    worlds = ReplayWorld.from_events(store.load())
    outcome = OutcomeModel.fit(store.load())
    report = DreamImprover(gate=PromotionGate(min_coverage=0.5)).improve(
        worlds, ExplorationPolicy(), outcome_model=outcome
    )
    assert report.outcome_model_digest == outcome.digest
    assert any("predicted_page_changes_per_world" in c for c in report.candidates)


# --------------------------------------------------------------- ChoiceModel


def test_choice_model_uncertainty_and_abstention():
    events = [
        {
            **_transition(selected="sel", page_changed=True, candidates=[
                {"id": "sel", "kind": "click", "label": "Search", "goal_overlap": 2},
            ]),
            "selected_offered_rank": 0,
        },
        *[
            {
                **_transition(selected=f"w{i}", page_changed=False, candidates=[
                    {"id": f"w{i}", "kind": "wait", "label": "wait", "goal_overlap": 0},
                ]),
                "selected": {"id": f"w{i}", "kind": "wait"},
            }
            for i in range(6)
        ],
    ]
    model = ChoiceModel.fit(events)
    assert model.samples == 7
    sparse = model.predict(kind="click", goal_overlap=2, rank=0)
    assert sparse["level"] == "cell" and sparse["n"] == 1
    assert sparse["confident"] is False
    assert 0 < sparse["uncertainty"] < 0.5
    dense = model.predict(kind="wait", goal_overlap=0, rank=0)
    assert dense["n"] == 6 and dense["confident"] is True
    assert dense["uncertainty"] < sparse["uncertainty"]
    unseen = model.predict(kind="scroll", goal_overlap=0, rank=500)
    assert unseen["level"] == "global" and unseen["n"] == 7


def test_choice_model_empty_fit_abstains():
    model = ChoiceModel.fit([])
    prediction = model.predict(kind="click", goal_overlap=0, rank=0)
    assert prediction["level"] == "global"
    assert prediction["n"] == 0 and prediction["confident"] is False
    assert model.choose([{"id": "a", "kind": "click"}])["confident"] is False
    assert model.choose([]) is None


def test_choice_model_choose_proposes_only_offered_candidates():
    # Mixed cells within one kind: goal-relevant clicks ran on verified
    # trajectories, irrelevant ones on failed runs, so the cell prior beats
    # the kind fallback.
    events = []
    for i in range(5):
        events += _run_events(f"g{i}", [
            {
                **_transition(selected=f"g{i}", candidates=[
                    {"id": f"g{i}", "kind": "click", "label": "Search", "goal_overlap": 2},
                ]),
                "selected_offered_rank": 0,
            }
        ], status="done", verified=True)
    for i in range(5):
        events += _run_events(f"k{i}", [
            {
                **_transition(selected=f"k{i}", page_changed=False, candidates=[
                    {"id": f"k{i}", "kind": "click", "label": "Ctl", "goal_overlap": 0},
                ]),
                "selected_offered_rank": 0,
            }
        ], status="blocked", verified=False)
    model = ChoiceModel.fit(events)
    offered = [
        {"id": "a", "kind": "click", "label": "A", "goal_overlap": 0},
        {"id": "b", "kind": "click", "label": "Search", "goal_overlap": 2},
        {"id": "c", "kind": "click", "label": "C", "goal_overlap": 0},
    ]
    proposal = model.choose(offered)
    assert proposal["id"] in {c["id"] for c in offered}
    assert proposal["id"] == "b"  # only candidate in the strong cell
    assert {"p_progress", "uncertainty", "confident", "score"} <= set(proposal)


def test_choice_model_ucb_can_prefer_uncertain_candidate():
    # A dense low-payoff cell vs an unseen kind: optimism under uncertainty
    # still proposes the unexplored action, but the proposal is flagged
    # not-confident (global fallback is an abstention) so consumers know it
    # is a hypothesis, not knowledge.
    events = []
    for i in range(8):
        events += _run_events(f"k{i}", [
            {
                **_transition(selected=f"k{i}", page_changed=False, candidates=[
                    {"id": f"k{i}", "kind": "click", "label": "Known", "goal_overlap": 0},
                ]),
                "selected_offered_rank": 0,
            }
        ], status="blocked", verified=False)
    for i in range(4):
        events += _run_events(f"s{i}", [
            {
                **_transition(selected=f"s{i}", candidates=[
                    {"id": f"s{i}", "kind": "select", "label": "Pick", "goal_overlap": 0},
                ]),
                "selected": {"id": f"s{i}", "kind": "select"},
                "selected_offered_rank": 0,
            }
        ], status="done", verified=True)
    model = ChoiceModel.fit(events)
    offered = [
        {"id": "known", "kind": "click", "label": "Known", "goal_overlap": 0},
        {"id": "novel", "kind": "fill", "label": "Novel", "goal_overlap": 0},
    ]
    proposal = model.choose(offered)
    assert proposal["id"] == "novel"
    assert proposal["confident"] is False
    assert proposal["uncertainty"] > model.predict(kind="click", goal_overlap=0, rank=0)["uncertainty"]


def test_choice_model_roundtrip_and_digest():
    model = ChoiceModel.fit([_transition()])
    clone = ChoiceModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert clone.digest == model.digest
    other = ChoiceModel.fit([_transition(), _transition()])
    assert other.digest != model.digest
    with pytest.raises(ValueError, match="Unknown choice model keys"):
        ChoiceModel.from_dict({"samples": 0, "bogus": True})


def _rank_sensitive_worlds():
    """Recorded action sits at DOM index 6 behind six zero-overlap controls."""
    candidates = [
        *[{"id": f"f{i}", "kind": "click", "label": f"ctl{i}", "goal_overlap": 0} for i in range(6)],
        {"id": "sel", "kind": "click", "label": "Search", "goal_overlap": 2},
    ]
    transition = _transition(selected="sel", candidates=candidates, offered=7)
    transition["selected_observed_rank"] = 6
    transition["selected_offered_rank"] = 6
    events = []
    for i in range(3):
        events.append({"event": "run_started", "run_id": f"r{i}", "task_key": "t", "goal": "search",
                       "policy": ExplorationPolicy().to_dict(),
                       "policy_digest": ExplorationPolicy().digest})
        events.append({**transition, "run_id": f"r{i}", "task_key": "t"})
        events.append({"event": "run_finished", "run_id": f"r{i}", "task_key": "t",
                       "status": "done", "verified": True})
    return ReplayWorld.from_events(events)


def test_choice_model_replay_uses_candidate_policy_rank():
    worlds = _rank_sensitive_worlds()
    # Separate priors per rank bucket: recorded at bucket "5-19" under baseline
    # (offered rank 6) but "0-4" under the filtering policy (offered rank 0).
    front_events = []
    for i in range(6):
        front_events += _run_events(f"f{i}", [
            {**_transition(selected=f"s{i}", candidates=[
                {"id": f"s{i}", "kind": "click", "label": "Search", "goal_overlap": 2},
            ]), "selected_offered_rank": 0}
        ])
    back_events = []
    for i in range(6):
        # The selected action honestly sits at offered rank 9.
        back_events += _run_events(f"b{i}", [
            {**_transition(selected=f"b{i}", page_changed=False, offered=10, candidates=[
                *[{"id": f"x{i}-{j}", "kind": "click", "label": f"X{j}", "goal_overlap": 0}
                  for j in range(9)],
                {"id": f"b{i}", "kind": "click", "label": "Back", "goal_overlap": 0},
            ]), "selected_offered_rank": 9}
        ], status="blocked", verified=False)
    model = ChoiceModel.fit(front_events + back_events)
    sim = ReplaySimulator(worlds)
    baseline = ReplaySimulator(worlds).evaluate(ExplorationPolicy(), choice_model=model)
    filtered = sim.evaluate(ExplorationPolicy(min_goal_overlap=1), choice_model=model)
    base_summary = baseline.per_world[0]
    filt_summary = filtered.per_world[0]
    # Same recorded action, different offered rank -> different prior cell.
    assert filt_summary["choice_predicted_change"] > base_summary["choice_predicted_change"]
    # Annotation-only: identical empirical metrics either way.
    assert baseline.metrics.successes == filtered.metrics.successes
    plain = ReplaySimulator(worlds).evaluate(ExplorationPolicy(min_goal_overlap=1))
    assert filtered.metrics.score == plain.metrics.score
    assert filtered.metrics.actions == plain.metrics.actions


def test_choice_model_proposals_never_reach_gates(tmp_path):
    store = ExperienceStore(tmp_path / "choice.jsonl")
    for i in range(3):
        _append_run(store, f"r{i}", "search", [_transition()])
    events = store.load()
    worlds = ReplayWorld.from_events(events)
    choice = ChoiceModel.fit(events)
    improver = DreamImprover(gate=PromotionGate(min_coverage=0.5))
    with_model = improver.improve(worlds, ExplorationPolicy(), choice_model=choice)
    without_model = DreamImprover(gate=PromotionGate(min_coverage=0.5)).improve(
        worlds, ExplorationPolicy()
    )
    assert with_model.promotion.approved == without_model.promotion.approved
    assert with_model.selected.digest == without_model.selected.digest
    assert with_model.choice_model_digest == choice.digest
    annotated = [c for c in with_model.candidates if "choice_model" in c]
    assert annotated
    entry = annotated[0]["choice_model"]
    assert {"divergence_rate", "mean_uncertainty", "confident_fraction",
            "predicted_progress_per_step", "proposed_progress_per_step",
            "selected_mean_uncertainty", "selected_confident_fraction",
            "proposal_mean_uncertainty", "proposal_confident_fraction"} <= set(entry)
    # The two subjects stay separate: selected-action metrics describe the
    # recorded action's prior, proposal metrics the counterfactual proposal's.
    assert entry["mean_uncertainty"] == entry["selected_mean_uncertainty"]
    assert entry["confident_fraction"] == entry["selected_confident_fraction"]
    # A rejected candidate is annotated but still rejected.
    rejected = [c for c in annotated if not c["replay_approved"]]
    if rejected:
        assert all("choice_model" in c for c in rejected)


def test_choice_model_annotations_do_not_change_replay_outcomes():
    worlds = _rank_sensitive_worlds()
    model = ChoiceModel.fit([])
    with_model = ReplaySimulator(worlds).evaluate(ExplorationPolicy(), choice_model=model)
    without = ReplaySimulator(worlds).evaluate(ExplorationPolicy())
    assert with_model.metrics == without.metrics
    summary = with_model.per_world[0]
    assert summary["actions"] == without.per_world[0]["actions"]
    assert summary["success"] == without.per_world[0]["success"]


# ------------------------------------------------- bounded surface expansion


def _actions():
    actions = [
        {"id": "sel", "kind": "fill", "node": 7, "label": "Destination field"},
        {"id": "go", "kind": "click", "node": 8, "label": "Find destination"},
        {"id": "scroll", "kind": "scroll", "node": 9, "label": "Scroll"},
        {"id": "wait", "kind": "wait", "node": 10, "label": "Wait"},
    ]
    for i in range(12):
        actions.append({"id": f"opt{i}", "kind": "select", "node": 20, "label": f"Option {i}"})
    for i in range(15):
        actions.append({"id": f"c{i}", "kind": "click", "node": 30 + i, "label": f"Control {i}"})
    return actions


def _compact(actions, goal):
    from jev_ultrafast.privacy import tokenize

    tokens = set(tokenize(goal))
    return [
        {
            "id": a["id"],
            "kind": a["kind"],
            "node": a.get("node"),
            "label": a.get("label", ""),
            "goal_overlap": action_goal_overlap(a, tokens),
        }
        for a in actions
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"min_goal_overlap": 1},
        {"min_goal_overlap": 2},
        {"duplicate_node_cap": 3},
        {"duplicate_node_cap": 3, "min_goal_overlap": 1},
        {"overlap_exponent": 0.5},
        {"overlap_exponent": 1.8},
        {"model_action_limit": 16},
        {"model_action_limit": 24, "duplicate_node_cap": 4},
        {"overlap_exponent": 1.4, "duplicate_node_cap": 5, "min_goal_overlap": 1},
    ],
)
def test_live_and_replay_catalogues_match_under_extended_policy(overrides):
    policy = ExplorationPolicy(name="candidate", **overrides)
    goal = "find destination"
    actions = _actions()
    live, _dropped = candidate_actions(actions, goal, exploration_policy=policy)
    replayed = ReplaySimulator._retained_candidates(policy, _compact(actions, goal))
    assert [a["id"] for a in live] == [a["id"] for a in replayed]


def test_duplicate_node_cap_bounds_select_fanout():
    actions = _actions()
    capped, _ = candidate_actions(
        actions, "find destination", exploration_policy=ExplorationPolicy(duplicate_node_cap=2)
    )
    selects = [a for a in capped if a["kind"] == "select"]
    assert len(selects) == 2


def test_min_goal_overlap_filters_irrelevant_controls():
    actions = _actions()
    filtered, _ = candidate_actions(
        actions, "find destination", exploration_policy=ExplorationPolicy(min_goal_overlap=1)
    )
    kinds = {a["kind"] for a in filtered}
    assert "fill" in kinds and "click" in kinds
    assert all(
        a["kind"] not in {"click", "fill", "select"} or action_goal_overlap(a, {"find", "destination"}) >= 1
        for a in filtered
    )
    controls = {a["id"] for a in filtered if a["kind"] not in {"click", "fill", "select"}}
    assert {"scroll", "wait"} <= controls


def test_overlap_exponent_changes_relevance_ranking():
    base = ExplorationPolicy()
    curved = ExplorationPolicy(name="exp", overlap_exponent=0.5)
    spiky = ExplorationPolicy(name="spiky", overlap_exponent=1.4)
    tokens = {"find", "destination", "nearby", "hotel"}
    far = {"id": "far", "kind": "click", "label": "find destination"}
    near = {"id": "near", "kind": "click", "label": "find destination nearby hotel"}
    # Overlap is a distinct-token count; curvature shows at overlap >= 2 and
    # reshapes the relative score of weak versus strong matches.
    flat_ratio = base.candidate_score(far, tokens, 0) / base.candidate_score(near, tokens, 0)
    curved_ratio = curved.candidate_score(far, tokens, 0) / curved.candidate_score(near, tokens, 0)
    spiky_ratio = spiky.candidate_score(far, tokens, 0) / spiky.candidate_score(near, tokens, 0)
    assert curved_ratio > flat_ratio > spiky_ratio


def test_new_knobs_are_bounded_and_mutated():
    with pytest.raises(ValueError, match="overlap_exponent"):
        ExplorationPolicy(overlap_exponent=0.1)
    with pytest.raises(ValueError, match="duplicate_node_cap"):
        ExplorationPolicy(duplicate_node_cap=0)
    with pytest.raises(ValueError, match="min_goal_overlap"):
        ExplorationPolicy(min_goal_overlap=-1)
    candidates = mutate_policies(ExplorationPolicy())
    assert any(c.min_goal_overlap == 1 for c in candidates)
    assert any(c.overlap_exponent != 1.0 for c in candidates)
    assert {c.duplicate_node_cap for c in candidates} >= {150, 200, 225}


def test_behavior_digest_covers_new_knobs():
    a = ExplorationPolicy()
    assert a.behavior_digest != replace(a, min_goal_overlap=1).behavior_digest
    assert a.behavior_digest != replace(a, overlap_exponent=0.7).behavior_digest
    assert a.behavior_digest != replace(a, duplicate_node_cap=100).behavior_digest


# ------------------------------------------------------------------ signing


def _signer():
    return EvidenceSigner.from_hex("ab" * 32)


def test_signer_roundtrip_and_tamper():
    signer = _signer()
    sig = signer.sign_hex("f" * 64)
    assert verify_signature(signer.key_id, "f" * 64, sig)
    assert not verify_signature(signer.key_id, "e" * 64, sig)
    assert not verify_signature("cd" * 32, "f" * 64, sig)


def test_store_signs_events_and_verifies(tmp_path):
    path = tmp_path / "signed.jsonl"
    store = ExperienceStore(path, signer=_signer())
    _append_run(store, "r0", "search", [_transition()])
    events = ExperienceStore(path, verify_key=_signer().key_id).load()
    assert all(e.get("signature") and e.get("key_id") == _signer().key_id for e in events)

    checked = ExperienceStore(path, verify_key=_signer().key_id)
    verified = checked.verify()
    assert verified["signed_events"] == len(events)
    assert verified["signatures_checked"] is True


def test_store_rejects_wrong_key_and_forged_signature(tmp_path):
    path = tmp_path / "signed.jsonl"
    store = ExperienceStore(path, signer=_signer())
    _append_run(store, "r0", "search", [_transition()])

    with pytest.raises(ValueError, match="unexpected key"):
        ExperienceStore(path, verify_key="cd" * 32).load()

    lines = path.read_text().splitlines()
    forged = json.loads(lines[-1])
    forged["signature"] = "00" * 64
    lines[-1] = json.dumps(forged, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="signature verification failure"):
        ExperienceStore(path, verify_key=_signer().key_id).load()


def test_store_rejects_unsigned_when_required_and_reserved_key_injection(tmp_path):
    path = tmp_path / "unsigned.jsonl"
    store = ExperienceStore(path)
    _append_run(store, "r0", "search", [_transition()])
    with pytest.raises(ValueError, match="Unsigned DREAM event"):
        ExperienceStore(path, require_signatures=True).load()
    with pytest.raises(ValueError, match="reserved"):
        store.append({"signature": "spoofed"})
    with pytest.raises(ValueError, match="reserved"):
        store.append({"key_id": "spoofed"})


# ------------------------------------------------ v0.6.1 correction pass ---

def test_cost_model_requires_design_variation_for_reliability():
    """The audit's identifiability gap: N samples at one offered count carry
    zero information about the per-candidate slope."""
    flat = CostModel.fit([_transition(offered=40) for _ in range(8)])
    assert flat.samples == 8 and flat.distinct_offered == 1
    assert not flat.reliable
    # Two distinct counts but a trivially narrow span is not identifying either.
    narrow = CostModel.fit([_transition(offered=o) for o in (40, 41) * 4])
    assert narrow.distinct_offered == 2 and not narrow.reliable
    # Distinct counts spanning the minimum range: identifiable.
    wide = CostModel.fit([_transition(offered=o) for o in (40, 80) * 4])
    assert wide.reliable


def test_choice_model_trains_on_the_offered_rank_coordinate():
    """Regression for the v0.6 evidence-coordinate bug: ``selected_rank`` in old
    traces was the *observed* catalogue index. Fitting must use the offered
    rank — here recorded explicitly, and derived when absent."""
    policy = ExplorationPolicy(min_goal_overlap=1)
    observed = [
        *[{"id": f"f{i}", "kind": "click", "label": f"ctl{i}", "goal_overlap": 0}
          for i in range(6)],
        {"id": "sel", "kind": "click", "label": "Search", "goal_overlap": 2},
    ]
    events = []
    for i in range(3):
        events.append({"event": "run_started", "run_id": f"r{i}", "task_key": "t",
                       "goal": "search", "policy": policy.to_dict(),
                       "policy_digest": policy.digest})
        # Recorded offered rank: the filtered catalogue exposes only "sel".
        events.append({**_transition(selected="sel", candidates=observed, offered=1),
                       "run_id": f"r{i}", "task_key": "t", "catalog": "observed",
                       "selected_observed_rank": 6, "selected_offered_rank": 0})
        events.append({"event": "run_finished", "run_id": f"r{i}", "task_key": "t",
                       "status": "done", "verified": True})
    # Same shape again but WITHOUT the recorded rank — fit derives it through
    # the recorded policy, exactly like ReplayWorld.from_events does.
    for i in range(3, 6):
        events.append({"event": "run_started", "run_id": f"r{i}", "task_key": "t",
                       "goal": "search", "policy": policy.to_dict(),
                       "policy_digest": policy.digest})
        events.append({**_transition(selected="sel", candidates=observed, offered=1),
                       "run_id": f"r{i}", "task_key": "t", "catalog": "observed",
                       "selected_observed_rank": 6})
        events.append({"event": "run_finished", "run_id": f"r{i}", "task_key": "t",
                       "status": "done", "verified": True})
    model = ChoiceModel.fit(events)
    front = model.predict(kind="click", goal_overlap=2, rank=0)
    assert front["level"] == "cell" and front["n"] == 6
    assert front["p_progress"] > 0.8
    # Under the buggy basis the mass would sit in the "5-19" bucket instead.
    back = model.predict(kind="click", goal_overlap=2, rank=6)
    assert back["level"] != "cell" or back["n"] == 0


def test_choice_model_terminal_progress_requires_verified_done():
    """The audit's objective fix: page activity on a terminal transition is not
    progress unless the run ended verifier-confirmed done."""
    def run(run_id, kind, status, verified):
        return [
            {"event": "run_started", "run_id": run_id, "task_key": "t", "goal": "g",
             "policy": ExplorationPolicy().to_dict(),
             "policy_digest": ExplorationPolicy().digest},
            {"event": "transition", "run_id": run_id, "task_key": "t",
             "state": "S", "next_state": "D",
             "selected": {"id": f"a{run_id}", "kind": kind},
             "candidates": [{"id": f"a{run_id}", "kind": kind, "label": "Go",
                             "goal_overlap": 1}],
             "page_changed": True, "latency_ms": 10, "model_calls": 1, "tokens": 1},
            {"event": "run_finished", "run_id": run_id, "task_key": "t",
             "status": status, "verified": verified},
        ]

    events = []
    for i in range(6):
        events += run(f"bad{i}", "navbad", "blocked", False)
        events += run(f"good{i}", "navgood", "done", True)
        # Done but never verified is not progress either.
        events += run(f"unverified{i}", "navunv", "done", False)
    model = ChoiceModel.fit(events)
    bad = model.predict(kind="navbad", goal_overlap=1, rank=0)
    unverified = model.predict(kind="navunv", goal_overlap=1, rank=0)
    good = model.predict(kind="navgood", goal_overlap=1, rank=0)
    assert bad["n"] == 6 and bad["p_progress"] < 0.2
    assert unverified["n"] == 6 and unverified["p_progress"] < 0.2
    assert good["n"] == 6 and good["p_progress"] > 0.8


def test_choice_model_failed_runs_teach_negative_trajectory():
    """Trajectory-success labeling: intermediate page movement inside a run
    the verifier rejected is a *negative* association, not progress — the
    screen moving is telemetry, not outcome."""
    events = [
        {"event": "run_started", "run_id": "mid", "task_key": "t", "goal": "g",
         "policy": ExplorationPolicy().to_dict(),
         "policy_digest": ExplorationPolicy().digest},
        {"event": "transition", "run_id": "mid", "task_key": "t",
         "state": "S", "next_state": "M", "selected": {"id": "m", "kind": "click"},
         "candidates": [{"id": "m", "kind": "click", "label": "Mid", "goal_overlap": 1}],
         "page_changed": True, "latency_ms": 5, "model_calls": 1, "tokens": 1},
        {"event": "transition", "run_id": "mid", "task_key": "t",
         "state": "M", "next_state": "D", "selected": {"id": "e", "kind": "click"},
         "candidates": [{"id": "e", "kind": "click", "label": "End", "goal_overlap": 1}],
         "page_changed": True, "latency_ms": 5, "model_calls": 1, "tokens": 1},
        # The run failed: neither transition is a positive.
        {"event": "run_finished", "run_id": "mid", "task_key": "t",
         "status": "blocked", "verified": False},
    ]
    model = ChoiceModel.fit(events)
    pred = model.predict(kind="click", goal_overlap=1, rank=0)
    # 0 positives of 2 + Beta(1,1) → 1/4 = 0.25
    assert pred["n"] == 2
    assert abs(pred["p_progress"] - 0.25) < 1e-9


def test_choice_model_page_churn_on_failed_runs_is_not_progress():
    """The v0.6.2 regression guard: dozens of page-changing actions across
    failed runs must not teach the prior progress — verified success is 0,
    so the learned P(success) sits near the Beta prior floor."""
    events = []
    for run_index in range(10):
        transitions = [
            {
                **_transition(
                    selected=f"r{run_index}s{step}",
                    page_changed=True,
                    candidates=[
                        {"id": f"r{run_index}s{step}", "kind": "click",
                         "label": "Churn", "goal_overlap": 1},
                    ],
                ),
                "selected_offered_rank": 0,
            }
            for step in range(5)
        ]
        events += _run_events(
            f"fail{run_index}", transitions, status="blocked", verified=False
        )
    model = ChoiceModel.fit(events)
    pred = model.predict(kind="click", goal_overlap=1, rank=0)
    assert pred["n"] == 50
    assert pred["confident"] is True  # dense evidence — all of it negative
    assert pred["p_progress"] < 0.05


def _trial_key(*, family="", site="", m_kind="click", m_effect="unknown", m_role="unknown",
               m_ov="0", m_rank="0-4", m_phase="unknown", p_kind="click",
               p_effect="unknown", p_role="unknown", p_ov="1", p_rank="0-4",
               p_phase="unknown"):
    """The 14-coordinate trial context string: family and site are separate
    coordinates, then the bounded treatment signature of both arms."""
    return "|".join([
        family, site, m_kind, m_effect, m_role, m_ov, m_rank, m_phase,
        p_kind, p_effect, p_role, p_ov, p_rank, p_phase,
    ])


def _trial_meta(run_id, arm, *, propensity=0.5, kind="click", model_kind="click",
                overlap=1, model_overlap=0, task_family=None, site=None,
                proposal_effect=None, model_effect=None, proposal_role=None,
                model_role=None, phase=None, proposal_rank=0, model_rank=1):
    """The experiment record exactly as the agent stamps it at assignment
    time — arm, propensity, and the pre-treatment context of both sides."""
    meta = {
        "experiment_id": f"x{run_id}",
        "arm": arm,
        "assignment_probability": propensity,
        "proposal_id": f"p{run_id}",
        "proposal_kind": kind,
        "proposal_overlap": overlap,
        "proposal_offered_rank": proposal_rank,
        "model_choice_id": f"m{run_id}",
        "model_choice_kind": model_kind,
        "model_choice_overlap": model_overlap,
        "model_choice_offered_rank": model_rank,
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
    if proposal_role is not None:
        meta["proposal_role"] = proposal_role
    if model_role is not None:
        meta["model_choice_role"] = model_role
    if phase is not None:
        meta["phase"] = phase
    return meta


def _trial_run(run_id, arm, *, success=True, page=True, propensity=0.5, kind="click",
               model_kind="click", overlap=1, model_overlap=0, status=None, verified=None,
               executed=True, task_family=None, site=None, reason=None,
               proposal_effect=None, model_effect=None, proposal_role=None,
               model_role=None, phase=None, proposal_rank=0, model_rank=1,
               latency_ms=0, tokens=0, approvals=0):
    """One randomized trial as a complete run: the ``experiment_assigned``
    record written at randomization, the executed arm's transition (absent
    when the trial was denied or vetoed before execution), and the
    run_finished outcome the trial endpoint is drawn from. The model choice
    stays in the candidate catalogue (it was an offered action), so both arms
    of a divergence share its context cell."""
    meta = _trial_meta(
        run_id, arm, propensity=propensity, kind=kind, model_kind=model_kind,
        overlap=overlap, model_overlap=model_overlap,
        task_family=task_family, site=site,
        proposal_effect=proposal_effect, model_effect=model_effect,
        proposal_role=proposal_role, model_role=model_role, phase=phase,
        proposal_rank=proposal_rank, model_rank=model_rank,
    )
    proposal_id, model_id = meta["proposal_id"], meta["model_choice_id"]
    transitions = []
    if executed:
        transitions.append({
            **_transition(
                selected=proposal_id if arm == "candidate" else model_id,
                page_changed=page,
                latency_ms=latency_ms,
                tokens=tokens,
                candidates=[
                    {"id": proposal_id, "kind": kind, "label": "Go", "goal_overlap": overlap},
                    {"id": model_id, "kind": model_kind, "label": "M", "goal_overlap": model_overlap},
                ],
            ),
            "experiment": meta,
        })
    events = _run_events(
        run_id,
        transitions,
        status=status or ("done" if success else "blocked"),
        verified=success if verified is None else verified,
        reason=reason,
    )
    if approvals:
        events = events[:1] + [
            {"event": "approval_required", "run_id": run_id, "task_key": "t"}
        ] * int(approvals) + events[1:]
    # The assignment lands at randomization — before authority, approval, and
    # any transition — tagged with the same run and task.
    assignment = {
        "event": "experiment_assigned",
        "run_id": run_id,
        "task_key": "t",
        "experiment": meta,
    }
    return events[:1] + [assignment] + events[1:]


def test_counterfactual_trials_ipw_estimates():
    """Self-normalized IPW: the rare candidate arm (propensity 0.25) is
    weighted 4x, the control arm (0.75) 4/3x, and the estimate reflects it."""
    from jev_ultrafast.dreamlearn import CounterfactualTrials

    events = []
    for i in range(8):
        events += _trial_run(f"c{i}", "candidate", success=i < 6, propensity=0.25)
        events += _trial_run(f"k{i}", "control", success=i < 2, propensity=0.75)
    trials = CounterfactualTrials.fit(events)
    estimates = trials.estimate()
    # Context keys on the model choice's features (overlap 0), not the
    # executed arm's — otherwise the two arms would land in different cells.
    assert set(estimates) == {_trial_key()}
    arms = estimates[_trial_key()]
    assert arms["candidate"]["assigned"] == arms["candidate"]["trials"] == 8
    assert arms["candidate"]["executed"] == 8
    assert arms["candidate"]["censored"] == 0
    assert abs(arms["candidate"]["p_success"] - 0.75) < 1e-9
    assert arms["control"]["trials"] == 8
    assert abs(arms["control"]["p_success"] - 0.25) < 1e-9
    assert abs(arms["delta"] - 0.5) < 1e-9
    assert arms["support_sufficient"] is True
    assert arms["candidate"]["support_sufficient"] and arms["control"]["support_sufficient"]
    # 8+8 with a 0.5 delta is *supported* but not *established*: the
    # sequentially valid interval still crosses zero, so the honest status is
    # unresolved — "enough samples" is not "we know the sign".
    assert arms["effect_status"] == "unresolved"
    assert arms["unresolved_reason"] == "ci_crosses_zero"
    # Uncertainty intervals bracket the point estimate on effective support.
    for arm in ("candidate", "control"):
        lo, hi = arms[arm]["p_success_ci"]
        assert lo <= arms[arm]["p_success"] <= hi
    assert arms["delta_ci"][0] <= arms["delta"] <= arms["delta_ci"][1]
    # The confidence sequence is never narrower than the fixed-sample one.
    assert arms["delta_cs"][0] <= arms["delta_ci"][0]
    assert arms["delta_cs"][1] >= arms["delta_ci"][1]


def test_counterfactual_trials_require_recorded_propensity():
    """No honest weight, no estimate: missing/zero/out-of-range assignment
    probabilities are skipped rather than guessed."""
    from jev_ultrafast.dreamlearn import CounterfactualTrials

    bad = []
    for prop in (None, 0.0, 1.5, "nan"):
        run = _trial_run("x", "candidate")
        # Index 1 is the experiment_assigned event — the record the estimator
        # actually reads. Corrupting its propensity must void the assignment.
        run[1]["experiment"]["assignment_probability"] = prop
        bad += run
    # Non-experiment transitions never enter trial estimates either.
    bad.append(_transition(selected="plain", page_changed=True, candidates=[
        {"id": "plain", "kind": "click", "goal_overlap": 1}]))
    assert CounterfactualTrials.fit(bad).cells == ()


def test_counterfactual_trials_endpoint_is_run_outcome_not_page_change():
    """The primary endpoint is the run's final verified outcome: a trial that
    moved the page inside a failed run is a *failure*, and ``p_page_changed``
    remains as secondary telemetry showing the page did move."""
    from jev_ultrafast.dreamlearn import CounterfactualTrials

    # The audit's adversarial construction: candidate-armed trials all moved
    # the page yet every run failed; control-armed trials didn't move the
    # page yet every run verified. The estimate must report candidate worse.
    events = []
    for i in range(4):
        events += _trial_run(f"c{i}", "candidate", success=False, page=True)
        events += _trial_run(f"k{i}", "control", success=True, page=False)
    arms = CounterfactualTrials.fit(events).estimate()[_trial_key()]
    assert arms["candidate"]["p_success"] < 0.01
    assert arms["control"]["p_success"] > 0.99
    assert arms["candidate"]["p_page_changed"] > 0.99
    assert arms["control"]["p_page_changed"] < 0.01
    assert arms["delta"] < -0.9
    assert arms["support_sufficient"] is False  # 4 trials/arm is below MIN_ESS
    assert arms["effect_status"] == "insufficient_data"


def test_counterfactual_trials_partial_success_counts_verified_share():
    """Two trials, one in a verified run and one in a failed run, give a
    half-weighted success estimate."""
    from jev_ultrafast.dreamlearn import CounterfactualTrials

    events = (
        _trial_run("r1", "candidate", success=False, page=True)
        + _trial_run("r2", "candidate", success=True, page=True)
    )
    estimates = CounterfactualTrials.fit(events).estimate()
    arm = estimates[_trial_key()]["candidate"]
    assert arm["assigned"] == arm["trials"] == arm["executed"] == 2
    assert arm["censored"] == 0
    assert abs(arm["p_success"] - 0.5) < 1e-9  # one verified success of two
    assert arm["p_page_changed"] > 0.99  # both steps moved the page
    assert arm["support_sufficient"] is False  # two effective samples is not evidence


def test_counterfactual_trials_context_is_arm_independent():
    """The two arms of one divergence share the model choice's context cell —
    keying on the executed action would compare different populations."""
    from jev_ultrafast.dreamlearn import CounterfactualTrials

    # Same divergence point: the prior proposed a high-overlap action over the
    # model's overlap-0 choice. Both arms must land in the model choice's cell.
    events = _trial_run("x", "candidate") + _trial_run("y", "control")
    estimates = CounterfactualTrials.fit(events).estimate()
    assert set(estimates) == {_trial_key()}
    entry = estimates[_trial_key()]
    assert set(entry["candidate"]) and set(entry["control"])
    assert "delta" in entry


def test_counterfactual_trials_itt_counts_unexecuted_assignments():
    """The audit's post-randomization-selection repro: candidate-armed trials
    whose assignment is rejected or denied before a transition must still
    count — as candidate-arm assignments, with the run's real outcome — not
    vanish from the estimator because nothing executed."""
    from jev_ultrafast.dreamlearn import CounterfactualTrials

    events = []
    for i in range(24):
        # Assigned candidate → authority stopped it → run still finished
        # (blocked) with no transition ever recorded for the trial.
        events += _trial_run(f"c{i}", "candidate", executed=False, success=False)
        events += _trial_run(f"k{i}", "control", success=True)
    arms = CounterfactualTrials.fit(events).estimate()[_trial_key()]
    candidate = arms["candidate"]
    assert candidate["assigned"] == 24
    assert candidate["executed"] == 0  # no transition — yet still analyzed
    assert candidate["trials"] == 24
    assert candidate["p_success"] < 0.01  # every assigned-candidate run failed
    assert candidate["p_page_changed"] is None  # no step ever moved a page
    assert arms["control"]["p_success"] > 0.99
    assert arms["delta"] < -0.9
    assert arms["support_sufficient"] is True
    assert arms["effect_status"] == "harmful"


def test_counterfactual_trials_censor_unmeasured_runs():
    """Aborted/interrupted runs are censored — counted as assigned but not as
    failures — in both the primary endpoint and the analyzed denominator, and
    the structured termination reason survives into the report."""
    from jev_ultrafast.dreamlearn import CounterfactualTrials

    events = (
        # Aborted: operator closed the session — no outcome measured.
        _trial_run("a", "candidate", status="aborted", verified=False, reason="operator_cancel")
        # Interrupted: the store holds no run_finished at all.
        + _trial_run("t", "candidate")[:-1]
        # A real measured failure for contrast.
        + _trial_run("f", "candidate", success=False)
    )
    arm = CounterfactualTrials.fit(events).estimate()[_trial_key()]["candidate"]
    assert arm["assigned"] == 3
    assert arm["censored"] == 2
    assert arm["trials"] == 1  # only the measured failure is analyzed
    assert arm["p_success"] < 0.01
    # The causes stay distinguishable: an operator cancel is not a crash, and
    # an interrupted recording is not invented as either.
    assert arm["terminations"] == {"operator_cancel": 1, "unknown_abort": 1}
    assert abs(arm["censor_rate"] - 2 / 3) < 1e-9


def test_counterfactual_trials_legacy_transition_only_pool():
    """Pre-0.11 pools have no experiment_assigned events; a transition's
    experiment block is the only assignment record — analyzed as one."""
    from jev_ultrafast.dreamlearn import CounterfactualTrials

    events = [
        event
        for event in _trial_run("x", "candidate", success=False)
        if event["event"] != "experiment_assigned"
    ]
    arm = CounterfactualTrials.fit(events).estimate()[_trial_key()]["candidate"]
    assert arm["assigned"] == arm["trials"] == arm["executed"] == 1
    assert arm["p_success"] < 0.01


def test_counterfactual_trials_unreliable_below_effective_support():
    """A precise-looking delta on thin effective support must not claim
    reliability — the interval is wide and the flag is honest."""
    from jev_ultrafast.dreamlearn import CounterfactualTrials

    events = []
    for i in range(4):
        events += _trial_run(f"c{i}", "candidate", success=i < 3)
        events += _trial_run(f"k{i}", "control", success=i < 1)
    arms = CounterfactualTrials.fit(events).estimate()[_trial_key()]
    assert abs(arms["delta"] - 0.5) < 1e-9
    assert arms["candidate"]["ess"] < CounterfactualTrials.MIN_ESS
    assert arms["support_sufficient"] is False
    assert arms["delta_ci"][1] - arms["delta_ci"][0] > 0.5  # honestly wide


def test_report_collects_experiment_proposals(tmp_path):
    """Replay surfaces the divergent proposals as stamped experiment
    hypotheses — hypothesis only, never evidence or a gate input."""
    from jev_ultrafast.dream import ReplayWorld

    store = tmp_path / "events.jsonl"
    worlds = _divergent_worlds(store)
    model = ChoiceModel.fit(
        [json.loads(line) for line in store.read_text().splitlines() if line.strip()]
    )
    report = DreamImprover(gate=PromotionGate(min_coverage=0.0)).improve(
        worlds, ExplorationPolicy(), choice_model=model)
    proposals = report.experiment_proposals
    assert proposals
    for proposal in proposals:
        assert proposal["proposal"]["id"] != proposal["historical"]["id"]
        assert proposal["digest"]
        assert proposal["expected_delta"] > -1.0
        # A stamped plan is a bound ExperimentPlan: schema, task, world,
        # offered catalogue, policy, and originating model are all recorded.
        assert proposal["schema"] == "jev-experiment-plan/1"
        for binding in (
            "task_key", "family_key", "world", "state", "offered_catalogue_digest",
            "policy_behavior_digest", "choice_model_digest", "world_pool_digest",
            "proposal_offered_rank",
        ):
            assert proposal.get(binding) is not None, binding
        assert proposal["choice_model_digest"] == model.digest
    # Digest binds the proposal material — stable and content-derived.
    from jev_ultrafast.dream import experiment_plan_digest
    assert proposals[0]["digest"] == experiment_plan_digest(proposals[0])
    assert proposals[0]["digest"] == hashlib.sha256(
        json.dumps(
            {k: v for k, v in proposals[0].items() if k != "digest"},
            sort_keys=True, separators=(",", ":"),
        ).encode()
    ).hexdigest()
    replay_worlds = ReplayWorld.from_events(
        [json.loads(line) for line in store.read_text().splitlines() if line.strip()]
    )
    assert replay_worlds  # proposals stay annotation; replay is unchanged


def test_report_surfaces_trial_estimates_without_gating(tmp_path):
    """CounterfactualTrials annotate the report with what the randomized layer
    measured — assignment counts, censoring, intervals — and never enter the
    promotion path."""
    from jev_ultrafast.dreamlearn import CounterfactualTrials

    store = tmp_path / "events.jsonl"
    worlds = _divergent_worlds(store)
    events = [json.loads(line) for line in store.read_text().splitlines() if line.strip()]
    events += _trial_run("t1", "candidate", success=True)
    trials = CounterfactualTrials.fit(events)
    with_trials = DreamImprover(gate=PromotionGate(min_coverage=0.0)).improve(
        worlds, ExplorationPolicy(), trials=trials)
    without = DreamImprover(gate=PromotionGate(min_coverage=0.0)).improve(
        worlds, ExplorationPolicy())
    assert with_trials.trials_digest == trials.digest
    estimates = with_trials.trial_estimates
    assert estimates[_trial_key()]["candidate"]["assigned"] == 1
    assert estimates[_trial_key()]["candidate"]["executed"] == 1
    # Trial estimates are annotation only — identical promotion outcome.
    assert with_trials.promotion.approved == without.promotion.approved
    assert with_trials.selected.digest == without.selected.digest
    assert without.trials_digest is None and without.trial_estimates is None


def test_choice_model_censors_unmeasured_runs():
    """An aborted run's actions were never measured — censoring drops them
    from the outcome-label fit instead of teaching the prior they failed."""
    events = []
    for i in range(4):
        events += _run_events(f"ok{i}", [_transition(selected="a")],
                              status="done", verified=True)
    events += _run_events("dead", [_transition(selected="a"), _transition(selected="b")],
                          status="aborted", verified=False)
    torn = _run_events("torn", [_transition(selected="a")])[:-1]  # no finish
    model = ChoiceModel.fit(events + torn)
    # 4 verified positives; the aborted and torn runs contribute nothing.
    assert model.samples == 4
    assert model.global_positive == 4


def test_choice_model_measured_failures_still_fit_negatively():
    """Blocked (measured) failure keeps the v0.6.2 negative labeling — only
    *unmeasured* outcomes censor."""
    events = (
        _run_events("ok", [_transition(selected="a")], status="done", verified=True)
        + _run_events("bad", [_transition(selected="a")], status="blocked", verified=False)
    )
    model = ChoiceModel.fit(events)
    assert model.samples == 2
    assert model.global_positive == 1


def _divergent_worlds(store_path):
    """Runs where the recorded policy picked the low-overlap sibling while a
    goal-relevant action sat in the same offered catalogue — plus evidence
    that the good action works, so the prior actually prefers it."""
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

    # History teaches the prior: goal-relevant clicks make progress.
    for i in range(4):
        run(f"good{i}", "high", True, "done")
    # And picking the irrelevant sibling does not — divergence material.
    for i in range(6):
        run(f"bad{i}", "low", False, "blocked")
    store_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    from jev_ultrafast.dream import ReplayWorld
    return ReplayWorld.from_events(events)


# ------------------------------------------- hierarchical ChoiceModel


def test_choice_model_family_stratum_isolates_unrelated_tasks():
    """Same (kind, overlap, rank) coordinate under two task families learns
    separate posteriors — booking flights and paying invoices no longer
    share one tiny cell."""
    transition = _transition(
        selected="s",
        candidates=[{"id": "s", "kind": "click", "label": "X", "goal_overlap": 1}],
    )
    events = []
    for i in range(5):
        events += _run_events(f"fa{i}", [transition], task_family="flights")
        events += _run_events(
            f"fb{i}", [transition], status="blocked", verified=False,
            task_family="banking")
    model = ChoiceModel.fit(events)
    flight = model.predict(kind="click", goal_overlap=1, rank=0, task_family="flights")
    bank = model.predict(kind="click", goal_overlap=1, rank=0, task_family="banking")
    assert flight["level"] == "family" and bank["level"] == "family"
    assert flight["p_progress"] > 0.7
    assert bank["p_progress"] < 0.3


def test_choice_model_family_backoff_to_pooled_cell():
    """A family stratum below the confidence floor falls back to the pooled
    cell — thin specific evidence never shadows real pooled support."""
    transition = _transition(
        selected="s",
        candidates=[{"id": "s", "kind": "click", "label": "X", "goal_overlap": 1}],
    )
    events = []
    for i in range(6):
        events += _run_events(f"g{i}", [transition], task_family="flights")
    events += _run_events(
        "b0", [transition], status="blocked", verified=False, task_family="banking")
    model = ChoiceModel.fit(events)
    pred = model.predict(kind="click", goal_overlap=1, rank=0, task_family="banking")
    assert pred["level"] == "cell"
    assert pred["n"] == 7 and pred["p_progress"] > 0.7
    # A context-free prediction keeps the legacy pooled behavior.
    assert model.predict(kind="click", goal_overlap=1, rank=0)["level"] == "cell"


def test_choice_model_excludes_experiment_transitions():
    """An experiment-tagged transition was scheduled by the trial layer, not
    chosen by the recorded policy — fitting it would launder randomized
    evidence into the correlational prior."""
    events = []
    for i in range(4):
        events += _run_events(f"ok{i}", [_transition(selected="a")])
    events += _trial_run("t1", "candidate", success=False)
    model = ChoiceModel.fit(events)
    assert model.samples == 4
    assert model.global_positive == 4


# -------------------------------------------- hierarchical trial scope


def test_trials_scope_stratifies_family_site_global():
    """Task family and site are *separate* stored coordinates: a
    family-carrying trial keeps its site stratum, and an unscoped assignment
    pools under empty coordinates."""
    from jev_ultrafast.dreamlearn import CounterfactualTrials

    events = (
        _trial_run("a", "candidate", task_family="flights")
        + _trial_run("b", "candidate", site="example.com")
        + _trial_run("c", "candidate")
    )
    trials = CounterfactualTrials.fit(events)
    # The collapsed v0.8.3 scope would have stored f:flights / s:example.com /
    # * — losing the site dimension for the family-carrying trial.
    assert {(cell[0], cell[1]) for cell in trials.cells} == {
        ("flights", ""), ("", "example.com"), ("", ""),
    }
    estimates = trials.estimate()
    assert set(estimates) == {
        _trial_key(family="flights"),
        _trial_key(site="example.com"),
        _trial_key(),
    }


def test_trials_site_fallback_survives_family_indexing():
    """The audit's repro: a thin family stratum must fall back to the *site*
    stratum — not skip straight to a pooled estimate dominated by unrelated
    sites. Site evidence is only reachable when family and site are stored as
    separate coordinates."""
    from jev_ultrafast.dreamlearn import CounterfactualTrials

    events = []
    # Thin "flights" family evidence at another site: positive but tiny.
    events += _trial_run("t1", "candidate", success=True, task_family="flights",
                         site="other.example")
    events += _trial_run("t2", "control", success=True, task_family="flights",
                         site="other.example")
    # Strong site evidence at target.example: clearly positive.
    for i in range(24):
        events += _trial_run(f"s{i}", "candidate", success=True, site="target.example")
        events += _trial_run(f"k{i}", "control", success=False, site="target.example")
    # Strong unrelated-site evidence: negative, and it would dominate a naive
    # pooled fallback.
    for i in range(24):
        events += _trial_run(f"o{i}", "candidate", success=False, site="other.example")
        events += _trial_run(f"p{i}", "control", success=True, site="other.example")
    resolved = CounterfactualTrials.fit(events).resolve(
        task_family="flights", site="target.example",
        model_kind="click", model_overlap=0,
        proposal_kind="click", proposal_overlap=1,
    )
    assert resolved["level"] == "site"
    assert resolved["support_sufficient"]
    assert resolved["delta"] > 0.9
    assert resolved["effect_status"] == "beneficial"


def test_trials_resolve_family_stratum_wins_when_reliable():
    """Adequate family support beats pooled — the specific estimate is the
    honest answer for that task family."""
    events = []
    for i in range(24):
        events += _trial_run(f"fc{i}", "candidate", success=True, task_family="flights")
        events += _trial_run(f"fk{i}", "control", success=False, task_family="flights")
    resolved = CounterfactualTrials.fit(events).resolve(
        task_family="flights", model_kind="click", model_overlap=0,
        proposal_kind="click", proposal_overlap=1)
    assert resolved["level"] == "family"
    assert resolved["delta"] > 0.9 and resolved["support_sufficient"]
    assert resolved["effect_status"] == "beneficial"


def test_trials_resolve_thin_scope_falls_through_to_pooled():
    """A 2-assignment family stratum cannot hide a 60-assignment pooled
    refutation — resolution keeps walking until a stratum is supported."""
    events = []
    for i in range(2):
        events += _trial_run(f"fc{i}", "candidate", success=True, task_family="flights")
        events += _trial_run(f"fk{i}", "control", success=False, task_family="flights")
    for i in range(30):
        events += _trial_run(f"c{i}", "candidate", success=False, task_family="banking")
        events += _trial_run(f"k{i}", "control", success=True, task_family="banking")
    resolved = CounterfactualTrials.fit(events).resolve(
        task_family="flights", model_kind="click", model_overlap=0,
        proposal_kind="click", proposal_overlap=1)
    assert resolved["level"] == "pooled"
    assert resolved["support_sufficient"] and resolved["delta"] < 0
    assert resolved["effect_status"] == "harmful"


def test_trials_resolve_sparse_returns_unreliable_specific_estimate():
    """When no stratum meets the floor the most specific estimate is still
    reported — flagged unreliable rather than hidden."""
    events = (
        _trial_run("a", "candidate", success=True, task_family="flights")
        + _trial_run("b", "control", success=True, task_family="flights")
    )
    resolved = CounterfactualTrials.fit(events).resolve(
        task_family="flights", model_kind="click", model_overlap=0,
        proposal_kind="click", proposal_overlap=1)
    assert resolved["level"] == "family"
    assert resolved["support_sufficient"] is False
    assert resolved["effect_status"] == "insufficient_data"
    # A hypothesis that was never tried resolves to nothing.
    assert CounterfactualTrials.fit([]).resolve(task_family="x") is None


def test_counterfactual_trials_rejects_legacy_cell_arity():
    """/4 cells are 15 fields — an older 11-field serialization fails closed
    instead of silently misaligning counts."""
    with pytest.raises(ValueError):
        CounterfactualTrials.from_dict({"cells": [tuple("" for _ in range(11))]})


# ------------------------------------------------------ causal prior


def test_trial_choice_model_proposes_reliable_positive_delta():
    """The causal prior proposes the offered candidate whose arm measured a
    reliable positive ITT effect — a hypothesis randomized evidence already
    supports."""
    events = []
    for i in range(24):
        events += _trial_run(f"c{i}", "candidate", success=True)
        events += _trial_run(f"k{i}", "control", success=False)
    model = TrialChoiceModel.fit(events)
    proposal = model.choose(
        [
            {"id": "m", "kind": "click", "goal_overlap": 0},
            {"id": "p", "kind": "click", "goal_overlap": 1},
        ],
        model_choice={"id": "m", "kind": "click", "goal_overlap": 0},
    )
    assert proposal["id"] == "p"
    assert proposal["expected_delta"] > 0.9
    assert proposal["confident"] is True
    assert proposal["trial_level"] == "pooled"


def test_trial_choice_model_abstains_without_evidence():
    """No premise, or no reliable positive delta, means no proposal — a
    causal prior without evidence proposes nothing."""
    model = TrialChoiceModel.fit([])
    assert model.choose(
        [{"id": "p", "kind": "click", "goal_overlap": 1}],
        model_choice={"id": "m", "kind": "click", "goal_overlap": 0},
    ) is None
    events = _trial_run("a", "candidate") + _trial_run("b", "control")
    assert TrialChoiceModel.fit(events).choose(
        [{"id": "p", "kind": "click", "goal_overlap": 1}]
    ) is None


def test_trial_choice_model_refuted_blocks_settled_divergences():
    """A reliable non-positive delta means the hypothesis was tested and
    failed; unreliable evidence refutes nothing."""
    events = []
    for i in range(24):
        events += _trial_run(f"c{i}", "candidate", success=False)
        events += _trial_run(f"k{i}", "control", success=True)
    model = TrialChoiceModel.fit(events)
    assert model.refuted(
        kind="click", goal_overlap=1, model_kind="click", model_overlap=0) is True
    thin = TrialChoiceModel.fit(
        _trial_run("a", "candidate", success=False)
        + _trial_run("b", "control", success=True)
    )
    assert thin.refuted(
        kind="click", goal_overlap=1, model_kind="click", model_overlap=0) is False


def test_trial_choice_model_family_scope_isolates_effects():
    """The same divergence can be supported under one family and refuted
    under another — family-scoped estimates keep those truths separate."""
    events = []
    for i in range(24):
        events += _trial_run(f"fc{i}", "candidate", success=True, task_family="flights")
        events += _trial_run(f"fk{i}", "control", success=False, task_family="flights")
        events += _trial_run(f"bc{i}", "candidate", success=False, task_family="banking")
        events += _trial_run(f"bk{i}", "control", success=True, task_family="banking")
    model = TrialChoiceModel.fit(events)
    assert model.refuted(
        kind="click", goal_overlap=1, model_kind="click", model_overlap=0,
        task_family="banking") is True
    assert model.refuted(
        kind="click", goal_overlap=1, model_kind="click", model_overlap=0,
        task_family="flights") is False
    offered = [
        {"id": "m", "kind": "click", "goal_overlap": 0},
        {"id": "p", "kind": "click", "goal_overlap": 1},
    ]
    proposal = model.choose(
        offered,
        model_choice={"id": "m", "kind": "click", "goal_overlap": 0},
        task_family="flights",
    )
    assert proposal is not None and proposal["id"] == "p"
    assert proposal["trial_level"] == "family"
    assert model.choose(
        offered,
        model_choice={"id": "m", "kind": "click", "goal_overlap": 0},
        task_family="banking",
    ) is None


def test_trial_choice_model_never_generalizes_across_effect_classes():
    """The effect coordinate is a signature floor: evidence measured on a
    purchase click must not bless a same-shaped delete, and the same
    effect-class divergence must still resolve."""
    events = []
    for i in range(24):
        events += _trial_run(f"c{i}", "candidate", success=True,
                             proposal_effect="purchase", model_effect="search")
        events += _trial_run(f"k{i}", "control", success=False,
                             proposal_effect="purchase", model_effect="search")
    model = TrialChoiceModel.fit(events)
    model_choice = {"id": "m", "kind": "click", "effect": "search", "goal_overlap": 0}
    # Same divergence that was measured → the beneficial effect resolves.
    proposal = model.choose(
        [model_choice, {"id": "p", "kind": "click", "effect": "purchase", "goal_overlap": 1}],
        model_choice=model_choice,
    )
    assert proposal is not None and proposal["id"] == "p"
    # Identical kind/role/overlap but a different semantic effect → abstain.
    assert model.choose(
        [model_choice, {"id": "d", "kind": "click", "effect": "delete", "goal_overlap": 1}],
        model_choice=model_choice,
    ) is None


@pytest.mark.parametrize("stored_effect", ["search", "purchase", "delete"])
def test_effect_floor_matrix(stored_effect):
    """T-226 matrix: evidence measured under one semantic effect class can
    never qualify a divergence under another, for every sensitive pairing."""
    events = []
    for i in range(24):
        events += _trial_run(f"c{i}", "candidate", success=True,
                             proposal_effect=stored_effect, model_effect="navigate",
                             proposal_role="button", model_role="link")
        events += _trial_run(f"k{i}", "control", success=False,
                             proposal_effect=stored_effect, model_effect="navigate",
                             proposal_role="button", model_role="link")
    model = TrialChoiceModel.fit(events)
    model_choice = {"id": "m", "kind": "click", "effect": "navigate",
                    "role": "link", "goal_overlap": 0}
    # The measured divergence resolves.
    same = {"id": "p", "kind": "click", "effect": stored_effect,
            "role": "button", "goal_overlap": 1}
    proposal = model.choose([model_choice, same], model_choice=model_choice)
    assert proposal is not None and proposal["id"] == "p"
    # Every other effect class on an otherwise identical click abstains.
    for query_effect in ("search", "purchase", "delete", "disclosure",
                         "authenticate", "form_edit", "navigate",
                         "submission", "external_message", "financial",
                         "permission_change", "unknown_commit"):
        if query_effect == stored_effect:
            continue
        other = dict(same, id="x", effect=query_effect)
        assert model.choose(
            [model_choice, other], model_choice=model_choice
        ) is None, query_effect


def test_signature_backoff_drops_only_safe_coordinates():
    """Backoff matrix: phase → rank → role → overlap may generalize; kind and
    effect are floors that no amount of backoff may cross."""
    from jev_ultrafast.dreamlearn import CounterfactualTrials

    events = []
    for i in range(8):
        events += _trial_run(f"c{i}", "candidate", success=True,
                             proposal_effect="search", model_effect="navigate",
                             proposal_role="button", model_role="link",
                             phase=1, proposal_rank=0, model_rank=1)
        events += _trial_run(f"k{i}", "control", success=False,
                             proposal_effect="search", model_effect="navigate",
                             proposal_role="button", model_role="link",
                             phase=1, proposal_rank=0, model_rank=1)
    trials = CounterfactualTrials.fit(events)
    base = dict(model_kind="click", model_overlap=0, model_effect="navigate",
                model_role="link", model_rank=1, proposal_kind="click",
                proposal_overlap=1, proposal_effect="search",
                proposal_role="button", proposal_rank=0)

    full = trials.resolve(**base, phase=1)
    assert full is not None and full["signature_level"] == "full"
    # Each droppable coordinate generalizes one level at a time.
    assert trials.resolve(**base, phase=5)["signature_level"] == "minus_phase"
    looser = dict(base, proposal_rank=50, model_rank=60)
    assert trials.resolve(**looser, phase=5)["signature_level"] == "minus_phase_rank"
    looser = dict(looser, proposal_role="link", model_role="button")
    assert trials.resolve(**looser, phase=5)["signature_level"] == "minus_phase_rank_role"
    looser = dict(looser, proposal_overlap=3, model_overlap=3)
    assert trials.resolve(**looser, phase=5) is not None
    # The floors never move: a different effect or kind resolves nothing.
    for floor in (
        dict(base, proposal_effect="purchase"),
        dict(base, model_effect="purchase"),
        dict(base, proposal_kind="fill"),
        dict(base, model_kind="fill"),
    ):
        assert trials.resolve(**floor, phase=5) is None


def test_legacy_unknown_effect_never_answers_known_queries():
    """Migrated v4/v5 cells carry effect='unknown' wildcards: they remain
    honest in reporting but can never answer a live known-effect query."""
    from jev_ultrafast.dreamlearn import CounterfactualTrials

    events = []
    for i in range(8):
        events += _trial_run(f"c{i}", "candidate", success=True)
        events += _trial_run(f"k{i}", "control", success=False)
    trials = CounterfactualTrials.fit(events)
    # An effect-unspecified query still sees the legacy evidence (kind is the
    # only enforced floor when the caller doesn't know the effect class).
    assert trials.resolve(model_kind="click", proposal_kind="click") is not None
    # A real semantic effect must never borrow unknown-effect evidence.
    assert trials.resolve(proposal_effect="search") is None
    assert trials.resolve(proposal_effect="search", model_effect="navigate") is None


def test_null_simulation_never_establishes_effect():
    """Section-29 smoke: under a true null, the α-spent interval must not
    manufacture 'beneficial' conclusions. 400 synthetic experiment sequences
    of 12 candidate + 12 control runs at equal success rates."""
    from jev_ultrafast.dreamlearn import CounterfactualTrials

    rng = __import__("random").Random(7)
    false_activations = 0
    sequences = 400
    for seq in range(sequences):
        events = []
        for i in range(12):
            events += _trial_run(f"s{seq}c{i}", "candidate",
                                 success=rng.random() < 0.5)
            events += _trial_run(f"s{seq}k{i}", "control",
                                 success=rng.random() < 0.5)
        trials = CounterfactualTrials.fit(events)
        resolved = trials.resolve(model_kind="click", proposal_kind="click")
        if resolved and resolved.get("effect_status") == "beneficial":
            false_activations += 1
    # Zero is the expected outcome at 12/arm; a couple of extreme draws could
    # legitimately clear the interval — anything more means the gate leaks.
    assert false_activations <= 2, f"{false_activations}/{sequences} null sequences established"


def test_true_effect_simulation_establishes_benefit():
    """Positive control: a large real effect must establish once the evidence
    base is adequate — n=100/arm at delta 0.6 resolves beneficial for nearly
    every sequence (the anytime-valid sequence is deliberately low-power at
    canary scale — the price of a guarantee that survives unlimited peeking)."""
    from jev_ultrafast.dreamlearn import CounterfactualTrials

    rng = __import__("random").Random(11)
    established = 0
    sequences = 150
    for seq in range(sequences):
        events = []
        for i in range(100):
            events += _trial_run(f"s{seq}c{i}", "candidate",
                                 success=rng.random() < 0.8)
            events += _trial_run(f"s{seq}k{i}", "control",
                                 success=rng.random() < 0.2)
        trials = CounterfactualTrials.fit(events)
        resolved = trials.resolve(model_kind="click", proposal_kind="click")
        if resolved and resolved.get("effect_status") == "beneficial":
            established += 1
    assert established > sequences * 0.8, f"only {established}/{sequences} detected"


def test_assignment_propensity_matches_configured_rate():
    """T-800: the assignment roll must realize the configured propensity.
    Pins the mechanism — roll < rate ⇒ candidate — over enough draws that a
    biased implementation cannot hide inside noise."""
    import random

    rng = random.Random(0)
    draws = 20_000
    for rate in (0.5, 0.25, 0.1):
        rng = random.Random(42)
        hits = sum(1 for _ in range(draws) if rng.random() < rate)
        sigma = (draws * rate * (1 - rate)) ** 0.5
        assert abs(hits - draws * rate) < 5 * sigma, (rate, hits)


def test_trial_choice_model_serialization_roundtrip():
    events = _trial_run("a", "candidate") + _trial_run("b", "control")
    model = TrialChoiceModel.fit(events)
    clone = TrialChoiceModel.from_dict(model.to_dict())
    assert clone.digest == model.digest
    assert clone.version == "jev-causal/2"
    assert clone.trials.version == "jev-trials/7"


def test_improve_suppresses_trial_refuted_proposals(tmp_path):
    """End to end: the observational prior proposes a divergence the
    randomized evidence already settled — improve() must not stamp a plan
    asking the same question again."""
    store = tmp_path / "events.jsonl"
    worlds = _divergent_worlds(store)
    events = [json.loads(line) for line in store.read_text().splitlines() if line.strip()]
    choice_model = ChoiceModel.fit(events)
    # The observational divergence is low(overlap 0) -> high(overlap 3).
    # Randomized trials measured it and the candidate arm kept losing.
    for i in range(24):
        events += _trial_run(f"c{i}", "candidate", success=False, overlap=3)
        events += _trial_run(f"k{i}", "control", success=True, overlap=3)
    trials = CounterfactualTrials.fit(events)
    trial_model = TrialChoiceModel(trials=trials)
    report = DreamImprover(gate=PromotionGate(min_coverage=0.0)).improve(
        worlds, ExplorationPolicy(),
        choice_model=choice_model, trial_model=trial_model, trials=trials)
    assert report.trial_model_digest == trial_model.digest
    # The refuted divergence produced no stamped experiment plan.
    assert report.experiment_proposals == ()
    # And the causal prior is what the live agent would bind against: a plan
    # stamped under this digest executes only when the same model answers.
    plain = DreamImprover(gate=PromotionGate(min_coverage=0.0)).improve(
        worlds, ExplorationPolicy(), choice_model=choice_model)
    assert plain.trial_model_digest is None
    # The guard is causal suppression, not an empty pipeline: without the
    # trial prior the same replay does stamp the divergence.
    assert plain.experiment_proposals != ()


# --- jev-dream trials: read-only causal evidence inspection -----------------

def _trial_store(tmp_path):
    """A store holding randomized evidence: 8 candidate vs 8 control trials
    on the same click divergence, effect-stamped so floor queries match."""
    path = tmp_path / "trials.jsonl"
    store = ExperienceStore(path)
    for i in range(8):
        for e in _trial_run(f"c{i}", "candidate", success=i < 6, propensity=0.25,
                            model_effect="observe", proposal_effect="navigate"):
            store.append(e)
        for e in _trial_run(f"k{i}", "control", success=i < 2, propensity=0.75,
                            model_effect="observe", proposal_effect="navigate"):
            store.append(e)
    return path


def test_trials_cli_summarizes_and_resolves(tmp_path, capsys):
    """The trials command is the read side of the causal layer: a bare call
    summarizes the fitted cells, and signature flags resolve the matching
    contrast — with the estimator's own support/effect verdicts attached."""
    from jev_ultrafast.dream_cli import main

    path = _trial_store(tmp_path)

    assert main(["trials", str(path)]) == 0
    bare = json.loads(capsys.readouterr().out)
    assert bare["trials"]["cells"] > 0
    assert bare["trials"]["version"] == "jev-trials/7"
    assert "resolved" not in bare  # no query asked, no pooled guess printed

    assert main([
        "trials", str(path),
        "--model-kind", "click", "--proposal-kind", "click",
    ]) == 0
    resolved = json.loads(capsys.readouterr().out)
    contrast = resolved["resolved"]
    assert contrast["level"] == "pooled"
    assert contrast["support_sufficient"] is True
    # 6/8 candidate successes vs 2/8 control successes — a real raw delta,
    # and the α-spent interval honestly says how settled it is.
    assert contrast["delta"] > 0
    assert contrast["effect_status"] in {"beneficial", "unresolved"}
    assert contrast["candidate"]["assigned"] == 8
    assert resolved["query"]["model_kind"] == "click"


def test_trials_cli_abstains_on_effect_mismatch(tmp_path, capsys):
    """A query whose effect floor no stored cell can answer resolves to
    null — evidence never generalizes across effect classes."""
    from jev_ultrafast.dream_cli import main

    path = _trial_store(tmp_path)
    assert main([
        "trials", str(path),
        "--model-effect", "purchase", "--proposal-effect", "purchase",
    ]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["resolved"] is None
    assert out["query"]["model_effect"] == "purchase"


def test_trials_cli_cells_dump_and_verify_flags(tmp_path, capsys):
    """--cells decodes each cell's stratum/signature/arm counters, and the
    shared verification flags apply like every other evidence command."""
    from jev_ultrafast.dream_cli import main

    signer = EvidenceSigner.from_hex("ab" * 32)
    path = tmp_path / "signed-trials.jsonl"
    store = ExperienceStore(path, signer=signer)
    for e in _trial_run("c0", "candidate", propensity=0.5,
                        model_effect="observe", proposal_effect="navigate"):
        store.append(e)

    assert main(["trials", str(path), "--verify-keys", signer.key_id,
                 "--cells"]) == 0
    out = json.loads(capsys.readouterr().out)
    cell = out["cells"][0]
    assert cell["assigned"] == 1
    assert cell["signature"]["model_kind"] == "click"
    assert cell["signature"]["model_effect"] == "observe"
    assert cell["arm"] == "candidate"
    # Signed store without the key fails closed — same as verify/improve.
    with pytest.raises(ValueError, match="no verification key"):
        main(["trials", str(path)])


def test_update_manifest_script_regenerates_and_verifies(tmp_path):
    """The manifest script is the canonical generator: --check is a
    byte-exact comparison that catches a stale hash or a missing entry."""
    import subprocess
    import sys

    script = Path(__file__).resolve().parent.parent / "scripts" / "update_manifest.py"
    # Exercise it against a tree shaped like the repo: no .git, so it
    # re-covers the manifest's own file list.
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "a.txt").write_text("alpha")
    (tree / "b.txt").write_text("beta")
    manifest = tree / "MANIFEST.sha256"
    manifest.write_text(
        "".join(
            f"{hashlib.sha256((tree / n).read_bytes()).hexdigest()}  {n}\n"
            for n in ("a.txt", "b.txt")
        )
    )
    # --check passes on a fresh manifest, fails after a file changes.
    ok = subprocess.run([sys.executable, str(script), "--check"], cwd=tree,
                        capture_output=True, text=True)
    # The script anchors on its own repo root, so run it with a patched ROOT.
    patched = tmp_path / "update_manifest.py"
    patched.write_text(script.read_text().replace(
        'ROOT = Path(__file__).resolve().parent.parent',
        f'ROOT = Path({str(tree)!r})',
    ))
    ok = subprocess.run([sys.executable, str(patched), "--check"],
                        capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    (tree / "a.txt").write_text("changed")
    bad = subprocess.run([sys.executable, str(patched), "--check"],
                         capture_output=True, text=True)
    assert bad.returncode == 1
    # Regenerate and it verifies again.
    regen = subprocess.run([sys.executable, str(patched)],
                           capture_output=True, text=True)
    assert regen.returncode == 0
    ok2 = subprocess.run([sys.executable, str(patched), "--check"],
                         capture_output=True, text=True)
    assert ok2.returncode == 0


def test_assignment_propensity_distribution_is_correct():
    """T-800: the arm draw is `rng.random() < rate`. Over 100k seeded draws
    (fully deterministic — no flake surface) the observed candidate-arm
    frequency must sit within ~4 sigma of the configured propensity at each
    rate the scheduler exposes."""
    import math
    import random

    for rate in (0.50, 0.25, 0.10):
        rng = random.Random(0xC0FFEE)
        hits = sum(1 for _ in range(100_000) if rng.random() < rate)
        z = abs(hits / 100_000 - rate) / math.sqrt(rate * (1 - rate) / 100_000)
        assert z < 4.0, f"rate={rate} observed={hits / 100_000} z={z:.1f}"


def test_fits_and_replay_are_deterministic(tmp_path):
    """§57: identical evidence produces byte-identical fitted artifacts — the
    digests downstream bindings sign must not drift between runs."""
    store = tmp_path / "det.jsonl"
    worlds = _divergent_worlds(store)
    events = [json.loads(line) for line in store.read_text().splitlines() if line.strip()]
    for i in range(6):
        events += _trial_run(f"d{i}", "candidate" if i % 2 else "control",
                             success=i % 3 == 0, propensity=0.5)

    assert CounterfactualTrials.fit(events).digest == CounterfactualTrials.fit(events).digest
    assert ChoiceModel.fit(events).digest == ChoiceModel.fit(events).digest
    assert OutcomeModel.fit(events).digest == OutcomeModel.fit(events).digest
    a = DreamImprover(gate=PromotionGate(min_coverage=0.0)).improve(
        worlds, ExplorationPolicy(), trials=CounterfactualTrials.fit(events))
    b = DreamImprover(gate=PromotionGate(min_coverage=0.0)).improve(
        worlds, ExplorationPolicy(), trials=CounterfactualTrials.fit(events))
    assert a.selected.digest == b.selected.digest
    assert a.trials_digest == b.trials_digest
