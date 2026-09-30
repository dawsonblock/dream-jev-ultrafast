import hashlib
import json
from dataclasses import replace

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
from jev_ultrafast.dreamlearn import ChoiceModel, CostModel, OutcomeModel, overlap_bucket, rank_bucket
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
    # Mixed cells within one kind: goal-relevant clicks changed pages,
    # irrelevant ones did not, so the cell prior beats the kind fallback.
    events = [
        *[
            {
                **_transition(selected=f"g{i}", page_changed=True, candidates=[
                    {"id": f"g{i}", "kind": "click", "label": "Search", "goal_overlap": 2},
                ]),
                "selected_offered_rank": 0,
            }
            for i in range(5)
        ],
        *[
            {
                **_transition(selected=f"k{i}", page_changed=False, candidates=[
                    {"id": f"k{i}", "kind": "click", "label": "Ctl", "goal_overlap": 0},
                ]),
                "selected_offered_rank": 0,
            }
            for i in range(5)
        ],
    ]
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
    events = [
        *[
            {
                **_transition(selected=f"k{i}", page_changed=False, candidates=[
                    {"id": f"k{i}", "kind": "click", "label": "Known", "goal_overlap": 0},
                ]),
                "selected_offered_rank": 0,
            }
            for i in range(8)
        ],
        *[
            {
                **_transition(selected=f"s{i}", page_changed=True, candidates=[
                    {"id": f"s{i}", "kind": "select", "label": "Pick", "goal_overlap": 0},
                ]),
                "selected": {"id": f"s{i}", "kind": "select"},
                "selected_offered_rank": 0,
            }
            for i in range(4)
        ],
    ]
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
    front_events = [
        {**_transition(selected=f"s{i}", page_changed=True, candidates=[
            {"id": f"s{i}", "kind": "click", "label": "Search", "goal_overlap": 2},
        ]), "selected_offered_rank": 0}
        for i in range(6)
    ]
    back_events = [
        # The selected action honestly sits at offered rank 9.
        {**_transition(selected=f"b{i}", page_changed=False, offered=10, candidates=[
            *[{"id": f"x{i}-{j}", "kind": "click", "label": f"X{j}", "goal_overlap": 0}
              for j in range(9)],
            {"id": f"b{i}", "kind": "click", "label": "Back", "goal_overlap": 0},
        ]), "selected_offered_rank": 9}
        for i in range(6)
    ]
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


def test_choice_model_nonterminal_steps_still_count_page_changes():
    """Intermediate transitions keep the page-change signal — only the run's
    terminal transition is gated on verified completion."""
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
        # The run failed: the terminal transition is not progress, but the
        # intermediate page change still counts.
        {"event": "run_finished", "run_id": "mid", "task_key": "t",
         "status": "blocked", "verified": False},
    ]
    model = ChoiceModel.fit(events)
    pred = model.predict(kind="click", goal_overlap=1, rank=0)
    # 1 positive of 2 + Beta(1,1) → 2/4 = 0.5
    assert pred["n"] == 2
    assert abs(pred["p_progress"] - 0.5) < 1e-9


def _trial(run_id, arm, *, positive=True, propensity=0.5, kind="click", model_kind="click"):
    """One experiment-tagged transition: an executed arm with a recorded
    assignment propensity — the denominator honest off-policy weights need."""
    aid = f"{arm}{run_id}"
    return {
        "run_id": run_id,
        **_transition(selected=aid, page_changed=positive, candidates=[
            {"id": aid, "kind": kind, "label": "Go", "goal_overlap": 1},
        ]),
        "experiment": {
            "arm": arm,
            "assignment_probability": propensity,
            "proposal_id": aid,
            "proposal_kind": kind,
            "model_choice_id": f"m{run_id}",
            "model_choice_kind": model_kind,
        },
    }


def test_counterfactual_trials_ipw_estimates():
    """Self-normalized IPW: the rare candidate arm (propensity 0.25) is
    weighted 4x, the control arm (0.75) 4/3x, and the estimate reflects it."""
    from jev_ultrafast.dreamlearn import CounterfactualTrials

    events = []
    for i in range(4):
        events.append(_trial(f"c{i}", "candidate", positive=i < 3, propensity=0.25))
        events.append(_trial(f"k{i}", "control", positive=i == 0, propensity=0.75))
    trials = CounterfactualTrials.fit(events)
    estimates = trials.estimate()
    assert set(estimates) == {"click|1"}
    arms = estimates["click|1"]
    assert arms["candidate"]["trials"] == 4
    assert abs(arms["candidate"]["p_progress"] - 0.75) < 1e-9
    assert arms["control"]["trials"] == 4
    assert abs(arms["control"]["p_progress"] - 0.25) < 1e-9
    assert abs(arms["delta"] - 0.5) < 1e-9
    assert arms["candidate"]["reliable"] and arms["control"]["reliable"]


def test_counterfactual_trials_require_recorded_propensity():
    """No honest weight, no estimate: missing/zero/out-of-range assignment
    probabilities are skipped rather than guessed."""
    from jev_ultrafast.dreamlearn import CounterfactualTrials

    bad = []
    for prop in (None, 0.0, 1.5, "nan"):
        t = _trial("x", "candidate")
        t["experiment"]["assignment_probability"] = prop
        bad.append(t)
    # Non-experiment transitions never enter trial estimates either.
    bad.append(_transition(selected="plain", page_changed=True, candidates=[
        {"id": "plain", "kind": "click", "goal_overlap": 1}]))
    assert CounterfactualTrials.fit(bad).cells == ()


def test_counterfactual_trials_terminal_requires_verified_done():
    """Experiment outcomes use the same verified-progress label as the prior:
    a terminal trial counts only when the run ended verifier-confirmed done."""
    from jev_ultrafast.dreamlearn import CounterfactualTrials

    events = [
        {"event": "run_started", "run_id": "r1", "task_key": "t",
         "policy": ExplorationPolicy().to_dict(),
         "policy_digest": ExplorationPolicy().digest},
        _trial("r1", "candidate", positive=True, propensity=0.5),
        # Failed run: the terminal trial is not progress even though the page moved.
        {"event": "run_finished", "run_id": "r1", "task_key": "t",
         "status": "blocked", "verified": False},
        {"event": "run_started", "run_id": "r2", "task_key": "t",
         "policy": ExplorationPolicy().to_dict(),
         "policy_digest": ExplorationPolicy().digest},
        _trial("r2", "candidate", positive=True, propensity=0.5),
        {"event": "run_finished", "run_id": "r2", "task_key": "t",
         "status": "done", "verified": True},
    ]
    estimates = CounterfactualTrials.fit(events).estimate()
    arm = estimates["click|1"]["candidate"]
    assert arm["trials"] == 2
    assert abs(arm["p_progress"] - 0.5) < 1e-9  # one verified success of two


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
    # Digest binds the proposal material — stable and content-derived.
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
