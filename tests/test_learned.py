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
from jev_ultrafast.dreamlearn import CostModel, OutcomeModel, overlap_bucket, rank_bucket
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
