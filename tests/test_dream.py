import json
import multiprocessing

import pytest

from jev_ultrafast.dream import (
    CanaryEvidence,
    CanaryGate,
    CanaryMetrics,
    DreamImprover,
    ExperienceStore,
    ExplorationPolicy,
    HealthGate,
    PolicyRegistry,
    PromotionGate,
    ReplaySimulator,
    ReplayWorld,
    candidate_catalog_digest,
    task_key,
)
from jev_ultrafast.model import candidate_actions
from jev_ultrafast.trace import DreamTraceRecorder


def _append_run(store, run_id, goal, transitions, *, status, verified, policy=None):
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


def _branching_world(tmp_path):
    store = ExperienceStore(tmp_path / "experience.jsonl")
    candidates = [
        {"id": "click-high", "kind": "click", "label": "Search nearby", "goal_overlap": 1},
        {"id": "fill-low", "kind": "fill", "label": "Destination", "goal_overlap": 0},
    ]
    _append_run(store, "bad", "find destination", [{
        "state": "S", "next_state": "B", "selected": {"id": "click-high", "kind": "click"},
        "candidates": candidates, "page_changed": True, "latency_ms": 50, "model_calls": 1,
        "tokens": 100, "stale_or_failure": 0, "risk_events": 0,
    }], status="blocked", verified=False)
    _append_run(store, "good", "find destination", [
        {
            "state": "S", "next_state": "G", "selected": {"id": "fill-low", "kind": "fill"},
            "candidates": candidates, "page_changed": True, "latency_ms": 50, "model_calls": 1,
            "tokens": 100, "stale_or_failure": 0, "risk_events": 0,
        },
        {
            "state": "G", "next_state": "H", "selected": {"id": "submit", "kind": "click"},
            "candidates": [{"id": "submit", "kind": "click", "label": "Search", "goal_overlap": 0}],
            "page_changed": True, "latency_ms": 50, "model_calls": 1,
            "tokens": 100, "stale_or_failure": 0, "risk_events": 0,
        },
    ], status="done", verified=True)
    return store, ReplayWorld.from_events(store.load())


def _efficiency_world(tmp_path):
    store = ExperienceStore(tmp_path / "efficiency.jsonl")
    goal = "find destination"
    candidates = [
        {"id": f"c{i}", "kind": "click", "label": f"Control {i}", "goal_overlap": 0}
        for i in range(229)
    ]
    candidates.insert(0, {"id": "selected", "kind": "fill", "label": "Destination", "goal_overlap": 5})
    _append_run(store, "success", goal, [{
        "state": "S", "next_state": "DONE", "selected": {"id": "selected", "kind": "fill"},
        "candidates": candidates, "page_changed": True, "latency_ms": 50, "model_calls": 1,
        "tokens": 1000, "stale_or_failure": 0, "risk_events": 0,
    }], status="done", verified=True)
    return ReplayWorld.from_events(store.load())


def _coverage_miss_world(tmp_path):
    store = ExperienceStore(tmp_path / "coverage.jsonl")
    goal = "click final control"
    candidates = [
        {"id": f"c{i}", "kind": "click", "label": f"Control {i}", "goal_overlap": 0}
        for i in range(20)
    ]
    _append_run(store, "late", goal, [{
        "state": "S", "next_state": "DONE", "selected": {"id": "c19", "kind": "click"},
        "candidates": candidates, "page_changed": True, "latency_ms": 10, "model_calls": 1,
        "tokens": 10, "stale_or_failure": 0, "risk_events": 0,
    }], status="done", verified=True)
    return ReplayWorld.from_events(store.load())


def test_exploration_policy_is_bounded_and_digest_stable():
    a = ExplorationPolicy()
    b = ExplorationPolicy.from_dict(a.to_dict())
    assert a.digest == b.digest
    with pytest.raises(ValueError, match="model_action_limit"):
        ExplorationPolicy(model_action_limit=500)
    with pytest.raises(ValueError, match="Unknown"):
        ExplorationPolicy.from_dict({**a.to_dict(), "exec_shell": True})


def test_live_candidate_selection_obeys_dream_policy():
    actions = [
        {"id": f"c{i}", "kind": "click", "node": i, "label": f"Other control {i}"}
        for i in range(20)
    ]
    actions += [
        {"id": "destination", "kind": "fill", "node": 100, "label": "Destination"},
        {"id": "wait", "kind": "wait", "label": "Wait"},
    ]
    policy = ExplorationPolicy(model_action_limit=16, click_quota=14, fill_quota=1)
    selected, omitted = candidate_actions(actions, "destination", exploration_policy=policy)
    assert len(selected) == 16
    assert any(a["id"] == "wait" for a in selected)
    assert any(a["id"] == "destination" for a in selected)
    assert omitted == len(actions) - 16


def test_experience_store_hash_chain_detects_tamper(tmp_path):
    path = tmp_path / "events.jsonl"
    store = ExperienceStore(path)
    store.append({"run_id": "1", "event": "a"})
    store.append({"run_id": "1", "event": "b"})
    assert len(store.load()) == 2
    lines = path.read_text().splitlines()
    payload = json.loads(lines[0])
    payload["event"] = "tampered"
    lines[0] = json.dumps(payload)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="integrity failure"):
        ExperienceStore(path).load()


def test_trace_recorder_bounds_and_redacts_sensitive_content(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_MODEL_PRIVACY", "basic")
    store = ExperienceStore(tmp_path / "trace.jsonl")
    recorder = DreamTraceRecorder(store, goal="Email me at person@example.com")
    page = {"fingerprint": "S", "url": "https://example.com/?token=secret"}
    recorder.start(page, ExplorationPolicy())
    recorder.finish(status="claimed_done", verified=False)
    events = store.load()
    assert "person@example.com" not in events[0]["goal"]
    assert "token=secret" not in events[0]["url"]


def test_replay_never_hallucinates_unrecorded_outcome(tmp_path):
    worlds = _coverage_miss_world(tmp_path)
    baseline = ReplaySimulator(worlds).evaluate(ExplorationPolicy())
    assert baseline.metrics.successes == 1
    constrained = ReplaySimulator(worlds).evaluate(ExplorationPolicy(model_action_limit=16))
    assert constrained.metrics.coverage_misses == 1
    assert constrained.metrics.successes == 0



def test_dream_improver_can_select_more_efficient_compatible_policy(tmp_path):
    worlds = _efficiency_world(tmp_path)
    report = DreamImprover(gate=PromotionGate(min_coverage=1.0)).improve(worlds, ExplorationPolicy())
    assert report.promotion.approved
    assert report.selected.digest != report.baseline.digest
    assert report.promotion.candidate.successes == 1
    assert report.promotion.candidate.offered_candidates < report.promotion.baseline.offered_candidates
    assert report.live_canary_required is True



def test_promotion_gate_rejects_low_coverage(tmp_path):
    worlds = _coverage_miss_world(tmp_path)
    sim = ReplaySimulator(worlds)
    baseline = sim.evaluate(ExplorationPolicy())
    candidate = sim.evaluate(ExplorationPolicy(model_action_limit=16))
    decision = PromotionGate(min_coverage=1.0).assess(baseline, candidate)
    assert not decision.approved
    assert "coverage" in decision.reason



def test_canary_gate_requires_real_verified_success():
    gate = CanaryGate(min_tasks=3, min_baseline_tasks=3)
    baseline = CanaryMetrics(tasks=3, verified_successes=2)
    assert not gate.assess(baseline, CanaryMetrics(tasks=2, verified_successes=2)).approved
    assert not gate.assess(baseline, CanaryMetrics(tasks=3, verified_successes=0, task_families=4)).approved
    assert gate.assess(baseline, CanaryMetrics(tasks=3, verified_successes=3, task_families=4)).approved


def test_policy_registry_requires_replay_then_canary(tmp_path):
    worlds = _efficiency_world(tmp_path)
    report = DreamImprover(gate=PromotionGate(min_coverage=1.0)).improve(worlds, ExplorationPolicy())
    registry = PolicyRegistry(tmp_path / "policy.json")
    staged = registry.stage(report)
    assert staged["digest"] == report.selected.digest
    with pytest.raises(ValueError, match="Unbound canary metrics"):
        registry.promote(CanaryMetrics(3, 3), CanaryMetrics(3, 3))
    permissive = CanaryGate(min_tasks=3, min_baseline_tasks=3, min_task_families=0, min_paired_task_families=0)
    with pytest.raises(ValueError, match="Canary promotion rejected"):
        registry.promote(
            CanaryMetrics(3, 3), CanaryMetrics(2, 2),
            gate=permissive, allow_unbound_metrics=True,
        )
    active = registry.promote(
        CanaryMetrics(3, 2, failures=1), CanaryMetrics(3, 3),
        gate=permissive, allow_unbound_metrics=True,
    )
    assert active["digest"] == report.selected.digest
    assert registry.active_policy().digest == report.selected.digest


def test_canary_metrics_are_derived_from_hash_chained_run_evidence(tmp_path):
    store = ExperienceStore(tmp_path / "canary.jsonl")
    baseline = ExplorationPolicy()
    candidate = ExplorationPolicy(name="candidate", version=2, model_action_limit=218)
    transition = [{
        "state": "S", "next_state": "D", "selected": {"id": "go", "kind": "click"},
        "candidates": [{"id": "go", "kind": "click", "label": "Go", "goal_overlap": 1}],
        "page_changed": True, "latency_ms": 15, "model_calls": 1, "tokens": 20,
        "stale_or_failure": 0, "risk_events": 0,
    }]
    _append_run(store, "b1", "goal", transition, status="done", verified=True, policy=baseline)
    _append_run(store, "c1", "goal", transition, status="done", verified=True, policy=candidate)
    _append_run(store, "c2", "goal", transition, status="blocked", verified=False, policy=candidate)
    metrics = CanaryMetrics.from_events(store.load(), candidate.digest)
    assert metrics.tasks == 2
    assert metrics.verified_successes == 1
    assert metrics.failures == 1
    assert metrics.latency_ms == 30


def test_agent_emits_replayable_trace_without_changing_executor_contract(tmp_path, monkeypatch):
    from jev_ultrafast import agent as loop

    first = {
        "fingerprint": "S",
        "url": "https://example.test/start",
        "title": "Start",
        "text": "Start",
        "actions": [{"id": "go", "kind": "click", "node": 1, "label": "Go", "role": "button"}],
    }
    second = {
        "fingerprint": "D",
        "url": "https://example.test/done",
        "title": "Done",
        "text": "Done",
        "actions": [],
    }

    class FakeBrowser:
        def __init__(self, _url):
            self.current = first

        def observe(self, screenshot=False):
            return self.current

        def fresh(self, _page, action=None):
            return True

        def act(self, action, page, text=None):
            assert action["id"] == "go" and page["fingerprint"] == "S" and text is None
            self.current = second

        def close(self):
            pass

    monkeypatch.setattr(loop, "Browser", FakeBrowser)
    path = tmp_path / "agent.jsonl"
    agent = loop.Agent(
        "https://example.test",
        "Open done",
        verifier=lambda page: {"passed": page["fingerprint"] == "D"},
        dream_store=path,
    )
    agent.state["decision"] = {
        "choice": "go", "operation": "CLICK", "target": "1", "probabilities": {"go": 1.0},
        "confidence": 1.0, "latency_ms": 5, "usage": {"total_tokens": 10},
    }
    agent.state["status"] = "predicted"
    agent.command("act", {"fingerprint": "S"})
    agent.state["decision"] = {
        "choice": "DONE", "operation": "DONE", "target": None, "probabilities": {"DONE": 1.0},
        "confidence": 1.0, "latency_ms": 1, "usage": {},
    }
    agent.state["status"] = "predicted"
    agent.command("act", {"fingerprint": "D"})
    agent.close()

    events = ExperienceStore(path).load()
    assert [event["event"] for event in events] == ["run_started", "transition", "run_finished"]
    worlds = ReplayWorld.from_events(events)
    replay = ReplaySimulator(worlds).evaluate(ExplorationPolicy())
    assert replay.metrics.verified_successes == 1



def _append_concurrent_worker(path, prefix, count):
    store = ExperienceStore(path)
    for index in range(count):
        store.append({"run_id": f"{prefix}-{index}", "event": "concurrent", "index": index})


def test_experience_store_serializes_multiple_processes(tmp_path):
    path = str(tmp_path / "multi.jsonl")
    context = multiprocessing.get_context("spawn")
    workers = [context.Process(target=_append_concurrent_worker, args=(path, f"w{i}", 10)) for i in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(10)
        assert worker.exitcode == 0
    store = ExperienceStore(path)
    verification = store.verify()
    assert verification["events"] == 40
    assert verification["head_hash"] != "0" * 64


def test_trace_records_candidate_catalogue_digest_and_rank(tmp_path):
    store = ExperienceStore(tmp_path / "trace-catalog.jsonl")
    recorder = DreamTraceRecorder(store, goal="Open target")
    page = {"fingerprint": "A", "url": "https://example.test"}
    recorder.start(page, ExplorationPolicy())
    candidates = [
        {"id": "other", "kind": "click", "label": "Other"},
        {"id": "target", "kind": "click", "label": "Target"},
    ]
    recorder.transition(
        before=page,
        after={"fingerprint": "B", "url": "https://example.test/done"},
        action=candidates[1],
        decision={"latency_ms": 1, "usage": {"total_tokens": 2}},
        candidates=candidates,
    )
    recorder.finish(status="done", verified=True)
    event = next(item for item in store.load() if item["event"] == "transition")
    assert event["selected_rank"] == 1
    assert event["candidate_digest"] == candidate_catalog_digest(event["candidates"])
    world = ReplayWorld.from_events(store.load())[0]
    assert world.trajectory[0].candidate_digest == event["candidate_digest"]


def _canary_pair_store(tmp_path, baseline, candidate, *, candidate_success=True):
    store = ExperienceStore(tmp_path / "paired-canary.jsonl")
    transition = [{
        "state": "S", "next_state": "D", "selected": {"id": "go", "kind": "click"},
        "candidates": [{"id": "go", "kind": "click", "label": "Go", "goal_overlap": 1}],
        "page_changed": True, "latency_ms": 10, "model_calls": 1, "tokens": 5,
        "stale_or_failure": 0, "risk_events": 0,
    }]
    for family in range(4):
        goal = f"task family {family}"
        for repeat in range(3):
            _append_run(
                store, f"b-{family}-{repeat}", goal, transition,
                status="done", verified=True, policy=baseline,
            )
            _append_run(
                store, f"c-{family}-{repeat}", goal, transition,
                status="done" if candidate_success else "blocked",
                verified=candidate_success, policy=candidate,
            )
    return store


def test_canary_evidence_is_paired_and_bound_to_store_head(tmp_path):
    baseline = ExplorationPolicy()
    candidate = ExplorationPolicy(name="candidate", version=2, model_action_limit=218)
    store = _canary_pair_store(tmp_path, baseline, candidate)
    events = store.load()
    evidence = CanaryEvidence.from_events(events, baseline.digest, candidate.digest)
    assert evidence.baseline.tasks == 12
    assert evidence.candidate.tasks == 12
    assert evidence.paired_task_families == 4
    assert evidence.ties == 4
    assert evidence.event_head_hash == store.head_hash()
    assert CanaryGate().assess(evidence.baseline, evidence.candidate, evidence=evidence).approved


def test_registry_promotes_only_bound_paired_canary_evidence(tmp_path):
    worlds = _efficiency_world(tmp_path)
    baseline = ExplorationPolicy()
    report = DreamImprover(gate=PromotionGate(min_coverage=1.0)).improve(worlds, baseline)
    registry = PolicyRegistry(tmp_path / "bound-policy.json")
    staged = registry.stage(report)
    store = _canary_pair_store(tmp_path, baseline, report.selected)
    active = registry.promote_from_store(store, baseline_policy_digest=baseline.digest)
    assert active["digest"] == staged["digest"]
    assert active["canary"]["paired_task_families"] == 4
    assert active["canary"]["event_head_hash"] == store.head_hash()


def test_registry_rejects_stale_replay_parent(tmp_path):
    worlds = _efficiency_world(tmp_path)
    baseline = ExplorationPolicy()
    report = DreamImprover(gate=PromotionGate(min_coverage=1.0)).improve(worlds, baseline)
    registry = PolicyRegistry(tmp_path / "stale-policy.json")
    registry.stage(report)
    permissive = CanaryGate(min_tasks=1, min_baseline_tasks=1, min_task_families=0, min_paired_task_families=0)
    registry.promote(
        CanaryMetrics(1, 1), CanaryMetrics(1, 1),
        gate=permissive, allow_unbound_metrics=True,
    )
    with pytest.raises(ValueError, match="baseline is stale"):
        registry.stage(report)


def test_registry_rollback_restores_prior_active_policy(tmp_path):
    baseline = ExplorationPolicy()
    worlds = _efficiency_world(tmp_path)
    registry = PolicyRegistry(tmp_path / "rollback-policy.json")

    first_report = DreamImprover(gate=PromotionGate(min_coverage=1.0)).improve(worlds, baseline)
    registry.stage(first_report)
    permissive = CanaryGate(min_tasks=1, min_baseline_tasks=1, min_task_families=0, min_paired_task_families=0)
    first = registry.promote(
        CanaryMetrics(1, 1), CanaryMetrics(1, 1), gate=permissive, allow_unbound_metrics=True,
    )

    second_report = DreamImprover(gate=PromotionGate(min_coverage=1.0)).improve(worlds, registry.active_policy())
    if not second_report.promotion.approved or second_report.selected.digest == first["digest"]:
        pytest.skip("fixture produced no second replay-approved policy")
    registry.stage(second_report)
    second = registry.promote(
        CanaryMetrics(1, 1), CanaryMetrics(1, 1), gate=permissive, allow_unbound_metrics=True,
    )
    assert second["digest"] != first["digest"]
    restored = registry.rollback(first["digest"])
    assert restored["digest"] == first["digest"]
    assert registry.active_policy().digest == first["digest"]


def test_canary_gate_rejects_zero_task_families():
    gate = CanaryGate(min_tasks=3, min_baseline_tasks=3)
    baseline = CanaryMetrics(tasks=3, verified_successes=3, task_families=4)
    candidate = CanaryMetrics(tasks=3, verified_successes=3, task_families=0)
    decision = gate.assess(baseline, candidate)
    assert not decision.approved
    assert "task families" in decision.reason


def test_experience_store_rejects_reserved_event_keys(tmp_path):
    store = ExperienceStore(tmp_path / "reserved.jsonl")
    for key in ("schema", "tcb_version", "recorded_at_ms", "prev_hash", "event_hash"):
        with pytest.raises(ValueError, match="reserved"):
            store.append({"run_id": "x", "event": "a", key: "spoof"})
    assert store.load() == []


def test_trace_recorder_rejects_reserved_payload_keys(tmp_path):
    store = ExperienceStore(tmp_path / "trace-reserved.jsonl")
    recorder = DreamTraceRecorder(store, goal="goal")
    for key in ("run_id", "task_key", "sequence"):
        with pytest.raises(ValueError, match="reserved"):
            recorder.event("policy_denied", **{key: "spoof"})
    assert store.load() == []


def test_aborted_runs_do_not_count_as_canary_outcomes(tmp_path):
    store = ExperienceStore(tmp_path / "aborted.jsonl")
    transition = [{
        "state": "S", "next_state": "D", "selected": {"id": "go", "kind": "click"},
        "candidates": [{"id": "go", "kind": "click", "label": "Go", "goal_overlap": 1}],
        "page_changed": True, "latency_ms": 10, "model_calls": 1, "tokens": 5,
        "stale_or_failure": 0, "risk_events": 0,
    }]
    policy = ExplorationPolicy()
    _append_run(store, "abandoned", "goal", transition, status="aborted", verified=False, policy=policy)
    _append_run(store, "blocked-run", "goal", transition, status="blocked", verified=False, policy=policy)
    metrics = CanaryMetrics.from_events(store.load(), policy.digest)
    assert metrics.tasks == 1
    assert metrics.run_ids == ("blocked-run",)


def test_candidate_runs_before_the_bound_time_do_not_count(tmp_path):
    baseline = ExplorationPolicy()
    candidate = ExplorationPolicy(name="candidate", version=2, model_action_limit=218)
    store = _canary_pair_store(tmp_path, baseline, candidate)
    events = store.load()
    future = max(event["recorded_at_ms"] for event in events) + 1
    evidence = CanaryEvidence.from_events(
        events, baseline.digest, candidate.digest, candidate_since_ms=future,
    )
    assert evidence.candidate.tasks == 0
    # Baseline evidence is intentionally unbounded below: the baseline has been live.
    assert evidence.baseline.tasks == 12


def test_promotion_binds_candidate_evidence_to_staging_time(tmp_path, monkeypatch):
    import jev_ultrafast.dream as dream_module

    worlds = _efficiency_world(tmp_path)
    baseline = ExplorationPolicy()
    report = DreamImprover(gate=PromotionGate(min_coverage=1.0)).improve(worlds, baseline)

    class Clock:
        t = 100.0

        @staticmethod
        def time():
            return Clock.t

    monkeypatch.setattr(dream_module, "time", Clock)
    store = ExperienceStore(tmp_path / "temporal.jsonl")
    transition = [{
        "state": "S", "next_state": "D", "selected": {"id": "go", "kind": "click"},
        "candidates": [{"id": "go", "kind": "click", "label": "Go", "goal_overlap": 1}],
        "page_changed": True, "latency_ms": 10, "model_calls": 1, "tokens": 5,
        "stale_or_failure": 0, "risk_events": 0,
    }]
    # Baseline and an identical candidate policy ran before this candidate was staged.
    _append_run(store, "b1", "goal", transition, status="done", verified=True, policy=baseline)
    _append_run(store, "pre-stage", "goal", transition, status="done", verified=True, policy=report.selected)
    Clock.t = 200.0
    registry = PolicyRegistry(tmp_path / "temporal-policy.json")
    registry.stage(report)
    Clock.t = 300.0
    _append_run(store, "post-stage", "goal", transition, status="done", verified=True, policy=report.selected)
    permissive = CanaryGate(
        min_tasks=1, min_baseline_tasks=1, min_task_families=1,
        min_paired_task_families=1, min_pair_coverage=0.0,
    )
    active = registry.promote_from_store(store, gate=permissive)
    assert active["canary"]["baseline"]["tasks"] == 1
    assert active["canary"]["candidate"]["tasks"] == 1
    assert tuple(active["canary"]["candidate"]["run_ids"]) == ("post-stage",)


def test_promotion_rejects_baseline_digest_mismatch(tmp_path):
    worlds = _efficiency_world(tmp_path)
    baseline = ExplorationPolicy()
    report = DreamImprover(gate=PromotionGate(min_coverage=1.0)).improve(worlds, baseline)
    registry = PolicyRegistry(tmp_path / "mismatch-policy.json")
    registry.stage(report)
    store = _canary_pair_store(tmp_path, baseline, report.selected)
    with pytest.raises(ValueError, match="parent digest"):
        registry.promote_from_store(store, baseline_policy_digest=report.selected.digest)


def test_promote_bound_rejects_unbound_evidence_digests(tmp_path):
    worlds = _efficiency_world(tmp_path)
    report = DreamImprover(gate=PromotionGate(min_coverage=1.0)).improve(worlds, ExplorationPolicy())
    registry = PolicyRegistry(tmp_path / "digest-policy.json")
    staged = registry.stage(report)
    permissive = CanaryGate(
        min_tasks=1, min_baseline_tasks=1, min_task_families=0, min_paired_task_families=0,
    )
    metrics = CanaryMetrics(12, 12, task_families=4)
    evidence = CanaryEvidence(
        baseline=metrics, candidate=metrics,
        paired_task_families=4, candidate_wins=0, baseline_wins=0, ties=4,
        event_head_hash="0" * 64, evidence_digest="0" * 64,
        baseline_digest=staged["parent_digest"], candidate_digest="f" * 64,
    )
    with pytest.raises(ValueError, match="staged policy digest"):
        registry._promote_bound(evidence, permissive)
    evidence = CanaryEvidence(
        baseline=metrics, candidate=metrics,
        paired_task_families=4, candidate_wins=0, baseline_wins=0, ties=4,
        event_head_hash="0" * 64, evidence_digest="0" * 64,
        baseline_digest="f" * 64, candidate_digest=staged["digest"],
    )
    with pytest.raises(ValueError, match="parent digest"):
        registry._promote_bound(evidence, permissive)


def _registry_concurrent_worker(path, rounds):
    registry = PolicyRegistry(path)
    for index in range(rounds):
        registry.suspend(f"drift {index}")
        registry.resume()


def test_policy_registry_serializes_concurrent_writers(tmp_path):
    worlds = _efficiency_world(tmp_path)
    report = DreamImprover(gate=PromotionGate(min_coverage=1.0)).improve(worlds, ExplorationPolicy())
    path = str(tmp_path / "locked-policy.json")
    registry = PolicyRegistry(path)
    registry.stage(report)
    permissive = CanaryGate(
        min_tasks=1, min_baseline_tasks=1, min_task_families=0, min_paired_task_families=0,
    )
    registry.promote(
        CanaryMetrics(1, 1), CanaryMetrics(1, 1), gate=permissive, allow_unbound_metrics=True,
    )
    base_revision = registry.load()["revision"]
    context = multiprocessing.get_context("spawn")
    workers = [
        context.Process(target=_registry_concurrent_worker, args=(path, 5))
        for _ in range(3)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(30)
        assert worker.exitcode == 0
    payload = registry.load()
    # Every suspend/resume pair must survive: lost updates would shrink the revision.
    assert payload["revision"] == base_revision + 30
    assert payload["active"]["suspended"] is False


def test_tokenizer_is_consistent_across_scoring_and_trace(tmp_path):
    from jev_ultrafast.model import _candidate_score, _goal_tokens
    from jev_ultrafast.privacy import tokenize

    goal = "Find the Gödel numbering paper"
    goal_tokens = _goal_tokens(goal)
    assert "gödel" in goal_tokens
    action = {"id": "a", "kind": "click", "label": "Gödel numbering", "node": 1}
    overlap = len(goal_tokens & set(tokenize(action["label"])))
    assert overlap == 2
    policy = ExplorationPolicy()
    # Default parameters: policy path and baseline path must score identically.
    assert _candidate_score(action, goal_tokens, 0) == policy.candidate_score(action, goal_tokens, 0)
    store = ExperienceStore(tmp_path / "tokens.jsonl")
    recorder = DreamTraceRecorder(store, goal=goal)
    recorder.start({"fingerprint": "A", "url": "https://example.test"}, policy)
    recorder.transition(
        before={"fingerprint": "A"}, after={"fingerprint": "B"},
        action=action, decision={"latency_ms": 1, "usage": {}}, candidates=[action],
    )
    event = next(item for item in store.load() if item["event"] == "transition")
    assert event["candidates"][0]["goal_overlap"] == overlap


def test_health_gate_can_suspend_drifted_active_policy(tmp_path):
    baseline = ExplorationPolicy()
    worlds = _efficiency_world(tmp_path)
    report = DreamImprover(gate=PromotionGate(min_coverage=1.0)).improve(worlds, baseline)
    registry = PolicyRegistry(tmp_path / "health-policy.json")
    registry.stage(report)
    canary_store = _canary_pair_store(tmp_path, baseline, report.selected)
    registry.promote_from_store(canary_store, baseline_policy_digest=baseline.digest)

    transition = [{
        "state": "S", "next_state": "D", "selected": {"id": "go", "kind": "click"},
        "candidates": [{"id": "go", "kind": "click", "label": "Go", "goal_overlap": 1}],
        "page_changed": True, "latency_ms": 10, "model_calls": 1, "tokens": 5,
        "stale_or_failure": 0, "risk_events": 0,
    }]
    for index in range(10):
        _append_run(
            canary_store, f"drift-{index}", f"drift task {index}", transition,
            status="done" if index < 5 else "blocked", verified=index < 5, policy=report.selected,
        )
    decision = registry.health_from_store(
        canary_store,
        gate=HealthGate(min_tasks=10, max_success_regression=0.1),
        recent_tasks=10,
        suspend_on_fail=True,
    )
    assert not decision.healthy
    assert registry.load()["active"]["suspended"] is True
    assert registry.active_policy().digest == ExplorationPolicy().digest


def test_dream_report_binds_world_pool_split_and_evidence_head(tmp_path):
    store, worlds = _branching_world(tmp_path)
    report = DreamImprover().improve(worlds, ExplorationPolicy(), evidence_head_hash=store.head_hash())
    payload = report.to_dict()
    assert len(payload["world_pool_digest"]) == 64
    assert len(payload["split_manifest_digest"]) == 64
    assert payload["evidence_head_hash"] == store.head_hash()
    assert payload["tcb_versions"]


def test_canary_gate_rejects_live_latency_and_token_regression():
    gate = CanaryGate(
        min_tasks=2,
        min_baseline_tasks=2,
        min_task_families=0,
        min_paired_task_families=0,
        max_latency_regression_ratio=0.25,
        max_token_regression_ratio=0.25,
    )
    baseline = CanaryMetrics(
        tasks=2, verified_successes=2, latency_ms=200, actions=4, tokens=2000,
    )
    slow = CanaryMetrics(
        tasks=2, verified_successes=2, latency_ms=400, actions=4, tokens=2000,
    )
    expensive = CanaryMetrics(
        tasks=2, verified_successes=2, latency_ms=200, actions=4, tokens=4000,
    )
    assert "latency" in gate.assess(baseline, slow).reason
    assert "token" in gate.assess(baseline, expensive).reason


def test_suspended_policy_falls_back_to_baseline_and_can_be_superseded(tmp_path):
    baseline = ExplorationPolicy()
    worlds = _efficiency_world(tmp_path)
    registry = PolicyRegistry(tmp_path / "suspend-supersede.json")
    report = DreamImprover(gate=PromotionGate(min_coverage=1.0)).improve(worlds, baseline)
    registry.stage(report)
    permissive = CanaryGate(
        min_tasks=1,
        min_baseline_tasks=1,
        min_task_families=0,
        min_paired_task_families=0,
    )
    active = registry.promote(
        CanaryMetrics(1, 1), CanaryMetrics(1, 1),
        gate=permissive, allow_unbound_metrics=True,
    )
    registry.suspend("drift")
    assert registry.active_policy().digest == baseline.digest
    replacement = DreamImprover(gate=PromotionGate(min_coverage=1.0)).improve(worlds, baseline)
    staged = registry.stage(replacement)
    assert staged["parent_digest"] == baseline.digest
    assert active["digest"] != baseline.digest


def test_experience_store_refuses_to_extend_corrupt_tail(tmp_path):
    path = tmp_path / "corrupt-tail.jsonl"
    store = ExperienceStore(path)
    store.append({"run_id": "1", "event": "a"})
    payload = json.loads(path.read_text().strip())
    payload["event"] = "tampered"
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="corrupt tail"):
        store.append({"run_id": "2", "event": "b"})


def test_v04_reads_v03_experience_events(tmp_path):
    path = tmp_path / "legacy.jsonl"
    payload = {
        "schema": "jev-dream/1",
        "tcb_version": "jev-ultrafast-tcb/0.3",
        "recorded_at_ms": 1,
        "prev_hash": "0" * 64,
        "run_id": "legacy",
        "event": "legacy_event",
    }
    payload["event_hash"] = ExperienceStore._event_hash(payload)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    loaded = ExperienceStore(path).load()
    assert loaded[0]["schema"] == "jev-dream/1"
    assert loaded[0]["tcb_version"] == "jev-ultrafast-tcb/0.3"
