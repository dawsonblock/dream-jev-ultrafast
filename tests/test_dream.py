import json
import multiprocessing

import pytest

from jev_ultrafast.dream import (
    TCB_VERSION,
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
    mutate_policies,
    split_worlds,
    task_key,
)
from jev_ultrafast.model import candidate_actions
from jev_ultrafast.trace import DreamTraceRecorder


def _append_run(store, run_id, goal, transitions, *, status, verified, policy=None, extra=None):
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


def test_canary_metrics_max_runs_zero_means_none(tmp_path):
    store = ExperienceStore(tmp_path / "canary.jsonl")
    policy = ExplorationPolicy()
    transition = [{
        "state": "S", "next_state": "D", "selected": {"id": "go", "kind": "click"},
        "candidates": [{"id": "go", "kind": "click", "label": "Go", "goal_overlap": 1}],
        "page_changed": True, "latency_ms": 5, "model_calls": 1, "tokens": 10,
        "stale_or_failure": 0, "risk_events": 0,
    }]
    _append_run(store, "r1", "goal", transition, status="done", verified=True, policy=policy)
    _append_run(store, "r2", "goal", transition, status="done", verified=True, policy=policy)
    events = store.load()
    assert CanaryMetrics.from_events(events, policy.digest, max_runs=0).tasks == 0
    assert CanaryMetrics.from_events(events, policy.digest, max_runs=-3).tasks == 0
    assert CanaryMetrics.from_events(events, policy.digest, max_runs=1).tasks == 1


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
    assert event["selected_observed_rank"] == 1
    # No separate offer was made (offered=None): the observed catalogue was the
    # offer, so no offered-rank coordinate is recorded and catalog is "recorded".
    assert event["selected_offered_rank"] is None
    assert event["catalog"] == "recorded"
    assert event["candidate_digest"] == candidate_catalog_digest(event["candidates"])
    world = ReplayWorld.from_events(store.load())[0]
    transition = world.trajectory[0]
    assert transition.candidate_digest == event["candidate_digest"]
    assert transition.selected_observed_rank == 1
    # Replay resolves the offered coordinate for a recorded catalogue to the
    # observed rank — observed and offered are the same catalogue here.
    assert transition.selected_offered_rank == 1


def _canary_pair_store(tmp_path, baseline, candidate, *, candidate_success=True,
                       name="paired-canary.jsonl"):
    store = ExperienceStore(tmp_path / name)
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


def test_canary_pairing_is_namespaced_by_task_family(tmp_path):
    """Audit regression: instance_id reused across families must never pair."""
    baseline = ExplorationPolicy()
    candidate = ExplorationPolicy(name="candidate", version=2, model_action_limit=218)
    store = ExperienceStore(tmp_path / "xfam.jsonl")
    transition = _single_transition()
    _append_run(store, "b-1", "book a flight", transition,
                status="done", verified=True, policy=baseline,
                extra={"instance_id": "inst-1", "task_family": "flights"})
    _append_run(store, "c-1", "book a hotel", transition,
                status="blocked", verified=False, policy=candidate,
                extra={"instance_id": "inst-1", "task_family": "hotels"})
    evidence = CanaryEvidence.from_events(store.load(), baseline.digest, candidate.digest)
    assert evidence.paired_instances == 0
    assert evidence.paired_task_families == 0


def _signed_registry(tmp_path, signer, verify_keys=None, name="signed-policy.json"):
    """Stage and promote through the real bound path under a signing key."""
    worlds = _efficiency_world(tmp_path)
    baseline = ExplorationPolicy()
    report = DreamImprover(gate=PromotionGate(min_coverage=1.0)).improve(worlds, baseline)
    registry = PolicyRegistry(tmp_path / name, signer=signer, verify_keys=verify_keys)
    registry.stage(report)
    store = _canary_pair_store(tmp_path, baseline, report.selected, name=f"canary-{name}.jsonl")
    active = registry.promote_from_store(store, baseline_policy_digest=baseline.digest)
    return registry, report, active


def test_promotion_writes_signed_attestation_and_reader_verifies(tmp_path):
    from jev_ultrafast.signing import EvidenceSigner

    signer = EvidenceSigner(b"a" * 32)
    registry, report, active = _signed_registry(
        tmp_path, signer, verify_keys={signer.key_id})
    att = active["attestation"]
    assert att["kind"] == "promotion_attestation"
    assert att["candidate_digest"] == report.selected.digest
    assert att["parent_digest"] == ExplorationPolicy().digest
    assert att["key_id"] == signer.key_id and att["signature"]
    # A fresh reader trusting the same key accepts the signed record.
    reader = PolicyRegistry(tmp_path / "signed-policy.json", verify_keys={signer.key_id})
    assert reader.active_policy().digest == report.selected.digest


def test_unsigned_active_record_fails_closed_when_keys_configured(tmp_path):
    from jev_ultrafast.signing import EvidenceSigner

    signer = EvidenceSigner(b"b" * 32)
    _signed_registry(tmp_path, signer)
    path = tmp_path / "signed-policy.json"
    payload = json.loads(path.read_text())
    payload["active"].pop("attestation")  # direct local write: strip authority
    path.write_text(json.dumps(payload))
    reader = PolicyRegistry(path, verify_keys={signer.key_id})
    # The tamper is caught by the chained state head before attestation checks
    # even run — either layer failing closed is the required behavior.
    with pytest.raises(ValueError, match="state digest|attestation"):
        reader.active_policy()


def test_forged_and_wrong_key_attestations_fail(tmp_path):
    from jev_ultrafast.signing import EvidenceSigner

    signer = EvidenceSigner(b"c" * 32)
    _signed_registry(tmp_path, signer, verify_keys={signer.key_id})
    path = tmp_path / "signed-policy.json"

    payload = json.loads(path.read_text())
    payload["active"]["attestation"]["signature"] = "0" * 128  # forged
    path.write_text(json.dumps(payload))
    # State head catches the write first; the attestation verifier is the
    # deeper layer for a payload whose head was legitimately produced.
    with pytest.raises(ValueError, match="state digest|attestation"):
        PolicyRegistry(path, verify_keys={signer.key_id}).load()

    attacker = EvidenceSigner(b"d" * 32)
    _signed_registry(tmp_path, signer, verify_keys={signer.key_id}, name="k.json")
    payload = json.loads((tmp_path / "k.json").read_text())
    material = {k: v for k, v in payload["active"]["attestation"].items()
                if k not in {"digest", "signature", "key_id"}}
    attacker_registry = PolicyRegistry(tmp_path / "attacker.json", signer=attacker)
    payload["active"]["attestation"] = attacker_registry._sign_attestation(material)
    (tmp_path / "k.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unexpected key|state digest"):
        PolicyRegistry(tmp_path / "k.json", verify_keys={signer.key_id}).load()


def test_promotion_attestation_is_domain_separated_from_events(tmp_path):
    """An attestation signature must not verify under the evidence-event frame."""
    from jev_ultrafast.signing import EvidenceSigner, verify_signature

    signer = EvidenceSigner(b"e" * 32)
    _, _, active = _signed_registry(tmp_path, signer)
    att = active["attestation"]
    assert not verify_signature(att["key_id"], att["digest"], att["signature"])


def test_rollback_preserves_signed_authority(tmp_path):
    from jev_ultrafast.signing import EvidenceSigner

    signer = EvidenceSigner(b"f" * 32)
    registry, report, _ = _signed_registry(tmp_path, signer, verify_keys={signer.key_id})
    # Second promotion: improve on top of the newly active policy, using a
    # different world pool so a distinct winner exists.
    worlds2 = _overlapping_distractor_world(tmp_path)
    second = DreamImprover(gate=PromotionGate(min_coverage=1.0)).improve(
        worlds2, report.selected)
    if not second.promotion.approved or second.selected.digest == report.selected.digest:
        pytest.skip("deterministic search did not find a second distinct winner")
    registry.stage(second)
    store2 = _canary_pair_store(tmp_path, report.selected, second.selected,
                                name="canary-second.jsonl")
    registry.promote_from_store(store2, baseline_policy_digest=report.selected.digest)
    target = registry.rollback()
    assert target["digest"] == report.selected.digest
    assert target["attestation"]["candidate_digest"] == report.selected.digest
    assert target["rollback_attestation"]["kind"] == "rollback_attestation"
    assert target["rollback_attestation"]["rollback_from_digest"] == second.selected.digest
    # Reader still verifies both attestations after reload.
    reader = PolicyRegistry(tmp_path / "signed-policy.json", verify_keys={signer.key_id})
    assert reader.active_policy().digest == report.selected.digest


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


def _overlapping_distractor_world(tmp_path):
    """Distractors share one goal token, so ``min_goal_overlap=1`` keeps them and
    a promoted policy still leaves contraction headroom for a second winner."""
    store = ExperienceStore(tmp_path / "overlap.jsonl")
    goal = "find destination"
    candidates = [
        {"id": f"c{i}", "kind": "click", "label": f"Find control {i}", "goal_overlap": 1}
        for i in range(229)
    ]
    candidates.insert(0, {"id": "selected", "kind": "fill", "label": "Destination", "goal_overlap": 5})
    _append_run(store, "success", goal, [{
        "state": "S", "next_state": "DONE", "selected": {"id": "selected", "kind": "fill"},
        "candidates": candidates, "page_changed": True, "latency_ms": 50, "model_calls": 1,
        "tokens": 1000, "stale_or_failure": 0, "risk_events": 0,
    }], status="done", verified=True)
    return ReplayWorld.from_events(store.load())


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

    second_worlds = _overlapping_distractor_world(tmp_path)
    second_report = DreamImprover(gate=PromotionGate(min_coverage=1.0)).improve(
        second_worlds, registry.active_policy()
    )
    assert second_report.promotion.approved
    assert second_report.selected.digest != first["digest"]
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


def test_experience_store_strict_mode_refuses_mid_chain_corruption(tmp_path):
    path = tmp_path / "strict.jsonl"
    store = ExperienceStore(path)
    store.append({"run_id": "1", "event": "a"})
    store.append({"run_id": "1", "event": "b"})
    store.append({"run_id": "1", "event": "c"})
    lines = path.read_text().splitlines()
    payload = json.loads(lines[1])
    payload["event"] = "tampered"
    lines[1] = json.dumps(payload)
    path.write_text("\n".join(lines) + "\n")
    # Tail-only appends remain O(1) and tolerate a stale mid-chain corruption;
    # strict mode refuses to grow a store that load() would reject.
    ExperienceStore(path).append({"run_id": "1", "event": "d"})
    with pytest.raises(ValueError, match="integrity failure"):
        ExperienceStore(path, strict=True).append({"run_id": "1", "event": "e"})


def test_canary_gate_compares_risk_rate_as_well_as_count():
    gate = CanaryGate(
        min_tasks=1, min_baseline_tasks=1, min_task_families=0,
        min_paired_task_families=0, max_extra_risk_events=5,
    )
    baseline = CanaryMetrics(tasks=12, verified_successes=12)
    # Fewer raw events than the allowance, but a worse rate per task.
    candidate = CanaryMetrics(tasks=20, verified_successes=20, risk_events=4)
    decision = gate.assess(baseline, candidate)
    assert not decision.approved
    assert "risk rate" in decision.reason
    strict = CanaryGate(
        min_tasks=1, min_baseline_tasks=1, min_task_families=0, min_paired_task_families=0,
    )
    assert not strict.assess(
        baseline, CanaryMetrics(tasks=12, verified_successes=12, risk_events=1)
    ).approved


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


def _write_legacy_tcb_run(path, run_id, tcb_version, goal, *, status="done", verified=True):
    """Append a complete hash-valid run stamped with an older TCB version."""
    key = task_key(goal)
    policy = ExplorationPolicy()
    events = [
        {"run_id": run_id, "task_key": key, "sequence": 1, "event": "run_started",
         "goal": goal, "state": "S", "policy": policy.to_dict(), "policy_digest": policy.digest},
        {"run_id": run_id, "task_key": key, "sequence": 2, "event": "transition",
         "state": "S", "next_state": "D", "selected": {"id": "go", "kind": "click"},
         "candidates": [{"id": "go", "kind": "click", "label": "Go", "goal_overlap": 1}],
         "page_changed": True, "latency_ms": 10, "model_calls": 1, "tokens": 5,
         "stale_or_failure": 0, "risk_events": 0},
        {"run_id": run_id, "task_key": key, "sequence": 3, "event": "run_finished",
         "status": status, "verified": verified},
    ]
    lines = path.read_text().splitlines() if path.exists() else []
    prev = json.loads(lines[-1])["event_hash"] if lines else "0" * 64
    for event in events:
        payload = {
            "schema": "jev-dream/2",
            "tcb_version": tcb_version,
            "recorded_at_ms": 1,
            "prev_hash": prev,
            **event,
        }
        payload["event_hash"] = ExperienceStore._event_hash(payload)
        lines.append(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        prev = payload["event_hash"]
    path.write_text("\n".join(lines) + "\n")


def test_new_events_use_current_tcb_and_read_legacy_tail(tmp_path):
    path = tmp_path / "mixed-tcb.jsonl"
    _write_legacy_tcb_run(path, "legacy", "jev-ultrafast-tcb/0.4", "goal")
    store = ExperienceStore(path)
    store.append({"run_id": "new", "event": "fresh"})
    events = store.load()
    assert events[0]["tcb_version"] == "jev-ultrafast-tcb/0.4"
    assert events[-1]["tcb_version"] == TCB_VERSION


def test_improve_rejects_replay_pools_mixing_tcb_generations(tmp_path):
    path = tmp_path / "mixed-pool.jsonl"
    _write_legacy_tcb_run(path, "old", "jev-ultrafast-tcb/0.4", "goal")
    transition = [{
        "state": "S", "next_state": "D", "selected": {"id": "go", "kind": "click"},
        "candidates": [{"id": "go", "kind": "click", "label": "Go", "goal_overlap": 1}],
        "page_changed": True, "latency_ms": 10, "model_calls": 1, "tokens": 5,
        "stale_or_failure": 0, "risk_events": 0,
    }]
    store = ExperienceStore(path)
    _append_run(store, "new", "goal", transition, status="done", verified=True)
    worlds = ReplayWorld.from_events(store.load())
    assert {world.tcb_version for world in worlds} == {"jev-ultrafast-tcb/0.4", TCB_VERSION}
    with pytest.raises(ValueError, match="TCB"):
        DreamImprover().improve(worlds, ExplorationPolicy())


def test_health_gate_marks_insufficient_coverage():
    gate = HealthGate(min_tasks=20)
    reference = CanaryMetrics(tasks=20, verified_successes=20)
    quiet = gate.assess(reference, CanaryMetrics(tasks=3, verified_successes=3))
    assert quiet.healthy and quiet.sufficient is False
    full = gate.assess(reference, CanaryMetrics(tasks=20, verified_successes=20))
    assert full.healthy and full.sufficient is True


def test_behavior_digest_separates_identity_from_behavior():
    a = ExplorationPolicy(name="a", version=1)
    b = ExplorationPolicy(name="b", version=3)
    assert a.digest != b.digest
    assert a.behavior_digest == b.behavior_digest
    changed = ExplorationPolicy(name="a", version=1, click_quota=100)
    assert changed.behavior_digest != a.behavior_digest


def test_mutations_are_bidirectional_and_behavior_deduped():
    base = ExplorationPolicy(name="mid", model_action_limit=100, max_actions=30, no_progress_window=4)
    candidates = mutate_policies(base)
    assert len(candidates) > 20
    assert len({c.behavior_digest for c in candidates}) == len(candidates)
    assert all(c.behavior_digest != base.behavior_digest for c in candidates)
    assert all(c.name != base.name and c.version == base.version + 1 for c in candidates)
    for field in ("model_action_limit", "click_quota", "fill_quota", "select_quota",
                  "click_bonus", "fill_bonus", "select_bonus", "max_actions",
                  "no_progress_window", "goal_overlap_weight", "order_penalty"):
        values = {getattr(c, field) for c in candidates}
        assert any(v < getattr(base, field) for v in values), field
        assert any(v > getattr(base, field) for v in values), field


def test_saturated_policy_mutations_stay_bounded():
    base = ExplorationPolicy(
        model_action_limit=250, click_quota=250, fill_quota=250,
        select_quota=250, max_actions=120, no_progress_window=10,
    )
    for candidate in mutate_policies(base):
        ExplorationPolicy.from_dict(candidate.to_dict())  # stays inside envelopes
        assert candidate.model_action_limit <= 250
        assert candidate.max_actions <= 120
        assert candidate.no_progress_window <= 10


def _single_transition():
    return [{
        "state": "S", "next_state": "D", "selected": {"id": "go", "kind": "click"},
        "candidates": [{"id": "go", "kind": "click", "label": "Go", "goal_overlap": 1}],
        "page_changed": True, "latency_ms": 10, "model_calls": 1, "tokens": 5,
        "stale_or_failure": 0, "risk_events": 0,
    }]


def test_trace_records_observed_and_offered_catalogues(tmp_path):
    store = ExperienceStore(tmp_path / "dual.jsonl")
    policy = ExplorationPolicy(model_action_limit=16, click_quota=5)
    recorder = DreamTraceRecorder(
        store, goal="Open target", task_family="Family-A", instance_id="inst-1",
    )
    page = {"fingerprint": "A", "url": "https://example.test"}
    recorder.start(page, policy)
    actions = [{"id": f"c{i}", "kind": "click", "node": i, "label": f"Control {i}"} for i in range(20)]
    actions.append({"id": "target", "kind": "fill", "node": 99, "label": "Target field"})
    offered, _ = candidate_actions(actions, "Open target", exploration_policy=policy)
    recorder.transition(
        before=page,
        after={"fingerprint": "B", "url": "https://example.test/done"},
        action=actions[-1],
        decision={"latency_ms": 1, "usage": {}},
        candidates=actions,
        offered=offered,
    )
    recorder.finish(status="done", verified=True)
    events = store.load()
    start = next(e for e in events if e["event"] == "run_started")
    assert start["task_family"] == "Family-A" and start["instance_id"] == "inst-1"
    event = next(e for e in events if e["event"] == "transition")
    assert event["catalog"] == "observed"
    assert len(event["candidates"]) == 21
    assert event["offered_count"] == len(offered) < len(event["candidates"])
    # The two rank coordinates are recorded separately: observed index 20 vs
    # the action's position inside the offered catalogue.
    assert event["selected_observed_rank"] == 20
    assert event["selected_offered_rank"] == [
        a.get("id") for a in offered
    ].index("target")
    worlds = ReplayWorld.from_events(events)
    assert worlds[0].family_key == "family-a"
    assert worlds[0].trajectory[0].offered_digest == event["offered_digest"]
    assert worlds[0].trajectory[0].catalog == "observed"
    assert worlds[0].trajectory[0].selected_observed_rank == 20
    assert worlds[0].trajectory[0].selected_offered_rank == event["selected_offered_rank"]


def test_trace_records_behavior_propensities(tmp_path):
    """Off-policy groundwork: the recorded behavior-policy probabilities are the
    denominator any future counterfactual estimate is weighted by."""
    store = ExperienceStore(tmp_path / "propensity.jsonl")
    policy = ExplorationPolicy(min_goal_overlap=1)
    recorder = DreamTraceRecorder(store, goal="Open target")
    page = {"fingerprint": "A", "url": "https://example.test"}
    recorder.start(page, policy)
    actions = [
        {"id": "f0", "kind": "click", "node": 1, "label": "Filler", "goal_overlap": 0},
        {"id": "target", "kind": "fill", "node": 2, "label": "Target field"},
    ]
    offered = [actions[1]]  # what the model actually saw
    decision = {
        "latency_ms": 1, "usage": {}, "confidence": 0.9,
        "operation": "TYPE_TEXT", "target": "1",
        "probabilities": {"target": 0.8, "f0": 0.2},
        "operation_probabilities": {"TYPE_TEXT": 0.7, "CLICK": 0.3},
        "target_probabilities": {"1": 0.8},
    }
    recorder.transition(
        before=page, after={"fingerprint": "B", "url": "https://example.test"},
        action=actions[1], decision=decision, candidates=actions, offered=offered,
    )
    recorder.finish(status="done", verified=True)
    event = next(e for e in store.load() if e["event"] == "transition")
    assert event["selected_observed_rank"] == 1
    assert event["selected_offered_rank"] == 0
    assert event["selected_propensity"] == 0.8
    assert event["operation_probability"] == 0.7
    assert event["target_probability"] == 0.8
    assert event["decision_confidence"] == 0.9
    assert event["action_probabilities"] == {"target": 0.8, "f0": 0.2}
    transition = ReplayWorld.from_events(store.load())[0].trajectory[0]
    assert transition.selected_observed_rank == 1
    assert transition.selected_offered_rank == 0
    assert transition.selected_propensity == 0.8


def test_replay_world_rejects_inconsistent_offered_metadata(tmp_path):
    """The audit's missing checks: offered_count and the recorded offered rank
    must both agree with the catalogue the recorded policy re-derives."""
    transition = _single_transition()
    transition[0]["catalog"] = "observed"
    transition[0]["offered_count"] = 2  # "go" alone re-derives to 1
    _append_run(
        ExperienceStore(tmp_path / "count.jsonl"), "run", "goal", transition,
        status="done", verified=True,
    )
    with pytest.raises(ValueError, match="offered count mismatch"):
        ReplayWorld.from_events(ExperienceStore(tmp_path / "count.jsonl").load())

    transition = _single_transition()
    transition[0]["catalog"] = "observed"
    transition[0]["offered_count"] = 1
    transition[0]["selected_offered_rank"] = 3  # "go" sits at rank 0
    _append_run(
        ExperienceStore(tmp_path / "rank.jsonl"), "run", "goal", transition,
        status="done", verified=True,
    )
    with pytest.raises(ValueError, match="selected offered-rank mismatch"):
        ReplayWorld.from_events(ExperienceStore(tmp_path / "rank.jsonl").load())


def test_offered_digest_must_match_recorded_policy(tmp_path):
    store = ExperienceStore(tmp_path / "offered.jsonl")
    transition = _single_transition()
    transition[0]["catalog"] = "observed"
    transition[0]["offered_digest"] = "0" * 64
    transition[0]["offered_count"] = 1
    _append_run(store, "run", "goal", transition, status="done", verified=True)
    with pytest.raises(ValueError, match="offered catalogue digest mismatch"):
        ReplayWorld.from_events(store.load())


def test_task_family_keeps_paraphrased_goals_in_one_partition(tmp_path):
    store = ExperienceStore(tmp_path / "fam.jsonl")
    transition = _single_transition()
    _append_run(store, "r1", "Find flights Zurich to London", transition,
                status="done", verified=True, extra={"task_family": "flights"})
    _append_run(store, "r2", "Search flights from Zurich to London", transition,
                status="done", verified=True, extra={"task_family": "flights"})
    _append_run(store, "r3", "Book a hotel in Paris", transition,
                status="done", verified=True, extra={"task_family": "hotels"})
    worlds = ReplayWorld.from_events(store.load())
    by_run = {world.key: world for world in worlds}
    assert by_run["r1"].task_key != by_run["r2"].task_key
    assert by_run["r1"].family_key == by_run["r2"].family_key == "flights"
    splits = split_worlds(worlds)
    assignment = {w.key: name for name, part in splits.items() for w in part}
    assert assignment["r1"] == assignment["r2"]
    assert assignment["r3"] != assignment["r1"]


def test_canary_evidence_pairs_by_instance_and_reports_sign_test(tmp_path):
    baseline = ExplorationPolicy()
    candidate = ExplorationPolicy(name="candidate", version=2, model_action_limit=218)
    store = ExperienceStore(tmp_path / "inst.jsonl")
    transition = _single_transition()
    # Equal aggregate success, split differently across instances: the candidate
    # wins inst 6-7 but loses inst 4-5, so pair outcomes are symmetric.
    baseline_wins = set(range(6))
    candidate_wins = set(range(4)) | {6, 7}
    for inst in range(8):
        meta = {"instance_id": f"inst-{inst}", "task_family": "fam"}
        _append_run(store, f"b{inst}", f"goal {inst}", transition,
                    status="done" if inst in baseline_wins else "blocked",
                    verified=inst in baseline_wins, policy=baseline, extra=meta)
        _append_run(store, f"c{inst}", f"goal {inst}", transition,
                    status="done" if inst in candidate_wins else "blocked",
                    verified=inst in candidate_wins, policy=candidate, extra=meta)
    evidence = CanaryEvidence.from_events(store.load(), baseline.digest, candidate.digest)
    assert evidence.paired_instances == 8
    assert evidence.paired_task_families == 1
    assert evidence.candidate.instance_ids == tuple(f"inst-{i}" for i in range(8))
    assert evidence.baseline_wins == 2 and evidence.candidate_wins == 2 and evidence.ties == 4
    assert evidence.sign_test_p_value == 1.0
    gate = CanaryGate(
        min_tasks=1, min_baseline_tasks=1, min_task_families=1,
        min_paired_task_families=1, min_pair_coverage=0.0, max_pair_sign_p=0.2,
    )
    assert "sign test" in gate.assess(evidence.baseline, evidence.candidate, evidence=evidence).reason


def test_baseline_window_bounds_canary_evidence(tmp_path):
    baseline = ExplorationPolicy()
    candidate = ExplorationPolicy(name="candidate", version=2, model_action_limit=218)
    store = _canary_pair_store(tmp_path, baseline, candidate)
    events = store.load()
    future = max(event["recorded_at_ms"] for event in events) + 1
    evidence = CanaryEvidence.from_events(
        events, baseline.digest, candidate.digest, baseline_since_ms=future,
    )
    assert evidence.baseline.tasks == 0
    assert evidence.candidate.tasks == 12


# --- registry state head and full attestation binding (v0.6.1) --------------

def test_registry_state_head_is_chained_and_tamper_evident(tmp_path):
    """Every write chains to the head it replaces; any post-write field edit —
    including mutable ones like ``suspended`` — invalidates the digest."""
    worlds = _efficiency_world(tmp_path)
    path = tmp_path / "chain.json"
    registry = PolicyRegistry(path)
    report = DreamImprover(gate=PromotionGate(min_coverage=1.0)).improve(
        worlds, ExplorationPolicy())
    registry.stage(report)
    first = json.loads(path.read_text())
    assert first["state_digest"]
    assert first["prev_state_digest"] == "0" * 64
    permissive = CanaryGate(
        min_tasks=1, min_baseline_tasks=1, min_task_families=0, min_paired_task_families=0)
    registry.promote(
        CanaryMetrics(1, 1), CanaryMetrics(1, 1), gate=permissive, allow_unbound_metrics=True)
    second = json.loads(path.read_text())
    assert second["prev_state_digest"] == first["state_digest"]
    registry.suspend("drift")
    third = json.loads(path.read_text())
    assert third["prev_state_digest"] == second["state_digest"]
    # Post-write tamper: flip a mutable field without resealing the head.
    third["active"]["suspended"] = False
    path.write_text(json.dumps(third))
    with pytest.raises(ValueError, match="state digest"):
        PolicyRegistry(path).load()


def test_promotion_attestation_must_bind_the_whole_record(tmp_path):
    """The audit's tamper reproduction: a valid signature must not float over a
    record whose parent, canary evidence, or promotion fields were edited
    after signing."""
    from jev_ultrafast.signing import EvidenceSigner

    signer = EvidenceSigner(b"t" * 32)
    registry, report, active = _signed_registry(
        tmp_path, signer, verify_keys={signer.key_id})
    att = active["attestation"]
    for field, value in (
        ("parent_digest", "tampered-parent"),
        ("promoted_at_ms", 1),
        ("promotion_revision", 99),
    ):
        tampered = {**active, field: value}
        with pytest.raises(ValueError, match="not bound"):
            registry._check_attestation(att, tampered, "active")
    # Canary evidence fields — including the health-check reference metrics
    # health_from_store() trusts — are bound through canary_digest.
    tampered = {**active, "canary": {**active["canary"], "evidence_digest": "x" * 64}}
    with pytest.raises(ValueError, match="not bound"):
        registry._check_attestation(att, tampered, "active")
    tampered = {
        **active,
        "canary": {
            **active["canary"],
            "candidate": {**active["canary"]["candidate"], "tasks": 999},
        },
    }
    with pytest.raises(ValueError, match="not bound"):
        registry._check_attestation(att, tampered, "active")
    # The untouched record still verifies.
    registry._check_attestation(att, active, "active")


def test_registry_state_transitions_are_signed_and_chained(tmp_path):
    from jev_ultrafast.signing import EvidenceSigner

    signer = EvidenceSigner(b"s" * 32)
    registry, report, active = _signed_registry(
        tmp_path, signer, verify_keys={signer.key_id})
    path = tmp_path / "signed-policy.json"
    promoted = json.loads(path.read_text())
    assert promoted["state_signature"] and promoted["state_key_id"] == signer.key_id

    suspended = registry.suspend("drift check")
    assert suspended["suspend_attestation"]["kind"] == "suspend_attestation"
    # The transition attestation chains to the head it replaced.
    assert suspended["suspend_attestation"]["prev_state_head"] == promoted["state_digest"]
    assert suspended["suspend_attestation"]["registry_revision"] == suspended["suspend_revision"]
    state = json.loads(path.read_text())
    assert state["prev_state_digest"] == promoted["state_digest"]

    resumed = registry.resume()
    assert resumed["resume_attestation"]["kind"] == "resume_attestation"
    assert resumed["resume_attestation"]["prev_state_head"] == state["state_digest"]

    # A fresh reader trusting the key verifies every transition attestation on
    # the record — the signatures are bound, not decorative.
    reader = PolicyRegistry(path, verify_keys={signer.key_id})
    record = reader.load()["active"]
    for key in ("attestation", "suspend_attestation", "resume_attestation"):
        reader._check_attestation(record[key], record, f"active {key}")


def test_signed_registry_state_fails_closed_without_keys(tmp_path, monkeypatch):
    from jev_ultrafast.signing import EvidenceSigner

    for var in (
        "JEV_EVIDENCE_SIGNING_KEY", "JEV_PROMOTION_SIGNING_KEY",
        "JEV_EVIDENCE_VERIFY_KEY", "JEV_EVIDENCE_VERIFY_KEYS",
        "JEV_PROMOTION_VERIFY_KEYS",
    ):
        monkeypatch.delenv(var, raising=False)
    signer = EvidenceSigner(b"k" * 32)
    _signed_registry(tmp_path, signer)
    # A reader with no configured authority cannot trust a signed state head.
    with pytest.raises(ValueError, match="no verification key"):
        PolicyRegistry(tmp_path / "signed-policy.json").load()


def test_unsigned_registry_state_fails_closed_with_keys(tmp_path):
    from jev_ultrafast.signing import EvidenceSigner

    worlds = _efficiency_world(tmp_path)
    path = tmp_path / "unsigned.json"
    registry = PolicyRegistry(path)
    report = DreamImprover(gate=PromotionGate(min_coverage=1.0)).improve(
        worlds, ExplorationPolicy())
    registry.stage(report)
    permissive = CanaryGate(
        min_tasks=1, min_baseline_tasks=1, min_task_families=0, min_paired_task_families=0)
    registry.promote(
        CanaryMetrics(1, 1), CanaryMetrics(1, 1), gate=permissive, allow_unbound_metrics=True)
    assert "state_signature" not in json.loads(path.read_text())
    signer = EvidenceSigner(b"u" * 32)
    with pytest.raises(ValueError, match="Unsigned policy registry"):
        PolicyRegistry(path, verify_keys={signer.key_id}).load()


def test_rollback_attestation_chains_to_the_replaced_head(tmp_path):
    from jev_ultrafast.signing import EvidenceSigner

    signer = EvidenceSigner(b"r" * 32)
    registry, report, _ = _signed_registry(
        tmp_path, signer, verify_keys={signer.key_id})
    worlds2 = _overlapping_distractor_world(tmp_path)
    second = DreamImprover(gate=PromotionGate(min_coverage=1.0)).improve(
        worlds2, report.selected)
    if not second.promotion.approved or second.selected.digest == report.selected.digest:
        pytest.skip("deterministic search did not find a second distinct winner")
    registry.stage(second)
    store2 = _canary_pair_store(
        tmp_path, report.selected, second.selected, name="canary-chain.jsonl")
    registry.promote_from_store(store2, baseline_policy_digest=report.selected.digest)
    before = json.loads((tmp_path / "signed-policy.json").read_text())
    target = registry.rollback()
    att = target["rollback_attestation"]
    assert att["prev_state_head"] == before["state_digest"]
    assert att["registry_revision"] == target["rollback_revision"]
    # The record's own promotion attestation still binds its original revision.
    assert target["attestation"]["registry_revision"] == target["promotion_revision"]
    reader = PolicyRegistry(tmp_path / "signed-policy.json", verify_keys={signer.key_id})
    assert reader.active_policy().digest == report.selected.digest


def test_registry_state_head_cannot_be_stripped_to_legacy(tmp_path):
    """Removing every state-head field must not downgrade a jev-dream/4 file
    to legacy format: the schema version itself promises the head exists."""
    from jev_ultrafast.signing import EvidenceSigner

    signer = EvidenceSigner(b"x" * 32)
    _signed_registry(tmp_path, signer, verify_keys={signer.key_id})
    path = tmp_path / "signed-policy.json"
    payload = json.loads(path.read_text())
    for key in ("state_digest", "state_signature", "state_key_id", "prev_state_digest"):
        payload.pop(key, None)
    path.write_text(json.dumps(payload))
    # Fails closed with keys, with only a signer, and with no keys at all —
    # a current-schema file without its head is tampered, not legacy.
    with pytest.raises(ValueError, match="state head"):
        PolicyRegistry(path, verify_keys={signer.key_id}).load()
    with pytest.raises(ValueError, match="state head"):
        PolicyRegistry(path, signer=signer).load()
    with pytest.raises(ValueError, match="state head"):
        PolicyRegistry(path).load()


def test_registry_schema_downgrade_cannot_launder_a_stripped_head(tmp_path):
    """Stripping the head AND flipping schema to jev-dream/3 must still fail
    closed under configured trust keys — otherwise the downgrade un-binds
    mutable fields (``suspended``) under a still-valid attestation."""
    from jev_ultrafast.signing import EvidenceSigner

    signer = EvidenceSigner(b"z" * 32)
    registry, _report, _active = _signed_registry(
        tmp_path, signer, verify_keys={signer.key_id})
    registry.suspend("drift detected")
    path = tmp_path / "signed-policy.json"
    payload = json.loads(path.read_text())
    assert payload["active"]["suspended"] is True
    for key in ("state_digest", "state_signature", "state_key_id", "prev_state_digest"):
        payload.pop(key, None)
    payload["schema"] = "jev-dream/3"  # the field `stripped` keys on
    payload["active"]["suspended"] = False  # the edit the attacker wants
    payload["active"].pop("suspend_attestation", None)
    payload["active"].pop("suspended_at_ms", None)
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="state head"):
        PolicyRegistry(path, verify_keys={signer.key_id}).load()
    with pytest.raises(ValueError, match="state head"):
        PolicyRegistry(path, signer=signer).load()
    # With no trust keys at all the file is unsigned data either way — legacy
    # compat stays honest, the downgrade gains the attacker nothing.
    assert PolicyRegistry(path).load()["active"]["suspended"] is False


def test_registry_orphaned_state_fields_are_rejected(tmp_path):
    """A partial strip that leaves signature fields behind is also evidence
    of tampering, even on a legacy-schema payload."""
    worlds = _efficiency_world(tmp_path)
    registry = PolicyRegistry(tmp_path / "orphan.json")
    report = DreamImprover(gate=PromotionGate(min_coverage=1.0)).improve(
        worlds, ExplorationPolicy())
    registry.stage(report)
    path = tmp_path / "orphan.json"
    payload = json.loads(path.read_text())
    payload["state_digest"] = None and payload.pop("state_digest")
    payload["state_signature"] = "00" * 64  # orphan field without the head
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="state head"):
        PolicyRegistry(path).load()


def test_registry_signature_strip_fails_under_a_signer(tmp_path):
    """A configured signer only ever produces signed writes, so a missing
    state signature is a strip — even when no verification keys are set."""
    from jev_ultrafast.signing import EvidenceSigner

    signer = EvidenceSigner(b"y" * 32)
    _signed_registry(tmp_path, signer)
    path = tmp_path / "signed-policy.json"
    payload = json.loads(path.read_text())
    payload.pop("state_signature")
    payload.pop("state_key_id")
    path.write_text(json.dumps(payload))
    # The digest still covers everything else, so only the unsigned-under-
    # signer check can catch this.
    with pytest.raises(ValueError, match="Unsigned policy registry state"):
        PolicyRegistry(path, signer=signer).load()


def test_registry_legacy_schema_without_state_head_still_loads(tmp_path):
    """Pre-0.7 files predate the state head entirely; compat mode is honest
    because their own schema declares it."""
    path = tmp_path / "legacy.json"
    legacy = {
        "schema": "jev-dream/3",
        "tcb_version": "jev-ultrafast-tcb/0.8",
        "revision": 3,
        "active": None,
        "staged": None,
        "history": [],
    }
    path.write_text(json.dumps(legacy))
    payload = PolicyRegistry(path).load()
    assert payload["revision"] == 3
    assert payload["active"] is None


def test_experiment_runs_are_excluded_from_canary_metrics(tmp_path):
    """A run carrying an assigned experiment arm is not a clean
    policy-performance sample; it must not qualify or disqualify a policy."""
    baseline = ExplorationPolicy()
    store = ExperienceStore(tmp_path / "exp.jsonl")
    transition = _single_transition()
    _append_run(store, "clean-1", "book a flight", transition,
                status="done", verified=True, policy=baseline)
    experiment_transition = {
        **_single_transition()[0],
        "experiment": {
            "arm": "candidate",
            "assignment_probability": 0.25,
            "proposal_id": "b",
            "proposal_kind": "click",
            "model_choice_id": "a",
            "model_choice_kind": "click",
        },
    }
    _append_run(store, "trial-1", "book a flight", [experiment_transition],
                status="done", verified=True, policy=baseline)
    metrics = CanaryMetrics.from_events(store.load(), baseline.digest)
    assert metrics.tasks == 1
    assert metrics.run_ids == ("clean-1",)
