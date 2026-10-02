"""Offline contracts for a dynamic operation/target policy. No paid APIs."""

import json
import time
from copy import deepcopy
from unittest.mock import Mock

import pytest

from jev_ultrafast import agent as loop
from jev_ultrafast import model
from jev_ultrafast.browser import IndeterminateMutation, StalePage, browser_operation, fingerprint


def page():
    state = {
        "url": "https://example.test/",
        "title": "Search",
        "text": "Search",
        "scroll": {"y": 0},
        "actions": [
            {"id": "e1", "kind": "fill", "label": "Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e2", "kind": "click", "label": "Open Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e3", "kind": "click", "label": "Go", "role": "button", "value": "", "node": 20},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ],
    }
    state["fingerprint"] = fingerprint(state)
    return state


def _candidate_rng():
    """Deterministic roll below every supported rate → the candidate arm."""
    rng = Mock()
    rng.random = Mock(return_value=0.0)
    return rng


def choice(ids, selected):
    return {"choice": selected, "confidence": 1.0, "probabilities": {i: float(i == selected) for i in ids}}


def decision(action="e1"):
    return {
        "choice": action,
        "operation": "TYPE_TEXT",
        "target": "1",
        "confidence": 1.0,
        "probabilities": {action: 1.0},
        "latency_ms": 10,
        "usage": {},
    }


@pytest.mark.parametrize("mutation", ["unknown", "nan", "missing", "negative", "non_max", "confidence"])
def test_invalid_choice_is_rejected(mutation):
    a = choice(["a", "b"], "a")
    if mutation == "unknown":
        a["choice"] = "invented"
    elif mutation == "nan":
        a["probabilities"]["a"] = float("nan")
    elif mutation == "missing":
        del a["probabilities"]["b"]
    elif mutation == "negative":
        a["probabilities"]["b"] = -1
    elif mutation == "non_max":
        a["choice"] = "b"
    else:
        a["confidence"] = 5
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        model.validate_choice(a, {"a", "b"})


def test_one_index_per_node_with_operation_specific_targets():
    elements, targets, controls = model.action_space(page()["actions"])
    assert len(elements) == 2
    assert elements[0]["operations"] == ["TYPE_TEXT", "CLICK"]
    assert targets["TYPE_TEXT"]["1"]["id"] == "e1"
    assert targets["CLICK"]["1"]["id"] == "e2"
    assert targets["CLICK"]["2"]["id"] == "e3"
    assert "WAIT" in controls


def test_all_heads_are_one_request_and_only_matching_head_executes(monkeypatch):
    calls = []

    def post(_url, _key, body):
        calls.append(body)
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "TYPE_TEXT"),
                "type_text_target": choice(["1"], "1"),
                "click_target": {"choice": "invented"},
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(page(), "Find a book", [])
    assert len(calls) == 1
    assert d["operation"] == "TYPE_TEXT" and d["target"] == "1" and d["choice"] == "e1"
    assert set(calls[0]["questions"]) == {"operation", "click_target", "type_text_target"}


def test_click_cannot_consume_a_text_target(monkeypatch):
    def post(_url, _key, body):
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "CLICK"),
                "type_text_target": choice(["1"], "1"),
                "click_target": choice(["1", "2", "999"], "999"),
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        model.choose(page(), "Find a book", [])


def test_target_head_receives_control_state_and_full_next_step_rules(monkeypatch):
    p = page()
    p["actions"].insert(0, {
        "id": "toggle", "kind": "click", "label": "Free cancellation", "node": 30,
        "role": "checkbox", "checked": "true", "selected": False,
    })

    def post(_url, _key, body):
        questions = body["questions"]
        target = questions["click_target"]
        assert target["criteria"]["1"]["checked"] == "true"
        assert target["criteria"]["1"]["selected"] is False
        assert questions["operation"]["instructions"]["rules"] in target["instructions"]["rules"]
        return {
            "model": "test",
            "answers": {
                "operation": choice(questions["operation"]["criteria"], "CLICK"),
                "click_target": choice(target["criteria"], "3"),
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(p, "Search with free cancellation", [])
    assert d["choice"] == "e3"


def test_quoted_task_text_still_uses_the_llm(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    post = Mock(return_value={"choices": [{"message": {"content": '{"text":"Zurich"}'}}]})
    monkeypatch.setattr(model, "post_json", post)
    context = model.field_context('Fly from "Zurich" to London', page()["actions"][0], page(), [])
    assert model.field_text(context)[0] == "Zurich"
    assert post.call_count == 1
    sent = json.loads(post.call_args.args[2]["messages"][1]["content"])
    assert sent["goal"] == 'Fly from "Zurich" to London'


def test_missing_text_credential_stops_before_guessing(monkeypatch):
    monkeypatch.delenv("TEXT_MODEL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TEXT_MODEL_API_KEY"):
        model.field_text({"goal": 'Enter "Zurich"'})


@pytest.fixture
def runner():
    a = loop.Agent.__new__(loop.Agent)
    a.screenshots = False
    a.pending_text = None
    a.input_guarantee = "atomic"
    p = page()
    a.state = {
        "browser": Mock(fresh=Mock(return_value=True), observe=Mock(return_value=p)),
        "page": p,
        "decision": decision(),
        "goal": "Find a book",
        "history": [],
        "decisions": [],
        "status": "predicted",
        "started_at": time.perf_counter(),
        "record": False,
        "text_calls": [],
    }
    return a


def test_stale_decision_is_consumed_before_any_mutation(runner):
    runner.state["browser"].fresh.return_value = False
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["browser"].act.assert_not_called()
    assert runner.state["decision"] is None


def test_generated_text_reused_only_for_identical_retry_context(runner, monkeypatch):
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 10}))
    monkeypatch.setattr(loop, "field_text", helper)
    runner.state["browser"].act.side_effect = [StalePage("Changed before input"), None]
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["decision"] = decision()
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert helper.call_count == 1
    assert runner.state["browser"].act.call_count == 2  # The first call rejects before any browser input.
    assert runner.pending_text is None


def test_changed_field_context_does_not_reuse_generated_text(runner, monkeypatch):
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 10}))
    monkeypatch.setattr(loop, "field_text", helper)
    runner.state["browser"].act.side_effect = [StalePage("Changed before input"), None]
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["page"]["text"] = "Different page context"
    runner.state["decision"] = decision()
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert helper.call_count == 2


def test_loading_waits_do_not_trigger_no_progress_stop(runner):
    for _ in range(5):
        runner.state["decision"] = decision("wait")
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert len(runner.state["history"]) == 5 and runner.state["status"] == "ready"


def test_stale_observation_preserves_executed_action(runner):
    runner.state["decision"] = decision("e3")
    runner.state["browser"].observe.side_effect = StalePage("changed")
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert runner.state["history"][-1]["action"] == "Go"
    runner.state["browser"].act.assert_called_once()


def test_observation_is_one_atomic_browser_read(monkeypatch):
    import jev_ultrafast.browser as browser

    p = page()
    cdp = Mock(return_value={"result": {"value": p}})
    monkeypatch.setattr(browser, "cdp", cdp)
    actual = browser_operation({"operation": "observe", "session": "test", "screenshot": False})
    assert actual["actions"] == p["actions"]
    assert cdp.call_count == 1
    assert cdp.call_args.args[0] == "Runtime.evaluate"


def test_executor_rejects_a_stale_page_before_browser_input(monkeypatch):
    import jev_ultrafast.browser as browser

    b = browser.Browser.__new__(browser.Browser)
    b.fresh = Mock(return_value=False)
    operation = Mock()
    monkeypatch.setattr(browser, "browser_operation", operation)
    with pytest.raises(StalePage):
        b.act(page()["actions"][0], page(), "book")
    operation.assert_not_called()


@pytest.mark.parametrize("response", [{"exceptionDetails": {}}, {"result": {}}])
def test_interrupted_dropdown_mutation_cannot_be_retried_as_stale(monkeypatch, response):
    import jev_ultrafast.browser as browser

    # A navigation can destroy the evaluation result after the change event already fired.
    if "exceptionDetails" in response:
        response["exceptionDetails"] = {"text": "Execution context destroyed"}
    cdp = Mock(return_value=response)
    monkeypatch.setattr(browser, "cdp", cdp)
    with pytest.raises(RuntimeError, match="Dropdown execution"):
        browser_operation({
            "operation": "act",
            "session": "test",
            "expected": {"page_key": [1], "guard": [1]},
            "action": {
                "id": "e1", "kind": "select", "node": 1, "value": "Design",
                "option_index": 1, "option_label": "Design",
            },
        })
    assert cdp.call_count == 1


def test_fingerprint_tracks_values_and_identity_not_screenshots():
    p = page()
    other = deepcopy(p)
    other["screenshot"] = "changed"
    assert fingerprint(p) == fingerprint(other)
    other["actions"][0]["node"] = 99
    assert fingerprint(p) != fingerprint(other)


@pytest.mark.parametrize("changed", ["Departure", "Where from?", "Where to?", "year"])
def test_flight_verification_rejects_wrong_trip(changed):
    from examples.flights import verify

    actual = {
        "url": "https://www.google.com/travel/flights/search?tfs=example",
        "text": "Track prices from Zürich to London departing 2026-09-20",
        "actions": [
            {"label": k, "value": v}
            for k, v in [
                ("Change ticket type. One way", "One way"),
                ("Where from?", "Zürich"),
                ("Where to?", "London"),
                ("Departure", "Sun, Sep 20"),
                ("Nonstop flight on Sunday, September 20. Select flight", ""),
            ]
        ],
    }
    assert verify(actual)["passed"]
    if changed == "year":
        actual["text"] = actual["text"].replace("2026", "2027")
    else:
        next(a for a in actual["actions"] if a["label"] == changed)["value"] = "wrong"
    assert not verify(actual)["passed"]


@pytest.mark.parametrize(
    "content", ["Thinking: Zurich", '{"text":null}', '{"text":"Zurich","extra":true}', '{"text":123}']
)
def test_text_helper_rejects_invalid_values(monkeypatch, content):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", Mock(return_value={"choices": [{"message": {"content": content}}]}))
    with pytest.raises(ValueError, match="nothing typed"):
        model.field_text({"goal": "Find a flight"})


def test_navigation_during_prediction_reobserves_without_action(runner):
    runner.state["browser"].fresh.side_effect = StalePage("Document navigating")
    runner.command("tick")
    assert runner.state["status"] == "ready"
    assert runner.state["decision"] is None
    runner.state["browser"].act.assert_not_called()


def test_trace_candidates_only_computed_when_recording(runner, monkeypatch, tmp_path):
    from jev_ultrafast.dream import ExperienceStore
    from jev_ultrafast.trace import DreamTraceRecorder

    spy = Mock(wraps=model.candidate_actions)
    monkeypatch.setattr(loop, "candidate_actions", spy)
    runner.state["decision"] = decision("e3")
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    # Only the snapshot() bookkeeping call; no trace-catalogue work without a recorder.
    assert spy.call_count == 1

    runner.dream_recorder = DreamTraceRecorder(
        ExperienceStore(tmp_path / "trace.jsonl"), goal="Find a book",
    )
    # A fresh observation installs a fresh actions list, so the cached
    # perception for the old page must not leak into the new step.
    runner.state["browser"].observe.return_value = page()
    runner.state["decision"] = decision("e3")
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    # One shared perception serves the recorder's offered catalogue and
    # snapshot() — the cache keys on the observed actions list, not per consumer.
    assert spy.call_count == 2


def test_snapshot_annotation_does_not_mutate_shared_perception(runner):
    runner.state["decision"] = decision("e3")
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    # Inspector elements carry node ids, but the cached perception is also the
    # model request body — annotating it in place would leak local metadata
    # into the next choose() call.
    cached_elements = runner._perception_cache[5]
    assert all("node" not in element for element in cached_elements)


def test_close_records_unfinished_run_as_aborted(tmp_path):
    from jev_ultrafast.dream import CanaryMetrics, ExperienceStore, ExplorationPolicy
    from jev_ultrafast.trace import DreamTraceRecorder

    store = ExperienceStore(tmp_path / "abort.jsonl")
    a = loop.Agent.__new__(loop.Agent)
    a.dream_recorder = DreamTraceRecorder(store, goal="Find a book")
    a.dream_recorder.start({"fingerprint": "A", "url": "https://example.test"}, ExplorationPolicy())
    a.state = {"status": "ready", "verified": False}
    a.browser = Mock()
    a.close()
    events = store.load()
    assert [event["event"] for event in events] == ["run_started", "run_finished"]
    assert events[-1]["status"] == "aborted"
    # Abandoned runs are not task outcomes and must not count as canary failures.
    assert CanaryMetrics.from_events(events, ExplorationPolicy().digest).tasks == 0


def test_close_preserves_terminal_status(tmp_path):
    from jev_ultrafast.dream import ExperienceStore, ExplorationPolicy
    from jev_ultrafast.trace import DreamTraceRecorder

    store = ExperienceStore(tmp_path / "done.jsonl")
    a = loop.Agent.__new__(loop.Agent)
    a.dream_recorder = DreamTraceRecorder(store, goal="Find a book")
    a.dream_recorder.start({"fingerprint": "A", "url": "https://example.test"}, ExplorationPolicy())
    a.state = {"status": "done", "verified": True}
    a.browser = Mock()
    a.close()
    assert store.load()[-1]["status"] == "done"


def test_close_releases_browser_when_finish_fails(tmp_path):
    """A failing run_finished append must not orphan the browser target."""
    from jev_ultrafast.dream import ExperienceStore, ExplorationPolicy
    from jev_ultrafast.trace import DreamTraceRecorder

    store = ExperienceStore(tmp_path / "fail.jsonl")
    a = loop.Agent.__new__(loop.Agent)
    recorder = DreamTraceRecorder(store, goal="Find a book")
    recorder.start({"fingerprint": "A", "url": "https://example.test"}, ExplorationPolicy())
    a.dream_recorder = Mock(wraps=recorder)
    a.dream_recorder.finished = False
    a.dream_recorder.finish.side_effect = RuntimeError("store full")
    a.state = {"status": "ready", "verified": False}
    a.browser = Mock()
    with pytest.raises(RuntimeError, match="store full"):
        a.close()
    a.browser.close.assert_called_once()


def _buy_runner(runner):
    from jev_ultrafast.policy import DefaultActionPolicy

    runner.policy = DefaultActionPolicy()
    p = runner.state["page"]
    p["actions"].append(
        {"id": "buy", "kind": "click", "label": "Buy now", "role": "button", "node": 30}
    )
    p["fingerprint"] = fingerprint(p)
    runner.state["decision"] = decision("buy")
    runner.state["status"] = "predicted"
    return p


def test_caller_supplied_approval_flag_is_rejected(runner):
    p = _buy_runner(runner)
    for body in ({"approved": True}, {"approved": False}, {"approved": "yes"}):
        with pytest.raises(ValueError, match="caller-supplied approval"):
            runner.command("act", {"fingerprint": p["fingerprint"], **body})
    runner.state["browser"].act.assert_not_called()


def test_consequential_action_executes_only_via_bound_approval(runner):
    p = _buy_runner(runner)
    state = runner.command("act", {"fingerprint": p["fingerprint"]})
    assert state["status"] == "approval_required"
    assert state["pending_approval"]["reason"]
    runner.state["browser"].act.assert_not_called()

    runner.command("approve")
    runner.state["browser"].act.assert_called_once()
    assert runner.state["history"][-1]["action"] == "Buy now"
    assert runner.state["status"] == "ready"
    assert runner.state.get("granted_approval") is None

    # The grant is consumed once: the next consequential decision needs a new approval.
    runner.state["decision"] = decision("buy")
    runner.state["status"] = "predicted"
    again = runner.command("act", {"fingerprint": p["fingerprint"]})
    assert again["status"] == "approval_required"
    assert runner.state["browser"].act.call_count == 1


def test_approval_grant_cannot_approve_a_different_action(runner):
    p = _buy_runner(runner)
    runner.command("act", {"fingerprint": p["fingerprint"]})
    assert runner.state["status"] == "approval_required"
    # Forge a grant aimed at an unrelated action; the pending action must still pause.
    runner.state["granted_approval"] = {"action_id": "e3", "fingerprint": p["fingerprint"]}
    runner.state["decision"] = decision("buy")
    runner.state["status"] = "predicted"
    state = runner.command("act", {"fingerprint": p["fingerprint"]})
    assert state["status"] == "approval_required"
    runner.state["browser"].act.assert_not_called()


def test_reject_clears_pending_and_records_rejection(runner):
    p = _buy_runner(runner)
    runner.command("act", {"fingerprint": p["fingerprint"]})
    state = runner.command("reject")
    assert state["status"] == "ready"
    assert state["pending_approval"] is None
    assert runner.state["history"][-1]["approval"] == "rejected"
    runner.state["browser"].act.assert_not_called()


def test_approved_execution_marks_risk_events_in_trace(runner, tmp_path):
    from jev_ultrafast.dream import ExperienceStore
    from jev_ultrafast.trace import DreamTraceRecorder

    p = _buy_runner(runner)
    runner.dream_recorder = DreamTraceRecorder(
        ExperienceStore(tmp_path / "approval.jsonl"), goal="Buy a book"
    )
    runner.command("act", {"fingerprint": p["fingerprint"]})
    runner.command("approve")
    events = ExperienceStore(tmp_path / "approval.jsonl").load()
    kinds = [event["event"] for event in events]
    # The mutation journal brackets the dispatch: attempted before execution,
    # confirmed after, then the observed transition.
    assert kinds == ["approval_required", "action_attempted", "action_confirmed", "transition"]
    assert events[-1]["risk_events"] == 1


def test_indeterminate_mutation_is_journaled_and_never_retried(runner, tmp_path):
    """An interrupted mutation may already have crossed its mutation point —
    the journal must record the attempt AND its indeterminate outcome, the run
    aborts under its own censoring reason, and the action is never retried."""
    from jev_ultrafast.dream import ExperienceStore
    from jev_ultrafast.trace import DreamTraceRecorder

    path = tmp_path / "run.jsonl"
    runner.dream_recorder = DreamTraceRecorder(ExperienceStore(path), goal="Find a book")
    runner.state["browser"].act.side_effect = IndeterminateMutation(
        "press acknowledged, result lost"
    )
    runner.state["decision"] = decision("e3")
    p = runner.state["page"]
    with pytest.raises(IndeterminateMutation):
        runner.command("act", {"fingerprint": p["fingerprint"]})
    runner.browser = runner.state["browser"]
    runner.close()

    runner.state["browser"].act.assert_called_once()  # no retry, no re-observe
    assert runner.state["abort_reason"] == "indeterminate_execution"
    assert runner.state["history"][-1]["execution"] == "indeterminate"
    assert runner.state["history"][-1]["page_changed"] is None

    events = ExperienceStore(path).load()
    kinds = [event["event"] for event in events]
    assert "action_attempted" in kinds
    assert "action_indeterminate" in kinds
    assert "action_confirmed" not in kinds
    assert "transition" not in kinds
    final = events[-1]
    assert final["event"] == "run_finished"
    assert final["status"] == "aborted"
    assert final["reason"] == "indeterminate_execution"


def test_premutation_stale_is_journaled_as_not_executed(runner, tmp_path):
    """A provably pre-mutation StalePage stays recoverable, but the journal
    must still close the attempt: attempted, then not_executed — never left
    dangling and never claimed indeterminate."""
    from jev_ultrafast.dream import ExperienceStore
    from jev_ultrafast.trace import DreamTraceRecorder

    path = tmp_path / "run.jsonl"
    runner.dream_recorder = DreamTraceRecorder(ExperienceStore(path), goal="Find a book")
    runner.state["browser"].act.side_effect = StalePage("Target changed")
    runner.state["decision"] = decision("e3")
    p = runner.state["page"]
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": p["fingerprint"]})

    events = ExperienceStore(path).load()
    kinds = [event["event"] for event in events]
    assert "action_attempted" in kinds
    assert "action_not_executed" in kinds
    assert "action_indeterminate" not in kinds
    assert "action_confirmed" not in kinds


def test_sensitive_generated_text_escalates_a_benign_field(runner, monkeypatch):
    """Audit regression: the policy runs again on the generated payload, so a
    field that describes itself as plain text cannot launder a sensitive fill."""
    from jev_ultrafast.policy import DefaultActionPolicy

    runner.policy = DefaultActionPolicy()
    runner.state["decision"] = decision("e1")  # "Search" fill: FORM_EDIT → allow
    monkeypatch.setattr(loop, "field_context", Mock(return_value={"goal": "x"}))
    monkeypatch.setattr(
        loop, "field_text",
        Mock(return_value=("4111 1111 1111 1111", {"model": "t", "latency_ms": 1})),
    )
    p = runner.state["page"]
    state = runner.command("act", {"fingerprint": p["fingerprint"]})
    assert state["status"] == "approval_required"
    assert "payload" in state["pending_approval"]["reason"]
    runner.state["browser"].act.assert_not_called()

    # Approval binds to this action on this page; the executed payload is the
    # exact text that was assessed (pending_text is reused verbatim).
    runner.command("approve")
    runner.state["browser"].act.assert_called_once()
    assert runner.state["browser"].act.call_args.kwargs.get("text") == "4111 1111 1111 1111" \
        or "4111 1111 1111 1111" in runner.state["browser"].act.call_args.args


def test_ordinary_generated_text_keeps_benign_fill_autonomous(runner, monkeypatch):
    from jev_ultrafast.policy import DefaultActionPolicy

    runner.policy = DefaultActionPolicy()
    runner.state["decision"] = decision("e1")
    monkeypatch.setattr(loop, "field_context", Mock(return_value={"goal": "x"}))
    monkeypatch.setattr(
        loop, "field_text", Mock(return_value=("Zurich", {"model": "t", "latency_ms": 1}))
    )
    p = runner.state["page"]
    runner.command("act", {"fingerprint": p["fingerprint"]})
    runner.state["browser"].act.assert_called_once()
    assert runner.state["status"] == "ready"


def test_input_guarantee_defaults_to_atomic_and_rejects_unknown(monkeypatch):
    monkeypatch.delenv("JEV_INPUT_GUARANTEE", raising=False)
    # Validation must not depend on a real browser starting.
    monkeypatch.setattr(loop, "Browser", lambda *a, **k: Mock(observe=Mock(return_value=page())))
    b = loop.Agent("http://x", "g")
    assert b.input_guarantee == "atomic"
    monkeypatch.setenv("JEV_INPUT_GUARANTEE", "trusted")
    c = loop.Agent("http://x", "g")
    assert c.input_guarantee == "trusted"
    with pytest.raises(ValueError, match="input_guarantee"):
        loop.Agent("http://x", "g", input_guarantee="magical")


def test_input_guarantee_reaches_the_browser_dispatch(runner):
    """The operator-declared guarantee is journaled and passed to act() —
    trusted stays non-transactional, but is never auto-escalated into."""
    runner.input_guarantee = "trusted"
    runner.state["decision"] = decision("e3")
    runner.state["decision"]["operation"] = "CLICK"
    runner.state["decision"]["target"] = "2"
    p = runner.state["page"]
    runner.command("act", {"fingerprint": p["fingerprint"]})
    runner.state["browser"].act.assert_called_once()
    assert runner.state["browser"].act.call_args.kwargs["guarantee"] == "trusted"


def test_routing_local_only_refuses_remote_decision_endpoint(monkeypatch):
    monkeypatch.setenv("JEV_MODEL_ROUTING", "local-only")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    with pytest.raises(ValueError, match="local-only"):
        model.SystemOneBackend(url="https://api.example.test/v1").decide({})


def test_routing_local_only_refuses_remote_text_endpoint(monkeypatch):
    monkeypatch.setenv("JEV_MODEL_ROUTING", "local-only")
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    with pytest.raises(ValueError, match="local-only"):
        model.field_text({"goal": "x"})


def test_routing_local_only_allows_loopback(monkeypatch):
    monkeypatch.setenv("JEV_MODEL_ROUTING", "local-only")
    post = Mock(return_value={"choices": [{"message": {"content": '{"text":"Zurich"}'}}]})
    monkeypatch.setattr(model, "post_json", post)
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.delenv("TEXT_MODEL_API_KEY", raising=False)
    model.field_text({"goal": "where"})
    assert post.called


def test_routing_sanitized_redacts_goal_pii(monkeypatch):
    monkeypatch.setenv("JEV_MODEL_ROUTING", "sanitized")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    captured = {}

    def post(_url, _key, body):
        captured.update(body)
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "DONE"),
            },
        }

    monkeypatch.setattr(model, "post_json", post)
    model.choose(page(), "Email the receipt to alice@example.com now", [])
    wire = json.dumps(captured["questions"])
    assert "alice@example.com" not in wire
    assert "[REDACTED_EMAIL]" in wire
    # Local scoring still sees the raw goal — the bound copy is outbound-only.
    ctx = model.field_context("Ping alice@example.com", page()["actions"][0], page(), [])
    assert ctx["goal"] == "Ping [REDACTED_EMAIL]"


def test_routing_public_sends_goal_verbatim(monkeypatch):
    monkeypatch.setenv("JEV_MODEL_ROUTING", "public")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    captured = {}

    def post(_url, _key, body):
        captured.update(body)
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "DONE"),
            },
        }

    monkeypatch.setattr(model, "post_json", post)
    model.choose(page(), "Email alice@example.com", [])
    assert "alice@example.com" in json.dumps(captured["questions"])


def test_routing_unknown_level_fails_closed(monkeypatch):
    monkeypatch.setenv("JEV_MODEL_ROUTING", "trust-me")
    with pytest.raises(ValueError, match="JEV_MODEL_ROUTING"):
        model.SystemOneBackend().decide({})


def test_remote_plaintext_http_endpoint_refused(monkeypatch):
    """Non-loopback http:// would carry the bearer key and full model
    context in cleartext — refused regardless of routing level."""
    monkeypatch.delenv("JEV_ALLOW_INSECURE_TRANSPORT", raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    with pytest.raises(ValueError, match="https"):
        model.SystemOneBackend(url="http://model.example.com/v1").decide({})
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", "http://model.example.com/v1")
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    with pytest.raises(ValueError, match="https"):
        model.field_text({"goal": "x"})


def test_insecure_transport_override_is_explicit(monkeypatch):
    monkeypatch.setenv("JEV_ALLOW_INSECURE_TRANSPORT", "1")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    post = Mock(return_value={"model": "t", "answers": {}})
    monkeypatch.setattr(model, "post_json", post)
    model.SystemOneBackend(url="http://model.example.com/v1").decide({})
    assert post.called
    # Loopback http never needed the override.
    monkeypatch.delenv("JEV_ALLOW_INSECURE_TRANSPORT")
    model.SystemOneBackend(url="http://127.0.0.1:9000/v1").decide({})


def test_model_boundaries_redact_history_text(monkeypatch):
    monkeypatch.setenv("JEV_MODEL_PRIVACY", "basic")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    captured = {}

    def post(_url, _key, body):
        captured.update(body)
        return {
            "model": "test",
            "answers": {"operation": choice(body["questions"]["operation"]["criteria"], "DONE")},
        }

    monkeypatch.setattr(model, "post_json", post)
    history = [
        {"action": "Email", "kind": "fill", "text": "private@example.com", "page_changed": True},
        {"action": "Search", "kind": "fill", "text": "flights", "page_changed": True},
    ]
    model.choose(page(), "Send the mail", history)
    recent = captured["state"]["recent_actions"]
    assert recent[0]["text"] == "[REDACTED_EMAIL]"
    assert "private@example.com" not in json.dumps(recent)
    assert recent[1]["text"] == "flights"

    context = model.field_context("Send the mail", page()["actions"][0], page(), history)
    assert context["recent_actions"][0]["text"] == "[REDACTED_EMAIL]"
    assert "private@example.com" not in json.dumps(context["recent_actions"])


def test_approval_grant_is_bound_to_the_pending_payload(runner, monkeypatch):
    """The grant covers the exact text that was pending — the operator approved
    this payload, not whatever a regenerated helper might later produce."""
    import hashlib

    from jev_ultrafast.policy import DefaultActionPolicy

    runner.policy = DefaultActionPolicy()
    p = runner.state["page"]
    p["actions"][0]["label"] = "Credit card number"  # target alone escalates
    p["fingerprint"] = fingerprint(p)
    text = "4111 1111 1111 1111"
    monkeypatch.setattr(loop, "field_context", Mock(return_value={"goal": "x"}))
    monkeypatch.setattr(
        loop, "field_text", Mock(return_value=(text, {"model": "t", "latency_ms": 1}))
    )
    state = runner.command("act", {"fingerprint": p["fingerprint"]})
    assert state["status"] == "approval_required"
    assert state["pending_approval"]["payload_digest"] == hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()
    runner.command("approve")
    assert runner.state.get("granted_approval") is None  # consumed once
    act_call = runner.state["browser"].act.call_args
    assert act_call.args[0]["id"] == "e1"
    assert act_call.kwargs["text"] == text


def test_approval_grant_does_not_cover_a_changed_payload(runner, monkeypatch):
    """A grant minted for payload A cannot execute regenerated payload B — a
    stale or shifted helper context forces a fresh approval round."""
    from jev_ultrafast.policy import DefaultActionPolicy

    runner.policy = DefaultActionPolicy()
    p = runner.state["page"]
    p["actions"][0]["label"] = "Credit card number"
    p["fingerprint"] = fingerprint(p)
    monkeypatch.setattr(loop, "field_context", Mock(return_value={"goal": "x"}))
    monkeypatch.setattr(
        loop, "field_text", Mock(return_value=("4111 1111 1111 1111", {"model": "t"}))
    )
    state = runner.command("act", {"fingerprint": p["fingerprint"]})
    assert state["status"] == "approval_required"
    # Forge a grant bound to a different payload: it must not be consumed.
    runner.state["granted_approval"] = {
        "action_id": "e1",
        "fingerprint": p["fingerprint"],
        "payload_digest": "0" * 64,
    }
    runner.state["decision"] = decision("e1")
    runner.state["status"] = "predicted"
    again = runner.command("act", {"fingerprint": p["fingerprint"]})
    assert again["status"] == "approval_required"
    runner.state["browser"].act.assert_not_called()


def test_experiment_candidate_arm_executes_the_proposal(runner, monkeypatch, tmp_path):
    """The scheduler deviates to the prior's proposal and records the arm,
    the assignment propensity, and the model choice it deviated from."""
    from jev_ultrafast.dream import ExperienceStore, ExplorationPolicy
    from jev_ultrafast.trace import DreamTraceRecorder

    store = ExperienceStore(tmp_path / "exp.jsonl")
    runner.dream_recorder = DreamTraceRecorder(store, goal="Find a book")
    runner.dream_recorder.start(runner.state["page"], ExplorationPolicy())
    proposal_model = Mock()
    proposal_model.choose = Mock(return_value={"id": "e3", "kind": "click"})
    runner.experiment = {"model": proposal_model, "rate": 0.5, "rng": _candidate_rng()}
    runner.state["decision"] = decision("e1")  # the model picked the fill
    runner.state["status"] = "predicted"
    p = runner.state["page"]
    state = runner.command("act", {"fingerprint": p["fingerprint"]})
    assert state["status"] == "ready"
    assert runner.state["browser"].act.call_count == 1
    assert runner.state["browser"].act.call_args.args[0]["id"] == "e3"
    assert runner.state["history"][-1]["action"] == "Go"
    trial = next(e for e in store.load() if e["event"] == "transition")
    assert trial["experiment"]["arm"] == "candidate"
    assert trial["experiment"]["assignment_probability"] == 0.5
    assert trial["experiment"]["model_choice_id"] == "e1"
    assert trial["experiment"]["proposal_id"] == "e3"
    assert trial["experiment"]["proposal_digest"]
    assert trial["selected"]["id"] == "e3"  # executed action is the record


def test_experiment_control_arm_executes_the_model_choice(runner, monkeypatch):
    proposal_model = Mock()
    proposal_model.choose = Mock(return_value={"id": "e3", "kind": "click"})
    rng = Mock()
    rng.random = Mock(return_value=0.9)  # above rate → control arm
    runner.experiment = {"model": proposal_model, "rate": 0.5, "rng": rng}
    monkeypatch.setattr(
        loop, "field_text", Mock(return_value=("book", {"model": "t", "latency_ms": 1}))
    )
    runner.state["decision"] = decision("e1")
    runner.state["status"] = "predicted"
    p = runner.state["page"]
    runner.command("act", {"fingerprint": p["fingerprint"]})
    assert runner.state["browser"].act.call_args.args[0]["id"] == "e1"
    assert runner.state["history"][-1]["action"] == "Search"


def test_experiment_deviation_never_bypasses_authority(runner):
    """A trial that substitutes a high-authority action still pauses for
    approval — the experiment layer has no authority of its own."""
    from jev_ultrafast.policy import DefaultActionPolicy

    runner.policy = DefaultActionPolicy()
    p = runner.state["page"]
    p["actions"].append(
        {"id": "buy", "kind": "click", "label": "Buy now", "role": "button", "node": 30}
    )
    p["fingerprint"] = fingerprint(p)
    proposal_model = Mock()
    proposal_model.choose = Mock(return_value={"id": "buy", "kind": "click"})
    runner.experiment = {"model": proposal_model, "rate": 0.5, "rng": _candidate_rng()}
    runner.state["decision"] = decision("e1")
    runner.state["status"] = "predicted"
    state = runner.command("act", {"fingerprint": p["fingerprint"]})
    assert state["status"] == "approval_required"
    assert state["pending_approval"]["action"]["id"] == "buy"
    runner.state["browser"].act.assert_not_called()


def test_experiment_deviated_trial_executes_the_approved_action(runner, tmp_path):
    """Approving a deviated trial runs the *approved* proposal — not the
    model's original choice — and keeps the experiment tag on the transition."""
    from jev_ultrafast.dream import ExperienceStore, ExplorationPolicy
    from jev_ultrafast.policy import DefaultActionPolicy
    from jev_ultrafast.trace import DreamTraceRecorder

    store = ExperienceStore(tmp_path / "exp.jsonl")
    runner.dream_recorder = DreamTraceRecorder(store, goal="Find a book")
    runner.dream_recorder.start(runner.state["page"], ExplorationPolicy())
    runner.policy = DefaultActionPolicy()
    p = runner.state["page"]
    p["actions"].append(
        {"id": "buy", "kind": "click", "label": "Buy now", "role": "button", "node": 30}
    )
    p["fingerprint"] = fingerprint(p)
    proposal_model = Mock()
    proposal_model.choose = Mock(return_value={"id": "buy", "kind": "click"})
    runner.experiment = {"model": proposal_model, "rate": 0.5, "rng": _candidate_rng()}
    runner.state["decision"] = decision("e1")
    runner.state["status"] = "predicted"

    state = runner.command("act", {"fingerprint": p["fingerprint"]})
    assert state["status"] == "approval_required"
    assert state["pending_approval"]["action"]["id"] == "buy"
    assert state["pending_approval"]["experiment"]["arm"] == "candidate"

    runner.command("approve")
    assert runner.state["browser"].act.call_args.args[0]["id"] == "buy"
    trial = next(e for e in store.load() if e["event"] == "transition")
    assert trial["experiment"]["arm"] == "candidate"
    assert trial["experiment"]["assignment_probability"] == 0.5
    assert trial["experiment"]["model_choice_id"] == "e1"
    assert trial["experiment"]["proposal_id"] == "buy"
    assert trial["selected"]["id"] == "buy"
    assert trial["risk_events"] == 1  # the approved action consumed the grant


def test_experiment_config_is_validated_before_browser_launch(monkeypatch):
    """A malformed experiment config raises in the constructor — not mid-run
    after the decision was already consumed."""
    browser_cls = Mock()
    monkeypatch.setattr(loop, "Browser", browser_cls)
    with pytest.raises(ValueError, match="rate"):
        loop.Agent("https://example.test", "goal", experiment={"model": Mock(), "rate": 0})
    with pytest.raises(ValueError, match="rate"):
        loop.Agent("https://example.test", "goal", experiment={"model": Mock(), "rate": 2.0})
    # Degenerate allocations are rejected: rate=1.0 assigns every run to the
    # candidate (no control support — nothing causal is identifiable), and a
    # rate below the arm floor gives the other arm essentially none.
    for degenerate in (1.0, 0.01, 0.999):
        with pytest.raises(ValueError, match="both arms"):
            loop.Agent("https://example.test", "goal", experiment={
                "model": Mock(), "rate": degenerate,
            })
    with pytest.raises(ValueError, match="model"):
        loop.Agent("https://example.test", "goal", experiment={"rate": 0.5})
    with pytest.raises(ValueError, match="dict"):
        loop.Agent("https://example.test", "goal", experiment="yes")
    browser_cls.assert_not_called()


def test_experiment_only_assigns_one_trial_per_run(runner):
    proposal_model = Mock()
    proposal_model.choose = Mock(return_value={"id": "e3", "kind": "click"})
    runner.experiment = {"model": proposal_model, "rate": 0.5, "rng": _candidate_rng()}
    p = runner.state["page"]
    runner.state["decision"] = decision("e1")
    runner.state["status"] = "predicted"
    runner.command("act", {"fingerprint": p["fingerprint"]})
    assert runner.state["experiment_spent"] is True
    # A later step is not deviated even though a proposal still exists.
    proposal_model.choose.reset_mock()
    runner.state["decision"] = decision("e3")
    runner.state["status"] = "predicted"
    runner.command("act", {"fingerprint": p["fingerprint"]})
    proposal_model.choose.assert_not_called()


def test_experiment_proposal_must_be_an_offered_action(runner, tmp_path):
    """A proposal outside the policy-filtered offered catalogue is rejected —
    existing in page["actions"] is not enough. The run just executes the
    model's own choice with no trial recorded."""
    from jev_ultrafast.dream import ExperienceStore, ExplorationPolicy
    from jev_ultrafast.trace import DreamTraceRecorder

    store = ExperienceStore(tmp_path / "exp.jsonl")
    runner.dream_recorder = DreamTraceRecorder(store, goal="Find a book")
    runner.dream_recorder.start(runner.state["page"], ExplorationPolicy())
    p = runner.state["page"]
    # "good" overlaps the goal ("book") so it survives min_goal_overlap=1;
    # "filtered" does not, so the policy never offers it to the model.
    p["actions"].append(
        {"id": "good", "kind": "click", "label": "book order", "role": "button", "node": 40}
    )
    p["actions"].append(
        {"id": "filtered", "kind": "click", "label": "unrelated", "role": "button", "node": 50}
    )
    p["fingerprint"] = fingerprint(p)
    runner.exploration_policy = ExplorationPolicy(min_goal_overlap=1)
    proposal_model = Mock()
    proposal_model.choose = Mock(return_value={"id": "filtered", "kind": "click"})
    runner.experiment = {"model": proposal_model, "rate": 0.5, "rng": _candidate_rng()}
    runner.state["decision"] = decision("e3")
    runner.state["status"] = "predicted"
    state = runner.command("act", {"fingerprint": p["fingerprint"]})
    assert state["status"] == "ready"
    # The filtered proposal never executed and no trial was assigned.
    runner.state["browser"].act.assert_called_once()
    assert runner.state["browser"].act.call_args.args[0]["id"] == "e3"
    assert all(e.get("experiment") is None for e in store.load())


def test_experiment_assignment_is_recorded_before_authority(runner, tmp_path):
    """The experiment_assigned event lands at randomization — before policy,
    approval, or execution — so an operator-rejected trial still marks the
    whole run experimental and unfit for canary."""
    from jev_ultrafast.dream import (
        CanaryMetrics,
        ExperienceStore,
        ExplorationPolicy,
    )
    from jev_ultrafast.policy import DefaultActionPolicy
    from jev_ultrafast.trace import DreamTraceRecorder

    store = ExperienceStore(tmp_path / "exp.jsonl")
    runner.dream_recorder = DreamTraceRecorder(store, goal="Find a book")
    runner.dream_recorder.start(runner.state["page"], ExplorationPolicy())
    runner.policy = DefaultActionPolicy()
    runner.exploration_policy = ExplorationPolicy()
    p = runner.state["page"]
    p["actions"].append(
        {"id": "buy", "kind": "click", "label": "Buy now", "role": "button", "node": 30}
    )
    p["fingerprint"] = fingerprint(p)
    proposal_model = Mock()
    proposal_model.choose = Mock(return_value={"id": "buy", "kind": "click"})
    runner.experiment = {"model": proposal_model, "rate": 0.5, "rng": _candidate_rng()}
    runner.state["decision"] = decision("e1")
    runner.state["status"] = "predicted"

    state = runner.command("act", {"fingerprint": p["fingerprint"]})
    assert state["status"] == "approval_required"
    assigned = next(e for e in store.load() if e["event"] == "experiment_assigned")
    assert assigned["experiment"]["arm"] == "candidate"
    assert assigned["experiment"]["proposal_id"] == "buy"
    # The frozen plan carries the full provenance binding.
    meta = assigned["experiment"]
    for field in (
        "proposal_digest", "policy_behavior_digest", "offered_catalogue_digest",
        "experiment_id",
    ):
        assert meta[field], field
    runner.command("reject")
    # The operator vetoed the trial: no transition exists, yet the run is
    # still unmistakably experimental and excluded from canary.
    runner.dream_recorder.finish(status="blocked", verified=False)
    events = store.load()
    assert not any(e["event"] == "transition" for e in events)
    assert CanaryMetrics.from_events(events, ExplorationPolicy().digest).tasks == 0


def _stamped_plan(runner, *, proposal_id="e3", model_choice_id="e1", **overrides):
    """A *valid* stamped experiment plan for the runner's live step — digest
    and every execution binding computed the way DreamImprover stamps them."""
    from jev_ultrafast.dream import (
        candidate_catalog_digest,
        experiment_plan_digest,
        task_key,
    )
    from jev_ultrafast.privacy import redact_text, tokenize
    from jev_ultrafast.trace import compact_candidate

    goal = runner.state["goal"]
    goal_tokens = set(tokenize(redact_text(goal, 4096)))
    offered = runner._perception()[0]
    policy = getattr(runner, "exploration_policy", None)
    plan = {
        "schema": "jev-experiment-plan/1",
        "world": "w0",
        "task_key": task_key(goal),
        # Family identity exactly as the live agent derives it: the normalized
        # task family, or the task key when no family was configured.
        "family_key": (
            str(getattr(runner, "task_family", None) or "").strip().lower()
            or task_key(goal)
        ),
        "step": 0,
        "state": runner.state["page"]["fingerprint"],
        "historical": {
            "id": model_choice_id,
            "kind": "fill",
            "goal_overlap": 1,
            "offered_rank": 0,
        },
        "proposal": {
            "id": proposal_id,
            "kind": "click",
            "p_progress": 0.9,
            "uncertainty": 0.1,
        },
        "proposal_offered_rank": next(
            i for i, candidate in enumerate(offered) if candidate["id"] == proposal_id
        ),
        "offered_catalogue_digest": candidate_catalog_digest(
            [compact_candidate(candidate, goal_tokens) for candidate in offered]
        ),
        "policy_behavior_digest": getattr(policy, "behavior_digest", None),
        "choice_model_digest": None,
        "world_pool_digest": "0" * 64,
        "expected_delta": 0.4,
    }
    plan.update(overrides)
    plan["digest"] = experiment_plan_digest(plan)
    return plan


def test_experiment_stamped_proposal_executes_the_plan(runner, tmp_path):
    """experiment={"proposals": [...]} consumes the stamped DreamReport plan
    itself — digest and every binding verified — rather than re-asking a live
    model."""
    from jev_ultrafast.dream import ExperienceStore, ExplorationPolicy
    from jev_ultrafast.trace import DreamTraceRecorder

    store = ExperienceStore(tmp_path / "exp.jsonl")
    runner.dream_recorder = DreamTraceRecorder(store, goal="Find a book")
    runner.dream_recorder.start(runner.state["page"], ExplorationPolicy())
    runner.exploration_policy = ExplorationPolicy()
    p = runner.state["page"]
    stamped = _stamped_plan(runner)
    runner.experiment = {"proposals": [stamped], "rate": 0.5, "rng": _candidate_rng()}
    runner.state["decision"] = decision("e1")
    runner.state["status"] = "predicted"
    runner.command("act", {"fingerprint": p["fingerprint"]})
    assert runner.state["browser"].act.call_args.args[0]["id"] == "e3"
    events = store.load()
    assigned = next(e for e in events if e["event"] == "experiment_assigned")
    trial = next(e for e in events if e["event"] == "transition")
    assert assigned["experiment"]["stamped_digest"] == stamped["digest"]
    assert assigned["experiment"]["experiment_id"] == stamped["digest"][:16]
    assert assigned["experiment"]["expected_delta"] == 0.4
    assert trial["experiment"]["stamped_digest"] == stamped["digest"]


def _experiment_runner(runner, tmp_path, plan, monkeypatch):
    """Wire a runner for stamped-plan tests: recorder, policy, experiment —
    and a stubbed text helper so the model's own fill can execute when a
    stale plan correctly suppresses the substitution."""
    from jev_ultrafast.dream import ExperienceStore, ExplorationPolicy
    from jev_ultrafast.trace import DreamTraceRecorder

    store = ExperienceStore(tmp_path / "exp.jsonl")
    runner.dream_recorder = DreamTraceRecorder(store, goal="Find a book")
    runner.dream_recorder.start(runner.state["page"], ExplorationPolicy())
    if getattr(runner, "exploration_policy", None) is None:
        runner.exploration_policy = ExplorationPolicy()
    monkeypatch.setattr(
        loop, "field_text", Mock(return_value=("book", {"model": "t", "latency_ms": 1}))
    )
    runner.experiment = {"proposals": [plan], "rate": 0.5, "rng": _candidate_rng()}
    runner.state["decision"] = decision("e1")
    runner.state["status"] = "predicted"
    return store


def test_experiment_stamped_plan_rejects_tampered_proposal(runner, tmp_path, monkeypatch):
    """The audit's repro: mutate the proposal after stamping without
    re-digesting → the digest check fails and the plan is discarded, never
    reinterpreted. The model's own choice executes; no trial is assigned."""
    p = runner.state["page"]
    plan = _stamped_plan(runner)
    plan["proposal"] = {"id": "e2", "kind": "click"}  # post-digest tamper
    store = _experiment_runner(runner, tmp_path, plan, monkeypatch)
    runner.command("act", {"fingerprint": p["fingerprint"]})
    assert runner.state["browser"].act.call_args.args[0]["id"] == "e1"
    events = store.load()
    stale = next(e for e in events if e["event"] == "experiment_plan_stale")
    assert "digest" in stale["reasons"]
    assert not any(e["event"] == "experiment_assigned" for e in events)


def test_experiment_stamped_plan_binds_task_and_state(runner, tmp_path, monkeypatch):
    """A plan stamped for a different goal is stale here — the page
    fingerprint alone must never qualify it (same page, different task)."""
    p = runner.state["page"]
    plan = _stamped_plan(runner, task_key="deadbeef")
    store = _experiment_runner(runner, tmp_path, plan, monkeypatch)
    runner.command("act", {"fingerprint": p["fingerprint"]})
    assert runner.state["browser"].act.call_args.args[0]["id"] == "e1"
    events = store.load()
    stale = next(e for e in events if e["event"] == "experiment_plan_stale")
    assert "task_key" in stale["reasons"]


def test_experiment_stamped_plan_binds_model_choice(runner, tmp_path, monkeypatch):
    """The hypothesis is 'when the model picks A, try B'. If the live model
    picks something else, the conditioned-on premise never held — stale."""
    p = runner.state["page"]
    plan = _stamped_plan(runner, model_choice_id="e2")  # model will pick e1
    store = _experiment_runner(runner, tmp_path, plan, monkeypatch)
    runner.command("act", {"fingerprint": p["fingerprint"]})
    assert runner.state["browser"].act.call_args.args[0]["id"] == "e1"
    events = store.load()
    stale = next(e for e in events if e["event"] == "experiment_plan_stale")
    assert "model_choice" in stale["reasons"]


def test_experiment_stamped_plan_binds_policy_and_catalogue(runner, tmp_path, monkeypatch):
    """A plan minted under a different policy/offered catalogue is stale — the
    proposal may still be offered, but the experiment that was designed is
    not the one being run."""
    from jev_ultrafast.dream import ExplorationPolicy

    p = runner.state["page"]
    # Stamped under different policy behavior than the live one — same offered
    # catalogue (no_progress_window doesn't filter candidates), different
    # behavior digest.
    runner.exploration_policy = ExplorationPolicy(no_progress_window=10)
    plan = _stamped_plan(runner)
    runner.exploration_policy = ExplorationPolicy()  # live policy differs
    store = _experiment_runner(runner, tmp_path, plan, monkeypatch)
    runner.command("act", {"fingerprint": p["fingerprint"]})
    assert runner.state["browser"].act.call_args.args[0]["id"] == "e1"
    stale = next(e for e in store.load() if e["event"] == "experiment_plan_stale")
    assert "policy_behavior" in stale["reasons"]


def test_experiment_stale_plan_suppresses_live_fallback(runner, tmp_path, monkeypatch):
    """A stamped plan that claims this state but fails validation suppresses
    the ad-hoc live-model substitution for the step — the experiment that
    runs is the stamped one or none."""
    p = runner.state["page"]
    plan = _stamped_plan(runner, task_key="deadbeef")
    store = _experiment_runner(runner, tmp_path, plan, monkeypatch)
    live = Mock()
    live.choose = Mock(return_value={"id": "e3", "kind": "click"})
    runner.experiment = {"proposals": [plan], "model": live, "rate": 0.5,
                         "rng": _candidate_rng()}
    runner.command("act", {"fingerprint": p["fingerprint"]})
    assert runner.state["browser"].act.call_args.args[0]["id"] == "e1"
    live.choose.assert_not_called()
    assert not any(e["event"] == "experiment_assigned" for e in store.load())


def test_experiment_stamped_plan_binds_task_family(runner, tmp_path, monkeypatch):
    """The audit's repro: a hypothesis generated under one task family must
    not execute under another. Same goal text, same page state, same model
    choice — the family binding alone makes the plan stale."""
    p = runner.state["page"]
    runner.task_family = "banking"  # live family
    plan = _stamped_plan(runner, family_key="flights")  # stamped under another
    store = _experiment_runner(runner, tmp_path, plan, monkeypatch)
    runner.command("act", {"fingerprint": p["fingerprint"]})
    assert runner.state["browser"].act.call_args.args[0]["id"] == "e1"
    stale = next(e for e in store.load() if e["event"] == "experiment_plan_stale")
    assert "task_family" in stale["reasons"]
    assert not any(e["event"] == "experiment_assigned" for e in store.load())


def test_experiment_stamped_plan_stale_when_newer_evidence_refutes(runner, tmp_path, monkeypatch):
    """Freshness: an old hypothesis does not survive newer randomized
    evidence. A live causal model that now refutes the plan's proposal makes
    the plan stale instead of silently re-running a settled question."""
    p = runner.state["page"]
    plan = _stamped_plan(runner)
    store = _experiment_runner(runner, tmp_path, plan, monkeypatch)
    refuting = Mock()
    refuting.refuted = Mock(return_value=True)
    refuting.digest = "refuting-digest"
    runner.experiment = {"proposals": [plan], "model": refuting, "rate": 0.5,
                         "rng": _candidate_rng()}
    runner.command("act", {"fingerprint": p["fingerprint"]})
    assert runner.state["browser"].act.call_args.args[0]["id"] == "e1"
    stale = next(e for e in store.load() if e["event"] == "experiment_plan_stale")
    assert "trial_refuted" in stale["reasons"]


def test_experiment_stamped_plan_signature_required_when_keys_trusted(runner, tmp_path, monkeypatch):
    """With trusted verification keys configured, plan provenance is
    mandatory: unsigned plans are stale, forged signatures are stale, and a
    correctly signed plan executes."""
    from jev_ultrafast.dream import (
        ExplorationPolicy,
        experiment_plan_digest,
        experiment_plan_signature,
    )
    from jev_ultrafast.signing import EvidenceSigner

    signer = EvidenceSigner(bytes(range(32)))
    runner.exploration_policy = ExplorationPolicy()  # stable stamping basis
    p = runner.state["page"]

    def last_stale(store):
        return [e for e in store.load() if e["event"] == "experiment_plan_stale"][-1]

    unsigned = _stamped_plan(runner)
    store = _experiment_runner(runner, tmp_path, unsigned, monkeypatch)
    runner.experiment_verify_keys = {signer.key_id}
    runner.command("act", {"fingerprint": p["fingerprint"]})
    assert runner.state["browser"].act.call_args.args[0]["id"] == "e1"
    assert "plan_unsigned" in last_stale(store)["reasons"]

    forged_plan = _stamped_plan(runner)
    forged_plan["authority"] = {"key_id": signer.key_id, "signature": "00" * 64}
    store2 = _experiment_runner(runner, tmp_path, forged_plan, monkeypatch)
    runner.command("act", {"fingerprint": p["fingerprint"]})
    assert "plan_signature" in last_stale(store2)["reasons"]

    signed = _stamped_plan(runner)
    signed["digest"] = experiment_plan_digest(signed)
    signed["authority"] = experiment_plan_signature(signed, signer)
    store3 = _experiment_runner(runner, tmp_path, signed, monkeypatch)
    runner.command("act", {"fingerprint": p["fingerprint"]})
    assert runner.state["browser"].act.call_args.args[0]["id"] == "e3"
    assert any(e["event"] == "experiment_assigned" for e in store3.load())


def test_experiment_stamped_plan_requires_current_model_when_configured(runner, tmp_path, monkeypatch):
    """``require_current_model`` turns the soft digest check into a hard one:
    a plan whose originating model is not available and current is stale."""
    p = runner.state["page"]
    plan = _stamped_plan(runner, choice_model_digest="some-digest")
    store = _experiment_runner(runner, tmp_path, plan, monkeypatch)
    runner.experiment = {"proposals": [plan], "rate": 0.5,
                         "rng": _candidate_rng(), "require_current_model": True}
    runner.command("act", {"fingerprint": p["fingerprint"]})
    assert runner.state["browser"].act.call_args.args[0]["id"] == "e1"
    stale = next(e for e in store.load() if e["event"] == "experiment_plan_stale")
    assert "choice_model" in stale["reasons"]


def test_experiment_assignment_meta_carries_itt_context(runner, tmp_path):
    """The assignment event is self-contained: the pre-treatment context the
    intention-to-treat estimator needs survives even when nothing executes."""
    from jev_ultrafast.dream import ExperienceStore, ExplorationPolicy
    from jev_ultrafast.trace import DreamTraceRecorder

    store = ExperienceStore(tmp_path / "exp.jsonl")
    runner.dream_recorder = DreamTraceRecorder(store, goal="Find a book")
    runner.dream_recorder.start(runner.state["page"], ExplorationPolicy())
    runner.task_family = "search"
    runner.instance_id = "inst-7"
    proposal_model = Mock()
    proposal_model.choose = Mock(return_value={"id": "e3", "kind": "click"})
    runner.experiment = {"model": proposal_model, "rate": 0.5, "rng": _candidate_rng()}
    runner.state["decision"] = decision("e1")
    runner.state["status"] = "predicted"
    p = runner.state["page"]
    runner.command("act", {"fingerprint": p["fingerprint"]})
    meta = next(e for e in store.load() if e["event"] == "experiment_assigned")["experiment"]
    for field in (
        "task_key", "task_family", "site", "instance_id", "state",
        "model_choice_id", "model_choice_kind", "model_choice_overlap",
        "model_choice_offered_rank", "proposal_id", "proposal_kind",
        "proposal_overlap", "proposal_offered_rank", "assignment_probability",
        "experiment_id",
    ):
        assert field in meta, field
    assert meta["task_family"] == "search"
    assert meta["site"] == "example.test"
    assert meta["instance_id"] == "inst-7"
    assert meta["model_choice_id"] == "e1" and meta["proposal_id"] == "e3"


def test_experiment_proposals_config_validated_in_init(monkeypatch):
    browser_cls = Mock()
    monkeypatch.setattr(loop, "Browser", browser_cls)
    # Stamped proposals alone are a valid experiment config — no model needed.
    loop.Agent("https://example.test", "goal", experiment={
        "proposals": [{"state": "s", "proposal": {"id": "x"}}], "rate": 0.5,
    })
    with pytest.raises(ValueError, match="proposals"):
        loop.Agent("https://example.test", "goal", experiment={
            "proposals": "notalist", "rate": 0.5,
        })


# ------------------------------------------------------- structured aborts


def test_close_records_operator_cancel_for_unfinished_run(runner, tmp_path):
    """An operator closing a mid-run session is censored with its cause —
    not scored as a task failure, and not confused with a crash."""
    from jev_ultrafast.dream import ExperienceStore, ExplorationPolicy
    from jev_ultrafast.trace import DreamTraceRecorder

    store = ExperienceStore(tmp_path / "exp.jsonl")
    runner.dream_recorder = DreamTraceRecorder(store, goal="Find a book")
    runner.dream_recorder.start(runner.state["page"], ExplorationPolicy())
    runner.browser = runner.state["browser"]  # close() tears the browser down
    runner.state["status"] = "ready"  # mid-run, no terminal decision
    runner.close()
    final = next(e for e in store.load() if e["event"] == "run_finished")
    assert final["status"] == "aborted"
    assert final["reason"] == "operator_cancel"


def test_run_records_structured_abort_reason_for_crashes(runner, tmp_path):
    """A run that dies mid-flight records *why*: a model timeout, a browser
    crash, and an unattributable exception are distinguishable in evidence,
    so censoring analysis can tell infrastructure from treatment-side
    hangs."""
    from jev_ultrafast.browser import BrowserError
    from jev_ultrafast.dream import ExperienceStore, ExplorationPolicy
    from jev_ultrafast.model import ModelConnectionError, ModelTimeoutError
    from jev_ultrafast.trace import DreamTraceRecorder

    cases = (
        (ModelTimeoutError("timed out"), "timeout"),
        (ModelConnectionError("connection refused"), "network_failure"),
        (BrowserError("cdp died"), "browser_crash"),
        (RuntimeError("boom"), "agent_exception"),
    )
    for exc, expected in cases:
        store = ExperienceStore(tmp_path / f"{expected}.jsonl")
        runner.dream_recorder = DreamTraceRecorder(store, goal="Find a book")
        runner.dream_recorder.start(runner.state["page"], ExplorationPolicy())
        runner.state["status"] = "ready"
        runner.state.pop("abort_reason", None)
        runner.browser = runner.state["browser"]

        def boom(*_args, **_kwargs):
            raise exc

        runner.command = boom
        with pytest.raises(type(exc)):
            list(runner.run())
        runner.close()
        final = next(e for e in store.load() if e["event"] == "run_finished")
        assert final["status"] == "aborted"
        assert final["reason"] == expected


# ------------------------------------------------------- causal policy modes


def _causal_evidence():
    """Randomized evidence: assigning the high-overlap click was beneficial
    for the divergence premise (model picked a low-overlap click). Enough
    per arm for the confidence sequence to establish the sign."""
    from jev_ultrafast.dream import ExplorationPolicy

    events = []
    for i in range(24):
        for arm, success in (("candidate", True), ("control", False)):
            meta = {
                "experiment_id": f"x{i}{arm}",
                "arm": arm,
                "assignment_probability": 0.5,
                "proposal_id": "e3",
                "proposal_kind": "click",
                "proposal_effect": "search",
                "proposal_role": "button",
                "proposal_overlap": 0,
                "proposal_offered_rank": 1,
                "model_choice_id": "e1",
                "model_choice_kind": "fill",
                "model_choice_effect": "form_edit",
                "model_choice_role": "textbox",
                "model_choice_overlap": 0,
                "model_choice_offered_rank": 0,
                "task_key": "t",
            }
            run_id = f"r{i}{arm}"
            events += [
                {"event": "run_started", "run_id": run_id, "task_key": "t", "goal": "g",
                 "policy": ExplorationPolicy().to_dict(),
                 "policy_digest": ExplorationPolicy().digest},
                {"event": "experiment_assigned", "run_id": run_id, "task_key": "t",
                 "experiment": meta},
                {"event": "run_finished", "run_id": run_id, "task_key": "t",
                 "status": "done" if success else "blocked", "verified": success},
            ]
    return events


def test_causal_policy_shadow_annotates_without_influencing(runner, monkeypatch):
    """Shadow mode reports what the combined policy would prefer — with
    provenance — and changes nothing about execution."""
    from jev_ultrafast.dreamlearn import CausalChoicePolicy, TrialChoiceModel

    model = TrialChoiceModel.fit(_causal_evidence())
    runner.experiment = {"causal_policy": CausalChoicePolicy(mode="shadow", trial_model=model)}
    # One click sibling only, so the ranking has a single unambiguous top.
    p = runner.state["page"]
    p["actions"] = [a for a in p["actions"] if a["id"] != "e2"]
    p["fingerprint"] = fingerprint(p)
    monkeypatch.setattr(loop, "choose", lambda *a, **k: decision("e1"))
    state = runner.command("predict")
    assert state["decision"]["choice"] == "e1"  # untouched
    shadow = state["causal_shadow"]
    assert shadow["shadow"] is True
    assert shadow["proposals"][0]["id"] == "e3"
    # No task family was configured, so the evidence backed off to the pooled
    # stratum — and the provenance says exactly that.
    assert shadow["proposals"][0]["source"] == "pooled_randomized"
    assert shadow["proposals"][0]["causal"]["effect_status"] == "beneficial"


def test_causal_policy_canary_proposal_enters_randomized_assignment(runner, tmp_path, monkeypatch):
    """Canary mode's proposal is a *trial* candidate: it runs only through
    the recorded random assignment, never deterministically."""
    from jev_ultrafast.dream import ExperienceStore, ExplorationPolicy
    from jev_ultrafast.dreamlearn import CausalChoicePolicy, TrialChoiceModel
    from jev_ultrafast.trace import DreamTraceRecorder

    store = ExperienceStore(tmp_path / "exp.jsonl")
    runner.dream_recorder = DreamTraceRecorder(store, goal="Find a book")
    runner.dream_recorder.start(runner.state["page"], ExplorationPolicy())
    runner.exploration_policy = ExplorationPolicy()
    model = TrialChoiceModel.fit(_causal_evidence())
    p = runner.state["page"]
    p["actions"] = [a for a in p["actions"] if a["id"] != "e2"]
    p["fingerprint"] = fingerprint(p)
    runner.experiment = {
        "causal_policy": CausalChoicePolicy(mode="canary", trial_model=model),
        "rng": _candidate_rng(),  # deterministic roll < rate
        "rate": 0.5,
    }
    runner.state["decision"] = decision("e1")
    runner.state["status"] = "predicted"
    p = runner.state["page"]
    runner.command("act", {"fingerprint": p["fingerprint"]})
    assert runner.state["browser"].act.call_args.args[0]["id"] == "e3"
    assigned = next(e for e in store.load() if e["event"] == "experiment_assigned")
    assert assigned["experiment"]["arm"] == "candidate"
    assert assigned["experiment"]["proposal_id"] == "e3"
    # The assignment carries the bounded treatment signature of both arms.
    meta = assigned["experiment"]
    assert meta["model_choice_effect"] and meta["proposal_effect"]
    assert meta["phase"] == 0


def test_causal_policy_active_override_is_recorded_and_canary_excluded(runner, tmp_path, monkeypatch):
    """Active mode executes an established proposal deterministically — still
    through the authority plane — records the override, and keeps the run out
    of policy-canary qualification."""
    from jev_ultrafast.dream import CanaryMetrics, ExperienceStore, ExplorationPolicy
    from jev_ultrafast.dreamlearn import CausalChoicePolicy, TrialChoiceModel
    from jev_ultrafast.trace import DreamTraceRecorder

    store = ExperienceStore(tmp_path / "exp.jsonl")
    runner.dream_recorder = DreamTraceRecorder(store, goal="Find a book")
    runner.dream_recorder.start(runner.state["page"], ExplorationPolicy())
    runner.exploration_policy = ExplorationPolicy()
    model = TrialChoiceModel.fit(_causal_evidence())
    p = runner.state["page"]
    p["actions"] = [a for a in p["actions"] if a["id"] != "e2"]
    p["fingerprint"] = fingerprint(p)
    runner.experiment = {"causal_policy": CausalChoicePolicy(mode="active", trial_model=model)}
    runner.state["decision"] = decision("e1")
    runner.state["status"] = "predicted"
    p = runner.state["page"]
    runner.command("act", {"fingerprint": p["fingerprint"]})
    assert runner.state["browser"].act.call_args.args[0]["id"] == "e3"
    events = store.load()
    applied = next(e for e in events if e["event"] == "causal_policy_applied")
    assert applied["selected"] == "e3"
    # The override is not a trial (no randomized assignment), and the run is
    # not a clean sample of the exploration policy under test.
    assert not any(e["event"] == "experiment_assigned" for e in events)
    assert CanaryMetrics.from_events(events, ExplorationPolicy().digest).tasks == 0
