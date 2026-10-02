"""Regression tests for v0.2 execution-integrity and policy boundaries."""

import re
from pathlib import Path
from unittest.mock import Mock

import pytest

from jev_ultrafast import agent as loop
from jev_ultrafast import browser, model
from jev_ultrafast.browser import IndeterminateMutation, StalePage, browser_operation, fingerprint
from jev_ultrafast.policy import DefaultActionPolicy
from jev_ultrafast.privacy import redact_text, sanitize_action, sanitize_url


def base_page(actions=None):
    state = {
        "url": "https://example.test/",
        "title": "Example",
        "text": "Example page",
        "scroll": {"y": 0},
        "page_key": [1],
        "guards": {"1": [1]},
        "actions": actions or [{"id": "wait", "kind": "wait", "label": "Wait"}],
    }
    state["fingerprint"] = fingerprint(state)
    return state


def terminal_runner(choice="DONE", verifier=None):
    a = loop.Agent.__new__(loop.Agent)
    a.screenshots = False
    a.pending_text = None
    a.verifier = verifier
    a.policy = None
    p = base_page()
    a.state = {
        "browser": Mock(fresh=Mock(return_value=True)),
        "page": p,
        "decision": {
            "choice": choice,
            "operation": choice,
            "target": None,
            "confidence": 1.0,
            "probabilities": {choice: 1.0},
            "latency_ms": 1,
            "usage": {},
        },
        "goal": "Verify the task",
        "history": [],
        "decisions": [],
        "status": "predicted",
        "started_at": None,
        "record": False,
        "text_calls": [],
        "pending_approval": None,
        "verification": None,
        "verification_failures": [],
        "verified": False,
        "plan": ["Verify the task"],
        "plan_index": 0,
        "elapsed_ms": 0,
    }
    return a


def test_snapshot_cache_is_not_in_main_world():
    source = Path(browser.__file__).with_name("snapshot.js").read_text()
    assert "globalThis.__jevFastV2" in source
    assert "window.__jevFast" not in source


def test_context_id_is_bound_to_runtime_evaluation(monkeypatch):
    p = base_page()
    cdp = Mock(return_value={"result": {"value": p}})
    monkeypatch.setattr(browser, "cdp", cdp)
    browser_operation({"operation": "observe", "session": "s", "context_id": 42, "screenshot": False})
    assert cdp.call_args.kwargs["contextId"] == 42


def test_mutation_without_expected_guard_fails_closed(monkeypatch):
    cdp = Mock()
    monkeypatch.setattr(browser, "cdp", cdp)
    with pytest.raises(StalePage, match="Missing execution guard"):
        browser_operation({
            "operation": "act",
            "session": "s",
            "context_id": 42,
            "action": {"id": "e1", "kind": "click", "node": 1},
        })
    cdp.assert_not_called()


def test_fill_lost_focus_never_inserts_text(monkeypatch):
    """Trusted-input fill: pre-press/pre-release checks pass, then the
    focus+insert evaluation reports focus loss — no text may be dispatched."""
    calls = []

    def fake_cdp(method, session_id=None, **kwargs):
        calls.append((method, kwargs))
        if method == "Runtime.evaluate":
            runtime_count = sum(1 for name, _ in calls if name == "Runtime.evaluate")
            if runtime_count == 1:
                return {"result": {"value": {"x": 10, "y": 10}}}
            if runtime_count <= 3:
                # Pre-press and pre-release identity checks pass; the atomic
                # focus+insert evaluation then reports the focus failure.
                return {"result": {"value": True}}
            return {"result": {"value": {"error": "focus"}}}
        return {}

    monkeypatch.setattr(browser, "cdp", fake_cdp)
    with pytest.raises(StalePage, match="failed before mutation"):
        browser_operation({
            "operation": "act",
            "session": "s",
            "context_id": 42,
            "expected": {"page_key": [1], "guard": [1]},
            "action": {"id": "e1", "kind": "fill", "node": 1},
            "text": "secret",
            "guarantee": "trusted",
        })
    assert not any(name == "Input.insertText" for name, _ in calls)


def test_atomic_click_mutates_inside_the_single_guarded_evaluation(monkeypatch):
    """DOM_ATOMIC: validation and mutation share one isolated-world turn, so
    no page script can interleave between them. No CDP input is dispatched."""
    calls = []

    def fake_cdp(method, session_id=None, **kwargs):
        calls.append(method)
        if method == "Runtime.evaluate":
            return {"result": {"value": {"x": 10, "y": 10, "clicked": True}}}
        return {}

    monkeypatch.setattr(browser, "cdp", fake_cdp)
    result = browser_operation({
        "operation": "act",
        "session": "s",
        "context_id": 42,
        "expected": {"page_key": [1], "guard": [1]},
        "action": {"id": "e1", "kind": "click", "node": 1},
        "guarantee": "atomic",
    })
    assert result == {"executed": "e1"}
    assert calls == ["Runtime.evaluate"]  # exactly one evaluation, no input dispatch


def test_unsupported_atomic_target_escalates_to_trusted_input(monkeypatch):
    """An element without HTMLElement.click() provably did not mutate when it
    returned 'unsupported', so it may escalate to the trusted-input path."""
    calls = []
    evaluations = 0

    def fake_cdp(method, session_id=None, **kwargs):
        nonlocal evaluations
        calls.append(method)
        if method == "Runtime.evaluate":
            evaluations += 1
            if evaluations == 1:
                return {"result": {"value": {"error": "unsupported"}}}
            if evaluations == 2:
                return {"result": {"value": {"x": 5, "y": 5}}}
            return {"result": {"value": True}}  # precheck passes twice
        return {}

    monkeypatch.setattr(browser, "cdp", fake_cdp)
    result = browser_operation({
        "operation": "act",
        "session": "s",
        "context_id": 42,
        "expected": {"page_key": [1], "guard": [1]},
        "action": {"id": "e1", "kind": "click", "node": 1},
        "guarantee": "atomic",
    })
    assert result == {"executed": "e1"}
    assert calls.count("Input.dispatchMouseEvent") == 2  # press + release


def test_atomic_fill_mismatch_is_not_retried(monkeypatch):
    calls = []

    def fake_cdp(method, session_id=None, **kwargs):
        calls.append(method)
        return {"result": {"value": {"x": 1, "y": 1, "inserted": True, "matched": False}}}

    monkeypatch.setattr(browser, "cdp", fake_cdp)
    with pytest.raises(RuntimeError, match="landed differently"):
        browser_operation({
            "operation": "act",
            "session": "s",
            "context_id": 42,
            "expected": {"page_key": [1], "guard": [1]},
            "action": {"id": "e1", "kind": "fill", "node": 1},
            "text": "secret",
        })
    assert calls == ["Runtime.evaluate"]  # single atomic evaluation


def test_duplicate_select_values_keep_distinct_exact_option_indices():
    actions = [
        {"id": "e1", "kind": "select", "node": 7, "role": "combobox", "label": "Plan → Standard",
         "value": "x", "current_value": "", "option_index": 1, "option_label": "Standard"},
        {"id": "e2", "kind": "select", "node": 7, "role": "combobox", "label": "Plan → Premium",
         "value": "x", "current_value": "", "option_index": 2, "option_label": "Premium"},
    ]
    elements, targets, _ = model.action_space(actions)
    assert [o["option_index"] for o in elements[0]["options"]] == [1, 2]
    assert targets["SELECT"]["1:2"]["option_index"] == 1
    assert targets["SELECT"]["1:3"]["option_index"] == 2


def test_select_target_keys_are_stable_across_selection_change():
    # The snapshot omits the currently-selected option, so positional keys
    # re-map after every selection: "6:1" meant Design, then All stays, then
    # Design — the model toggled the same key between different options
    # forever. Keys must encode the DOM option_index instead.
    before = [
        {"id": "e1", "kind": "select", "node": 6, "role": "combobox", "label": "Cat → Design",
         "value": "design", "current_value": "All", "option_index": 1, "option_label": "Design"},
        {"id": "e2", "kind": "select", "node": 6, "role": "combobox", "label": "Cat → Nature",
         "value": "nature", "current_value": "All", "option_index": 2, "option_label": "Nature"},
    ]
    # After Design is selected, the DOM omits it from the offered options —
    # All stays (index 0) is now the first entry but must keep its own key.
    after = [
        {"id": "e3", "kind": "select", "node": 6, "role": "combobox", "label": "Cat → All",
         "value": "all", "current_value": "Design", "option_index": 0, "option_label": "All"},
        {"id": "e4", "kind": "select", "node": 6, "role": "combobox", "label": "Cat → Nature",
         "value": "nature", "current_value": "Design", "option_index": 2, "option_label": "Nature"},
    ]
    _, targets_before, _ = model.action_space(before)
    _, targets_after, _ = model.action_space(after)
    assert targets_before["SELECT"]["1:2"]["option_index"] == 1  # Design before
    assert targets_after["SELECT"]["1:1"]["option_index"] == 0   # All after
    assert targets_after["SELECT"]["1:3"]["option_index"] == 2   # Nature keeps its key


def test_large_dropdown_cannot_evict_other_operation_types():
    actions = [
        {"id": f"s{i}", "kind": "select", "node": 1, "role": "combobox", "label": f"Category → {i}",
         "value": str(i), "option_index": i, "option_label": str(i)}
        for i in range(400)
    ]
    actions += [
        {"id": "fill", "kind": "fill", "node": 2, "role": "textbox", "label": "Destination", "value": ""},
        {"id": "click", "kind": "click", "node": 3, "role": "button", "label": "Search", "value": ""},
        {"id": "wait", "kind": "wait", "label": "Wait"},
    ]
    candidates, omitted = model.candidate_actions(actions, "Search destination", limit=250)
    ids = {a["id"] for a in candidates}
    assert {"fill", "click", "wait"} <= ids
    assert len(candidates) == 250 and omitted == len(actions) - 250
    assert candidates.index(next(a for a in candidates if a["id"] == "fill")) < 250


def test_basic_privacy_redacts_incidental_secrets(monkeypatch):
    monkeypatch.setenv("JEV_MODEL_PRIVACY", "basic")
    assert "dawson@example.com" not in redact_text("Contact dawson@example.com")
    action = {"label": "Credit card number", "value": "4111111111111111"}
    assert sanitize_action(action)["value"] == "[REDACTED]"


def test_destructive_account_action_requires_approval():
    result = DefaultActionPolicy().assess({"kind": "click", "label": "Delete account"})
    assert result.level == "require_approval"


def test_done_without_verifier_is_only_claimed_done():
    a = terminal_runner()
    state = a.command("act", {"fingerprint": a.state["page"]["fingerprint"]})
    assert state["status"] == "claimed_done"
    assert state["verified"] is False


def test_done_with_independent_verifier_becomes_verified_done():
    a = terminal_runner(verifier=lambda page: {"passed": page["url"] == "https://example.test/"})
    state = a.command("act", {"fingerprint": a.state["page"]["fingerprint"]})
    assert state["status"] == "done"
    assert state["verified"] is True


def test_failed_verifier_returns_agent_to_ready_state():
    a = terminal_runner(verifier=lambda _page: {"passed": False, "reason": "missing result"})
    state = a.command("act", {"fingerprint": a.state["page"]["fingerprint"]})
    assert state["status"] == "ready"
    assert state["history"][-1]["kind"] == "verification"
    assert state["verification_failures"][-1]["reason"] == "missing result"


def test_local_systemone_backend_can_run_without_bearer_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("JEV_DECISION_API_KEY", raising=False)
    post = Mock(return_value={"model": "local", "answers": {}})
    monkeypatch.setattr(model, "post_json", post)
    backend = model.SystemOneBackend(url="http://127.0.0.1:9000/v1/systemone")
    assert backend.decide({"model": "local"})["model"] == "local"
    assert post.call_args.args[1] is None


def test_local_text_backend_can_run_without_api_key(monkeypatch):
    monkeypatch.delenv("TEXT_MODEL_API_KEY", raising=False)
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", "http://localhost:8000/v1")
    post = Mock(return_value={"choices": [{"message": {"content": '{"text":"Zurich"}'}}]})
    monkeypatch.setattr(model, "post_json", post)
    assert model.field_text({"goal": "Enter Zurich"})[0] == "Zurich"
    assert post.call_args.args[1] is None


def test_model_url_redacts_sensitive_query_values(monkeypatch):
    monkeypatch.setenv("JEV_MODEL_PRIVACY", "basic")
    clean = sanitize_url("https://example.test/callback?code=abc123&city=Zurich&email=dawson@example.com")
    assert "abc123" not in clean
    assert "dawson%40example.com" not in clean and "dawson@example.com" not in clean
    assert "city=Zurich" in clean


@pytest.mark.parametrize("param", [
    "api_key", "apikey", "access_key", "access_token", "refresh_token", "id_token",
    "client_secret", "private_key", "session_id", "csrf", "xsrf", "jwt", "otp", "passwd",
])
def test_common_sensitive_query_names_are_redacted(monkeypatch, param):
    monkeypatch.setenv("JEV_MODEL_PRIVACY", "basic")
    clean = sanitize_url(f"https://example.test/cb?{param}=s3cr3tv4lu3&city=Zurich")
    assert "s3cr3tv4lu3" not in clean
    assert "city=Zurich" in clean


@pytest.mark.parametrize("error", [
    "stale-page", "stale-target", "unavailable", "offscreen",
    "covered", "readonly", "unsupported-select", "stale-option",
])
def test_select_premutation_errors_reobserve_instead_of_aborting(monkeypatch, error):
    # The guarded script returns these errors before its single mutation statement,
    # so re-observing is safe: no dropdown value could have been written.
    cdp = Mock(return_value={"result": {"value": {"error": error}}})
    monkeypatch.setattr(browser, "cdp", cdp)
    with pytest.raises(StalePage):
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


def test_verifier_internal_typeerror_is_not_retried_as_one_argument():
    calls = []

    def verifier(_page, _snapshot):
        calls.append(True)
        raise TypeError("internal verifier bug")

    a = terminal_runner(verifier=verifier)
    with pytest.raises(TypeError, match="internal verifier bug"):
        a.command("act", {"fingerprint": a.state["page"]["fingerprint"]})
    assert len(calls) == 1


def test_verifier_signature_dispatch_covers_supported_shapes():
    def two(_page, _snapshot):
        return {"passed": True}

    def optional(_page, _snapshot=None):
        return {"passed": True}

    def variadic(*args):
        assert len(args) == 2
        return {"passed": True}

    def one(_page):
        return True

    for verifier in (two, optional, variadic, one):
        a = terminal_runner(verifier=verifier)
        state = a.command("act", {"fingerprint": a.state["page"]["fingerprint"]})
        assert state["status"] == "done"


def test_isolated_expressions_invoke_their_argument():
    """The (fn)(arg) templates must actually call the function with the JSON arg.

    A missing call paren turns `((input) => ...){...}` into a SyntaxError that
    only surfaces against a real browser — mocks can't catch it (v0.4 live run).
    """
    src = Path(browser.__file__).read_text()
    joins = [m.start() for m in re.finditer(r'"""\s*\+\s*json\.dumps\(', src)]
    assert joins, "expected templated isolated-world expressions"
    for pos in joins:
        assert src.rstrip()[:pos].rstrip().endswith("("), (
            "expression argument is concatenated without a call paren"
        )


def test_fresh_treats_unreachable_guard_as_stale():
    b = browser.Browser.__new__(browser.Browser)
    b._isolated = Mock(side_effect=StalePage("Isolated execution context changed"))
    page = {"marker": "m", "page_key": "k", "guards": {"0": "g"}}
    assert b.fresh(page) is False
    assert b.fresh(page, {"kind": "click", "node": 0}) is False


def test_close_best_effort_when_daemon_gone(monkeypatch):
    """close() runs inside __exit__ and __init__ failure paths; a dead daemon
    must not mask the original error or leak the target."""
    b = browser.Browser.__new__(browser.Browser)
    b.target, b.session = "t1", "s1"
    b.world_context, b.world_frame = 7, "f"
    monkeypatch.setattr(browser, "cdp", Mock(side_effect=RuntimeError("daemon gone")))
    b.close()  # must not raise
    assert (b.target, b.session, b.world_context, b.world_frame) == (None, None, None, None)


def test_select_guard_normalizes_option_value_like_snapshot():
    """snapshot.js records clip(o.value): whitespace-normalized. The guarded
    select must apply identical normalization or a value with irregular
    spacing can never match and retries forever."""
    src = Path(browser.__file__).read_text()
    assert "String(o.value).replace(" in src
    assert "String(o.label).replace(" in src


def test_atomic_click_lost_result_is_indeterminate_not_stale(monkeypatch):
    """e.click() can navigate before Runtime.evaluate replies — the context is
    destroyed on the way back, so a lost result is NOT 'not executed'. The
    outcome is indeterminate and must never be retried as a clean stale."""
    calls = []

    def fake_cdp(method, session_id=None, **kwargs):
        calls.append(method)
        if method == "Runtime.evaluate":
            raise RuntimeError("Execution context was destroyed")
        return {}

    monkeypatch.setattr(browser, "cdp", fake_cdp)
    with pytest.raises(IndeterminateMutation):
        browser_operation({
            "operation": "act",
            "session": "s",
            "context_id": 42,
            "expected": {"page_key": [1], "guard": [1]},
            "action": {"id": "e1", "kind": "click", "node": 1},
            "guarantee": "atomic",
        })
    assert calls == ["Runtime.evaluate"]  # one dispatch, no retry


def test_atomic_mutation_missing_result_is_indeterminate(monkeypatch):
    """A mutating evaluation that returns no value at all cannot prove the
    mutation did not run — 'validated and returned nothing' is
    indistinguishable from 'mutated, then the result was lost'."""
    monkeypatch.setattr(browser, "cdp", Mock(return_value={"result": {}}))
    with pytest.raises(IndeterminateMutation):
        browser_operation({
            "operation": "act",
            "session": "s",
            "context_id": 42,
            "expected": {"page_key": [1], "guard": [1]},
            "action": {"id": "e1", "kind": "click", "node": 1},
            "guarantee": "atomic",
        })


@pytest.mark.parametrize("shapeless", [{}, 0, False, "", "ok", [1]])
def test_atomic_mutation_shapeless_result_is_indeterminate(monkeypatch, shapeless):
    """A mutating call whose acknowledgement is not the expected object —
    empty, scalar, or a wrong type — can no more prove 'not executed' than a
    missing result can. Falsy shapes used to take the retryable-stale branch
    (and a truthy non-dict would have crashed on .get())."""
    monkeypatch.setattr(
        browser, "cdp", Mock(return_value={"result": {"value": shapeless}})
    )
    with pytest.raises(IndeterminateMutation):
        browser_operation({
            "operation": "act",
            "session": "s",
            "context_id": 42,
            "expected": {"page_key": [1], "guard": [1]},
            "action": {"id": "e1", "kind": "click", "node": 1},
            "guarantee": "atomic",
        })


@pytest.mark.parametrize("shapeless", [None, {}, 0, "ok"])
def test_trusted_validation_shapeless_result_is_stale(monkeypatch, shapeless):
    """Under 'trusted' the guarded script only validates and reports
    coordinates — it provably never mutates, so a missing or shapeless
    result is a failed check (retryable stale), not an ambiguous mutation."""
    monkeypatch.setattr(
        browser, "cdp", Mock(return_value={"result": {"value": shapeless}})
    )
    with pytest.raises(StalePage):
        browser_operation({
            "operation": "act",
            "session": "s",
            "context_id": 42,
            "expected": {"page_key": [1], "guard": [1]},
            "action": {"id": "e1", "kind": "click", "node": 1},
            "guarantee": "trusted",
        })


def test_atomic_fill_lost_result_is_indeterminate(monkeypatch):
    """execCommand('insertText') may fire handlers that navigate before the
    evaluation returns; same indeterminate class as click."""
    monkeypatch.setattr(
        browser, "cdp", Mock(side_effect=RuntimeError("Execution context destroyed"))
    )
    with pytest.raises(IndeterminateMutation):
        browser_operation({
            "operation": "act",
            "session": "s",
            "context_id": 42,
            "expected": {"page_key": [1], "guard": [1]},
            "action": {"id": "e1", "kind": "fill", "node": 1},
            "text": "secret",
            "guarantee": "atomic",
        })


def test_trusted_prerelease_exception_still_dispatches_release(monkeypatch):
    """After mousePressed the pointer must never be stranded: a pre-release
    check that throws still attempts mouseReleased under finally, and the
    click outcome is indeterminate — not a clean retry."""
    calls = []

    def fake_cdp(method, session_id=None, **kwargs):
        calls.append(method)
        if method == "Runtime.evaluate":
            n = sum(1 for name in calls if name == "Runtime.evaluate")
            if n == 1:
                return {"result": {"value": {"x": 5, "y": 5}}}  # validation
            if n == 2:
                return {"result": {"value": True}}  # pre-press identity check
            raise RuntimeError("Execution context destroyed")  # pre-release dies
        return {}

    monkeypatch.setattr(browser, "cdp", fake_cdp)
    with pytest.raises(IndeterminateMutation):
        browser_operation({
            "operation": "act",
            "session": "s",
            "context_id": 42,
            "expected": {"page_key": [1], "guard": [1]},
            "action": {"id": "e1", "kind": "click", "node": 1},
            "guarantee": "trusted",
        })
    # Press dispatched once; the cleanup release was still attempted.
    assert calls.count("Input.dispatchMouseEvent") == 2


def test_trusted_release_dispatch_failure_is_indeterminate(monkeypatch):
    """If the release itself is lost after a validated press, the click
    outcome is unknowable — indeterminate, never a transport crash or stale."""
    calls = []

    def fake_cdp(method, session_id=None, **kwargs):
        calls.append(method)
        if method == "Runtime.evaluate":
            n = sum(1 for name in calls if name == "Runtime.evaluate")
            if n == 1:
                return {"result": {"value": {"x": 5, "y": 5}}}
            return {"result": {"value": True}}  # both identity checks pass
        if method == "Input.dispatchMouseEvent" and kwargs.get("type") == "mouseReleased":
            raise RuntimeError("socket closed")
        return {}

    monkeypatch.setattr(browser, "cdp", fake_cdp)
    with pytest.raises(IndeterminateMutation):
        browser_operation({
            "operation": "act",
            "session": "s",
            "context_id": 42,
            "expected": {"page_key": [1], "guard": [1]},
            "action": {"id": "e1", "kind": "click", "node": 1},
            "guarantee": "trusted",
        })


def test_trusted_insert_lost_result_is_indeterminate(monkeypatch):
    """Trusted-path text insertion mutates inside the evaluation; a lost
    result means the fill may already have happened."""
    calls = []

    def fake_cdp(method, session_id=None, **kwargs):
        calls.append(method)
        if method == "Runtime.evaluate":
            n = sum(1 for name in calls if name == "Runtime.evaluate")
            if n == 1:
                return {"result": {"value": {"x": 5, "y": 5}}}
            if n <= 3:
                return {"result": {"value": True}}  # pre-press/pre-release pass
            return {"result": {}}  # insert evaluation returns no value
        return {}

    monkeypatch.setattr(browser, "cdp", fake_cdp)
    with pytest.raises(IndeterminateMutation):
        browser_operation({
            "operation": "act",
            "session": "s",
            "context_id": 42,
            "expected": {"page_key": [1], "guard": [1]},
            "action": {"id": "e1", "kind": "fill", "node": 1},
            "text": "secret",
            "guarantee": "trusted",
        })
