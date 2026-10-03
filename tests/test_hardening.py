"""Regression tests for v0.2 execution-integrity and policy boundaries."""

import json
import re
import shutil
import subprocess
import time
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


# ------------------------------------------------- snapshot.js (real Node run)
#
# The snapshot only runs inside a page, but its catalogue-completeness contract
# is too important to leave to source inspection. A minimal stub DOM is enough:
# the script reads elements through a small, fully stubbable surface.


_SNAPSHOT_HARNESS = r"""
const fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf8');
const spec = JSON.parse(process.argv[3]);

class El {
  constructor(tag, attrs) {
    this.tagName = tag.toUpperCase();
    this.attrs = attrs || {};
    this.type = String(this.attrs.type || '');
    this.value = this.attrs.value || '';
    this.name = this.attrs.name || '';
    this.id = this.attrs.id || '';
    this.target = this.attrs.target || '';
    this.labels = [];
    this.childNodes = [];
    this.isConnected = true;
    this.disabled = false;
    this.readOnly = false;
    this.checked = false;
    this.selectedIndex = -1;
    this.isContentEditable = false;
    this.form = null;
    this.parentElement = {innerText: 'page'};
  }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  hasAttribute(k) { return k in this.attrs; }
  closest() { return null; }
  matches(sel) {
    return sel.split(',').some(part => {
      part = part.trim();
      if (part === ':disabled') return this.disabled;
      // Compound forms the catalogue actually uses: tag, [attr], [attr="v"],
      // tag[attr], tag[attr="v"]. (:not(...) compounds stay unsupported —
      // contenteditable is not exercised by these fixtures.)
      const m = part.match(/^(?:([A-Za-z]+))?(?:\[([^\]=]+)(?:="([^"]*)")?\])?$/);
      if (!m || (!m[1] && !m[2])) return false;
      const [, tag, attr, attrVal] = m;
      if (tag && this.tagName !== tag.toUpperCase()) return false;
      if (attr) {
        const v = this.getAttribute(attr);
        if (attrVal === undefined ? v === null : v !== attrVal) return false;
      }
      return true;
    });
  }
  checkVisibility() { return true; }
  getBoundingClientRect() {
    return {x: 10, y: 10, width: 100, height: 20, top: 10, bottom: 30, left: 10, right: 110};
  }
  querySelector() { return null; }
}

const els = spec.map(s => {
  const e = new El(s.tag, {
    type: s.type, 'aria-label': s.label, autocomplete: s.autocomplete,
    name: s.name, id: s.id, href: s.href, target: s.target, role: s.role,
    ...(s.download ? {download: s.download} : {}),
  });
  if (s.tag === 'select') {
    e.options = [];
    for (let i = 0; i < s.options; i++) {
      e.options.push({selected: i === 0, disabled: false, closest: () => null,
        label: 'Option ' + i, value: 'v' + i});
    }
    e.selectedOptions = e.options.filter(o => o.selected);
    e.selectedIndex = 0;
    e.multiple = false;
  }
  // Optional form association: enough DOM truth for ctxOf to compute submit
  // semantics, form method/origin, and the scoped sensitive-field inventory.
  if (s.form) {
    const form = new El('form', {method: s.form_method || 'post',
      action: s.form_action || null, role: s.form_role || null});
    const markers = s.scope_markers || [];
    form.querySelector = sel => markers.some(m => sel.includes(m)) ? form : null;
    e.form = form;
    e.closest = sel =>
      sel.split(',').map(x => x.trim()).includes('form') ? form : null;
  }
  return e;
});

globalThis.location = {href: 'https://t.test/', origin: 'https://t.test'};
globalThis.scrollX = 0;
globalThis.scrollY = 0;
globalThis.innerWidth = 1280;
globalThis.innerHeight = 800;
globalThis.NodeFilter = {SHOW_TEXT: 4};
globalThis.document = {
  body: {},
  title: 'T',
  documentElement: {scrollHeight: 800},
  getElementById: () => null,
  querySelectorAll: () => els,
  createTreeWalker: () => ({nextNode: () => null}),
  createRange: () => ({
    selectNodeContents() {},
    getBoundingClientRect() {
      return {width: 0, height: 0, top: 0, bottom: 0, left: 0, right: 0};
    },
  }),
};

console.log(JSON.stringify(eval(src)));
"""


def _run_snapshot_js(spec, tmp_path):
    """Run snapshot.js in Node against a stub DOM built from ``spec``."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required for the browser snapshot contract")
    harness = tmp_path / "snapshot_harness.js"
    harness.write_text(_SNAPSHOT_HARNESS)
    result = subprocess.run(
        [node, str(harness),
         str(Path(browser.__file__).with_name("snapshot.js")), json.dumps(spec)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def test_giant_select_cannot_starve_the_action_catalogue(tmp_path):
    """A 1500-option <select> early in DOM order used to consume the whole
    1200-action catalogue before later controls were ever represented — the
    anti-crowding ranking in Python then had nothing to rank. The node-fair
    merge guarantees every node's first action lands before any node's
    second, so the cap cuts a giant select's own tail, not the controls
    behind it."""
    page = _run_snapshot_js([
        {"tag": "select", "label": "Country", "options": 1500},
        {"tag": "input", "type": "search", "label": "Search"},
        {"tag": "button", "label": "Go"},
    ], tmp_path)
    actions = page["actions"]
    # 1499 selectable options + fill + open-click + go-click = 1502 raw.
    assert page["raw_action_count"] == 1502
    assert page["omitted_actions"] == 302
    assert len(actions) == 1203  # 1200 catalogue + 2 key controls + wait sentinel
    # First pass of the interleave: each node's head action, in DOM order.
    assert actions[1]["kind"] == "fill" and actions[1]["label"] == "Search"
    assert actions[2]["kind"] == "click" and actions[2]["label"] == "Go"
    # Everything the cap removed was select tail — a prefix of its options
    # survives, in order.
    select_indexes = [a["option_index"] for a in actions if a["kind"] == "select"]
    assert select_indexes == list(range(1, 1198))
    # Keyboard-scroll controls land between the catalogue and the sentinel.
    assert actions[-3]["kind"] == "key" and actions[-3]["key"] == "PageDown"
    assert actions[-2]["kind"] == "key" and actions[-2]["key"] == "PageUp"
    assert actions[-1]["kind"] == "wait"


def test_catalogue_stays_complete_below_the_cap(tmp_path):
    """Below the cap nothing is dropped and DOM order is preserved node by
    node (each node's bucket still interleaves deterministically)."""
    page = _run_snapshot_js([
        {"tag": "input", "type": "search", "label": "Search"},
        {"tag": "select", "label": "Country", "options": 3},
        {"tag": "button", "label": "Go"},
    ], tmp_path)
    actions = page["actions"]
    assert page["omitted_actions"] == 0
    kinds = [(a["kind"], a["label"]) for a in actions]
    assert kinds[0] == ("fill", "Search")
    assert ("click", "Go") in kinds
    # Both non-selected options are represented before the catalogue ends.
    assert [a["option_index"] for a in actions if a["kind"] == "select"] == [1, 2]


def _node_guard(page):
    """The recorded execution guard for the page's first action."""
    return page["guards"][str(page["actions"][0]["node"])]


def test_execution_guard_binds_authority_context(tmp_path):
    """classify_effect() reads ctxOf() — form membership, submit semantics,
    field class, scoped sensitive inventory, messaging/external/download —
    and the pre-mutation guard must change when any of it does. Otherwise a
    page could flip authority-relevant attributes between the policy
    decision and dispatch, and the same authorized action would execute
    under a weaker classification than its real semantics."""
    def guard(spec):
        return _node_guard(_run_snapshot_js(spec, tmp_path))

    # Identity invariance: the same DOM produces the same guard.
    text = guard([{"tag": "input", "type": "text", "label": "Note"}])
    assert guard([{"tag": "input", "type": "text", "label": "Note"}]) == text

    # Target field class: text -> email / otp / money all change authority.
    email = guard([{"tag": "input", "type": "email", "label": "Note"}])
    otp = guard([{"tag": "input", "type": "text", "label": "Note",
                  "autocomplete": "one-time-code"}])
    money = guard([{"tag": "input", "type": "text", "label": "Note",
                    "name": "card_number"}])
    assert len({json.dumps(g, sort_keys=True) for g in (text, email, otp, money)}) == 4

    # Submit semantics: the auditor's exact bypass — a form button whose
    # type flips from button to submit must not keep the old guard.
    benign = guard([{"tag": "button", "type": "button", "label": "Continue", "form": True}])
    submit = guard([{"tag": "button", "type": "submit", "label": "Continue", "form": True}])
    assert benign != submit

    # Form method and destination origin are authority context too.
    get_form = guard([{"tag": "button", "type": "submit", "label": "Continue",
                       "form": True, "form_method": "get"}])
    cross = guard([{"tag": "button", "type": "submit", "label": "Continue",
                    "form": True, "form_action": "https://evil.example/"}])
    assert benign != get_form and benign != cross

    # A sensitive sibling field appearing in the scope is context.
    with_pw = guard([{"tag": "button", "type": "submit", "label": "Continue",
                      "form": True, "scope_markers": ["password"]}])
    assert with_pw != benign

    # Messaging / external / download destinations.
    link = guard([{"tag": "a", "href": "https://t.test/docs", "label": "Open"}])
    mailto = guard([{"tag": "a", "href": "mailto:a@example.com", "label": "Open"}])
    blank = guard([{"tag": "a", "href": "https://t.test/docs", "label": "Open",
                    "target": "_blank"}])
    download = guard([{"tag": "a", "href": "https://t.test/f", "label": "Open",
                       "download": "f"}])
    assert len({json.dumps(g, sort_keys=True) for g in (link, mailto, blank, download)}) == 4


def test_sanitize_url_strips_userinfo_and_path_secrets(monkeypatch):
    """A credential in URL userinfo or a token in the path must not cross
    the model boundary while the run claims redaction."""
    monkeypatch.setenv("JEV_MODEL_PRIVACY", "basic")
    out = sanitize_url("https://alice:secret@example.test/home")
    assert "alice" not in out and "secret" not in out
    assert "example.test/home" in out
    # Secret-shaped and email-shaped path segments.
    assert "sk-abcdefghijklmnop1234" not in sanitize_url(
        "https://example.test/reset/sk-abcdefghijklmnop1234")
    assert "alice@example.com" not in sanitize_url(
        "https://example.test/users/alice@example.com")
    # A bare opaque token segment (no keyword prefix, mixed letters+digits).
    assert "a1b2c3d4e5f6a1b2c3d4e5f6" not in sanitize_url(
        "https://example.test/session/a1b2c3d4e5f6a1b2c3d4e5f6")
    # Ordinary slugs survive — the token mask only fires on opaque segments.
    assert sanitize_url("https://example.test/docs/getting-started") == \
        "https://example.test/docs/getting-started"
    # Query-param redaction is unchanged.
    out = sanitize_url("https://example.test/cb?token=abc12345&q=flights")
    assert "token=%5BREDACTED%5D" in out and "q=flights" in out
    # An encoded '?' inside a path must not re-enter the emitted URL as a
    # real query boundary carrying unredacted key=value material.
    out = sanitize_url("https://example.test/r%3Ftoken%3Dsupersecretvalue")
    assert "supersecretvalue" not in out and "token=[REDACTED]" in out
    # The same mask applies to sensitive key=value pairs written plainly in a
    # path segment, in a fragment (OAuth-style tokens), and nested inside a
    # query parameter's decoded value.
    assert "hunter2" not in sanitize_url("https://example.test/cb/session=hunter2")
    assert "hunter2" not in sanitize_url("https://app.test/w#access_token=hunter2")
    assert "hunter2" not in sanitize_url(
        "https://example.test/?next=/cb%3Ftoken%3Dhunter2")
    # A decoded %2F stays inside its segment rather than becoming structure.
    assert sanitize_url("https://example.test/s/a%2Fb").endswith("/s/a%2Fb")
    # Pure-digit path ids are token-shaped too (a 24-digit run exceeds the
    # card-pattern bound yet is still account-id material).
    assert "123456789012345678901234" not in sanitize_url(
        "https://example.test/acct/123456789012345678901234")
    # Long ordinary slugs — alpha+hyphen, no digits — still survive.
    assert sanitize_url(
        "https://example.test/docs/getting-started-with-the-new-platform"
    ).endswith("/docs/getting-started-with-the-new-platform")


# --- Phase 16: iframe / shadow DOM / tab / keyboard-scroll / upload ----------


def test_upload_targets_expand_per_declared_file():
    """UPLOAD mirrors select's option axis: element_index:file_index keys over
    the operator-declared basenames, never paths."""
    actions = [
        {"id": "e1", "kind": "upload", "node": 9, "role": "button",
         "label": "Choose file — Attach", "ctx": {"field": "file"}},
    ]
    elements, targets, controls = model.action_space(actions, upload_names=["a.pdf", "b.txt"])
    assert elements[0]["operations"] == ["UPLOAD"]
    assert [f["name"] for f in elements[0]["files"]] == ["a.pdf", "b.txt"]
    assert targets["UPLOAD"]["1:1"]["file_index"] == 0
    assert targets["UPLOAD"]["1:2"]["file_index"] == 1
    assert targets["UPLOAD"]["1:2"]["id"] == "e1"
    assert not controls


def test_upload_actions_are_inert_without_declared_files():
    """A page can offer a file input while the operator declared no files —
    the action stays in the observed catalogue but never reaches the model."""
    actions = [
        {"id": "e1", "kind": "upload", "node": 9, "role": "button",
         "label": "Choose file — Attach", "ctx": {"field": "file"}},
        {"id": "e2", "kind": "click", "node": 2, "role": "button", "label": "Go"},
    ]
    offered, _ = model.candidate_actions(actions, "upload a file", upload_names=())
    assert [a["id"] for a in offered] == ["e2"]
    elements, targets, _ = model.action_space(actions, upload_names=())
    assert "UPLOAD" not in targets
    offered, _ = model.candidate_actions(actions, "upload a file", upload_names=["a.pdf"])
    assert "e1" in {a["id"] for a in offered}


def test_uploads_are_ranked_regulars_not_controls():
    """Upload actions join the ranked catalogue (kind bonus like select's),
    not the always-offered control list — a page of file inputs cannot
    starve the fill/click budget the way unranked controls could."""
    actions = [
        {"id": f"u{i}", "kind": "upload", "node": i, "role": "button",
         "label": f"Choose file {i}", "ctx": {"field": "file"}}
        for i in range(4)
    ] + [
        {"id": "f1", "kind": "fill", "node": 90, "role": "textbox", "label": "Search"},
        {"id": "wait", "kind": "wait", "label": "Wait"},
    ]
    offered, _ = model.candidate_actions(actions, "upload", upload_names=["a.pdf"], limit=250)
    assert {a["id"] for a in offered} == {"u0", "u1", "u2", "u3", "f1", "wait"}
    _, targets, controls = model.action_space(offered, upload_names=["a.pdf"])
    assert "UPLOAD" in targets and "WAIT" in controls


def test_agent_upload_resolution_binds_declared_files_only(tmp_path):
    """_resolve_upload maps the decision's file index to the allowlist —
    a missing or out-of-range selection fails before any CDP call."""
    f = tmp_path / "attach.txt"
    f.write_text("data")
    agent = loop.Agent.__new__(loop.Agent)
    agent.uploads = [{"name": f.name, "path": str(f), "size": 4}]
    assert agent._resolve_upload({"target": "3:1"})["path"] == str(f)
    with pytest.raises(ValueError, match="no file selection"):
        agent._resolve_upload({"target": "3"})
    with pytest.raises(ValueError, match="outside the declared set"):
        agent._resolve_upload({"target": "3:2"})
    with pytest.raises(ValueError, match="outside the declared set"):
        agent._resolve_upload({"target": "3:0"})


def test_resolve_uploads_validates_the_allowlist(tmp_path, monkeypatch):
    good = tmp_path / "a.txt"
    good.write_text("x")
    entries = loop._resolve_uploads([str(good), str(good)])  # duplicates dedup
    assert len(entries) == 1 and entries[0]["name"] == "a.txt"
    with pytest.raises(ValueError, match="does not exist"):
        loop._resolve_uploads([str(tmp_path / "missing.bin")])
    with pytest.raises(ValueError, match="regular file"):
        loop._resolve_uploads([str(tmp_path)])
    monkeypatch.setenv("JEV_UPLOADS", str(good))
    assert loop._resolve_uploads(None)[0]["path"] == str(good.resolve())


def test_upload_declaration_snapshots_content_into_private_staging(tmp_path):
    """Declaration copies the exact declared bytes under a content hash —
    the staged snapshot, not the mutable source path, is the thing an
    approval grant and dispatch can ever reach."""
    import hashlib

    src = tmp_path / "attach.pdf"
    src.write_bytes(b"original-content-x")
    entries = loop._resolve_uploads([str(src)])
    entry = entries[0]
    assert entry["content_sha256"] == hashlib.sha256(b"original-content-x").hexdigest()
    assert entry["size"] == len(b"original-content-x")
    staged = Path(entry["staged_path"])
    # The page still observes the real basename; the staging root is private.
    assert staged.name == "attach.pdf"
    assert staged.parent.parent.name.startswith("jev-uploads-")
    assert staged.read_bytes() == b"original-content-x"


def test_same_size_source_replacement_cannot_reach_the_staged_upload(tmp_path):
    """The audit's TOCTOU case: path+size proved nothing. A file swapped for
    same-size different content after declaration must leave the approved
    bytes untouched — the snapshot is what dispatch sends."""
    src = tmp_path / "attach.pdf"
    src.write_bytes(b"original-content-x")
    entry = loop._resolve_uploads([str(src)])[0]
    src.write_bytes(b"swapped-content-yy")  # same byte length, same name
    staged = Path(entry["staged_path"])
    assert staged.read_bytes() == b"original-content-x"
    import hashlib

    assert hashlib.sha256(staged.read_bytes()).hexdigest() == entry["content_sha256"]
    # The declared path is kept only for provenance — never for dispatch.
    assert entry["path"] == str(src.resolve()) != entry["staged_path"]


def _upload_decision():
    return {
        "choice": "u1",
        "operation": "UPLOAD",
        "target": "1:1",
        "confidence": 1.0,
        "probabilities": {"u1": 1.0},
        "latency_ms": 1,
        "usage": {},
    }


def _upload_agent(tmp_path, payload=b"original-content-x"):
    """A runner-shaped agent carrying one declared upload on a file field."""
    from unittest.mock import Mock

    from jev_ultrafast.policy import DefaultActionPolicy

    src = tmp_path / "attach.pdf"
    src.write_bytes(payload)
    a = loop.Agent.__new__(loop.Agent)
    a.screenshots = False
    a.pending_text = None
    a.input_guarantee = "atomic"
    a.policy = DefaultActionPolicy()
    a.uploads = loop._resolve_uploads([str(src)])
    p = base_page(
        actions=[
            {"id": "u1", "kind": "upload", "node": 5, "role": "button",
             "label": "Attach", "ctx": {"field": "file"}},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ]
    )
    a.state = {
        "browser": Mock(fresh=Mock(return_value=True), observe=Mock(return_value=p)),
        "page": p,
        "decision": _upload_decision(),
        "goal": "Send the file",
        "history": [],
        "decisions": [],
        "status": "predicted",
        "started_at": time.perf_counter(),
        "record": False,
        "text_calls": [],
    }
    return a, src


def test_upload_approval_digest_binds_content_not_path(tmp_path):
    """The pending grant's payload_digest covers the staged content hash —
    replacing the declared file after approval changes nothing the grant
    covers, and dispatch sends the staged snapshot."""
    import hashlib

    a, src = _upload_agent(tmp_path)
    p = a.state["page"]
    state = a.command("act", {"fingerprint": p["fingerprint"]})
    assert state["status"] == "approval_required"
    entry = a.uploads[0]
    expected = hashlib.sha256(
        f"jev-upload/v1\n{entry['content_sha256']}\n{entry['size']}\n{entry['name']}".encode()
    ).hexdigest()
    assert state["pending_approval"]["payload_digest"] == expected

    # Same-size replacement of the declared file: the grant still covers the
    # ORIGINAL bytes — digest identical, staged snapshot untouched.
    src.write_bytes(b"swapped-content-yy")
    a.state["decision"] = _upload_decision()
    a.state["status"] = "predicted"
    again = a.command("act", {"fingerprint": p["fingerprint"]})
    assert again["status"] == "approval_required"
    assert again["pending_approval"]["payload_digest"] == expected

    a.command("approve")
    act_call = a.state["browser"].act.call_args
    assert act_call.kwargs["file_path"] == entry["staged_path"]


def test_staged_upload_tampering_fails_closed(tmp_path):
    """If the staged snapshot itself is altered, the re-hash at decision time
    raises before the grant comparison — dispatch never runs."""
    a, _src = _upload_agent(tmp_path)
    p = a.state["page"]
    Path(a.uploads[0]["staged_path"]).write_bytes(b"tampered-bytes!!!")
    with pytest.raises(ValueError, match="staged upload content changed"):
        a.command("act", {"fingerprint": p["fingerprint"]})
    a.state["browser"].act.assert_not_called()


def test_classify_effect_new_kinds():
    from jev_ultrafast.policy import Effect, classify_effect
    assert classify_effect({"kind": "key", "key": "PageDown"}) == Effect.OBSERVE
    assert classify_effect({"kind": "key", "key": "Home"}) == Effect.OBSERVE
    # A non-whitelisted key is never silently autonomous.
    assert classify_effect({"kind": "key", "key": "Enter"}) == Effect.UNKNOWN_COMMIT
    assert classify_effect({"kind": "key"}) == Effect.UNKNOWN_COMMIT
    assert classify_effect({"kind": "scroll", "node": 3}) == Effect.OBSERVE
    # Upload classifies as an edit that escalates on the file-field context.
    assert classify_effect({"kind": "upload", "ctx": {"field": "file"}}) == Effect.SUBMISSION
    assert classify_effect({"kind": "upload", "ctx": {}}) == Effect.FORM_EDIT


def test_fresh_guard_covers_new_node_kinds(monkeypatch):
    """fresh() must give scroll-node and upload actions the per-node guard,
    not just the page marker — a scrollable container mutating post-decision
    is a stale target."""
    calls = []

    class FakeBrowser(browser.Browser):
        def __init__(self):
            pass

        def _isolated(self, expression):
            calls.append(expression)
            return ["pk", "gr"]

    b = FakeBrowser()
    page = {"page_key": "pk", "marker": "m", "guards": {"7": "gr"}}
    assert b.fresh(page, {"kind": "scroll", "node": 7}) is True
    assert b.fresh(page, {"kind": "upload", "node": 7}) is True
    assert b.fresh(page, {"kind": "key", "key": "PageDown"}) is False  # marker != page marker
    # Node-bound actions compare against BOTH page key and node guard.
    assert b.fresh({"page_key": "other", "guards": {"7": "gr"}}, {"kind": "scroll", "node": 7}) is False
    assert b.fresh({"page_key": "pk", "guards": {"7": "changed"}}, {"kind": "scroll", "node": 7}) is False
