"""Regression tests for v0.2 execution-integrity and policy boundaries."""

import re
from pathlib import Path
from unittest.mock import Mock

import pytest

from jev_ultrafast import agent as loop
from jev_ultrafast import browser, model
from jev_ultrafast.browser import StalePage, browser_operation, fingerprint
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
    calls = []

    def fake_cdp(method, session_id=None, **kwargs):
        calls.append((method, kwargs))
        if method == "Runtime.evaluate":
            runtime_count = sum(1 for name, _ in calls if name == "Runtime.evaluate")
            if runtime_count == 1:
                return {"result": {"value": {"x": 10, "y": 10}}}
            return {"result": {"value": False}}
        return {}

    monkeypatch.setattr(browser, "cdp", fake_cdp)
    with pytest.raises(StalePage, match="lost focus"):
        browser_operation({
            "operation": "act",
            "session": "s",
            "context_id": 42,
            "expected": {"page_key": [1], "guard": [1]},
            "action": {"id": "e1", "kind": "fill", "node": 1},
            "text": "secret",
        })
    assert not any(name == "Input.insertText" for name, _ in calls)


def test_duplicate_select_values_keep_distinct_exact_option_indices():
    actions = [
        {"id": "e1", "kind": "select", "node": 7, "role": "combobox", "label": "Plan → Standard",
         "value": "x", "current_value": "", "option_index": 1, "option_label": "Standard"},
        {"id": "e2", "kind": "select", "node": 7, "role": "combobox", "label": "Plan → Premium",
         "value": "x", "current_value": "", "option_index": 2, "option_label": "Premium"},
    ]
    elements, targets, _ = model.action_space(actions)
    assert [o["option_index"] for o in elements[0]["options"]] == [1, 2]
    assert targets["SELECT"]["1:1"]["option_index"] == 1
    assert targets["SELECT"]["1:2"]["option_index"] == 2


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


def test_select_guard_normalizes_option_value_like_snapshot():
    """snapshot.js records clip(o.value): whitespace-normalized. The guarded
    select must apply identical normalization or a value with irregular
    spacing can never match and retries forever."""
    src = Path(browser.__file__).read_text()
    assert "String(o.value).replace(" in src
    assert "String(o.label).replace(" in src
