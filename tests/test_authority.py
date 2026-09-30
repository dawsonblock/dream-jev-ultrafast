"""v0.5 authority-integrity tests: effect classification, TOCTOU aborts,
torn-tail evidence recovery, and signing-key rotation."""

import json

import pytest

from jev_ultrafast import browser
from jev_ultrafast.browser import StalePage, browser_operation
from jev_ultrafast.dream import ExperienceStore
from jev_ultrafast.policy import DefaultActionPolicy, Effect, PolicyDecision, classify_effect
from jev_ultrafast.signing import EvidenceSigner, frame_digest, verify_signature


def click(label, **kw):
    return {"kind": "click", "label": label, **kw}


# --- deterministic effect classification ---------------------------------

@pytest.mark.parametrize("action,effect", [
    ({"kind": "wait"}, Effect.OBSERVE),
    ({"kind": "scroll"}, Effect.OBSERVE),
    (click("City", role="textbox"), Effect.FORM_EDIT),
    (click("Refundable", role="checkbox"), Effect.FORM_EDIT),
    (click("Plan A", role="radio"), Effect.FORM_EDIT),
    (click("Overview", role="tab"), Effect.NAVIGATE),
    (click("About", role="link"), Effect.NAVIGATE),
    (click("Sign in", role="link"), Effect.NAVIGATE),   # nav to login ≠ authenticating
    (click("Sign in", role="button"), Effect.AUTHENTICATE),
    (click("Search", role="button"), Effect.SEARCH),
    (click("Go", role="button"), Effect.SEARCH),
    (click("Show more", role="button"), Effect.OBSERVE),
    # The motivating case: a bare progression label is an opaque commit.
    (click("Continue", role="button"), Effect.UNKNOWN_COMMIT),
    (click("Yes", role="button"), Effect.UNKNOWN_COMMIT),
    (click("Flarble", role="button"), Effect.UNKNOWN_COMMIT),
    (click("Buy now", role="button"), Effect.PURCHASE),
    (click("Transfer funds", role="button"), Effect.FINANCIAL),
    (click("Delete account", role="button"), Effect.DELETE),
    (click("Send", role="button"), Effect.EXTERNAL_MESSAGE),
    (click("Allow notifications", role="button"), Effect.PERMISSION_CHANGE),
    # High-risk labels escalate even on links — structure cannot downgrade text.
    (click("Delete account", role="link"), Effect.DELETE),
    (click("Buy now", role="link"), Effect.PURCHASE),
])
def test_classify_effect_matrix(action, effect):
    assert classify_effect(action) == effect


@pytest.mark.parametrize("action,effect", [
    # Form submit membership: field inventory decides the class.
    (click("Continue", ctx={"form": True, "submit": True}), Effect.SUBMISSION),
    (click("Continue", ctx={"form": True, "submit": True, "fields": {"password": True}}),
     Effect.AUTHENTICATE),
    (click("Confirm", ctx={"form": True, "submit": True, "fields": {"money": True}}),
     Effect.PURCHASE),
    (click("Upload", ctx={"form": True, "submit": True, "fields": {"file": True}}),
     Effect.SUBMISSION),
    (click("Go", ctx={"form": True, "submit": True, "method": "get"}), Effect.SEARCH),
    (click("Go", ctx={"form": True, "submit": True, "fields": {"search": True}}), Effect.SEARCH),
    # Outbound channels.
    (click("Email us", role="link", ctx={"messaging": True}), Effect.EXTERNAL_MESSAGE),
    (click("Download report", role="link", ctx={"download": True}), Effect.SUBMISSION),
    (click("Partner site", role="link", ctx={"external": True}), Effect.NAVIGATE),
    (click("Share externally", role="link", ctx={"external": True}), Effect.EXTERNAL_MESSAGE),
    # Modal scope: dismiss autonomous, anything else is an unknown commit.
    (click("Cancel", ctx={"modal": True}), Effect.OBSERVE),
    (click("Not now", ctx={"modal": True}), Effect.OBSERVE),
    (click("OK", ctx={"modal": True}), Effect.UNKNOWN_COMMIT),
    (click("Continue", ctx={"modal": True}), Effect.UNKNOWN_COMMIT),
])
def test_classify_effect_structural_ctx(action, effect):
    assert classify_effect(action) == effect


@pytest.mark.parametrize("action,level", [
    (click("Continue", role="button"), "require_approval"),
    (click("Buy now", role="button"), "require_approval"),
    (click("About", role="link"), "allow"),
    (click("Refundable", role="checkbox"), "allow"),
    (click("Cancel", ctx={"modal": True}), "allow"),
    ({"kind": "fill", "label": "Email"}, "allow"),
    ({"kind": "wait"}, "allow"),
])
def test_authority_floor(action, level):
    decision = DefaultActionPolicy().assess(action)
    assert decision.level == level
    assert decision.effect


def test_legacy_text_backstop_still_escalates():
    # Context-free "place order" hits the legacy pattern list even without ctx.
    decision = DefaultActionPolicy().assess(click("Place order"))
    assert decision.level == "require_approval"


def test_adviser_escalates_but_never_downgrades():
    cautious = DefaultActionPolicy(adviser=lambda *a, **k: PolicyDecision("require_approval", "ml"))
    assert cautious.assess(click("About", role="link")).level == "require_approval"

    denying = DefaultActionPolicy(adviser=lambda *a, **k: "deny")
    assert denying.assess(click("About", role="link")).level == "deny"

    # An adviser verdict of "allow" cannot soften a deterministic approval floor.
    optimistic = DefaultActionPolicy(adviser=lambda *a, **k: "allow")
    assert optimistic.assess(click("Buy now", role="button")).level == "require_approval"
    assert optimistic.assess(click("Continue", role="button")).level == "require_approval"


# --- TOCTOU: press/release revalidation ------------------------------------

def _op(action, text=None):
    request = {
        "operation": "act",
        "session": "s",
        "context_id": 42,
        "expected": {"page_key": [1], "guard": [1]},
        "action": action,
    }
    if text is not None:
        request["text"] = text
    return request


def _fake_cdp(calls, evaluate_results):
    sequence = iter(evaluate_results)

    def fake(method, session_id=None, **kwargs):
        calls.append((method, kwargs))
        if method == "Runtime.evaluate":
            return {"result": {"value": next(sequence)}}
        return {}

    return fake


def test_pre_press_identity_failure_blocks_input(monkeypatch):
    calls = []
    # resolve → {x,y}; pre-press → False
    monkeypatch.setattr(browser, "cdp", _fake_cdp(calls, [{"x": 5, "y": 5}, False]))
    with pytest.raises(StalePage, match="pre-press"):
        browser_operation(_op({"id": "e", "kind": "click", "node": 1}))
    assert not any(m == "Input.dispatchMouseEvent" for m, _ in calls)


def test_pre_release_failure_still_releases_pointer(monkeypatch):
    calls = []
    # resolve → {x,y}; pre-press → True; pre-release → False
    monkeypatch.setattr(browser, "cdp", _fake_cdp(calls, [{"x": 5, "y": 5}, True, False]))
    with pytest.raises(StalePage, match="press and release"):
        browser_operation(_op({"id": "e", "kind": "click", "node": 1}))
    events = [kw.get("type") for m, kw in calls if m == "Input.dispatchMouseEvent"]
    # Both halves of the gesture were dispatched: press landed, the release is
    # sent for input-state hygiene, and the operation still reports aborted.
    assert events == ["mousePressed", "mouseReleased"]


def test_fill_atomic_insert_paths(monkeypatch):
    calls = []
    # resolve, pre-press, pre-release all pass; insert reports focus failure.
    fake = _fake_cdp(calls, [{"x": 5, "y": 5}, True, True, {"error": "focus"}])
    monkeypatch.setattr(browser, "cdp", fake)
    with pytest.raises(StalePage, match="failed before mutation"):
        browser_operation(_op({"id": "e", "kind": "fill", "node": 1}, text="typed"))
    assert not any(m == "Input.insertText" for m, _ in calls)

    calls.clear()
    # Insert ran but the landed value differs → non-retryable mutation error.
    fake = _fake_cdp(calls, [{"x": 5, "y": 5}, True, True, {"inserted": True, "matched": False, "value": "x"}])
    monkeypatch.setattr(browser, "cdp", fake)
    with pytest.raises(RuntimeError, match="landed differently"):
        browser_operation(_op({"id": "e", "kind": "fill", "node": 1}, text="typed"))

    calls.clear()
    # Happy path: everything happens in-world; no CDP key/text dispatch at all.
    fake = _fake_cdp(calls, [{"x": 5, "y": 5}, True, True,
                             {"inserted": True, "matched": True, "value": "typed"}])
    monkeypatch.setattr(browser, "cdp", fake)
    browser_operation(_op({"id": "e", "kind": "fill", "node": 1}, text="typed"))
    assert not any(m.startswith("Input.") for m, _ in calls if m != "Input.dispatchMouseEvent")


# --- evidence store: torn tail, rotation, domain separation -----------------

def _minimal_event():
    return {"run_id": "r", "task_key": "t", "sequence": 1, "event": "run_finished",
            "status": "done", "verified": True}


def test_torn_tail_recovers_and_append_seals(tmp_path):
    path = tmp_path / "store.jsonl"
    store = ExperienceStore(path)
    store.append(_minimal_event())
    store.append(_minimal_event())
    # Simulate a crash mid-write: a recognizable event prefix, no newline.
    with path.open("a") as handle:
        handle.write('{"schema":"jev-dre')
    assert len(ExperienceStore(path).load()) == 2
    assert ExperienceStore(path).verify()["torn_tail_recovered"] is True
    # Append repairs the fragment and continues the chain on the last good event.
    ExperienceStore(path).append(_minimal_event())
    events = ExperienceStore(path, strict=True).load()
    assert len(events) == 3
    assert path.read_bytes().endswith(b"\n")


def test_corrupt_tail_and_mid_chain_stay_fatal(tmp_path):
    path = tmp_path / "store.jsonl"
    store = ExperienceStore(path)
    store.append(_minimal_event())
    # Garbage tail that is not an event prefix: fail closed.
    with path.open("ab") as handle:
        handle.write(b'garbage-without-newline')
    with pytest.raises(ValueError, match="Invalid DREAM event JSON|Unrecoverable"):
        ExperienceStore(path).load()

    # A complete but hash-invalid record is tampering, not a torn write.
    path2 = tmp_path / "store2.jsonl"
    store2 = ExperienceStore(path2)
    store2.append(_minimal_event())
    lines = path2.read_text().splitlines()
    bad = json.loads(lines[-1])
    bad["verified"] = False  # hash no longer matches
    path2.write_text(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match="integrity failure"):
        ExperienceStore(path2).load()


def test_complete_record_missing_newline_is_sealed(tmp_path):
    path = tmp_path / "store.jsonl"
    store = ExperienceStore(path)
    store.append(_minimal_event())
    data = path.read_bytes()
    path.write_bytes(data[:-1])  # strip only the trailing newline
    assert len(ExperienceStore(path).load()) == 1
    ExperienceStore(path).append(_minimal_event())
    assert len(ExperienceStore(path, strict=True).load()) == 2


def test_key_rotation_and_domain_separation(tmp_path):
    old_key, new_key = EvidenceSigner.from_hex("ab" * 32), EvidenceSigner.from_hex("cd" * 32)
    path = tmp_path / "rot.jsonl"
    ExperienceStore(path, signer=old_key).append(_minimal_event())
    ExperienceStore(path, signer=new_key).append(_minimal_event())

    # The rotated key set accepts both segments; either key alone does not.
    events = ExperienceStore(path, verify_keys={old_key.key_id, new_key.key_id}).load()
    assert len(events) == 2
    with pytest.raises(ValueError, match="unexpected key"):
        ExperienceStore(path, verify_keys={old_key.key_id}).load()

    # Domain separation: a signature over the raw digest bytes is not an
    # evidence signature.
    raw_sig = new_key._key.sign(bytes.fromhex("ab" * 32)).hex()
    assert not verify_signature(new_key.key_id, "ab" * 32, raw_sig)
    assert verify_signature(new_key.key_id, "ab" * 32, new_key._key.sign(frame_digest("ab" * 32)).hex())


def test_cli_reads_signed_store_with_rotation_flags(tmp_path):
    from jev_ultrafast.dream_cli import main

    signer = EvidenceSigner.from_hex("ab" * 32)
    path = tmp_path / "signed.jsonl"
    ExperienceStore(path, signer=signer).append(_minimal_event())

    assert main(["verify", str(path), "--verify-keys", signer.key_id]) == 0
    # Without a configured key the same signed store fails closed.
    with pytest.raises(ValueError, match="no verification key"):
        main(["verify", str(path)])
    with pytest.raises(ValueError, match="unexpected key"):
        main(["verify", str(path), "--verify-keys", "cd" * 32])
