"""v0.5 authority-integrity tests: effect classification, TOCTOU aborts,
torn-tail evidence recovery, and signing-key rotation."""

import json
from pathlib import Path

import pytest

from jev_ultrafast import browser
from jev_ultrafast.browser import IndeterminateMutation, StalePage, browser_operation
from jev_ultrafast.dream import ExperienceStore
from jev_ultrafast.policy import (
    DefaultActionPolicy,
    Effect,
    PolicyDecision,
    assess_payload,
    classify_effect,
    classify_payload,
)
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
    ({"kind": "fill", "label": "Email"}, "require_approval"),  # contact data is a disclosure
    ({"kind": "fill", "label": "Search"}, "allow"),
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

def _op(action, text=None, guarantee="trusted"):
    request = {
        "operation": "act",
        "session": "s",
        "context_id": 42,
        "expected": {"page_key": [1], "guard": [1]},
        "action": action,
        "guarantee": guarantee,
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


def test_pre_release_failure_is_indeterminate_and_releases_pointer(monkeypatch):
    calls = []
    # resolve → {x,y}; pre-press → True; pre-release → False
    monkeypatch.setattr(browser, "cdp", _fake_cdp(calls, [{"x": 5, "y": 5}, True, False]))
    with pytest.raises(IndeterminateMutation, match="indeterminate"):
        browser_operation(_op({"id": "e", "kind": "click", "node": 1}))
    events = [kw.get("type") for m, kw in calls if m == "Input.dispatchMouseEvent"]
    # Both halves of the gesture were dispatched: press landed, the release is
    # sent for input-state hygiene under finally, and the outcome is reported
    # indeterminate — the press may already have run, so it is not retryable.
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


def test_append_tail_check_never_reads_whole_file(tmp_path, monkeypatch):
    """Scale guard (qualification §52): the per-append torn-tail check inspects
    only the tail bytes. A whole-file ``read_bytes`` made every append O(log
    size) and the store O(n²) — ~4.5 min for 100k events where the fast path
    is under 30s."""
    path = tmp_path / "bounded-tail.jsonl"
    store = ExperienceStore(path)
    store.append(_minimal_event())
    store.append(_minimal_event())
    original = Path.read_bytes

    def no_full_reads(self):  # Path.read_bytes reads the whole file by design
        raise AssertionError("append performed a whole-file read")

    monkeypatch.setattr(Path, "read_bytes", no_full_reads)
    store.append(_minimal_event())  # healthy tail: last-byte check only
    monkeypatch.setattr(Path, "read_bytes", original)

    # Boundary cases still work: a fragment spanning chunk reads and a tail
    # that IS the whole file (no newline at all).
    path.write_bytes(path.read_bytes()[:-1])  # complete record, newline lost
    monkeypatch.setattr(Path, "read_bytes", no_full_reads)
    store.append(_minimal_event())  # backward fragment scan + separator seal
    monkeypatch.setattr(Path, "read_bytes", original)
    assert len(ExperienceStore(path, strict=True).load()) == 4


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


# --- classifier monotonicity: no structural shortcut may bypass risk ------

def test_sensitive_fill_escalates_past_form_edit():
    """The audit's case: a credit-card fill must not be classified FORM_EDIT."""
    action = {
        "kind": "fill", "label": "Credit card number", "role": "textbox",
        "ctx": {"form": True, "fields": {"money": True}},
    }
    result = DefaultActionPolicy().assess(action)
    assert result.level == "require_approval"
    assert classify_effect(action) in {Effect.FINANCIAL, Effect.DISCLOSURE}

    # A benign-looking fill on a target that IS a credential field.
    pw = {"kind": "fill", "label": "Enter", "role": "textbox",
          "ctx": {"field": "auth"}}
    assert classify_effect(pw) == Effect.AUTHENTICATE

    # The audit's toggle case: "Share publicly" must not ride the FORM_EDIT
    # shortcut even though the role alone would be a reversible edit.
    toggle = {"kind": "click", "label": "Share publicly", "role": "switch"}
    assert classify_effect(toggle) == Effect.EXTERNAL_MESSAGE


def test_monotonic_classifier_never_returns_below_the_floor():
    """Every candidate effect meets or exceeds the structural floor — nothing
    in the classifier can downgrade a dangerous label or context."""
    floor = DefaultActionPolicy().assess({"kind": "fill", "label": "City"})
    assert floor.effect == Effect.FORM_EDIT
    for extra in (
        {"ctx": {"field": "money"}},   # the target itself is a payment field
        {"ctx": {"field": "email"}},   # contact data: disclosure
        {"ctx": {"field": "auth"}},    # credential field
        {"label": "Share publicly"},
        {"label": "Pay now"},
    ):
        action = {"kind": "fill", "label": "City", **extra}
        assert DefaultActionPolicy().assess(action).level == "require_approval", extra


# --- privacy: labels and option labels are external-facing text ------------

def test_action_labels_and_options_are_redacted_before_model_boundary():
    from jev_ultrafast.privacy import sanitize_action

    action = {
        "kind": "fill",
        "label": "Card for dawson@example.com",
        "option_label": "Card 4111 1111 1111 1111",
        "ctx": {"field": "money"},
    }
    clean = sanitize_action(action)
    assert "dawson@example.com" not in clean["label"]
    assert "4111" not in clean["option_label"]
    # Structural authority context survives — redaction is text-only.
    assert clean["ctx"]["field"] == "money"
    # And the policy still classifies the ORIGINAL action, not the redacted one.
    assert DefaultActionPolicy().assess(action).level == "require_approval"


# --- candidate catalogue digest binds every replay-relevant field ----------

def test_catalogue_digest_binds_replay_relevant_fields():
    from jev_ultrafast.dream import candidate_catalog_digest

    base = [{
        "id": "a", "kind": "select", "node": 3, "role": "combobox",
        "label": "Plan", "value": "x", "current_value": "",
        "option_index": 1, "option_label": "Standard", "checked": None,
        "ctx": {"form": True}, "goal_overlap": 1,
    }]
    digest = candidate_catalog_digest(base)
    for field, value in (
        ("node", 999),            # duplicate_node_cap groups by node
        ("role", "listbox"),
        ("option_index", 2),      # select replay uses the exact index
        ("option_label", "Premium"),
        ("current_value", "x"),
        ("checked", True),
        ("ctx", {"form": False}),
        ("goal_overlap", 0),
        ("kind", "click"),
    ):
        mutated = [dict(base[0], **{field: value})]
        assert candidate_catalog_digest(mutated) != digest, field


# --- chain-head anchoring ---------------------------------------------------

def _two_event_store(tmp_path, signer, anchor=None, name="anchored.jsonl"):
    path = tmp_path / name
    store = ExperienceStore(
        path, signer=signer, verify_keys={signer.key_id},
        anchor_path=tmp_path / "store.head" if anchor is None else anchor,
    )
    store.append(_minimal_event())
    store.append({**_minimal_event(), "sequence": 2})
    return path, store


def test_chain_head_anchor_detects_truncated_tail(tmp_path):
    signer = EvidenceSigner.from_hex("11" * 32)
    path, _ = _two_event_store(tmp_path, signer)
    # Attacker deletes the last signed event entirely.
    data = path.read_bytes()
    path.write_bytes(data[: data.index(b"\n") + 1])
    reader = ExperienceStore(
        path, verify_keys={signer.key_id}, anchor_path=tmp_path / "store.head")
    with pytest.raises(ValueError, match="anchor"):
        reader.load()


def test_anchor_detects_signed_tail_disguised_as_torn_write(tmp_path):
    """The audit's attack: replace the last SIGNED event with something that
    looks like a crash-torn fragment. Recovery would silently drop it; the
    anchor exposes the missing head."""
    signer = EvidenceSigner.from_hex("22" * 32)
    path, _ = _two_event_store(tmp_path, signer)
    first = path.read_bytes().split(b"\n")[0]
    path.write_bytes(first + b'\n{"event":"B')   # torn-looking tail
    reader = ExperienceStore(
        path, verify_keys={signer.key_id}, anchor_path=tmp_path / "store.head")
    with pytest.raises(ValueError, match="anchor"):
        reader.load()
    # And a writer cannot append past the gap to silently re-anchor.
    writer = ExperienceStore(
        path, signer=signer, verify_keys={signer.key_id},
        anchor_path=tmp_path / "store.head")
    with pytest.raises(ValueError, match="anchor"):
        writer.append({**_minimal_event(), "sequence": 3})


def test_missing_anchor_and_forged_anchor_fail_closed(tmp_path):
    signer = EvidenceSigner.from_hex("33" * 32)
    path = tmp_path / "unsigned-anchor-store.jsonl"
    store = ExperienceStore(path, signer=signer)
    store.append(_minimal_event())
    # Anchor configured after the fact but never written.
    reader = ExperienceStore(
        path, verify_keys={signer.key_id}, anchor_path=tmp_path / "missing.head")
    with pytest.raises(ValueError, match="anchor missing"):
        reader.load()

    path2, _ = _two_event_store(tmp_path, signer, anchor=tmp_path / "s2.head")
    anchor = json.loads((tmp_path / "s2.head").read_text())
    anchor["sequence"] = 99  # forged checkpoint
    (tmp_path / "s2.head").write_text(json.dumps(anchor))
    with pytest.raises(ValueError, match="anchor"):
        ExperienceStore(
            path2, verify_keys={signer.key_id},
            anchor_path=tmp_path / "s2.head").load()


def test_consistent_anchor_passes_and_reanchor_restores(tmp_path):
    signer = EvidenceSigner.from_hex("44" * 32)
    path, store = _two_event_store(tmp_path, signer)
    reader = ExperienceStore(
        path, verify_keys={signer.key_id}, anchor_path=tmp_path / "store.head")
    report = reader.verify()
    assert report["anchor"]["consistent"] and report["anchor"]["signed"]
    assert report["anchor"]["sequence"] == 2

    # After confirmed truncation the operator re-anchors explicitly.
    data = path.read_bytes()
    path.write_bytes(data[: data.index(b"\n") + 1])
    writer = ExperienceStore(
        path, signer=signer, verify_keys={signer.key_id},
        anchor_path=tmp_path / "store.head")
    with pytest.raises(ValueError, match="anchor"):
        writer.load()
    writer.reanchor()
    assert ExperienceStore(
        path, verify_keys={signer.key_id}, anchor_path=tmp_path / "store.head"
    ).load()[0]["event"] == "run_finished"


# --- unknown kinds and the payload authority signal ------------------------

@pytest.mark.parametrize("kind", ["upload", "script", "future_mutation", "drag", "hotkey"])
def test_unknown_mutation_kinds_fail_closed(kind):
    """Audit regression: an unrecognized kind classifies UNKNOWN_COMMIT and
    must escalate — the v0.6 kind shortcut let it pass as ``allow``."""
    decision = DefaultActionPolicy().assess({"kind": kind, "label": "Anything"})
    assert classify_effect({"kind": kind, "label": "Anything"}) == Effect.UNKNOWN_COMMIT
    assert decision.level == "require_approval"
    assert decision.effect == Effect.UNKNOWN_COMMIT


def test_only_non_mutating_kinds_stay_autonomous():
    policy = DefaultActionPolicy()
    assert policy.assess({"kind": "wait"}).level == "allow"
    assert policy.assess({"kind": "scroll"}).level == "allow"


def test_payload_classification_is_a_second_authority_signal():
    """The target's self-description is page-controlled; the generated text is
    the second, independent signal — sensitive content cannot hide behind a
    benign field label."""
    assert classify_payload("4111 1111 1111 1111") == Effect.FINANCIAL
    assert classify_payload("dawson@example.com") == Effect.DISCLOSURE
    assert classify_payload("sk-ABCDEFGHIJKLMNOPQRST") == Effect.DISCLOSURE
    assert classify_payload("123-45-6789") == Effect.DISCLOSURE
    assert classify_payload(
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3OCJ9.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJVadQssw5c"
    ) == Effect.DISCLOSURE
    assert classify_payload("Zurich") == Effect.OBSERVE
    assert classify_payload("") == Effect.OBSERVE
    assert classify_payload(None) == Effect.OBSERVE
    assert assess_payload("4111 1111 1111 1111").level == "require_approval"
    assert assess_payload("dawson@example.com").level == "require_approval"
    assert assess_payload("Zurich").level == "allow"
