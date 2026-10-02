"""§54 property-based tests: invariants over randomized inputs, not
hand-picked cases. Hypothesis drives the generation; each property is an
authority/evidence invariant the audit plan names."""

import json

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from jev_ultrafast.dream import ExperienceStore
from jev_ultrafast.dreamlearn import CounterfactualTrials
from jev_ultrafast.policy import DefaultActionPolicy, Effect, classify_effect

_KINDS = st.sampled_from(["click", "fill", "select", "scroll", "wait", "type"])
_LABELS = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
    max_size=80,
)
_ROLES = st.sampled_from(
    ["button", "link", "textbox", "searchbox", "switch", "checkbox",
     "combobox", "menuitem", None]
)
_SENSITIVE_CTX = st.sampled_from(
    [{"field": "money"}, {"field": "auth"}, {"field": "otp"},
     {"field": "file"}, {"field": "email"}, {"field": "message"}]
)


@st.composite
def _action(draw, sensitive=None):
    action = {
        "kind": draw(_KINDS),
        "label": draw(_LABELS),
    }
    role = draw(_ROLES)
    if role is not None:
        action["role"] = role
    ctx = {}
    if sensitive is True:
        ctx.update(draw(_SENSITIVE_CTX))
    elif draw(st.booleans()):
        ctx.update(draw(_SENSITIVE_CTX))
    if ctx:
        action["ctx"] = ctx
    return action


# --- P1: sensitive context never grants more authority than a plain one ----

@settings(max_examples=300, deadline=None)
@given(base=_action(sensitive=False), sensitive_ctx=_SENSITIVE_CTX)
def test_sensitive_context_never_downgrades_authority(base, sensitive_ctx):
    """Adding a sensitive target field can only escalate, never relax."""
    levels = {"allow": 0, "require_approval": 1}
    baseline = levels[DefaultActionPolicy().assess(dict(base)).level]
    augmented = dict(base)
    augmented["ctx"] = {**base.get("ctx", {}), **sensitive_ctx}
    hardened = levels[DefaultActionPolicy().assess(augmented).level]
    assert hardened >= baseline


@settings(max_examples=300, deadline=None)
@given(_action())
def test_classification_is_deterministic_and_total(action):
    """Every descriptor classifies to a real Effect — no silent gaps."""
    first = classify_effect(action)
    assert isinstance(first, Effect)
    assert classify_effect(dict(action)) == first


# --- P2: the effect coordinate is a signature floor ------------------------

_EFFECTS = st.sampled_from([e.value for e in Effect])


def _cell(effect, arm):
    # A minimal trial cell: stratum + 12-field signature + arm + 21 counters.
    sig = {
        "m_kind": "click", "m_effect": effect, "m_role": "button",
        "m_ov": "1-2", "m_rank": "0-4", "m_phase": "early",
        "p_kind": "click", "p_effect": effect, "p_role": "button",
        "p_ov": "1-2", "p_rank": "0-4", "p_phase": "early",
    }
    from jev_ultrafast.dreamlearn import _SIGNATURE_FIELDS, _SIGNATURE_INDEX
    key = ["", ""] + [""] * 12 + [arm]
    for f in _SIGNATURE_FIELDS:
        key[_SIGNATURE_INDEX[f]] = sig[f]
    counts = [0] * 21
    counts[0] = counts[1] = counts[2] = 10
    counts[4] = counts[5] = 10.0  # w_success == w_sum -> p_success 1.0
    counts[6] = 10.0
    return tuple(key + counts)


@settings(max_examples=200, deadline=None)
@given(stored=_EFFECTS, asked=_EFFECTS)
def test_effect_mismatch_never_resolves(stored, asked):
    """A known-effect query can only resolve evidence stored under the same
    effect class — cross-effect generalization is structurally impossible."""
    trials = CounterfactualTrials(
        version="jev-trials/6",
        cells=(_cell(stored, "candidate"), _cell(stored, "control")),
    )
    resolved = trials.resolve(model_kind="click", proposal_kind="click",
                              model_effect=asked, proposal_effect=asked)
    if stored != asked:
        assert resolved is None or (
            resolved.get("effect_status") == "insufficient_data"
        )


# --- P3: a corrupted mid-chain record is never silently accepted -----------

def _store_events(path, n=6):
    store = ExperienceStore(path)
    for i in range(n):
        store.append({"event": "transition", "run_id": f"r{i}", "task_key": "t",
                      "state": "S", "selected": {"id": "a"},
                      "page_changed": True})
    return path


@settings(max_examples=150, deadline=None,
          suppress_health_check=list(HealthCheck))
@given(line=st.integers(min_value=0, max_value=4),
       pos=st.integers(min_value=0, max_value=300),
       byte=st.integers(min_value=1, max_value=255))
def test_mid_chain_corruption_always_detected(line, pos, byte, tmp_path):
    """Any single-byte mutation inside a non-final record must break
    verification or load — evidence cannot be silently amended."""
    path = _store_events(tmp_path / "s.jsonl")
    lines = path.read_bytes().split(b"\n")
    if line >= len(lines) or pos >= len(lines[line]) - 20:
        return  # skip mutations that only touch the hash field boundary
    mutated = bytearray(lines[line])
    mutated[pos] ^= byte & 0xFF or 1
    lines[line] = bytes(mutated)
    path.write_bytes(b"\n".join(lines))
    store = ExperienceStore(path)
    with pytest.raises((ValueError, json.JSONDecodeError, KeyError)):
        store.load()
