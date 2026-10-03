"""_learning.signatures — extracted from jev_ultrafast.dreamlearn."""

from __future__ import annotations

from .common import _stable_hash

__all__ = [
    'TERMINATION_REASONS',
    '_OVERLAP_BUCKETS',
    '_PHASE_BUCKETS',
    '_RANK_BUCKETS',
    '_REASON_INDEX',
    '_SIGNATURE_DROP_FIELDS',
    '_SIGNATURE_FIELDS',
    '_SIGNATURE_INDEX',
    '_bucket',
    '_coord',
    '_effect_bucket',
    '_overlap_of',
    '_rank_of',
    '_role_bucket',
    '_signature_masks',
    '_termination_reason',
    '_trial_context',
    'action_key',
    'overlap_bucket',
    'phase_bucket',
    'rank_bucket',
]


_OVERLAP_BUCKETS = ("0", "1", "2", "3+")


_RANK_BUCKETS = ("0-4", "5-19", "20-99", "100+")


def _bucket(value: int, edges: tuple[int, ...], labels: tuple[str, ...]) -> str:
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def overlap_bucket(overlap: int) -> str:
    return _bucket(max(0, int(overlap)), (1, 2, 3), _OVERLAP_BUCKETS)


def rank_bucket(rank: int | None) -> str:
    if rank is None or rank < 0:
        return "unknown"
    return _bucket(rank, (5, 20, 100), _RANK_BUCKETS)


TERMINATION_REASONS = (
    "operator_cancel",
    "browser_crash",
    "agent_exception",
    "network_failure",
    "timeout",
    "recorder_shutdown",
    "indeterminate_execution",
    "unverified_claim",
    "unknown_abort",
)


_REASON_INDEX = {reason: index for index, reason in enumerate(TERMINATION_REASONS)}


def _termination_reason(final: dict | None) -> str:
    """Why a censored run produced no measured outcome.

    New evidence records the reason on ``run_finished`` (``operator_cancel``
    for a session closed mid-run, the exception class for a crash/timeout/
    network failure, ``recorder_shutdown`` when the store itself failed).
    Older evidence without a reason is ``unknown_abort``; a run with no
    ``run_finished`` at all is ``unknown_abort`` too — the recording was
    interrupted, and pretending to know why would be invention.
    """
    if final is None:
        return "unknown_abort"
    status = str(final.get("status") or "")
    if status == "claimed_done":
        return "unverified_claim"
    reason = str(final.get("reason") or "").strip().lower()
    return reason if reason in _REASON_INDEX else "unknown_abort"


_PHASE_BUCKETS = ("0-2", "3-9", "10+", "unknown")


def phase_bucket(step) -> str:
    """Workflow-phase bucket of a step index — early/mid/late/unknown."""
    if step is None:
        return "unknown"
    try:
        value = max(0, int(step))
    except (TypeError, ValueError):
        return "unknown"
    return _bucket(value, (3, 10), _PHASE_BUCKETS)


def _coord(value) -> str:
    """A trial-context coordinate, safe for the ``|``-joined context key.

    Page metadata is attacker-shaped input: a role like ``"a|b"`` would let two
    distinct context tuples collide under ``"|".join`` — silently merging two
    cells into one estimate key. The join delimiter can never survive into a
    stored coordinate.
    """
    text = str(value).replace("|", " ").strip()
    return text if text else "unknown"


def _role_bucket(role) -> str:
    return _coord(str(role or "").lower())


def _effect_bucket(effect) -> str:
    """Effect class of an action, as the deterministic classifier names it."""
    if effect is None:
        return "unknown"
    return _coord(str(getattr(effect, "value", effect)).lower())


_SIGNATURE_FIELDS = (
    "m_kind", "m_effect", "m_role", "m_ov", "m_rank", "m_phase",
    "p_kind", "p_effect", "p_role", "p_ov", "p_rank", "p_phase",
)


def _overlap_of(meta_key: str, cand_id_key: str, meta: dict, transition: dict | None) -> str:
    """Overlap bucket for a context member, from the assignment record first.

    Assignment meta carries the pre-treatment overlap directly; for records
    that lack it, resolve the candidate's recorded ``goal_overlap`` from the
    executed transition's catalogue by id.
    """
    raw = meta.get(meta_key)
    if raw is not None:
        try:
            return overlap_bucket(int(raw))
        except (TypeError, ValueError):
            pass
    if transition is not None:
        cand = next(
            (
                c
                for c in transition.get("candidates") or ()
                if c.get("id") == meta.get(cand_id_key)
            ),
            None,
        )
        if cand is not None:
            try:
                return overlap_bucket(int(cand.get("goal_overlap", 0) or 0))
            except (TypeError, ValueError):
                pass
    return "unknown"


def _rank_of(meta: dict, key: str) -> str:
    raw = meta.get(key)
    if raw is None:
        return "unknown"
    try:
        return rank_bucket(int(raw))
    except (TypeError, ValueError):
        return "unknown"


def _trial_context(meta: dict, transition: dict | None) -> tuple[str, ...]:
    """The cell a trial belongs to — 14 coordinates, no information collapsed.

    ``(task_family, site, m_kind, m_effect, m_role, m_ov, m_rank, m_phase,
    p_kind, p_effect, p_role, p_ov, p_rank, p_phase)``.

    Task family and site are *separate* coordinates: a trial carrying both must
    stay indexable under either (the v0.8.3 ``scope`` collapsed them into one
    mutually exclusive key, so a family-carrying trial lost its site stratum
    and resolution could never fall back to it). Resolution walks
    family+site → site → family → pooled over these stored cells.

    The remaining coordinates are the bounded *treatment signature* of the
    pre-treatment divergence: the model choice's kind/effect-class/role/
    overlap/offered-rank/phase and the proposal's same six. Keying on the
    divergence (not the executed action) keeps both arms of one assignment in
    one cell. Assignment events carry every coordinate directly; the
    transition's selected action and candidate list are fallbacks for older
    records, and anything still unknown stays ``"unknown"`` — a wildcard
    dimension during resolution, never a fabricated value.
    """
    family = str(meta.get("task_family") or "").replace("|", " ").strip().lower()
    site = str(meta.get("site") or "").replace("|", " ").strip().lower()
    selected = (transition.get("selected") or {}) if transition is not None else {}
    m_kind = str(meta.get("model_choice_kind") or "unknown").replace("|", " ")
    if m_kind == "unknown":
        m_kind = str(selected.get("kind") or "unknown").replace("|", " ")
    p_kind = str(meta.get("proposal_kind") or "unknown").replace("|", " ")
    if p_kind == "unknown" and transition is not None:
        p_cand = next(
            (
                c
                for c in transition.get("candidates") or ()
                if c.get("id") == meta.get("proposal_id")
            ),
            None,
        )
        if p_cand is not None:
            p_kind = str(p_cand.get("kind") or "unknown").replace("|", " ")
    return (
        family,
        site,
        m_kind,
        _effect_bucket(meta.get("model_choice_effect", selected.get("effect"))),
        _role_bucket(meta.get("model_choice_role")),
        _overlap_of("model_choice_overlap", "model_choice_id", meta, transition),
        _rank_of(meta, "model_choice_offered_rank"),
        phase_bucket(meta.get("phase")),
        p_kind,
        _effect_bucket(meta.get("proposal_effect")),
        _role_bucket(meta.get("proposal_role")),
        _overlap_of("proposal_overlap", "proposal_id", meta, transition),
        _rank_of(meta, "proposal_offered_rank"),
        phase_bucket(meta.get("phase")),
    )


_SIGNATURE_INDEX = {field: 2 + index for index, field in enumerate(_SIGNATURE_FIELDS)}


_CTX_KEY_FIELDS = ("field", "method", "same_origin")


def _ctx_semantics(ctx) -> str:
    """Authority-semantics fingerprint of a target's observed context.

    The same redacted-label text can sit on controls that do very different
    things — a "Continue" button that submits a same-origin search form and
    a "Continue" that posts cross-origin are different actions. ``ctx`` is
    the derived-flag context the browser records (form destination class,
    field semantics, structural scope): folded into the key, it separates
    semantic twins without persisting any free text.
    """
    if not isinstance(ctx, dict):
        return "unknown"
    flags = (
        "form" if ctx.get("form") else "-",
        "submit" if ctx.get("submit") else "-",
        "ext" if ctx.get("external") else "-",
        "msg" if ctx.get("messaging") else "-",
        "dl" if ctx.get("download") else "-",
        "modal" if ctx.get("modal") else "-",
        "row" if ctx.get("row") else "-",
    )
    values = tuple(
        str(ctx.get(field) if ctx.get(field) is not None else "?")
        .replace("|", " ")
        for field in _CTX_KEY_FIELDS
    )
    return "/".join(flags) + "#" + "/".join(values)


def action_key(kind=None, effect=None, role=None, label=None, *,
               site=None, ctx=None) -> str | None:
    """Stable semantic identity of one concrete action, across runs.

    Treatment-signature coordinates answer "was an action of this *class*
    randomized?"; ``action_key`` answers the stricter Phase-3 question "was
    *this exact action* randomized?" — the distinction between an established
    class effect and treatment-class exchangeability assumed for a
    never-randomized control. The key binds site, operation kind, effect
    class, element role, the action's authority-semantics context, and the
    action's label digested *before* privacy redaction: two controls whose
    labels redact to the same placeholder (``alice@…`` / ``bob@…`` →
    ``[email]``) or that share redacted text under different semantics are
    still different keys. The raw label is normalized and hashed locally —
    only the digest ever leaves the process, never the text — and an action
    with no readable identity at all returns ``None``, never a fabricated
    identity.
    """
    if label is None or not str(label).strip():
        return None
    text = " ".join(str(label).split()).lower()
    if not text:
        return None
    components = (
        _coord(str(site or "unknown").lower()),
        str(kind or "unknown").replace("|", " ") or "unknown",
        _effect_bucket(effect),
        _role_bucket(role),
        text,
        _ctx_semantics(ctx),
    )
    return _stable_hash("|".join(components))


_SIGNATURE_DROP_FIELDS = (
    ("phase", ("m_phase", "p_phase")),
    ("rank", ("m_rank", "p_rank")),
    ("role", ("m_role", "p_role")),
    ("overlap", ("m_ov", "p_ov")),
)


def _signature_masks(known: list[str]) -> list[tuple[str, tuple[str, ...]]]:
    """Progressive backoff over the *known* signature coordinates.

    Each mask names the coordinates still enforced; the rest are wildcards
    merged by summation. The order is most specific first, dropping phase,
    rank, role, and overlap — a bounded, auditable hierarchy that lets thin
    signature cells borrow support from coarser ones without ever mixing
    unrelated strata or inventing values. ``kind`` and ``effect`` are the
    floor: neither is ever dropped, so click evidence can never answer a fill
    proposal and purchase evidence can never answer a search proposal.
    """
    enforced = list(known)
    masks = [("full", tuple(enforced))]
    dropped: list[str] = []
    for name, fields in _SIGNATURE_DROP_FIELDS:
        dropped.append(name)
        enforced = [f for f in enforced if f not in fields]
        label = "minus_" + "_".join(dropped)
        if masks[-1][1] != tuple(enforced):
            masks.append((label, tuple(enforced)))
    return masks
