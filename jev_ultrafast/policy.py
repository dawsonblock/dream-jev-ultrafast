"""Small fail-closed policy gate for browser mutations with external consequences."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Effect(str, Enum):
    """Semantic effect class of a candidate browser mutation."""

    OBSERVE = "observe"
    NAVIGATE = "navigate"
    FORM_EDIT = "form_edit"
    SEARCH = "search"
    AUTHENTICATE = "authenticate"
    SUBMISSION = "submission"
    EXTERNAL_MESSAGE = "external_message"
    PURCHASE = "purchase"
    FINANCIAL = "financial"
    DELETE = "delete"
    PERMISSION_CHANGE = "permission_change"
    DISCLOSURE = "disclosure"
    UNKNOWN_COMMIT = "unknown_commit"


# Authority floor per effect class. The deterministic classifier computes this
# floor from structural context (form membership, submit semantics, field
# inventory, destination origin, modal scope); text signals and any learned
# adviser may only *raise* authority, never lower it.
_AUTHORITY = {
    Effect.OBSERVE: "allow",
    Effect.NAVIGATE: "allow",
    Effect.FORM_EDIT: "allow",
    Effect.SEARCH: "allow",
    Effect.AUTHENTICATE: "require_approval",
    Effect.SUBMISSION: "require_approval",
    Effect.EXTERNAL_MESSAGE: "require_approval",
    Effect.PURCHASE: "require_approval",
    Effect.FINANCIAL: "require_approval",
    Effect.DELETE: "require_approval",
    Effect.PERMISSION_CHANGE: "require_approval",
    Effect.DISCLOSURE: "require_approval",
    Effect.UNKNOWN_COMMIT: "require_approval",
}

_DISMISS_LABELS = re.compile(
    r"\b(cancel|close|dismiss|back|not now|no thanks|later|skip|never mind|decline|keep)\b", re.I
)
_BENIGN_LABELS = re.compile(
    r"\b(show more|load more|see all|read more|expand|collapse|menu|filter|sort|"
    r"details|view|open|play|pause|mute|zoom|refresh|reload|previous|more info)\b", re.I
)
_PURCHASE_LABELS = re.compile(
    r"\b(buy|purchase|order|checkout|add to (?:cart|basket|bag)|cart|basket|pre-?order|"
    r"subscribe|book now|reserve|pay)\b", re.I
)
_FINANCIAL_LABELS = re.compile(
    r"\b(transfer|wire|withdraw|deposit|send money|payment|invoice|billing|credit card|"
    r"card (?:number|ending|details)|cvv|cvc|ssn|social security|routing|iban|account number)\b", re.I
)
_SECRET_LABELS = re.compile(
    r"\b(api[-_ ]?key|secret|private key|access token|auth token|passphrase|"
    r"otp|one[- ]time code|verification code|2fa|backup codes?)\b", re.I
)
_DELETE_LABELS = re.compile(r"\b(delete|remove|erase|uninstall|deactivate|close account|terminate)\b", re.I)
_MESSAGE_LABELS = re.compile(
    r"\b(send|post|comment|reply|share|publish|tweet|message|email|invite|review|rate)\b", re.I
)
_AUTH_LABELS = re.compile(r"\b(sign in|log ?in|sign up|register|create account|password|username)\b", re.I)
_PERMISSION_LABELS = re.compile(
    r"\b(allow|grant|permission|access|notifications?|location|camera|microphone|cookies)\b", re.I
)
_SEARCH_LABELS = re.compile(r"\b(search|find|go|lookup|query|filters?)\b", re.I)
_COMMIT_LABELS = re.compile(
    r"\b(confirm|ok|accept|agree|continue|next|proceed|apply|save|update|submit|yes|done|finish|start)\b", re.I
)


# Authority rank used to pick the strictest applicable classification.
_AUTHORITY_RANK = {"allow": 0, "require_approval": 1, "deny": 2}

_EDIT_KINDS = {"fill", "select"}
_TOGGLE_ROLES = {"checkbox", "radio", "switch", "option", "menuitemradio"}
_EDITOR_ROLES = {"textbox", "combobox", "searchbox", "spinbutton"}

# Per-target field kind (ctx.field from snapshot.js) → minimum effect for a
# fill/select landing on it. Entering data is itself a disclosure event: page
# JavaScript observes input events before any submit ever runs.
_FIELD_EFFECT = {
    "auth": Effect.AUTHENTICATE,
    "otp": Effect.AUTHENTICATE,
    "money": Effect.FINANCIAL,
    "file": Effect.SUBMISSION,
    "email": Effect.DISCLOSURE,
    "tel": Effect.DISCLOSURE,
    "message": Effect.EXTERNAL_MESSAGE,
}


def _submit_effect(ctx: dict) -> Effect:
    fields = ctx.get("fields") or {}
    if fields.get("password"):
        return Effect.AUTHENTICATE
    if fields.get("money"):
        return Effect.PURCHASE
    if fields.get("file"):
        return Effect.SUBMISSION
    if ctx.get("method") == "get" or fields.get("search"):
        return Effect.SEARCH
    return Effect.SUBMISSION


def _label_effect(label: str, *, is_link: bool) -> Effect | None:
    """First matching high-risk label pattern, or None.

    Exception: an auth-labelled link merely navigates to a sign-in page — the
    AUTHENTICATE boundary is the credential submission, not the route to it.
    """
    for pattern, effect in (
        (_FINANCIAL_LABELS, Effect.FINANCIAL),
        (_PURCHASE_LABELS, Effect.PURCHASE),
        (_DELETE_LABELS, Effect.DELETE),
        (_MESSAGE_LABELS, Effect.EXTERNAL_MESSAGE),
        (_SECRET_LABELS, Effect.DISCLOSURE),
        (_AUTH_LABELS, Effect.AUTHENTICATE),
        (_PERMISSION_LABELS, Effect.PERMISSION_CHANGE),
    ):
        if is_link and effect == Effect.AUTHENTICATE:
            continue
        if pattern.search(label):
            return effect
    return None


def classify_effect(action) -> Effect:
    """Deterministic effect classification from structural DOM context.

    Monotonic escalation: structural floors (kind, role, form membership) are
    combined with field sensitivity and label signals, and the *highest*
    authority wins. There are no early returns past the non-mutating kinds —
    a fill into a card field or a "Allow notifications" switch cannot short-
    circuit to FORM_EDIT before escalation runs.
    """
    kind = action.get("kind")
    if kind in {"scroll", "wait"}:
        return Effect.OBSERVE

    ctx = action.get("ctx") or {}
    label = str(action.get("label") or "")
    role = action.get("role")
    is_link = role == "link" or ctx.get("external")

    candidates: list[Effect] = []

    if kind in _EDIT_KINDS:
        # Entering or changing field data is a form edit at minimum, and a
        # disclosure/commit event when the target is sensitive.
        candidates.append(Effect.FORM_EDIT)
        field_effect = _FIELD_EFFECT.get(ctx.get("field"))
        if field_effect is not None:
            candidates.append(field_effect)
        if ctx.get("messaging"):
            candidates.append(Effect.EXTERNAL_MESSAGE)
    elif kind != "click":
        return Effect.UNKNOWN_COMMIT
    else:
        if role in _TOGGLE_ROLES or role in _EDITOR_ROLES:
            candidates.append(Effect.FORM_EDIT)
        elif role == "tab":
            candidates.append(Effect.NAVIGATE)
        if ctx.get("messaging"):
            candidates.append(Effect.EXTERNAL_MESSAGE)
        if ctx.get("download"):
            candidates.append(Effect.SUBMISSION)
        if ctx.get("submit") and ctx.get("form"):
            candidates.append(_submit_effect(ctx))
        if is_link:
            candidates.append(Effect.NAVIGATE)
        if ctx.get("modal"):
            candidates.append(Effect.OBSERVE if _DISMISS_LABELS.search(label) else Effect.UNKNOWN_COMMIT)

    # High-risk label signals escalate every kind — including toggles and
    # edits, which is the point of the monotonic structure.
    label_hit = _label_effect(label, is_link=is_link)
    if label_hit is not None:
        candidates.append(label_hit)

    if not candidates:
        # Bare clicks with no structural or risk signal: query affordances
        # stay autonomous; progression labels are opaque commits.
        if _SEARCH_LABELS.search(label):
            candidates.append(Effect.SEARCH)
        elif _COMMIT_LABELS.search(label):
            candidates.append(Effect.UNKNOWN_COMMIT)
        elif _BENIGN_LABELS.search(label):
            candidates.append(Effect.OBSERVE)
        else:
            candidates.append(Effect.UNKNOWN_COMMIT)

    return max(candidates, key=lambda e: _AUTHORITY_RANK[_AUTHORITY[e]])


@dataclass(frozen=True)
class PolicyDecision:
    level: str
    reason: str = ""
    effect: str = ""


class DefaultActionPolicy:
    """Deterministic effect-classification gate with an escalation-only adviser.

    The floor is computed from structural context by ``classify_effect`` —
    unknown commit-shaped controls require approval even when their labels are
    innocuous ("Continue" inside an unscoped workflow). The legacy text-pattern
    list and the optional learned ``adviser`` may only raise authority; a
    classifier/adviser verdict can never downgrade a deterministic high-risk
    classification to autonomous.
    """

    _approval_patterns = (
        (re.compile(r"\b(buy now|place order|confirm (?:purchase|order)|pay now|purchase|checkout)\b", re.I),
         "purchase or payment"),
        (re.compile(r"\b(transfer|send money|wire (?:money|funds)|withdraw)\b", re.I),
         "movement of money"),
        (re.compile(r"\b(delete|close|remove) (?:my )?(?:account|profile|workspace|organization)\b", re.I),
         "destructive account action"),
        (re.compile(r"\b(publish|post publicly|send (?:message|email)|submit application)\b", re.I),
         "external communication or submission"),
        (re.compile(r"\b(confirm deletion|permanently delete|erase all)\b", re.I),
         "irreversible destructive action"),
    )

    def __init__(self, adviser=None):
        """``adviser(action, page=, goal=) -> PolicyDecision|str|None`` may only
        escalate an allowed action to require_approval/deny — never the reverse."""
        self.adviser = adviser

    def assess(self, action, *, page=None, goal=None) -> PolicyDecision:
        effect = classify_effect(action)
        if action.get("kind") not in {"click", "select", "fill"}:
            return PolicyDecision("allow", f"effect:{effect.value}", effect.value)
        floor = _AUTHORITY[effect]
        reason = f"effect:{effect.value}"
        label = str(action.get("label", ""))[:512]
        for pattern, legacy_reason in self._approval_patterns:
            if pattern.search(label):
                floor = "require_approval"
                reason = f"{reason}; {legacy_reason}"
                break
        decision = PolicyDecision(floor, reason, effect.value)
        if self.adviser is not None and decision.level == "allow":
            advisory = self.adviser(action, page=page, goal=goal)
            advisory_level = getattr(advisory, "level", advisory)
            if advisory_level == "deny":
                decision = PolicyDecision("deny", "adviser denied", effect.value)
            elif advisory_level == "require_approval":
                decision = PolicyDecision("require_approval", "adviser escalation", effect.value)
        return decision
