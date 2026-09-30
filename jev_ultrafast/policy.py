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
_FINANCIAL_LABELS = re.compile(r"\b(transfer|wire|withdraw|deposit|send money|payment|invoice|billing)\b", re.I)
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


def classify_effect(action) -> Effect:
    """Deterministic effect classification from structural DOM context.

    Uses the bounded ``ctx`` flags emitted by snapshot.js plus the (already
    sanitized) label text. Structure wins over text: a form submit stays a
    SUBMISSION no matter what the button says, and a link stays a NAVIGATE.
    Text patterns only escalate an ambiguous context to a stricter class.
    """
    kind = action.get("kind")
    if kind in {"scroll", "wait"}:
        return Effect.OBSERVE
    if kind in {"fill", "select"}:
        return Effect.FORM_EDIT
    if kind != "click":
        return Effect.UNKNOWN_COMMIT

    ctx = action.get("ctx") or {}
    label = str(action.get("label") or "")

    # Reversible state toggles are form edits; tabs are in-page navigation.
    # A click on an editor role is just the open/focus affordance for it.
    role = action.get("role")
    if role in {"checkbox", "radio", "switch", "option", "menuitemradio"}:
        return Effect.FORM_EDIT
    if role in {"textbox", "combobox", "searchbox", "spinbutton"}:
        return Effect.FORM_EDIT
    if role == "tab":
        return Effect.NAVIGATE

    # Outbound channels and downloads leave the browser's trust boundary.
    if ctx.get("messaging"):
        return Effect.EXTERNAL_MESSAGE
    if ctx.get("download"):
        return Effect.SUBMISSION

    # Deterministic label signals escalate ANY context — a link labelled
    # "Delete account" or a submit labelled "Pay" is not navigation. Absence
    # of a match never lowers the floor the structure already set. Exception:
    # an auth-labelled link merely navigates to a sign-in page — the AUTHENTICATE
    # boundary is the credential submission, not the route to it.
    is_link = action.get("role") == "link" or ctx.get("external")
    for pattern, effect in (
        (_FINANCIAL_LABELS, Effect.FINANCIAL),
        (_PURCHASE_LABELS, Effect.PURCHASE),
        (_DELETE_LABELS, Effect.DELETE),
        (_MESSAGE_LABELS, Effect.EXTERNAL_MESSAGE),
        (_AUTH_LABELS, Effect.AUTHENTICATE),
        (_PERMISSION_LABELS, Effect.PERMISSION_CHANGE),
    ):
        if is_link and effect == Effect.AUTHENTICATE:
            continue
        if pattern.search(label):
            return effect

    if ctx.get("submit") and ctx.get("form"):
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

    # Links with no commit context are navigation.
    if is_link:
        return Effect.NAVIGATE

    # Modal/dialog actions commit or dismiss. Dismissal is autonomous;
    # anything else inside a modal is an unknown commit by definition.
    if ctx.get("modal"):
        return Effect.OBSERVE if _DISMISS_LABELS.search(label) else Effect.UNKNOWN_COMMIT

    # A bare query affordance ("Go", "Search") stays autonomous; progression
    # labels ("Continue", "Next") are treated as opaque commits.
    if _SEARCH_LABELS.search(label):
        return Effect.SEARCH
    if _COMMIT_LABELS.search(label):
        return Effect.UNKNOWN_COMMIT
    if _BENIGN_LABELS.search(label):
        return Effect.OBSERVE
    # A control with no form, no modal scope, no link semantics, and no
    # recognizable affordance label is an opaque state mutation.
    return Effect.UNKNOWN_COMMIT


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
