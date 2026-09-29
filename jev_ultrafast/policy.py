"""Small fail-closed policy gate for browser mutations with external consequences."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyDecision:
    level: str
    reason: str = ""


class DefaultActionPolicy:
    """Require caller approval for obviously consequential commit actions.

    This is intentionally narrow: it is an enforcement backstop, not a semantic
    classifier for the whole web. Applications can inject a stricter policy.
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

    def assess(self, action, *, page=None, goal=None):
        if action.get("kind") not in {"click", "select"}:
            return PolicyDecision("allow")
        label = str(action.get("label", ""))[:512]
        for pattern, reason in self._approval_patterns:
            if pattern.search(label):
                return PolicyDecision("require_approval", reason)
        return PolicyDecision("allow")
