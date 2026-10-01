"""Bound and redact incidental page data before it leaves the browser process."""

from __future__ import annotations

import os
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_API_SECRET = re.compile(r"\b(?:sk|pk|api|token)[-_][A-Za-z0-9_-]{16,}\b", re.I)
_SENSITIVE_QUERY = re.compile(
    r"^(?:access_?token|refresh_?token|id_?token|token|auth(?:entication|orization)?|code|"
    r"api_?key|apikey|access_?key|client_?secret|private_?key|secret(?:_?key)?|key|"
    r"pass(?:word|wd)|pwd|session(?:_?id)?|sig(?:nature)?|jwt|bearer|csrf|xsrf|otp|totp|ssn)$",
    re.I,
)
_SENSITIVE_LABEL = re.compile(
    r"\b(password|passcode|secret|api key|access token|credit card|card number|cvv|cvc|"
    r"social security|\bsin\b|routing number|bank account|account number)\b",
    re.I,
)


def tokenize(value):
    """Shared Unicode-aware tokenization for candidate scoring and trace evidence."""
    token = []
    for char in str(value).lower():
        if char.isalnum():
            token.append(char)
        elif token:
            joined = "".join(token)
            if len(joined) >= 2:
                yield joined
            token = []
    if token:
        joined = "".join(token)
        if len(joined) >= 2:
            yield joined


def _mode():
    return os.environ.get("JEV_MODEL_PRIVACY", "basic").strip().lower()


def redact_text(value, limit=6000):
    text = str(value or "")[:limit]
    if _mode() in {"off", "none", "0", "false"}:
        return text
    text = _EMAIL.sub("[REDACTED_EMAIL]", text)
    text = _CARD.sub("[REDACTED_NUMBER]", text)
    text = _API_SECRET.sub("[REDACTED_SECRET]", text)
    return text



def sanitize_url(value, limit=4096):
    raw = str(value or "")[:limit]
    if _mode() in {"off", "none", "0", "false"}:
        return raw
    try:
        parts = urlsplit(raw)
        query = urlencode([
            (key, "[REDACTED]" if _SENSITIVE_QUERY.match(key) else redact_text(val, 512))
            for key, val in parse_qsl(parts.query, keep_blank_values=True)
        ], doseq=True)
        fragment = redact_text(parts.fragment, 512)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, query, fragment))[:limit]
    except ValueError:
        return redact_text(raw, limit)

def sanitize_action(action):
    clean = dict(action)
    # Labels are page-controlled text and can carry emails, card numbers, or
    # tokens — they cross every external model boundary, so redact them too.
    clean["label"] = redact_text(str(clean.get("label", "")), 256)
    if "value" in clean:
        if _mode() not in {"off", "none", "0", "false"} and _SENSITIVE_LABEL.search(clean["label"]):
            clean["value"] = "[REDACTED]"
        else:
            clean["value"] = redact_text(clean.get("value", ""), 512)
    if "current_value" in clean:
        clean["current_value"] = redact_text(clean.get("current_value", ""), 512)
    if "option_label" in clean:
        clean["option_label"] = redact_text(str(clean.get("option_label", "")), 256)
    return clean


def action_goal_overlap(action, goal_tokens, clean=None):
    """Goal-token overlap computed on the sanitized candidate view.

    Live candidate scoring and trace compaction must agree on overlap, so both
    tokenize the sanitized action fields rather than raw page text. Callers
    that already hold ``sanitize_action(action)`` may pass it as ``clean`` to
    skip re-redacting the same dict — common in per-candidate hot loops.
    """
    if clean is None:
        clean = sanitize_action(action)
    searchable = " ".join(
        str(clean.get(k, "")) for k in ("label", "value", "current_value", "option_label")
    ).lower()
    return len(set(goal_tokens) & set(tokenize(searchable)))


def sanitize_page(page):
    return {
        "url": sanitize_url(page.get("url", ""), 4096),
        "title": redact_text(page.get("title", ""), 256),
        "text": redact_text(page.get("text", ""), 6000),
    }
