"""Bound and redact incidental page data before it leaves the browser process."""

from __future__ import annotations

import os
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_API_SECRET = re.compile(r"\b(?:sk|pk|api|token)[-_][A-Za-z0-9_-]{16,}\b", re.I)
_SENSITIVE_QUERY = re.compile(
    r"^(?:access_?token|token|auth|authorization|code|key|password|secret|session|sig|signature)$", re.I
)
_SENSITIVE_LABEL = re.compile(
    r"\b(password|passcode|secret|api key|access token|credit card|card number|cvv|cvc|"
    r"social security|\bsin\b|routing number|bank account|account number)\b",
    re.I,
)


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
    clean["label"] = str(clean.get("label", ""))[:256]
    if "value" in clean:
        if _mode() not in {"off", "none", "0", "false"} and _SENSITIVE_LABEL.search(clean["label"]):
            clean["value"] = "[REDACTED]"
        else:
            clean["value"] = redact_text(clean.get("value", ""), 512)
    if "current_value" in clean:
        clean["current_value"] = redact_text(clean.get("current_value", ""), 512)
    if "option_label" in clean:
        clean["option_label"] = str(clean.get("option_label", ""))[:256]
    return clean


def sanitize_page(page):
    return {
        "url": sanitize_url(page.get("url", ""), 4096),
        "title": redact_text(page.get("title", ""), 256),
        "text": redact_text(page.get("text", ""), 6000),
    }
