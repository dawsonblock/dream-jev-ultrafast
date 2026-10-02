"""Bound and redact incidental page data before it leaves the browser process."""

from __future__ import annotations

import os
import re
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_API_SECRET = re.compile(r"\b(?:sk|pk|api|token)[-_][A-Za-z0-9_-]{16,}\b", re.I)
# Opaque bearer-shaped material that appears without a keyword prefix: JWTs
# and long mixed-character path segments (reset tokens, session ids, hashes).
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
# A whole path segment that is one opaque token: 24+ mixed alphanumeric
# characters (hex/base32/base64url material — real slugs stay readable) or a
# pure 16+ digit run, which is card/account-id shaped, never prose.
_PATH_TOKEN = re.compile(
    r"(?:(?=.*\d)(?=.*[a-z])[a-z0-9_\-]{24,}|\d{16,})", re.I
)
_SENSITIVE_KEYS = (
    r"access_?token|refresh_?token|id_?token|token|auth(?:entication|orization)?|code|"
    r"api_?key|apikey|access_?key|client_?secret|private_?key|secret(?:_?key)?|key|"
    r"pass(?:word|wd)|pwd|session(?:_?id)?|sig(?:nature)?|jwt|bearer|csrf|xsrf|otp|totp|ssn"
)
_SENSITIVE_QUERY = re.compile(rf"^(?:{_SENSITIVE_KEYS})$", re.I)
# key=value material wherever it appears — decoded path segments, fragments,
# and nested-URL query values can all carry credential pairs that never went
# through the top-level query-parameter redaction.
_SENSITIVE_KV = re.compile(rf"\b(?:{_SENSITIVE_KEYS})=[^&#\s;]*", re.I)
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


# ---------------------------------------------------------------------------
# Model routing security levels.
#
# Every value sent to a model endpoint is bound to one routing level before
# inference starts (JEV_MODEL_ROUTING):
#
#   local-only  — every model endpoint must resolve to a loopback host; any
#                 configured remote URL fails closed before the request body
#                 is even built. Nothing the operator types leaves the machine.
#   sanitized   — (default) remote endpoints are allowed, but goal/objective
#                 text crosses the wire only after the same redaction pass as
#                 page content. Residual risk: secrets the regexes do not
#                 recognise still travel — declare local-only for those runs.
#   public      — the task is declared non-sensitive; goal text is sent
#                 verbatim. Explicit opt-in, never the silent default.
#
# The level classifies the *run*, not individual fields: regex redaction is a
# floor, not a guarantee, which is why "local-only" exists.
# ---------------------------------------------------------------------------

ROUTING_LEVELS = ("local-only", "sanitized", "public")
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def routing_level():
    level = os.environ.get("JEV_MODEL_ROUTING", "sanitized").strip().lower()
    if level not in ROUTING_LEVELS:
        # An unrecognized level is a misconfiguration, not a permission —
        # fail closed rather than guessing how much may leave the machine.
        raise ValueError(
            f"JEV_MODEL_ROUTING must be one of {', '.join(ROUTING_LEVELS)}"
        )
    if level == "sanitized" and _mode() in {"off", "none", "0", "false"}:
        # The two flags must not compose into raw remote transmission:
        # "sanitized" is the promise that outbound text passed the redaction
        # pass, and privacy=off disables that pass while remote endpoints
        # remain reachable. Declare the intent instead — 'public' sends
        # verbatim, 'local-only' never leaves the machine.
        raise ValueError(
            "JEV_MODEL_ROUTING=sanitized cannot combine with "
            "JEV_MODEL_PRIVACY=off — sanitized means outbound text was "
            "redacted. Set JEV_MODEL_ROUTING=public to send raw text "
            "explicitly, or local-only to keep the run on-machine."
        )
    return level


def loopback_endpoint(url):
    host = (urlsplit(str(url)).hostname or "").lower()
    return host in _LOOPBACK_HOSTS


def assert_endpoint_allowed(url):
    """Fail closed when the routing level forbids a remote model endpoint."""
    if routing_level() == "local-only" and not loopback_endpoint(url):
        raise ValueError(
            "JEV_MODEL_ROUTING=local-only refuses remote model endpoint "
            f"{urlsplit(str(url)).netloc or url!s}; point *_BASE_URL at a local "
            "server or relax the routing level for this run."
        )


def assert_transport_secure(url):
    """Require TLS for every non-loopback model endpoint.

    Loopback HTTP is fine (same machine); a remote http:// endpoint would
    carry the Authorization header and the entire model context in cleartext.
    ``JEV_ALLOW_INSECURE_TRANSPORT=1`` is the explicit dangerous-development
    escape hatch — never set it for a qualified run.
    """
    parts = urlsplit(str(url))
    scheme = parts.scheme.lower()
    if scheme == "https" or loopback_endpoint(url):
        return
    if scheme == "http" and os.environ.get("JEV_ALLOW_INSECURE_TRANSPORT") == "1":
        return
    raise ValueError(
        f"Refusing {'plaintext ' if scheme == 'http' else 'non-TLS '}"
        f"model endpoint {parts.netloc or url!s}: non-loopback model traffic "
        "requires https (or JEV_ALLOW_INSECURE_TRANSPORT=1 for local "
        "development only)."
    )


def outbound_text(value, limit=6000):
    """Bound free-text that may reach a model endpoint to the routing level."""
    if routing_level() == "public":
        return str(value or "")[:limit]
    return redact_text(value, limit)


def redact_text(value, limit=6000):
    text = str(value or "")[:limit]
    if _mode() in {"off", "none", "0", "false"}:
        return text
    text = _EMAIL.sub("[REDACTED_EMAIL]", text)
    text = _CARD.sub("[REDACTED_NUMBER]", text)
    text = _API_SECRET.sub("[REDACTED_SECRET]", text)
    text = _JWT.sub("[REDACTED_SECRET]", text)
    return text


def _mask_sensitive_kv(match):
    """``token=abc`` → ``token=[REDACTED]`` — keep the key name, drop the value."""
    return match.group(0).split("=", 1)[0] + "=[REDACTED]"


def _path_segment(segment):
    """Sanitize one raw URL path segment.

    Secrets hide behind percent-escapes, so decode before the redaction pass —
    then re-encode the structural characters decoding could introduce: an
    escaped ``?``/``#``/``/`` must not re-enter the emitted URL as real
    structure (a decoded ``?`` would otherwise carry ``key=value`` material
    that never passed the sensitive-parameter mask)."""
    decoded = _SENSITIVE_KV.sub(
        _mask_sensitive_kv, redact_text(unquote(segment), 512)
    )
    if _PATH_TOKEN.fullmatch(decoded):
        return "[REDACTED]"
    return (
        decoded.replace("%", "%25")
        .replace("?", "%3F")
        .replace("#", "%23")
        .replace("/", "%2F")
    )


def sanitize_url(value, limit=4096):
    raw = str(value or "")[:limit]
    if _mode() in {"off", "none", "0", "false"}:
        return raw
    try:
        parts = urlsplit(raw)
        # URL userinfo carries verbatim credentials (http://user:pass@host) —
        # never forward it to a model endpoint.
        netloc = parts.netloc.rsplit("@", 1)[-1]
        # Paths carry reset/session tokens, emails, and account ids. Segments
        # are decoded and redacted individually so a decoded %2F cannot
        # smuggle a token past a segment boundary.
        path = "/".join(_path_segment(segment) for segment in parts.path.split("/"))
        query = urlencode([
            (key, "[REDACTED]" if _SENSITIVE_QUERY.match(key)
             else _SENSITIVE_KV.sub(_mask_sensitive_kv, redact_text(val, 512)))
            for key, val in parse_qsl(parts.query, keep_blank_values=True)
        ], doseq=True)
        # Fragments carry OAuth-style key=value tokens (#access_token=…).
        fragment = _SENSITIVE_KV.sub(
            _mask_sensitive_kv, redact_text(unquote(parts.fragment), 512)
        )
        return urlunsplit((parts.scheme, netloc, path, query, fragment))[:limit]
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
