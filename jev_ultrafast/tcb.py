"""The trusted computing base boundary.

``MANIFEST.sha256`` lists the whole release set; this module names the
subset that is the *trusted* base — the files recursive improvement may
never modify. Candidates may change only validated ``ExplorationPolicy``
data; everything here — browser execution, the authority/approval
boundary, privacy, evidence capture and validation, replay scoring,
causal estimation, promotion, signing, and the qualification tooling —
is outside candidate modification authority by construction, and this
list makes that boundary machine-checkable.

Two kinds of membership:

- ``TCB_FILES`` — individually named trusted files.
- ``TCB_PACKAGE_PREFIXES`` — directories whose *every* file is trusted.
  A new file landing under one of these prefixes is a TCB change whether
  or not it is listed here yet; ``scripts/check_tcb.py`` fails closed on
  unclassified entries so code cannot silently join the trusted surface.

This module is itself TCB: a candidate must not shrink the boundary by
editing the list.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

# Directories whose every member is trusted. Checked before TCB_FILES so
# a new module inside the package is treated as trusted immediately.
TCB_PACKAGE_PREFIXES = (
    "jev_ultrafast/_dream/",
    "jev_ultrafast/_learning/",
)

# Individually trusted files outside the package prefixes.
#
#   agent.py / browser.py / snapshot.js — the only mutation path: approval
#       capability, isolated-world execution, atomic/trusted guarantees.
#   model.py — the finite-choice boundary: candidates must map to observed
#       elements and supported operations; validation lives here.
#   policy.py — deterministic effect classification + approval boundary.
#   privacy.py — redaction, routing levels, security profiles.
#   trace.py — live evidence capture.
#   signing.py — domain-separated Ed25519.
#   dream.py / dreamlearn.py / __init__.py — facades re-export trusted
#       code; a modified facade can redirect the trusted surface.
#   tcb.py — this boundary definition; shrinking it is a TCB change.
#   scripts/{qualify,update_manifest,sign_manifest,env_digest,
#            verify_release}.py —
#       release qualification and provenance tooling; verify_release is
#       additionally the externally bootstrapped verifier meant to run
#       from outside the artifact.
TCB_FILES = frozenset(
    {
        "jev_ultrafast/__init__.py",
        "jev_ultrafast/agent.py",
        "jev_ultrafast/browser.py",
        "jev_ultrafast/dream.py",
        "jev_ultrafast/dreamlearn.py",
        "jev_ultrafast/model.py",
        "jev_ultrafast/policy.py",
        "jev_ultrafast/privacy.py",
        "jev_ultrafast/signing.py",
        "jev_ultrafast/snapshot.js",
        "jev_ultrafast/tcb.py",
        "jev_ultrafast/trace.py",
        "scripts/env_digest.py",
        "scripts/qualify.py",
        "scripts/sign_manifest.py",
        "scripts/update_manifest.py",
        "scripts/verify_release.py",
    }
)


def is_tcb(path: str) -> bool:
    """Whether ``path`` (repo-relative, posix separators) is trusted code."""
    normalized = path.lstrip("./")
    if normalized in TCB_FILES:
        return True
    return any(
        normalized.startswith(prefix) for prefix in TCB_PACKAGE_PREFIXES
    )


def tcb_paths() -> frozenset:
    """The individually listed trusted files (prefix members excluded —
    they are resolved by enumeration, not named one by one)."""
    return TCB_FILES


# ---------------------------------------------------------------------------
# Runtime release verification (qualified profile).
#
# ``JEV_SECURITY_PROFILE=qualified`` requires the running tree to re-verify
# its signed release before the agent starts: the manifest must carry a
# valid domain-separated signature under a pinned key, every manifest-listed
# file must hash to its recorded digest, and the trusted-base boundary must
# be intact. This is defense-in-depth on top of the external bootstrap
# (scripts/verify_release.py) — it catches partial tampering and corrupted
# installs at startup; it is not a substitute for verifying the artifact
# before it is ever executed.
# ---------------------------------------------------------------------------

_MANIFEST = "MANIFEST.sha256"
_SIG_FILE = "MANIFEST.sig"
_SIG_SCHEMA = "jev-manifest-sig/1"
_SIG_DOMAIN = b"jev-dream/release-manifest/v1:"
_IGNORE_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".hypothesis"}

# A signed release proves *integrity* (this tree is what a release authority
# signed). A *qualified* run asserts a separate claim — that the full
# qualification pipeline ran over exactly this manifest and a qualification
# authority attested the verdict. ``RELEASE_QUALIFICATION.json`` is the
# machine-readable form of that claim: a ``jev-qualify/*`` report with
# ``release_qualified == true``, bound to the manifest digest, and signed
# under a key pinned in ``JEV_QUALIFY_VERIFY_KEYS``. Like ``MANIFEST.sig``
# it travels outside the manifest — it binds the manifest digest, so the
# manifest cannot list it without a fixpoint.
_QUALIFICATION_REPORT = "RELEASE_QUALIFICATION.json"
_QUALIFY_SIG_SCHEMA = "jev-qualify-sig/1"
_QUALIFY_SIG_DOMAIN = b"jev-dream/qualify-report/v1:"
_TCB_VERSION_RE = re.compile(
    r'TCB_VERSION\s*=\s*"(jev-ultrafast-tcb/[0-9.]+)"')


def _sig_valid(
    key_id: str, digest: str, signature: str, *, domain: bytes = _SIG_DOMAIN
) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(key_id)).verify(
            bytes.fromhex(signature), domain + digest.encode("ascii")
        )
        return True
    except (InvalidSignature, ValueError):
        return False


def _report_digest(report: dict) -> str:
    """Canonical report digest — the qualification signature covers the
    report with its ``signature`` block stripped (``qualify.py`` form)."""
    clean = {k: v for k, v in report.items() if k != "signature"}
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _qualification_violations(
    root: Path, manifest_digest: str, qualify_keys: set[str]
) -> list[str]:
    """Every qualification-evidence violation (empty = qualified).

    Fail-closed on each axis: the report must exist, be signed under a
    pinned qualification key, declare ``release_qualified == true``, and
    bind the *current* manifest digest — a report validating a different
    artifact is stale evidence, not qualification.
    """
    violations: list[str] = []
    if not qualify_keys:
        violations.append(
            "JEV_QUALIFY_VERIFY_KEYS: no pinned qualification key — an "
            "unpinned report accepts anything signed"
        )
    report_path = root / _QUALIFICATION_REPORT
    if not report_path.is_file():
        violations.append(
            f"{_QUALIFICATION_REPORT}: missing — a qualified run requires "
            "signed qualification evidence bound to this manifest"
        )
        return violations
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return violations + [f"{_QUALIFICATION_REPORT}: unreadable ({exc})"]
    if not isinstance(report, dict) or not str(
        report.get("schema") or ""
    ).startswith("jev-qualify/"):
        return violations + [
            f"{_QUALIFICATION_REPORT}: unrecognized qualification report"
        ]
    if report.get("release_qualified") is not True:
        violations.append(
            f"{_QUALIFICATION_REPORT}: release_qualified is not true — "
            "validation evidence is not a release verdict"
        )
    bound = (report.get("provenance") or {}).get("manifest_digest")
    if bound != manifest_digest:
        violations.append(
            f"{_QUALIFICATION_REPORT}: qualifies a different manifest — "
            "stale or transplanted qualification evidence"
        )
    sig = report.get("signature")
    if not isinstance(sig, dict) or sig.get("schema") != _QUALIFY_SIG_SCHEMA:
        violations.append(
            f"{_QUALIFICATION_REPORT}: unsigned or unrecognized "
            "signature block"
        )
    elif qualify_keys:
        digest = _report_digest(report)
        if sig.get("report_digest") != digest:
            violations.append(
                f"{_QUALIFICATION_REPORT}: report digest mismatch — "
                "modified after signing"
            )
        elif str(sig.get("key_id")) not in qualify_keys:
            violations.append(
                f"{_QUALIFICATION_REPORT}: signed by an unexpected key"
            )
        elif not _sig_valid(
            str(sig.get("key_id")),
            digest,
            str(sig.get("signature") or ""),
            domain=_QUALIFY_SIG_DOMAIN,
        ):
            violations.append(
                f"{_QUALIFICATION_REPORT}: signature verification failure"
            )
    return violations


def verify_installation(
    root: Path | None = None,
    keys: set[str] | None = None,
    qualify_keys: set[str] | None = None,
    expect_tcb_version: str | None = None,
    expect_manifest_digest: str | None = None,
) -> list[str]:
    """Every release violation in the running tree (empty = verified).

    ``root`` defaults to the installed package's parent (the release or
    checkout root); ``keys`` to ``JEV_MANIFEST_VERIFY_KEYS`` and
    ``qualify_keys`` to ``JEV_QUALIFY_VERIFY_KEYS``. All fail closed: no
    manifest, no signature, no pinned key, or no signed qualification
    report bound to this manifest means there is nothing a qualified run
    may stand on.

    Anti-rollback pins (``expect_tcb_version`` /
    ``expect_manifest_digest``, or ``JEV_EXPECT_TCB_VERSION`` /
    ``JEV_EXPECT_MANIFEST_DIGEST``) close the gap a valid signature cannot:
    the same key may have signed an older, weaker release — authenticity
    is not recency. When pinned, drift fails closed.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    if keys is None:
        keys = {
            k.strip()
            for k in os.environ.get("JEV_MANIFEST_VERIFY_KEYS", "").split(",")
            if k.strip()
        }
    if qualify_keys is None:
        qualify_keys = {
            k.strip()
            for k in os.environ.get("JEV_QUALIFY_VERIFY_KEYS", "").split(",")
            if k.strip()
        }
    if expect_tcb_version is None:
        expect_tcb_version = os.environ.get("JEV_EXPECT_TCB_VERSION") or None
    if expect_manifest_digest is None:
        expect_manifest_digest = (
            os.environ.get("JEV_EXPECT_MANIFEST_DIGEST") or None
        )
    violations: list[str] = []
    if not keys:
        violations.append(
            "JEV_MANIFEST_VERIFY_KEYS: no pinned release key — an unpinned "
            "verifier accepts anything signed"
        )
    manifest = root / _MANIFEST
    if not manifest.is_file():
        return violations + [f"{_MANIFEST}: missing — no release set to verify"]
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if expect_manifest_digest and digest != expect_manifest_digest.strip().lower():
        violations.append(
            f"{_MANIFEST}: digest does not match the pinned release — "
            "a validly-signed older release is not the approved one"
        )
    sig = root / _SIG_FILE
    if not sig.is_file():
        violations.append(f"{_SIG_FILE}: missing — unsigned release manifest")
    else:
        try:
            block = json.loads(sig.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            violations.append(f"{_SIG_FILE}: unreadable ({exc})")
            block = None
        if isinstance(block, dict):
            if block.get("schema") != _SIG_SCHEMA:
                violations.append(f"{_SIG_FILE}: unrecognized signature block")
            elif block.get("manifest_digest") != digest:
                violations.append(
                    f"{_SIG_FILE}: signed digest does not match {_MANIFEST}"
                )
            elif str(block.get("key_id")) not in keys:
                violations.append(
                    f"{_SIG_FILE}: manifest signed by an unexpected key"
                )
            elif not _sig_valid(
                str(block.get("key_id")), digest,
                str(block.get("signature") or ""),
            ):
                violations.append(f"{_SIG_FILE}: signature verification failure")
    entries = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            d, _, name = line.partition("  ")
            entries[name.strip()] = d.strip()
    if not entries:
        violations.append(f"{_MANIFEST}: empty or unparsable")
    for name, recorded in sorted(entries.items()):
        path = root / name
        if not path.is_file():
            violations.append(f"{name}: manifest-listed file missing from tree")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != recorded:
            violations.append(f"{name}: content digest differs from manifest")
    for name in TCB_FILES:
        if name not in entries:
            violations.append(f"{name}: trusted file absent from {_MANIFEST}")
    for prefix in TCB_PACKAGE_PREFIXES:
        directory = root / prefix
        if not directory.is_dir():
            violations.append(f"{prefix}: trusted package missing from tree")
            continue
        for path in directory.rglob("*"):
            if path.is_file() and not (set(path.parts) & _IGNORE_DIRS):
                rel = path.relative_to(root).as_posix()
                if rel not in entries:
                    violations.append(
                        f"{rel}: trusted package member absent from {_MANIFEST}"
                    )
    # Anti-rollback TCB pin: parsed as text, never executed — the file is
    # part of the artifact under verification.
    if expect_tcb_version:
        common = root / "jev_ultrafast/_dream/common.py"
        found = None
        if common.is_file():
            match = _TCB_VERSION_RE.search(
                common.read_text(encoding="utf-8", errors="replace")
            )
            found = match.group(1) if match else None
        if found != expect_tcb_version:
            violations.append(
                f"TCB_VERSION: expected {expect_tcb_version}, found "
                f"{found or 'unreadable'} — an older signed TCB is not the "
                "approved one"
            )
    # A signed release proves integrity; *qualified* asserts the separate
    # claim that the full qualification pipeline attested exactly this
    # manifest — the signed report must travel with the tree.
    violations += _qualification_violations(root, digest, qualify_keys)
    return violations
