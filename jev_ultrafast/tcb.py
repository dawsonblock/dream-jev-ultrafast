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
#   scripts/{qualify,update_manifest,sign_manifest,env_digest}.py —
#       release qualification and provenance tooling.
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
