"""Verify the trusted computing base (Phase 17).

``jev_ultrafast.tcb`` names the boundary: the files recursive improvement
may never modify — browser execution, authority/approval, privacy,
evidence, replay scoring, causal estimation, promotion, signing, and the
qualification tooling itself. This check fails closed when:

1. a TCB file is missing or its content digest differs from
   ``MANIFEST.sha256`` (drift inside the trusted base — the signed
   manifest is the reference, so a signature over it also covers the TCB);
2. a file exists under a TCB package prefix (``jev_ultrafast/_dream/``,
   ``jev_ultrafast/_learning/``) that the manifest does not cover — every
   prefix member is trusted by construction, so trusted code outside the
   signed release set is unaccountable;
3. a TCB file is absent from ``MANIFEST.sha256`` — the signed release set
   must cover the whole trusted base.

    uv run python scripts/check_tcb.py            # check, write nothing
"""

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jev_ultrafast.tcb import (  # noqa: E402
    TCB_FILES,
    TCB_PACKAGE_PREFIXES,
)

MANIFEST = "MANIFEST.sha256"
_IGNORE_DIRS = {
    "__pycache__", ".pytest_cache", ".ruff_cache", ".hypothesis",
}


def _manifest_digests(root: Path) -> dict[str, str]:
    manifest = root / MANIFEST
    if not manifest.exists():
        return {}
    entries = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            digest, _, name = line.partition("  ")
            entries[name.strip()] = digest.strip()
    return entries


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(root: Path = ROOT) -> list[str]:
    """Return the list of TCB violations under ``root`` (empty = clean)."""
    violations = []
    digests = _manifest_digests(root)

    listed = set(TCB_FILES)
    # Prefix members are trusted too — enumerate them from the manifest so
    # the check travels with the release set.
    listed.update(
        name for name in digests
        if any(name.startswith(p) for p in TCB_PACKAGE_PREFIXES)
    )

    for name in sorted(listed):
        path = root / name
        recorded = digests.get(name)
        if recorded is None:
            violations.append(
                f"{name}: trusted file absent from {MANIFEST}"
            )
            continue
        if not path.is_file():
            violations.append(f"{name}: trusted file missing from tree")
            continue
        if _digest(path) != recorded:
            violations.append(
                f"{name}: content digest differs from {MANIFEST} — "
                "trusted-base drift"
            )

    # Completeness in the other direction: every file living under a TCB
    # package prefix is trusted by construction (is_tcb), so it must be
    # covered by the manifest — trusted code outside the signed release
    # set is unaccountable. Enumeration is over the filesystem itself —
    # an unlisted file cannot hide behind the manifest's file set.
    for prefix in TCB_PACKAGE_PREFIXES:
        base = root / prefix
        if not base.is_dir():
            violations.append(f"{prefix}: trusted package missing")
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            if any(part in _IGNORE_DIRS for part in rel.split("/")[:-1]):
                continue
            if rel not in digests:
                violations.append(
                    f"{rel}: trusted package member absent from {MANIFEST} — "
                    "the signed release set must cover the whole trusted base"
                )
    return violations


def main() -> int:
    violations = check(ROOT)
    if violations:
        for violation in violations:
            print(f"TCB: {violation}", file=sys.stderr)
        return 1
    print(
        f"TCB: trusted base verified "
        f"({len(TCB_FILES)} listed files + prefix members)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
