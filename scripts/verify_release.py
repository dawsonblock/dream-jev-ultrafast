"""Externally bootstrapped release verification.

The trust root lives *here*, not inside the artifact: this script is
self-contained (stdlib + ``cryptography``), imports nothing from
``jev_ultrafast``, and can be copied out of the tree and run against any
release directory. The artifact under verification never supplies —
or executes — the code that verifies it:

    known verifier + pinned public key
        -> verify MANIFEST.sig
        -> hash artifact contents against MANIFEST.sha256
        -> check the trusted-base boundary and TCB version
        -> check the generated validation report binds this manifest
        -> only then trust the release set

    uv run python scripts/verify_release.py --key <pubkey-hex> [--root DIR]
        [--expect-tcb-version jev-ultrafast-tcb/0.16] [--require-validation]
    JEV_MANIFEST_VERIFY_KEYS=<k1>,<k2> uv run python scripts/verify_release.py

Checks, all fail-closed:

1. ``MANIFEST.sig`` — schema ``jev-manifest-sig/1``, pinned key id, digest
   field matching the real manifest digest, valid Ed25519 signature.
2. Every file the manifest lists exists in the tree and hashes to the
   recorded digest — the ``sha256sum -c`` equivalent, in-process.
3. TCB boundary — the embedded floor below is an *external* expectation
   (deliberately duplicated from ``jev_ultrafast/tcb.py``, not imported):
   every floor file must be manifest-covered, and any file under a
   trusted package prefix present in the tree but absent from the
   manifest is unclassified trusted code — rejected.
4. TCB version — parsed as text from ``_dream/common.py`` (never
   executed); with ``--expect-tcb-version`` a drift fails closed.
5. ``VALIDATION.generated.md`` — when present, it must declare the same
   ``manifest_digest`` this run computed; a report validating a
   different artifact is stale evidence, not provenance.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

MANIFEST = "MANIFEST.sha256"
SIG_FILE = "MANIFEST.sig"
SIG_SCHEMA = "jev-manifest-sig/1"
SIG_DOMAIN = b"jev-dream/release-manifest/v1:"
VALIDATION_REPORT = "VALIDATION.generated.md"

# The trusted-base floor — an EXTERNAL expectation duplicated on purpose:
# importing ``jev_ultrafast.tcb`` would execute code from the artifact
# being verified. Prefix members are trusted by construction, so a file
# landing under one of these directories joins the trusted surface
# whether or not it is individually named.
TCB_PACKAGE_PREFIXES = (
    "jev_ultrafast/_dream/",
    "jev_ultrafast/_learning/",
)
TCB_FILES = (
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
)
_IGNORE_DIRS = {
    "__pycache__", ".pytest_cache", ".ruff_cache", ".hypothesis",
}
_TCB_VERSION_RE = re.compile(
    r'TCB_VERSION\s*=\s*"(jev-ultrafast-tcb/[0-9.]+)"')
_REPORT_DIGEST_RE = re.compile(r"manifest_digest: `([0-9a-f]{64})`")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_entries(root: Path) -> dict[str, str]:
    """``name -> recorded digest`` from a sha256sum-format manifest."""
    manifest = root / MANIFEST
    if not manifest.is_file():
        return {}
    entries = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            digest, _, name = line.partition("  ")
            entries[name.strip()] = digest.strip()
    return entries


def _verify_hex(public_key_hex: str, digest_hex: str, signature_hex: str) -> bool:
    try:
        key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        key.verify(bytes.fromhex(signature_hex),
                   SIG_DOMAIN + digest_hex.encode("ascii"))
        return True
    except (InvalidSignature, ValueError):
        return False


def check(
    root: Path,
    keys: set[str],
    *,
    expect_tcb_version: str | None = None,
    require_validation: bool = False,
) -> list[str]:
    """Every release violation under ``root`` (empty = verified)."""
    root = Path(root)
    violations: list[str] = []

    # -- 1/2: manifest + signature -------------------------------------
    manifest_path = root / MANIFEST
    if not manifest_path.is_file():
        return [f"{MANIFEST}: missing — no release set to verify"]
    manifest_digest = _sha256(manifest_path)
    sig_path = root / SIG_FILE
    if not sig_path.is_file():
        violations.append(f"{SIG_FILE}: missing — unsigned release manifest")
    else:
        try:
            block = json.loads(sig_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            violations.append(f"{SIG_FILE}: unreadable ({exc})")
            block = None
        if isinstance(block, dict):
            if block.get("schema") != SIG_SCHEMA:
                violations.append(f"{SIG_FILE}: unrecognized signature block")
            elif block.get("manifest_digest") != manifest_digest:
                violations.append(
                    f"{SIG_FILE}: signed digest does not match {MANIFEST}"
                )
            elif str(block.get("key_id")) not in keys:
                violations.append(
                    f"{SIG_FILE}: manifest signed by an unexpected key"
                )
            elif not _verify_hex(
                str(block.get("key_id")), manifest_digest,
                str(block.get("signature") or ""),
            ):
                violations.append(f"{SIG_FILE}: signature verification failure")

    entries = _manifest_entries(root)
    if not entries:
        violations.append(f"{MANIFEST}: empty or unparsable")
    for name, recorded in sorted(entries.items()):
        path = root / name
        if not path.is_file():
            violations.append(f"{name}: manifest-listed file missing from tree")
        elif _sha256(path) != recorded:
            violations.append(f"{name}: content digest differs from manifest")

    # -- 3: TCB boundary ------------------------------------------------
    for name in TCB_FILES:
        if name not in entries:
            violations.append(f"{name}: trusted file absent from {MANIFEST}")
    for prefix in TCB_PACKAGE_PREFIXES:
        directory = root / prefix
        if not directory.is_dir():
            violations.append(f"{prefix}: trusted package missing from tree")
            continue
        for path in directory.rglob("*"):
            if not path.is_file() or (
                set(path.parts) & _IGNORE_DIRS
            ):
                continue
            rel = path.relative_to(root).as_posix()
            if rel not in entries:
                violations.append(
                    f"{rel}: trusted package member absent from {MANIFEST}"
                )

    # -- 4: TCB version ---------------------------------------------------
    common = root / "jev_ultrafast/_dream/common.py"
    tcb_version = None
    if common.is_file():
        match = _TCB_VERSION_RE.search(common.read_text(encoding="utf-8"))
        tcb_version = match.group(1) if match else None
    if tcb_version is None:
        violations.append("TCB_VERSION: not found in _dream/common.py")
    elif expect_tcb_version is not None and tcb_version != expect_tcb_version:
        violations.append(
            f"TCB_VERSION: expected {expect_tcb_version}, found {tcb_version}"
        )

    # -- 5: validation-report binding ------------------------------------
    report = root / VALIDATION_REPORT
    if not report.is_file():
        if require_validation:
            violations.append(
                f"{VALIDATION_REPORT}: required but absent — no generated "
                "validation evidence travels with this release"
            )
    else:
        match = _REPORT_DIGEST_RE.search(
            report.read_text(encoding="utf-8", errors="replace"))
        if not match:
            violations.append(
                f"{VALIDATION_REPORT}: does not declare the manifest digest "
                "it validates — an unbound report is not provenance"
            )
        elif match.group(1) != manifest_digest:
            violations.append(
                f"{VALIDATION_REPORT}: validates a different artifact "
                "(stale generated report)"
            )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent.parent),
                        help="release tree to verify (default: repo root)")
    parser.add_argument("--key", action="append", default=[],
                        help="accepted manifest verify key (repeatable); "
                             "JEV_MANIFEST_VERIFY_KEYS is also honored")
    parser.add_argument("--expect-tcb-version", default=None,
                        help="required TCB_VERSION value; "
                             "JEV_EXPECT_TCB_VERSION is also honored")
    parser.add_argument("--require-validation", action="store_true",
                        help="fail when VALIDATION.generated.md is absent")
    args = parser.parse_args()

    keys = {k.strip() for k in args.key if k and k.strip()}
    keys.update(
        k.strip()
        for k in os.environ.get("JEV_MANIFEST_VERIFY_KEYS", "").split(",")
        if k.strip()
    )
    if not keys:
        print("no verification key configured — refusing to accept an "
              "unpinned manifest", file=sys.stderr)
        return 1
    expect = (args.expect_tcb_version
              or os.environ.get("JEV_EXPECT_TCB_VERSION") or None)
    violations = check(
        Path(args.root), keys,
        expect_tcb_version=expect,
        require_validation=args.require_validation,
    )
    root = Path(args.root)
    if violations:
        print(f"release verification FAILED ({root})", file=sys.stderr)
        for violation in violations:
            print(f"  ✗ {violation}", file=sys.stderr)
        return 1
    print(f"release verified: {root}")
    print(f"  manifest digest …{_sha256(root / MANIFEST)[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
