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
        -> enumerate the tree: no unmanifested file anywhere
        -> check the generated validation report binds this manifest
        -> check signed qualification evidence binds this manifest
        -> only then trust the release set

    uv run python scripts/verify_release.py --key <pubkey-hex> [--root DIR]
        [--expect-tcb-version jev-ultrafast-tcb/0.16] [--require-validation]
        [--require-qualification] [--qualify-key <pubkey-hex>]
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
5. Full-set enumeration — every regular file in the tree must be either
   manifest-listed or inside the tiny exclusion set (the manifest itself,
   its signature, the signed qualification report, the generated
   validation report, and runtime caches mirrored from
   ``update_manifest.py``). An added executable, data file, or config
   anywhere else is rejected: a signature can only vouch for a fully
   specified release set.
6. ``VALIDATION.generated.md`` — when present, it must declare the same
   ``manifest_digest`` this run computed; a report validating a
   different artifact is stale evidence, not provenance.
7. ``RELEASE_QUALIFICATION.json`` — qualification evidence is a separate
   claim from integrity: when present it must be a ``jev-qualify/*``
   report signed under a pinned qualification key
   (``--qualify-key``/``JEV_QUALIFY_VERIFY_KEYS``) and bound to this
   manifest digest; ``--require-qualification`` additionally requires
   ``release_qualified == true`` — the distinction between a *signed*
   release and a *qualified* one.
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
# The signed qualification report binds the manifest digest — like
# MANIFEST.sig it cannot be manifest-listed (the manifest's digest is not
# fixed until the manifest is), so it travels beside the release set.
QUALIFICATION_REPORT = "RELEASE_QUALIFICATION.json"
QUALIFY_SIG_SCHEMA = "jev-qualify-sig/1"
QUALIFY_SIG_DOMAIN = b"jev-dream/qualify-report/v1:"

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
# Full-set enumeration exclusions — mirrored from update_manifest.py's
# packaged-tree enumeration so both tools accept the same release set:
# VCS/runtime detritus only, plus the files that cannot be manifest-listed
# by construction (the manifest, its signature, and the generated reports
# that bind the manifest's own digest).
_ENUM_IGNORE_DIRS = _IGNORE_DIRS | {
    ".git", ".venv", "artifacts", "dist", ".jev", "mutants", "htmlcov",
    "node_modules",
}
_ENUM_IGNORE_NAMES = {
    ".env", ".DS_Store", ".coverage", "coverage.xml", "setup.cfg",
    MANIFEST, SIG_FILE, VALIDATION_REPORT, QUALIFICATION_REPORT,
}
_ENUM_IGNORE_SUFFIXES = (".pyc", ".jsonl.lock")
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


def _verify_hex(
    public_key_hex: str,
    digest_hex: str,
    signature_hex: str,
    *,
    domain: bytes = SIG_DOMAIN,
) -> bool:
    try:
        key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        key.verify(bytes.fromhex(signature_hex),
                   domain + digest_hex.encode("ascii"))
        return True
    except (InvalidSignature, ValueError):
        return False


def _report_digest(report: dict) -> str:
    """Canonical report digest — the signature covers the report with its
    ``signature`` block stripped (``qualify.py`` canonical form)."""
    clean = {k: v for k, v in report.items() if k != "signature"}
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _extra_files(root: Path, entries: dict[str, str]) -> list[str]:
    """Regular files in the tree the manifest does not cover.

    A signature over the manifest vouches for a *fully specified* release
    set — anything else present (an added script, data file, config, or
    binary) is outside the signed claim and fails closed. Exclusions are
    deliberately tiny: the manifest and its signature, the generated
    reports that bind the manifest's own digest, and VCS/runtime detritus
    mirrored from update_manifest.py's packaged enumeration.
    """
    extra = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        parts = path.relative_to(root).as_posix().split("/")
        if any(
            part in _ENUM_IGNORE_DIRS or part.endswith(".egg-info")
            for part in parts[:-1]
        ):
            continue
        name = parts[-1]
        if name in _ENUM_IGNORE_NAMES or name.endswith(_ENUM_IGNORE_SUFFIXES):
            continue
        rel = "/".join(parts)
        if rel not in entries:
            extra.append(rel)
    return sorted(extra)


def check(
    root: Path,
    keys: set[str],
    *,
    expect_tcb_version: str | None = None,
    expect_manifest_digest: str | None = None,
    require_validation: bool = False,
    require_qualification: bool = False,
    qualify_keys: set[str] | None = None,
) -> list[str]:
    """Every release violation under ``root`` (empty = verified)."""
    root = Path(root)
    violations: list[str] = []

    # -- 1/2: manifest + signature -------------------------------------
    manifest_path = root / MANIFEST
    if not manifest_path.is_file():
        return [f"{MANIFEST}: missing — no release set to verify"]
    manifest_digest = _sha256(manifest_path)
    # Anti-rollback: a validly-signed manifest is authenticity, not
    # recency — the caller may pin the exact release it approved.
    if (
        expect_manifest_digest is not None
        and manifest_digest != expect_manifest_digest.strip().lower()
    ):
        violations.append(
            f"{MANIFEST}: digest {manifest_digest} does not match pinned "
            f"release {expect_manifest_digest.strip().lower()}"
        )
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

    # -- 5: full-set enumeration -----------------------------------------
    # Every regular file in the tree must be manifest-listed or inside the
    # tiny exclusion set — an added file anywhere is outside the signed
    # claim even when it sits nowhere near the trusted prefixes.
    for rel in _extra_files(root, entries):
        violations.append(f"{rel}: present in tree but absent from {MANIFEST}")

    # -- 6: validation-report binding ------------------------------------
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

    # -- 7: qualification evidence -----------------------------------------
    # A signed release proves integrity; *qualification* is the separate
    # claim that the full gate pipeline attested this manifest. The report
    # must be signed under a pinned qualification key and bound to this
    # manifest digest — a markdown report is commentary, this is the claim.
    qpath = root / QUALIFICATION_REPORT
    if not qpath.is_file():
        if require_qualification:
            violations.append(
                f"{QUALIFICATION_REPORT}: required but absent — no signed "
                "qualification evidence travels with this release"
            )
    else:
        try:
            qreport = json.loads(qpath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            qreport = None
            violations.append(f"{QUALIFICATION_REPORT}: unreadable ({exc})")
        if isinstance(qreport, dict):
            if not str(qreport.get("schema") or "").startswith("jev-qualify/"):
                violations.append(
                    f"{QUALIFICATION_REPORT}: unrecognized qualification report"
                )
            if require_qualification and qreport.get("release_qualified") is not True:
                violations.append(
                    f"{QUALIFICATION_REPORT}: release_qualified is not true — "
                    "validation evidence is not a release verdict"
                )
            bound = (qreport.get("provenance") or {}).get("manifest_digest")
            if bound != manifest_digest:
                violations.append(
                    f"{QUALIFICATION_REPORT}: qualifies a different manifest "
                    "— stale or transplanted qualification evidence"
                )
            qkeys = set(qualify_keys or ())
            qsig = qreport.get("signature")
            if not qkeys:
                violations.append(
                    "no pinned qualification key — an unpinned report "
                    "accepts anything signed"
                )
            elif not isinstance(qsig, dict) or qsig.get("schema") != QUALIFY_SIG_SCHEMA:
                violations.append(
                    f"{QUALIFICATION_REPORT}: unsigned or unrecognized "
                    "signature block"
                )
            else:
                qdigest = _report_digest(qreport)
                if qsig.get("report_digest") != qdigest:
                    violations.append(
                        f"{QUALIFICATION_REPORT}: report digest mismatch — "
                        "modified after signing"
                    )
                elif str(qsig.get("key_id")) not in qkeys:
                    violations.append(
                        f"{QUALIFICATION_REPORT}: signed by an unexpected key"
                    )
                elif not _verify_hex(
                    str(qsig.get("key_id")),
                    qdigest,
                    str(qsig.get("signature") or ""),
                    domain=QUALIFY_SIG_DOMAIN,
                ):
                    violations.append(
                        f"{QUALIFICATION_REPORT}: signature verification failure"
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
    parser.add_argument("--expect-manifest-digest", default=None,
                        help="required MANIFEST.sha256 digest; "
                             "JEV_EXPECT_MANIFEST_DIGEST is also honored")
    parser.add_argument("--require-validation", action="store_true",
                        help="fail when VALIDATION.generated.md is absent")
    parser.add_argument("--require-qualification", action="store_true",
                        help="fail unless RELEASE_QUALIFICATION.json is "
                             "signed under a pinned key and declares "
                             "release_qualified == true")
    parser.add_argument("--qualify-key", action="append", default=[],
                        help="accepted qualification-report verify key "
                             "(repeatable); JEV_QUALIFY_VERIFY_KEYS is "
                             "also honored")
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
    qualify_keys = {k.strip() for k in args.qualify_key if k and k.strip()}
    qualify_keys.update(
        k.strip()
        for k in os.environ.get("JEV_QUALIFY_VERIFY_KEYS", "").split(",")
        if k.strip()
    )
    expect = (args.expect_tcb_version
              or os.environ.get("JEV_EXPECT_TCB_VERSION") or None)
    expect_digest = (args.expect_manifest_digest
                     or os.environ.get("JEV_EXPECT_MANIFEST_DIGEST") or None)
    violations = check(
        Path(args.root), keys,
        expect_tcb_version=expect,
        expect_manifest_digest=expect_digest,
        require_validation=args.require_validation,
        require_qualification=args.require_qualification,
        qualify_keys=qualify_keys,
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
