"""Staged qualification pipeline (qualification §64).

Runs the qualification gates in layer order and emits a ``jev-qualify/3``
report — the single artifact a release decision reads. The report separates
three claims that used to be conflated in ``overall``:

- ``validation_status`` — did the stages run here pass? (the old ``overall``)
- ``provenance_status`` — is the source manifest signed and verified under
  pinned keys? (``unsigned`` | ``verified`` | ``failed``)
- ``release_qualified`` — the release verdict. Always ``false`` for a bounded
  run: bounded validation is not a release profile. ``--full`` requires every
  listed condition — signed+verified manifest under pinned keys, every
  required stage run and passed, the report itself signed, and the signature
  verifying under a pinned accepted key — else ``release_blockers`` names
  what failed closed.

``--sign``/``JEV_QUALIFY_SIGNING_KEY`` binds the report to a release
authority (``jev-qualify-sig/1``); ``--verify-report`` checks a signed report
fail-closed under pinned keys.

    uv run python scripts/qualify.py [--full] [--out report.json]
        [--report-md VALIDATION.generated.md] [--sign SEED_HEX]
    uv run python scripts/qualify.py --verify-report report.json --key PUB_HEX

Layers:
    Q0  static integrity      manifest, compile, ruff, JS syntax, lock, build,
                              reproducible double-build
    Q1  offline correctness   full pytest suite (with committed coverage floor)
    Q2  browser execution     live Chrome guard suite (skipped, not passed,
                              when no CDP endpoint is reachable)
    Q3  adversarial           structured fuzz, crash-injection, TCB mutation
                              sweep; live race loops when Chrome is up
    Q4  statistical           causal null/power/multiplicity grid, propensity

``--full`` swaps the bounded Q3/Q4 sizes for the plan's qualification volumes
(1M fuzz cases, 1,000 crash kills, 100k-sequence grid). A stage that cannot
run on this host (no Chrome, missing tool) is reported ``skipped`` — never
``passed`` — so the report honestly separates executed evidence from absence.
Every check declares a ``requirement`` — ``mandatory`` (must run and pass),
``environment-dependent`` (needs a host capability; a skip is honest
absence, a run-and-fail is a real failure), or ``optional`` (advisory,
never gates). Undeclared checks are treated as mandatory.
Integrity is not provenance: a manifest that hashes correctly still says
nothing about who produced it, and a validation pass says nothing about
release authority.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Report signing/verification is deliberately self-contained: it uses the
# cryptography package directly and never imports jev_ultrafast, so
# --verify-report on an untrusted artifact cannot execute package code from
# the tree being qualified.
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

ROOT = Path(__file__).resolve().parent.parent
PY = [sys.executable]
_TALLY = re.compile(r"(\d+)\s+(passed|failed|skipped|deselected|xpassed|xfailed)")
REPORT_SCHEMA = "jev-qualify/4"
# Check requirement taxonomy (Phase 9): every check declares whether its
# evidence is mandatory, gated on a host capability the runner may not
# have (environment-dependent), or advisory-only. An undeclared check is
# treated as mandatory — a missing declaration must never silently
# downgrade a gate to advisory.
_MANDATORY = "mandatory"
_ENV_DEPENDENT = "environment-dependent"
_OPTIONAL = "optional"
REQUIREMENT_TAXONOMY = {
    _MANDATORY: "must run and pass — absence of a run is absence of evidence",
    _ENV_DEPENDENT: (
        "requires a host capability (Chrome CDP, node, a signed tree); "
        "a skip is honest absence, a run-and-fail is a real failure"
    ),
    _OPTIONAL: "advisory only — reported, never gates",
}
REPORT_SIG_SCHEMA = "jev-qualify-sig/1"
REPORT_SIG_DOMAIN = b"jev-dream/qualify-report/v1:"


def _report_digest(report: dict) -> str:
    """Canonical digest of the report with any signature block stripped."""
    clean = {k: v for k, v in report.items() if k != "signature"}
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sign_report(report: dict, seed_hex: str) -> dict:
    key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed_hex.strip()))
    digest = _report_digest(report)
    report["signature"] = {
        "schema": REPORT_SIG_SCHEMA,
        "kind": "qualification_report_signature",
        "report_digest": digest,
        "key_id": key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex(),
        "signature": key.sign(REPORT_SIG_DOMAIN + digest.encode("ascii")).hex(),
    }
    return report


def _report_signature_valid(report: dict, keys: set[str]) -> bool:
    """True iff the report's signature block verifies under ``keys``."""
    sig = report.get("signature") if isinstance(report, dict) else None
    if not isinstance(sig, dict) or sig.get("schema") != REPORT_SIG_SCHEMA:
        return False
    digest = _report_digest(report)
    if sig.get("report_digest") != digest or sig.get("key_id") not in keys:
        return False
    try:
        Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(sig["key_id"])
        ).verify(
            bytes.fromhex(str(sig.get("signature"))),
            REPORT_SIG_DOMAIN + digest.encode("ascii"),
        )
        return True
    except (InvalidSignature, ValueError):
        return False


def _verify_report(path: Path, keys: set[str]) -> int:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"report unreadable ({exc})", file=sys.stderr)
        return 1
    sig = report.get("signature") if isinstance(report, dict) else None
    if not isinstance(sig, dict) or sig.get("schema") != REPORT_SIG_SCHEMA:
        print("report is unsigned or has an unrecognized signature block",
              file=sys.stderr)
        return 1
    digest = _report_digest(report)
    if sig.get("report_digest") != digest:
        print("report digest mismatch — report was modified after signing",
              file=sys.stderr)
        return 1
    if sig.get("key_id") not in keys:
        print("report signed by an unexpected key", file=sys.stderr)
        return 1
    if not _report_signature_valid(report, keys):
        print("report signature verification failure", file=sys.stderr)
        return 1
    print(f"report verified under key {str(sig['key_id'])[:16]}…"
          f" — release_qualified={report.get('release_qualified')}")
    return 0


def _cdp_up() -> bool:
    url = os.environ.get("BU_CDP_URL", "http://127.0.0.1:9222")
    try:
        urllib.request.urlopen(url.rstrip("/") + "/json/version", timeout=2)
        return True
    except Exception:
        return False


def _run(cmd: list[str], timeout: int = 1800) -> dict:
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              timeout=timeout)
        return {"cmd": " ".join(cmd), "status": "passed" if proc.returncode == 0
                else "failed", "exit": proc.returncode,
                "seconds": round(time.monotonic() - t0, 1),
                "tail": (proc.stdout + proc.stderr).strip().splitlines()[-3:]}
    except subprocess.TimeoutExpired:
        return {"cmd": " ".join(cmd), "status": "failed", "exit": "timeout",
                "seconds": timeout, "tail": []}


def _double_build() -> dict:
    """Reproducibility: two consecutive ``uv build`` runs must produce
    byte-identical artifacts. Implemented in-process rather than shelling to
    ``sha256sum``/``shasum`` so the gate runs identically on macOS, Linux,
    and Windows checkouts."""
    t0 = time.monotonic()

    def build_once() -> dict | None:
        shutil.rmtree(ROOT / "dist", ignore_errors=True)
        try:
            proc = subprocess.run(["uv", "build", "-q"], cwd=ROOT,
                                  capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode != 0:
            return None
        dist = ROOT / "dist"
        try:
            return {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(dist.iterdir()) if p.is_file()
            }
        except OSError:
            return None

    first = build_once()
    if first is None:
        return {"cmd": "uv build ×2 compare", "status": "failed", "exit": 1,
                "seconds": round(time.monotonic() - t0, 1),
                "tail": ["first uv build failed"]}
    second = build_once()
    if second is None:
        return {"cmd": "uv build ×2 compare", "status": "failed", "exit": 1,
                "seconds": round(time.monotonic() - t0, 1),
                "tail": ["second uv build failed"]}
    diff = sorted(k for k in set(first) | set(second)
                  if first.get(k) != second.get(k))
    return {"cmd": "uv build ×2 compare",
            "status": "passed" if not diff else "failed",
            "exit": 0 if not diff else 1,
            "seconds": round(time.monotonic() - t0, 1),
            "tail": [f"{len(first)} artifacts byte-identical" if not diff
                     else f"nondeterministic artifacts: {diff}"]}


def _env_digest() -> dict:
    """Run env_digest.py so the report binds results to the exact machine,
    commit, tree, lockfile, and manifest it ran on — not a host name."""
    try:
        proc = subprocess.run(
            PY + ["scripts/env_digest.py"], cwd=ROOT,
            capture_output=True, text=True, timeout=120)
        if proc.returncode == 0:
            return json.loads(proc.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return {"error": "env_digest unavailable"}


def _manifest_provenance() -> dict:
    """Manifest identity + signature status for the report header."""
    manifest = ROOT / "MANIFEST.sha256"
    block = {
        "manifest_digest": hashlib.sha256(manifest.read_bytes()).hexdigest()
        if manifest.exists() else None,
        "manifest_signed": False,
        "signature_key_id": None,
        "signature_verified": "skipped",
    }
    sig = ROOT / "MANIFEST.sig"
    if sig.exists():
        try:
            sig_block = json.loads(sig.read_text(encoding="utf-8"))
            block["manifest_signed"] = sig_block.get("schema") == "jev-manifest-sig/1"
            block["signature_key_id"] = sig_block.get("key_id")
            if not os.environ.get("JEV_MANIFEST_VERIFY_KEYS", "").strip():
                block["signature_verified"] = "skipped — no pinned verify keys"
            else:
                # The Q0 gate runs sign_manifest --verify under the pinned
                # keys; the stage's own check record is the verdict.
                block["signature_verified"] = "delegated to Q0 gate"
        except (OSError, json.JSONDecodeError):
            block["signature_verified"] = "unreadable"
    return block


def _provenance_status(provenance: dict, q0_stage: dict | None) -> str:
    """unsigned | verified | failed — from the manifest signature *and* the
    Q0 gate's actual verification verdict, never from file existence alone."""
    if not provenance.get("manifest_signed"):
        return "unsigned"
    verdict = None
    for check in (q0_stage or {}).get("checks", ()):
        if "sign_manifest.py" in str(check.get("cmd")):
            verdict = check.get("status")
    if verdict == "passed":
        return "verified"
    if verdict == "skipped":
        return "unsigned"  # an unverified signature is no provenance
    return "failed"


# Stable blocker slugs per check command — the report names *what* failed or
# was absent, not just which stage box contained it.
_CHECK_SLUGS = (
    ("update_manifest.py", "manifest_integrity"),
    ("sign_manifest.py", "manifest_signature"),
    ("compileall", "compile"),
    ("ruff", "ruff"),
    ("node --check", "js_syntax"),
    ("uv lock", "lockfile"),
    ("×2", "reproducible_build"),
    ("uv build", "build"),
    ("check_guards.py", "browser_suite"),
    ("e2e_check.py", "browser_e2e"),
    ("race_check.py", "race_check"),
    ("fuzz_check.py", "fuzz_suite"),
    ("crash_check.py", "crash_injection"),
    ("mutate_check.py", "mutation_sweep"),
    ("keydrill_check.py", "key_drill"),
    ("simulate_causal.py", "causal_grid"),
    ("test_authority.py", "anchor_suite"),
    ("test_dream.py", "anchor_suite"),
    ("pytest", "pytest_suite"),
)


def _check_slug(cmd) -> str:
    text = cmd if isinstance(cmd, str) else " ".join(str(c) for c in cmd)
    for needle, slug in _CHECK_SLUGS:
        if needle in text:
            return slug
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")[:48] or "check"


def _release_verdict(
    stages: list[dict],
    *,
    full: bool,
    provenance: dict,
    provenance_status: str,
    report_key_id: str | None,
    accepted_keys: set[str],
) -> tuple[bool, list[str]]:
    """Release qualification is a separate claim from validation.

    Bounded mode is a validation profile — it reports ``release_qualified``
    ``false`` unconditionally. Full mode requires every release condition:
    signed manifest verified under pinned keys, every gating check run and
    passed (``mandatory`` and ``environment-dependent`` checks only —
    ``optional`` checks report but never gate; nothing skipped counts as
    evidence), and the qualification report itself signed by a key the
    caller accepts. ``release_blockers`` names each unsatisfied condition
    in deterministic order.
    """
    if not full:
        return False, ["bounded_mode"]
    blockers: list[str] = []
    if not (ROOT / "MANIFEST.sha256").exists():
        blockers.append("manifest_missing")
    if not provenance.get("manifest_signed"):
        blockers.append("unsigned_manifest")
    elif provenance_status == "failed":
        blockers.append("untrusted_manifest_key")
    elif provenance_status != "verified":
        blockers.append("manifest_signature_unverified")
    for stage in stages:
        for check in stage.get("checks", ()):
            status = check.get("status")
            # Advisory checks report but never gate; undeclared checks are
            # mandatory. Environment-dependent checks that skipped still
            # block release — a capability the host lacked is evidence the
            # release does not have, not evidence it earned.
            if check.get("requirement", _MANDATORY) == _OPTIONAL:
                continue
            if status in {"skipped", "failed"}:
                blockers.append(f"{_check_slug(check.get('cmd'))}_{status}")
    if report_key_id is None:
        blockers.append("report_unsigned")
    elif not accepted_keys:
        blockers.append("report_verify_keys_missing")
    elif report_key_id not in accepted_keys:
        blockers.append("report_key_untrusted")
    return not blockers, blockers


def _pytest_tally(stage: dict) -> dict:
    """Extract pass/fail/skip counts from the pytest check's tail lines."""
    tally = {}
    for check in stage["checks"]:
        for line in check.get("tail", []):
            for count, kind in _TALLY.findall(line):
                tally[kind] = int(count)
    return tally


def _write_markdown(report: dict, path: Path) -> None:
    """Render the machine-generated validation report.

    VALIDATION.md narrates *what* the gates are; this file is generated from
    the exact artifact the pipeline just ran — counts here are authoritative,
    prose elsewhere is commentary.
    """
    env = report["environment"]
    validation = report.get("validation_status", report.get("overall"))
    qualified = bool(report.get("release_qualified"))
    lines = [
        "# Validation report (generated)",
        "",
        "Generated by `scripts/qualify.py` — do not edit by hand. Every count",
        "below comes from the run that produced this file.",
        "",
        f"## Validation: **{str(validation).upper()}** — "
        f"Release qualified: **{'YES' if qualified else 'NO'}**",
        "",
        f"- schema: `{report['schema']}` / mode: `{report['mode']}`",
        f"- validation_status: `{validation}`",
        f"- provenance_status: `{report.get('provenance_status')}`",
        f"- release_qualified: `{qualified}`",
        "- release_blockers: "
        + (", ".join(f"`{b}`" for b in report.get("release_blockers") or [])
           or "none"),
        f"- generated_at_ms: {report['generated_at_ms']}",
        "",
        "## Provenance",
        "",
        f"- manifest_digest: `{report['provenance'].get('manifest_digest')}`",
        f"- manifest_signed: {report['provenance'].get('manifest_signed')}",
        f"- signature_key_id: `{report['provenance'].get('signature_key_id')}`",
        f"- signature_verified: {report['provenance'].get('signature_verified')}",
        f"- report_signed: {bool(report.get('signature'))}"
        + (f" (key `{str(report.get('signature', {}).get('key_id'))[:16]}…`)"
           if report.get("signature") else ""),
        "",
        "## Environment",
        "",
        f"- os: {env.get('os')} {env.get('machine')} (kernel {env.get('kernel')})",
        f"- python: {env.get('python')} · node: {env.get('node')} · chrome: {env.get('chrome')}",
        f"- git_commit: `{env.get('git_commit')}` (dirty: {env.get('git_dirty')})",
        f"- tree_digest: `{env.get('tree_digest')}`",
        f"- lock_digest: `{env.get('lock_digest')}`",
        f"- test_suite_digest: `{env.get('test_suite_digest')}`",
        f"- env_digest: `{env.get('env_digest')}`",
        "",
        "## Stages",
        "",
        "| stage | status | detail |",
        "|---|---|---|",
    ]
    for s in report["stages"]:
        tally = s.get("test_tally") or {}
        checks = s.get("checks", ())
        ran = sum(1 for c in checks if c.get("status") != "skipped")
        skipped = sum(1 for c in checks if c.get("status") == "skipped")
        checks_note = (
            f"{ran} check{'s' if ran != 1 else ''} ran"
            + (f", {skipped} skipped" if skipped else "")
        )
        detail = " ".join(f"{v} {k}" for k, v in sorted(tally.items()))
        lines.append(
            f"| {s['stage']} | {s['status']} | {checks_note} — {detail or s['description']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stage(name: str, description: str, checks: list[dict]) -> dict:
    results = []
    for check in checks:
        requirement = check.get("requirement", _MANDATORY)
        display = check["cmd"] if isinstance(check["cmd"], str) else " ".join(check["cmd"])
        if check.get("skip"):
            results.append({"cmd": display, "status": "skipped",
                            "requirement": requirement,
                            "reason": check["skip"]})
            print(f"    skip   {display[:72]} ({check['skip']})", flush=True)
            continue
        print(f"    run    {display[:72]}", flush=True)
        if "fn" in check:
            result = check["fn"]()
        else:
            result = _run(check["cmd"], check.get("timeout", 1800))
        result["requirement"] = requirement
        results.append(result)
        print(f"    {results[-1]['status']:6} ({results[-1]['seconds']}s)",
              flush=True)
    # Stage status is decided by gating checks only — an advisory check
    # failing reports the failure without failing the stage. A stage made
    # entirely of advisory checks falls back to the plain tally.
    gating = [r for r in results if r.get("requirement") != _OPTIONAL]
    decisive = gating or results
    status = ("failed" if any(r["status"] == "failed" for r in decisive)
              else "passed" if any(r["status"] == "passed" for r in decisive)
              else "skipped")
    return {"stage": name, "description": description, "status": status,
            "checks": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="plan-volume Q3/Q4 (1M fuzz, 1k kills, 100k grid)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--report-md", default=None,
                        help="also render a generated markdown report (e.g. "
                             "VALIDATION.generated.md)")
    parser.add_argument("--sign", metavar="SEED_HEX", default=None,
                        help="sign the report with this Ed25519 seed "
                             "(JEV_QUALIFY_SIGNING_KEY is also honored)")
    parser.add_argument("--verify-report", metavar="PATH", default=None,
                        help="verify a signed report and exit")
    parser.add_argument("--key", action="append", default=[],
                        help="accepted report verify key (repeatable); "
                             "JEV_QUALIFY_VERIFY_KEYS is also honored")
    args = parser.parse_args()

    if args.verify_report:
        keys = {k.strip() for k in args.key if k and k.strip()}
        keys.update(k.strip() for k in
                    os.environ.get("JEV_QUALIFY_VERIFY_KEYS", "").split(",")
                    if k.strip())
        if not keys:
            print("no verification key configured — refusing to accept an "
                  "unpinned report", file=sys.stderr)
            return 1
        return _verify_report(Path(args.verify_report), keys)

    full = args.full
    chrome = _cdp_up()
    node = shutil.which("node")

    stages = []

    print("Q0 — static integrity", flush=True)
    # Release provenance: a qualification run that will feed a release
    # decision (--full) requires MANIFEST.sig to verify under configured
    # trusted keys. In bounded/dev mode an unsigned tree is reported skipped
    # — honestly absent, never passed.
    manifest_signed = (ROOT / "MANIFEST.sig").exists()
    # A signed manifest must verify under pinned keys (a signature nobody
    # checks is not provenance); under --full an unsigned tree fails closed —
    # release qualification requires provenance, not just integrity.
    if manifest_signed or full:
        sig_check = {"cmd": PY + ["scripts/sign_manifest.py", "--verify"],
                     "requirement": _ENV_DEPENDENT}
    else:
        sig_check = {"cmd": ["python3", "scripts/sign_manifest.py", "--verify"],
                     "requirement": _ENV_DEPENDENT,
                     "skip": "unsigned tree — no release provenance to verify"}
    q0 = [
        {"cmd": PY + ["scripts/update_manifest.py", "--check"],
         "requirement": _MANDATORY},
        # The trusted computing base (Phase 17): drift inside the TCB, or
        # unclassified code under a trusted package prefix, fails closed.
        {"cmd": PY + ["scripts/check_tcb.py"],
         "requirement": _MANDATORY},
        sig_check,
        {"cmd": PY + ["-m", "compileall", "-q", "jev_ultrafast", "tests"],
         "requirement": _MANDATORY},
        {"cmd": PY + ["-m", "ruff", "check", "."],
         "requirement": _MANDATORY},
    ]
    for js in ("jev_ultrafast/snapshot.js", "jev_ultrafast/static/app.js"):
        q0.append({"cmd": [node, "--check", js],
                   "requirement": _ENV_DEPENDENT} if node
                  else {"cmd": ["node", "--check", js],
                        "requirement": _ENV_DEPENDENT,
                        "skip": "node not on PATH"})
    q0 += [
        {"cmd": ["uv", "lock", "--check"], "requirement": _MANDATORY},
        {"cmd": ["uv", "build", "-q"], "requirement": _MANDATORY},
        # Reproducibility: two consecutive builds must be byte-identical.
        # In-process so the gate needs no coreutils/BSD shasum on the host.
        {"cmd": "uv build ×2 compare", "fn": _double_build,
         "requirement": _MANDATORY},
    ]
    stages.append(_stage("Q0-static", "manifest, compile, lint, syntax, lock, "
                         "build, reproducibility", q0))

    print("Q1 — deterministic offline correctness", flush=True)
    stages.append(_stage("Q1-offline", "full pytest suite + coverage floor",
                         [{"cmd": PY + ["-m", "pytest", "-q"],
                           "requirement": _MANDATORY}]))

    print("Q2 — browser execution guards", flush=True)
    stages.append(_stage("Q2-browser", "live Chrome guard suite + E2E scenario", [
        {"cmd": PY + ["scripts/check_guards.py"],
         "timeout": 900, "requirement": _ENV_DEPENDENT} if chrome else
        {"cmd": ["python3", "scripts/check_guards.py"],
         "requirement": _ENV_DEPENDENT,
         "skip": "no CDP endpoint reachable"},
        {"cmd": PY + ["scripts/e2e_check.py"],
         "timeout": 900, "requirement": _ENV_DEPENDENT} if chrome else
        {"cmd": ["python3", "scripts/e2e_check.py"],
         "requirement": _ENV_DEPENDENT,
         "skip": "no CDP endpoint reachable"},
    ]))

    print("Q3 — adversarial / failure injection", flush=True)
    stages.append(_stage("Q3-adversarial", "fuzz, crash injection, mutation "
                         "sweep, live race loops", [
        {"cmd": PY + ["scripts/fuzz_check.py", "--cases",
                      "1000000" if full else "50000"],
         "requirement": _MANDATORY},
        {"cmd": PY + ["scripts/crash_check.py", "--iterations",
                      "1000" if full else "30"],
         "requirement": _MANDATORY},
        {"cmd": PY + ["scripts/mutate_check.py"], "timeout": 3600,
         "requirement": _MANDATORY},
        {"cmd": PY + ["scripts/keydrill_check.py"],
         "requirement": _MANDATORY},
        # Independent anchors are a qualification requirement, not optional
        # hardening: the anchor-enforcement slice must hold (rollback,
        # truncation, forged/missing anchor — all fail closed).
        {"cmd": PY + ["-m", "pytest", "-q", "--no-cov",
                      "tests/test_authority.py::test_chain_head_anchor_detects_truncated_tail",
                      "tests/test_authority.py::test_anchor_digest_tamper_stays_fatal",
                      "tests/test_authority.py::test_missing_anchor_and_forged_anchor_fail_closed",
                      "tests/test_dream.py::test_registry_anchor_detects_snapshot_restore",
                      "tests/test_dream.py::test_registry_anchor_missing_and_unsigned_fail_closed",
                      "tests/test_dream.py::test_registry_anchor_detects_deleted_registry"],
         "requirement": _MANDATORY},
        {"cmd": PY + ["scripts/race_check.py", "--nav-iterations",
                      "1000" if full else "200", "--churn-iterations", "0"],
         "timeout": 3600, "requirement": _ENV_DEPENDENT}
            if chrome else {"cmd": ["race_check --nav-iterations"],
                            "requirement": _ENV_DEPENDENT,
                            "skip": "no CDP endpoint reachable"},
        {"cmd": PY + ["scripts/race_check.py", "--churn-iterations",
                      "10000" if full else "500", "--nav-iterations", "0"],
         "timeout": 3600, "requirement": _ENV_DEPENDENT}
            if chrome else {"cmd": ["race_check --churn-iterations"],
                            "requirement": _ENV_DEPENDENT,
                            "skip": "no CDP endpoint reachable"},
    ]))

    print("Q4 — statistical / recursive-improvement", flush=True)
    stages.append(_stage("Q4-statistical", "causal grid + propensity", [
        {"cmd": PY + ["scripts/simulate_causal.py", "--sequences",
                      "100000" if full else "2000"], "timeout": 7200,
         "requirement": _MANDATORY},
    ]))

    overall = ("failed" if any(s["status"] == "failed" for s in stages)
               else "passed" if all(s["status"] == "passed" for s in stages)
               else "partial")
    for s in stages:
        s["test_tally"] = _pytest_tally(s)
    provenance = _manifest_provenance()
    provenance_status = _provenance_status(
        provenance, next((s for s in stages if s["stage"] == "Q0-static"), None)
    )
    # Release authority: the report must be signed AND its signature must
    # verify under a key the caller pinned as accepted — a signature over
    # the wrong key, or one nobody can check, is not provenance.
    seed = args.sign or os.environ.get("JEV_QUALIFY_SIGNING_KEY", "").strip()
    accepted_keys = {k.strip() for k in args.key if k and k.strip()}
    accepted_keys.update(
        k.strip()
        for k in os.environ.get("JEV_QUALIFY_VERIFY_KEYS", "").split(",")
        if k.strip()
    )
    report_key_id = None
    if seed:
        report_key_id = Ed25519PrivateKey.from_private_bytes(
            bytes.fromhex(seed.strip())
        ).public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    release_qualified, blockers = _release_verdict(
        stages,
        full=full,
        provenance=provenance,
        provenance_status=provenance_status,
        report_key_id=report_key_id,
        accepted_keys=accepted_keys,
    )
    report = {
        "schema": REPORT_SCHEMA,
        "mode": "full" if full else "bounded",
        "generated_at_ms": int(time.time() * 1000),
        # ``overall`` is retained as the deprecated alias of
        # ``validation_status`` for one release; consumers must migrate.
        "overall": overall,
        "validation_status": overall,
        "provenance_status": provenance_status,
        "release_qualified": release_qualified,
        "release_blockers": blockers,
        "release_required": bool(full),
        "requirement_taxonomy": dict(REQUIREMENT_TAXONOMY),
        "environment": _env_digest(),
        "provenance": provenance,
        "stages": [{k: v for k, v in s.items() if k != "checks"}
                   | {"checks": [{k: v for k, v in c.items() if k != "tail"}
                                 for c in s["checks"]]}
                   for s in stages],
    }
    # The report binds source artifact + environment + the release verdict;
    # signing binds all of it to a release authority — a detached or mutated
    # report must not be readable as qualified evidence for some other tree.
    if seed:
        _sign_report(report, seed)
        if accepted_keys and not _report_signature_valid(report, accepted_keys):
            # A just-produced signature that fails verification is a defect,
            # not a release: report honestly what happened — unsigned, with
            # the blocker recorded — rather than emitting broken evidence.
            report.pop("signature", None)
            report["release_qualified"] = False
            if "report_signature_invalid" not in report["release_blockers"]:
                report["release_blockers"].append("report_signature_invalid")
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"\nreport → {args.out}")
    if args.report_md:
        _write_markdown(report, Path(args.report_md))
        print(f"validation report → {args.report_md}")
    print(f"\nvalidation: {overall.upper()}")
    print(f"release qualified: {'YES' if report['release_qualified'] else 'NO'}")
    for blocker in report["release_blockers"]:
        print(f"  blocker: {blocker}")
    for s in stages:
        print(f"  {s['status']:7} {s['stage']} — {s['description']}")
    # Bounded runs exit on validation alone; --full is a release profile —
    # release qualification failure is a non-zero exit even when every stage
    # executed cleanly.
    if full:
        return 0 if report["release_qualified"] else 1
    return 0 if overall == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
