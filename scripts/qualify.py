"""Staged qualification pipeline (qualification §64).

Runs the qualification gates in layer order and emits a ``jev-qualify/1``
report — the single artifact a release decision reads.

    uv run python scripts/qualify.py [--full] [--out report.json]

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

ROOT = Path(__file__).resolve().parent.parent
PY = [sys.executable]
_TALLY = re.compile(r"(\d+)\s+(passed|failed|skipped|deselected|xpassed|xfailed)")
REPORT_SIG_SCHEMA = "jev-qualify-sig/1"
REPORT_SIG_DOMAIN = b"jev-dream/qualify-report/v1:"


def _report_digest(report: dict) -> str:
    """Canonical digest of the report with any signature block stripped."""
    clean = {k: v for k, v in report.items() if k != "signature"}
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sign_report(report: dict, seed_hex: str) -> dict:
    sys.path.insert(0, str(ROOT))
    from jev_ultrafast.signing import EvidenceSigner

    signer = EvidenceSigner.from_hex(seed_hex.strip())
    digest = _report_digest(report)
    report["signature"] = {
        "schema": REPORT_SIG_SCHEMA,
        "kind": "qualification_report_signature",
        "report_digest": digest,
        "key_id": signer.key_id,
        "signature": signer.sign_hex(digest, domain=REPORT_SIG_DOMAIN),
    }
    return report


def _verify_report(path: Path, keys: set[str]) -> int:
    sys.path.insert(0, str(ROOT))
    from jev_ultrafast.signing import verify_signature

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
    if not verify_signature(sig["key_id"], digest, str(sig.get("signature")),
                            domain=REPORT_SIG_DOMAIN):
        print("report signature verification failure", file=sys.stderr)
        return 1
    print(f"report verified under key {str(sig['key_id'])[:16]}…")
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
        except (OSError, json.JSONDecodeError):
            block["signature_verified"] = "unreadable"
    return block


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
    lines = [
        "# Validation report (generated)",
        "",
        "Generated by `scripts/qualify.py` — do not edit by hand. Every count",
        "below comes from the run that produced this file.",
        "",
        f"- schema: `{report['schema']}` / mode: `{report['mode']}`",
        f"- overall: **{report['overall']}**",
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
        detail = " ".join(f"{v} {k}" for k, v in sorted(tally.items()))
        lines.append(f"| {s['stage']} | {s['status']} | {detail or s['description']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stage(name: str, description: str, checks: list[dict]) -> dict:
    results = []
    for check in checks:
        if check.get("skip"):
            results.append({"cmd": check["cmd"], "status": "skipped",
                            "reason": check["skip"]})
            print(f"    skip   {' '.join(check['cmd'])[:72]} "
                  f"({check['skip']})", flush=True)
            continue
        print(f"    run    {' '.join(check['cmd'])[:72]}", flush=True)
        results.append(_run(check["cmd"], check.get("timeout", 1800)))
        print(f"    {results[-1]['status']:6} ({results[-1]['seconds']}s)",
              flush=True)
    status = ("failed" if any(r["status"] == "failed" for r in results)
              else "passed" if any(r["status"] == "passed" for r in results)
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
        sig_check = {"cmd": PY + ["scripts/sign_manifest.py", "--verify"]}
    else:
        sig_check = {"cmd": ["python3", "scripts/sign_manifest.py", "--verify"],
                     "skip": "unsigned tree — no release provenance to verify"}
    q0 = [
        {"cmd": PY + ["scripts/update_manifest.py", "--check"]},
        sig_check,
        {"cmd": PY + ["-m", "compileall", "-q", "jev_ultrafast", "tests"]},
        {"cmd": PY + ["-m", "ruff", "check", "."]},
    ]
    for js in ("jev_ultrafast/snapshot.js", "jev_ultrafast/static/app.js"):
        q0.append({"cmd": [node, "--check", js]} if node
                  else {"cmd": ["node", "--check", js],
                        "skip": "node not on PATH"})
    q0 += [
        {"cmd": ["uv", "lock", "--check"]},
        {"cmd": ["uv", "build", "-q"]},
        # Reproducibility: two consecutive builds must be byte-identical.
        {"cmd": ["sh", "-c",
                 "rm -rf dist && uv build -q && sha256sum dist/* > /tmp/jev-b1.sha && "
                 "rm -rf dist && uv build -q && sha256sum -c /tmp/jev-b1.sha"]},
    ]
    stages.append(_stage("Q0-static", "manifest, compile, lint, syntax, lock, "
                         "build, reproducibility", q0))

    print("Q1 — deterministic offline correctness", flush=True)
    stages.append(_stage("Q1-offline", "full pytest suite + coverage floor",
                         [{"cmd": PY + ["-m", "pytest", "-q"]}]))

    print("Q2 — browser execution guards", flush=True)
    stages.append(_stage("Q2-browser", "live Chrome guard suite + E2E scenario", [
        {"cmd": PY + ["scripts/check_guards.py"],
         "timeout": 900} if chrome else
        {"cmd": ["python3", "scripts/check_guards.py"],
         "skip": "no CDP endpoint reachable"},
        {"cmd": PY + ["scripts/e2e_check.py"],
         "timeout": 900} if chrome else
        {"cmd": ["python3", "scripts/e2e_check.py"],
         "skip": "no CDP endpoint reachable"},
    ]))

    print("Q3 — adversarial / failure injection", flush=True)
    stages.append(_stage("Q3-adversarial", "fuzz, crash injection, mutation "
                         "sweep, live race loops", [
        {"cmd": PY + ["scripts/fuzz_check.py", "--cases",
                      "1000000" if full else "50000"]},
        {"cmd": PY + ["scripts/crash_check.py", "--iterations",
                      "1000" if full else "30"]},
        {"cmd": PY + ["scripts/mutate_check.py"], "timeout": 3600},
        {"cmd": PY + ["scripts/keydrill_check.py"]},
        # Independent anchors are a qualification requirement, not optional
        # hardening: the anchor-enforcement slice must hold (rollback,
        # truncation, forged/missing anchor — all fail closed).
        {"cmd": PY + ["-m", "pytest", "-q", "--no-cov",
                      "tests/test_authority.py::test_chain_head_anchor_detects_truncated_tail",
                      "tests/test_authority.py::test_anchor_digest_tamper_stays_fatal",
                      "tests/test_authority.py::test_missing_anchor_and_forged_anchor_fail_closed",
                      "tests/test_dream.py::test_registry_anchor_detects_snapshot_restore",
                      "tests/test_dream.py::test_registry_anchor_missing_and_unsigned_fail_closed",
                      "tests/test_dream.py::test_registry_anchor_detects_deleted_registry"]},
        {"cmd": PY + ["scripts/race_check.py", "--nav-iterations",
                      "1000" if full else "200", "--churn-iterations", "0"],
         "timeout": 3600}
            if chrome else {"cmd": ["race_check --nav-iterations"],
                            "skip": "no CDP endpoint reachable"},
        {"cmd": PY + ["scripts/race_check.py", "--churn-iterations",
                      "10000" if full else "500", "--nav-iterations", "0"],
         "timeout": 3600}
            if chrome else {"cmd": ["race_check --churn-iterations"],
                            "skip": "no CDP endpoint reachable"},
    ]))

    print("Q4 — statistical / recursive-improvement", flush=True)
    stages.append(_stage("Q4-statistical", "causal grid + propensity", [
        {"cmd": PY + ["scripts/simulate_causal.py", "--sequences",
                      "100000" if full else "2000"], "timeout": 7200},
    ]))

    overall = ("failed" if any(s["status"] == "failed" for s in stages)
               else "passed" if all(s["status"] == "passed" for s in stages)
               else "partial")
    for s in stages:
        s["test_tally"] = _pytest_tally(s)
    report = {
        "schema": "jev-qualify/2",
        "mode": "full" if full else "bounded",
        "generated_at_ms": int(time.time() * 1000),
        "overall": overall,
        "environment": _env_digest(),
        "provenance": _manifest_provenance(),
        "stages": [{k: v for k, v in s.items() if k != "checks"}
                   | {"checks": [{k: v for k, v in c.items() if k != "tail"}
                                 for c in s["checks"]]}
                   for s in stages],
    }
    # The report binds source artifact + environment; signing binds it to a
    # release authority — a detached or mutated report must not be readable
    # as qualified evidence for some other tree.
    seed = args.sign or os.environ.get("JEV_QUALIFY_SIGNING_KEY", "").strip()
    if seed:
        _sign_report(report, seed)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"\nreport → {args.out}")
    if args.report_md:
        _write_markdown(report, Path(args.report_md))
        print(f"validation report → {args.report_md}")
    print(f"\nqualification: {overall.upper()}")
    for s in stages:
        print(f"  {s['status']:7} {s['stage']} — {s['description']}")
    return 0 if overall == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
