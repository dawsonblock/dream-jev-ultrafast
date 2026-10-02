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
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = [sys.executable]


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
    args = parser.parse_args()

    full = args.full
    chrome = _cdp_up()
    node = shutil.which("node")

    stages = []

    print("Q0 — static integrity", flush=True)
    q0 = [
        {"cmd": PY + ["scripts/update_manifest.py", "--check"]},
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
    stages.append(_stage("Q2-browser", "live Chrome guard suite", [
        {"cmd": PY + ["scripts/check_guards.py"],
         "timeout": 900} if chrome else
        {"cmd": ["python3", "scripts/check_guards.py"],
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
    report = {
        "schema": "jev-qualify/1",
        "mode": "full" if full else "bounded",
        "generated_at_ms": int(time.time() * 1000),
        "overall": overall,
        "stages": [{k: v for k, v in s.items() if k != "checks"}
                   | {"checks": [{k: v for k, v in c.items() if k != "tail"}
                                 for c in s["checks"]]}
                   for s in stages],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"\nreport → {args.out}")
    print(f"\nqualification: {overall.upper()}")
    for s in stages:
        print(f"  {s['status']:7} {s['stage']} — {s['description']}")
    return 0 if overall == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
