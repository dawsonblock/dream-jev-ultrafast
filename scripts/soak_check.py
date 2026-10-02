"""Sustained-duration soak for the execution + evidence layers (§49).

Round-robins the existing probes — nav-race, TOCTOU churn, crash injection,
evidence append — until the wall-clock deadline, so latent failures that only
surface under accumulated runtime state (leaked targets, lock contention,
anchor drift, accumulating chains) get hours of exposure rather than a fixed
iteration count.

    uv run python scripts/soak_check.py --minutes 1440    # the 24h gate
    uv run python scripts/soak_check.py --minutes 10      # smoke

Each round runs the probes as subprocesses (fresh stores, clean Chrome state)
and accumulates per-probe pass/fail tallies. Any probe failure is reported
immediately and fails the run; the final line summarizes rounds and wall time.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def _round(i: int, results: dict) -> bool:
    """One soak round: a slice of each probe. Returns False on any failure."""
    probes = [
        ("nav-race", [PY, "scripts/race_check.py", "--nav-iterations", "100",
                      "--churn-iterations", "0"]),
        ("churn", [PY, "scripts/race_check.py", "--nav-iterations", "0",
                   "--churn-iterations", "200"]),
        ("crash", [PY, "scripts/crash_check.py", "--iterations", "10",
                   "--writers", "4"]),
        ("fuzz", [PY, "scripts/fuzz_check.py", "--cases", "5000"]),
    ]
    ok = True
    for name, cmd in probes:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        results[name]["rounds"] += 1
        if proc.returncode == 0:
            results[name]["passed"] += 1
        else:
            results[name]["failed"] += 1
            ok = False
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-4:]
            print(f"soak: round {i} probe {name} FAILED:", flush=True)
            for line in tail:
                print(f"      {line}", flush=True)
    return ok


def _preflight() -> bool:
    """Fail fast when the environment cannot run the probes at all.

    Without this a soak launched against a dead/missing browser fails every
    round quietly for the full deadline — wall-clock spent on nothing.
    """
    proc = subprocess.run(
        [PY, "scripts/race_check.py", "--nav-iterations", "1",
         "--churn-iterations", "0"],
        cwd=ROOT, capture_output=True, text=True)
    if proc.returncode == 0:
        return True
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
    print("soak: preflight FAILED — probe environment is not up:", flush=True)
    for line in tail:
        print(f"      {line}", flush=True)
    print("soak: start Chrome with remote debugging (or set "
          "BU_CDP_URL=http://127.0.0.1:9222), then retry", flush=True)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=10)
    parser.add_argument("--max-consecutive-failures", type=int, default=3,
                        help="abort the soak after this many fully-failed "
                             "rounds in a row (default 3)")
    args = parser.parse_args()

    if not _preflight():
        return 1

    deadline = time.monotonic() + args.minutes * 60
    results = {n: {"rounds": 0, "passed": 0, "failed": 0}
               for n in ("nav-race", "churn", "crash", "fuzz")}
    i = 0
    ok = True
    consecutive_failures = 0
    while time.monotonic() < deadline:
        i += 1
        round_ok = _round(i, results)
        ok = round_ok and ok
        consecutive_failures = 0 if round_ok else consecutive_failures + 1
        if consecutive_failures >= args.max_consecutive_failures:
            print(f"soak: ABORT — {consecutive_failures} consecutive rounds "
                  f"with probe failures; the environment is dead, not drifting",
                  flush=True)
            break
    mins = (time.monotonic() - (deadline - args.minutes * 60)) / 60
    summary = ", ".join(
        f"{n}: {r['passed']}/{r['rounds']} ok" for n, r in results.items())
    print(f"soak: {i} rounds in {mins:.1f}min — {summary}")
    if not ok:
        print("soak: FAIL — probe failures above")
        return 1
    print("soak: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
