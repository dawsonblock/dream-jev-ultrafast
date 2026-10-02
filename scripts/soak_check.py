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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=10)
    args = parser.parse_args()

    deadline = time.monotonic() + args.minutes * 60
    results = {n: {"rounds": 0, "passed": 0, "failed": 0}
               for n in ("nav-race", "churn", "crash", "fuzz")}
    i = 0
    ok = True
    while time.monotonic() < deadline:
        i += 1
        ok = _round(i, results) and ok
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
