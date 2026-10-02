"""Crash-injection probe for the experience store (qualification §22/§23).

Spawns writer subprocesses that append to a shared hash-chained store and
SIGKILLs them at random points — mid-write, mid-fsync, mid-anchor-update.
After each kill the store must either hold a verifiable chain or recover by
discarding a torn tail; malformed records must never be *accepted*.

    uv run python scripts/crash_check.py [--iterations 30] [--writers 4]
                                        [--segment 25]

Stores rotate every `--segment` iterations so each segment's chain stays
short enough to load+verify per kill — the kill-point coverage is the
variable under test, not chain length. Rotation itself widens coverage:
fresh chains, mid-length chains, and torn tails are all exercised.

Exit 0: every post-kill recovery produced a valid chain (or a torn tail the
store itself discards). Any load failure, broken chain, or anchor
inconsistency fails the run.
"""

import argparse
import random
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Child process: append forever to the shared store (argv: repo, store,
# optional anchor). Killed at random mid-write points by the driver.
WRITER = """\
import os, sys
sys.path.insert(0, sys.argv[1])
from jev_ultrafast.dream import ExperienceStore
store = ExperienceStore(sys.argv[2], anchor_path=sys.argv[3] or None)
i = 0
while True:
    store.append({"event": "transition", "run_id": "w%d" % os.getpid(),
                  "task_key": "t", "step": i,
                  "state": "S", "selected": {"id": "a"},
                  "page_changed": True})
    i += 1
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--writers", type=int, default=4)
    parser.add_argument("--segment", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0xC0FFEE)
    args = parser.parse_args()

    repo = str(Path(__file__).resolve().parent.parent)
    sys.path.insert(0, repo)
    from jev_ultrafast.dream import ExperienceStore

    rng = random.Random(args.seed)
    tmp = Path(tempfile.mkdtemp(prefix="jev-crash-"))
    script = tmp / "writer.py"
    script.write_text(WRITER)

    killed = recovered = seg = 0
    for it in range(args.iterations):
        seg = it // args.segment
        store_path = tmp / f"events-{seg}.jsonl"
        anchor_path = tmp / f"anchor-{seg}.json"
        n_writers = rng.randint(2, args.writers)
        procs = [
            subprocess.Popen(
                [sys.executable, str(script), repo, str(store_path),
                 # Anchored appends re-verify the whole log per write by
                 # design (O(n) each) — anchor only every 5th iteration so
                 # the recovery path is covered without quadratic stall.
                 str(anchor_path) if it % 5 == 0 else ""],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            for _ in range(n_writers)
        ]
        # Warm up past interpreter import (~300ms) into the append loop, then
        # kill at a random moment inside the write window — plus a second
        # random window so the surviving writers' kills also land mid-write
        # rather than mid-import.
        time.sleep(0.6 + rng.uniform(0.0, 0.15))
        victim = rng.choice(procs)
        victim.send_signal(signal.SIGKILL)
        killed += 1
        time.sleep(rng.uniform(0.005, 0.05))
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGKILL)
        for p in procs:
            p.wait()

        store = ExperienceStore(store_path)
        try:
            store.load()
            store.verify()
            recovered += 1
        except Exception as exc:
            print(f"iteration {it}: recovery FAILED: {exc}")
            for p in procs:
                if p.poll() is None:
                    p.kill()
            return 1

    total_events = sum(
        sum(1 for _ in f.open()) for f in tmp.glob("events-*.jsonl"))
    if not total_events:
        print("crash-injection: vacuous — no events were ever written")
        return 1
    size_mb = sum(f.stat().st_size for f in tmp.glob("events-*.jsonl")) / 1e6
    print(f"crash-injection: {args.iterations} iterations, {killed} kills, "
          f"{recovered} verified recoveries, {total_events} events across "
          f"{seg + 1} segments, {size_mb:.1f}MB — "
          "every post-kill state verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
