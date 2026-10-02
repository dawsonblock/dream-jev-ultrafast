"""Latency benchmarks for hot paths (qualification §50).

Measures p50/p95/p99/max for the per-decision and evidence operations.
Browser paths run only when BU_CDP_URL is reachable.

    uv run python scripts/bench.py [--iters 200]
    BU_CDP_URL=http://127.0.0.1:9222 uv run python scripts/bench.py
"""

import argparse
import tempfile
import time
from pathlib import Path


def _percentiles(samples):
    samples = sorted(samples)
    n = len(samples)
    def pct(p):
        return samples[min(n - 1, int(p * n))] * 1000
    return f"p50={pct(.5):.1f}ms p95={pct(.95):.1f}ms p99={pct(.99):.1f}ms max={samples[-1]*1000:.1f}ms"


def _bench(label, fn, iters):
    samples = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t)
    print(f"{label:<34}{_percentiles(samples)}  (n={iters})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
    from test_learned import _run_events, _transition

    from jev_ultrafast.dream import ExperienceStore, PolicyRegistry
    from jev_ultrafast.dreamlearn import CounterfactualTrials
    from jev_ultrafast.policy import DefaultActionPolicy, classify_effect

    # Effect classification — per-candidate, every decision cycle.
    action = {"kind": "fill", "label": "Search", "role": "searchbox",
              "ctx": {"form": True}}
    _bench("classify_effect", lambda: classify_effect(action), args.iters)
    policy = DefaultActionPolicy()
    _bench("authority assess", lambda: policy.assess(action), args.iters)

    # Evidence append/verify on a realistic store.
    tmp = Path(tempfile.mkdtemp(prefix="jev-bench-"))
    store = ExperienceStore(tmp / "events.jsonl")
    for i in range(2000):
        for e in _run_events(f"r{i}", [_transition(selected="a")]):
            store.append(e)
    counter = {"i": 0}
    def _append():
        counter["i"] += 1
        store.append({"event": "transition", "run_id": f"b{counter['i']}",
                      "task_key": "t", "state": "S",
                      "selected": {"id": "a"}, "page_changed": True})
    _bench("evidence append (fsync)", _append, min(args.iters, 100))
    _bench("evidence verify (6k-event chain)", store.verify, 5)

    # Causal resolve over a fitted trial set.
    trial_events = []
    for i in range(40):
        for e in _run_events(f"t{i}", [_transition(selected="a")]):
            trial_events.append(e)
    trials = CounterfactualTrials.fit(trial_events)
    _bench("causal resolve",
           lambda: trials.resolve(model_kind="click", proposal_kind="click"),
           args.iters)

    # Registry read path (lock + file check + anchor verification).
    registry_path = tmp / "registry.json"
    _bench("registry load", lambda: PolicyRegistry(registry_path).load(),
           min(args.iters, 100))

    # Browser paths only when a CDP endpoint is reachable.
    import os
    import urllib.request
    cdp = os.environ.get("BU_CDP_URL", "http://127.0.0.1:9222")
    try:
        urllib.request.urlopen(cdp + "/json/version", timeout=1)
    except OSError:
        print("(browser paths skipped — no CDP endpoint)")
        return 0
    from jev_ultrafast.browser import Browser
    browser = Browser(cdp)
    browser.evaluate("document.body.innerHTML=" + repr(
        '<button id="b1">Go</button><input id="i1" aria-label="Site">'))
    _bench("observe (snapshot)", lambda: browser.observe(screenshot=False),
           min(args.iters, 100))
    page = browser.observe(screenshot=False)
    click = next(a for a in page["actions"] if a["kind"] == "click")
    def _atomic_click():
        p = browser.observe(screenshot=False)
        browser.act(click, p)
    _bench("atomic click (observe+act)", _atomic_click, min(args.iters, 50))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
