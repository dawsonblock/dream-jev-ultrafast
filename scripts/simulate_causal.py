"""Monte-Carlo qualification of the causal estimator — no model calls.

Feeds synthetic randomized-trial evidence (known ground truth) through the
real ``CounterfactualTrials`` pipeline and reports the empirical behavior the
qualification plan asks about:

- false-positive rate under a true null (delta = 0)
- power to establish real effects of varying size
- behavior under unequal recorded propensities
- system-level error when many divergences are inspected (multiplicity)

Usage:
    uv run python scripts/simulate_causal.py            # bounded default grid
    uv run python scripts/simulate_causal.py --sequences 5000   # heavier run

This measures the *estimator's* operating characteristics; it is a harness,
not a unit test, because production qualification wants far more sequences
than a test suite should spend time on.
"""

import argparse
import random

from jev_ultrafast.dreamlearn import CounterfactualTrials


def trial_events(run_id, arm, *, success, propensity, rng, base):
    """One randomized trial as run_started + experiment_assigned + transition +
    run_finished, shaped exactly like the agent records it."""
    meta = {
        "experiment_id": f"x{run_id}",
        "arm": arm,
        "assignment_probability": propensity if arm == "candidate" else 1.0 - propensity,
        "proposal_id": f"p{run_id}",
        "proposal_kind": "click",
        "proposal_overlap": 1,
        "proposal_offered_rank": 0,
        "model_choice_id": f"m{run_id}",
        "model_choice_kind": "click",
        "model_choice_overlap": 0,
        "model_choice_offered_rank": 1,
        "task_key": "t",
    }
    pid, mid = meta["proposal_id"], meta["model_choice_id"]
    return [
        {"event": "run_started", "run_id": run_id, "task_key": "t", "goal": "g"},
        {"event": "experiment_assigned", "run_id": run_id, "task_key": "t",
         "experiment": meta},
        {"event": "transition", "run_id": run_id, "task_key": "t",
         "state": "S", "next_state": "D",
         "selected": {"id": pid if arm == "candidate" else mid, "kind": "click"},
         "candidates": [
             {"id": pid, "kind": "click", "label": "Go", "goal_overlap": 1},
             {"id": mid, "kind": "click", "label": "M", "goal_overlap": 0},
         ],
         "page_changed": True, "latency_ms": 0, "model_calls": 1, "tokens": 0,
         "stale_or_failure": 0, "risk_events": 0, "experiment": meta},
        {"event": "run_finished", "run_id": run_id, "task_key": "t",
         "status": "done" if success else "blocked", "verified": bool(success)},
    ]


def simulate_sequence(rng, *, n_per_arm, delta, base, propensity):
    """One experiment sequence: n_per_arm assigned to each arm under a known
    true delta (candidate arm success = base + delta)."""
    events = []
    for i in range(n_per_arm):
        events += trial_events(f"c{i}", "candidate",
                               success=rng.random() < base + delta,
                               propensity=propensity, rng=rng, base=base)
        events += trial_events(f"k{i}", "control",
                               success=rng.random() < base,
                               propensity=1.0 - propensity, rng=rng, base=base)
    return events


def measure(sequences, *, n_per_arm, delta, base, propensity, seed):
    rng = random.Random(seed)
    status_counts = {"beneficial": 0, "harmful": 0, "unresolved": 0, "none": 0}
    for seq in range(sequences):
        events = simulate_sequence(rng, n_per_arm=n_per_arm, delta=delta,
                                   base=base, propensity=propensity)
        resolved = CounterfactualTrials.fit(events).resolve(
            model_kind="click", proposal_kind="click")
        status_counts[resolved["effect_status"] if resolved else "none"] += 1
    return status_counts


def multiplicity_probe(sequences, *, divergences, n_per_arm, base, seed):
    """K independent null divergences per sequence: probability that ANY of
    them resolves 'beneficial' — the system-level false-activation rate."""
    rng = random.Random(seed)
    any_false = 0
    for seq in range(sequences):
        events = []
        # K divergences separated by task_family so each resolves its own cell.
        for d in range(divergences):
            for i in range(n_per_arm):
                for arm, p_succ in (("candidate", base), ("control", base)):
                    meta_events = trial_events(
                        f"s{seq}d{d}{arm[0]}{i}", arm,
                        success=rng.random() < p_succ,
                        propensity=0.5, rng=rng, base=base)
                    for event in meta_events:
                        # The stratum lives on the experiment meta — the
                        # assignment record's task_family is what cells key on.
                        if isinstance(event.get("experiment"), dict):
                            event["experiment"]["task_family"] = f"fam{d}"
                    events += meta_events
        trials = CounterfactualTrials.fit(events)
        for d in range(divergences):
            resolved = trials.resolve(model_kind="click", proposal_kind="click",
                                    task_family=f"fam{d}")
            if resolved and resolved.get("effect_status") == "beneficial":
                any_false += 1
                break
    return any_false


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=int, default=400)
    parser.add_argument(
        "--probe-sequences", type=int, default=None,
        help="Sequences for the multiplicity probe (default: sequences//4, "
        "capped — probe cost scales with divergence count)",
    )
    args = parser.parse_args()

    print(f"CounterfactualTrials Monte-Carlo — {args.sequences} sequences/point")
    print(f"{'grid':<46}{'beneficial':>11}{'harmful':>9}{'unresolved':>11}{'none':>6}")
    grid = [
        ("null  delta=0.00  n=12  p=0.50", dict(n_per_arm=12, delta=0.0,  base=0.5, propensity=0.5)),
        ("null  delta=0.00  n=30  p=0.50", dict(n_per_arm=30, delta=0.0,  base=0.5, propensity=0.5)),
        ("null  delta=0.00  n=30  p=0.25", dict(n_per_arm=30, delta=0.0,  base=0.5, propensity=0.25)),
        ("delta=+0.01       n=30  p=0.50", dict(n_per_arm=30, delta=0.01, base=0.5, propensity=0.5)),
        ("delta=+0.05       n=30  p=0.50", dict(n_per_arm=30, delta=0.05, base=0.5, propensity=0.5)),
        ("delta=+0.10       n=30  p=0.50", dict(n_per_arm=30, delta=0.10, base=0.5, propensity=0.5)),
        ("delta=-0.05       n=30  p=0.50", dict(n_per_arm=30, delta=-0.05, base=0.5, propensity=0.5)),
        ("delta=-0.10       n=30  p=0.50", dict(n_per_arm=30, delta=-0.10, base=0.55, propensity=0.5)),
        ("delta=+0.30       n=30  p=0.50", dict(n_per_arm=30, delta=0.30, base=0.5, propensity=0.5)),
        ("delta=+0.60       n=30  p=0.50", dict(n_per_arm=30, delta=0.60, base=0.2, propensity=0.5)),
        ("delta=-0.30       n=30  p=0.50", dict(n_per_arm=30, delta=-0.30, base=0.8, propensity=0.5)),
        ("delta=+0.30       n=30  p=0.10", dict(n_per_arm=30, delta=0.30, base=0.5, propensity=0.10)),
    ]
    for label, kw in grid:
        counts = measure(args.sequences, seed=hash(label) & 0xFFFF, **kw)
        print(f"{label:<46}{counts['beneficial']:>11}{counts['harmful']:>9}"
              f"{counts['unresolved']:>11}{counts['none']:>6}")

    print("\nMultiplicity probe (all-null divergences, per-sequence family strata)")
    probe_seq = args.probe_sequences or min(2000, max(100, args.sequences // 4))
    for divergences in (2, 5, 10, 25, 50, 100):
        hits = multiplicity_probe(probe_seq,
                                  divergences=divergences, n_per_arm=12,
                                  base=0.5, seed=divergences)
        print(f"  {divergences:>3} divergences: {hits}/{probe_seq} sequences had >=1 false "
              f"'beneficial' ({hits / probe_seq:.1%})")


if __name__ == "__main__":
    main()
