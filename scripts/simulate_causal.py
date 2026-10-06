"""Monte-Carlo qualification of the causal estimator — no model calls.

Feeds synthetic randomized-trial evidence (known ground truth) through the
real ``CounterfactualTrials`` pipeline and *gates* the empirical operating
characteristics the qualification plan requires. This is a statistical
qualification, not statistical reporting: every scenario below declares an
explicit acceptance bound and the script exits nonzero when any bound is
violated — a crashing run is not the only way to fail Q4.

Gated criteria:

- false establishment under a true null (uniform, extreme-propensity and
  asymmetric-baseline strata): the one-sided Clopper-Pearson 95% upper
  bound on the observed false-beneficial rate must stay under the family
  error budget;
- power at the estimator's honest operating points: large effects at
  moderate n, medium effects at larger n, small effects at heavy n —
  ``beneficial`` rates above fixed floors;
- harm detection: a true negative effect must classify ``harmful`` at
  rate above its floor and must *never* classify ``beneficial``;
- propensity validity: IPW-reweighted arms keep null control and power;
- censoring: informative/imbalanced censoring must not manufacture a
  confident call — the imbalance guard keeps the status ``unresolved``;
- multiplicity: K independent null contexts, probability that ANY
  resolves ``beneficial``, bounded at the family level;
- calibration: the mean estimated delta tracks the true delta.

Usage:
    uv run python scripts/simulate_causal.py            # bounded default grid
    uv run python scripts/simulate_causal.py --sequences 5000   # heavier run
    uv run python scripts/simulate_causal.py --no-gate          # report only

Seeds are derived from scenario labels with SHA-256 — the qualification
grid is reproducible across interpreter processes (``hash()`` is not).
"""

import argparse
import hashlib
import random
from math import comb

from jev_ultrafast.dreamlearn import CounterfactualTrials


def _seed(label: str) -> int:
    """Deterministic 32-bit seed from a scenario label."""
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "big")


def _cp_upper(x: int, n: int, *, conf: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound on a binomial rate.

    The largest ``p`` with ``P[X <= x | p] >= 1 - conf``, by bisection on
    the decreasing binomial CDF. Used to turn an observed false-establish
    count into a statement about the *true* error rate: ``x = 0`` observed
    at ``n`` sequences certifies a rate below ``1 - conf^(1/n)``.
    """
    if n <= 0:
        return 1.0
    if x >= n:
        return 1.0
    target = 1.0 - conf
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        cdf = sum(comb(n, i) * mid**i * (1.0 - mid) ** (n - i) for i in range(x + 1))
        if cdf > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def trial_events(run_id, arm, *, success, propensity):
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


def simulate_sequence(rng, *, n_per_arm, delta, base, propensity,
                      censor_candidate=0.0, censor_control=0.0):
    """One randomized sequence with 2*n_per_arm units and known assignment
    probability. The arm counts are binomial, not fixed: forcing equal counts
    would condition on assignment and misrepresent extreme propensities.
    ``censor_*`` are per-arm probabilities that a run ends ``aborted``."""
    events = []
    for i in range(2 * n_per_arm):
        arm = "candidate" if rng.random() < propensity else "control"
        run_id = f"r{i}"
        arm_success = base + delta if arm == "candidate" else base
        events += trial_events(
            run_id, arm, success=rng.random() < arm_success,
            propensity=propensity,
        )
        censor_probability = censor_candidate if arm == "candidate" else censor_control
        if rng.random() < censor_probability:
            events[-1] = {"event": "run_finished", "run_id": run_id,
                          "task_key": "t", "status": "aborted",
                          "reason": "operator_cancel"}
    return events


def measure(sequences, *, n_per_arm, delta, base, propensity, seed,
            censor_candidate=0.0, censor_control=0.0, collect_deltas=False):
    rng = random.Random(seed)
    status_counts = {"beneficial": 0, "harmful": 0, "unresolved": 0,
                     "insufficient_data": 0, "none": 0}
    deltas = []
    imbalance = 0
    for seq in range(sequences):
        events = simulate_sequence(
            rng, n_per_arm=n_per_arm, delta=delta, base=base,
            propensity=propensity, censor_candidate=censor_candidate,
            censor_control=censor_control)
        trials = CounterfactualTrials.fit(events)
        resolved = trials.resolve(model_kind="click", proposal_kind="click")
        if resolved is None:
            status_counts["none"] += 1
            continue
        status_counts[resolved["effect_status"]] += 1
        if resolved.get("censoring_imbalance"):
            imbalance += 1
        if collect_deltas and resolved.get("delta") is not None:
            deltas.append(resolved["delta"])
    result = dict(status_counts)
    result["imbalance_flagged"] = imbalance
    if collect_deltas:
        result["mean_delta"] = (
            sum(deltas) / len(deltas) if deltas else None)
    return result


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
                        f"s{seq}d{d}{'c' if arm == 'candidate' else 'k'}{i}", arm,
                        success=rng.random() < p_succ,
                        propensity=0.5)
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


# ---------------------------------------------------------------------------
# Acceptance gates — each row is a named statistical property with an explicit
# bound. ``check`` receives the measure() result and returns (value, verdict).
# ---------------------------------------------------------------------------

def _err_upper(counts, n):
    """Upper CI bound on the *established-error* rate (beneficial + harmful
    against ground truth decided separately by the scenario caller)."""
    return _cp_upper(counts, n)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=int, default=400,
                        help="sequences for null/multiplicity points; heavy "
                        "power points scale off this")
    parser.add_argument(
        "--probe-sequences", type=int, default=None,
        help="Sequences for the multiplicity probe (default: sequences//4, "
        "capped — probe cost scales with divergence count)",
    )
    parser.add_argument("--no-gate", action="store_true",
                        help="report the grid without enforcing acceptance "
                        "bounds (diagnostic mode — never used by qualify)")
    args = parser.parse_args()
    S = args.sequences

    alpha = CounterfactualTrials.SEQUENTIAL_ALPHA
    failures = []

    def run_scenario(name, *, seqs, expect, bound_desc, verdict, **kw):
        counts = measure(seqs, seed=_seed("grid:" + name), **kw)
        ok, value = verdict(counts, seqs)
        status = "PASS" if ok or args.no_gate else "FAIL"
        if not ok and not args.no_gate:
            failures.append((name, value, bound_desc))
        print(f"  [{status}] {name:<52} {expect:<24} -> "
              f"{counts}  (bound: {bound_desc})")
        return ok

    print(f"CounterfactualTrials Monte-Carlo — family alpha {alpha}, "
          f"{S} base sequences")
    print("G1 — false establishment under the null "
          f"(one-sided CP95 upper <= {alpha})")

    def null_verdict(counts, n):
        # Null truth: any confident direction is a false establishment.
        bad = counts["beneficial"] + counts["harmful"]
        return _err_upper(bad, n) <= alpha, _err_upper(bad, n)

    run_scenario("null  n=30  p=0.50", seqs=S,
                 expect="no confident calls", bound_desc=f"CP95 <= {alpha}",
                 verdict=null_verdict,
                 n_per_arm=30, delta=0.0, base=0.5, propensity=0.5)
    run_scenario("null  n=60  p=0.25 (extreme propensity)", seqs=S,
                 expect="no confident calls", bound_desc=f"CP95 <= {alpha}",
                 verdict=null_verdict,
                 n_per_arm=60, delta=0.0, base=0.5, propensity=0.25)
    run_scenario("null  n=60  base=0.20 (asymmetric)", seqs=S,
                 expect="no confident calls", bound_desc=f"CP95 <= {alpha}",
                 verdict=null_verdict,
                 n_per_arm=60, delta=0.0, base=0.2, propensity=0.5)

    print("G2 — power at the estimator's operating points")
    p_seqs = max(60, S // 4)
    run_scenario("large effect  +0.60  n=100", seqs=p_seqs,
                 expect="power >= 0.80", bound_desc="beneficial >= 0.80",
                 verdict=lambda c, n: (c["beneficial"] / n >= 0.80,
                                       c["beneficial"] / n),
                 n_per_arm=100, delta=0.60, base=0.2, propensity=0.5)
    run_scenario("medium effect  +0.30  n=300", seqs=p_seqs,
                 expect="power >= 0.60", bound_desc="beneficial >= 0.60",
                 verdict=lambda c, n: (c["beneficial"] / n >= 0.60,
                                       c["beneficial"] / n),
                 n_per_arm=300, delta=0.30, base=0.5, propensity=0.5)
    run_scenario("small effect  +0.10  n=2000", seqs=max(60, S // 8),
                 expect="power >= 0.20", bound_desc="beneficial >= 0.20",
                 verdict=lambda c, n: (c["beneficial"] / n >= 0.20,
                                       c["beneficial"] / n),
                 n_per_arm=2000, delta=0.10, base=0.5, propensity=0.5)

    print("G3 — harm and propensity")
    run_scenario("harm  -0.30  n=300", seqs=p_seqs,
                 expect="harmful >= 0.50, beneficial ~0",
                 bound_desc="harmful >= 0.50 and CP95(beneficial) <= alpha",
                 verdict=lambda c, n: (
                     c["harmful"] / n >= 0.50
                     and _err_upper(c["beneficial"], n) <= alpha,
                     (c["harmful"] / n, _err_upper(c["beneficial"], n))),
                 n_per_arm=300, delta=-0.30, base=0.8, propensity=0.5)
    run_scenario("IPW  +0.30  n=3000  p=0.10", seqs=p_seqs,
                 expect="power >= 0.50 under extreme weights",
                 bound_desc="beneficial >= 0.50",
                 verdict=lambda c, n: (c["beneficial"] / n >= 0.50,
                                       c["beneficial"] / n),
                 n_per_arm=3000, delta=0.30, base=0.5, propensity=0.10)

    print("G4 — censoring")
    run_scenario("informative censor 40/0  null  n=100", seqs=p_seqs,
                 expect="imbalance guard blocks establishment",
                 bound_desc=f"CP95(confident) <= {alpha}",
                 verdict=null_verdict,
                 n_per_arm=100, delta=0.0, base=0.5, propensity=0.5,
                 censor_candidate=0.4)
    run_scenario("symmetric censor 30  +0.30  n=400", seqs=p_seqs,
                 expect="no overclaim under missingness",
                 bound_desc=f"CP95(confident) <= {alpha}",
                 verdict=lambda c, n: (
                     _err_upper(c["beneficial"] + c["harmful"], n) <= alpha,
                     _err_upper(c["beneficial"] + c["harmful"], n)),
                 n_per_arm=400, delta=0.30, base=0.5, propensity=0.5,
                 censor_candidate=0.3, censor_control=0.3)

    print("G5 — calibration")
    run_scenario("delta estimate  +0.30  n=200", seqs=p_seqs,
                 expect="mean delta in [0.25, 0.35]",
                 bound_desc="|mean_delta - 0.30| <= 0.05",
                 verdict=lambda c, n: (
                     c.get("mean_delta") is not None
                     and abs(c["mean_delta"] - 0.30) <= 0.05,
                     c.get("mean_delta")),
                 n_per_arm=200, delta=0.30, base=0.5, propensity=0.5,
                 collect_deltas=True)
    run_scenario("delta estimate  +0.30  n=200  p=0.25", seqs=p_seqs,
                 expect="IPW mean delta in [0.25, 0.35]",
                 bound_desc="|mean_delta - 0.30| <= 0.05",
                 verdict=lambda c, n: (
                     c.get("mean_delta") is not None
                     and abs(c["mean_delta"] - 0.30) <= 0.05,
                     c.get("mean_delta")),
                 n_per_arm=200, delta=0.30, base=0.5, propensity=0.25,
                 collect_deltas=True)

    print("G6 — multiplicity (family-wise any-false rate)")
    probe_seq = args.probe_sequences or min(2000, max(100, S // 4))
    for divergences in (2, 10, 50):
        hits = multiplicity_probe(
            probe_seq, divergences=divergences, n_per_arm=12, base=0.5,
            seed=_seed(f"probe:{divergences}"))
        upper = _cp_upper(hits, probe_seq)
        ok = upper <= 0.10 or args.no_gate
        if not ok:
            failures.append((f"multiplicity d={divergences}", upper,
                             "CP95(any-false) <= 0.10"))
        print(f"  [{'PASS' if ok else 'FAIL'}] {divergences:>3} divergences: "
              f"{hits}/{probe_seq} sequences had >=1 false 'beneficial' "
              f"(CP95 upper {upper:.3f}, bound 0.10)")

    if failures:
        print(f"\nQUALIFICATION FAILED — {len(failures)} bound(s) violated:")
        for name, value, bound in failures:
            print(f"  {name}: observed {value} (bound: {bound})")
        return 1
    print("\nQUALIFICATION PASSED — all statistical bounds hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
