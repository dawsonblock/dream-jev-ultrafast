"""Operational CLI for DREAM-Jev improvement, qualification, and rollback.

The CLI never auto-promotes a replay winner. Activation requires hash-verified
live canary traces through ``promote``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dream import DreamImprover, ExperienceStore, ExplorationPolicy, PolicyRegistry, ReplayWorld
from .dreamlearn import ChoiceModel, CostModel, CounterfactualTrials, OutcomeModel

COMMANDS = {
    "improve", "verify", "promote", "status", "rollback", "health",
    "suspend", "resume", "registry-reanchor",
}


def _add_registry_anchor_arg(cmd):
    cmd.add_argument(
        "--registry-anchor",
        help="Registry head anchor file. Registry writes checkpoint the signed "
        "state head here; reads fail when the file no longer reaches the "
        "anchored head (wholesale snapshot restore).",
    )


def _add_verify_args(cmd):
    """Store-authenticity flags shared by every command that reads evidence."""
    cmd.add_argument("--verify-key", help="Hex Ed25519 public key; verify event signatures under this key")
    cmd.add_argument(
        "--verify-keys",
        help="Comma- or space-separated trusted hex Ed25519 public keys (key rotation; also JEV_EVIDENCE_VERIFY_KEYS)",
    )
    cmd.add_argument(
        "--require-signatures",
        action="store_true",
        help="Reject unsigned events (requires a verification key)",
    )
    cmd.add_argument(
        "--anchor",
        help="Chain-head anchor file. Appends checkpoint the signed head here; "
        "reads fail when the log no longer reaches the anchored head (tail truncation).",
    )


def _store(args):
    return ExperienceStore(
        args.experience,
        verify_key=getattr(args, "verify_key", None),
        verify_keys=(getattr(args, "verify_keys", None) or "").replace(",", " ").split(),
        require_signatures=getattr(args, "require_signatures", False),
        anchor_path=getattr(args, "anchor", None),
    )


def _registry(args):
    return PolicyRegistry(
        args.registry,
        anchor_path=getattr(args, "registry_anchor", None),
    )


def build_parser():
    parser = argparse.ArgumentParser(description="DREAM-Jev replay improvement and policy qualification controls.")
    sub = parser.add_subparsers(dest="command", required=True)

    improve = sub.add_parser("improve", help="Replay experience and propose a bounded exploration-policy update")
    improve.add_argument("experience", help="Path to Jev DREAM JSONL experience store")
    improve.add_argument("--report", default="dream-report.json", help="Write replay report to this JSON path")
    improve.add_argument("--registry", help="Optional policy registry. Its active policy is used as the baseline.")
    _add_registry_anchor_arg(improve)
    improve.add_argument("--stage", action="store_true", help="Stage a replay-approved changed policy in --registry")
    improve.add_argument(
        "--cost-model",
        action="store_true",
        help="Fit a learned cost model on the store and use it to prioritize among replay-qualified candidates",
    )
    improve.add_argument(
        "--outcome-model",
        action="store_true",
        help="Fit a coarse outcome model and annotate candidates with predicted progress (never gates)",
    )
    improve.add_argument(
        "--choice-model",
        action="store_true",
        help="Fit the uncertainty-aware choice prior; annotates candidates with "
        "policy-dependent predictions and counterfactual proposals (never gates)",
    )
    _add_verify_args(improve)

    verify = sub.add_parser("verify", help="Verify the experience-store hash chain and print its evidence head")
    verify.add_argument("experience")
    _add_verify_args(verify)

    promote = sub.add_parser("promote", help="Promote the staged policy from bound live-canary evidence")
    promote.add_argument("experience")
    promote.add_argument("--registry", required=True)
    _add_registry_anchor_arg(promote)
    promote.add_argument(
        "--baseline-digest",
        help="Explicit baseline policy digest; must match the staged policy parent digest",
    )
    promote.add_argument(
        "--baseline-since-ms",
        type=int,
        help="Only count baseline runs finishing at or after this timestamp (matched-time canary window)",
    )
    _add_verify_args(promote)

    status = sub.add_parser("status", help="Show registry state")
    status.add_argument("--registry", required=True)
    _add_registry_anchor_arg(status)

    rollback = sub.add_parser("rollback", help="Roll back to a prior active policy")
    rollback.add_argument("--registry", required=True)
    rollback.add_argument("--digest", help="Specific prior digest; defaults to most recent")
    _add_registry_anchor_arg(rollback)

    health = sub.add_parser("health", help="Evaluate recent active-policy traces for post-promotion drift")
    health.add_argument("experience")
    health.add_argument("--registry", required=True)
    _add_registry_anchor_arg(health)
    health.add_argument("--recent-tasks", type=int, default=20)
    health.add_argument("--suspend-on-fail", action="store_true")
    _add_verify_args(health)

    suspend = sub.add_parser("suspend", help="Suspend the active learned policy and fall back to baseline")
    suspend.add_argument("--registry", required=True)
    suspend.add_argument("--reason", required=True)
    _add_registry_anchor_arg(suspend)

    resume = sub.add_parser("resume", help="Resume a suspended active policy")
    resume.add_argument("--registry", required=True)
    _add_registry_anchor_arg(resume)

    reanchor = sub.add_parser(
        "registry-reanchor",
        help="Re-anchor the registry state head after a confirmed anchor gap",
    )
    reanchor.add_argument("--registry", required=True)
    reanchor.add_argument("--registry-anchor", required=True, help="Anchor file to re-write")
    return parser


def _json(value):
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Backward compatibility with v0.3: ``jev-dream EXPERIENCE ...``.
    if argv and argv[0] not in COMMANDS and not argv[0].startswith("-"):
        argv.insert(0, "improve")
    args = build_parser().parse_args(argv)

    if args.command == "verify":
        _json(_store(args).verify())
        return 0

    if args.command == "status":
        _json(_registry(args).load())
        return 0

    if args.command == "rollback":
        active = _registry(args).rollback(args.digest)
        _json({"rolled_back": True, "active_digest": active["digest"], "active": active})
        return 0

    if args.command == "suspend":
        active = _registry(args).suspend(args.reason)
        _json({"suspended": True, "active_digest": active["digest"], "reason": active.get("suspension_reason")})
        return 0

    if args.command == "resume":
        active = _registry(args).resume()
        _json({"resumed": True, "active_digest": active["digest"]})
        return 0

    if args.command == "registry-reanchor":
        _json({"reanchored": True, "anchor": _registry(args).reanchor()})
        return 0

    if args.command == "promote":
        store = _store(args)
        active = _registry(args).promote_from_store(
            store,
            baseline_policy_digest=args.baseline_digest,
            baseline_since_ms=args.baseline_since_ms,
        )
        _json({
            "promoted": True,
            "active_digest": active["digest"],
            "evidence_digest": active.get("canary", {}).get("evidence_digest"),
            "event_head_hash": active.get("canary", {}).get("event_head_hash"),
        })
        return 0

    if args.command == "health":
        decision = _registry(args).health_from_store(
            _store(args),
            recent_tasks=args.recent_tasks,
            suspend_on_fail=args.suspend_on_fail,
        )
        _json({
            "healthy": decision.healthy,
            "sufficient": decision.sufficient,
            "reason": decision.reason,
            "reference": decision.reference.__dict__,
            "observed": decision.observed.__dict__,
            "suspend_on_fail": args.suspend_on_fail,
        })
        return 0 if decision.healthy else 2

    store = _store(args)
    events = store.load()
    worlds = ReplayWorld.from_events(events)
    registry = _registry(args) if args.registry else None
    baseline = registry.active_policy() if registry else ExplorationPolicy()
    cost_model = CostModel.fit(events) if args.cost_model else None
    outcome_model = OutcomeModel.fit(events) if args.outcome_model else None
    choice_model = ChoiceModel.fit(events) if args.choice_model else None
    # Randomized arm assignments are causal evidence, not observational trace
    # — fit them separately and always, so the report carries what the
    # experiment layer actually measured rather than mixing trial transitions
    # into the correlational priors alone.
    trials = CounterfactualTrials.fit(events)
    report = DreamImprover().improve(
        worlds,
        baseline,
        evidence_head_hash=store.head_hash(),
        cost_model=cost_model,
        outcome_model=outcome_model,
        choice_model=choice_model,
        trials=trials,
    )
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    staged = None
    if args.stage:
        if registry is None:
            raise SystemExit("--stage requires --registry")
        staged = registry.stage(report)

    _json({
        "worlds": sum(report.split_sizes.values()),
        "split_sizes": report.split_sizes,
        "baseline": report.baseline.digest,
        "selected": report.selected.digest,
        "replay_approved": report.promotion.approved,
        "reason": report.promotion.reason,
        "world_pool_digest": report.world_pool_digest,
        "evidence_head_hash": report.evidence_head_hash,
        "live_canary_required": report.live_canary_required,
        "experiment_proposals": len(report.experiment_proposals),
        "trial_contexts": len(report.trial_estimates or {}),
        "trials_digest": report.trials_digest,
        "staged": bool(staged),
        "report": str(output),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
