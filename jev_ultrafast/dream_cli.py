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

COMMANDS = {"improve", "verify", "promote", "status", "rollback", "health", "suspend", "resume"}


def build_parser():
    parser = argparse.ArgumentParser(description="DREAM-Jev replay improvement and policy qualification controls.")
    sub = parser.add_subparsers(dest="command", required=True)

    improve = sub.add_parser("improve", help="Replay experience and propose a bounded exploration-policy update")
    improve.add_argument("experience", help="Path to Jev DREAM JSONL experience store")
    improve.add_argument("--report", default="dream-report.json", help="Write replay report to this JSON path")
    improve.add_argument("--registry", help="Optional policy registry. Its active policy is used as the baseline.")
    improve.add_argument("--stage", action="store_true", help="Stage a replay-approved changed policy in --registry")

    verify = sub.add_parser("verify", help="Verify the experience-store hash chain and print its evidence head")
    verify.add_argument("experience")

    promote = sub.add_parser("promote", help="Promote the staged policy from bound live-canary evidence")
    promote.add_argument("experience")
    promote.add_argument("--registry", required=True)
    promote.add_argument(
        "--baseline-digest",
        help="Override the baseline policy digest used for paired canary evidence",
    )

    status = sub.add_parser("status", help="Show registry state")
    status.add_argument("--registry", required=True)

    rollback = sub.add_parser("rollback", help="Roll back to a prior active policy")
    rollback.add_argument("--registry", required=True)
    rollback.add_argument("--digest", help="Specific prior digest; defaults to most recent")

    health = sub.add_parser("health", help="Evaluate recent active-policy traces for post-promotion drift")
    health.add_argument("experience")
    health.add_argument("--registry", required=True)
    health.add_argument("--recent-tasks", type=int, default=20)
    health.add_argument("--suspend-on-fail", action="store_true")

    suspend = sub.add_parser("suspend", help="Suspend the active learned policy and fall back to baseline")
    suspend.add_argument("--registry", required=True)
    suspend.add_argument("--reason", required=True)

    resume = sub.add_parser("resume", help="Resume a suspended active policy")
    resume.add_argument("--registry", required=True)
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
        _json(ExperienceStore(args.experience).verify())
        return 0

    if args.command == "status":
        _json(PolicyRegistry(args.registry).load())
        return 0

    if args.command == "rollback":
        active = PolicyRegistry(args.registry).rollback(args.digest)
        _json({"rolled_back": True, "active_digest": active["digest"], "active": active})
        return 0

    if args.command == "suspend":
        active = PolicyRegistry(args.registry).suspend(args.reason)
        _json({"suspended": True, "active_digest": active["digest"], "reason": active.get("suspension_reason")})
        return 0

    if args.command == "resume":
        active = PolicyRegistry(args.registry).resume()
        _json({"resumed": True, "active_digest": active["digest"]})
        return 0

    if args.command == "promote":
        store = ExperienceStore(args.experience)
        active = PolicyRegistry(args.registry).promote_from_store(
            store,
            baseline_policy_digest=args.baseline_digest,
        )
        _json({
            "promoted": True,
            "active_digest": active["digest"],
            "evidence_digest": active.get("canary", {}).get("evidence_digest"),
            "event_head_hash": active.get("canary", {}).get("event_head_hash"),
        })
        return 0

    if args.command == "health":
        decision = PolicyRegistry(args.registry).health_from_store(
            ExperienceStore(args.experience),
            recent_tasks=args.recent_tasks,
            suspend_on_fail=args.suspend_on_fail,
        )
        _json({
            "healthy": decision.healthy,
            "reason": decision.reason,
            "reference": decision.reference.__dict__,
            "observed": decision.observed.__dict__,
            "suspend_on_fail": args.suspend_on_fail,
        })
        return 0 if decision.healthy else 2

    store = ExperienceStore(args.experience)
    events = store.load()
    worlds = ReplayWorld.from_events(events)
    registry = PolicyRegistry(args.registry) if args.registry else None
    baseline = registry.active_policy() if registry else ExplorationPolicy()
    report = DreamImprover().improve(worlds, baseline, evidence_head_hash=store.head_hash())
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
        "staged": bool(staged),
        "report": str(output),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
