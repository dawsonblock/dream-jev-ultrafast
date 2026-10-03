"""_dream.canary — extracted from jev_ultrafast.dream."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from .common import _stable_hash

if TYPE_CHECKING:
    from typing import Iterable


__all__ = [
    'CanaryDecision',
    'CanaryEvidence',
    'CanaryGate',
    'CanaryMetrics',
    'CanaryRunSummary',
    '_canary_run_summaries',
    '_metrics_from_canary_runs',
    '_sign_test_p_value',
]


@dataclass(frozen=True)
class CanaryMetrics:
    tasks: int
    verified_successes: int
    failures: int = 0
    risk_events: int = 0
    latency_ms: int = 0
    actions: int = 0
    model_calls: int = 0
    tokens: int = 0
    offered_candidates: int = 0
    task_families: int = 0
    run_ids: tuple[str, ...] = ()
    task_keys: tuple[str, ...] = ()
    family_keys: tuple[str, ...] = ()
    instance_ids: tuple[str, ...] = ()

    @property
    def success_rate(self) -> float:
        return self.verified_successes / self.tasks if self.tasks else 0.0

    @property
    def failure_rate(self) -> float:
        return self.failures / self.tasks if self.tasks else 0.0

    @property
    def risk_rate(self) -> float:
        return self.risk_events / self.tasks if self.tasks else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.latency_ms / self.tasks if self.tasks else 0.0

    @property
    def avg_actions(self) -> float:
        return self.actions / self.tasks if self.tasks else 0.0

    @property
    def avg_tokens(self) -> float:
        return self.tokens / self.tasks if self.tasks else 0.0

    @classmethod
    def from_events(
        cls,
        events: Iterable[dict],
        policy_digest: str,
        *,
        max_runs: int | None = None,
        since_ms: int | None = None,
    ) -> "CanaryMetrics":
        runs = _canary_run_summaries(events, policy_digest, since_ms=since_ms)
        if max_runs is not None:
            # ``-0`` slices to the whole list, so a non-positive limit must not
            # silently mean "all runs" — it must mean none.
            limit = int(max_runs)
            runs = runs[-limit:] if limit > 0 else []
        return _metrics_from_canary_runs(runs)


@dataclass(frozen=True)
class CanaryRunSummary:
    run_id: str
    task_key: str
    verified_success: bool
    risk_events: int
    latency_ms: int
    actions: int
    model_calls: int
    tokens: int
    offered_candidates: int
    finished_at_ms: int
    task_family: str = ""
    instance_id: str = ""
    pair_key: str = ""


def _canary_run_summaries(
    events: Iterable[dict], policy_digest: str, *, since_ms: int | None = None
) -> list[CanaryRunSummary]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        if event.get("run_id"):
            grouped[event["run_id"]].append(event)
    summaries = []
    for run_id, run_events in grouped.items():
        run_events.sort(key=lambda e: (e.get("sequence", 0), e.get("recorded_at_ms", 0)))
        start = next((e for e in run_events if e.get("event") == "run_started"), None)
        if not start or start.get("policy_digest") != policy_digest:
            continue
        # A run that executed an experiment arm is not a clean
        # policy-performance sample: the deviated step belongs to the
        # scheduler, not to the policy being qualified. Trials qualify
        # evidence only through CounterfactualTrials, never promotion. The
        # same holds for a run whose action was overridden by an active
        # causal policy — that choice was the causal prior's, not the
        # exploration policy's.
        if any(event.get("experiment") for event in run_events) or any(
            event.get("event") == "causal_policy_applied" for event in run_events
        ):
            continue
        final = next((e for e in reversed(run_events) if e.get("event") == "run_finished"), None)
        # Runs abandoned before a terminal decision are not task outcomes.
        if not final or final.get("status") == "aborted":
            continue
        finished_at_ms = max(int(final.get("recorded_at_ms", 0)), int(start.get("recorded_at_ms", 0)))
        if since_ms is not None and finished_at_ms < since_ms:
            continue
        task_key = start.get("task_key", "")
        task_family = str(start.get("task_family") or "").strip().lower() or task_key
        instance_id = str(start.get("instance_id") or "").strip()
        risk = latency = actions = model_calls = tokens = offered = 0
        for event in run_events:
            if event.get("event") == "transition":
                risk += max(0, int(event.get("risk_events", 0)))
                latency += max(0, int(event.get("latency_ms", 0)))
                actions += 1
                model_calls += max(0, int(event.get("model_calls", 0)))
                tokens += max(0, int(event.get("tokens", 0)))
                offered_count = event.get("offered_count")
                offered += int(offered_count) if offered_count is not None else len(event.get("candidates", ()))
        summaries.append(CanaryRunSummary(
            run_id=run_id,
            task_key=task_key,
            verified_success=final.get("status") == "done" and bool(final.get("verified")),
            risk_events=risk,
            latency_ms=latency,
            actions=actions,
            model_calls=model_calls,
            tokens=tokens,
            offered_candidates=offered,
            finished_at_ms=finished_at_ms,
            task_family=task_family,
            instance_id=instance_id,
            # Namespaced by family: an instance_id only means "the same task
            # instance" *within* a family — bare instance ids collide across
            # families (flights#1 ≠ hotels#1).
            pair_key=f"{task_family}\x00{instance_id}" if instance_id else task_key,
        ))
    summaries.sort(key=lambda run: (run.finished_at_ms, run.run_id))
    return summaries


def _metrics_from_canary_runs(runs: Iterable[CanaryRunSummary]) -> CanaryMetrics:
    runs = list(runs)
    keys = tuple(sorted({run.task_key for run in runs if run.task_key}))
    families = tuple(sorted({run.task_family or run.task_key for run in runs if run.task_key or run.task_family}))
    instances = tuple(sorted({run.instance_id for run in runs if run.instance_id}))
    successes = sum(int(run.verified_success) for run in runs)
    return CanaryMetrics(
        tasks=len(runs),
        verified_successes=successes,
        failures=len(runs) - successes,
        risk_events=sum(run.risk_events for run in runs),
        latency_ms=sum(run.latency_ms for run in runs),
        actions=sum(run.actions for run in runs),
        model_calls=sum(run.model_calls for run in runs),
        tokens=sum(run.tokens for run in runs),
        offered_candidates=sum(run.offered_candidates for run in runs),
        task_families=len(families),
        run_ids=tuple(run.run_id for run in runs),
        task_keys=keys,
        family_keys=families,
        instance_ids=instances,
    )


def _sign_test_p_value(candidate_wins: int, baseline_wins: int) -> float:
    """Exact two-sided sign-test p-value over paired outcomes (ties excluded)."""
    n = candidate_wins + baseline_wins
    if n == 0:
        return 1.0
    k = min(candidate_wins, baseline_wins)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


@dataclass(frozen=True)
class CanaryEvidence:
    baseline: CanaryMetrics
    candidate: CanaryMetrics
    paired_task_families: int
    candidate_wins: int
    baseline_wins: int
    ties: int
    event_head_hash: str
    evidence_digest: str
    baseline_digest: str = ""
    candidate_digest: str = ""
    paired_instances: int = 0
    sign_test_p_value: float = 1.0

    @classmethod
    def from_events(
        cls,
        events: Iterable[dict],
        baseline_digest: str,
        candidate_digest: str,
        *,
        candidate_since_ms: int | None = None,
        baseline_since_ms: int | None = None,
    ):
        events = list(events)
        baseline_runs = _canary_run_summaries(events, baseline_digest, since_ms=baseline_since_ms)
        candidate_runs = _canary_run_summaries(events, candidate_digest, since_ms=candidate_since_ms)
        baseline = _metrics_from_canary_runs(baseline_runs)
        candidate = _metrics_from_canary_runs(candidate_runs)
        base_groups: dict[str, list[CanaryRunSummary]] = defaultdict(list)
        cand_groups: dict[str, list[CanaryRunSummary]] = defaultdict(list)
        for run in baseline_runs:
            base_groups[run.pair_key or run.task_key].append(run)
        for run in candidate_runs:
            cand_groups[run.pair_key or run.task_key].append(run)
        shared = sorted(set(base_groups) & set(cand_groups))
        candidate_wins = baseline_wins = ties = 0
        pair_rows = []
        paired_families = set()
        for key in shared:
            b = base_groups[key]
            c = cand_groups[key]
            # A pair key embeds its family, so equal keys should imply equal
            # families; a mismatch means the evidence is structurally corrupt.
            if b[0].task_family != c[0].task_family:
                raise ValueError(f"Canary pair key {key!r} spans inconsistent task families")
            paired_families.add((b[0].task_family or b[0].task_key))
            b_rate = sum(int(r.verified_success) for r in b) / len(b)
            c_rate = sum(int(r.verified_success) for r in c) / len(c)
            if c_rate > b_rate:
                candidate_wins += 1
            elif b_rate > c_rate:
                baseline_wins += 1
            else:
                ties += 1
            pair_rows.append({
                "pair_key": key,
                "task_key": b[0].task_key,
                "instance_id": b[0].instance_id or c[0].instance_id,
                "task_family": b[0].task_family or c[0].task_family,
                "baseline_runs": [r.run_id for r in b],
                "candidate_runs": [r.run_id for r in c],
                "baseline_success_rate": b_rate,
                "candidate_success_rate": c_rate,
                "baseline_risk": sum(r.risk_events for r in b),
                "candidate_risk": sum(r.risk_events for r in c),
            })
        head = events[-1].get("event_hash", "0" * 64) if events else "0" * 64
        material = {
            "baseline_digest": baseline_digest,
            "candidate_digest": candidate_digest,
            "baseline": asdict(baseline),
            "candidate": asdict(candidate),
            "pairs": pair_rows,
            "event_head_hash": head,
        }
        digest = _stable_hash(json.dumps(material, sort_keys=True, separators=(",", ":")))
        return cls(
            baseline=baseline,
            candidate=candidate,
            paired_task_families=len(paired_families),
            candidate_wins=candidate_wins,
            baseline_wins=baseline_wins,
            ties=ties,
            event_head_hash=head,
            evidence_digest=digest,
            baseline_digest=baseline_digest,
            candidate_digest=candidate_digest,
            paired_instances=len(shared),
            sign_test_p_value=_sign_test_p_value(candidate_wins, baseline_wins),
        )


@dataclass(frozen=True)
class CanaryDecision:
    approved: bool
    reason: str


class CanaryGate:
    """Require matched real executions before replay-selected policy activation.

    The default gate is a paired-evidence requirement, not just a
    non-regression check: alongside the aggregate floors it demands an exact
    two-sided sign test over ``(task_family, instance_id)`` outcome pairs with
    ``max_pair_sign_p`` — promotion needs *demonstrated* improvement, not
    merely "no worse". Because a sign test ignores ties, p ≤ 0.05 needs at
    least six non-tied pairs all won by the candidate, so the bound evidence
    must exceed the raw task minimums before promotion can pass. Setting
    ``max_pair_sign_p=None`` explicitly waives the significance check for
    controlled testing; leaving the field unset keeps it on.
    """

    def __init__(
        self,
        *,
        min_tasks: int = 12,
        min_baseline_tasks: int = 12,
        min_task_families: int = 4,
        min_paired_task_families: int = 4,
        min_pair_coverage: float = 0.80,
        max_success_regression: float = 0.0,
        max_extra_risk_events: int = 0,
        max_extra_risk_rate: float = 0.0,
        max_failure_rate_regression: float = 0.0,
        max_latency_regression_ratio: float = 0.25,
        max_action_regression_ratio: float = 0.25,
        max_token_regression_ratio: float = 0.25,
        max_pair_sign_p: float | None = 0.05,
    ):
        self.min_tasks = min_tasks
        self.min_baseline_tasks = min_baseline_tasks
        self.min_task_families = min_task_families
        self.min_paired_task_families = min_paired_task_families
        self.min_pair_coverage = min_pair_coverage
        self.max_success_regression = max_success_regression
        self.max_extra_risk_events = max_extra_risk_events
        self.max_extra_risk_rate = max_extra_risk_rate
        self.max_failure_rate_regression = max_failure_rate_regression
        self.max_latency_regression_ratio = max_latency_regression_ratio
        self.max_action_regression_ratio = max_action_regression_ratio
        self.max_token_regression_ratio = max_token_regression_ratio
        self.max_pair_sign_p = max_pair_sign_p

    def assess(
        self,
        baseline: CanaryMetrics,
        candidate: CanaryMetrics,
        *,
        evidence: CanaryEvidence | None = None,
    ) -> CanaryDecision:
        if candidate.tasks < self.min_tasks:
            return CanaryDecision(False, f"need at least {self.min_tasks} candidate canary tasks")
        if baseline.tasks < self.min_baseline_tasks:
            return CanaryDecision(False, f"need at least {self.min_baseline_tasks} baseline canary tasks")
        if candidate.task_families < self.min_task_families:
            return CanaryDecision(False, f"need at least {self.min_task_families} candidate task families")
        if baseline.tasks and candidate.success_rate + self.max_success_regression < baseline.success_rate:
            return CanaryDecision(False, "candidate canary regressed verified success")
        if candidate.failure_rate > baseline.failure_rate + self.max_failure_rate_regression and baseline.tasks:
            return CanaryDecision(False, "candidate canary increased failure rate")
        if candidate.risk_events > baseline.risk_events + self.max_extra_risk_events:
            return CanaryDecision(False, "candidate canary increased risk events")
        # Rates as well as counts: unequal task counts must not let a busier
        # candidate accumulate more risk per task than the baseline.
        if baseline.tasks and candidate.risk_rate > baseline.risk_rate + self.max_extra_risk_rate:
            return CanaryDecision(False, "candidate canary increased risk rate")
        if candidate.verified_successes == 0:
            return CanaryDecision(False, "candidate canary has no verified successes")
        if (
            baseline.avg_latency_ms > 0
            and candidate.avg_latency_ms > baseline.avg_latency_ms * (1 + self.max_latency_regression_ratio)
        ):
            return CanaryDecision(False, "candidate canary latency regression")
        if (
            baseline.avg_actions > 0
            and candidate.avg_actions > baseline.avg_actions * (1 + self.max_action_regression_ratio)
        ):
            return CanaryDecision(False, "candidate canary action-count regression")
        if (
            baseline.avg_tokens > 0
            and candidate.avg_tokens > baseline.avg_tokens * (1 + self.max_token_regression_ratio)
        ):
            return CanaryDecision(False, "candidate canary token regression")
        if evidence is not None:
            if evidence.paired_task_families < self.min_paired_task_families:
                return CanaryDecision(False, f"need at least {self.min_paired_task_families} paired task families")
            pair_denominator = max(1, evidence.candidate.task_families)
            if evidence.paired_task_families / pair_denominator < self.min_pair_coverage:
                return CanaryDecision(False, "insufficient paired task-family coverage")
            if evidence.baseline_wins > evidence.candidate_wins:
                return CanaryDecision(False, "candidate lost more paired task families than it won")
            if self.max_pair_sign_p is not None and evidence.sign_test_p_value > self.max_pair_sign_p:
                return CanaryDecision(
                    False,
                    f"paired sign test lacks confidence (p={evidence.sign_test_p_value:.3f})",
                )
        return CanaryDecision(True, "bound live canary gates passed" if evidence else "live canary gates passed")
