"""DREAM-Jev: replay-based meta-exploration for Jev Ultrafast.

This module intentionally improves only bounded exploration policy parameters.
The browser executor, approval policy, verifier contract, and isolation boundary
remain outside the self-improvement surface.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import fmean
from typing import Iterable

from .privacy import tokenize

SCHEMA_VERSION = "jev-dream/2"
SUPPORTED_SCHEMAS = {"jev-dream/1", SCHEMA_VERSION}
TCB_VERSION = "jev-ultrafast-tcb/0.4"
SUPPORTED_TCB_VERSIONS = {"jev-ultrafast-tcb/0.3", TCB_VERSION}
ACTION_KINDS = ("click", "fill", "select", "scroll", "wait")


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@contextmanager
def _file_lock(lock_path: Path):
    """Cross-process advisory lock shared by the experience store and the registry."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def task_key(goal: str) -> str:
    """Stable task identity without requiring plaintext goal storage."""
    return _stable_hash(" ".join(str(goal).split()).strip().lower())


def new_run_id() -> str:
    return uuid.uuid4().hex


def candidate_catalog_digest(candidates: Iterable[dict]) -> str:
    """Digest the replay-relevant candidate catalogue in its original order."""
    compact = []
    for candidate in candidates:
        compact.append({
            "id": candidate.get("id"),
            "kind": candidate.get("kind"),
            "label": candidate.get("label", ""),
            "value": candidate.get("value", ""),
            "goal_overlap": max(0, int(candidate.get("goal_overlap", 0))),
        })
    return _stable_hash(json.dumps(compact, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


@dataclass(frozen=True)
class ExplorationPolicy:
    """Bounded policy surface that DREAM-Jev is allowed to optimize.

    These knobs influence candidate allocation and search patience only. They do
    not bypass action validation, approval requirements, completion verification,
    or browser execution guards.
    """

    name: str = "baseline"
    version: int = 1
    goal_overlap_weight: float = 10.0
    order_penalty: float = 0.00001
    click_bonus: float = 0.5
    fill_bonus: float = 1.5
    select_bonus: float = 1.0
    model_action_limit: int = 250
    click_quota: int = 150
    fill_quota: int = 45
    select_quota: int = 55
    max_actions: int = 60
    no_progress_window: int = 3

    def __post_init__(self):
        if not self.name or self.version < 1:
            raise ValueError("Exploration policy needs a name and positive version")
        for value in (
            self.goal_overlap_weight,
            self.order_penalty,
            self.click_bonus,
            self.fill_bonus,
            self.select_bonus,
        ):
            if not math.isfinite(value):
                raise ValueError("Exploration policy weights must be finite")
        if not 16 <= self.model_action_limit <= 250:
            raise ValueError("model_action_limit must be between 16 and 250")
        for quota in (self.click_quota, self.fill_quota, self.select_quota):
            if not 1 <= quota <= 250:
                raise ValueError("candidate quotas must be between 1 and 250")
        if not 1 <= self.max_actions <= 120:
            raise ValueError("max_actions must be between 1 and 120")
        if not 2 <= self.no_progress_window <= 10:
            raise ValueError("no_progress_window must be between 2 and 10")

    def candidate_score(self, action: dict, goal_tokens: set[str], order: int) -> float:
        searchable = " ".join(
            str(action.get(k, "")) for k in ("label", "value", "current_value", "option_label")
        ).lower()
        tokens = set(tokenize(searchable))
        overlap = len(goal_tokens & tokens)
        bonus = {
            "fill": self.fill_bonus,
            "select": self.select_bonus,
            "click": self.click_bonus,
        }.get(action.get("kind"), 0.0)
        return overlap * self.goal_overlap_weight + bonus - order * self.order_penalty

    def quota_for(self, kind: str) -> int:
        return {
            "click": self.click_quota,
            "fill": self.fill_quota,
            "select": self.select_quota,
        }.get(kind, self.model_action_limit)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "ExplorationPolicy":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown exploration policy keys: {sorted(unknown)}")
        return cls(**payload)

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return _stable_hash(encoded)


@dataclass(frozen=True)
class ObjectiveWeights:
    success: float = 100.0
    verified: float = 20.0
    page_progress: float = 0.5
    latency_seconds: float = 0.25
    action: float = 0.15
    model_call: float = 0.10
    thousand_tokens: float = 0.02
    offered_candidate: float = 0.002
    stale_or_failure: float = 2.0
    risk_event: float = 5.0
    coverage_miss: float = 4.0


@dataclass(frozen=True)
class ReplayMetrics:
    score: float
    worlds: int
    successes: int
    verified_successes: int
    actions: int
    model_calls: int
    tokens: int
    latency_ms: int
    offered_candidates: int
    page_progress: int
    stale_or_failures: int
    risk_events: int
    coverage_misses: int
    coverage: float

    @property
    def success_rate(self) -> float:
        return self.successes / self.worlds if self.worlds else 0.0

    @property
    def score_per_world(self) -> float:
        return self.score / self.worlds if self.worlds else 0.0


class ExperienceStore:
    """Append-only JSONL event store with cross-process serialization.

    Each event is hash-linked to the previous event. The lock file prevents two
    Jev processes from reading the same head and creating a forked chain. Reads
    verify the complete chain; appends only read the tail while holding the
    operating-system lock, keeping append cost effectively constant.
    """

    def __init__(self, path: str | os.PathLike, *, strict: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._lock = threading.Lock()
        # Default appends verify only the tail so they stay O(1). ``strict``
        # verifies the whole chain first so a mid-chain corruption cannot keep
        # accumulating events into a store that load() would reject.
        self.strict = strict

    @staticmethod
    def _event_hash(payload: dict) -> str:
        material = {k: v for k, v in payload.items() if k != "event_hash"}
        return _stable_hash(json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False))

    def _tail_event_unlocked(self) -> dict | None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return None
        with self.path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            pos = handle.tell()
            data = b""
            while pos > 0:
                take = min(4096, pos)
                pos -= take
                handle.seek(pos)
                data = handle.read(take) + data
                lines = [line for line in data.splitlines() if line.strip()]
                if len(lines) >= 2 or pos == 0:
                    if not lines:
                        return None
                    try:
                        return json.loads(lines[-1].decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ValueError("Invalid DREAM event JSON at store tail") from exc
        return None

    _RESERVED_EVENT_KEYS = {"schema", "tcb_version", "recorded_at_ms", "prev_hash", "event_hash"}

    def append(self, event: dict) -> dict:
        if self._RESERVED_EVENT_KEYS & event.keys():
            raise ValueError(
                f"DREAM event cannot set reserved chain keys: {sorted(self._RESERVED_EVENT_KEYS & event.keys())}"
            )
        with self._lock, _file_lock(self.lock_path):
            if self.strict:
                self._load_unlocked()
            tail = self._tail_event_unlocked()
            if tail:
                if tail.get("schema") not in SUPPORTED_SCHEMAS or tail.get("tcb_version") not in SUPPORTED_TCB_VERSIONS:
                    raise ValueError("Cannot append to an unsupported DREAM store tail")
                if tail.get("event_hash") != self._event_hash(tail):
                    raise ValueError("Cannot append to a DREAM store with a corrupt tail")
            previous = tail.get("event_hash") if tail else "0" * 64
            payload = {
                "schema": SCHEMA_VERSION,
                "tcb_version": TCB_VERSION,
                "recorded_at_ms": int(time.time() * 1000),
                "prev_hash": previous,
                **event,
            }
            payload["event_hash"] = self._event_hash(payload)
            line = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return payload

    def _load_unlocked(self) -> list[dict]:
        if not self.path.exists():
            return []
        events = []
        expected_prev = "0" * 64
        with self.path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid DREAM event JSON on line {line_no}") from exc
                if event.get("schema") not in SUPPORTED_SCHEMAS:
                    raise ValueError(f"Unsupported DREAM schema on line {line_no}")
                if event.get("tcb_version") not in SUPPORTED_TCB_VERSIONS:
                    raise ValueError(f"Unsupported DREAM TCB version on line {line_no}")
                if event.get("prev_hash") != expected_prev:
                    raise ValueError(f"DREAM hash-chain predecessor mismatch on line {line_no}")
                if event.get("event_hash") != self._event_hash(event):
                    raise ValueError(f"DREAM hash-chain integrity failure on line {line_no}")
                expected_prev = event["event_hash"]
                events.append(event)
        return events

    def load(self) -> list[dict]:
        with self._lock, _file_lock(self.lock_path):
            return self._load_unlocked()

    def verify(self) -> dict:
        events = self.load()
        return {
            "events": len(events),
            "head_hash": events[-1]["event_hash"] if events else "0" * 64,
            "schemas": sorted({event["schema"] for event in events}),
            "tcb_versions": sorted({event["tcb_version"] for event in events}),
        }

    def head_hash(self) -> str:
        return self.verify()["head_hash"]


@dataclass(frozen=True)
class RecordedTransition:
    run_id: str
    task_key: str
    state: str
    next_state: str
    selected_id: str
    selected_kind: str
    candidate_actions: tuple[dict, ...]
    candidate_digest: str
    selected_rank: int | None
    page_changed: bool
    latency_ms: int
    model_calls: int
    tokens: int
    stale_or_failure: int
    risk_events: int
    terminal: str | None = None
    verified: bool = False


@dataclass
class ReplayWorld:
    """One recorded online run used as an empirical replay world.

    Worlds are intentionally not merged across live runs. This avoids stitching
    together a counterfactual trajectory from states that only happen to share a
    fingerprint. Multiple runs of the same task remain separate worlds in the
    simulator pool, matching the paper's history-of-trees framing.
    """

    key: str
    task_key: str
    goal: str
    start_state: str
    tcb_version: str
    trajectory: tuple[RecordedTransition, ...]

    @property
    def digest(self) -> str:
        payload = {
            "key": self.key,
            "task_key": self.task_key,
            "start_state": self.start_state,
            "tcb_version": self.tcb_version,
            "trajectory": [asdict(item) for item in self.trajectory],
        }
        return _stable_hash(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))

    @classmethod
    def from_events(cls, events: Iterable[dict]) -> list["ReplayWorld"]:
        runs: dict[str, list[dict]] = defaultdict(list)
        for event in events:
            run_id = event.get("run_id")
            if run_id:
                runs[run_id].append(event)

        worlds = []
        for run_id, run_events in runs.items():
            run_events.sort(key=lambda e: (e.get("sequence", 0), e.get("recorded_at_ms", 0)))
            start = next((e for e in run_events if e.get("event") == "run_started"), None)
            if not start:
                continue
            key = start["task_key"]
            run_tcbs = {e.get("tcb_version") for e in run_events}
            if len(run_tcbs) != 1:
                raise ValueError(f"Run {run_id} crosses DREAM TCB versions")
            run_tcb = next(iter(run_tcbs))
            final = next((e for e in reversed(run_events) if e.get("event") == "run_finished"), None)
            raw_transitions = [e for e in run_events if e.get("event") == "transition"]
            transitions = []
            for index, event in enumerate(raw_transitions):
                # Dynamic pages may change between actions without a Jev mutation.
                # Preserve the ordered empirical trajectory instead of inventing an
                # action-caused edge or rejecting valid asynchronous state drift.
                terminal = None
                verified = False
                if index == len(raw_transitions) - 1 and final:
                    terminal = final.get("status")
                    verified = bool(final.get("verified"))
                event_candidates = tuple(event.get("candidates", ()))
                observed_digest = candidate_catalog_digest(event_candidates)
                stored_digest = event.get("candidate_digest")
                if stored_digest and stored_digest != observed_digest:
                    raise ValueError(f"Run {run_id} has a candidate catalogue digest mismatch")
                selected_id = event["selected"]["id"]
                selected_rank = event.get("selected_rank")
                if selected_rank is None:
                    selected_rank = next(
                        (i for i, item in enumerate(event_candidates) if item.get("id") == selected_id),
                        None,
                    )
                tr = RecordedTransition(
                    run_id=run_id,
                    task_key=key,
                    state=event["state"],
                    next_state=event["next_state"],
                    selected_id=selected_id,
                    selected_kind=event["selected"].get("kind", ""),
                    candidate_actions=event_candidates,
                    candidate_digest=observed_digest,
                    selected_rank=selected_rank,
                    page_changed=bool(event.get("page_changed")),
                    latency_ms=max(0, int(event.get("latency_ms", 0))),
                    model_calls=max(0, int(event.get("model_calls", 1))),
                    tokens=max(0, int(event.get("tokens", 0))),
                    stale_or_failure=max(0, int(event.get("stale_or_failure", 0))),
                    risk_events=max(0, int(event.get("risk_events", 0))),
                    terminal=terminal,
                    verified=verified,
                )
                transitions.append(tr)
            worlds.append(cls(
                key=run_id,
                task_key=key,
                goal=start.get("goal", ""),
                start_state=start.get("state", "ROOT"),
                tcb_version=run_tcb,
                trajectory=tuple(transitions),
            ))
        return worlds

    def recorded_actions(self) -> tuple[str, ...]:
        return tuple(t.selected_id for t in self.trajectory)


def replay_pool_digest(worlds: Iterable[ReplayWorld]) -> str:
    digests = sorted(world.digest for world in worlds)
    return _stable_hash(json.dumps(digests, separators=(",", ":")))


def split_manifest_digest(splits: dict[str, list[ReplayWorld]]) -> str:
    payload = {name: sorted(world.digest for world in worlds) for name, worlds in sorted(splits.items())}
    return _stable_hash(json.dumps(payload, sort_keys=True, separators=(",", ":")))


@dataclass(frozen=True)
class ReplayResult:
    metrics: ReplayMetrics
    per_world: tuple[dict, ...]


class ReplaySimulator:
    """Conservative replay over realized browser trajectories.

    The exploration policy is allowed to change candidate *allocation* and
    stopping budgets. Replay follows a recorded transition only if the action
    actually taken online would still be offered by the candidate policy. It
    never assumes that the fixed decision model would choose a different action
    merely because the candidate catalogue changed.
    """

    def __init__(self, worlds: Iterable[ReplayWorld], weights: ObjectiveWeights | None = None):
        self.worlds = tuple(worlds)
        self.weights = weights or ObjectiveWeights()

    def evaluate(self, policy: ExplorationPolicy, *, max_steps: int | None = None) -> ReplayResult:
        summaries = [self._evaluate_world(world, policy, max_steps=max_steps) for world in self.worlds]
        return ReplayResult(metrics=self._aggregate(summaries), per_world=tuple(summaries))

    def _evaluate_world(self, world: ReplayWorld, policy: ExplorationPolicy, *, max_steps: int | None):
        summary = self._empty_world(world)
        step_limit = min(policy.max_actions, max_steps or policy.max_actions)
        no_progress = 0
        for transition in world.trajectory[:step_limit]:
            offered = self._retained_candidates(policy, transition.candidate_actions)
            summary["offered_candidates"] += len(offered)
            if transition.selected_id not in {item["id"] for item in offered}:
                summary["coverage_misses"] += 1
                break
            summary["actions"] += 1
            summary["model_calls"] += transition.model_calls
            summary["tokens"] += transition.tokens
            summary["latency_ms"] += transition.latency_ms
            summary["page_progress"] += int(transition.page_changed)
            summary["stale_or_failures"] += transition.stale_or_failure
            summary["risk_events"] += transition.risk_events
            if transition.selected_kind != "wait" and not transition.page_changed:
                no_progress += 1
            else:
                no_progress = 0
            if transition.terminal:
                summary["status"] = transition.terminal
                summary["verified"] = transition.verified
                summary["success"] = transition.terminal == "done" and transition.verified
                break
            if no_progress >= policy.no_progress_window:
                summary["status"] = "blocked"
                break
        summary["covered_steps"] = summary["actions"]
        return summary

    @staticmethod
    def _retained_candidates(policy: ExplorationPolicy, candidates: Iterable[dict]) -> list[dict]:
        candidates = [dict(c) for c in candidates if c.get("id")]
        controls = [c for c in candidates if c.get("kind") not in {"click", "fill", "select"}]
        regular = [(index, c) for index, c in enumerate(candidates) if c.get("kind") in {"click", "fill", "select"}]

        def score(pair):
            index, action = pair
            overlap = max(0, int(action.get("goal_overlap", 0)))
            bonus = {
                "fill": policy.fill_bonus,
                "select": policy.select_bonus,
                "click": policy.click_bonus,
            }.get(action.get("kind"), 0.0)
            return overlap * policy.goal_overlap_weight + bonus - index * policy.order_penalty

        ranked = sorted(regular, key=score, reverse=True)
        quotas = {"click": policy.click_quota, "fill": policy.fill_quota, "select": policy.select_quota}
        regular_budget = max(0, policy.model_action_limit - len(controls))
        counts = {kind: 0 for kind in quotas}
        selected = []
        used = set()
        for index, action in ranked:
            kind = action["kind"]
            if counts[kind] >= quotas[kind] or len(selected) >= regular_budget:
                continue
            selected.append((index, action))
            used.add(index)
            counts[kind] += 1
        if len(selected) < regular_budget:
            for index, action in ranked:
                if index in used:
                    continue
                selected.append((index, action))
                if len(selected) >= regular_budget:
                    break
        result = [action for _, action in sorted(selected, key=lambda pair: pair[0])]
        result.extend(controls)
        return result[: policy.model_action_limit]

    @staticmethod
    def _empty_world(world: ReplayWorld, coverage_miss: int = 0):
        return {
            "world": world.key,
            "task_key": world.task_key,
            "success": False,
            "verified": False,
            "status": None,
            "actions": 0,
            "model_calls": 0,
            "tokens": 0,
            "latency_ms": 0,
            "offered_candidates": 0,
            "page_progress": 0,
            "stale_or_failures": 0,
            "risk_events": 0,
            "coverage_misses": coverage_miss,
            "covered_steps": 0,
        }

    def _aggregate(self, summaries: list[dict]) -> ReplayMetrics:
        w = self.weights
        worlds = len(summaries)
        successes = sum(int(s["success"]) for s in summaries)
        verified = sum(int(s["verified"] and s["success"]) for s in summaries)
        actions = sum(s["actions"] for s in summaries)
        model_calls = sum(s["model_calls"] for s in summaries)
        tokens = sum(s["tokens"] for s in summaries)
        latency = sum(s["latency_ms"] for s in summaries)
        offered = sum(s["offered_candidates"] for s in summaries)
        progress = sum(s["page_progress"] for s in summaries)
        failures = sum(s["stale_or_failures"] for s in summaries)
        risk = sum(s["risk_events"] for s in summaries)
        misses = sum(s["coverage_misses"] for s in summaries)
        potential = actions + misses
        coverage = actions / potential if potential else (1.0 if worlds else 0.0)
        score = (
            w.success * successes
            + w.verified * verified
            + w.page_progress * progress
            - w.latency_seconds * (latency / 1000)
            - w.action * actions
            - w.model_call * model_calls
            - w.thousand_tokens * (tokens / 1000)
            - w.offered_candidate * offered
            - w.stale_or_failure * failures
            - w.risk_event * risk
            - w.coverage_miss * misses
        )
        return ReplayMetrics(
            score=score,
            worlds=worlds,
            successes=successes,
            verified_successes=verified,
            actions=actions,
            model_calls=model_calls,
            tokens=tokens,
            latency_ms=latency,
            offered_candidates=offered,
            page_progress=progress,
            stale_or_failures=failures,
            risk_events=risk,
            coverage_misses=misses,
            coverage=coverage,
        )


@dataclass(frozen=True)
class PromotionDecision:
    approved: bool
    reason: str
    baseline: ReplayMetrics
    candidate: ReplayMetrics
    validation: ReplayMetrics | None = None
    holdout: ReplayMetrics | None = None


class PromotionGate:
    """Fail-closed replay promotion gate.

    A candidate must improve the training objective and avoid regressions in
    verified success, risk, coverage, and normalized objective on every
    available validation/holdout split. Replay approval remains provisional; a
    bound live-canary evidence bundle is still required before activation.
    """

    def __init__(
        self,
        *,
        min_score_gain_per_world: float = 0.0,
        min_coverage: float = 0.75,
        max_success_regression: float = 0.0,
        max_risk_regression: int = 0,
        max_objective_regression_per_world: float = 0.0,
    ):
        self.min_score_gain_per_world = min_score_gain_per_world
        self.min_coverage = min_coverage
        self.max_success_regression = max_success_regression
        self.max_risk_regression = max_risk_regression
        self.max_objective_regression_per_world = max_objective_regression_per_world

    def _split_failure(self, label: str, baseline: ReplayMetrics, candidate: ReplayMetrics, *, require_gain=False):
        if candidate.worlds == 0:
            return f"no {label} replay worlds"
        if candidate.coverage < self.min_coverage:
            return f"insufficient {label} coverage"
        if candidate.success_rate + self.max_success_regression < baseline.success_rate:
            return f"{label} success regression"
        if candidate.risk_events > baseline.risk_events + self.max_risk_regression:
            return f"{label} risk regression"
        delta = candidate.score_per_world - baseline.score_per_world
        if require_gain and delta <= self.min_score_gain_per_world:
            return f"candidate did not improve {label} replay objective"
        if not require_gain and delta < -self.max_objective_regression_per_world:
            return f"{label} objective regression"
        return None

    def assess(
        self,
        baseline: ReplayResult,
        candidate: ReplayResult,
        *,
        validation_baseline: ReplayResult | None = None,
        validation_candidate: ReplayResult | None = None,
        holdout_baseline: ReplayResult | None = None,
        holdout_candidate: ReplayResult | None = None,
    ) -> PromotionDecision:
        b, c = baseline.metrics, candidate.metrics
        reason = self._split_failure("training", b, c, require_gain=True)
        if reason:
            return PromotionDecision(False, reason, b, c)

        validation_metrics = holdout_metrics = None
        for label, vb, vc in (
            ("validation", validation_baseline, validation_candidate),
            ("holdout", holdout_baseline, holdout_candidate),
        ):
            if (vb is None) != (vc is None):
                return PromotionDecision(False, f"incomplete {label} evidence", b, c)
            if vb and vc:
                split_reason = self._split_failure(label, vb.metrics, vc.metrics)
                if split_reason:
                    return PromotionDecision(
                        False,
                        split_reason,
                        b,
                        c,
                        vc.metrics if label == "validation" else None,
                        vc.metrics if label == "holdout" else None,
                    )
                if label == "validation":
                    validation_metrics = vc.metrics
                else:
                    holdout_metrics = vc.metrics
        return PromotionDecision(
            True,
            "replay gates passed; bound live canary still required",
            b,
            c,
            validation_metrics,
            holdout_metrics,
        )


def split_worlds(worlds: Iterable[ReplayWorld]):
    """Deterministic, disjoint split that keeps each task family in one partition."""
    groups: dict[str, list[ReplayWorld]] = defaultdict(list)
    for world in worlds:
        groups[world.task_key].append(world)
    ordered = sorted(groups.items(), key=lambda item: (int(item[0][:16], 16), item[0]))
    n = len(ordered)
    if n == 0:
        return {"train": [], "validation": [], "holdout": []}
    if n == 1:
        assignments = (n, 0, 0)
    elif n == 2:
        assignments = (1, 1, 0)
    else:
        holdout_n = max(1, round(n * 0.15))
        validation_n = max(1, round(n * 0.15))
        if holdout_n + validation_n >= n:
            holdout_n = validation_n = 1
        assignments = (n - validation_n - holdout_n, validation_n, holdout_n)
    train_n, validation_n, _holdout_n = assignments
    train_groups = ordered[:train_n]
    validation_groups = ordered[train_n : train_n + validation_n]
    holdout_groups = ordered[train_n + validation_n :]

    def flatten(items):
        return [world for _key, group in items for world in group]

    return {
        "train": flatten(train_groups),
        "validation": flatten(validation_groups),
        "holdout": flatten(holdout_groups),
    }


def mutate_policies(base: ExplorationPolicy) -> list[ExplorationPolicy]:
    """Deterministic, bounded candidate generator for offline dreaming.

    The generator changes only whitelisted exploration knobs. It never emits code.
    """
    candidates = [base]
    specs = [
        {"goal_overlap_weight": base.goal_overlap_weight * 1.20},
        {"goal_overlap_weight": max(1.0, base.goal_overlap_weight * 0.50)},
        {"goal_overlap_weight": max(0.1, base.goal_overlap_weight * 0.25)},
        {"goal_overlap_weight": max(0.1, base.goal_overlap_weight * 0.05)},
        {"fill_bonus": base.fill_bonus + 0.5},
        {"fill_bonus": base.fill_bonus + 3.0},
        {"select_bonus": base.select_bonus + 0.5},
        {"select_bonus": base.select_bonus + 3.0},
        {"click_bonus": base.click_bonus + 0.5},
        {"click_bonus": base.click_bonus + 3.0},
        {"model_action_limit": max(64, min(250, base.model_action_limit - 32))},
        {"model_action_limit": max(64, min(250, base.model_action_limit + 0))},
        {"click_quota": max(40, min(250, base.click_quota - 20)), "fill_quota": min(80, base.fill_quota + 10)},
        {"no_progress_window": min(6, base.no_progress_window + 1)},
        {"max_actions": max(20, base.max_actions - 10)},
    ]
    for index, changes in enumerate(specs, 1):
        # Avoid duplicate candidates such as action_limit=250 + 0.
        candidate = replace(base, name=f"{base.name}-dream-{index}", version=base.version + 1, **changes)
        if candidate.digest not in {p.digest for p in candidates}:
            candidates.append(candidate)
    return candidates


@dataclass(frozen=True)
class DreamReport:
    selected: ExplorationPolicy
    baseline: ExplorationPolicy
    promotion: PromotionDecision
    candidates: tuple[dict, ...]
    split_sizes: dict[str, int]
    world_pool_digest: str
    split_manifest_digest: str
    evidence_head_hash: str | None = None
    tcb_versions: tuple[str, ...] = ()
    live_canary_required: bool = True

    def to_dict(self):
        return {
            "schema": SCHEMA_VERSION,
            "selected": self.selected.to_dict(),
            "selected_digest": self.selected.digest,
            "baseline": self.baseline.to_dict(),
            "baseline_digest": self.baseline.digest,
            "promotion": {
                "approved": self.promotion.approved,
                "reason": self.promotion.reason,
                "baseline": asdict(self.promotion.baseline),
                "candidate": asdict(self.promotion.candidate),
                "validation": asdict(self.promotion.validation) if self.promotion.validation else None,
                "holdout": asdict(self.promotion.holdout) if self.promotion.holdout else None,
            },
            "candidates": list(self.candidates),
            "split_sizes": self.split_sizes,
            "world_pool_digest": self.world_pool_digest,
            "split_manifest_digest": self.split_manifest_digest,
            "evidence_head_hash": self.evidence_head_hash,
            "tcb_versions": list(self.tcb_versions),
            "live_canary_required": self.live_canary_required,
            "tcb_version": TCB_VERSION,
        }


class DreamImprover:
    """Evaluate bounded policy variants using historical replay.

    Every candidate is evaluated on train and, when available, validation and
    holdout worlds. Selection is among candidates that pass all replay gates,
    rather than simply picking the training winner and checking it afterward.
    This prevents a slightly weaker but generalizing candidate from being hidden
    by an overfit training winner.
    """

    def __init__(self, *, weights: ObjectiveWeights | None = None, gate: PromotionGate | None = None):
        self.weights = weights or ObjectiveWeights()
        self.gate = gate or PromotionGate()

    @staticmethod
    def _robust_gain(base_results: dict[str, ReplayResult | None], candidate_results: dict[str, ReplayResult | None]):
        deltas = []
        for name in ("train", "validation", "holdout"):
            base_result, candidate_result = base_results.get(name), candidate_results.get(name)
            if base_result is not None and candidate_result is not None and base_result.metrics.worlds:
                deltas.append(candidate_result.metrics.score_per_world - base_result.metrics.score_per_world)
        return min(deltas) if deltas else float("-inf")

    def improve(
        self,
        worlds: Iterable[ReplayWorld],
        base: ExplorationPolicy,
        *,
        evidence_head_hash: str | None = None,
    ) -> DreamReport:
        worlds = list(worlds)
        splits = split_worlds(worlds)
        pool_digest = replay_pool_digest(worlds)
        split_digest = split_manifest_digest(splits)
        tcb_versions = tuple(sorted({world.tcb_version for world in worlds}))
        split_sizes = {k: len(v) for k, v in splits.items()}
        train = splits["train"]
        if not train:
            empty = ReplaySimulator([], self.weights).evaluate(base)
            decision = PromotionDecision(False, "no replay worlds", empty.metrics, empty.metrics)
            return DreamReport(
                base, base, decision, (), split_sizes, pool_digest, split_digest, evidence_head_hash, tcb_versions
            )

        simulators = {
            name: (ReplaySimulator(items, self.weights) if items else None)
            for name, items in splits.items()
        }
        base_results = {
            name: (sim.evaluate(base) if sim else None)
            for name, sim in simulators.items()
        }

        evaluated = []
        passing: list[tuple[float, float, ExplorationPolicy, PromotionDecision]] = []
        for policy in mutate_policies(base):
            candidate_results = {
                name: (sim.evaluate(policy) if sim else None)
                for name, sim in simulators.items()
            }
            if policy.digest == base.digest:
                decision = PromotionDecision(
                    False,
                    "baseline candidate",
                    base_results["train"].metrics,
                    candidate_results["train"].metrics,
                )
            else:
                decision = self.gate.assess(
                    base_results["train"],
                    candidate_results["train"],
                    validation_baseline=base_results["validation"],
                    validation_candidate=candidate_results["validation"],
                    holdout_baseline=base_results["holdout"],
                    holdout_candidate=candidate_results["holdout"],
                )
            robust_gain = self._robust_gain(base_results, candidate_results)
            average_gain = fmean([
                candidate_results[name].metrics.score_per_world - base_results[name].metrics.score_per_world
                for name in ("train", "validation", "holdout")
                if base_results[name] is not None and candidate_results[name] is not None
            ])
            evaluated.append({
                "policy": policy.to_dict(),
                "digest": policy.digest,
                "train": asdict(candidate_results["train"].metrics),
                "validation": (
                    asdict(candidate_results["validation"].metrics) if candidate_results["validation"] else None
                ),
                "holdout": asdict(candidate_results["holdout"].metrics) if candidate_results["holdout"] else None,
                "replay_approved": decision.approved,
                "reason": decision.reason,
                "robust_gain_per_world": robust_gain,
                "average_gain_per_world": average_gain,
            })
            if decision.approved:
                passing.append((robust_gain, average_gain, policy, decision))

        if passing:
            passing.sort(key=lambda item: (item[0], item[1], item[2].digest), reverse=True)
            _robust, _average, selected, promotion = passing[0]
        else:
            selected = base
            base_train = base_results["train"]
            promotion = PromotionDecision(
                False,
                "no changed policy passed all replay gates",
                base_train.metrics,
                base_train.metrics,
                base_results["validation"].metrics if base_results["validation"] else None,
                base_results["holdout"].metrics if base_results["holdout"] else None,
            )

        return DreamReport(
            selected=selected,
            baseline=base,
            promotion=promotion,
            candidates=tuple(evaluated),
            split_sizes=split_sizes,
            world_pool_digest=pool_digest,
            split_manifest_digest=split_digest,
            evidence_head_hash=evidence_head_hash,
            tcb_versions=tcb_versions,
        )


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
            runs = runs[-max(0, int(max_runs)) :]
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
        final = next((e for e in reversed(run_events) if e.get("event") == "run_finished"), None)
        # Runs abandoned before a terminal decision are not task outcomes.
        if not final or final.get("status") == "aborted":
            continue
        finished_at_ms = max(int(final.get("recorded_at_ms", 0)), int(start.get("recorded_at_ms", 0)))
        if since_ms is not None and finished_at_ms < since_ms:
            continue
        risk = latency = actions = model_calls = tokens = offered = 0
        for event in run_events:
            if event.get("event") == "transition":
                risk += max(0, int(event.get("risk_events", 0)))
                latency += max(0, int(event.get("latency_ms", 0)))
                actions += 1
                model_calls += max(0, int(event.get("model_calls", 0)))
                tokens += max(0, int(event.get("tokens", 0)))
                offered += len(event.get("candidates", ()))
        summaries.append(CanaryRunSummary(
            run_id=run_id,
            task_key=start.get("task_key", ""),
            verified_success=final.get("status") == "done" and bool(final.get("verified")),
            risk_events=risk,
            latency_ms=latency,
            actions=actions,
            model_calls=model_calls,
            tokens=tokens,
            offered_candidates=offered,
            finished_at_ms=finished_at_ms,
        ))
    summaries.sort(key=lambda run: (run.finished_at_ms, run.run_id))
    return summaries



def _metrics_from_canary_runs(runs: Iterable[CanaryRunSummary]) -> CanaryMetrics:
    runs = list(runs)
    keys = tuple(sorted({run.task_key for run in runs if run.task_key}))
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
        task_families=len(keys),
        run_ids=tuple(run.run_id for run in runs),
        task_keys=keys,
    )


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

    @classmethod
    def from_events(
        cls,
        events: Iterable[dict],
        baseline_digest: str,
        candidate_digest: str,
        *,
        candidate_since_ms: int | None = None,
    ):
        events = list(events)
        baseline_runs = _canary_run_summaries(events, baseline_digest)
        candidate_runs = _canary_run_summaries(events, candidate_digest, since_ms=candidate_since_ms)
        baseline = _metrics_from_canary_runs(baseline_runs)
        candidate = _metrics_from_canary_runs(candidate_runs)
        base_groups: dict[str, list[CanaryRunSummary]] = defaultdict(list)
        cand_groups: dict[str, list[CanaryRunSummary]] = defaultdict(list)
        for run in baseline_runs:
            base_groups[run.task_key].append(run)
        for run in candidate_runs:
            cand_groups[run.task_key].append(run)
        shared = sorted(set(base_groups) & set(cand_groups))
        candidate_wins = baseline_wins = ties = 0
        pair_rows = []
        for key in shared:
            b = base_groups[key]
            c = cand_groups[key]
            b_rate = sum(int(r.verified_success) for r in b) / len(b)
            c_rate = sum(int(r.verified_success) for r in c) / len(c)
            if c_rate > b_rate:
                candidate_wins += 1
            elif b_rate > c_rate:
                baseline_wins += 1
            else:
                ties += 1
            pair_rows.append({
                "task_key": key,
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
            paired_task_families=len(shared),
            candidate_wins=candidate_wins,
            baseline_wins=baseline_wins,
            ties=ties,
            event_head_hash=head,
            evidence_digest=digest,
            baseline_digest=baseline_digest,
            candidate_digest=candidate_digest,
        )


@dataclass(frozen=True)
class CanaryDecision:
    approved: bool
    reason: str


class CanaryGate:
    """Require matched real executions before replay-selected policy activation."""

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
        return CanaryDecision(True, "bound live canary gates passed" if evidence else "live canary gates passed")


@dataclass(frozen=True)
class HealthDecision:
    healthy: bool
    reason: str
    reference: CanaryMetrics
    observed: CanaryMetrics


class HealthGate:
    """Detect post-promotion drift from hash-verified active-policy traces."""

    def __init__(self, *, min_tasks: int = 20, max_success_regression: float = 0.10, max_extra_risk_rate: float = 0.0):
        self.min_tasks = min_tasks
        self.max_success_regression = max_success_regression
        self.max_extra_risk_rate = max_extra_risk_rate

    def assess(self, reference: CanaryMetrics, observed: CanaryMetrics) -> HealthDecision:
        if observed.tasks < self.min_tasks:
            return HealthDecision(True, "insufficient recent tasks for drift decision", reference, observed)
        if reference.tasks and observed.success_rate + self.max_success_regression < reference.success_rate:
            return HealthDecision(False, "active policy verified-success drift", reference, observed)
        if observed.risk_rate > reference.risk_rate + self.max_extra_risk_rate:
            return HealthDecision(False, "active policy risk-rate drift", reference, observed)
        return HealthDecision(True, "active policy health gates passed", reference, observed)


class PolicyRegistry:
    """Atomic registry with lineage binding, evidence binding, suspension, and rollback.

    Only bounded ``ExplorationPolicy`` data can be promoted. Staging is bound to
    the active parent policy and replay evidence digests, preventing a report
    generated against stale policy state from being activated later.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._lock = threading.RLock()

    @staticmethod
    def _empty():
        return {
            "schema": SCHEMA_VERSION,
            "tcb_version": TCB_VERSION,
            "revision": 0,
            "active": None,
            "staged": None,
            "history": [],
        }

    @staticmethod
    def _validate_record(record: dict | None, label: str):
        if not record:
            return
        policy = ExplorationPolicy.from_dict(record["policy"])
        if policy.digest != record.get("digest"):
            raise ValueError(f"Policy registry {label} digest mismatch")

    def load(self) -> dict:
        with self._lock, _file_lock(self.lock_path):
            return self._load_unlocked()

    def _load_unlocked(self) -> dict:
        if not self.path.exists():
            return self._empty()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema") not in SUPPORTED_SCHEMAS or payload.get("tcb_version") not in SUPPORTED_TCB_VERSIONS:
            raise ValueError("Unsupported policy registry schema/TCB version")
        payload.setdefault("revision", 0)
        payload.setdefault("active", None)
        payload.setdefault("staged", None)
        payload.setdefault("history", [])
        self._validate_record(payload.get("active"), "active")
        self._validate_record(payload.get("staged"), "staged")
        for index, record in enumerate(payload.get("history", [])):
            self._validate_record(record, f"history[{index}]")
        # In-memory migration; the next mutation writes the current schema/TCB.
        payload["schema"] = SCHEMA_VERSION
        payload["tcb_version"] = TCB_VERSION
        return payload

    def _write(self, payload: dict):
        payload = dict(payload)
        payload["schema"] = SCHEMA_VERSION
        payload["tcb_version"] = TCB_VERSION
        payload["revision"] = int(payload.get("revision", 0)) + 1
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)

    def _current_baseline_digest(self, payload: dict) -> str:
        active = payload.get("active")
        return active["digest"] if active and not active.get("suspended") else ExplorationPolicy().digest

    def stage(self, report: DreamReport) -> dict:
        if not report.promotion.approved or report.selected.digest == report.baseline.digest:
            raise ValueError("Only a replay-approved changed policy can be staged")
        with self._lock, _file_lock(self.lock_path):
            return self._stage_unlocked(report)

    def _stage_unlocked(self, report: DreamReport) -> dict:
        payload = self._load_unlocked()
        parent_digest = self._current_baseline_digest(payload)
        if report.baseline.digest != parent_digest:
            raise ValueError("Replay report baseline is stale relative to the active policy")
        payload["staged"] = {
            "policy": report.selected.to_dict(),
            "digest": report.selected.digest,
            "parent_digest": parent_digest,
            "replay_report_hash": _stable_hash(json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":"))),
            "world_pool_digest": report.world_pool_digest,
            "split_manifest_digest": report.split_manifest_digest,
            "evidence_head_hash": report.evidence_head_hash,
            "tcb_versions": list(report.tcb_versions),
            "staged_at_ms": int(time.time() * 1000),
        }
        self._write(payload)
        return payload["staged"]

    def _promote_bound(self, evidence: CanaryEvidence, gate: CanaryGate | None = None) -> dict:
        with self._lock, _file_lock(self.lock_path):
            payload = self._load_unlocked()
            staged = payload.get("staged")
            if not staged:
                raise ValueError("No staged policy")
            current_parent = self._current_baseline_digest(payload)
            if current_parent != staged.get("parent_digest"):
                raise ValueError("Staged policy parent no longer matches the active policy")
            if evidence.candidate_digest and evidence.candidate_digest != staged["digest"]:
                raise ValueError("Canary evidence is not bound to the staged policy digest")
            if evidence.baseline_digest and evidence.baseline_digest != staged.get("parent_digest"):
                raise ValueError("Canary evidence baseline does not match the staged policy parent digest")
            decision = (gate or CanaryGate()).assess(evidence.baseline, evidence.candidate, evidence=evidence)
            if not decision.approved:
                raise ValueError(f"Canary promotion rejected: {decision.reason}")
            previous = payload.get("active")
            if previous:
                payload["history"].append({
                    **previous,
                    "deactivated_at_ms": int(time.time() * 1000),
                    "deactivation_reason": "superseded",
                })
            payload["active"] = {
                **staged,
                "promoted_at_ms": int(time.time() * 1000),
                "suspended": False,
                "canary": {
                    "baseline": asdict(evidence.baseline),
                    "candidate": asdict(evidence.candidate),
                    "paired_task_families": evidence.paired_task_families,
                    "candidate_wins": evidence.candidate_wins,
                    "baseline_wins": evidence.baseline_wins,
                    "ties": evidence.ties,
                    "event_head_hash": evidence.event_head_hash,
                    "evidence_digest": evidence.evidence_digest,
                    "reason": decision.reason,
                },
            }
            payload["staged"] = None
            self._write(payload)
            return payload["active"]

    def promote(
        self,
        baseline: CanaryMetrics,
        candidate: CanaryMetrics,
        gate: CanaryGate | None = None,
        *,
        allow_unbound_metrics: bool = False,
    ) -> dict:
        if not allow_unbound_metrics:
            raise ValueError("Unbound canary metrics are disabled; use promote_from_store()")
        payload = self.load()
        staged = payload.get("staged")
        if not staged:
            raise ValueError("No staged policy")
        decision = (gate or CanaryGate()).assess(baseline, candidate)
        if not decision.approved:
            raise ValueError(f"Canary promotion rejected: {decision.reason}")
        head = "0" * 64
        material = {
            "baseline": asdict(baseline),
            "candidate": asdict(candidate),
            "staged_digest": staged["digest"],
            "mode": "unbound_metrics",
        }
        evidence = CanaryEvidence(
            baseline=baseline,
            candidate=candidate,
            paired_task_families=0,
            candidate_wins=0,
            baseline_wins=0,
            ties=0,
            event_head_hash=head,
            evidence_digest=_stable_hash(json.dumps(material, sort_keys=True, separators=(",", ":"))),
            baseline_digest=staged.get("parent_digest", ""),
            candidate_digest=staged["digest"],
        )
        # Explicit compatibility escape hatch; bypass paired-family requirement.
        permissive = CanaryGate(
            min_tasks=(gate.min_tasks if gate else 12),
            min_baseline_tasks=(gate.min_baseline_tasks if gate else 12),
            min_task_families=0,
            min_paired_task_families=0,
            min_pair_coverage=0.0,
            max_success_regression=(gate.max_success_regression if gate else 0.0),
            max_extra_risk_events=(gate.max_extra_risk_events if gate else 0),
            max_extra_risk_rate=(gate.max_extra_risk_rate if gate else 0.0),
            max_failure_rate_regression=(gate.max_failure_rate_regression if gate else 0.0),
            max_latency_regression_ratio=(gate.max_latency_regression_ratio if gate else 0.25),
            max_action_regression_ratio=(gate.max_action_regression_ratio if gate else 0.25),
            max_token_regression_ratio=(gate.max_token_regression_ratio if gate else 0.25),
        )
        return self._promote_bound(evidence, permissive)

    def promote_from_store(
        self,
        store: ExperienceStore,
        *,
        baseline_policy_digest: str | None = None,
        gate: CanaryGate | None = None,
    ) -> dict:
        payload = self.load()
        staged = payload.get("staged")
        if not staged:
            raise ValueError("No staged policy")
        parent_digest = staged.get("parent_digest")
        if baseline_policy_digest is None:
            baseline_policy_digest = parent_digest
        if not baseline_policy_digest:
            raise ValueError("Cannot resolve baseline policy digest")
        # Bound promotion: evidence must pair the candidate against its staged parent.
        if baseline_policy_digest != parent_digest:
            raise ValueError("Baseline digest must match the staged policy parent digest")
        events = store.load()
        evidence = CanaryEvidence.from_events(
            events,
            baseline_policy_digest,
            staged["digest"],
            candidate_since_ms=staged.get("staged_at_ms"),
        )
        return self._promote_bound(evidence, gate=gate)

    def staged_policy(self) -> ExplorationPolicy:
        staged = self.load().get("staged")
        if not staged:
            raise ValueError("No staged policy")
        return ExplorationPolicy.from_dict(staged["policy"])

    def active_policy(self, default: ExplorationPolicy | None = None) -> ExplorationPolicy:
        active = self.load().get("active")
        if not active or active.get("suspended"):
            return default or ExplorationPolicy()
        return ExplorationPolicy.from_dict(active["policy"])

    def suspend(self, reason: str) -> dict:
        with self._lock, _file_lock(self.lock_path):
            payload = self._load_unlocked()
            active = payload.get("active")
            if not active:
                raise ValueError("No active policy to suspend")
            active["suspended"] = True
            active["suspended_at_ms"] = int(time.time() * 1000)
            active["suspension_reason"] = str(reason)[:512]
            self._write(payload)
            return active

    def resume(self) -> dict:
        with self._lock, _file_lock(self.lock_path):
            payload = self._load_unlocked()
            active = payload.get("active")
            if not active:
                raise ValueError("No active policy to resume")
            active["suspended"] = False
            active.pop("suspended_at_ms", None)
            active.pop("suspension_reason", None)
            self._write(payload)
            return active

    def rollback(self, digest: str | None = None) -> dict:
        with self._lock, _file_lock(self.lock_path):
            payload = self._load_unlocked()
            history = payload.get("history", [])
            if not history:
                raise ValueError("No prior active policy is available for rollback")
            index = None
            if digest is None:
                index = len(history) - 1
            else:
                for i in range(len(history) - 1, -1, -1):
                    if history[i].get("digest") == digest:
                        index = i
                        break
            if index is None:
                raise ValueError("Requested rollback digest is not in registry history")
            target = history.pop(index)
            current = payload.get("active")
            if current:
                history.append({
                    **current,
                    "deactivated_at_ms": int(time.time() * 1000),
                    "deactivation_reason": "rollback",
                })
            target = {k: v for k, v in target.items() if k not in {"deactivated_at_ms", "deactivation_reason"}}
            target["suspended"] = False
            target["rollback_at_ms"] = int(time.time() * 1000)
            target["rollback_from_digest"] = current.get("digest") if current else None
            payload["active"] = target
            payload["history"] = history
            payload["staged"] = None
            self._write(payload)
            return target

    def health_from_store(
        self,
        store: ExperienceStore,
        *,
        gate: HealthGate | None = None,
        recent_tasks: int = 20,
        suspend_on_fail: bool = False,
    ) -> HealthDecision:
        payload = self.load()
        active = payload.get("active")
        if not active:
            raise ValueError("No active policy")
        canary = active.get("canary", {})
        reference_payload = canary.get("candidate")
        if not reference_payload:
            raise ValueError("Active policy has no bound canary reference metrics")
        reference = CanaryMetrics(**reference_payload)
        observed = CanaryMetrics.from_events(store.load(), active["digest"], max_runs=recent_tasks)
        decision = (gate or HealthGate()).assess(reference, observed)
        if not decision.healthy and suspend_on_fail:
            self.suspend(decision.reason)
        return decision


def summarize_usage(usage: dict | None) -> int:
    if not isinstance(usage, dict):
        return 0
    for key in ("total_tokens", "total", "tokens"):
        value = usage.get(key)
        if isinstance(value, (int, float)) and value >= 0:
            return int(value)
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
    if isinstance(prompt, (int, float)) and isinstance(completion, (int, float)):
        return max(0, int(prompt + completion))
    return 0
