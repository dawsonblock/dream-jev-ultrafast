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

from .privacy import action_goal_overlap
from .signing import (
    ANCHOR_DOMAIN,
    EXPERIMENT_PLAN_DOMAIN,
    EXPERIMENT_SIGNING_KEY_ENV,
    PROMOTION_SIGNING_KEY_ENV,
    PROMOTION_VERIFY_KEYS_ENV,
    REGISTRY_ANCHOR_DOMAIN,
    REGISTRY_STATE_DOMAIN,
    EvidenceSigner,
    verify_keys_from_env,
    verify_signature,
)
from .signing import (
    ATTESTATION_DOMAIN as ATTESTATION_SIG_DOMAIN,
)

SCHEMA_VERSION = "jev-dream/4"
SUPPORTED_SCHEMAS = {"jev-dream/1", "jev-dream/2", "jev-dream/3", SCHEMA_VERSION}
# 0.10 adds experiment-tagged transitions: trial runs carry an ``experiment``
# block and are excluded from canary metrics, so pools that predate the
# distinction must not silently co-mingle with it.
# 0.11 relabels propensity semantics (deterministic argmax selection records
# behavior_propensity=None; only a randomized assignment is a propensity) and
# adds experiment_assigned events recorded at randomization time, before the
# authority plane — pools that predate it lack the boundary between a trial
# that executed and a trial that was assigned but vetoed.
# 0.12 makes stamped experiment plans immutable and bound — digest, task key,
# state, model choice, offered catalogue, and policy behavior are verified at
# execution — and enriches experiment_assigned with the pre-treatment context
# (choice/proposal overlap and offered ranks, task identity) the
# intention-to-treat estimator groups on.
# 0.13 adds structured censoring (run_finished.reason: operator_cancel,
# browser_crash, timeout, …), the bounded treatment signature (effect class,
# role, offered rank, workflow phase) on assignment records, and
# domain-separated experiment-plan signatures — pools that predate it cannot
# distinguish why a run was censored, and their trial cells lack the signature
# coordinates that keep class-level deltas from silently generalizing.
# 0.14 tags active causal-policy overrides on the transition (causal_override)
# so an overridden step is evidence of scheduler influence, not an on-policy
# choice the observational priors may fit — pools that predate the tag would
# read that step as the recorded policy's own selection.
# 0.15 adds the mutation execution journal (action_attempted before dispatch,
# action_confirmed / action_indeterminate after) and the
# ``indeterminate_execution`` censoring reason: a mutation whose
# acknowledgement was lost is recorded as unknowable, never as a retryable
# stale page — pools that predate it cannot distinguish a press that may have
# landed from a click that provably never ran.
TCB_VERSION = "jev-ultrafast-tcb/0.15"
SUPPORTED_TCB_VERSIONS = {
    "jev-ultrafast-tcb/0.3",
    "jev-ultrafast-tcb/0.4",
    "jev-ultrafast-tcb/0.5",
    "jev-ultrafast-tcb/0.6",
    "jev-ultrafast-tcb/0.7",
    "jev-ultrafast-tcb/0.8",
    "jev-ultrafast-tcb/0.9",
    "jev-ultrafast-tcb/0.10",
    "jev-ultrafast-tcb/0.11",
    "jev-ultrafast-tcb/0.12",
    "jev-ultrafast-tcb/0.13",
    "jev-ultrafast-tcb/0.14",
    TCB_VERSION,
}
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


def _fsync_dir(path: Path):
    """Durable-rename bookkeeping: fsync the directory after atomic file writes."""
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def task_key(goal: str) -> str:
    """Stable task identity without requiring plaintext goal storage."""
    return _stable_hash(" ".join(str(goal).split()).strip().lower())


def new_run_id() -> str:
    return uuid.uuid4().hex


def candidate_catalog_digest(candidates: Iterable[dict]) -> str:
    """Digest the replay-relevant candidate catalogue in its original order.

    Every field that can alter retention, ordering, grouping, or policy
    evaluation is bound — including ``node``, which ``duplicate_node_cap``
    groups by. Two catalogues differing only in node identity are different
    replay evidence.
    """
    compact = []
    for candidate in candidates:
        compact.append({
            "id": candidate.get("id"),
            "kind": candidate.get("kind"),
            "node": candidate.get("node"),
            "role": candidate.get("role"),
            "label": candidate.get("label", ""),
            "value": candidate.get("value", ""),
            "current_value": candidate.get("current_value"),
            "option_index": candidate.get("option_index"),
            "option_label": candidate.get("option_label"),
            "checked": candidate.get("checked"),
            "ctx": candidate.get("ctx"),
            "goal_overlap": max(0, int(candidate.get("goal_overlap", 0))),
        })
    return _stable_hash(json.dumps(compact, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


EXPERIMENT_PLAN_SCHEMA = "jev-experiment-plan/1"


def experiment_plan_digest(plan: dict) -> str:
    """Canonical content digest of a stamped experiment plan.

    The digest covers every field of the plan — hypothesis, provenance, and
    the bindings revalidated at execution (task key, family key, state
    fingerprint, model choice, offered catalogue, policy behavior) — so a
    stamped plan is an immutable artifact: any post-hoc edit invalidates the
    digest instead of silently reinterpreting the plan. The ``digest`` field
    and the ``authority`` signature block are excluded: the signature covers
    the digest, not the other way around.
    """
    material = {k: v for k, v in plan.items() if k not in {"digest", "authority"}}
    return _stable_hash(json.dumps(material, sort_keys=True, separators=(",", ":")))


def experiment_plan_signature(plan: dict, signer) -> dict:
    """The domain-separated authority block for a stamped plan.

    SHA-256 alone establishes *integrity* — anyone who can write the plan can
    compute a valid digest. An Ed25519 signature over
    ``jev-dream/experiment-plan/v1:<digest>`` establishes *provenance*: the
    plan was stamped by whoever holds the improvement key. Execution
    distinguishes ``digest_valid`` from ``authority_signature_valid``; with
    trusted verification keys configured, an unsigned or forged plan fails
    closed.
    """
    if signer is None:
        return {}
    return {
        "key_id": signer.key_id,
        "signature": signer.sign_hex(plan["digest"], EXPERIMENT_PLAN_DOMAIN),
        "domain": EXPERIMENT_PLAN_DOMAIN.decode("ascii"),
    }


def experiment_plan_authority(plan: dict, trusted_keys) -> dict:
    """Verify a plan's signature block.

    Returns ``{"present", "signature_valid", "trusted"}``. ``trusted`` is
    only meaningful when a trusted-key set is configured; a present signature
    is always checked for self-consistency, so a forged self-claim fails even
    in unsigned-compatibility mode.
    """
    authority = plan.get("authority")
    if not isinstance(authority, dict) or not authority.get("signature"):
        return {"present": False, "signature_valid": False, "trusted": False}
    key_id = str(authority.get("key_id") or "")
    digest = str(plan.get("digest") or "")
    valid = bool(
        key_id
        and digest
        and verify_signature(
            key_id, digest, str(authority.get("signature") or ""), EXPERIMENT_PLAN_DOMAIN
        )
    )
    trusted = bool(valid and trusted_keys and key_id in set(trusted_keys))
    return {"present": True, "signature_valid": valid, "trusted": trusted}


def experiment_plan_signer_from_env() -> EvidenceSigner | None:
    """The improvement authority's plan signer.

    ``JEV_EXPERIMENT_SIGNING_KEY`` takes precedence, falling back to the
    promotion key and then the evidence key — the same precedence chain the
    policy registry uses for promotion authority. No key configured means
    plans are stamped digest-only and the execution plane stays in explicit
    unsigned-compatibility mode.
    """
    return (
        EvidenceSigner.from_env(EXPERIMENT_SIGNING_KEY_ENV)
        or EvidenceSigner.from_env(PROMOTION_SIGNING_KEY_ENV)
        or EvidenceSigner.from_env()
    )


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
    overlap_exponent: float = 1.0
    order_penalty: float = 0.00001
    click_bonus: float = 0.5
    fill_bonus: float = 1.5
    select_bonus: float = 1.0
    model_action_limit: int = 250
    duplicate_node_cap: int = 250
    min_goal_overlap: int = 0
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
            self.overlap_exponent,
            self.order_penalty,
            self.click_bonus,
            self.fill_bonus,
            self.select_bonus,
        ):
            if not math.isfinite(value):
                raise ValueError("Exploration policy weights must be finite")
        if not 0.25 <= self.overlap_exponent <= 2.0:
            raise ValueError("overlap_exponent must be between 0.25 and 2.0")
        if not 16 <= self.model_action_limit <= 250:
            raise ValueError("model_action_limit must be between 16 and 250")
        if not 1 <= self.duplicate_node_cap <= 250:
            raise ValueError("duplicate_node_cap must be between 1 and 250")
        if not 0 <= self.min_goal_overlap <= 5:
            raise ValueError("min_goal_overlap must be between 0 and 5")
        for quota in (self.click_quota, self.fill_quota, self.select_quota):
            if not 1 <= quota <= 250:
                raise ValueError("candidate quotas must be between 1 and 250")
        if not 1 <= self.max_actions <= 120:
            raise ValueError("max_actions must be between 1 and 120")
        if not 2 <= self.no_progress_window <= 10:
            raise ValueError("no_progress_window must be between 2 and 10")

    def candidate_score(self, action: dict, goal_tokens: set[str], order: int, overlap=None) -> float:
        if overlap is None:
            overlap = action_goal_overlap(action, goal_tokens)
        bonus = {
            "fill": self.fill_bonus,
            "select": self.select_bonus,
            "click": self.click_bonus,
        }.get(action.get("kind"), 0.0)
        return (overlap ** self.overlap_exponent) * self.goal_overlap_weight + bonus - order * self.order_penalty

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

    _BEHAVIOR_FIELDS = (
        "goal_overlap_weight",
        "overlap_exponent",
        "order_penalty",
        "click_bonus",
        "fill_bonus",
        "select_bonus",
        "model_action_limit",
        "duplicate_node_cap",
        "min_goal_overlap",
        "click_quota",
        "fill_quota",
        "select_quota",
        "max_actions",
        "no_progress_window",
    )

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return _stable_hash(encoded)

    @property
    def behavior_digest(self) -> str:
        """Identity-free digest of only the parameters that change runtime behavior.

        ``digest`` binds name/version lineage; ``behavior_digest`` deduplicates
        candidates that differ in identity but not in what the agent would do.
        """
        material = {field: getattr(self, field) for field in self._BEHAVIOR_FIELDS}
        return _stable_hash(json.dumps(material, sort_keys=True, separators=(",", ":")))


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
    estimated_tokens: int = 0
    estimated_latency_ms: int = 0
    estimated_score: float | None = None

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

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        strict: bool = False,
        signer=None,
        verify_key: str | None = None,
        verify_keys=None,
        require_signatures: bool = False,
        anchor_path: str | os.PathLike | None = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._lock = threading.Lock()
        # Default appends verify only the tail so they stay O(1). ``strict``
        # verifies the whole chain first so a mid-chain corruption cannot keep
        # accumulating events into a store that load() would reject.
        self.strict = strict
        # Optional chain-head anchor: after each append the current head is
        # written to a separate (signed) checkpoint file that may live on
        # different storage. Reads fail closed when the anchor records a head
        # the log no longer reaches — the difference between a torn crash
        # write and deliberate tail truncation becomes observable.
        self.anchor_path = Path(anchor_path) if anchor_path else None
        # Optional authenticity: a signer object exposing ``key_id`` and
        # ``sign_hex`` (Ed25519 EvidenceSigner by default via env key). The
        # signer interface is injectable so private material can live outside
        # the agent process. ``verify_keys`` is the trusted set of hex Ed25519
        # public keys — a set rather than one key so rotation keeps older
        # signed events verifiable. ``require_signatures`` additionally
        # rejects unsigned events.
        self.signer = signer if signer is not None else EvidenceSigner.from_env()
        keys = {k.strip() for k in (verify_keys or []) if k and k.strip()}
        if verify_key and verify_key.strip():
            keys.add(verify_key.strip())
        keys.update(verify_keys_from_env())
        self.verify_keys = keys
        # Single-key form retained for callers inspecting configuration.
        self.verify_key = next(iter(keys)) if len(keys) == 1 else None
        self.require_signatures = bool(require_signatures or os.environ.get("JEV_REQUIRE_SIGNED_EVIDENCE"))

    @staticmethod
    def _event_hash(payload: dict) -> str:
        material = {k: v for k, v in payload.items() if k not in {"event_hash", "signature", "key_id"}}
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

    _RESERVED_EVENT_KEYS = {
        "schema",
        "tcb_version",
        "recorded_at_ms",
        "prev_hash",
        "event_hash",
        "signature",
        "key_id",
    }

    def append(self, event: dict) -> dict:
        if self._RESERVED_EVENT_KEYS & event.keys():
            raise ValueError(
                f"DREAM event cannot set reserved chain keys: {sorted(self._RESERVED_EVENT_KEYS & event.keys())}"
            )
        with self._lock, _file_lock(self.lock_path):
            needs_separator = self._truncate_torn_tail_unlocked()
            if self.strict or self.anchor_path is not None:
                # Anchored stores check the log against the checkpoint BEFORE
                # appending — otherwise an append would re-anchor over a
                # truncated tail and erase the very gap the anchor detects.
                events, _ = self._load_unlocked()
                if self.anchor_path is not None:
                    self._check_anchor_unlocked(events)
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
            if self.signer is not None:
                payload["key_id"] = self.signer.key_id
                payload["signature"] = self.signer.sign_hex(payload["event_hash"])
            line = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            created = not self.path.exists()
            with self.path.open("a", encoding="utf-8") as handle:
                # A complete record whose terminating newline was lost to a
                # crash needs the separator restored before the next event.
                handle.write(("\n" if needs_separator else "") + line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if created:
                _fsync_dir(self.path.parent)
            if self.anchor_path is not None:
                # Checkpoint the new head under the same lock so the anchor is
                # never ahead of a head that did not survive.
                self._write_anchor_unlocked(
                    payload["event_hash"], self._anchor_sequence_unlocked())
            return payload

    _ANCHOR_CONTENT_DOMAIN = "jev-dream/chain-head-anchor/v1"

    def _anchor_sequence_unlocked(self) -> int:
        anchor = self._read_anchor()
        if anchor is not None:
            return int(anchor.get("sequence", 0)) + 1
        # Bootstrap: count the records the log currently holds (O(n) once).
        if not self.path.exists():
            return 0
        return sum(1 for raw in self.path.read_bytes().split(b"\n") if raw.strip())

    def _read_anchor(self) -> dict | None:
        if self.anchor_path is None or not self.anchor_path.exists():
            return None
        try:
            anchor = json.loads(self.anchor_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("Invalid DREAM chain-head anchor file") from exc
        if not isinstance(anchor, dict) or anchor.get("kind") != "chain_head_anchor":
            raise ValueError("Invalid DREAM chain-head anchor file")
        return anchor

    def _write_anchor_unlocked(self, head: str, sequence: int) -> dict | None:
        if self.anchor_path is None:
            return None
        material = {
            "kind": "chain_head_anchor",
            "head": head,
            "sequence": sequence,
            "signed_at_ms": int(time.time() * 1000),
        }
        digest = _stable_hash(
            f"{self._ANCHOR_CONTENT_DOMAIN}\n"
            + json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        )
        anchor = {**material, "digest": digest}
        if self.signer is not None:
            anchor["key_id"] = self.signer.key_id
            anchor["signature"] = self.signer.sign_hex(digest, domain=ANCHOR_DOMAIN)
        self.anchor_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.anchor_path.with_suffix(self.anchor_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(anchor, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.anchor_path)
        _fsync_dir(self.anchor_path.parent)
        return anchor

    def _check_anchor_unlocked(self, events: list[dict]) -> dict | None:
        """Anchor consistency when ``anchor_path`` is configured.

        Fails closed when the anchor is missing while the store holds events,
        when a signed anchor fails verification, when an unsigned anchor is
        found while trust keys are configured, or when the anchor head
        disagrees with the recomputed log head (truncation or stale anchor).
        """
        if self.anchor_path is None:
            return None
        anchor = self._read_anchor()
        if anchor is None:
            if events:
                raise ValueError("DREAM chain-head anchor missing while the store holds events")
            return None
        material = {k: v for k, v in anchor.items() if k not in {"digest", "signature", "key_id"}}
        digest = _stable_hash(
            f"{self._ANCHOR_CONTENT_DOMAIN}\n"
            + json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        )
        if anchor.get("digest") != digest:
            raise ValueError("DREAM chain-head anchor digest mismatch")
        signature = anchor.get("signature")
        if signature is not None:
            if not self.verify_keys:
                raise ValueError("Signed DREAM anchor but no verification key is configured")
            if anchor.get("key_id") not in self.verify_keys:
                raise ValueError("DREAM anchor signed by an unexpected key")
            if not verify_signature(
                anchor["key_id"], digest, str(signature), domain=ANCHOR_DOMAIN
            ):
                raise ValueError("DREAM anchor signature verification failure")
        elif self.verify_keys or self.require_signatures:
            raise ValueError("Unsigned DREAM chain-head anchor while trust keys are configured")
        head = events[-1]["event_hash"] if events else "0" * 64
        if anchor.get("head") != head or int(anchor.get("sequence", -1)) != len(events):
            raise ValueError(
                "DREAM chain-head anchor disagrees with the log: evidence tail "
                "truncated or anchor stale"
            )
        return anchor

    def reanchor(self) -> dict:
        """Write a fresh anchor for the current verified head.

        This is the operator-level resolution after a confirmed truncation or
        a lost anchor file. It is deliberately explicit — appends never
        silently re-anchor, so a gap between anchor and log always surfaces.
        """
        with self._lock, _file_lock(self.lock_path):
            events, _ = self._load_unlocked()
            head = events[-1]["event_hash"] if events else "0" * 64
            anchor = self._write_anchor_unlocked(head, len(events))
            if anchor is None:
                raise ValueError("No anchor_path configured for this store")
            return anchor

    def _truncate_torn_tail_unlocked(self) -> bool:
        """Discard a crash-torn final fragment; return True when a separator is owed.

        A write is ``line + "\\n"``, so a file not ending in a newline carries
        an unfinished final record. Truncating is allowed only when the
        fragment is a recognizable prefix of an event object (``{`` or
        ``{"...``) or pure whitespace — anything else means corruption or
        tampering and stays fail-closed. A complete record missing only its
        newline is left in place; the caller must restore the separator.
        """
        if not self.path.exists() or self.path.stat().st_size == 0:
            return False
        size = self.path.stat().st_size
        with self.path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) == b"\n":
                return False
            # The fragment is the bytes after the last newline — scan backward
            # in chunks so the healthy-tail check stays O(fragment), not
            # O(file); only a fragment spanning the whole file degrades to a
            # full read, which is the worst case either way.
            pos = size
            frag = b""
            while pos > 0:
                take = min(65536, pos)
                pos -= take
                handle.seek(pos)
                chunk = handle.read(take)
                nl = chunk.rfind(b"\n")
                if nl >= 0:
                    frag = chunk[nl + 1:] + frag
                    break
                frag = chunk + frag
        stripped = frag.strip()
        try:
            parsed = json.loads(frag) if stripped else None
        except (json.JSONDecodeError, UnicodeDecodeError):
            parsed = None
        if isinstance(parsed, dict):
            # A complete record whose terminating newline was lost to a crash —
            # keep it; the caller restores the separator before the next event.
            return True
        if parsed is not None or (stripped and stripped != b"{" and not stripped.startswith(b'{"')):
            raise ValueError("Unrecoverable DREAM store tail: not a torn event prefix")
        offset = size - len(frag)
        with self.path.open("r+b") as handle:
            handle.truncate(offset)
            handle.flush()
            os.fsync(handle.fileno())
        return False

    def _load_unlocked(self) -> tuple[list[dict], int | None]:
        """Validate the chain; return ``(events, torn_tail_offset | None)``.

        A clearly incomplete final fragment is reported, not fatal: the writer
        crashed between or during writes. Every other decode failure, and any
        parsed record with a bad hash, link, or signature, stays fail-closed.
        """
        if not self.path.exists():
            return [], None
        data = self.path.read_bytes()
        lines = data.split(b"\n")
        last_idx = len(lines) - 1
        events = []
        torn_offset = None
        expected_prev = "0" * 64
        offset = 0
        for i, raw in enumerate(lines):
            line_no = i + 1
            line_start = offset
            offset += len(raw) + 1
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                if i == last_idx and (raw.strip() == b"{" or raw.strip().startswith(b'{"')):
                    torn_offset = line_start
                    break
                raise ValueError(f"Invalid DREAM event JSON on line {line_no}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"Invalid DREAM event JSON on line {line_no}")
            if event.get("schema") not in SUPPORTED_SCHEMAS:
                raise ValueError(f"Unsupported DREAM schema on line {line_no}")
            if event.get("tcb_version") not in SUPPORTED_TCB_VERSIONS:
                raise ValueError(f"Unsupported DREAM TCB version on line {line_no}")
            if event.get("prev_hash") != expected_prev:
                raise ValueError(f"DREAM hash-chain predecessor mismatch on line {line_no}")
            if event.get("event_hash") != self._event_hash(event):
                raise ValueError(f"DREAM hash-chain integrity failure on line {line_no}")
            signature = event.get("signature")
            if signature is not None:
                if not self.verify_keys:
                    raise ValueError(
                        f"Signed DREAM event on line {line_no} but no verification key is configured"
                    )
                key_id = event.get("key_id")
                if key_id not in self.verify_keys:
                    raise ValueError(f"DREAM evidence signed by an unexpected key on line {line_no}")
                if not verify_signature(key_id, event["event_hash"], signature):
                    raise ValueError(f"DREAM evidence signature verification failure on line {line_no}")
            elif self.require_signatures:
                raise ValueError(f"Unsigned DREAM event on line {line_no} with signatures required")
            expected_prev = event["event_hash"]
            events.append(event)
        return events, torn_offset

    def load(self) -> list[dict]:
        with self._lock, _file_lock(self.lock_path):
            events, _ = self._load_unlocked()
            self._check_anchor_unlocked(events)
            return events

    def verify(self) -> dict:
        with self._lock, _file_lock(self.lock_path):
            events, torn_offset = self._load_unlocked()
            anchor = self._check_anchor_unlocked(events)
        signed = sum(1 for event in events if event.get("signature"))
        report = {
            "events": len(events),
            "head_hash": events[-1]["event_hash"] if events else "0" * 64,
            "schemas": sorted({event["schema"] for event in events}),
            "tcb_versions": sorted({event["tcb_version"] for event in events}),
            "signed_events": signed,
            "unsigned_events": len(events) - signed,
            "signature_key_ids": sorted({event["key_id"] for event in events if event.get("key_id")}),
            "signatures_checked": bool(self.verify_keys) if signed else None,
            "torn_tail_recovered": torn_offset is not None,
        }
        if self.anchor_path is not None:
            report["anchor"] = {
                "path": str(self.anchor_path),
                "present": anchor is not None,
                "consistent": anchor is not None,
                "sequence": anchor.get("sequence") if anchor else None,
                "signed": bool(anchor.get("signature")) if anchor else False,
            }
        return report

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
    selected_observed_rank: int | None
    page_changed: bool
    latency_ms: int
    model_calls: int
    tokens: int
    stale_or_failure: int
    risk_events: int
    terminal: str | None = None
    verified: bool = False
    catalog: str = "recorded"
    offered_digest: str | None = None
    offered_count: int | None = None
    selected_offered_rank: int | None = None
    selected_propensity: float | None = None
    experiment: dict | None = None
    causal_override: dict | None = None


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
    family_key: str = ""

    @property
    def digest(self) -> str:
        payload = {
            "key": self.key,
            "task_key": self.task_key,
            "family_key": self.family_key,
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
            family_key = str(start.get("task_family") or "").strip().lower() or key
            run_policy = None
            if start.get("policy") is not None:
                run_policy = ExplorationPolicy.from_dict(start["policy"])
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
                observed_rank = event.get("selected_observed_rank")
                if observed_rank is None:
                    # Pre-0.9 traces recorded the observed-catalogue rank under
                    # the ambiguous name ``selected_rank``.
                    observed_rank = event.get("selected_rank")
                if observed_rank is None:
                    observed_rank = next(
                        (i for i, item in enumerate(event_candidates) if item.get("id") == selected_id),
                        None,
                    )
                stored_offered = event.get("offered_digest")
                recorded_offered_rank = event.get("selected_offered_rank")
                recorded_offered_rank = (
                    int(recorded_offered_rank) if recorded_offered_rank is not None else None
                )
                # A *distinct* offered catalogue exists only when the trace says
                # so: catalog=="observed", an offered digest/count, or an
                # explicit offered rank. A recorded catalogue means the observed
                # catalogue itself was the offer.
                has_offered = (
                    event.get("catalog") == "observed"
                    or stored_offered is not None
                    or event.get("offered_count") is not None
                    or recorded_offered_rank is not None
                )
                if has_offered and run_policy is not None:
                    # The recorded policy must be able to re-derive the catalogue
                    # the model was offered from the observed catalogue, or the
                    # trace evidence is inconsistent.
                    expected_offered = ReplaySimulator._retained_candidates(run_policy, event_candidates)
                    if stored_offered is not None and candidate_catalog_digest(expected_offered) != stored_offered:
                        raise ValueError(f"Run {run_id} has an offered catalogue digest mismatch")
                    if event.get("offered_count") is not None and int(event["offered_count"]) != len(
                        expected_offered
                    ):
                        raise ValueError(f"Run {run_id} has an offered count mismatch")
                    selected_offered_rank = next(
                        (i for i, item in enumerate(expected_offered) if item.get("id") == selected_id),
                        None,
                    )
                    if recorded_offered_rank is not None and recorded_offered_rank != selected_offered_rank:
                        raise ValueError(f"Run {run_id} has a selected offered-rank mismatch")
                elif has_offered:
                    # Offered catalogue recorded but the policy is unknown; the
                    # recorded coordinate is the only honest value available.
                    selected_offered_rank = recorded_offered_rank
                else:
                    # "recorded" catalogues: offered == observed, so a recorded
                    # offered rank must equal the observed rank.
                    if recorded_offered_rank is not None and recorded_offered_rank != observed_rank:
                        raise ValueError(f"Run {run_id} has a selected offered-rank mismatch")
                    selected_offered_rank = observed_rank
                # ``selected_propensity`` (v0.7.x traces) recorded a model score
                # mislabeled as a propensity; ``behavior_propensity`` records
                # the honest value — None for deterministic argmax selection,
                # the assignment probability for a randomized trial step.
                propensity = event.get("selected_propensity")
                if propensity is None:
                    propensity = event.get("behavior_propensity")
                tr = RecordedTransition(
                    run_id=run_id,
                    task_key=key,
                    state=event["state"],
                    next_state=event["next_state"],
                    selected_id=selected_id,
                    selected_kind=event["selected"].get("kind", ""),
                    candidate_actions=event_candidates,
                    candidate_digest=observed_digest,
                    selected_observed_rank=observed_rank,
                    selected_offered_rank=selected_offered_rank,
                    selected_propensity=float(propensity) if propensity is not None else None,
                    page_changed=bool(event.get("page_changed")),
                    latency_ms=max(0, int(event.get("latency_ms", 0))),
                    model_calls=max(0, int(event.get("model_calls", 1))),
                    tokens=max(0, int(event.get("tokens", 0))),
                    stale_or_failure=max(0, int(event.get("stale_or_failure", 0))),
                    risk_events=max(0, int(event.get("risk_events", 0))),
                    terminal=terminal,
                    verified=verified,
                    catalog=str(event.get("catalog") or "recorded"),
                    offered_digest=stored_offered,
                    offered_count=(
                        max(0, int(event["offered_count"]))
                        if event.get("offered_count") is not None
                        else len(event_candidates)
                    ),
                    experiment=event.get("experiment"),
                    causal_override=event.get("causal_override"),
                )
                transitions.append(tr)
            worlds.append(cls(
                key=run_id,
                task_key=key,
                goal=start.get("goal", ""),
                start_state=start.get("state", "ROOT"),
                tcb_version=run_tcb,
                trajectory=tuple(transitions),
                family_key=family_key,
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

    def evaluate(
        self,
        policy: ExplorationPolicy,
        *,
        max_steps: int | None = None,
        cost_model=None,
        outcome_model=None,
        choice_model=None,
        trial_model=None,
        collect_proposals: bool = False,
    ) -> ReplayResult:
        estimated = bool(cost_model is not None and getattr(cost_model, "reliable", False))
        summaries = [
            self._evaluate_world(
                world,
                policy,
                max_steps=max_steps,
                cost_model=cost_model if estimated else None,
                outcome_model=outcome_model,
                choice_model=choice_model,
                trial_model=trial_model,
                collect_proposals=collect_proposals,
            )
            for world in self.worlds
        ]
        return ReplayResult(metrics=self._aggregate(summaries, estimated=estimated), per_world=tuple(summaries))

    def _evaluate_world(
        self,
        world: ReplayWorld,
        policy: ExplorationPolicy,
        *,
        max_steps: int | None,
        cost_model=None,
        outcome_model=None,
        choice_model=None,
        trial_model=None,
        collect_proposals: bool = False,
    ):
        summary = self._empty_world(world)
        step_limit = min(policy.max_actions, max_steps or policy.max_actions)
        no_progress = 0
        for step_index, transition in enumerate(world.trajectory[:step_limit]):
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
            if cost_model is not None:
                # Anchored efficiency estimate: keep the recorded measurement and
                # adjust only by the learned marginal cost of the offered-count
                # delta. This is hypothesis prioritization, not evidence.
                recorded_offered = transition.offered_count or len(transition.candidate_actions)
                predicted = cost_model.predict(len(offered))
                recorded = cost_model.predict(recorded_offered)
                summary["est_tokens"] += max(0.0, transition.tokens + predicted["tokens"] - recorded["tokens"])
                summary["est_latency_ms"] += max(
                    0.0, transition.latency_ms + predicted["latency_ms"] - recorded["latency_ms"]
                )
            if outcome_model is not None or choice_model is not None:
                selected_overlap = next(
                    (
                        int(c.get("goal_overlap", 0))
                        for c in transition.candidate_actions
                        if c.get("id") == transition.selected_id
                    ),
                    0,
                )
            if outcome_model is not None:
                prediction = outcome_model.predict(
                    kind=transition.selected_kind,
                    goal_overlap=selected_overlap,
                    rank=transition.selected_offered_rank,
                )
                summary["predicted_change"] += prediction["p_page_changed"]
                summary["prediction_samples"] += prediction["n"]
            if choice_model is not None or trial_model is not None:
                # The candidate policy's recomputed offered rank — the rank the
                # recorded action would have had under *this* policy's offered
                # catalogue — so the signal is policy-dependent. Counterfactual
                # proposals annotate the summary only; they are never evidence
                # and never reach a gate.
                offered_rank = next(
                    (
                        i for i, item in enumerate(offered)
                        if item.get("id") == transition.selected_id
                    ),
                    None,
                )
                prediction = None
                if choice_model is not None:
                    prediction = choice_model.predict(
                        kind=transition.selected_kind,
                        goal_overlap=selected_overlap,
                        rank=offered_rank,
                        task_family=world.family_key,
                    )
                    summary["choice_predicted_change"] += prediction["p_progress"]
                    summary["choice_uncertainty"] += prediction["uncertainty"]
                    summary["choice_confident"] += int(prediction["confident"])
                # The causal prior speaks first: a candidate arm with a
                # reliable positive measured effect is a hypothesis already
                # supported by randomized evidence. The observational prior
                # proposes only what trials have not refuted — a reliably
                # non-positive divergence is settled, not re-tested.
                proposal = None
                proposal_source = None
                if trial_model is not None:
                    proposal = trial_model.choose(
                        offered,
                        model_choice={
                            "id": transition.selected_id,
                            "kind": transition.selected_kind,
                            "role": next(
                                (
                                    c.get("role")
                                    for c in transition.candidate_actions
                                    if c.get("id") == transition.selected_id
                                ),
                                None,
                            ),
                            # The effect class recorded when the transition was
                            # compacted (full structural context); older stores
                            # carry none and resolve as an honest wildcard —
                            # never re-classify a compacted candidate whose ctx
                            # is gone, a degraded class could match the wrong
                            # signature cell instead of matching everything.
                            "effect": next(
                                (
                                    c.get("effect")
                                    for c in transition.candidate_actions
                                    if c.get("id") == transition.selected_id
                                ),
                                None,
                            ),
                            "goal_overlap": selected_overlap,
                        },
                        task_family=world.family_key,
                        phase=step_index,
                    )
                    if proposal is not None:
                        proposal_source = trial_model
                if proposal is None and choice_model is not None:
                    proposal = choice_model.choose(offered, task_family=world.family_key)
                    if proposal is not None:
                        proposal_overlap = next(
                            (
                                int(c.get("goal_overlap", 0) or 0)
                                for c in offered
                                if c.get("id") == proposal["id"]
                            ),
                            0,
                        )
                        if trial_model is not None and trial_model.refuted(
                            kind=str(proposal.get("kind") or "unknown"),
                            goal_overlap=proposal_overlap,
                            model_kind=transition.selected_kind,
                            model_overlap=selected_overlap,
                            model_effect=next(
                                (
                                    c.get("effect")
                                    for c in transition.candidate_actions
                                    if c.get("id") == transition.selected_id
                                ),
                                None,
                            ),
                            model_role=next(
                                (
                                    c.get("role")
                                    for c in transition.candidate_actions
                                    if c.get("id") == transition.selected_id
                                ),
                                None,
                            ),
                            model_rank=offered_rank,
                            proposal_effect=next(
                                (
                                    c.get("effect")
                                    for c in offered
                                    if c.get("id") == proposal["id"]
                                ),
                                None,
                            ),
                            proposal_role=next(
                                (
                                    c.get("role")
                                    for c in offered
                                    if c.get("id") == proposal["id"]
                                ),
                                None,
                            ),
                            proposal_rank=next(
                                (
                                    i
                                    for i, item in enumerate(offered)
                                    if item.get("id") == proposal["id"]
                                ),
                                None,
                            ),
                            phase=step_index,
                            task_family=world.family_key,
                        ):
                            proposal = None
                        else:
                            proposal_source = choice_model
                if proposal is not None:
                    summary["choice_proposed_change"] += proposal.get("p_progress") or 0.0
                    summary["choice_proposal_uncertainty"] += proposal.get("uncertainty") or 0.0
                    summary["choice_proposal_confident"] += int(bool(proposal.get("confident")))
                    divergent = proposal["id"] != transition.selected_id
                    summary["choice_divergences"] += int(divergent)
                    if collect_proposals and divergent:
                        # An experiment candidate: the state, the action the
                        # recorded policy actually took, and the action the
                        # prior would substitute — plus the bindings a live
                        # agent revalidates before executing the plan: the
                        # task, the model-choice the hypothesis is conditioned
                        # on, the offered catalogue it diverged inside, and
                        # the policy/model identities that generated it.
                        # Executing it for real — under the same authority
                        # plane — is how a counterfactual hypothesis becomes
                        # signed evidence.
                        expected_delta = proposal.get("expected_delta")
                        if expected_delta is None:
                            expected_delta = (
                                proposal["p_progress"] - prediction["p_progress"]
                                if prediction is not None
                                else 0.0
                            )
                        summary["experiment_proposals"].append({
                            "world": world.key,
                            "task_key": world.task_key,
                            "family_key": world.family_key,
                            "step": step_index,
                            "state": transition.state,
                            "historical": {
                                "id": transition.selected_id,
                                "kind": transition.selected_kind,
                                "effect": next(
                                    (
                                        c.get("effect")
                                        for c in transition.candidate_actions
                                        if c.get("id") == transition.selected_id
                                    ),
                                    None,
                                ),
                                "role": next(
                                    (
                                        c.get("role")
                                        for c in transition.candidate_actions
                                        if c.get("id") == transition.selected_id
                                    ),
                                    None,
                                ),
                                "goal_overlap": selected_overlap,
                                "offered_rank": offered_rank,
                                "propensity": transition.selected_propensity,
                            },
                            # The stamped proposal carries the recorded
                            # signature coordinates — the scheduler and any
                            # later refutation query resolve against exactly
                            # the treatment class this plan was stamped under.
                            "proposal": {
                                **proposal,
                                "effect": next(
                                    (
                                        c.get("effect")
                                        for c in offered
                                        if c.get("id") == proposal["id"]
                                    ),
                                    proposal.get("effect"),
                                ),
                                "role": next(
                                    (
                                        c.get("role")
                                        for c in offered
                                        if c.get("id") == proposal["id"]
                                    ),
                                    proposal.get("role"),
                                ),
                            },
                            "proposal_offered_rank": next(
                                (
                                    i
                                    for i, item in enumerate(offered)
                                    if item.get("id") == proposal["id"]
                                ),
                                None,
                            ),
                            "offered_catalogue_digest": candidate_catalog_digest(offered),
                            "policy_behavior_digest": policy.behavior_digest,
                            # The digest of whichever prior generated this
                            # hypothesis — observational ChoiceModel or causal
                            # TrialChoiceModel — bound into the stamped plan,
                            # plus the explicit channel so a live agent can
                            # tell causal provenance from correlational.
                            "choice_model_digest": getattr(proposal_source, "digest", None),
                            "origin": (
                                "randomized" if proposal_source is trial_model else "observational"
                            ),
                            "causal_model_digest": (
                                getattr(trial_model, "digest", None)
                                if proposal_source is trial_model
                                else None
                            ),
                            "selected_prior": (
                                {
                                    "p_progress": prediction["p_progress"],
                                    "uncertainty": prediction["uncertainty"],
                                    "confident": prediction["confident"],
                                    "source": prediction.get("source"),
                                }
                                if prediction is not None
                                else None
                            ),
                            "expected_delta": expected_delta,
                            "expected_uncertainty": (
                                proposal.get("uncertainty")
                                if proposal.get("uncertainty") is not None
                                else (prediction or {}).get("uncertainty")
                            ),
                            "created_at_ms": int(time.time() * 1000),
                        })
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
        """Re-derive the offered catalogue; must match model.candidate_actions.

        Stored ``goal_overlap`` is computed on sanitized labels, as is live
        scoring, so the replayed catalogue is identical to what the model was
        offered online. Candidates lacking ``node`` metadata (older schema)
        are treated as unique nodes so ``duplicate_node_cap`` cannot merge them.
        """
        candidates = [dict(c) for c in candidates if c.get("id")]
        controls = [c for c in candidates if c.get("kind") not in {"click", "fill", "select"}]
        regular = []
        for index, candidate in enumerate(candidates):
            if candidate.get("kind") not in {"click", "fill", "select"}:
                continue
            overlap = max(0, int(candidate.get("goal_overlap", 0)))
            if overlap < policy.min_goal_overlap:
                continue
            regular.append((index, candidate, overlap))

        def score(pair):
            index, action, overlap = pair
            bonus = {
                "fill": policy.fill_bonus,
                "select": policy.select_bonus,
                "click": policy.click_bonus,
            }.get(action.get("kind"), 0.0)
            overlap_term = (overlap ** policy.overlap_exponent) * policy.goal_overlap_weight
            return overlap_term + bonus - index * policy.order_penalty

        def node_group(index, action):
            node = action.get("node")
            return ("n", node) if node is not None else ("~", index)

        ranked = sorted(regular, key=score, reverse=True)
        quotas = {"click": policy.click_quota, "fill": policy.fill_quota, "select": policy.select_quota}
        regular_budget = max(0, policy.model_action_limit - len(controls))
        counts = {kind: 0 for kind in quotas}
        node_counts: dict = {}
        selected = []
        used = set()
        for index, action, _overlap in ranked:
            kind = action["kind"]
            group = node_group(index, action)
            if (
                counts[kind] >= quotas[kind]
                or len(selected) >= regular_budget
                or node_counts.get(group, 0) >= policy.duplicate_node_cap
            ):
                continue
            selected.append((index, action))
            used.add(index)
            counts[kind] += 1
            node_counts[group] = node_counts.get(group, 0) + 1
        if len(selected) < regular_budget:
            for index, action, _overlap in ranked:
                if index in used:
                    continue
                group = node_group(index, action)
                if node_counts.get(group, 0) >= policy.duplicate_node_cap:
                    continue
                selected.append((index, action))
                used.add(index)
                node_counts[group] = node_counts.get(group, 0) + 1
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
            "est_tokens": 0.0,
            "est_latency_ms": 0.0,
            "predicted_change": 0.0,
            "prediction_samples": 0,
            "choice_predicted_change": 0.0,
            "choice_proposed_change": 0.0,
            "choice_uncertainty": 0.0,
            "choice_confident": 0,
            "choice_divergences": 0,
            "choice_proposal_uncertainty": 0.0,
            "choice_proposal_confident": 0,
            "experiment_proposals": [],
        }

    def _aggregate(self, summaries: list[dict], *, estimated: bool = False) -> ReplayMetrics:
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

        def score(token_total, latency_total):
            return (
                w.success * successes
                + w.verified * verified
                + w.page_progress * progress
                - w.latency_seconds * (latency_total / 1000)
                - w.action * actions
                - w.model_call * model_calls
                - w.thousand_tokens * (token_total / 1000)
                - w.offered_candidate * offered
                - w.stale_or_failure * failures
                - w.risk_event * risk
                - w.coverage_miss * misses
            )

        estimated_score = None
        est_tokens = est_latency = 0
        if estimated:
            est_tokens = sum(s["est_tokens"] for s in summaries)
            est_latency = sum(s["est_latency_ms"] for s in summaries)
            estimated_score = score(est_tokens, est_latency)
        return ReplayMetrics(
            score=score(tokens, latency),
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
            estimated_tokens=int(round(est_tokens)),
            estimated_latency_ms=int(round(est_latency)),
            estimated_score=estimated_score,
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
    """Deterministic, disjoint split that keeps each task family in one partition.

    A family is the explicit ``task_family`` recorded at run start, falling back
    to the goal-hash task key for traces without family metadata.
    """
    groups: dict[str, list[ReplayWorld]] = defaultdict(list)
    for world in worlds:
        groups[world.family_key or world.task_key].append(world)
    ordered = sorted(groups.items(), key=lambda item: (int(_stable_hash(item[0])[:16], 16), item[0]))
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
    """Deterministic, bidirectional, bounded candidate generator for offline dreaming.

    Every knob is perturbed in both directions within the ``ExplorationPolicy``
    envelopes, so the search can expand as well as contract the candidate space.
    The generator changes only whitelisted exploration knobs. It never emits
    code, and candidates identical in behavior to the base are dropped by
    ``behavior_digest`` rather than kept as distinct digests.
    """
    specs = [
        {"goal_overlap_weight": max(0.1, base.goal_overlap_weight * factor)}
        for factor in (0.25, 0.5, 0.75, 1.25, 1.5, 2.0)
    ]
    specs += [
        {field: max(-2.0, min(10.0, getattr(base, field) + delta))}
        for field in ("click_bonus", "fill_bonus", "select_bonus")
        for delta in (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0)
    ]
    specs += [
        {field: max(1, min(250, getattr(base, field) + delta))}
        for field in ("click_quota", "fill_quota", "select_quota")
        for delta in (-25, -10, 10, 25)
    ]
    specs += [
        {"model_action_limit": max(16, min(250, base.model_action_limit + delta))}
        for delta in (-32, -16, 16, 32)
    ]
    specs += [
        {"max_actions": max(4, min(120, base.max_actions + delta))}
        for delta in (-16, -8, 8, 16)
    ]
    specs += [
        {"no_progress_window": max(2, min(10, base.no_progress_window + delta))}
        for delta in (-1, 1)
    ]
    specs += [
        {"order_penalty": max(0.0, base.order_penalty * factor)}
        for factor in (0.2, 5.0)
    ]
    specs += [
        {"overlap_exponent": max(0.25, min(2.0, base.overlap_exponent * factor))}
        for factor in (0.7, 1.4)
    ]
    specs += [
        {"duplicate_node_cap": max(1, min(250, base.duplicate_node_cap + delta))}
        for delta in (-100, -50, -25, 25, 100)
    ]
    specs += [
        {"min_goal_overlap": max(0, min(5, base.min_goal_overlap + delta))}
        for delta in (-1, 1)
    ]
    candidates = []
    seen = {base.behavior_digest}
    for index, changes in enumerate(specs, 1):
        candidate = replace(base, name=f"{base.name}-dream-{index}", version=base.version + 1, **changes)
        if candidate.behavior_digest in seen:
            continue
        seen.add(candidate.behavior_digest)
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
    cost_model_digest: str | None = None
    outcome_model_digest: str | None = None
    choice_model_digest: str | None = None
    trial_model_digest: str | None = None
    experiment_proposals: tuple[dict, ...] = ()
    trials_digest: str | None = None
    trial_estimates: dict | None = None

    def to_dict(self):
        return {
            "schema": SCHEMA_VERSION,
            "selected": self.selected.to_dict(),
            "selected_digest": self.selected.digest,
            "selected_behavior_digest": self.selected.behavior_digest,
            "baseline": self.baseline.to_dict(),
            "baseline_digest": self.baseline.digest,
            "baseline_behavior_digest": self.baseline.behavior_digest,
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
            "cost_model_digest": self.cost_model_digest,
            "outcome_model_digest": self.outcome_model_digest,
            "choice_model_digest": self.choice_model_digest,
            # Digest of the causal prior (TrialChoiceModel) that generated any
            # trial-backed proposals — distinct from the observational prior.
            "trial_model_digest": self.trial_model_digest,
            "experiment_proposals": list(self.experiment_proposals),
            # Randomized-trial estimates are annotation, never gate input:
            # the ITT arm estimates exist so an operator (or a future learned
            # layer) can see what assigned experiments measured.
            "trials_digest": self.trials_digest,
            "trial_estimates": self.trial_estimates,
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
        cost_model=None,
        outcome_model=None,
        choice_model=None,
        trial_model=None,
        trials=None,
        scheduler=None,
        plan_signer=None,
    ) -> DreamReport:
        worlds = list(worlds)
        splits = split_worlds(worlds)
        pool_digest = replay_pool_digest(worlds)
        split_digest = split_manifest_digest(splits)
        tcb_versions = tuple(sorted({world.tcb_version for world in worlds}))
        if len(tcb_versions) > 1:
            raise ValueError(
                "Replay worlds mix DREAM TCB versions; improve each evidence generation separately"
            )
        split_sizes = {k: len(v) for k, v in splits.items()}
        train = splits["train"]
        if not train:
            empty = ReplaySimulator([], self.weights).evaluate(base)
            decision = PromotionDecision(False, "no replay worlds", empty.metrics, empty.metrics)
            return DreamReport(
                base,
                base,
                decision,
                (),
                split_sizes,
                pool_digest,
                split_digest,
                evidence_head_hash,
                tcb_versions,
            )

        simulators = {
            name: (ReplaySimulator(items, self.weights) if items else None)
            for name, items in splits.items()
        }
        base_results = {
            name: (
                sim.evaluate(
                    base, cost_model=cost_model, outcome_model=outcome_model,
                    choice_model=choice_model,
                    trial_model=trial_model,
                    # Experiment proposals come from the baseline evaluation:
                    # they are "what the prior would test differently under the
                    # currently deployed behavior", not candidate-policy notes.
                    collect_proposals=(name == "train"),
                ) if sim else None
            )
            for name, sim in simulators.items()
        }

        def robust_estimated_gain(candidate_results):
            deltas = []
            for name in ("train", "validation", "holdout"):
                base_result, candidate_result = base_results.get(name), candidate_results.get(name)
                if (
                    base_result is None
                    or candidate_result is None
                    or not base_result.metrics.worlds
                    or base_result.metrics.estimated_score is None
                    or candidate_result.metrics.estimated_score is None
                ):
                    continue
                deltas.append(
                    candidate_result.metrics.estimated_score / candidate_result.metrics.worlds
                    - base_result.metrics.estimated_score / base_result.metrics.worlds
                )
            return min(deltas) if deltas else None

        evaluated = []
        passing: list[tuple[float, float, float, ExplorationPolicy, PromotionDecision]] = []
        for policy in mutate_policies(base):
            candidate_results = {
                name: (
                    sim.evaluate(
                        policy, cost_model=cost_model, outcome_model=outcome_model,
                        choice_model=choice_model, trial_model=trial_model,
                    ) if sim else None
                )
                for name, sim in simulators.items()
            }
            if policy.behavior_digest == base.behavior_digest:
                decision = PromotionDecision(
                    False,
                    "baseline behavior",
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
            estimated_gain = robust_estimated_gain(candidate_results)
            entry = {
                "policy": policy.to_dict(),
                "digest": policy.digest,
                "behavior_digest": policy.behavior_digest,
                "train": asdict(candidate_results["train"].metrics),
                "validation": (
                    asdict(candidate_results["validation"].metrics) if candidate_results["validation"] else None
                ),
                "holdout": asdict(candidate_results["holdout"].metrics) if candidate_results["holdout"] else None,
                "replay_approved": decision.approved,
                "reason": decision.reason,
                "robust_gain_per_world": robust_gain,
                "average_gain_per_world": average_gain,
            }
            if estimated_gain is not None:
                entry["estimated_gain_per_world"] = estimated_gain
            if outcome_model is not None and outcome_model.samples:
                entry["predicted_page_changes_per_world"] = fmean(
                    s["predicted_change"] for s in candidate_results["train"].per_world
                )
            if choice_model is not None and choice_model.samples:
                # Advisory annotation only: the choice prior evaluated under the
                # candidate's own offered ordering, plus how often its
                # counterfactual proposal differs from the action the recorded
                # policy actually took. None of this enters a gate. The two
                # subjects are reported separately: the *selected* action's
                # prior uncertainty/confidence and the *proposal's* — mixing
                # them would make a confident divergent proposal look like a
                # confident recorded action or vice versa.
                pw = candidate_results["train"].per_world
                steps = sum(s["actions"] for s in pw)
                if steps:
                    selected_uncertainty = fmean(
                        s["choice_uncertainty"] / max(1, s["actions"]) for s in pw
                    )
                    selected_confident = sum(s["choice_confident"] for s in pw) / steps
                    entry["choice_model"] = {
                        "predicted_progress_per_step": fmean(
                            s["choice_predicted_change"] / max(1, s["actions"]) for s in pw
                        ),
                        "proposed_progress_per_step": fmean(
                            s["choice_proposed_change"] / max(1, s["actions"]) for s in pw
                        ),
                        "divergence_rate": sum(s["choice_divergences"] for s in pw) / steps,
                        "selected_mean_uncertainty": selected_uncertainty,
                        "selected_confident_fraction": selected_confident,
                        "proposal_mean_uncertainty": fmean(
                            s["choice_proposal_uncertainty"] / max(1, s["actions"]) for s in pw
                        ),
                        "proposal_confident_fraction": sum(
                            s["choice_proposal_confident"] for s in pw
                        ) / steps,
                        # Pre-split names retained as aliases; their subject was
                        # always the recorded (selected) action.
                        "mean_uncertainty": selected_uncertainty,
                        "confident_fraction": selected_confident,
                    }
            evaluated.append(entry)
            if decision.approved:
                passing.append((robust_gain, average_gain, estimated_gain or float("-inf"), policy, decision))

        if passing:
            passing.sort(
                key=lambda item: (item[0], item[1], item[2], item[3].digest), reverse=True
            )
            _robust, _average, _estimated, selected, promotion = passing[0]
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

        experiment_proposals = ()
        if (choice_model is not None or trial_model is not None) and base_results[
            "train"
        ] is not None:
            collected = [
                proposal
                for summary in base_results["train"].per_world
                for proposal in summary.get("experiment_proposals", ())
            ]
            # The scheduler decides which *unresolved* hypothesis is worth a
            # real browser experiment (expected information gain × practical
            # importance ÷ evidence coverage); settled hypotheses are dropped
            # entirely. Without a scheduler the historical expected-delta
            # order is the fallback. A proposal is a hypothesis to test under
            # the real authority plane, not a finding.
            if scheduler is not None:
                collected = scheduler.rank(collected, estimates=trials)
            else:
                collected.sort(key=lambda p: (-p["expected_delta"], p["world"], p["step"]))
            stamped = []
            for proposal in collected[:24]:
                # A stamped plan is an immutable ExperimentPlan: the digest
                # binds the hypothesis *and* the bindings the agent
                # revalidates live (task, family, state, model choice, offered
                # catalogue, policy behavior, originating model, world pool
                # and evidence head). A plan whose bindings no longer hold is
                # stale and must be discarded, never reinterpreted. With a
                # signer configured the plan additionally carries a
                # domain-separated Ed25519 authority block — digest integrity
                # alone is not provenance.
                item = {
                    k: v
                    for k, v in {"schema": EXPERIMENT_PLAN_SCHEMA, **proposal}.items()
                    if k not in {"estimate", "scheduler"}
                }
                item["world_pool_digest"] = pool_digest
                item["evidence_head_hash"] = evidence_head_hash
                item["digest"] = experiment_plan_digest(item)
                authority = experiment_plan_signature(item, plan_signer)
                if authority:
                    item["authority"] = authority
                stamped.append(item)
            experiment_proposals = tuple(stamped)

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
            experiment_proposals=experiment_proposals,
            cost_model_digest=(cost_model.digest if cost_model is not None and cost_model.samples else None),
            outcome_model_digest=(
                outcome_model.digest if outcome_model is not None and outcome_model.samples else None
            ),
            choice_model_digest=(
                choice_model.digest if choice_model is not None and choice_model.samples else None
            ),
            trial_model_digest=(
                trial_model.digest
                if trial_model is not None and getattr(trial_model, "trials", None) is not None
                else None
            ),
            # Randomized assignments are the only causal evidence the store
            # holds — surface the fitted estimates so they annotate the report
            # instead of staying a diagnostic object nobody reads.
            trials_digest=trials.digest if trials is not None else None,
            trial_estimates=trials.estimate() if trials is not None else None,
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


@dataclass(frozen=True)
class HealthDecision:
    healthy: bool
    reason: str
    reference: CanaryMetrics
    observed: CanaryMetrics
    sufficient: bool = True


class HealthGate:
    """Detect post-promotion drift from hash-verified active-policy traces."""

    def __init__(self, *, min_tasks: int = 20, max_success_regression: float = 0.10, max_extra_risk_rate: float = 0.0):
        self.min_tasks = min_tasks
        self.max_success_regression = max_success_regression
        self.max_extra_risk_rate = max_extra_risk_rate

    def assess(self, reference: CanaryMetrics, observed: CanaryMetrics) -> HealthDecision:
        if observed.tasks < self.min_tasks:
            return HealthDecision(
                True, "insufficient recent tasks for drift decision", reference, observed, sufficient=False
            )
        if reference.tasks and observed.success_rate + self.max_success_regression < reference.success_rate:
            return HealthDecision(False, "active policy verified-success drift", reference, observed)
        if observed.risk_rate > reference.risk_rate + self.max_extra_risk_rate:
            return HealthDecision(False, "active policy risk-rate drift", reference, observed)
        return HealthDecision(True, "active policy health gates passed", reference, observed)


ATTESTATION_DOMAIN = "jev-dream/promotion-attestation/v1"  # content prefix (inside the digest)
# ATTESTATION_SIG_DOMAIN is the signature frame — separate from evidence events.


class PolicyRegistry:
    """Atomic registry with lineage binding, evidence binding, suspension, and rollback.

    Only bounded ``ExplorationPolicy`` data can be promoted. Staging is bound to
    the active parent policy and replay evidence digests, preventing a report
    generated against stale policy state from being activated later.

    When a ``signer`` is configured (``JEV_PROMOTION_SIGNING_KEY``, falling back
    to the evidence key), every promotion attaches a signed ``attestation``
    binding the candidate digest, parent digest, replay/canary evidence digests,
    and the registry revision at promotion time. When ``verify_keys`` is
    configured (``JEV_PROMOTION_VERIFY_KEYS`` or the evidence verify keys), any
    ``active`` record without a valid, correctly bound attestation fails closed
    on load — the registry becomes a view over signed authority rather than
    the authority itself.
    """

    def __init__(self, path: str | os.PathLike, *, signer=None, verify_keys=None, anchor_path=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.anchor_path = Path(anchor_path) if anchor_path else None
        self._lock = threading.RLock()
        self.signer = signer if signer is not None else (
            EvidenceSigner.from_env(PROMOTION_SIGNING_KEY_ENV) or EvidenceSigner.from_env()
        )
        keys = {k.strip() for k in (verify_keys or []) if k and k.strip()}
        keys.update(verify_keys_from_env(PROMOTION_VERIFY_KEYS_ENV))
        # Evidence verification keys are a valid promotion authority too.
        keys.update(verify_keys_from_env())
        self.verify_keys = keys

    # Content hash domain for the registry head anchor — distinct from both
    # the registry state domain (which signs the payload) and the evidence
    # chain-head anchor domain (which checkpoints the event log).
    _REGISTRY_ANCHOR_CONTENT_DOMAIN = "jev-dream/registry-head-anchor/v1"

    def _read_anchor(self) -> dict | None:
        if self.anchor_path is None or not self.anchor_path.exists():
            return None
        try:
            anchor = json.loads(self.anchor_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("Invalid registry head anchor file") from exc
        if not isinstance(anchor, dict) or anchor.get("kind") != "registry_head_anchor":
            raise ValueError("Invalid registry head anchor file")
        return anchor

    def _write_anchor_unlocked(self, head: str | None, revision: int) -> dict | None:
        if self.anchor_path is None:
            return None
        material = {
            "kind": "registry_head_anchor",
            "head": head,
            "revision": revision,
            "signed_at_ms": int(time.time() * 1000),
        }
        digest = _stable_hash(
            f"{self._REGISTRY_ANCHOR_CONTENT_DOMAIN}\n"
            + json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        )
        anchor = {**material, "digest": digest}
        if self.signer is not None:
            anchor["key_id"] = self.signer.key_id
            anchor["signature"] = self.signer.sign_hex(digest, domain=REGISTRY_ANCHOR_DOMAIN)
        self.anchor_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.anchor_path.with_suffix(self.anchor_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(anchor, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.anchor_path)
        _fsync_dir(self.anchor_path.parent)
        return anchor

    def _check_anchor_unlocked(self, state_digest: str, revision: int) -> dict | None:
        """Anchor consistency when ``anchor_path`` is configured.

        The chained signed state head proves authenticity of what was written,
        but a wholesale restore of an older complete signed file replays
        cleanly — the external anchor is the freshness check that detects it.
        Fails closed on a missing anchor, a bad digest, signature problems, or
        a head/revision that disagrees with the verified payload.
        """
        if self.anchor_path is None:
            return None
        anchor = self._read_anchor()
        if anchor is None:
            raise ValueError("Registry head anchor missing while the registry holds state")
        material = {k: v for k, v in anchor.items() if k not in {"digest", "signature", "key_id"}}
        digest = _stable_hash(
            f"{self._REGISTRY_ANCHOR_CONTENT_DOMAIN}\n"
            + json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        )
        if anchor.get("digest") != digest:
            raise ValueError("Registry head anchor digest mismatch")
        signature = anchor.get("signature")
        if signature is not None:
            trusted = set(self.verify_keys)
            if self.signer is not None:
                trusted.add(self.signer.key_id)
            if not trusted:
                raise ValueError("Signed registry anchor but no verification key is configured")
            if anchor.get("key_id") not in trusted:
                raise ValueError("Registry anchor signed by an unexpected key")
            if not verify_signature(
                str(anchor["key_id"]), digest, str(signature), domain=REGISTRY_ANCHOR_DOMAIN
            ):
                raise ValueError("Registry anchor signature verification failure")
        elif self.verify_keys or self.signer is not None:
            raise ValueError("Unsigned registry head anchor while trust keys are configured")
        if anchor.get("head") != state_digest or int(anchor.get("revision", -1)) != revision:
            raise ValueError(
                "Registry head anchor disagrees with the registry: an older "
                "snapshot was restored or the anchor is stale — resolve with "
                "an explicit reanchor()"
            )
        return anchor

    def reanchor(self) -> dict:
        """Write a fresh anchor for the current verified registry head.

        Operator-level resolution after a confirmed anchor gap — a restore, a
        lost anchor file, or a deliberate reset. The registry's own chained
        state is still verified by the load; only the anchor check is skipped,
        because the anchor is the thing being repaired. Never implicit: normal
        reads and writes checkpoint the anchor but never re-anchor a gap.
        """
        with self._lock, _file_lock(self.lock_path):
            payload = self._load_unlocked(check_anchor=False)
            anchor = self._write_anchor_unlocked(
                payload.get("state_digest"), int(payload.get("revision", 0))
            )
            if anchor is None:
                raise ValueError("No anchor_path configured for this registry")
            return anchor

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

    @staticmethod
    def _attestation_digest(material: dict) -> str:
        canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return _stable_hash(f"{ATTESTATION_DOMAIN}\n{canonical}")

    def _sign_attestation(self, material: dict) -> dict:
        digest = self._attestation_digest(material)
        return {**material, "digest": digest, "key_id": self.signer.key_id,
                "signature": self.signer.sign_hex(digest, domain=ATTESTATION_SIG_DOMAIN)}

    # Attestation kind → record fields the signed material must equal. The
    # signature authenticates the *statement*; these bindings authenticate
    # that the statement describes this exact record — otherwise a signed
    # promotion could ride along on a record whose parent, canary evidence,
    # promotion timestamp, or revision was edited after signing.
    _ATTESTATION_BINDINGS = {
        "promotion_attestation": {
            "candidate_digest": ("digest",),
            "parent_digest": ("parent_digest",),
            "replay_report_hash": ("replay_report_hash",),
            "world_pool_digest": ("world_pool_digest",),
            "split_manifest_digest": ("split_manifest_digest",),
            "evidence_head_hash": ("evidence_head_hash",),
            "evidence_digest": ("canary", "evidence_digest"),
            "event_head_hash": ("canary", "event_head_hash"),
            "promoted_at_ms": ("promoted_at_ms",),
            "registry_revision": ("promotion_revision",),
        },
        # Registry-state transition attestations chain to the state head they
        # replace (binding the head of the write that contains them would be
        # circular). Fields that later transitions legitimately mutate — the
        # suspended flag, suspension timestamps — are deliberately not bound;
        # the signed chained state head covers the payload as written.
        "suspend_attestation": {
            "candidate_digest": ("digest",),
            "prev_state_head": ("suspend_prev_state_head",),
            "registry_revision": ("suspend_revision",),
        },
        "resume_attestation": {
            "candidate_digest": ("digest",),
            "prev_state_head": ("resume_prev_state_head",),
            "registry_revision": ("resume_revision",),
        },
        "rollback_attestation": {
            "candidate_digest": ("digest",),
            "rollback_from_digest": ("rollback_from_digest",),
            "promoted_at_ms": ("rollback_at_ms",),
            "registry_revision": ("rollback_revision",),
            "prev_state_head": ("rollback_prev_state_head",),
        },
    }

    def _check_attestation_binding(self, att: dict, record: dict, label: str):
        kind = att.get("kind")
        bindings = self._ATTESTATION_BINDINGS.get(kind, {"candidate_digest": ("digest",)})
        mismatched = []
        for field, path in bindings.items():
            if field not in att:
                continue
            current = record
            for key in path:
                current = current.get(key) if isinstance(current, dict) else None
            if current != att[field]:
                mismatched.append(field)
        if kind == "promotion_attestation":
            if "behavior_digest" in att:
                behavior = None
                policy = record.get("policy")
                if isinstance(policy, dict):
                    try:
                        behavior = ExplorationPolicy.from_dict(policy).behavior_digest
                    except (ValueError, TypeError):
                        behavior = None
                if att["behavior_digest"] != behavior:
                    mismatched.append("behavior_digest")
            # The whole canary block — including the reference metrics that
            # health_from_store() later trusts — is bound by a content digest
            # rather than field-by-field paths.
            if "canary_digest" in att:
                canary = record.get("canary")
                digest = (
                    _stable_hash(json.dumps(canary, sort_keys=True, separators=(",", ":")))
                    if isinstance(canary, dict) else None
                )
                if att["canary_digest"] != digest:
                    mismatched.append("canary_digest")
        if mismatched:
            raise ValueError(
                f"Policy registry {label} attestation is not bound to the record: "
                f"{sorted(mismatched)}"
            )

    def _check_attestation(self, att: dict, record: dict, label: str):
        material = {k: v for k, v in att.items() if k not in {"digest", "signature", "key_id"}}
        digest = self._attestation_digest(material)
        if att.get("digest") != digest:
            raise ValueError(f"Policy registry {label} attestation digest mismatch")
        if att.get("key_id") not in self.verify_keys:
            raise ValueError(f"Policy registry {label} attestation signed by an unexpected key")
        if not verify_signature(
            att["key_id"], digest, str(att.get("signature") or ""),
            domain=ATTESTATION_SIG_DOMAIN,
        ):
            raise ValueError(f"Policy registry {label} attestation signature invalid")
        self._check_attestation_binding(att, record, label)

    def _verify_attestation(self, record: dict | None, label: str):
        """Verify a record's promotion attestation when trust keys are configured."""
        if record is None or not self.verify_keys:
            return
        att = record.get("attestation")
        if not isinstance(att, dict):
            raise ValueError(f"Policy registry {label} record lacks a promotion attestation")
        self._check_attestation(att, record, label)
        for key in ("rollback_attestation", "suspend_attestation", "resume_attestation"):
            transition_att = record.get(key)
            if isinstance(transition_att, dict):
                self._check_attestation(transition_att, record, f"{label} {key}")

    def load(self) -> dict:
        with self._lock, _file_lock(self.lock_path):
            return self._load_unlocked()

    def _load_unlocked(self, *, check_anchor: bool = True) -> dict:
        if not self.path.exists():
            # A deleted registry under a live anchor is not an empty registry —
            # the anchor still points at a head the file must reach. Only an
            # explicitly anchored *empty* state (head=None) may read as empty.
            if check_anchor:
                anchor = self._read_anchor()
                if anchor is not None and anchor.get("head") is not None:
                    raise ValueError(
                        "Policy registry file missing while a head anchor exists"
                    )
            return self._empty()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema") not in SUPPORTED_SCHEMAS or payload.get("tcb_version") not in SUPPORTED_TCB_VERSIONS:
            raise ValueError("Unsupported policy registry schema/TCB version")
        # Chained registry state head (jev-dream/4): the digest covers the whole
        # written payload — revision, active, staged, history, prev_state_digest —
        # so post-write edits to any field (including mutable ones such as
        # ``suspended``) are caught, and under a signer the head cannot be
        # resealed without the key. A wholesale restore of an older signed file
        # still replays cleanly; defeating that needs an externally anchored
        # latest head, the same residual the evidence anchor carries.
        state_digest = payload.get("state_digest")
        stripped = (
            payload.get("schema") == SCHEMA_VERSION
            or payload.get("state_signature") is not None
            or payload.get("state_key_id") is not None
            or payload.get("prev_state_digest") is not None
        )
        if state_digest is None and (stripped or self.verify_keys or self.signer is not None):
            # Every jev-dream/4 write stamps a state head; its absence — or a
            # partial strip leaving orphan chain fields — is tampering, not a
            # legacy file. And under configured trust keys a headless file can
            # never prove lineage: downgrading the schema field itself to
            # jev-dream/3 must not launder a stripped signed registry into a
            # "legacy" one whose mutable fields (``suspended``) attestations
            # deliberately leave unbound. Genuine pre-/4 files carry no
            # attestations and were already unloadable under keys; re-anchor
            # them by loading without keys and letting a trusted write restamp.
            raise ValueError("Policy registry state head is missing or stripped")
        if state_digest is not None:
            if state_digest != self._state_digest(payload):
                raise ValueError("Policy registry state digest mismatch")
            signature = payload.get("state_signature")
            if signature is not None:
                trusted = set(self.verify_keys)
                if self.signer is not None:
                    trusted.add(self.signer.key_id)
                if not trusted:
                    raise ValueError(
                        "Signed policy registry state but no verification key is configured"
                    )
                if payload.get("state_key_id") not in trusted:
                    raise ValueError("Policy registry state signed by an unexpected key")
                if not verify_signature(
                    str(payload["state_key_id"]), str(state_digest), str(signature),
                    domain=REGISTRY_STATE_DOMAIN,
                ):
                    raise ValueError("Policy registry state signature verification failure")
            elif self.verify_keys or self.signer is not None:
                # A configured signer produces signed writes only, so an
                # unsigned head means the signature was stripped — not that
                # the writer lacked a key.
                raise ValueError(
                    "Unsigned policy registry state while a signing/verification key is configured"
                )
        if check_anchor and state_digest is not None:
            # Freshness, not authenticity: the signed head above proves this
            # payload was written by the key holder, but a wholesale restore
            # of an older complete signed file replays cleanly. The external
            # anchor is the checkpoint that detects it.
            self._check_anchor_unlocked(state_digest, int(payload.get("revision", 0)))
        payload.setdefault("revision", 0)
        payload.setdefault("active", None)
        payload.setdefault("staged", None)
        payload.setdefault("history", [])
        self._validate_record(payload.get("active"), "active")
        self._validate_record(payload.get("staged"), "staged")
        for index, record in enumerate(payload.get("history", [])):
            self._validate_record(record, f"history[{index}]")
        # Promotion attestations: the active record must prove signed
        # qualification when verification keys are configured; staged and
        # history records are verified whenever they carry one.
        if self.verify_keys:
            self._verify_attestation(payload.get("active"), "active")
        for label, record in (
            [("staged", payload.get("staged"))]
            + [(f"history[{i}]", r) for i, r in enumerate(payload.get("history", []))]
        ):
            if record and record.get("attestation") is not None:
                self._verify_attestation(record, label)
        # In-memory migration; the next mutation writes the current schema/TCB.
        payload["schema"] = SCHEMA_VERSION
        payload["tcb_version"] = TCB_VERSION
        return payload

    @staticmethod
    def _state_digest(payload: dict) -> str:
        material = {
            k: v for k, v in payload.items()
            if k not in {"state_digest", "state_key_id", "state_signature"}
        }
        canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return _stable_hash(f"{REGISTRY_STATE_DOMAIN.decode()}\n{canonical}")

    def _write(self, payload: dict):
        payload = dict(payload)
        payload["schema"] = SCHEMA_VERSION
        payload["tcb_version"] = TCB_VERSION
        payload["revision"] = int(payload.get("revision", 0)) + 1
        # Chain each write to the head it replaces so history cannot be
        # reordered, truncated, or rewritten without invalidating the digest.
        payload["prev_state_digest"] = payload.get("state_digest") or "0" * 64
        payload.pop("state_signature", None)
        payload.pop("state_key_id", None)
        payload["state_digest"] = self._state_digest(payload)
        if self.signer is not None:
            payload["state_key_id"] = self.signer.key_id
            payload["state_signature"] = self.signer.sign_hex(
                payload["state_digest"], domain=REGISTRY_STATE_DOMAIN
            )
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)
        # fsync the directory so the rename itself survives a crash; the file
        # fsync alone does not make the new directory entry durable.
        _fsync_dir(self.path.parent)
        # Checkpoint the new head into the external anchor *after* the payload
        # is durable. If this step fails, the anchor stays behind and every
        # subsequent read fails closed until an explicit reanchor().
        if self.anchor_path is not None:
            self._write_anchor_unlocked(
                payload.get("state_digest"), int(payload.get("revision", 0))
            )

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
            if evidence.candidate_digest != staged["digest"]:
                raise ValueError("Canary evidence is not bound to the staged policy digest")
            if evidence.baseline_digest != staged.get("parent_digest"):
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
            promoted_at = int(time.time() * 1000)
            # The revision _write() will assign this promotion.
            promotion_revision = int(payload.get("revision", 0)) + 1
            canary = {
                "baseline": asdict(evidence.baseline),
                "candidate": asdict(evidence.candidate),
                "paired_task_families": evidence.paired_task_families,
                "candidate_wins": evidence.candidate_wins,
                "baseline_wins": evidence.baseline_wins,
                "ties": evidence.ties,
                "event_head_hash": evidence.event_head_hash,
                "evidence_digest": evidence.evidence_digest,
                "reason": decision.reason,
            }
            payload["active"] = {
                **staged,
                "promoted_at_ms": promoted_at,
                "promotion_revision": promotion_revision,
                "suspended": False,
                "canary": canary,
            }
            if self.signer is not None:
                # The promotion itself is signed authority: the attestation
                # binds candidate, parent, the exact evidence digests that
                # qualified it, the whole canary block (so the health-check
                # reference metrics cannot be edited under a valid signature),
                # and the registry revision _write will assign.
                payload["active"]["attestation"] = self._sign_attestation({
                    "kind": "promotion_attestation",
                    "candidate_digest": staged["digest"],
                    "behavior_digest": ExplorationPolicy.from_dict(staged["policy"]).behavior_digest,
                    "parent_digest": staged.get("parent_digest"),
                    "replay_report_hash": staged.get("replay_report_hash"),
                    "world_pool_digest": staged.get("world_pool_digest"),
                    "split_manifest_digest": staged.get("split_manifest_digest"),
                    "evidence_head_hash": staged.get("evidence_head_hash"),
                    "evidence_digest": evidence.evidence_digest,
                    "event_head_hash": evidence.event_head_hash,
                    "canary_digest": _stable_hash(
                        json.dumps(canary, sort_keys=True, separators=(",", ":"))
                    ),
                    "registry_revision": promotion_revision,
                    "promoted_at_ms": promoted_at,
                })
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
        # Compatibility escape hatch, feature-gated twice: the caller must
        # pass the flag AND the environment must explicitly opt in, so no code
        # path can reach unbound promotion silently. The gate fails closed —
        # absent or empty JEV_ALLOW_UNBOUND_METRICS rejects the call.
        if os.environ.get("JEV_ALLOW_UNBOUND_METRICS") != "1":
            raise ValueError(
                "Unbound canary metrics require JEV_ALLOW_UNBOUND_METRICS=1 in the "
                "environment; normal activation is promote_from_store() with "
                "bound paired evidence."
            )
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
            # Unbound evidence contains no outcome pairs, so a sign test could
            # never pass here; the compatibility path deliberately skips it —
            # the significance floor applies to bound paired evidence only.
            max_pair_sign_p=None,
        )
        return self._promote_bound(evidence, permissive)

    def promote_from_store(
        self,
        store: ExperienceStore,
        *,
        baseline_policy_digest: str | None = None,
        baseline_since_ms: int | None = None,
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
            baseline_since_ms=baseline_since_ms,
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
            if self.signer is not None:
                # The transition is a signed statement chained to the state
                # head it replaces; the payload's signed state head then covers
                # the whole post-transition registry.
                active["suspend_revision"] = int(payload.get("revision", 0)) + 1
                active["suspend_prev_state_head"] = payload.get("state_digest") or "0" * 64
                active["suspend_attestation"] = self._sign_attestation({
                    "kind": "suspend_attestation",
                    "candidate_digest": active["digest"],
                    "registry_revision": active["suspend_revision"],
                    "prev_state_head": active["suspend_prev_state_head"],
                    "suspended_at_ms": active["suspended_at_ms"],
                    "reason": active["suspension_reason"],
                })
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
            if self.signer is not None:
                active["resume_revision"] = int(payload.get("revision", 0)) + 1
                active["resume_prev_state_head"] = payload.get("state_digest") or "0" * 64
                active["resume_attestation"] = self._sign_attestation({
                    "kind": "resume_attestation",
                    "candidate_digest": active["digest"],
                    "registry_revision": active["resume_revision"],
                    "prev_state_head": active["resume_prev_state_head"],
                    "resumed_at_ms": int(time.time() * 1000),
                })
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
            # A rolled-back record re-enters authority: when verification keys
            # are configured its existing attestations must hold against the
            # record as written — verify before stamping the new transition
            # fields, since a prior rollback attestation binds the prior
            # rollback_revision.
            if self.verify_keys:
                self._verify_attestation(target, "rollback")
            target["rollback_at_ms"] = int(time.time() * 1000)
            target["rollback_from_digest"] = current.get("digest") if current else None
            # The revision _write() will assign this rollback; the record's
            # promotion_revision stays untouched so its promotion attestation
            # still verifies against the original promotion.
            target["rollback_revision"] = int(payload.get("revision", 0)) + 1
            target["rollback_prev_state_head"] = payload.get("state_digest") or "0" * 64
            if self.signer is not None:
                # Sign the rollback transition itself so authority restoration
                # is attested rather than only the original promotion, chained
                # to the registry state head it replaces.
                target["rollback_attestation"] = self._sign_attestation({
                    "kind": "rollback_attestation",
                    "candidate_digest": target["digest"],
                    "rollback_from_digest": target["rollback_from_digest"],
                    "registry_revision": target["rollback_revision"],
                    "prev_state_head": target["rollback_prev_state_head"],
                    "promoted_at_ms": target["rollback_at_ms"],
                })
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
