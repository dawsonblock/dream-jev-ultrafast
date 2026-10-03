"""_dream.registry — extracted from jev_ultrafast.dream."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from ..privacy import unbound_metrics_permitted
from ..signing import ATTESTATION_DOMAIN as ATTESTATION_SIG_DOMAIN
from ..signing import (
    PROMOTION_SIGNING_KEY_ENV,
    PROMOTION_VERIFY_KEYS_ENV,
    REGISTRY_ANCHOR_DOMAIN,
    REGISTRY_STATE_DOMAIN,
    EvidenceSigner,
    verify_keys_from_env,
    verify_signature,
)
from .canary import CanaryEvidence, CanaryGate, CanaryMetrics
from .common import (
    SCHEMA_VERSION,
    SUPPORTED_SCHEMAS,
    SUPPORTED_TCB_VERSIONS,
    TCB_VERSION,
    _file_lock,
    _fsync_dir,
    _stable_hash,
)
from .health import HealthGate
from .policy import ExplorationPolicy

if TYPE_CHECKING:
    from .evidence import ExperienceStore
    from .health import HealthDecision
    from .improver import DreamReport

__all__ = [
    'ATTESTATION_DOMAIN',
    'POLICY_STATES',
    'POLICY_TRANSITIONS',
    'PolicyRegistry',
]


ATTESTATION_DOMAIN = "jev-dream/promotion-attestation/v1"  # content prefix (inside the digest)

# The explicit promotion lifecycle (Phase 20). Every registry record
# carries a declared ``status``; lifecycle moves walk the transition
# table below rather than implicitly flipping flags, and a record whose
# declared status contradicts its slot or flags fails closed at load.
#
#   staged     — replay-approved candidate awaiting canary qualification
#   active     — the policy currently holding authority
#   suspended  — active policy with authority withdrawn (health/violation)
#   retired    — a record in history; authority it once held is released
POLICY_STATES = ("staged", "active", "suspended", "retired")

# The legal edges of the lifecycle. Promotion moves staged → active and
# retires the incumbent (active or suspended); suspension and resume
# shuttle active ↔ suspended; rollback moves retired → active and
# retires whatever held the active slot. The two self-loops are the
# idempotent edges: re-suspending a suspended policy or resuming an
# active one is a race-safe no-op (a monitor re-checking a suspended
# policy must not crash), never a state change. Every other edge is
# illegal and fails closed — there is no path that mints authority
# without the staged → evidence-bound → active sequence.
POLICY_TRANSITIONS = {
    "staged": ("active",),
    "active": ("active", "suspended", "retired"),
    "suspended": ("active", "suspended", "retired"),
    "retired": ("active",),
}


def _assert_transition(current: str, target: str, label: str):
    """Enforce the declared lifecycle: an edge outside
    ``POLICY_TRANSITIONS`` is a corruption/bug — never a silent move."""
    if target not in POLICY_TRANSITIONS.get(current, ()):
        raise ValueError(
            f"Illegal policy lifecycle transition {current} -> {target} "
            f"for {label} record"
        )


def _slot_status(record: dict, slot: str) -> str:
    """Derive a record's status from the slot it occupies and its flags."""
    if slot == "staged":
        return "staged"
    if slot == "history":
        return "retired"
    return "suspended" if record.get("suspended") else "active"


def _record_status(record: dict, slot: str) -> str:
    """The record's canonical status, checked against its slot.

    Records written before the field existed derive status from slot +
    ``suspended`` (legacy migration; the next write persists it). A
    declared status that is unknown, or that contradicts the slot's
    derived status, is corruption — fail closed rather than let a forged
    field repaint a record's authority."""
    derived = _slot_status(record, slot)
    declared = record.get("status")
    if declared is None:
        record["status"] = derived
        return derived
    if declared not in POLICY_STATES or declared != derived:
        raise ValueError(
            f"Policy registry {slot} record declares status "
            f"{declared!r}, inconsistent with {derived!r}"
        )
    return declared


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
        # Lifecycle state machine (Phase 20): every record's declared
        # status must agree with the slot it occupies — a contradictory
        # field is corruption, a missing one is migrated from the slot.
        for slot, record in (
            ("active", payload.get("active")),
            ("staged", payload.get("staged")),
        ) + tuple(
            ("history", r) for r in payload.get("history", [])
        ):
            if record:
                _record_status(record, slot)
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
            "status": "staged",
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
            _assert_transition("staged", "active", "staged")
            previous = payload.get("active")
            if previous:
                _assert_transition(
                    _record_status(previous, "active"), "retired", "previous"
                )
                payload["history"].append({
                    **previous,
                    "status": "retired",
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
                "status": "active",
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
        # absent or empty JEV_ALLOW_UNBOUND_METRICS rejects the call, and
        # JEV_SECURITY_PROFILE=strict refuses the hatch outright.
        if not unbound_metrics_permitted():
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
            _assert_transition(
                _record_status(active, "active"), "suspended", "active"
            )
            active["status"] = "suspended"
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
            _assert_transition(
                _record_status(active, "active"), "active", "active"
            )
            active["status"] = "active"
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
                _assert_transition(
                    _record_status(current, "active"), "retired", "active"
                )
                history.append({
                    **current,
                    "status": "retired",
                    "deactivated_at_ms": int(time.time() * 1000),
                    "deactivation_reason": "rollback",
                })
            target = {k: v for k, v in target.items() if k not in {"deactivated_at_ms", "deactivation_reason"}}
            _assert_transition(
                _record_status(target, "history"), "active", "rollback target"
            )
            target["status"] = "active"
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
        # suspended -> suspended is not a legal transition: a policy whose
        # authority is already withdrawn reports the unhealthy decision but
        # cannot be suspended a second time.
        if not decision.healthy and suspend_on_fail and not active.get("suspended"):
            self.suspend(decision.reason)
        return decision
