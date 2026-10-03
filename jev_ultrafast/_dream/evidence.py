"""_dream.evidence — extracted from jev_ultrafast.dream."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from ..privacy import signed_evidence_required
from ..signing import ANCHOR_DOMAIN, EvidenceSigner, verify_keys_from_env, verify_signature
from .common import (
    SCHEMA_VERSION,
    SUPPORTED_SCHEMAS,
    SUPPORTED_TCB_VERSIONS,
    TCB_VERSION,
    _file_lock,
    _fsync_dir,
    _stable_hash,
)

__all__ = [
    'ExperienceStore',
]


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
        # require_signatures is the explicit knob; JEV_REQUIRE_SIGNED_EVIDENCE
        # opts in via the environment; JEV_SECURITY_PROFILE=strict implies it
        # — a hardened posture cannot consume unsigned evidence as provenance.
        self.require_signatures = bool(
            require_signatures or signed_evidence_required()
        )

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
