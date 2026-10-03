"""_dream.common — extracted from jev_ultrafast.dream."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Iterable

__all__ = [
    'ACTION_KINDS',
    'SCHEMA_VERSION',
    'SUPPORTED_SCHEMAS',
    'SUPPORTED_TCB_VERSIONS',
    'TCB_VERSION',
    '_file_lock',
    '_fsync_dir',
    '_stable_hash',
    'candidate_catalog_digest',
    'new_run_id',
    'summarize_usage',
    'task_key',
]


SCHEMA_VERSION = "jev-dream/4"


SUPPORTED_SCHEMAS = {"jev-dream/1", "jev-dream/2", "jev-dream/3", SCHEMA_VERSION}


TCB_VERSION = "jev-ultrafast-tcb/0.17"


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
    "jev-ultrafast-tcb/0.15",
    "jev-ultrafast-tcb/0.16",
    TCB_VERSION,
}


ACTION_KINDS = ("click", "fill", "select", "upload", "scroll", "key", "wait")


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
