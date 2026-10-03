"""_learning.common — extracted from jev_ultrafast.dreamlearn."""

from __future__ import annotations

__all__ = [
    '_stable_hash',
]


def _stable_hash(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()
