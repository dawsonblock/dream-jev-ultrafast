"""Ed25519 signatures for DREAM qualification evidence.

The hash chain detects corruption; signatures provide authenticity. A signed
``event_hash`` proves the evidence was written by whoever holds the key, so an
attacker cannot silently recompute the chain after rewriting the store.

Signatures are domain-separated: the signed message is
``SIGNING_DOMAIN + digest_hex`` rather than the raw digest, so a signature over
evidence can never be reinterpreted as a signature over anything else.

Deployment boundary: ``EvidenceSigner`` reads a seed from
``JEV_EVIDENCE_SIGNING_KEY`` by default. Deployments that must keep private key
material outside the agent process can inject any signer object exposing
``key_id`` and ``sign_hex`` (for example a call into a KMS/HSM-backed
service); such a signer must sign ``frame_digest(digest_hex)`` — the helper is
exported for exactly that purpose. ``ExperienceStore`` only requires the
interface.

Verification keys rotate through ``verify_keys`` sets: every signed event
carries its ``key_id`` (the public key in hex), and readers accept events
signed by any currently trusted key, so old segments stay verifiable after a
rotation. ``JEV_EVIDENCE_VERIFY_KEYS`` accepts a comma- or space-separated
list; ``JEV_EVIDENCE_VERIFY_KEY`` remains as the single-key form.
"""

from __future__ import annotations

import os

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

SIGNING_KEY_ENV = "JEV_EVIDENCE_SIGNING_KEY"
VERIFY_KEY_ENV = "JEV_EVIDENCE_VERIFY_KEY"
VERIFY_KEYS_ENV = "JEV_EVIDENCE_VERIFY_KEYS"

SIGNING_DOMAIN = b"jev-dream/evidence-event/v1:"


def frame_digest(digest_hex: str) -> bytes:
    """Domain-separation frame: the exact bytes an evidence signature covers."""
    return SIGNING_DOMAIN + digest_hex.encode("ascii")


def _public_key_hex(private_key: Ed25519PrivateKey) -> str:
    return private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()


def verify_keys_from_env() -> set[str]:
    """Accepted verification keys from ``JEV_EVIDENCE_VERIFY_KEY(S)``."""
    keys: set[str] = set()
    single = os.environ.get(VERIFY_KEY_ENV, "").strip()
    if single:
        keys.add(single)
    multi = os.environ.get(VERIFY_KEYS_ENV, "").replace(",", " ")
    keys.update(part for part in multi.split() if part)
    return keys


class EvidenceSigner:
    """Ed25519 signer over event digests. ``key_id`` is the public key in hex."""

    def __init__(self, seed: bytes):
        if len(seed) != 32:
            raise ValueError("Ed25519 seed must be 32 bytes")
        self._key = Ed25519PrivateKey.from_private_bytes(seed)
        self.key_id = _public_key_hex(self._key)

    @classmethod
    def from_hex(cls, seed_hex: str) -> "EvidenceSigner":
        return cls(bytes.fromhex(seed_hex.strip()))

    @classmethod
    def from_env(cls, env_var: str = SIGNING_KEY_ENV) -> "EvidenceSigner | None":
        seed = os.environ.get(env_var, "").strip()
        return cls.from_hex(seed) if seed else None

    def sign_hex(self, digest_hex: str) -> str:
        """Sign the domain-separated frame of a hex digest; returns hex."""
        return self._key.sign(frame_digest(digest_hex)).hex()


def verify_signature(public_key_hex: str, digest_hex: str, signature_hex: str) -> bool:
    """Return True when ``signature_hex`` attests to the framed ``digest_hex``."""
    try:
        key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        key.verify(bytes.fromhex(signature_hex), frame_digest(digest_hex))
        return True
    except (InvalidSignature, ValueError):
        return False


def key_id_for(public_key_hex: str) -> str:
    """Short stable identity for a verification key (first 16 hex of sha256)."""
    import hashlib

    return hashlib.sha256(bytes.fromhex(public_key_hex)).hexdigest()[:16]
