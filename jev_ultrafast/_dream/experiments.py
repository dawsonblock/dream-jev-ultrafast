"""_dream.experiments — extracted from jev_ultrafast.dream."""

from __future__ import annotations

import json

from ..signing import (
    EXPERIMENT_PLAN_DOMAIN,
    EXPERIMENT_SIGNING_KEY_ENV,
    PROMOTION_SIGNING_KEY_ENV,
    EvidenceSigner,
    verify_signature,
)
from .common import _stable_hash

__all__ = [
    'EXPERIMENT_PLAN_SCHEMA',
    'experiment_plan_authority',
    'experiment_plan_digest',
    'experiment_plan_signature',
    'experiment_plan_signer_from_env',
]


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
