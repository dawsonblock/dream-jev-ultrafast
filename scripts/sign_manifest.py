"""Sign and verify MANIFEST.sha256 — the release-provenance half of the
signed-manifest gate.

The manifest covers file *contents*; the signature covers the manifest's own
SHA-256 digest, so a signed manifest binds the whole release set without the
manifest needing to hash its own signature file.

    # release machine (key never leaves it):
    uv run python scripts/sign_manifest.py --sign <ed25519-seed-hex>
    # anywhere else:
    uv run python scripts/sign_manifest.py --verify --key <pubkey-hex>
    JEV_MANIFEST_VERIFY_KEYS=<k1>,<k2> uv run python scripts/sign_manifest.py --verify

``--sign`` writes MANIFEST.sig (schema ``jev-manifest-sig/1``). ``--verify``
fails closed: missing MANIFEST.sig, unknown signing key, digest mismatch, or a
bad signature all exit 1. Verification does not re-check individual files —
that is ``update_manifest.py --check`` / ``sha256sum -c``.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = "MANIFEST.sha256"
SIG_FILE = "MANIFEST.sig"
SIG_SCHEMA = "jev-manifest-sig/1"
SIG_DOMAIN = b"jev-dream/release-manifest/v1:"


def _manifest_digest(root: Path) -> str:
    path = root / MANIFEST
    if not path.exists():
        raise SystemExit(f"{MANIFEST} not found")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sign(root: Path, seed_hex: str) -> int:
    sys.path.insert(0, str(root))
    from jev_ultrafast.signing import EvidenceSigner

    signer = EvidenceSigner.from_hex(seed_hex.strip())
    digest = _manifest_digest(root)
    block = {
        "schema": SIG_SCHEMA,
        "kind": "release_manifest_signature",
        "manifest_digest": digest,
        "signed_at_ms": int(time.time() * 1000),
        "key_id": signer.key_id,
        "signature": signer.sign_hex(digest, domain=SIG_DOMAIN),
    }
    out = root / SIG_FILE
    out.write_text(
        json.dumps(block, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"{SIG_FILE}: manifest digest {digest[:16]}… signed by "
          f"{signer.key_id[:16]}…")
    return 0


def _verify(root: Path, keys: set[str]) -> int:
    sys.path.insert(0, str(root))
    from jev_ultrafast.signing import verify_signature

    sig_path = root / SIG_FILE
    if not sig_path.exists():
        print(f"{SIG_FILE}: missing — unsigned release manifest", file=sys.stderr)
        return 1
    try:
        block = json.loads(sig_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"{SIG_FILE}: unreadable ({exc})", file=sys.stderr)
        return 1
    if not isinstance(block, dict) or block.get("schema") != SIG_SCHEMA:
        print(f"{SIG_FILE}: unrecognized signature block", file=sys.stderr)
        return 1
    key_id = block.get("key_id")
    signature = block.get("signature")
    digest = _manifest_digest(root)
    if block.get("manifest_digest") != digest:
        print(f"{SIG_FILE}: signed digest does not match {MANIFEST} — "
              "manifest changed after signing", file=sys.stderr)
        return 1
    if key_id not in keys:
        print(f"{SIG_FILE}: manifest signed by an unexpected key", file=sys.stderr)
        return 1
    if not signature or not verify_signature(key_id, digest, str(signature),
                                             domain=SIG_DOMAIN):
        print(f"{SIG_FILE}: signature verification failure", file=sys.stderr)
        return 1
    print(f"{SIG_FILE}: manifest digest {digest[:16]}… verified under key "
          f"{str(key_id)[:16]}…")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sign", metavar="SEED_HEX",
                        help="Ed25519 seed hex; writes MANIFEST.sig")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--key", action="append", default=[],
                        help="accepted verify key (repeatable); "
                             "JEV_MANIFEST_VERIFY_KEYS is also honored")
    args = parser.parse_args()

    keys = {k.strip() for k in args.key if k and k.strip()}
    env = os.environ.get("JEV_MANIFEST_VERIFY_KEYS", "")
    keys.update(k.strip() for k in env.split(",") if k.strip())

    if args.sign:
        return _sign(ROOT, args.sign)
    if args.verify:
        if not keys:
            print("no verification key configured — refusing to accept an "
                  "unpinned manifest", file=sys.stderr)
            return 1
        return _verify(ROOT, keys)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
