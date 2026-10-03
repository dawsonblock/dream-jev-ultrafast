"""Phase 17 — trusted computing base boundary checks."""

import hashlib
import importlib.util
import json
from pathlib import Path

from jev_ultrafast.tcb import TCB_FILES, is_tcb, verify_installation

ROOT = Path(__file__).resolve().parent.parent


def _load_check_tcb():
    """Import scripts/check_tcb.py as a module (scripts aren't a package)."""
    spec = importlib.util.spec_from_file_location(
        "jev_check_tcb", ROOT / "scripts" / "check_tcb.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tcb_tree(tmp_path):
    """Minimal consistent tree: stub content for every listed TCB file
    plus one package-prefix member, with a matching MANIFEST.sha256."""
    files = sorted(TCB_FILES) + [
        "jev_ultrafast/_dream/_stub_member.py",
        "jev_ultrafast/_learning/_stub_member.py",
    ]
    lines = []
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"stub:{name}".encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}")
    (tmp_path / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    return tmp_path


def test_tcb_clean_tree_passes(tmp_path):
    check_tcb = _load_check_tcb()
    assert check_tcb.check(_tcb_tree(tmp_path)) == []


def test_tcb_detects_drifted_trusted_file(tmp_path):
    """A byte changed inside the trusted base fails closed — the manifest
    is the reference and a signature over it covers the TCB."""
    check_tcb = _load_check_tcb()
    root = _tcb_tree(tmp_path)
    (root / "jev_ultrafast/policy.py").write_bytes(b"tampered")
    violations = check_tcb.check(root)
    assert any("policy.py" in v and "drift" in v for v in violations)


def test_tcb_rejects_trusted_code_outside_manifest(tmp_path):
    """A file under a TCB prefix is trusted by construction — it must be
    covered by the signed release set."""
    check_tcb = _load_check_tcb()
    root = _tcb_tree(tmp_path)
    extra = root / "jev_ultrafast/_dream/sneaky.py"
    extra.write_bytes(b"# new trusted member, never signed")
    violations = check_tcb.check(root)
    assert any("sneaky.py" in v and "absent" in v for v in violations)


def test_tcb_rejects_missing_and_unlisted(tmp_path):
    """A listed TCB file deleted from the tree, or dropped from the
    manifest, is a violation either way."""
    check_tcb = _load_check_tcb()
    root = _tcb_tree(tmp_path)
    (root / "jev_ultrafast/agent.py").unlink()
    manifest = root / "MANIFEST.sha256"
    kept = [
        line for line in manifest.read_text().splitlines()
        if "signing.py" not in line
    ]
    manifest.write_text("\n".join(kept) + "\n")
    violations = check_tcb.check(root)
    assert any("agent.py" in v and "missing" in v for v in violations)
    assert any("signing.py" in v and "absent" in v for v in violations)


def test_is_tcb_boundary_classification():
    # Prefix members are trusted automatically — new code inside the
    # trusted packages can never be *non*-TCB by accident (fail closed
    # for any write-guard classifying a candidate's output target).
    assert is_tcb("jev_ultrafast/_dream/registry.py")
    assert is_tcb("jev_ultrafast/_learning/causal.py")
    assert is_tcb("jev_ultrafast/_dream/future_module.py")
    # The individually named boundary files.
    assert is_tcb("jev_ultrafast/policy.py")
    assert is_tcb("scripts/qualify.py")
    # The boundary definition is itself TCB — a candidate must not
    # shrink the boundary by editing the list.
    assert is_tcb("jev_ultrafast/tcb.py")
    # Non-TCB: tests, docs, the trace-viewer UI.
    assert not is_tcb("jev_ultrafast/static/app.js")
    assert not is_tcb("tests/test_tcb.py")
    assert not is_tcb("docs/dream-rsi-integration.md")


# ------------------------------------------- runtime release verification
#
# JEV_SECURITY_PROFILE=qualified re-verifies the installed tree against its
# signed release manifest before the agent starts; the external verifier
# (scripts/verify_release.py) does the same from outside the artifact, and
# additionally pins the TCB version and binds the generated validation
# report to the manifest digest it validated.

_SIG_DOMAIN = b"jev-dream/release-manifest/v1:"
_QUALIFY_DOMAIN = b"jev-dream/qualify-report/v1:"
_SIGNER_SEED = "33" * 32
_QUALIFY_SEED = "55" * 32
_OTHER_SEED = "44" * 32


def _signer_key_id(seed_hex):
    from jev_ultrafast.signing import EvidenceSigner

    return EvidenceSigner.from_hex(seed_hex).key_id


def _sign_manifest(root: Path, seed_hex: str) -> None:
    """Write MANIFEST.sig for ``root`` the way scripts/sign_manifest.py does."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        PublicFormat,
    )

    digest = hashlib.sha256(
        (root / "MANIFEST.sha256").read_bytes()
    ).hexdigest()
    private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed_hex))
    key_id = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    block = {
        "schema": "jev-manifest-sig/1",
        "kind": "release_manifest_signature",
        "manifest_digest": digest,
        "signed_at_ms": 1,
        "key_id": key_id,
        "signature": private.sign(_SIG_DOMAIN + digest.encode("ascii")).hex(),
    }
    (root / "MANIFEST.sig").write_text(json.dumps(block, sort_keys=True))


def _release_tree(tmp_path) -> Path:
    """A minimal *signed* release tree: the TCB floor, one prefix member with
    a TCB_VERSION declaration, a manifest, and a valid signature."""
    files = sorted(TCB_FILES) + [
        "jev_ultrafast/_dream/common.py",
        "jev_ultrafast/_dream/_stub_member.py",
        "jev_ultrafast/_learning/_stub_member.py",
        "jev_ultrafast/static/app.js",
    ]
    lines = []
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            'TCB_VERSION = "jev-ultrafast-tcb/0.17"\n'
            if name.endswith("common.py")
            else f"stub:{name}\n"
        )
        path.write_text(content)
        lines.append(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}"
        )
    (tmp_path / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    _sign_manifest(tmp_path, _SIGNER_SEED)
    _sign_qualification(tmp_path, _QUALIFY_SEED)
    return tmp_path


def _sign_qualification(root: Path, seed_hex: str, *, qualified=True,
                        manifest_digest=None) -> Path:
    """Write RELEASE_QUALIFICATION.json the way scripts/qualify.py --sign
    emits it: a jev-qualify report bound to the manifest digest and signed
    under the qualification domain."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        PublicFormat,
    )

    if manifest_digest is None:
        manifest_digest = hashlib.sha256(
            (root / "MANIFEST.sha256").read_bytes()
        ).hexdigest()
    private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed_hex))
    key_id = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    report = {
        "schema": "jev-qualify/4",
        "mode": "full",
        "release_qualified": qualified,
        "provenance": {"manifest_digest": manifest_digest},
    }
    digest = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report["signature"] = {
        "schema": "jev-qualify-sig/1",
        "kind": "qualification_report_signature",
        "report_digest": digest,
        "key_id": key_id,
        "signature": private.sign(
            _QUALIFY_DOMAIN + digest.encode("ascii")
        ).hex(),
    }
    path = root / "RELEASE_QUALIFICATION.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return path


def _verify(root, keys=None, qualify_keys=None, **pins):
    return verify_installation(
        root,
        {_signer_key_id(_SIGNER_SEED)} if keys is None else keys,
        {_signer_key_id(_QUALIFY_SEED)} if qualify_keys is None else qualify_keys,
        **pins,
    )


def test_verify_installation_signed_tree_passes(tmp_path):
    assert _verify(_release_tree(tmp_path)) == []


def test_verify_installation_fails_closed_on_tampering(tmp_path):
    # A byte changed anywhere in the release set.
    root = _release_tree(tmp_path)
    (root / "jev_ultrafast/policy.py").write_text("tampered")
    assert any("policy.py" in v for v in _verify(root))
    # Trusted code that never made the signed manifest.
    root = _release_tree(tmp_path)
    (root / "jev_ultrafast/_dream/sneaky.py").write_text("# unmanifested")
    assert any("sneaky.py" in v for v in _verify(root))


def test_verify_installation_fails_closed_on_provenance(tmp_path):
    # Unsigned tree — no provenance at all.
    root = _tcb_tree(tmp_path)
    assert any("MANIFEST.sig" in v for v in _verify(root))
    # Signed under a key the caller never pinned.
    root = _release_tree(tmp_path)
    assert any(
        "unexpected key" in v
        for v in _verify(root, keys={_signer_key_id(_OTHER_SEED)})
    )
    # Manifest regenerated after signing — the signature no longer covers it.
    root = _release_tree(tmp_path)
    (root / "MANIFEST.sha256").write_text("changed\n")
    assert any("does not match" in v for v in _verify(root))
    # No pinned key configured — an unpinned verifier accepts anything.
    root = _release_tree(tmp_path)
    assert any("VERIFY_KEYS" in v for v in _verify(root, keys=set()))


def test_verify_installation_requires_signed_qualification(tmp_path):
    """``qualified`` asserts qualification evidence, not just integrity: a
    signed release tree still fails closed without a signed qualification
    report bound to this manifest's digest."""
    # Missing report — integrity proven, qualification absent.
    root = _release_tree(tmp_path)
    (root / "RELEASE_QUALIFICATION.json").unlink()
    assert any(
        "RELEASE_QUALIFICATION" in v for v in _verify(root)
    )
    # Unsigned report — anybody could have written it.
    root = _release_tree(tmp_path)
    report = json.loads((root / "RELEASE_QUALIFICATION.json").read_text())
    del report["signature"]
    (root / "RELEASE_QUALIFICATION.json").write_text(json.dumps(report))
    assert any("unsigned" in v for v in _verify(root))
    # Signed by a key the caller never pinned as a qualification authority.
    root = _release_tree(tmp_path)
    assert any(
        "unexpected key" in v
        for v in _verify(root, qualify_keys={_signer_key_id(_OTHER_SEED)})
    )
    # Report bound to a different manifest — stale evidence for this tree.
    root = _release_tree(tmp_path)
    _sign_qualification(root, _QUALIFY_SEED, manifest_digest="0" * 64)
    assert any("different manifest" in v for v in _verify(root))
    # release_qualified false — validation evidence is not a release verdict.
    root = _release_tree(tmp_path)
    _sign_qualification(root, _QUALIFY_SEED, qualified=False)
    assert any("not true" in v for v in _verify(root))
    # Report tampered after signing — the digest no longer covers it.
    root = _release_tree(tmp_path)
    report = json.loads((root / "RELEASE_QUALIFICATION.json").read_text())
    report["release_qualified"] = False
    (root / "RELEASE_QUALIFICATION.json").write_text(json.dumps(report))
    assert any("digest mismatch" in v for v in _verify(root))


def test_verify_installation_anti_rollback_pins(tmp_path):
    """A validly-signed older/weaker release is authentic but not the
    approved one: TCB-version and manifest-digest pins fail closed on
    drift and pass on the exact pinned release."""
    root = _release_tree(tmp_path)
    digest = hashlib.sha256(
        (root / "MANIFEST.sha256").read_bytes()).hexdigest()
    assert _verify(
        root,
        expect_tcb_version="jev-ultrafast-tcb/0.17",
        expect_manifest_digest=digest,
    ) == []
    assert any(
        "TCB_VERSION" in v
        for v in _verify(root, expect_tcb_version="jev-ultrafast-tcb/0.16")
    )
    assert any(
        "pinned release" in v
        for v in _verify(root, expect_manifest_digest="0" * 64)
    )


def _load_verify_release():
    spec = importlib.util.spec_from_file_location(
        "jev_verify_release", ROOT / "scripts" / "verify_release.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verify_release_external_verifier(tmp_path):
    """The external verifier runs the full release contract from outside the
    artifact: signature, hashes, TCB floor, TCB version, report binding."""
    verify_release = _load_verify_release()
    root = _release_tree(tmp_path)
    key = _signer_key_id(_SIGNER_SEED)
    qkey = _signer_key_id(_QUALIFY_SEED)
    assert verify_release.check(root, {key}, qualify_keys={qkey}) == []
    # TCB version pin — matching expectation passes, drift fails closed.
    assert verify_release.check(
        root, {key}, expect_tcb_version="jev-ultrafast-tcb/0.17",
        qualify_keys={qkey}) == []
    assert any(
        "TCB_VERSION" in v for v in verify_release.check(
            root, {key}, expect_tcb_version="jev-ultrafast-tcb/0.99",
            qualify_keys={qkey})
    )


def test_verify_release_binds_validation_report(tmp_path):
    """A generated report traveling with the release must name the manifest
    digest it validates — a stale or unbound report is not provenance."""
    verify_release = _load_verify_release()
    root = _release_tree(tmp_path)
    key = _signer_key_id(_SIGNER_SEED)
    qkey = _signer_key_id(_QUALIFY_SEED)
    digest = hashlib.sha256(
        (root / "MANIFEST.sha256").read_bytes()).hexdigest()
    (root / "VALIDATION.generated.md").write_text(
        f"# report\n- manifest_digest: `{digest}`\n")
    assert verify_release.check(root, {key}, qualify_keys={qkey}) == []
    # Stale report — validates a different artifact.
    (root / "VALIDATION.generated.md").write_text(
        f"# report\n- manifest_digest: `{'0' * 64}`\n")
    assert any(
        "different artifact" in v for v in verify_release.check(
            root, {key}, qualify_keys={qkey})
    )
    # Unbound report — no manifest digest declared at all.
    (root / "VALIDATION.generated.md").write_text("# report\nno binding\n")
    assert any(
        "manifest digest" in v for v in verify_release.check(
            root, {key}, qualify_keys={qkey})
    )
    # Required-but-absent is a release failure under --require-validation.
    (root / "VALIDATION.generated.md").unlink()
    assert verify_release.check(root, {key}, qualify_keys={qkey}) == []
    assert any(
        "required but absent" in v for v in verify_release.check(
            root, {key}, require_validation=True, qualify_keys={qkey})
    )


def test_verify_release_full_set_enumeration(tmp_path):
    """The external verifier rejects files the manifest never covered —
    anywhere in the tree, not only under trusted prefixes. A signature
    vouches for a fully specified release set, not a subset."""
    verify_release = _load_verify_release()
    key = _signer_key_id(_SIGNER_SEED)
    root = _release_tree(tmp_path)
    assert verify_release.check(
        root, {key}, qualify_keys={_signer_key_id(_QUALIFY_SEED)}) == []
    # Outside every trusted prefix — an added script is outside the claim.
    (root / "extra.py").write_text("# smuggled\n")
    assert any(
        "extra.py" in v for v in verify_release.check(root, {key})
    )
    (root / "extra.py").unlink()
    sub = root / "docs"
    sub.mkdir()
    (sub / "runme.sh").write_text("echo hi\n")
    assert any(
        "runme.sh" in v for v in verify_release.check(root, {key})
    )
    # Excluded names stay excluded: the signature and reports are not
    # manifest-listed by construction.
    (sub / "runme.sh").unlink()
    assert verify_release.check(
        root, {key}, qualify_keys={_signer_key_id(_QUALIFY_SEED)}) == []


def test_verify_release_qualification_evidence(tmp_path):
    """Qualification is a separate claim from integrity: the signed report
    must verify under a pinned qualification key and bind this manifest —
    and --require-qualification demands release_qualified == true."""
    verify_release = _load_verify_release()
    key = _signer_key_id(_SIGNER_SEED)
    qkey = _signer_key_id(_QUALIFY_SEED)
    root = _release_tree(tmp_path)
    assert verify_release.check(
        root, {key}, require_qualification=True,
        qualify_keys={qkey}) == []
    # No pinned qualification key — a present report cannot be accepted.
    assert any(
        "qualification key" in v for v in verify_release.check(root, {key})
    )
    # Unsigned report — rejected even though the release itself is signed.
    report = json.loads((root / "RELEASE_QUALIFICATION.json").read_text())
    del report["signature"]
    (root / "RELEASE_QUALIFICATION.json").write_text(json.dumps(report))
    assert any(
        "unsigned" in v
        for v in verify_release.check(root, {key}, qualify_keys={qkey})
    )
    # Signed by a key the caller never pinned for qualification.
    root = _release_tree(tmp_path)
    assert any(
        "unexpected key" in v
        for v in verify_release.check(
            root, {key}, qualify_keys={_signer_key_id(_OTHER_SEED)})
    )
    # Bound to a different manifest — stale evidence.
    root = _release_tree(tmp_path)
    _sign_qualification(root, _QUALIFY_SEED, manifest_digest="0" * 64)
    assert any(
        "different manifest" in v
        for v in verify_release.check(root, {key}, qualify_keys={qkey})
    )
    # release_qualified false — the require flag turns honest "not
    # qualified" into a violation; without it the report still verifies.
    root = _release_tree(tmp_path)
    _sign_qualification(root, _QUALIFY_SEED, qualified=False)
    assert verify_release.check(root, {key}, qualify_keys={qkey}) == []
    assert any(
        "not true" in v
        for v in verify_release.check(
            root, {key}, require_qualification=True, qualify_keys={qkey})
    )
    # Absent entirely — absence is only a violation when required.
    root = _release_tree(tmp_path)
    (root / "RELEASE_QUALIFICATION.json").unlink()
    assert verify_release.check(root, {key}) == []
    assert any(
        "required but absent" in v
        for v in verify_release.check(
            root, {key}, require_qualification=True, qualify_keys={qkey})
    )


def test_verify_release_manifest_digest_pin(tmp_path):
    """Anti-rollback: the caller may pin the exact manifest digest it
    approved — an older validly-signed release fails closed."""
    verify_release = _load_verify_release()
    key = _signer_key_id(_SIGNER_SEED)
    root = _release_tree(tmp_path)
    digest = hashlib.sha256(
        (root / "MANIFEST.sha256").read_bytes()).hexdigest()
    assert verify_release.check(
        root, {key}, expect_manifest_digest=digest,
        qualify_keys={_signer_key_id(_QUALIFY_SEED)}) == []
    assert any(
        "pinned" in v
        for v in verify_release.check(
            root, {key}, expect_manifest_digest="0" * 64)
    )
