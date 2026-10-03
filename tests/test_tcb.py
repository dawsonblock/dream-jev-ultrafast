"""Phase 17 — trusted computing base boundary checks."""

import hashlib
import importlib.util
from pathlib import Path

from jev_ultrafast.tcb import TCB_FILES, is_tcb

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
