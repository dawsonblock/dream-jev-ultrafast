"""Regenerate and verify MANIFEST.sha256 — the release integrity manifest.

The manifest covers every file in the release set — tracked plus untracked
non-ignored files in a worktree, or the manifest's own file list in a
packaged tree that has no ``.git`` — in sorted order, in ``sha256sum``
format (``<hash>  <path>``). The manifest file itself is never listed.

    uv run python scripts/update_manifest.py            # regenerate
    uv run python scripts/update_manifest.py --check    # verify, write nothing

``--check`` is the portable equivalent of ``sha256sum -c MANIFEST.sha256``
(which CI runs) for platforms without coreutils: it regenerates the expected
content and compares byte-for-byte, so a stale hash, a missing file, a wrong
path, or a formatting drift all fail the same way.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = "MANIFEST.sha256"


def _tracked_files(root: Path) -> list[str]:
    """The manifest's file set: tracked *and* untracked non-ignored files in a
    worktree (a not-yet-staged release file must not be silently dropped),
    else the existing manifest's own list — a packaged tree has no .git and
    that list is still the authoritative release set."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout
        files = [name for name in out.splitlines() if name.strip()]
    except (OSError, subprocess.CalledProcessError):
        files = []
    if not files:
        manifest = root / MANIFEST
        if not manifest.exists():
            raise SystemExit(
                "no git worktree and no MANIFEST.sha256 to re-cover; "
                "nothing to regenerate from"
            )
        files = [
            line.split("  ", 1)[-1].strip()
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return sorted(name for name in files if name != MANIFEST)


def _render(root: Path, files: list[str]) -> str:
    lines = []
    for name in files:
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}")
    return "\n".join(lines) + "\n"


def main() -> int:
    check = "--check" in sys.argv
    files = _tracked_files(ROOT)
    expected = _render(ROOT, files)
    manifest = ROOT / MANIFEST
    if not check:
        manifest.write_text(expected, encoding="utf-8")
        print(f"{MANIFEST}: regenerated over {len(files)} files")
        return 0
    actual = manifest.read_text(encoding="utf-8") if manifest.exists() else ""
    if actual == expected:
        print(f"{MANIFEST}: {len(files)} entries verified")
        return 0
    expected_lines = expected.splitlines()
    actual_lines = actual.splitlines()
    for i, want in enumerate(expected_lines):
        got = actual_lines[i] if i < len(actual_lines) else "<missing>"
        if got != want:
            print(f"{MANIFEST}: first mismatch at line {i + 1}:", file=sys.stderr)
            print(f"  expected: {want}", file=sys.stderr)
            print(f"  actual:   {got}", file=sys.stderr)
            break
    else:
        print(
            f"{MANIFEST}: {len(actual_lines)} entries vs {len(expected_lines)} expected",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
