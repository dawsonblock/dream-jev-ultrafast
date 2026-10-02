"""Emit the run-environment digest the qualification plan requires (§3).

Every official release/qualification run records: OS, kernel, Python, Node,
Chrome, commit, tree digest, lock digest, suite digest, timestamp — so a
result is attributable to an exact environment, not a host name.

    uv run python scripts/env_digest.py            # JSON to stdout
    uv run python scripts/env_digest.py --save OUT # also writes OUT.json
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(cmd):
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, check=True, cwd=ROOT
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _tree_digest() -> str | None:
    """Digest over git's tracked-file set + contents — the exact tree tested."""
    files = _run(["git", "ls-files"])
    if files is None:
        return None
    h = hashlib.sha256()
    for name in sorted(files.splitlines()):
        p = ROOT / name
        if p.is_file():
            h.update(name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _chrome_version():
    cdp = os.environ.get("BU_CDP_URL", "http://127.0.0.1:9222")
    try:
        with urllib.request.urlopen(cdp + "/json/version", timeout=2) as r:
            return json.loads(r.read()).get("Browser")
    except (OSError, ValueError):
        return None


def _tests_digest() -> str:
    h = hashlib.sha256()
    for p in sorted((ROOT / "tests").glob("test_*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", help="Also write the digest JSON to this path")
    args = parser.parse_args()

    digest = {
        "schema": "jev-env-digest/1",
        "os": platform.system(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "node": _run(["node", "--version"]),
        "chrome": _chrome_version(),
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain"])),
        "tree_digest": _tree_digest(),
        "lock_digest": _sha256(ROOT / "uv.lock"),
        "manifest_digest": _sha256(ROOT / "MANIFEST.sha256"),
        "test_suite_digest": _tests_digest(),
        "recorded_at_ms": int(time.time() * 1000),
    }
    digest["env_digest"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in digest.items() if k != "env_digest"},
            sort_keys=True, separators=(",", ":"),
        ).encode()
    ).hexdigest()
    out = json.dumps(digest, indent=2, sort_keys=True)
    print(out)
    if args.save:
        Path(args.save).write_text(out + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
