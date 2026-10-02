"""Bounded mutation sweep over trust-boundary code (qualification §55).

Applies a curated set of single-line mutations to the evidence/causal TCB —
each one weakens a fail-closed gate — and asserts the focused test files turn
red. A mutant that leaves the suite green is a *survivor*: a gate the tests
claim to hold that nothing actually detects.

    uv run python scripts/mutate_check.py

Exit 0: every mutant was killed by the suite. Exit 1 lists survivors. The
working tree is restored after every probe, killed or surviving.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (file, exact needle, replacement, pytest targets, what the gate stops)
MUTANTS = [
    ("jev_ultrafast/dream.py",
     "elif self.require_signatures:",
     "elif False:",
     ["tests/test_authority.py"],
     "unsigned evidence accepted under required signing"),
    ("jev_ultrafast/dream.py",
     "        if anchor.get(\"digest\") != digest:\n"
     "            raise ValueError(\"DREAM chain-head anchor digest mismatch\")",
     "        if False:\n"
     "            raise ValueError(\"DREAM chain-head anchor digest mismatch\")",
     ["tests/test_authority.py"],
     "forged chain-head anchor digest accepted"),
    ("jev_ultrafast/dream.py",
     "if anchor.get(\"head\") != head or int(anchor.get(\"sequence\", -1)) != len(events):",
     "if False:",
     ["tests/test_authority.py"],
     "stale/truncated-anchor disagreement accepted"),
    ("jev_ultrafast/dream.py",
     "if event.get(\"prev_hash\") != expected_prev:",
     "if False:",
     ["tests/test_authority.py", "tests/test_learned.py"],
     "hash-chain link not verified on load"),
    ("jev_ultrafast/dream.py",
     "if event.get(\"event_hash\") != self._event_hash(event):",
     "if False:",
     ["tests/test_authority.py"],
     "record integrity not recomputed on load"),
    ("jev_ultrafast/dream.py",
     "if self._RESERVED_EVENT_KEYS & event.keys():",
     "if False:",
     ["tests/test_authority.py"],
     "callers can forge reserved chain keys"),
    ("jev_ultrafast/dream.py",
     "            signature = event.get(\"signature\")\n            if signature is not None:",
     "            signature = event.get(\"signature\")\n            if False and signature is not None:",
     ["tests/test_authority.py"],
     "event signatures never verified"),
    ("jev_ultrafast/signing.py",
     "    except (InvalidSignature, ValueError):\n        return False",
     "    except (InvalidSignature, ValueError):\n        return True",
     ["tests/test_authority.py"],
     "bad/forged signature accepted (except path returns True)"),
    ("jev_ultrafast/dreamlearn.py",
     '    text = str(value).replace("|", " ").strip()',
     '    text = str(value).strip()',
     ["tests/test_learned.py", "tests/test_causal_validity.py"],
     "`|`-delimiter injection merges distinct trial contexts"),
    ("jev_ultrafast/dreamlearn.py",
     "        enforced = [f for f in enforced if f not in fields]",
     "        enforced = []",
     ["tests/test_learned.py", "tests/test_causal_validity.py"],
     "signature backoff drops the kind/effect floor — cross-kind generalization"),
    ("jev_ultrafast/dreamlearn.py",
     '    if status in {"aborted", "claimed_done"}:\n        return "censored"',
     '    if status in {"aborted", "claimed_done"}:\n        return "failure"',
     ["tests/test_learned.py", "tests/test_causal_validity.py"],
     "censored runs counted as measured failures"),
    ("jev_ultrafast/dreamlearn.py",
     "    n = max(1.0, float(looks))",
     "    n = 1e9",
     ["tests/test_learned.py", "tests/test_causal_validity.py"],
     "sequential alpha collapses to zero — every peek claims establishment"),
]


def _run_mutant(path: Path, needle: str, replacement: str,
                tests: list[str]) -> str:
    """Apply the mutation, run the focused suite, restore. Returns the
    outcome label."""
    original = path.read_text(encoding="utf-8")
    count = original.count(needle)
    if count != 1:
        return f"SETUP-ERROR needle occurs {count}x (expected 1)"
    mutated = original.replace(needle, replacement, 1)
    try:
        path.write_text(mutated, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", *tests, "-x", "-q",
             "-p", "no:cacheprovider", "--no-cov"],
            cwd=ROOT, capture_output=True, text=True, timeout=600,
        )
        if proc.returncode == 0:
            return "SURVIVED"
        return "killed"
    except subprocess.TimeoutExpired:
        return "killed (timeout — suite cannot pass under the mutation)"
    finally:
        path.write_text(original, encoding="utf-8")


def main() -> int:
    survivors = []
    print(f"mutate-check: {len(MUTANTS)} curated mutants over the TCB\n")
    for i, (rel, needle, replacement, tests, what) in enumerate(MUTANTS, 1):
        path = ROOT / rel
        outcome = _run_mutant(path, needle, replacement, tests)
        tag = "KILLED" if outcome.startswith("killed") else outcome
        print(f"  [{i:2}/{len(MUTANTS)}] {tag:9} {rel} — {what}")
        if not outcome.startswith("killed"):
            survivors.append((rel, what, outcome))
    print()
    if survivors:
        print(f"{len(survivors)} mutant(s) SURVIVED — gates with no detecting test:")
        for rel, what, outcome in survivors:
            print(f"  {rel}: {what} [{outcome}]")
        return 1
    print(f"all {len(MUTANTS)} mutants killed — every fail-closed gate has a "
          "detecting test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
