"""Structured fuzzing of the trust-boundary parse surfaces (qualification §53).

Four families, all in-process and deterministic:

- ``chain-mutate`` — build a real hash-chained store, then mutate raw lines
  (byte flips, mid-chain JSON injection, prev_hash/schema/signature forgery,
  reorders, duplicates, torn tails, blank-line injection). Oracle: ``load()``
  + ``verify()`` must either raise or return a strict prefix of the original
  chain — silent history alteration is impossible.
- ``event-hash`` — hostile event dicts (reserved keys, ``|`` delimiters,
  unicode/control chars, NaN/Infinity, deep nesting) through the canonical
  dumps → hash → loads round-trip. Oracle: the parsed record re-hashes
  identically, i.e. serialization is injective on this domain.
- ``real-append`` — hostile dicts through ``ExperienceStore.append`` itself
  (fsync path). Reserved keys must raise; survivors must reload cleanly.
- ``context-join`` — the ``|``-joined treatment-signature coordinates under
  attacker-shaped strings. Oracle: no coordinate may inject a delimiter, so
  the joined key splits back to exactly the sanitized fields — two distinct
  context tuples can never merge into one estimate key.

    uv run python scripts/fuzz_check.py [--cases 1000000] [--seed ...]

Exit 0: every oracle held. Any accepted-corruption or unexpected raise fails.
"""

import argparse
import json
import random
import string
import sys
import tempfile
from pathlib import Path

HOSTILE_POOL = (
    list(string.printable)
    + ["|", "||", "|||", "\x00", "\x01", "é", "中", "🦊", "\u202e", "f:*", "s:|",
       " ", "", "unknown", "*", "null", "NaN", "0"*80, "A"*300]
)


def _hostile_str(rng: random.Random) -> str:
    n = rng.randrange(0, 6)
    if n == 0:
        return ""
    return "".join(rng.choice(HOSTILE_POOL) for _ in range(n))[:120]


def _hostile_value(rng: random.Random, depth: int = 0):
    r = rng.random()
    if depth < 3 and r < 0.15:
        return [_hostile_value(rng, depth + 1) for _ in range(rng.randrange(4))]
    if depth < 3 and r < 0.30:
        return {_hostile_str(rng): _hostile_value(rng, depth + 1)
                for _ in range(rng.randrange(4))}
    if r < 0.45:
        return _hostile_str(rng)
    if r < 0.55:
        return rng.choice([0, -1, 2**63, -2**63, 3.14, float("nan"),
                           float("inf"), float("-inf"), True, None])
    if r < 0.62:
        return rng.randrange(-10**9, 10**9)
    return _hostile_str(rng)


def fuzz_event_hash(rng: random.Random, cases: int, event_hash) -> int:
    """dumps→hash→loads must be injective: every parseable record re-hashes
    to itself. Non-dict/non-JSON-serializable payloads are out of domain —
    ``append`` never sees them (callers pass dicts)."""
    ok = 0
    for _ in range(cases):
        ev = {}
        for _ in range(rng.randrange(1, 8)):
            ev[_hostile_str(rng) or "k"] = _hostile_value(rng)
        try:
            digest = event_hash(ev)
            line = json.dumps(ev, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False)
            back = json.loads(line)
        except (TypeError, ValueError, OverflowError):
            continue  # unserializable — append() rejects these too
        if not isinstance(back, dict):
            print(f"event-hash: non-dict round-trip from {ev!r}")
            return -1
        if event_hash(back) != digest:
            print(f"event-hash: round-trip changed digest for {ev!r}")
            return -1
        ok += 1
    return ok


def fuzz_context_join(rng: random.Random, cases: int, coord) -> int:
    """Sanitized coordinates must never inject the ``|`` delimiter, so the
    joined signature key is always field-aligned: split must recover exactly
    the sanitized fields."""
    ok = 0
    for _ in range(cases):
        fields = [_hostile_str(rng) for _ in range(rng.randrange(1, 13))]
        safe = [coord(f) for f in fields]
        joined = "|".join(safe)
        if any("|" in s for s in safe):
            print(f"context-join: delimiter survived sanitization in {fields!r}")
            return -1
        if joined.count("|") != len(fields) - 1:
            print(f"context-join: delimiter count wrong for {fields!r}")
            return -1
        if joined.split("|") != safe:
            print(f"context-join: split round-trip mismatch for {fields!r}")
            return -1
        ok += 1
    return ok


def fuzz_real_append(rng: random.Random, cases: int, store_cls,
                     reserved: set, tmp: Path) -> int:
    """append() under hostile payloads: reserved chain keys must raise;
    serializable survivors must be reloadable; unserializable payloads may
    raise but must never leave a silently-corrupt tail."""
    path = tmp / "fuzz-append.jsonl"
    store = store_cls(path)
    ok = appended = 0
    for i in range(cases):
        ev = {"event": "fuzz", "i": i}
        for _ in range(rng.randrange(0, 6)):
            key = rng.choice(list(reserved)) if rng.random() < 0.4 else _hostile_str(rng) or "k"
            ev[key] = _hostile_value(rng)
        if reserved & ev.keys():
            try:
                store.append(ev)
            except ValueError:
                ok += 1
                continue
            print(f"real-append: reserved keys accepted: {sorted(reserved & ev.keys())}")
            return -1
        try:
            store.append(ev)
            appended += 1
        except (TypeError, ValueError, OverflowError):
            pass  # unserializable — rejection is the correct outcome
        # Every state, accepted or rejected, must still load as a valid chain.
        store.load()
        store.verify()
        ok += 1
    return ok


def fuzz_chain_mutate(rng: random.Random, cases: int, store_cls,
                      tmp: Path) -> int:
    """Line-level corruption of a real chain. Accepted states must be a
    strict prefix of the original (torn-tail discard, blank-line skip, or a
    fully-intact final record); anything else must raise."""
    src = tmp / "fuzz-src.jsonl"
    store = store_cls(src)
    base_records = [
        {"event": "transition", "run_id": "f", "task_key": "t", "step": i,
         "state": "S", "selected": {"id": "a"}, "page_changed": True}
        for i in range(6)
    ]
    originals = [store.append(r) for r in base_records]
    base_lines = src.read_bytes().split(b"\n")[:-1]  # drop trailing empty
    ok = 0
    dst = tmp / "fuzz-mut.jsonl"
    for _ in range(cases):
        lines = list(base_lines)
        n = len(lines)
        m = rng.random()
        if m < 0.18:  # byte flip inside a random line
            i = rng.randrange(n)
            b = bytearray(lines[i])
            b[rng.randrange(len(b))] ^= 1 << rng.randrange(8)
            lines[i] = bytes(b)
        elif m < 0.30:  # truncate the tail mid-line
            cut = rng.randrange(1, len(lines[-1]))
            lines[-1] = lines[-1][:cut]
        elif m < 0.42:  # inject hostile raw bytes as a new mid/final line
            blob = rng.choice([b"{", b'{"a"', b"garbage", b'{"schema":"x"}',
                               b'"just a string"', b"[1,2]", b"null",
                               _hostile_str(rng).encode("utf-8", "replace")])
            lines.insert(rng.randrange(n + 1), blob)
        elif m < 0.52:  # schema/tcb_version swap on a random line
            i = rng.randrange(n)
            e = json.loads(lines[i])
            if rng.random() < 0.5:
                e["schema"] = rng.choice(["jev-dream/0", "jev-dream/99", None, 7])
            else:
                e["tcb_version"] = rng.choice(["jev-ultrafast-tcb/0.0", "x", None])
            lines[i] = json.dumps(e, sort_keys=True,
                                  separators=(",", ":")).encode()
        elif m < 0.62:  # forge signature/prev_hash/event_hash fields
            i = rng.randrange(n)
            e = json.loads(lines[i])
            field = rng.choice(["signature", "prev_hash", "event_hash",
                                "key_id", "recorded_at_ms"])
            e[field] = rng.choice(["f" * 64, "0" * 64, "forged", None,
                                   rng.randrange(10**15)])
            lines[i] = json.dumps(e, sort_keys=True,
                                  separators=(",", ":")).encode()
        elif m < 0.72:  # reorder two lines
            i, j = rng.randrange(n), rng.randrange(n)
            lines[i], lines[j] = lines[j], lines[i]
        elif m < 0.80:  # duplicate a line
            lines.insert(rng.randrange(n + 1), lines[rng.randrange(n)])
        elif m < 0.92:  # blank / whitespace-only line injection (harmless)
            lines.insert(rng.randrange(n + 1),
                         rng.choice([b"", b" ", b"\t", b"  "]))
        else:  # drop a middle line (gap in history)
            if n > 2:
                del lines[rng.randrange(1, n - 1)]

        dst.write_bytes(b"\n".join(lines) + (b"\n" if rng.random() < 0.8 else b""))
        candidate = store_cls(dst)
        try:
            events = candidate.load()
            candidate.verify()
        except ValueError:
            ok += 1  # fail-closed — the expected outcome for real corruption
            continue
        # Accepted: the surviving events must be a prefix of the original
        # chain (torn-tail discard or harmless whitespace), byte-identical
        # in every kept record.
        if len(events) > len(originals):
            print("chain-mutate: mutation GREW the chain")
            return -1
        for k, ev in enumerate(events):
            if ev.get("event_hash") != originals[k].get("event_hash"):
                print(f"chain-mutate: record {k} silently altered and accepted")
                return -1
        ok += 1
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0xF00D)
    args = parser.parse_args()

    repo = str(Path(__file__).resolve().parent.parent)
    sys.path.insert(0, repo)
    from jev_ultrafast.dream import ExperienceStore
    from jev_ultrafast.dreamlearn import _coord

    rng = random.Random(args.seed)
    tmp = Path(tempfile.mkdtemp(prefix="jev-fuzz-"))

    # Budget: cheap in-memory invariants carry the volume; the fsync-bound
    # real-append family keeps file-level coverage honest.
    budgets = {
        "context-join": int(args.cases * 0.50),
        "event-hash": int(args.cases * 0.45),
        "chain-mutate": int(args.cases * 0.049),
        "real-append": int(args.cases * 0.001),
    }
    results = {}
    t0 = __import__("time").monotonic()

    r = fuzz_context_join(rng, budgets["context-join"], _coord)
    if r < 0:
        return 1
    results["context-join"] = r
    print(f"context-join: {r} cases PASS", flush=True)

    r = fuzz_event_hash(rng, budgets["event-hash"], ExperienceStore._event_hash)
    if r < 0:
        return 1
    results["event-hash"] = r
    print(f"event-hash: {r} cases PASS", flush=True)

    r = fuzz_chain_mutate(rng, budgets["chain-mutate"], ExperienceStore, tmp)
    if r < 0:
        return 1
    results["chain-mutate"] = r
    print(f"chain-mutate: {r} cases PASS", flush=True)

    r = fuzz_real_append(rng, budgets["real-append"], ExperienceStore,
                         ExperienceStore._RESERVED_EVENT_KEYS, tmp)
    if r < 0:
        return 1
    results["real-append"] = r
    print(f"real-append: {r} cases PASS", flush=True)

    total = sum(results.values())
    print(f"fuzz: {total} structured cases in "
          f"{__import__('time').monotonic() - t0:.0f}s — all oracles held")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
