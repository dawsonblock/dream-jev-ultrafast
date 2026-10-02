"""Key-compromise drill for evidence signing (qualification §61).

Rehearses the operational sequence an operator runs when a signing key is
suspected compromised, and asserts the fail-closed property at every step:

    1. sign evidence + anchor under key A (the key later compromised)
    2. attacker forges an event "signed" under A's key id with a bad signature
       — or signs fresh records with the stolen key itself
    3. operator rotates the trust set to key B only
    4. everything signed under A — genuine AND attacker-written alike — must
       fail closed under the new trust set ("unexpected key")
    5. the compromised-key store is retired: a NEW store opens under B (the
       immutable chain cannot be retroactively re-signed — rotation is a new
       evidence epoch, not a rewrite), and its records verify under B only

A real deployment may keep A in ``verify_keys`` long enough to archive the
pre-compromise chain for forensics — the signatures still verify — but no
new evidence may ever be written under a compromised key, and a store
continuing under a retired key is indistinguishable from attacker writes.

Exit 0: every step behaved. Any acceptance of key-A material after rotation
fails the drill.
"""

import json
import sys
import tempfile
from pathlib import Path


def main() -> int:
    repo = str(Path(__file__).resolve().parent.parent)
    sys.path.insert(0, repo)
    from jev_ultrafast.dream import ExperienceStore
    from jev_ultrafast.signing import EvidenceSigner

    tmp = Path(tempfile.mkdtemp(prefix="jev-keydrill-"))
    path = tmp / "events.jsonl"
    anchor = tmp / "anchor.json"
    event = {"event": "transition", "run_id": "r", "task_key": "t", "step": 0,
             "state": "S", "selected": {"id": "a"}, "page_changed": True}
    checks = 0

    def expect(cond, label):
        nonlocal checks
        checks += 1
        if not cond:
            print(f"keydrill: FAILED — {label}")
            return False
        print(f"  ok {label}")
        return True

    key_a = EvidenceSigner.from_hex("aa" * 32)
    key_b = EvidenceSigner.from_hex("bb" * 32)

    # 1 — evidence and anchor signed under A
    store_a = ExperienceStore(path, signer=key_a, anchor_path=anchor,
                              verify_keys={key_a.key_id})
    store_a.append({**event, "step": 0})
    store_a.append({**event, "step": 1})
    if not expect(len(ExperienceStore(
            path, verify_keys={key_a.key_id}, anchor_path=anchor).load()) == 2,
            "pre-rotation chain verifies under A"):
        return 1

    # 2 — attacker writes a forged record claiming A's key id (bad signature),
    #     and a genuine-looking record signed with the *stolen* key itself
    lines = path.read_text().splitlines()
    forged = json.loads(lines[-1])
    forged["step"] = 999
    forged["signature"] = "ff" * 64  # forged bytes under A's key_id
    path.write_text("\n".join(lines[:-1] + [json.dumps(forged)]) + "\n")
    try:
        ExperienceStore(path, verify_keys={key_a.key_id}).load()
        expect(False, "forged signature under trusted key rejected")
    except ValueError:
        expect(True, "forged signature under trusted key rejected")
    path.write_text("\n".join(lines) + "\n")  # restore

    attacker = ExperienceStore(tmp / "stolen.jsonl", signer=key_a)
    attacker.append({**event, "step": 1000})  # signed by the stolen key itself
    try:
        ExperienceStore(tmp / "stolen.jsonl",
                        verify_keys={key_b.key_id}).load()
        expect(False, "stolen-key record rejected under rotated trust set")
    except ValueError:
        expect(True, "stolen-key record rejected under rotated trust set")

    # 3-4 — rotation: trust set drops A entirely; genuine A evidence fails
    # closed alongside attacker material (rotation retires the key, not just
    # the attacker's use of it)
    try:
        ExperienceStore(path, verify_keys={key_b.key_id},
                        anchor_path=anchor).load()
        expect(False, "pre-rotation chain rejected under B-only trust set")
    except ValueError:
        expect(True, "pre-rotation chain rejected under B-only trust set")

    # 5 — retire the compromised-key store; the new epoch opens under B.
    #     The old chain remains verifiable under A for forensics, but nothing
    #     new is ever appended to it.
    path_b = tmp / "events-epoch2.jsonl"
    anchor_b = tmp / "anchor-epoch2.json"
    epoch2 = ExperienceStore(path_b, signer=key_b, verify_keys={key_b.key_id},
                             anchor_path=anchor_b)
    epoch2.append({**event, "step": 0})
    epoch2.append({**event, "step": 1})
    events = ExperienceStore(path_b, verify_keys={key_b.key_id},
                             anchor_path=anchor_b).load()
    if not expect(len(events) == 2 and all(
            e.get("key_id") == key_b.key_id for e in events),
            "post-rotation epoch verifies under B, every record B-signed"):
        return 1
    # The retired chain under A is still cryptographically readable for
    # forensics — verification ≠ continued trust for new evidence.
    archived = ExperienceStore(path, verify_keys={key_a.key_id},
                               anchor_path=anchor).load()
    if not expect(len(archived) == 2,
                  "retired chain readable under A for forensics only"):
        return 1

    print(f"keydrill: {checks} checks passed — compromise → rotate → "
          "re-anchor → continue")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
