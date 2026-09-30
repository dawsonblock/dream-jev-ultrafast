# Jev Ultrafast v0.4.1 — DREAM-Jev Qualification Control Plane

v0.4.1 is a patch on v0.4.0: the first live Chrome guard run exposed that the guarded mutation expressions concatenated their argument without call parentheses, so `act`/`select`/`fill` could not execute against a real page. All three expressions now invoke their argument, `fresh()` fails closed when the guard context is destroyed, contenteditable coverage includes `""`/`plaintext-only`, and the 23-check live suite is recorded in `VALIDATION.md`.

v0.4 does not widen browser authority. It hardens the evidence and deployment path around the bounded DREAM-Jev exploration policy introduced in v0.3.

## Release blockers addressed

1. **Multi-process trace forks** — `ExperienceStore` now serializes writers with an OS lock file and verifies the full SHA-256 chain on read.
2. **Weak replay provenance** — reports bind the world pool, split manifest, store head, and represented TCB versions.
3. **Training-winner overfit** — every policy mutation is evaluated on all available splits; selection is only among candidates that pass train/validation/holdout gates.
4. **Unpaired canaries** — live promotion derives matched task-family evidence from hash-verified baseline/candidate runs.
5. **Efficiency regression at promotion** — canary gates bound live latency, action-count, and token regressions in addition to success/risk.
6. **Stale policy promotion** — staging records a parent digest and promotion rechecks lineage.
7. **No recovery path** — registry history, suspension, health checks, and rollback are now first-class.
8. **Unbound metric promotion** — disabled by default; `promote_from_store()` is the normal activation path.

## Default canary gate

The candidate must have at least 12 runs across four task families, with at least 12 baseline runs and four paired task families. At least 80% of candidate task families must be paired. Verified success, failure rate, and risk cannot regress; paired-family losses cannot exceed wins; average live latency, actions, and tokens cannot regress by more than 25%.

These thresholds are intentionally minimum operational gates. They are not claims of statistical significance or cross-site reliability.

## Compatibility

- Reads v0.3 `jev-dream/1` and `jev-ultrafast-tcb/0.3`/`0.4` trace events.
- New events are `jev-dream/2` / `jev-ultrafast-tcb/0.5`: post-review hardening unified the tokenizer used for recorded candidate `goal_overlap`, and replay improvement refuses pools that mix TCB generations.
- The v0.3 CLI form `jev-dream EXPERIENCE ...` remains accepted and maps to `jev-dream improve EXPERIENCE ...`.
- Old staged registry entries lack v0.4 lineage/evidence fields and should be restaged under v0.4 before promotion.
