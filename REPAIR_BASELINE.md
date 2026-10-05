# Main 18 repair baseline

Frozen before runtime edits on 2026-10-05.

- Supplied archive SHA-256: `c7ae424f8022cc829beff33546373432e1a4b47112c1ee0eb487dbff3a957004`.
  The archive is not present; this value is supplied, not independently verified.
- Independently verified SHA-256 of `MANIFEST.sha256`:
  `ea224ce450b6645ee1b97a8be4c31b49573c790a7d4ddd034b0833f8db5ec15d`.
- TCB version: `jev-ultrafast-tcb/0.18`.
- Package version: `0.9.2`.
- Locked offline suite: **594 passed, 0 failed, 0 skipped**; coverage 86.49%.
- Compile, Ruff, manifest check (108 entries), and TCB check passed.
- Baseline Q4 invocation help passed. The full baseline simulation was not run.

## Current behavior and call paths

`scripts/simulate_causal.py` → `CounterfactualTrials.fit()` → `resolve()` →
`_arm_cs_bounds()`. The generator creates `n_per_arm` candidate observations
and `n_per_arm` control observations irrespective of propensity. Its helper
complements the control propensity; its caller has already complemented it.
Thus its skewed-propensity power assertions do not describe randomized traffic.

Assignments declare class and derived action hypotheses in `fit()`.
`_alpha_allocations()` uses a shared 0.05 budget with inverse-square spending
`0.05 / (zeta(2) * ordinal**2)`. Ordering is the earliest assignment wall-clock
timestamp, assignment digest, then class/action tag. Legacy unordered payloads
receive reconstructed or equal-share allocation.

`action_key()` in `_learning/signatures.py` SHA-256 hashes normalized site,
operation, effect, role, label, and bounded authority-context semantics. It is a
semantic fingerprint, not an operational destination/structure identity, and
its unkeyed low-entropy label hash permits dictionary guessing.

`_arm_cs_bounds()` rescales accumulated IPW sufficient statistics by
`scale = total_weight / analyzed`. It then selects the count epoch
`k = floor(log2(analyzed)) + 1`, spends alpha proportional to `k**-1.5`,
and uses `lambda = sqrt(8 * log(4 / epoch_alpha) / 2**(k-1))`.
The radius is `(lambda * rescaled_weight_square_sum / 8
+ log(4 / epoch_alpha) / lambda) / analyzed`.
Retrospective rescaling does not match the claimed fixed-increment proof.

`resolve()` walks context and signature backoff and reports both class and
exact-action estimates. These adaptive exploratory looks are not a single
preregistered deployment contrast.

## Qualification state

The bundled `VALIDATION.generated.md` reports 587 historical passing tests,
bounded validation, unsigned provenance, and `release_qualified: False`.
It embeds manifest digest
`b3054050c297d3af48cc557e3f6eaa9cd8b41cb9d27e6df7ef23de3332340c6a`,
not this baseline's digest. Its Darwin/Chrome execution is historical evidence,
not evidence from this repair environment.

No signed release, live browser qualification, independent sealed benchmark,
or final archive verification has been performed here. Those are release
blockers, not successful or implicitly skipped acceptance criteria.
