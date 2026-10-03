# Baseline — pre-change qualification snapshot

Frozen before the staged hardening program. Generated from the tree itself;
do not edit by hand in later phases — regenerate.

## Tree identity

| field | value |
| --- | --- |
| package version | `0.9.2` (`pyproject.toml`) |
| TCB version | `jev-ultrafast-tcb/0.16` |
| git commit | `20f01dae45cf3f5ca02970be3886d7d2f34eaeed` (`validation: re-qualify on a775687 with clean tree`) |
| git dirty | `false` |
| tree_digest | `e736f8e31dbb0ff39b5dfa39e82b67eba152c3c69c0bb3a963b742398725a718` |
| lock_digest | `8ee55088451f489c6ff9e9703615c6fbcc86d8673f759454e671335a1f5570ab` |
| manifest_digest | `5029473b95ea727327901ca8325cf93a5ff72a6379ba8ae7594329b10b1050fb` |
| test_suite_digest | `8195d71da6ce6d842011b1a3b6cc5200b577736985e7868996727d561ae9e071` |
| env_digest | `36700f438657bcf6dd48cdfaca44d81b47cb178326ea2fb7073b00670d2be6f8` |
| MANIFEST.sig | **absent** — tree is unsigned at baseline |
| host | Darwin arm64, python 3.12.9, node v24.16.0, **no Chrome/CDP reachable** |

## Pre-change check results (all actually run on this host)

| check | result |
| --- | --- |
| `python -m compileall -q jev_ultrafast tests` | exit 0 |
| `python scripts/update_manifest.py --check` | exit 0 — 78 entries verified |
| `ruff check .` | exit 0 — clean |
| `pytest -q` (committed gate, with coverage) | **446 passed, 0 skipped, 0 failed** in ~16–32 s; coverage **86.73%** (floor 80%) |
| `node --check jev_ultrafast/static/app.js` | exit 0 |
| `node --check jev_ultrafast/snapshot.js` | exit 0 |
| `uv build` | exit 0 |
| `scripts/check_guards.py`, `e2e_check.py`, `race_check.py` | **unavailable on this host** (no CDP endpoint) — not run, not claimed |

## Module map (sizes at baseline)

| file | lines | role |
| --- | --- | --- |
| `jev_ultrafast/dream.py` | 3316 | replay worlds, experience store (hash chain + anchor), replay simulator, promotion/canary/health gates, `PolicyRegistry`, `ExplorationPolicy`, `mutate_policies` |
| `jev_ultrafast/dreamlearn.py` | 2503 | learned priors (`CostModel`, `OutcomeModel`, `ChoiceModel`), `CounterfactualTrials`, `TrialChoiceModel`, `ExperimentScheduler`, `CausalChoicePolicy` |
| `jev_ultrafast/agent.py` | 1336 | agent loop, approval capability, evidence recording |
| `jev_ultrafast/browser.py` | 529 | CDP/browser-harness executor, isolated world, stale/indeterminate outcomes |
| `jev_ultrafast/model.py` | 427 | candidate budgeting, finite-choice backend, validation |
| `jev_ultrafast/dream_cli.py` | 404 | `jev-dream` CLI |
| `jev_ultrafast/policy.py` | 309 | deterministic effect classifier + authority floors |
| `jev_ultrafast/privacy.py` | 251 | bounded/redacted model observation |
| `jev_ultrafast/trace.py` | 220 | hash-chained experience capture |
| `jev_ultrafast/signing.py` | 117 | Ed25519 domains, `EvidenceSigner`, `verify_signature` |

## Call graphs (as-built)

```
TrialChoiceModel.rank(candidates, model_choice, task_family, site, phase)
  └─ _estimate(...) → CounterfactualTrials.resolve(...)
        └─ per (stratum × signature-mask): _contrast(...)
              └─ _arm_entry, _newcombe (report CI), _arm_confidence_bounds
                 + _sequential_alpha (decision CS), Manski delta_bounds,
                 censoring-imbalance gate
  → emits per candidate: {id, kind, expected_delta, utility_delta,
    effect_status, support_sufficient, trial_level, signature_level,
    source="randomized"}   — NO control_p / NO p_progress at baseline

TrialChoiceModel.choose(...)  → same _estimate; beneficial-only; emits
  p_progress = clamp(control_p + delta, 0, 1)   [correct]
TrialChoiceModel.predict(...) → same; p_progress identical formula or None
TrialChoiceModel.refuted(...) → effect_status == "harmful" only

CausalChoicePolicy.rank(candidates, model_choice, ...)
  ├─ trial_model.rank(...)  → causal entries keyed by candidate id
  ├─ choice_model.predict(...) per candidate → observational entry
  ├─ source: pooled_randomized (supported && trial_level=="pooled")
  │          | randomized (causal entry present)
  │          | observational (otherwise)
  ├─ score = w_causal·expected_delta + w_observational·(p_obs − 0.5)
  └─ executable = (mode=="active"
                   && causal.effect_status=="beneficial"
                   && expected_delta > min_causal_delta)
     — NO trial_level restriction; pooled evidence CAN be executable

CausalChoicePolicy.proposal(...) → rank(); shadow→None; active→None unless
  top executable; canary→top entry.
  p_progress = clamp(0.5 + expected_delta, 0, 1)   ← KNOWN DEFECT (phase 1)

ExperimentScheduler.rank(hypotheses, estimates)
  → skips settled (beneficial|harmful); scores importance ×
    uncertainty_width/(1+support) / coverage penalty; deterministic order

CounterfactualTrials.fit(events) → cells (jev-trials/7)
CounterfactualTrials.estimate(context) → per-context arm entries + contrast
CounterfactualTrials.resolve(...) → first support_sufficient contrast over
  strata × signature masks; else most-specific insufficient_data fallback

ExplorationPolicy (frozen dataclass) → mutate_policies(base)
  → fixed scalar perturbations of whitelisted knobs (weights, bonuses,
    quotas, limits, windows, penalties, exponents, caps, min_overlap);
    dedup by behavior_digest

DreamImprover.improve(events, ...)
  → ReplayWorld.from_events → split_worlds (train/validation/holdout)
  → ReplaySimulator.evaluate per split → PromotionGate.assess
  → optional stage via PolicyRegistry.stage (parent-digest bound)

PolicyRegistry (jev-dream/4 state chain + optional Ed25519 state sig +
  external head anchor + promotion/suspend/resume/rollback attestations)
  stage → promote_from_store (CanaryEvidence bound to staged digest) →
  CanaryGate.assess → signed attestation → active; suspend/rollback
  re-verify attestation.
```

## Causal stratum hierarchy (as-built)

Resolution order in `CounterfactualTrials.resolve` (outermost first):

1. `family+site` (only when both are supplied)
2. `site`
3. `family`
4. `pooled` (always last)

Within a stratum, treatment-signature masks back off `full → −phase → −rank
→ −role → −overlap`; `kind` and `effect` are floors and never dropped. The
first stratum with `support_sufficient` (ESS ≥ `MIN_ESS = 8.0`) answers; a
supported-but-unresolved specific stratum still answers (honest "unknown"),
and a thin specific stratum cannot shadow a supported broader one.

Effect statuses: `beneficial` | `harmful` | `unresolved` |
`insufficient_data`. Establishment requires: support on both arms, no
censoring imbalance (`CENSOR_RATE_MAX 0.5`, `CENSOR_RATE_GAP 0.25`),
anytime-valid confidence sequence entirely above `min_effect`
(`MIN_EFFECT = 0.0`, `SEQUENTIAL_ALPHA = 0.05` Bonferroni-split over the
tracked hypothesis family), and the Manski `delta_bounds` on the same side
of the threshold.

## Evidence levels that can become executable today

`CausalChoicePolicy(mode="active")` marks a proposal `executable` when:

- `causal.effect_status == "beneficial"`, and
- `expected_delta > min_causal_delta` (default 0.0).

`support_sufficient` is implied (a `beneficial` status is unreachable
without it). **`trial_level` is not inspected** — an estimate that resolved
at `pooled` activates exactly like `family+site`. The candidate is by
construction in the offered catalogue (entries are built only from offered
candidates). `source` is provenance labeling only; it does not gate.

## Qualification semantics (as-built, `scripts/qualify.py`, schema `jev-qualify/2`)

- `overall`: `failed` if any stage failed; `passed` iff **every** stage is
  `passed`; else `partial`. A skipped stage therefore degrades to `partial`,
  never silently `passed`.
- Exit code: `0` iff `overall == "passed"`, else `1`.
- Bounded vs `--full`: same stages; `--full` swaps volumes (fuzz 50k→1M,
  crash 30→1000, race 200/500→1k/10k, causal grid 2k→100k) **and** requires
  `MANIFEST.sig` — in bounded mode an unsigned tree reports the signature
  check `skipped`; under `--full` it runs `sign_manifest.py --verify`, which
  fails closed when the file is absent/forged/untrusted.
- Manifest: `MANIFEST.sha256` verified by `update_manifest.py --check`
  (byte-for-byte regeneration). `MANIFEST.sig` (`jev-manifest-sig/1`)
  verified by `sign_manifest.py --verify` under pinned
  `JEV_MANIFEST_VERIFY_KEYS`; `_manifest_provenance()` records
  `manifest_signed`, `signature_key_id`, and delegates verification verdict
  to the Q0 stage record.
- Report signing: optional `--sign SEED_HEX` / `JEV_QUALIFY_SIGNING_KEY`
  emits a `jev-qualify-sig/1` block over the canonical report digest
  (domain `jev-dream/qualify-report/v1:`). `--verify-report PATH --key …`
  verifies fail-closed and refuses to run without a pinned key.
- **Recorded gap (Phase 3 target):** a bounded run reports
  `overall: passed` + exit 0 with `manifest_signed: false` and no report
  signature — "passed" is validation status, but nothing distinguishes it
  from release qualification. Even `--full` does not require the report
  itself to be signed/verified: if all stages pass without `--sign`, exit
  is still 0. No `release_qualified` / `release_blockers` fields exist.
- Chrome-dependent checks (Q2, Q3 races) are `skipped` when no CDP endpoint
  answers — reported as absence, never as pass.

## TCB components (security boundary — never self-modifiable)

- `jev_ultrafast/browser.py` + `snapshot.js`: isolated-world
  (`jev-ultrafast-v2`) read/validate/mutate; `StalePage` /
  `IndeterminateMutation` / `BrowserError` outcomes; atomic vs
  trusted-nontransactional guarantees.
- `jev_ultrafast/policy.py`: deterministic monotonic `classify_effect`
  (structural floors; text/labels escalate only) + `DefaultActionPolicy`
  approval boundary + `assess_payload`.
- `jev_ultrafast/agent.py`: the only caller of `act()`; `approve()` issues
  one-shot grants bound to pending action + generated payload digest + page
  fingerprint; pre-mutation guard re-binds the full `ctxOf` authority
  context.
- `jev_ultrafast/dream.py`: `ExperienceStore` (SHA-256 hash chain, cross-
  process file lock, torn-tail truncation, external chain-head anchor),
  `ReplaySimulator`, `PromotionGate`, `CanaryGate`, `HealthGate`,
  `PolicyRegistry` (state-digest chain, optional state signature, anchor,
  attestations).
- `jev_ultrafast/dreamlearn.py`: `CounterfactualTrials` estimator (the
  causal decision channel), `TrialChoiceModel`, `ExperimentScheduler`;
  `CostModel`/`OutcomeModel`/`ChoiceModel` are advisory only.
- `jev_ultrafast/signing.py`: domain-separated Ed25519
  (`evidence-event/v1`, `promotion-attestation/v1`, `chain-head-anchor/v1`,
  `registry-state/v1`, `registry-head-anchor/v1`, `experiment-plan/v1`).
- `scripts/qualify.py`, `scripts/update_manifest.py`,
  `scripts/sign_manifest.py`, `scripts/env_digest.py`: release
  qualification + provenance tooling (self-contained, no
  `jev_ultrafast` imports on the verify path).

## Schemas on disk at baseline

`jev-dream/4` (store/registry; reads /1–/4) · `jev-experiment-plan/1` ·
`jev-trials/7` · `jev-causal/2` · `jev-cost/2` · `jev-outcome/2` ·
`jev-choice/5` · `jev-qualify/2` + `jev-qualify-sig/1` ·
`jev-manifest-sig/1` · `jev-env-digest/1`.

## Known limitations at baseline

1. `CausalChoicePolicy.proposal` reconstructs candidate probability as
   `0.5 + expected_delta` — fabricates a 0.5 control rate (Phase 1).
2. Pooled randomized evidence can mark a proposal `executable` in active
   mode — no context-specific stratum requirement (Phase 2).
3. `overall: passed` is ambiguous with release qualification; report
   signing optional even under `--full` (Phase 3).
4. `dream.py` (3316) and `dreamlearn.py` (2503) are monoliths — large TCB /
   review units (Phase 4).
5. DOM reader scope excludes shadow roots, frames, canvas, uploads,
   pop-ups, nested scrolling, multi-select (Phase 6).
6. `mutate_policies` mutates a fixed scalar list only — the evolvable
   artifact is numbers, not structure (Phases 7–8).
7. Promotion is implicit state in registry records, not an explicit
   constitution/state machine (Phase 9).
8. No Chrome on this host — live guard evidence must come from CI or a
   dedicated automation Chrome.
