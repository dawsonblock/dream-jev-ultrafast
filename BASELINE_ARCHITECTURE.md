# Baseline Architecture — current-behavior map

Phase-0 snapshot for the repair/hardening/RSI program, taken on the
post-`_dream`/`_learning`-split tree. `BASELINE.md` is the older
pre-hardening freeze (monolithic `dream.py`/`dreamlearn.py` era); this
document describes the tree as it exists **now**. No runtime behavior was
modified to produce it.

## Tree identity

| field | value |
| --- | --- |
| package version | `0.9.2` (`pyproject.toml`, `jev_ultrafast.__version__`) |
| TCB version | `jev-ultrafast-tcb/0.16` |
| git commit | `60f35f0614ff6729c706530c4450517dc1c395b8` (`causal: add hierarchical outcome vectors without weakening the primary endpoint`) |
| git dirty at snapshot | `false` (after ruff --fix on two test-file imports) |
| tree_digest | `37c20a629434956abfca3d3aa7286905e873843ea530a3cfb7c64f9988518632` |
| lock_digest | `8ee55088451f489c6ff9e9703615c6fbcc86d8673f759454e671335a1f5570ab` |
| manifest_digest | `d187368a93432873ef16d86c025a20f8500ebb2c3f6e499132dbfd64c1696cfc` |
| test_suite_digest | `81ecd36211bb433ba4aa173448a1690c7233e0842f3e810f02d366a6c54bd34d` |
| env_digest | `ff98d890b5542bba951688f4a4735fbbec73526dba62c1e67b6ebf75a816f13e` |
| MANIFEST.sig | **absent** — tree is unsigned at baseline |
| host | Darwin arm64, python 3.12.9, node v24.16.0, **no Chrome/CDP reachable** |

## Module map (post-split)

| file | lines | role |
| --- | --- | --- |
| `jev_ultrafast/dream.py` | 42 | compatibility facade over `_dream/` (re-exports + monkeypatch-forwarding) |
| `jev_ultrafast/dreamlearn.py` | 83 | compatibility facade over `_learning/` (same mechanism) |
| `jev_ultrafast/_dream/policy.py` | 239 | `ExplorationPolicy`, `ObjectiveWeights`, `ReplayMetrics`, `mutate_policies` |
| `jev_ultrafast/_dream/replay.py` | 804 | `RecordedTransition`, `ReplayWorld`, `ReplaySimulator`, `split_worlds` |
| `jev_ultrafast/_dream/evidence.py` | 410 | `ExperienceStore` (SHA-256 hash chain, file lock, torn-tail, anchor) |
| `jev_ultrafast/_dream/experiments.py` | — | `jev-experiment-plan/1` digest/signature/authority helpers |
| `jev_ultrafast/_dream/registry.py` | 803 | `PolicyRegistry` — state-digest chain, attestations, stage/promote/suspend/resume/rollback |
| `jev_ultrafast/_dream/canary.py` | 403 | `CanaryEvidence`, `CanaryGate`, `CanaryMetrics`, sign test |
| `jev_ultrafast/_dream/promotion.py` | — | `PromotionGate`, `PromotionDecision` |
| `jev_ultrafast/_dream/health.py` | — | `HealthGate`, `HealthDecision` (success-regression + risk-rate monitors) |
| `jev_ultrafast/_dream/improver.py` | 414 | `DreamImprover`, `DreamReport` — replay-world improvement pipeline |
| `jev_ultrafast/_dream/common.py` | 156 | digests, file lock, `candidate_catalog_digest`, `task_key` |
| `jev_ultrafast/_learning/causal.py` | 1507 | `CounterfactualTrials` (jev-trials/8), `TrialChoiceModel` (jev-causal/2), `UtilityWeights`, outcome vectors |
| `jev_ultrafast/_learning/observational.py` | 566 | `CostModel`, `OutcomeModel`, `ChoiceModel` |
| `jev_ultrafast/_learning/policy.py` | 310 | `CausalChoicePolicy` (shadow/canary/active) |
| `jev_ultrafast/_learning/scheduler.py` | 152 | `ExperimentScheduler` (uncertainty/support priority heuristic) |
| `jev_ultrafast/_learning/signatures.py` | 258 | treatment-signature coordinates, masks, termination reasons |
| `jev_ultrafast/_learning/confidence.py` | 164 | Wilson/Newcombe, anytime-valid CS, sequential alpha spend |
| `jev_ultrafast/_learning/censoring.py` | — | `_run_outcome` terminal classification |
| `jev_ultrafast/agent.py` | 1336 | agent loop, approval capability, experiment assignment, evidence recording |
| `jev_ultrafast/browser.py` | 529 | CDP executor, isolated world, atomic/trusted guarantees |
| `jev_ultrafast/model.py` | 427 | candidate budgeting, finite-choice backend, validation |
| `jev_ultrafast/policy.py` | 309 | deterministic monotonic effect classifier + approval boundary |
| `jev_ultrafast/privacy.py` | 251 | bounded/redacted model observation |
| `jev_ultrafast/trace.py` | 220 | hash-chained live experience capture |
| `jev_ultrafast/signing.py` | — | domain-separated Ed25519 |
| `jev_ultrafast/__init__.py` | 74 | **eager** re-exports (imports `agent`→`browser` unconditionally — Phase 10 target) |

## Call graphs (as-built, post-split)

```
TrialChoiceModel.rank(candidates, model_choice, task_family, site, phase)
  └─ _estimate(...) → CounterfactualTrials.resolve(...)
        └─ per (stratum × signature-mask): _contrast(...)
              └─ _arm_entry, _newcombe (report CI), _arm_confidence_bounds
                 + _sequential_alpha (decision CS), Manski delta_bounds,
                 censoring-imbalance gate, secondary_delta, safety block
  → emits per candidate: {id, kind, expected_delta, control_p,
    p_progress (= clamp(control_p+delta)), uncertainty, delta_ci,
    utility_delta, effect_status, support_sufficient, trial_level,
    signature_level, safety_regression, secondary_delta,
    outcome_vector, source="randomized"}

TrialChoiceModel.choose(...)  → same _estimate; beneficial-only;
  _candidate_probability → (control_p, p_progress)
TrialChoiceModel.predict(...) → same machinery, single-action form
TrialChoiceModel.refuted(...) → effect_status == "harmful" only

CausalChoicePolicy.rank(candidates, model_choice, ...)
  ├─ trial_model.rank(...)  → causal entries keyed by candidate id
  ├─ choice_model.predict(...) per candidate → observational entry
  ├─ source: pooled_randomized (supported && pooled)
  │          | randomized (causal entry present)
  │          | observational
  ├─ score = w_causal·expected_delta + w_observational·(p_obs − 0.5)
  │   [single mixed score — Phase 6 splits it]
  ├─ _execution_blocker: no_causal_evidence | harmful |
  │   insufficient_support | unresolved | below_effect_threshold |
  │   missing_probability | safety_regression | pooled_only |
  │   unsupported_stratum
  └─ executable = mode=="active" && blocker is None

CausalChoicePolicy.proposal(...) → rank(); shadow→None; active→top iff
  executable; canary→first non-harmful/non-safety-regressing entry.
  p_progress = causal_p when supported causal else observational —
  control_p+delta propagated, never 0.5+delta (Phase 1 DONE)

ExperimentScheduler.rank(hypotheses, estimates)
  → skips settled (beneficial|harmful); score =
    importance × uncertainty_width/(1+support_fraction) /
    (1+coverage/coverage_scale); deterministic order.
    [Heuristic, not formal VOI — Phase 14 renames]

CounterfactualTrials.fit(events)
  → groups events by run_id; per run: dedup identical
    (experiment_id, proposal_id, model_choice_id, arm) assignment records
    (duplicates counted, NOT rejected) — EVERY distinct assignment record
    is folded into cells (Phase 4 target: ≥2 distinct assignments/run
    must invalidate the unit); runless ("loose") events recorded as
    censored unknown_abort with dedup; cells = jev-trials/8

DreamImprover.improve(events, ...)
  → ReplayWorld.from_events → split_worlds (train/validation/holdout,
    family-disjoint) → ReplaySimulator.evaluate per split →
    PromotionGate.assess → stage via PolicyRegistry (parent-digest bound)

PolicyRegistry (jev-dream/4 state chain + Ed25519 state sig + external
  head anchor + stage/promotion/suspend/resume/rollback attestations)
  stage → promote_from_store (CanaryEvidence bound to staged digest) →
  CanaryGate.assess → signed attestation → active; suspend/rollback
  re-verify attestation. Lifecycle is implicit in records — no explicit
  PROPOSED→…→ACTIVE state machine (Phase 20 target).
```

## Causal stratum hierarchy (as-built)

`CounterfactualTrials.resolve` walks (outermost first):

1. `family+site` (only when both supplied)
2. `site`
3. `family`
4. `pooled`

Within a stratum, signature masks back off `full → −phase → −rank →
−role → −overlap`; `kind` and `effect` are floors, never dropped. First
stratum with `support_sufficient` (ESS ≥ 8.0) answers; a supported
specific stratum answers even when unresolved; a thin specific stratum
cannot shadow a supported broader one.

Effect statuses: `beneficial` | `harmful` | `unresolved` |
`insufficient_data`. Establishment requires both-arm support, no
censoring imbalance (`CENSOR_RATE_MAX 0.5`, `CENSOR_RATE_GAP 0.25`),
anytime-valid CS entirely past `min_effect` (`MIN_EFFECT 0.0`,
`SEQUENTIAL_ALPHA 0.05` Bonferroni-split over `hypothesis_count` =
distinct cell contexts — *implicit* multiplicity accounting; Phase 5
makes it an explicit registered family), and Manski `delta_bounds`
agreeing.

## Evidence ladder (as-built)

| level | what it may do today |
| --- | --- |
| pooled randomized | annotate, schedule experiments, **nominate canaries**; CANNOT execute in `active` (`pooled_only` blocker — Phase 2 DONE) |
| context-specific randomized (`family+site`/`site`/`family`) | shadow, canary, active **at treatment-class granularity** — no exact-action check (Phase 3 target) |
| observational | annotate + score contribution only |
| none | nothing |

`active_trial_levels` defaults to `("family+site","site","family")`;
unknown level names fail closed at construction.

## Qualification semantics (as-built, `scripts/qualify.py`, `jev-qualify/3`)

- `validation_status`: `failed` if any stage failed; `passed` iff every
  stage `passed`; else `partial`.
- `provenance_status`: `unsigned` | `verified` | `failed` (manifest sig
  + Q0 verdict).
- `release_qualified`: always `false` in bounded mode; under `--full`
  requires signed+verified manifest under pinned keys, every check run
  and passed (any `skipped`/`failed` check is a blocker), report signed
  under an accepted key. `release_blockers` names each failure.
- Exit: `0` iff `release_qualified` under `--full`; bounded exits on
  `validation_status == "passed"`.
- Checks are implicitly all-mandatory — no `mandatory`/`optional`/
  `environment-dependent` declarations (Phase 9 target). Chrome-bound
  checks skip honestly when no CDP endpoint answers.

## TCB components (security boundary — never self-modifiable)

- `jev_ultrafast/browser.py` + `snapshot.js` — isolated world, stale /
  indeterminate outcomes, atomic vs trusted-nontransactional.
- `jev_ultrafast/policy.py` — `classify_effect`, `DefaultActionPolicy`,
  `assess_payload`.
- `jev_ultrafast/agent.py` — only caller of `act()`; `approve()` one-shot
  grants bound to pending action + payload digest + page fingerprint.
- `jev_ultrafast/_dream/` — `ExperienceStore`, `ReplaySimulator`,
  `PromotionGate`, `CanaryGate`, `HealthGate`, `PolicyRegistry`,
  experiment-plan binding.
- `jev_ultrafast/_learning/` — `CounterfactualTrials`,
  `TrialChoiceModel`, `ExperimentScheduler`; `CostModel`/`OutcomeModel`/
  `ChoiceModel` advisory only.
- `jev_ultrafast/signing.py` — Ed25519 domains.
- `scripts/qualify.py`, `update_manifest.py`, `sign_manifest.py`,
  `env_digest.py` — release qualification + provenance.

## Schemas on disk at baseline

`jev-dream/4` · `jev-experiment-plan/1` · `jev-trials/8` ·
`jev-causal/2` · `jev-cost/2` · `jev-outcome/2` · `jev-choice/5` ·
`jev-qualify/3` + `jev-qualify-sig/1` · `jev-manifest-sig/1` ·
`jev-env-digest/1` · `jev-outcome-vector/1`.

## Phase status at baseline (verified against source)

| plan phase | status |
| --- | --- |
| 0 baseline | this document (supersedes `BASELINE.md` scope) |
| 1 causal probability propagation | **done** (commit `1e0f26e` + tests) |
| 2 pooled ≠ active | **done** (commit `9f0dd15` + tests) |
| 3 exact-action generalization | **not done** — class evidence can activate a never-randomized action |
| 4 one-assignment-per-unit | **not done** — distinct assignments all fold into cells |
| 5 hypothesis registration | **not done** — multiplicity is implicit cell-count Bonferroni |
| 6 separate causal/observational scores | **not done** — single mixed `score` |
| 7 provenance dimensions | **partial** — `source` label only; pooled+insufficient mislabels `randomized` |
| 8 release qualification | **done** (commit `3a4f6ad`: validation/provenance/release_qualified/blockers) |
| 9 mandatory/optional checks | **partial** — all checks implicitly mandatory; no declarations |
| 10 lazy browser decoupling | **not done** — `__init__.py` eagerly imports `agent` |
| 11 security profiles | **not done** |
| 12 module split | **done** (commit `4345907`) |
| 13 replay categories | **not done** — `coverage_miss` is only a penalty weight |
| 14 scheduler terminology | **not done** — heuristic named "information gain" in docs |
| 15 hierarchical endpoints | **done** (commit `60f35f0`: jev-outcome-vector/1) |
| 16 browser expansion | **not done** — no iframe/shadow/tab/scroll-kb/upload support |
| 17 TCB manifest | **not done** |
| 18 policy DSL | **not done** |
| 19 structural mutation | **not done** — scalar perturbations only |
| 20 promotion state machine | **partial** — implicit lifecycle in registry records |
| 21 rollback/health monitoring | **partial** — HealthGate exists; crash matrix thin |
| 22 version hygiene | **not done** — CHANGELOG reuses `v0.9.2` ×4 |
| 23 adversarial scientific qual. | partial (existing causal validity tests) |
| 24 adversarial security qual. | partial (authority tests) |
| 25 final release qualification | not done |

## Known limitations at baseline

1. Class-level causal evidence can mark a never-randomized action
   executable (Phase 3).
2. `CounterfactualTrials.fit` accepts ≥2 distinct assignments per run
   without rejection (Phase 4).
3. No hypothesis pre-registration; multiplicity is inferred from cells
   (Phase 5).
4. Single mixed score couples causal and observational channels in
   every mode (Phase 6).
5. Provenance is a single label; pooled+insufficient entries label
   `randomized` (Phase 7).
6. `import jev_ultrafast.dreamlearn` pulls `agent`/`browser` via
   `__init__.py` (Phase 10).
7. No explicit mandatory/optional check taxonomy (Phase 9); no security
   profiles (Phase 11); no TCB manifest (Phase 17); no replay verdict
   categories (Phase 13); scheduler heuristic mislabeled (Phase 14).
8. `CHANGELOG.md` reuses `v0.9.2` for four different sections (Phase
   22).
9. No Chrome on this host — live guard evidence must come from CI or a
   dedicated automation Chrome.
