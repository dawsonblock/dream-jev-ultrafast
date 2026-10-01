# Validation — Jev Ultrafast v0.9.0 DREAM-Jev

Validation date: 2026-09-30, updated for the v0.7.1 hardening pass, the v0.8.0 audit-correction pass, the v0.8.1 stability patch, the v0.8.2 causal-integrity pass, and the v0.8.3 causal-model/hierarchical-context pass (v0.5.1 applied the independent-audit authority patch; v0.6.0 added the `ChoiceModel` counterfactual layer; v0.7.0 added real randomized counterfactual trials; v0.7.1 fixed trial-approval resume and registry lineage; v0.8.0 restores the v0.6.2 hardening this line had dropped — trajectory-success labeling, honest propensity fields, the registry head anchor, pre-authority assignment recording, and the run-level trial endpoint — and wires stamped `experiment_proposals` into the agent; v0.8.1 fixes select-option target-key stability, OpenAI-compatible server compatibility for the text helper, and adds a local Ollama decision backend for key-free runs; v0.8.2 makes trial analysis intention-to-treat over assignment events, makes stamped plans genuinely immutable and bound, censors unmeasured run outcomes in learned models, and hardens the local backend request path; v0.8.3 adds the separate causal `TrialChoiceModel` decision prior and hierarchical task-family/site context across `ChoiceModel` and `CounterfactualTrials`; v0.9.0 makes the causal layer validity-first — support vs. effect certainty, sequentially valid establishment, structured censoring with an imbalance gate, true family/site coordinates with a bounded treatment signature, bound and optionally signed experiment plans, a hypothesis scheduler, multi-objective utility, and the operator-gated `CausalChoicePolicy`).

## Reproduced in this build environment

- `uv run pytest`: **361 passed** against the project environment (including the new `tests/test_causal_validity.py` adversarial suite); browser/CDP calls in unit tests remain mocked by the tests themselves.
- `uv run python scripts/check_guards.py` against a dedicated headless Chrome 154 instance (`BU_CDP_URL=http://127.0.0.1:9222`, throwaway `--user-data-dir`): **all 34 live browser guard checks passed** — re-run at v0.9.0 (Chrome 154.0.8037.59) after the structured-abort changes to `browser.py`/`model.py` — including the adversarial mid-input cases (post-observe commit swap, same-label node swap between press and release, pre-release overlay, pre-release geometry drift, mid-press page mutation, pointer-state hygiene after abort, nested-descendant interception, post-observe hidden/detached targets, structured `ctx` propagation) and the atomic-guarantee check proving `DOM_ATOMIC` clicks dispatch no press/release window for mid-event sabotage. Earlier validation against Chrome 152 also passed; the first v0.4.0 live run had exposed a latent defect that mocked tests could not see, which is why CI now runs this suite in a `live-guards` job on every change.
- `uv run ruff check .`: passed.
- `uv build`: passed.
- `python -m compileall -q jev_ultrafast tests examples scripts`: passed.
- `node --check jev_ultrafast/snapshot.js`: passed.
- `node --check jev_ultrafast/static/app.js`: passed.
- `shasum -a 256 -c MANIFEST.sha256`: all tracked files verify.
- TOML parse for `pyproject.toml`: passed.
- JSON parse for bundled JSON evidence files: passed.
- DREAM CLI smoke: `verify` and `improve --stage` completed against synthetic hash-chained evidence; the staged record bound the parent policy, world-pool digest, split-manifest digest, evidence head, and TCB versions.
- Cross-process serialization tests: four spawned processes appended 40 events to one JSONL store and the resulting chain verified; three spawned processes ran 30 serialized suspend/resume registry writes without lost updates.

## Remaining environment limitation

- The live guard suite ran against a dedicated throwaway Chrome profile on this machine. Re-run `scripts/check_guards.py` on each deployment machine with its intended Chrome/Browser Harness installation before unattended use.

## Documented decisions

- `candidate_since_ms` uses `>=` when binding canary evidence to `staged_at_ms`: a candidate run finishing in the same millisecond as staging is admissible, and a stricter `>` would silently drop valid evidence.
- `ExperienceStore(strict=False)` remains the default; the O(n) full-chain verify is opt-in for qualification-critical stores.
- `demo.py` binds `127.0.0.1` only, with token, Host, and Origin checks; it is a local inspector by design.

## New v0.4 regression coverage

The test suite now covers the v0.3 contracts plus:

- cross-process serialization of hash-chained experience-store appends;
- refusal to extend a store whose tail event is corrupted;
- backward reading of v0.3 `jev-dream/1` / TCB 0.3 events;
- candidate-catalogue digests and selected-action rank evidence;
- replay report binding to world-pool, split-manifest, evidence-head, and TCB-version evidence;
- train/validation/holdout-aware candidate selection rather than training-winner-only selection;
- unbound canary-metric promotion disabled by default;
- matched canary evidence tied to the event-store head;
- default paired task-family canary requirements;
- live latency and token regression rejection;
- stale replay-parent rejection in the policy registry;
- bound promotion from hash-verified canary runs;
- post-promotion health drift detection and policy suspension;
- suspension fallback to baseline policy;
- policy-registry history and rollback;
- end-to-end Agent → trace → replay verified-success integration with a fake browser boundary;
- shared Unicode tokenization across live candidate scoring, policy scoring, and recorded goal overlap;
- expanded sensitive URL-query redaction (api/access/refresh/id tokens, client secrets, session ids, CSRF/JWT/OTP names);
- zero task-family canary candidates rejected by the default gate;
- canary candidate evidence bound to runs recorded after staging;
- canary evidence baseline digest pinned to the staged parent digest;
- reserved experience-store and trace event keys rejected;
- provably pre-mutation select errors retried via re-observation while interrupted select evaluation remains non-retryable;
- signature-based verifier dispatch so internal verifier TypeErrors are never masked;
- explicit `aborted` run status excluded from canary outcome metrics;
- trace candidate catalogues computed only when a DREAM recorder is active;
- cross-process serialization of policy-registry stage/promote/suspend/resume/rollback writes;
- canary risk comparison per task (rate) as well as absolute count;
- optional `strict` experience stores that verify the full hash chain before every append;
- replay improvement refuses pools that mix DREAM TCB generations (pre-unification traces recorded `goal_overlap` with the legacy tokenizer);
- health decisions mark insufficient observed coverage (`sufficient=false`) separately from drift outcomes;
- promotion requires non-empty bound evidence digests in every path;
- isolated-world `(fn)(arg)` expression templates verified to invoke their argument;
- `fresh()` reports stale rather than raising when the guard context is destroyed or unreachable;
- `CanaryMetrics.from_events` treats a non-positive `max_runs` as "no runs" rather than slicing to the whole history;
- the select execution guard normalizes option values exactly as `snapshot.js` records them;
- registry writes are fsynced before the atomic rename; StalePage messages carry the underlying JS error;
- `act()` rejects caller-supplied `approved` flags; approval is a one-shot server-side grant bound to the pending action id and page fingerprint, consumed by exactly one `act()`;
- action-history `text`/`action` values are redacted at every external model boundary (decision and text-helper contexts);
- transition events record the pre-policy observed catalogue plus an `offered_digest`/`offered_count` of the post-policy catalogue, verified against the recorded policy when replay worlds are built;
- `ExplorationPolicy.behavior_digest` deduplicates behavior-identical candidates while `digest` keeps name/version lineage binding;
- `mutate_policies()` perturbs every knob bidirectionally inside the `ExplorationPolicy` envelopes (no `+0` no-op variants remain);
- `task_family`/`instance_id` run metadata groups splits by family and pairs canary evidence by instance, with an exact two-sided sign-test p-value and an optional `max_pair_sign_p` gate plus `--baseline-since-ms` matched-time windows;
- registry atomic renames and first experience-store writes fsync the containing directory;
- `CostModel` linear token/latency estimates anchored to recorded transition values — reported as `estimated_*` metrics that prioritize among replay-passing candidates and are tested never to flip a gate;
- `OutcomeModel` bucketed Beta-smoothed `P(page_changed)` priors annotating candidates (`predicted_page_changes_per_world`) without entering any gate;
- the bounded learnable surface extended to `overlap_exponent`, `duplicate_node_cap`, and `min_goal_overlap`, all still replay-qualifiable candidate-allocation knobs; live `candidate_actions` and replay `_retained_candidates` are tested to produce identical catalogues under each knob;
- Ed25519 evidence signatures on `event_hash` via injectable signers (`JEV_EVIDENCE_SIGNING_KEY`), verified on load under `JEV_EVIDENCE_VERIFY_KEY`/`--verify-key` with fail-closed wrong-key, forged-signature, and unsigned-event (`require_signatures`) rejection;
- new events are `jev-dream/3` / `jev-ultrafast-tcb/0.6`; pre-0.6 stores remain readable and mixed-TCB replay pools are still rejected.

## v0.5 authority-integrity coverage

- deterministic `Effect` classification matrix: scroll/wait→OBSERVE, fill/select/editor roles/toggles→FORM_EDIT, tabs→NAVIGATE, links→NAVIGATE (including auth-labelled links), bare "Search"/"Go"→SEARCH, bare "Continue"/"Yes"/unknown labels→UNKNOWN_COMMIT, label-escalated PURCHASE/FINANCIAL/DELETE/EXTERNAL_MESSAGE/PERMISSION_CHANGE on any role;
- structural `ctx` classification: form submit membership with password→AUTHENTICATE, money→PURCHASE, file→SUBMISSION, GET/search-field→SEARCH, generic→SUBMISSION; messaging/download/external destinations; modal scope (dismiss autonomous, everything else UNKNOWN_COMMIT);
- high-risk labels escalate links ("Delete account" link→DELETE) and submits ("Confirm" in money form→PURCHASE); absence of a match never lowers the structural floor;
- legacy text-pattern backstop preserved (`place order` → require_approval without ctx);
- adviser hook escalation to require_approval/deny and impossibility of downgrading a deterministic floor;
- TOCTOU: pre-press identity failure dispatches no input; pre-release failure still dispatches `mouseReleased` for pointer hygiene then raises `StalePage`; atomic fill aborts before mutation on focus failure, errors non-retryably when the landed value differs, and performs zero `Input.*` text/key dispatches on the happy path;
- torn-tail recovery: incomplete `{"...` final fragment is excluded from `load()`, reported via `verify()["torn_tail_recovered"]`, and truncated by the next `append()`; garbage tails, non-object tails, mid-chain corruption, and hash-invalid complete records stay fail-closed; a complete record missing only its newline is sealed rather than truncated;
- domain-separated signatures (`jev-dream/evidence-event/v1:<digest>`): a raw-digest Ed25519 signature is not accepted as evidence;
- verification-key rotation: events signed under old and new keys load under the rotated key set and fail closed under a partial set;
- new events are `jev-dream/3` / `jev-ultrafast-tcb/0.7`; signatures written before the domain-separation change will not verify under this frame.

## v0.5.1 audit-patch coverage

- **Monotonic effect classification**: the audit's case — `fill` "Credit card number" on a `ctx.fields.money` form — no longer short-circuits to `FORM_EDIT`; per-target `ctx.field` (auth/otp/money/file/email/tel/message) escalates fill/select edits to `AUTHENTICATE`/`FINANCIAL`/`SUBMISSION`/`DISCLOSURE`/`EXTERNAL_MESSAGE`, and high-risk labels evaluate before every structural shortcut including toggles ("Share publicly" switch → `EXTERNAL_MESSAGE`).
- **Canary pair namespacing**: `pair_key = task_family\x00instance_id`; identical `instance_id` values across families never pair, equal pair keys with inconsistent family metadata raise, and `paired_task_families` counts namespaced families.
- **Signed promotion attestation**: `_promote_bound` attaches a `jev-dream/promotion-attestation/v1:`-framed signature binding candidate/behavior/parent digests, replay-report/world-pool/split digests, canary evidence digest, chain head, and registry revision; `active_policy()` and registry load fail closed on missing/forged/unexpected-key attestations when verification keys are configured; a signature valid under the evidence frame does not verify under the attestation frame; rollback re-verifies the original attestation and signs a `rollback_attestation` for the transition.
- **Chain-head anchoring**: `--anchor PATH` checkpoints the signed head after every append; reads *and* appends fail closed when the log no longer reaches the anchored head — covering both plain tail truncation and the signed-event-disguised-as-torn-fragment attack; missing anchors on non-empty stores fail closed; forged/re-keyed anchors fail closed; `reanchor()` is the explicit operator resolution.
- **Privacy**: `sanitize_action` redacts `label` and `option_label` (email/card/secret patterns) in addition to values; structural `ctx` survives for authority classification, which still sees the original action.
- **Catalogue digest completeness**: `candidate_catalog_digest` binds `node`, `role`, `option_index`, `option_label`, `current_value`, `checked`, and `ctx` alongside id/kind/label/value/goal_overlap — a mutation test flips each field and asserts the digest changes.
- **Execution guarantees**: `DOM_ATOMIC` (single isolated-world validate+mutate turn, default for click/fill/select) versus `TRUSTED_INPUT_NONTRANSACTIONAL` (CDP mouse input with pre-press/pre-release checks; `unsupported` programmatic elements escalate to it only because the atomic evaluation provably did not mutate); mocked tests assert the atomic path dispatches zero `Input.*` events and a single evaluation.
- New events are `jev-dream/3` / `jev-ultrafast-tcb/0.8`.

## v0.6.0 counterfactual-layer coverage

- `ChoiceModel` (Level 3) fits the same `(kind, overlap, rank)` cells as `OutcomeModel` but returns posterior standard deviation and an explicit `confident` flag — sparse cells and the global fallback abstain rather than guessing.
- During replay it is queried under the *candidate* policy's recomputed offered rank, so a filtering policy provably changes which cell the recorded action lands in — the signal is policy-dependent, not a descriptive prior. (v0.6.0 recorded `selected_rank` on the *observed* basis — a different coordinate — which is why v0.6.1 below splits the two ranks explicitly.)
- `choose()` proposes only from the offered catalogue using an upper-confidence score (`mean + 0.5·std`); tests cover abstention on empty/sparse fits, proposals restricted to observed candidates, deterministic digest/round-trip, and UCB preferring an unexplored action while flagging it not-confident.
- Counterfactual proposals are annotation-only (`choice_model` block: `divergence_rate`, `mean_uncertainty`, `confident_fraction`, `predicted/proposed_page_changes_per_step`); tests assert identical `ReplayMetrics`, promotion decisions, and selected digests with and without the model — a proposal can never open a gate or become evidence.

## v0.6.1 correction-pass coverage

- **Dual-rank evidence**: transition events record `selected_observed_rank` and `selected_offered_rank` as separate coordinates (the v0.6 `selected_rank` was the observed index and is never read as an offered rank); `ReplayWorld.from_events()` re-derives the offered catalogue under the recorded policy and fails closed on offered-digest, `offered_count`, or recorded offered-rank mismatches; learned models fit on the offered coordinate only, and a "recorded" catalogue resolves its offered rank to the observed index.
- **ChoiceModel objective**: the learned target is the run's final independently verified outcome applied to *every* step of its trajectory (v0.6.2 semantics, restored in v0.8.0) — a page change inside a failed run teaches the prior nothing — and censored runs (aborted/interrupted/unverifiable) drop out of the fit entirely; tests cover blocked and unverified terminal runs scoring below the Beta prior floor.
- **Split reporting**: `choice_model` entries keep the two subjects separate — `selected_mean_uncertainty`/`selected_confident_fraction` describe the recorded action's prior, `proposal_mean_uncertainty`/`proposal_confident_fraction` the counterfactual proposal's (`mean_uncertainty`/`confident_fraction` remain as selected-action aliases), and the predicted/proposed magnitudes are renamed `*_progress_per_step` to match the verified-progress target.
- **Behavior propensities**: transitions record `operation_probability`, `target_probability`, `decision_confidence`, and the offered `action_probabilities`/`operation_probabilities` distributions. (v0.7.x also recorded the selected head probability as `selected_propensity`; v0.8.0 corrects the semantics — see below.)
- **Fail-closed kinds**: `DefaultActionPolicy.assess()` no longer short-circuits unrecognized kinds to `allow`; unknown mutation primitives classify `UNKNOWN_COMMIT` → `require_approval` (scroll/wait remain the only autonomous non-mutations).
- **Payload authority**: after `field_text()` returns, `assess_payload(text)` classifies the concrete value being disclosed — card numbers → FINANCIAL, emails/secrets/identity numbers → DISCLOSURE — and the stricter of target effect and payload effect applies before the browser mutation; a field that describes itself as plain text cannot launder a sensitive generated value.
- **Full attestation binding**: under verification keys the promotion attestation must reproduce every signed field from the record — parent digest, replay digests, `canary_digest` (which covers the health-check reference metrics `health_from_store` trusts), `promoted_at_ms`, `promotion_revision` — not only the candidate digest; a valid signature over a tampered record fails closed.
- **Signed registry state**: every registry write chains `prev_state_digest → state_digest` over the entire payload and, under a signer, signs the head under `jev-dream/registry-state/v1:`; suspend/resume/rollback attach per-record transition attestations chaining to the replaced head; a signed state read with no keys, an unsigned state read with keys, and any post-write field edit all fail closed. A wholesale restore of an older signed file remains the documented residual — the state head is authenticity, not freshness — the externally anchored registry head that defeats a wholesale snapshot restore ships in v0.8.0 (`PolicyRegistry(anchor_path=...)`, `registry-reanchor`).
- **CostModel identifiability**: `reliable` requires ≥8 samples *and* ≥2 distinct offered counts spanning ≥4 — constant-count evidence cannot look reliable.
- New events are `jev-dream/4` / `jev-ultrafast-tcb/0.9`. Signed registries written before v0.6.1 fail closed at the first keyed read — the attestation now binds `promotion_revision`/`canary_digest`, which old records never stored; rebuild them by re-promoting under the new schema.

## v0.7.0 counterfactual-experiment coverage

- **Proposal extraction**: divergent `ChoiceModel` annotations on baseline replay worlds are emitted as `DreamReport.experiment_proposals` — stamped hypotheses with expected delta, world/step/action provenance, and the model-predicted counterfactual; they remain annotation, never gate input or evidence.
- **Payload-bound approval**: `pending_approval`/`granted_approval` now carry `payload_digest` over the exact generated text — a grant is consumed once and an approved fill cannot silently mutate into a different payload; approving the field approves the *value*, not the field type.
- **Registry downgrade fail-closed**: a `jev-dream/4` registry file missing its state-head material (`state_digest`/`prev_state_digest`/`state_signature`) fails closed on load instead of falling back to the unsigned legacy path — mutable fields like `suspended` can no longer be unprotected by stripping the head. When a signer or verification keys are configured, the head must also verify.
- **Real trials, one per run**: `Agent(experiment={"model", "rate", "rng"})` executes a divergent proposal (candidate arm) or the model's own choice (control arm) with a recorded `assignment_probability`, at most one deviation per run so outcomes stay attributable; the deviated action passes through the same policy/approval/payload checks as any selection — the experiment hook cannot reach around the authority plane.
- **Canary exclusion**: transitions carrying an `experiment` block are excluded from `CanaryEvidence` — a trial never counts toward its own promotion; qualification still requires the replay + bound live-canary path.
- **CounterfactualTrials**: `CounterfactualTrials.fit` produces self-normalized IPW estimates of verified progress per arm with `effective_sample_size`, so thin propensity support surfaces as thin rather than confident; terminal labels keep the verified-progress semantics (terminal transition positive only on `done` + verifier pass).
- New transitions are `jev-ultrafast-tcb/0.10`: evidence pools that predate experiment semantics never silently mix with trial runs.

## v0.7.1 trial-integrity fixes

- **Approval resumes the approved action**: `pending_approval` now carries the deviated action and its experiment metadata, and `approve()` rewrites the resumed decision to the approved action id — approving a candidate-arm proposal can no longer silently execute the model's original choice with `experiment: null`.
- **Registry lineage cannot be schema-laundered**: a missing state head fails closed whenever `verify_keys` or a `signer` is configured, so flipping `schema` to `jev-dream/3` no longer downgrades a stripped signed registry into a "legacy" file whose mutable `suspended` flag rides under a still-valid attestation.
- **Trial contexts key on the model choice**: `CounterfactualTrials` buckets both arms of one divergence by the model choice's `(kind, overlap)` — not the executed action's — and `estimate()` reports `delta_reliable` only when both arms clear the ESS floor.
- **Live proposals see real features**: offered candidates are annotated with `goal_overlap` before `ChoiceModel.choose`, matching the training basis; `experiment` config is validated in `__init__` before the browser launches.

## v0.8.0 audit-correction pass

The v0.7 experiment path is kept but re-hardened against the audit findings; several v0.6.2 fixes that had silently regressed are restored.

- **Offered-catalogue boundary enforced**: a trial proposal must be a member of `offered_now` — the post-policy candidate set the model actually saw — not merely present in `page["actions"]`. A custom proposal model can no longer escape the bounded candidate surface.
- **Immutable assignment with full provenance**: the frozen assignment binds `proposal_digest`, `stamped_digest` (when executing a stamped report plan), `choice_model_digest`, `policy_behavior_digest`, `offered_catalogue_digest`, `proposal_p_progress`, `proposal_uncertainty`, `expected_delta`, and `experiment_id`. Approval/resume consumes the frozen assignment rather than re-deriving it.
- **`experiment_assigned` at randomization**: assignment is evidence before the authority plane — a denied, rejected, stale, or aborted trial still marks the run experimental, and canary exclusion keys on the assignment record, not on a surviving transition.
- **Run-level verified endpoint**: `CounterfactualTrials` estimates `p_success` — the run's final independently verified outcome — as the primary endpoint (jev-trials/2). `p_page_changed` survives only as secondary telemetry; a trial that moved the page inside a failed run is a failure, fixing the audit's exact-reverse estimate.
- **Trajectory-success labeling restored (jev-choice/3)**: `ChoiceModel` again labels every step of a run by the run's final verified outcome — correlation, not per-action causal credit — so page churn inside failed runs teaches the prior nothing. Ten failed runs of page-changing actions now yield `p_progress` near the Beta floor instead of 0.79.
- **Honest propensity fields**: deterministic argmax selection records `selection_mode: "argmax"`, `behavior_propensity: null`, and model scores under `selected_model_score`/`joint_model_score` — a head probability is never again weightable as a sampling propensity. Trial steps record the real propensity: the scheduler's `assignment_probability`.
- **Stamped plans executable**: `Agent(experiment={"proposals": [...], "rate": ε})` consumes `DreamReport.experiment_proposals` directly, matched to the live state by fingerprint — the stamped hypothesis is the artifact that runs, closing the report→agent disconnect.
- **Registry head anchor restored**: `PolicyRegistry(anchor_path=...)` checkpoints the signed state head in an external file (`jev-dream/registry-head-anchor/v1:`-signed when a signer is configured); every load verifies the anchor agrees — a wholesale restore of an older complete signed file fails closed, and `reanchor()`/`jev-dream registry-reanchor` is the explicit operator repair. Deleting the registry under a live anchor fails closed too.
- **Approval UI shows the payload**: `pending_approval.payload_preview` carries the exact generated text so the operator approves the value the digest binds, and the demo status line renders it.
- **Release integrity**: `MANIFEST.sha256` is regenerated and CI verifies it (`sha256sum -c MANIFEST.sha256`), closing the stale-manifest gap.
- New transitions are `jev-ultrafast-tcb/0.11`: pools that predate assignment-event semantics never silently mix with trial evidence.

## v0.8.1 stability patch

- **Select target keys are now stable identities**: option sub-targets previously numbered positionally over the filtered option list, and the snapshot omits the currently-selected option — so the same `N:M` key silently re-mapped to a different option after every selection, which produced an observe→select→observe toggle loop. Keys now encode the DOM `option_index`; `test_select_target_keys_are_stable_across_selection_change` covers it.
- **Text helper compatibility**: `TEXT_MODEL_REASONING=none` now omits the reasoning field entirely instead of sending `{"reasoning": {"enabled": false}}` — strict OpenAI-compatible servers (Ollama `/v1`) reject the key outright. Field generation also pins `temperature: 0.2` for consistency, and the shared model client honors `JEV_MODEL_TIMEOUT` (default 25s) since local inference legitimately exceeds it.
- **Local decision backend**: `scripts/local_backend.py` serves the finite-choice decision protocol on loopback backed by an Ollama model (default `qwen3:8b`), enabling fully key-free runs — decisions, text, and browser all local. Grounded operation prompts preview what each operation would act on, satisfied controls are surfaced via current-value/goal-token overlap, and the shim fails honestly rather than fabricating a choice when the model's answer is invalid. Verified end-to-end against the bundled fixture: real Chrome execution with all authority-plane checks intact.

## v0.8.2 causal-integrity pass

The remaining audit findings were about the *statistics* of the experiment layer, not browser authority — this pass closes them without touching the authority plane.

- **Intention-to-treat trial analysis (jev-trials/3)**: `CounterfactualTrials.fit` now groups runs by their `experiment_assigned` events — the randomized assignment is the unit of analysis. An assignment denied by policy, vetoed at approval, or never reached by the browser still enters its arm's estimate under the run's real terminal outcome, eliminating post-randomization selection (the audit's repro: 4 candidate assignments vetoed pre-execution vanished from the estimator; they now count). Per assignment the report carries `assigned`/`trials`/`executed`/`censored`, the IPW `p_success`, a Wilson `p_success_ci`, effective sample size, and `reliable` against a raised `MIN_ESS = 8` floor; `delta`/`delta_ci`/`delta_reliable` appear only where both arms have evidence. Legacy pools without assignment events fall back to analyzing experiment-tagged transitions; pools with assignments never mix the two framings.
- **Censored outcomes, not fabricated failures**: `_run_outcome` classifies every run `success`/`failure`/`censored`. `aborted`, `claimed_done` (no verifier ran), and torn runs with no `run_finished` are censored — `ChoiceModel` drops their transitions from outcome labeling and `CounterfactualTrials` excludes them from the endpoint (an unrelated crash is missing data, not evidence the arm failed). Measured non-successes — `blocked`, `done` without verifier pass — remain negative. Context-free legacy transitions keep their conservative non-positive labeling.
- **Immutable bound experiment plans (`jev-experiment-plan/1`)**: stamped `experiment_proposals` now carry and bind `task_key`, `family_key`, world/step provenance, `state` fingerprint, `historical` model-choice identity, `proposal`, `proposal_offered_rank`, `offered_catalogue_digest`, `policy_behavior_digest`, `choice_model_digest`, `world_pool_digest`, and `evidence_head_hash`; `experiment_plan_digest` covers the whole artifact. The agent re-verifies every binding live — digest, task, state, model choice, offered membership, offered-catalogue and policy-behavior digests, and the choice-model identity — and a plan that claims the current state but fails a binding is discarded as stale (recorded as `experiment_plan_stale`), suppressing ad-hoc substitution for that step. The audit's tamper repro (mutating `proposal` after stamping, then running under a different model choice) no longer executes.
- **Assignment metadata is self-contained**: `experiment_assigned` now records `task_key`, `task_family`, `instance_id`, `state`, both sides' `kind`/`goal_overlap`/`offered_rank`, `assignment_probability`, and `experiment_id` — everything the ITT estimator groups on exists even when no transition ever follows.
- **Trial estimates surface in reports**: `DreamReport` carries `trials_digest`/`trial_estimates` and `jev-dream improve` fits `CounterfactualTrials` unconditionally — annotation only, never a gate or promotion input.
- **Local backend hardening**: `scripts/local_backend.py` now requires a loopback Host header at its bound port, rejects cross-site Origins, serves only `POST /v1/systemone` (GET and other paths 404), bounds bodies at 4 MiB, and honors an optional `JEV_LOCAL_TOKEN` bearer token (`JEV_DECISION_API_KEY` on the agent). `tests/test_local_backend.py` exercises the full gate set over a real loopback socket with Ollama mocked.
- New events are `jev-ultrafast-tcb/0.12`; TCB 0.11 stores remain supported and their legacy experiment-tagged transitions still analyze under the fallback path.

## v0.8.3 causal model and hierarchical context

The audit's two deferred items — "feed randomized causal evidence into a separate learned decision model" and "add hierarchical context so unrelated websites/task families do not share one posterior" — land here, still annotation-only: nothing in this section is a promotion gate.

- **Separate causal decision model (`jev-causal/1`)**: `TrialChoiceModel` consumes randomized evidence only — `CounterfactualTrials` assignments, never observational transitions — and reports the ITT effect of *assigning* a divergence. `choose` proposes the offered candidate with a reliable positive delta (strongest first); `refuted` reports a divergence whose delta is reliably non-positive, so the proposal layer stops re-stamping an experiment the randomized layer already settled. No premise or no reliable effect means abstention.
- **Causal-first proposals in replay**: `_evaluate_world` asks `TrialChoiceModel` before `ChoiceModel`; an observational divergence that trials refute is suppressed rather than stamped. The stamped plan's `choice_model_digest` binds whichever prior generated the hypothesis — a causal-origin plan only executes under the same causal model. `DreamReport.trial_model_digest` surfaces the causal prior's identity alongside the existing model digests.
- **Hierarchical observational prior (`jev-choice/5`)**: `ChoiceModel` carries a task-family stratum over the same `(kind, overlap, offered-rank)` cells — a family-specific posterior wins when it meets `MIN_CONFIDENT`, otherwise the pooled cell answers. Experiment-tagged transitions are excluded from the fit entirely: a trial-scheduled action is the scheduler's choice, and fitting it as the policy's own would launder randomized evidence into the correlational prior.
- **Hierarchical trial contexts (`jev-trials/4`)**: cells key on a five-part context — scope (`f:<task_family>`, `s:<site>`, or `*`), model-choice kind/overlap, proposal kind/overlap — so a "book a flight" divergence stops pooling with a "pay an invoice" one. `resolve(task_family, site, ...)` walks family → site → pooled, returning the first stratum whose contrast meets the reliability floor (a thin specific stratum cannot shadow a reliable broader refutation) and falling back to the most specific unreliable estimate when nothing clears it. `from_dict` fails closed on the old 11-field cell arity.
- **Live context flows**: `experiment_assigned` records `site` beside `task_family`, and the live experiment path passes `model_choice`/`task_family`/`site` into the proposal model (signature-tolerantly, so a foreign model still works). Serialization: `TrialChoiceModel` round-trips with a stable digest; `CounterfactualTrials` cells are now 15 fields.

## v0.9.0 causal-validity pass

The v0.8.3 audit's finding was precise: `delta_reliable` tested sample *support*, not whether the treatment effect's sign was established, and the surrounding layer had differential-censoring, indexing, binding, and generalization defects. v0.9.0 fixes the semantics; nothing in this section changes the authority plane or any promotion gate.

- **Support vs. effect certainty (`jev-trials/5`)**: `support_sufficient` (both arms meet `MIN_ESS` on weighted evidence) is reported separately from `effect_status` — `beneficial` requires the sequentially valid interval to lie entirely above the practical threshold (`min_effect`, configurable per task family on `TrialChoiceModel`), `harmful` entirely below its negative, and anything else is `unresolved` with a reason (`ci_crosses_zero`, `censoring_imbalance`). The audit's 5/8-vs-4/8 repro now stays unresolved instead of becoming a confident proposal, and 4/8-vs-5/8 no longer permanently refutes.
- **Sequential establishment**: effect status is decided by an anytime-valid interval built from a summable α-spending sequence (`α_n = α·ζ(3/2)⁻¹·n^-3/2` over integer looks), so re-peeking after every batch cannot inflate a single hypothesis's error rate; the fixed-sample Newcombe interval remains for reporting, and a test pins that establishment is strictly harder than one fixed-sample look. Multiplicity across simultaneously tracked hypotheses is explicitly *not* corrected by this bound — the conservative direction is always `unresolved` — and per-hypothesis α budgeting remains roadmap.
- **Structured censoring**: `run_finished` records a reason (`operator_cancel`, `browser_crash`, `agent_exception`, `network_failure`, `timeout`, `recorder_shutdown`, `unverified_claim`, `unknown_abort`); `model.py` raises distinguishable `ModelTimeoutError`/`ModelConnectionError` subclasses and `browser.py` normalizes transport failures to `BrowserError` so the agent can classify aborts honestly. Per-arm `censor_rate` and reason breakdowns are reported, and the imbalance gate refuses to establish an effect when either arm is censored above 50% or the arms differ by more than 25 points — the audit's 48-assigned/8-analyzed/40-aborted candidate now reads `unresolved: censoring_imbalance`, not `beneficial`.
- **True family/site hierarchy + treatment signature**: cells store `task_family` and `site` as separate coordinates (v4 scopes migrate in place), so resolution walks `family+site → site → family → pooled` and the audit's flights/thin-family/target.example-site repro now answers at `site` instead of a pooled estimate dominated by unrelated sites. Each arm is additionally keyed by a bounded treatment signature — kind, effect class, element role, overlap bucket, offered rank bucket, workflow phase — and resolution backs off over those coordinates (phase → rank → role → effect → overlap) with `kind` as a floor: click evidence never answers a fill proposal, and the returned `signature_level` names exactly how much generalization happened.
- **Bound, fresh, optionally signed plans**: `_match_stamped_plan` now compares `family_key` (the audit's cross-family repro is stale, not executable), makes a live causal model's `refuted` verdict stale the plan (newer randomized evidence beats an old hypothesis), and supports `require_current_model` for the "originating model must be available and current" mode. Plans can carry a domain-separated Ed25519 `authority` block (`jev-dream/experiment-plan/v1:`); the digest excludes it, execution distinguishes `digest_valid` from `authority_signature_valid`, and with `JEV_EXPERIMENT_VERIFY_KEYS` configured unsigned (`plan_unsigned`), forged (`plan_signature`), or untrusted-key plans fail closed. `JEV_EXPERIMENT_SIGNING_KEY` (falling back to promotion/evidence keys) signs at stamping time.
- **Duplicate assignments count once**: a repeated `(run, experiment_id, proposal, model_choice, arm)` record is one unit of analysis; the duplicate count is reported on the model instead of double-weighting an arm.
- **`ExperimentScheduler`**: ranks unresolved hypotheses by utility-weighted practical importance × remaining sequential-interval width ÷ evidence coverage, and excludes settled (`beneficial`/`harmful`) hypotheses entirely — the trial budget goes to uncertainty that still exists. Deterministic ordering keeps replays reproducible; its bookkeeping is stripped before stamping.
- **`UtilityWeights`**: per-arm utility is success minus latency, token, approval, and censoring costs. Statistical significance alone is not an improvement; hard safety/authority constraints are deliberately outside the utility function.
- **`CausalChoicePolicy`**: combines the observational and randomized channels with per-entry provenance (`observational` / `randomized` / `pooled_randomized`) under operator-gated modes — `shadow` (annotation only), `canary` (proposal enters the randomized assignment), `active` (a randomized-*established* beneficial proposal overrides the model's choice deterministically, still through the full authority plane, recorded as `causal_policy_applied`, and excluded from policy-canary qualification).
- **Provenance is permanent**: `ChoiceModel.predict` reports `source: "observational"`; `TrialChoiceModel` reports `source: "randomized"` and never reads observational transitions (tested on an experiment-free store); the two channels are never silently mixed.
- **Evidence is `jev-ultrafast-tcb/0.13`** (structured reasons, treatment-signature coordinates, signed plans); 0.12 and older stores remain readable, and mixed-TCB replay pools are still rejected.

## Deployment gate

Before production activation:

1. Run `uv sync --frozen` with network/package cache available.
2. Run `uv run ruff check .` and `uv run pytest` (both pass in this environment).
3. Run `uv run python scripts/check_guards.py` against the intended Chrome/Browser Harness installation.
4. Re-run representative browser benchmarks; the bundled v0.1 speed evidence is historical, not a fresh v0.4 latency claim.
5. Collect matched baseline/candidate canary runs across the intended site/task distribution. The built-in 12-run/four-family thresholds are minimum gates, not statistical proof.
6. Activate only through `jev-dream promote` / `PolicyRegistry.promote_from_store(...)`. For authenticated qualification evidence, sign store events (`JEV_EVIDENCE_SIGNING_KEY`, preferably via an injected signer holding the key outside the agent process) and require verification (`JEV_EVIDENCE_VERIFY_KEY`, `JEV_REQUIRE_SIGNED_EVIDENCE`) on the qualification host.
7. Run periodic `jev-dream health`; use suspension or rollback on meaningful post-promotion drift.
