# Validation — Jev Ultrafast v0.6.0 DREAM-Jev

Validation date: 2026-09-29 (v0.5.0 validated same-day; v0.5.1 applies the independent-audit authority patch: monotonic classification, family-namespaced canary pairing, signed promotion attestations, chain-head anchoring, label redaction, complete catalogue digests, and explicit execution guarantees; v0.6.0 adds the uncertainty-aware `ChoiceModel` counterfactual layer — annotation-only, never evidence).

## Reproduced in this build environment

- `uv run pytest`: **237 passed** against the project environment with `browser-harness==0.1.13` installed; browser/CDP calls in unit tests remain mocked by the tests themselves.
- `uv run python scripts/check_guards.py` against a dedicated headless Chrome 154 instance (`BU_CDP_URL=http://127.0.0.1:9222`, throwaway `--user-data-dir`): **all 34 live browser guard checks passed**, including the adversarial mid-input cases (post-observe commit swap, same-label node swap between press and release, pre-release overlay, pre-release geometry drift, mid-press page mutation, pointer-state hygiene after abort, nested-descendant interception, post-observe hidden/detached targets, structured `ctx` propagation) and the atomic-guarantee check proving `DOM_ATOMIC` clicks dispatch no press/release window for mid-event sabotage. Earlier validation against Chrome 152 also passed; the first v0.4.0 live run had exposed a latent defect that mocked tests could not see, which is why CI now runs this suite in a `live-guards` job on every change.
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
- During replay it is queried under the *candidate* policy's recomputed offered rank (the same basis as the recorded `selected_rank`), so a filtering policy provably changes which cell the recorded action lands in — the signal is policy-dependent, not a descriptive prior.
- `choose()` proposes only from the offered catalogue using an upper-confidence score (`mean + 0.5·std`); tests cover abstention on empty/sparse fits, proposals restricted to observed candidates, deterministic digest/round-trip, and UCB preferring an unexplored action while flagging it not-confident.
- Counterfactual proposals are annotation-only (`choice_model` block: `divergence_rate`, `mean_uncertainty`, `confident_fraction`, `predicted/proposed_page_changes_per_step`); tests assert identical `ReplayMetrics`, promotion decisions, and selected digests with and without the model — a proposal can never open a gate or become evidence.

## Deployment gate

Before production activation:

1. Run `uv sync --frozen` with network/package cache available.
2. Run `uv run ruff check .` and `uv run pytest` (both pass in this environment).
3. Run `uv run python scripts/check_guards.py` against the intended Chrome/Browser Harness installation.
4. Re-run representative browser benchmarks; the bundled v0.1 speed evidence is historical, not a fresh v0.4 latency claim.
5. Collect matched baseline/candidate canary runs across the intended site/task distribution. The built-in 12-run/four-family thresholds are minimum gates, not statistical proof.
6. Activate only through `jev-dream promote` / `PolicyRegistry.promote_from_store(...)`. For authenticated qualification evidence, sign store events (`JEV_EVIDENCE_SIGNING_KEY`, preferably via an injected signer holding the key outside the agent process) and require verification (`JEV_EVIDENCE_VERIFY_KEY`, `JEV_REQUIRE_SIGNED_EVIDENCE`) on the qualification host.
7. Run periodic `jev-dream health`; use suspension or rollback on meaningful post-promotion drift.
