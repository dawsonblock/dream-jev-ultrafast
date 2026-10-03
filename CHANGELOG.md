# Changelog

Detailed release history for Jev Ultrafast. Cut releases live on the
[GitHub releases page](https://github.com/dawsonblock/dream-jev-ultrafast/releases);
this file documents what changed and why, in the project's own words.

## Unreleased

### release provenance, upload binding, and structural policy programs

An independent audit (P0–P2) closed the remaining security/scientific gaps
and made the mutable artifact genuinely more expressive:

- **Upload approval now binds the file's contents, not its path**
  (TOCTOU). Declared upload files are snapshotted into a private staging
  directory at agent construction, the approval capability digest covers
  the staged content hash + byte length, and dispatch sends the staged
  snapshot — replacing the source file after approval can never change
  the bytes that were reviewed. Staging is cleaned on `Agent.close()`.
- **Exact-action evidence is a measured contrast, not a presence flag**
  (`jev-trials/11`). The trial store keeps per-`action_key` cells
  (`action_index` + `action_cells`), and `resolve` annotates a separate
  `action_exact` contrast — the exact action's own support, effect, and
  confidence sequence — alongside the class-level estimate. Active-mode
  deployment requires the action's own evidence to be sufficient and
  beneficial, a documented `hierarchical_shrinkage` rule for thin
  per-action data, or a verified external confirmation; a class effect
  can no longer carry an exact action whose own randomized data is thin
  or harmful (`deployment` block reports the estimator and blockers).
- **Multiplicity is allocated online and irrevocably.** Hypothesis
  contexts register at declaration time in `family_order`, and each
  declaration spends a fixed geometric share of the family α
  (`SEQUENTIAL_ALPHA` × rate × (1 − rate)^index — a budget of 0.05 that
  later hypotheses can never retroactively loosen for earlier ones).
  Contrasts report `family_alpha`; stores without declaration order use
  a deterministic equal split across known contexts.
- **The one-trial invariant holds on legacy evidence too.** The
  transition-tag migration path deduplicates identical rewrites,
  quarantines runs carrying two *distinct* experiment decisions, and
  rejects assignments ordered after the terminal event — the same rules
  the modern `experiment_assigned` path enforces.
- **External/manual action confirmations are signed evidence.** Raw
  `confirmed_action_keys` no longer admit an action: confirmations are
  domain-separated Ed25519 attestations
  (`jev-dream/action-confirmation/v1`, `mint_action_confirmation`)
  verified against `JEV_CONFIRMATION_VERIFY_KEYS`, optionally scoped to
  task family/site — unsigned confirmations fail closed when keys are
  configured.
- **`ExplorationPolicy` is a real declarative program, not just a knob
  vector.** Two new validated fields — `rules` (guarded score adjustments)
  and `stop_when` (a stopping expression) — interpret a small checked AST
  (`_dream/policylang.py`: feature/const/cmp/and/or/not/add/mul nodes
  over a bounded feature vocabulary). Live candidate scoring, replay
  filtering, and run termination all evaluate the program; digests stay
  canonical, and the baseline digest is unchanged (empty programs
  serialize as before). `mutate_policies()` grows genuine structural
  mutations alongside the scalar ones — rule insertion/removal,
  expression-constant perturbation, and stop-condition rewrites — 79
  baseline candidates where there were 73.
- **Release verification is externally bootstrapped**
  (`scripts/verify_release.py`). A self-contained verifier — stdlib +
  `cryptography`, zero `jev_ultrafast` imports — that can be copied out
  of the tree: pinned-key `MANIFEST.sig` verification, full-set hashing,
  the TCB floor embedded as an external expectation rather than imported,
  a `--expect-tcb-version` pin, and a check that a shipped
  `VALIDATION.generated.md` declares the manifest digest it actually
  validates (`--require-validation` makes absence a failure). The
  internal `scripts/check_tcb.py` remains the CI consistency check and
  now says so.
- **`JEV_SECURITY_PROFILE=qualified` — strict plus provenance.** Every
  strict refusal applies (it is a hardened profile), and the agent
  additionally re-verifies the installed tree against its signed
  manifest under `JEV_MANIFEST_VERIFY_KEYS` at construction —
  `tcb.verify_installation()` hashes the release set and checks the TCB
  boundary before any run starts; unsigned, unpinned, or drifted trees
  fail closed (TCB `jev-ultrafast-tcb/0.17`).

### exact-action generalization and experimental-unit enforcement

A fifth-pass scientific audit found two causal-evidence gaps plus a ranking
ambiguity, all fixed inside the existing causal layer:

- **Class-level evidence can no longer masquerade as exact-action
  evidence** (`jev-trials/9`, `jev-causal/3`). `CounterfactualTrials.fit`
  now indexes every randomized proposal's `action_key` — a stable hash of
  kind + effect class + role + the `redact_text`-normalized label — and
  `resolve` annotates each contrast with `generalization_level`
  (`exact` / `same_context_class` / `cross_context_class` /
  `pooled_class` / `none` — the last for queries with no provable action
  identity), `exact_action_randomized` (mirroring `exact` exactly),
  `action_randomized_anywhere`, `action_randomized_in_context`,
  `context_randomized`, `treatment_class_randomized`, and `action_key`.
  The agent stamps `proposal_action_key` into `experiment_assigned`
  metadata; older evidence recovers the identity from the executed
  transition's catalogue, and an unrecoverable identity is never invented.
  `CausalChoicePolicy` gains `active_generalization_levels` (default
  `("exact",)`) plus two canary-confirmation channels for
  `same_context_class` evidence — `action_randomized_in_context`, derived
  from the trial store itself, and `confirmed_action_keys` for
  confirmations evidenced outside this model — so a never-randomized
  action cannot enter `active` on class-level evidence alone; unknown
  generalization names fail closed at construction, and unverifiable
  levels report `unverified_generalization`.
- **One randomized assignment per experimental unit — enforced, not
  trusted.** Two *different* assignment decisions in one run quarantine
  the whole unit under the new `invalid_units` counter (identical rewrites
  still deduplicate under `duplicates`); an assignment ordered after the
  run's terminal event is likewise invalid; malformed arms or propensities
  outside (0, 1] are `rejected` before they can shape a cell. Runless
  records remain censored-by-construction — their mass still feeds the
  censoring bounds — which is stricter than silently dropping them.
- **Ranking scores are separated by purpose.** Every ranked entry now
  carries `causal_score`, `observational_score`,
  `experiment_priority_score`, and `deployment_score` instead of one
  overloaded `score`, and `active` ordering uses causal terms only — an
  observational prior can annotate an entry but can never outrank superior
  causal evidence. Provenance is reported as independent dimensions
  (`evidence_origin`, `trial_level`, `support_status`, `effect_status`,
  `generalization_level`): an *unsupported* pooled contrast is now
  honestly `pooled_randomized`/`insufficient` rather than wearing the
  context-specific `randomized` label.
- **Multiplicity is now an explicit registered family, not inferred cell
  count** (`jev-trials/10`). Every parseable `experiment_assigned` record
  registers its hypothesis context *before* outcomes are observed —
  including declarations whose units were later quarantined
  (`invalid_units`), rejected as malformed, or fully censored — and
  `hypothesis_count` (the Bonferroni denominator behind every confidence
  sequence) is the registered contexts unioned with observed cells. A
  declared-but-unanalyzable test therefore still pays its share of α
  instead of silently disappearing from the error budget. Contrasts
  report `hypothesis_registered` — both arms declared the answering
  context (every contributing context, when backoff merges several) — as
  preregistration provenance, and `CounterfactualTrials.hypothesis_summary`
  exposes the declared/observed/unobserved bookkeeping. Stores written
  before registration existed keep the cell-derived family and report
  `hypothesis_registered: False` rather than borrowing authority.

### adversarial-page authority binding

A fourth-pass source audit found that the authority plane was weaker at the
*mutation* boundary than at the *decision* boundary, plus provenance and
privacy gaps in the release tooling:

- **Authority context is part of the execution transaction** (TCB
  `jev-ultrafast-tcb/0.16`). `snapshot.js` computes `ctxOf(e)` — submit
  semantics, form method/origin, scoped sensitive-field inventory, the
  target's own field class, messaging/external/download flags, modal scope —
  and `classify_effect()` decides authority from exactly that context. The
  pre-mutation guard, however, compared only identity/value/style signals,
  so a page could flip `button.type` to `submit`, change `autocomplete` to
  `one-time-code`, repoint a form cross-origin, or inject a password sibling
  between the policy decision and dispatch without tripping the stale
  guard. The guard now embeds the full `ctxOf` result, and every guarded
  evaluation (`fresh`, atomic mutate, trusted pre-press/pre-release)
  re-derives it inside the isolated world — a mismatch is a stale target,
  never a mutation under a weaker classification. Evidence `0.15` remains
  readable; 0.16 marks the executor generation where the invariant closed.
  `scripts/check_guards.py` grows a live authority-context section: eight
  semantic mutations (type/autocomplete/method/action/sibling-inventory/
  href-scheme/target/download flips) must each invalidate the action's
  guard, and the atomic dispatch path must refuse a post-decision submit
  flip before `e.click()`.
- **Packaged-tree `--check` rejects files the manifest does not list.**
  `update_manifest.py` previously verified only listed files in a tree
  without `.git` — an extra file added after unpacking passed silently. In
  packaged mode it now enumerates the files actually present (with the same
  narrow ignore set as `.gitignore`) and fails closed on anything unlisted,
  so a signed manifest fully characterizes the unpacked release.
- **Signature verification never executes the artifact.** `sign_manifest.py`
  and `qualify.py` previously inserted the release root into `sys.path` and
  imported `jev_ultrafast.signing` to verify — running package code from the
  very tree under verification. Both now carry ~20 lines of self-contained
  Ed25519 (`cryptography` only), so `--verify`/`--verify-report` cannot be
  subverted by the artifact. A regression test runs both verifiers in a
  fresh interpreter and asserts `jev_ultrafast` never enters `sys.modules`.
- **URL paths and userinfo no longer carry secrets to the model.**
  `sanitize_url()` redacted query/fragment but forwarded `parts.netloc`
  (including `user:pass@` userinfo) and the raw path — reset tokens,
  addresses, and opaque session identifiers in path segments could cross
  the remote boundary under nominal redaction. Userinfo is stripped, and
  each raw path segment is decoded, redacted, and masked when token-shaped;
  ordinary slugs survive. Decoding is confined to the segment it came from —
  an escaped `?`/`#`/`/` is re-encoded, never re-emitted as real URL
  structure — and sensitive `key=value` material is masked in decoded paths,
  fragments (`#access_token=…`), and nested-URL query values, so an encoded
  `token=` cannot bypass the sensitive-parameter mask. JWTs are also now
  caught by `redact_text` generally.
- **`sanitized` routing rejects `JEV_MODEL_PRIVACY=off`.** Two separately
  named controls could silently compose into raw remote transmission —
  sanitized promised redaction while `off` disabled the pass. The routing
  level now fails closed with a pointer to the honest combinations:
  `public` for verbatim text, `local-only` to stay on-machine.
- **`Agent.snapshot()` returns a deep copy.** The shallow spread left
  `page`, `history`, `decisions` and other nested structures shared with
  live agent state — an in-process consumer could mutate internals through
  the returned dict. The snapshot is now fully detached.

### security profiles, replay verdicts, and qualification taxonomy

- **Named security profiles bound how the knobs compose**
  (`JEV_SECURITY_PROFILE`). The individual controls — routing level,
  privacy mode, insecure-transport flag, input guarantee, signed-evidence
  requirement, unbound-metrics hatch — were grown one defect at a time,
  and no single declaration described the posture a qualified run should
  hold. `standard` (the default) is the status quo; `strict` refuses
  `JEV_MODEL_ROUTING=public` and `JEV_MODEL_PRIVACY=off` under any
  routing level, refuses `JEV_ALLOW_INSECURE_TRANSPORT` even where it
  would apply, refuses `JEV_INPUT_GUARANTEE=trusted` (atomic
  isolated-world execution only), keeps `JEV_ALLOW_UNBOUND_METRICS`
  closed, and implies `JEV_REQUIRE_SIGNED_EVIDENCE`. A knob weakening
  below the declared profile fails closed naming the profile — never
  silently ignored — and an unrecognized profile value is itself a
  misconfiguration that fails closed.
- **Replay verdicts are named, not just penalized.** Each replayed world
  now reports `verdict` — `verified_success`, `measured_terminal`,
  `coverage_miss`, `blocked`, or `inconclusive` — and `ReplayMetrics`
  carries `verdicts` counts alongside the existing score fields. A world
  the policy simply declined to act on is now distinguishable from one
  that ran to a measured terminal outcome, and an inconclusive world
  (insufficient evidence to classify) is no longer conflated with either.
- **`import jev_ultrafast` no longer starts the browser stack.** The
  browser-coupled exports (`Agent`, `Browser`, `run`) resolve lazily via
  `__getattr__`, so the learning and DREAM layers import without pulling
  in the CDP harness; `from jev_ultrafast import Agent` behaves exactly
  as before.
- **`qualify.py` checks declare their requirement.** Each check reports
  `requirement` — `mandatory`, `environment-dependent` (needs a host
  capability; a skip is an honest absence, never a pass), or `optional`
  (advisory, never gates the verdict) — and the summary tallies them.
  Checks that predate the taxonomy read as `mandatory`, which is the
  conservative interpretation.
- **Scheduler terminology corrected.** The exploration scheduler's
  heuristic is a novelty/coverage prior; docs no longer call it
  "information gain", which is a measured quantity the heuristic does
  not compute.

### trusted base, policy language, lifecycle, and health monitors

- **The trusted computing base is now a machine-checkable boundary**
  (`jev_ultrafast/tcb.py`, `scripts/check_tcb.py`). `TCB_FILES` names the
  individually trusted files — execution, authority/approval, privacy,
  evidence, replay scoring, causal estimation, promotion, signing,
  qualification tooling — and `TCB_PACKAGE_PREFIXES` makes every file
  under `jev_ultrafast/_dream/` and `_learning/` trusted by
  construction, so code can never be *non*-TCB inside the trusted
  packages by accident. `check_tcb.py` (a mandatory Q0 check) fails
  closed on missing or drifted TCB files, on TCB files absent from the
  signed manifest, and on package members the manifest does not cover.
- **`ExplorationPolicy` is a declared DSL, not an incidental
  dataclass.** `_FIELD_SPECS` declares every field's type and inclusive
  envelope once — validation, `from_dict`, `behavior_digest`, and
  introspection via `ExplorationPolicy.schema()` all derive from it.
  Type violations (a string or bool where an integer is declared, a
  non-dict payload) are declared `ValueError` rejections, never leaked
  `TypeError`s; the envelopes themselves are unchanged.
- **Mutation is structural as well as scalar.** `mutate_policies` still
  walks each knob bidirectionally, and now also emits envelope-boundary
  probes, joint moves of coupled knob groups (patience, kind budgets,
  selectivity), single-knob ablations back to the class default, and
  knock-outs zeroing each kind prior — every candidate still inside the
  declared DSL, still deduplicated by `behavior_digest`.
- **The promotion lifecycle is an explicit state machine**
  (`POLICY_STATES`, `POLICY_TRANSITIONS`). Records carry a declared
  `status` — `staged`, `active`, `suspended`, `retired` — every
  transition walks the declared edge table (`_assert_transition`), and a
  record whose declared status contradicts its slot or flags fails
  closed at load even without trust keys. Idempotent self-loops keep
  re-suspend and re-resume race-safe; records written before the field
  existed derive status from slot at load.
- **Health monitoring sees the crash matrix.** `CanaryMetrics` gains
  `abandoned_runs` (starts that never produced a measured outcome) and
  `crash_runs` (the candidate-attributable subset — browser crashes,
  agent exceptions, indeterminate executions), computed under exactly
  the promotion-path membership filter. `HealthGate` checks attrition
  *before* task sufficiency — a policy dying on every start can no
  longer read as "insufficient data" — and now monitors the same
  dimensions the promotion gate measured: failure-rate drift,
  zero-success collapse, and latency/action/token regressions, on top of
  the existing success and risk monitors. Older canary blocks rebuild
  with zero defaults and stay verifiable.

### browser coverage: frames, shadow DOM, tabs, keyboard scroll, upload

The observation/execution loop grew the five capabilities the baseline
called out, each under the same authority invariants rather than beside
them:

- **Same-origin frames and open shadow roots are traversed.** The
  snapshot walks every reachable document (depth/breadth caps bound the
  walk); elements inside them get the same node identity, guard
  recording, page-key membership (re-queried *live* on every guard call
  — a mid-press appended input still invalidates), effect context, and
  text inclusion as top-document controls. Hit-tests run in the
  element's own document *and* the top document through `viewRect`
  frame-chain translation, so an overlay in either document is a
  `covered` rejection; `ownerDocument` scoping fixes focus/selection/
  `execCommand` for frame fills and `aria-labelledby` for shadow
  scopes. Cross-origin frames are counted (`unreachable_frames`) and
  never traversed; closed shadow roots remain indistinguishable from
  "no shadow" — honest gaps, not silent ones.
- **Popup/new-tab ownership.** `Browser` tracks the owned tab cluster
  (targets whose opener chain leads back to the session, transitively).
  `observe()` adopts the newest owned tab — the page a click produced is
  the page the user sees — and falls back to the surviving opener when a
  popup closes; `close()` tears down the whole cluster. `act()` never
  syncs: a decision always executes on the tab it was made on.
- **Keyboard scroll.** Whitelisted scroll keys (PageDown/PageUp/Home/End)
  dispatch through real CDP key events after an editable-focus check
  that descends the frame chain — a scroll key landing in a field would
  edit, so it refuses as stale instead. Non-whitelisted keys classify
  `unknown_commit` and gate like any unclassified commit.
- **Nested scroll containers.** Scrollable elements holding actionable
  content earn guarded `scroll` node actions executed by in-world
  `scrollBy` inside the same atomic validate-and-mutate turn as
  click/fill — no physical input to spoof, no fixed wheel coordinate to
  miss the inner scroller.
- **File upload through an operator allowlist.** File inputs are
  offered as `upload` actions; `Agent(uploads=[...])` / `JEV_UPLOADS`
  declares the attachable set (validated as real regular files at
  construction, ≤32, deduplicated). `UPLOAD` targets carry
  `element:file_index` like select options — the model names a declared
  basename, never a path — and dispatch resolves the index server-side,
  revalidates page and guard in-world, then sets `input.files` through
  `DOM.setFileInputFiles`. It is non-transactional by construction
  (there is no in-world setter), so strict profile refuses declared
  uploads outright, a lost acknowledgement is `IndeterminateMutation`,
  and the approval digest binds path+size.

## v0.9.2

Three workstreams shipped together as v0.9.2 (the release cut in
`f59f2b5`); they are kept as separate subsections rather than separate
version headings — the `v0.9.2` label is unique to this release.

### qualification correctness

A third-pass audit went after scientific-method defects rather than authority
defects: the executor held, but several places the evidence could not carry
the claims being made on it. All ten recommendations implemented:

- **Action-catalogue truncation is node-fair.** `snapshot.js` previously
  expanded every control inline in DOM order and then dropped everything past
  1,200 actions — one giant `<select>` early in the page could starve every
  later control out of the Python-side catalogue entirely. Options are now
  collected per-node and merged round-robin before the bound applies, so a
  big select loses its own deepest alternatives instead of hiding unrelated
  controls. A Node-driven regression test proves a 1,500-option select cannot
  starve the controls after it.
- **The replay holdout is actually held out.** `DreamImprover.improve()` used
  to evaluate every candidate on train, validation, *and* holdout, and
  `_robust_gain`/candidate sorting consumed all three — the holdout was a
  second validation set. Selection now runs on train+validation only; the
  single selected candidate is then assessed on holdout once, reported as
  `holdout` promotion evidence. A spy test proves non-selected candidates
  never touch the holdout pool.
- **Degenerate randomization is rejected.** `experiment.rate` must keep both
  arms supported (`0.05 ≤ rate ≤ 0.95`); `rate=1.0` produced all-candidate
  "experiments" with zero control support — evidence that could never
  identify a causal effect. The bound also caps inverse-propensity weights.
- **Censoring is a bound, not a hope.** Trial cells (`jev-trials/7`) carry
  the censored weight per arm. Estimates report Manski worst/best-case
  `delta_bounds` — every censored unit re-counted as failure, then as
  success — and establishment requires the conservative bound to agree with
  the confidence sequence: `beneficial` only when the worst-case bound still
  clears the threshold, else `unresolved: censoring_bounds_cross_threshold`.
  Legacy v6 cells load with a conservatively imputed censor mass.
- **Sequential inference is a real confidence sequence.** The approximate
  α-spent interval is replaced by `delta_cs`: per-arm KL confidence bounds
  (self-normalized IPW statistics) union-bounded across looks by the summable
  schedule — coverage holds at *every* sample size simultaneously, which is
  the property unlimited peeking actually needs. The fixed-sample Newcombe
  interval stays for reporting only; `sequential_alpha` reports the per-look
  spend. Honest trade-off: the sequence is deliberately low-power at canary
  scale — a δ≈0.6 effect needs ~60–100 per arm to establish, so thin evidence
  now honestly reports `unresolved` instead of pretending.
- **Multiplicity is controlled.** Concurrent hypotheses share the family
  error budget — the per-look α is Bonferroni-split across the tracked
  hypothesis count (`hypothesis_count` is reported per contrast). A hundred
  simultaneous null divergences no longer each get a full α.
- **Trusted input is operator-selectable.** `JEV_INPUT_GUARANTEE=trusted` (or
  `Agent(input_guarantee=…)`) routes click/fill through the non-transactional
  CDP input path for `isTrusted`-gated sites. Automatic escalation still
  happens only after the atomic path provably did not mutate (`unsupported`);
  a silent no-op synthetic click is indistinguishable from one that landed,
  so it is never retried — the guarantee that matters is preserved by
  declaration, not by guessing. The dispatched guarantee is journaled on
  `action_attempted`.
- **Model routing has security levels.** `JEV_MODEL_ROUTING` binds the run to
  `local-only` (every model endpoint must be loopback; remote URLs fail
  closed before the request is built), `sanitized` (default — goal/objective
  text crosses the wire only through the redaction pass, same as page
  content), or `public` (explicit opt-in; goal sent verbatim). The goal used
  to cross to remote backends unsanitized even when page content was
  redacted.
- **Provenance is a qualification requirement.** `scripts/qualify.py` now
  verifies `MANIFEST.sig` against pinned keys (`JEV_MANIFEST_VERIFY_KEYS`) as
  a Q0 gate — under `--full` an unsigned or unverifiable manifest fails
  closed — and runs the anchor-enforcement test slice (rollback, truncation,
  forged/missing anchor on evidence store *and* registry) as a Q3 gate.
- **Validation documentation is generated.** `qualify.py --report-md` emits a
  generated report (`jev-qualify/2` JSON + markdown) bound to the exact
  artifact: manifest digest, signature key identity, git commit + dirty flag,
  environment digest, and per-stage pass/fail/skip tallies — the counts are
  produced by the run, not transcribed. The external Inter stylesheet was
  dropped from the inspector (fully local as documented), and the read-only
  store test now skips under a privileged runner instead of failing
  spuriously.
- **Review-pass release hardening.** A post-merge self-audit caught four
  release-engineering defects the feature work left behind: `MANIFEST.sig`
  (and `VALIDATION.generated.md`) are excluded from `update_manifest.py`'s
  file set — the signature authenticates the manifest's own digest, so
  listing it would create an unsatisfiable self-reference and a signed
  worktree could never pass `--check`; `.gitattributes` pins LF endings so
  a Windows CRLF checkout cannot silently invalidate every manifest hash;
  the Q0 reproducibility gate computes the double-build in-process instead
  of shelling to `sha256sum` (absent on stock macOS); and the loopback
  endpoint policy is defined once in `privacy.py` rather than duplicated in
  `model.py`. `.env.example` documents `JEV_MODEL_ROUTING`,
  `JEV_ALLOW_INSECURE_TRANSPORT`, and `JEV_INPUT_GUARANTEE`.

### audit remediation

A second-pass audit found four defects that mattered precisely because everything else was already tight: none of them let the model or learner self-authorize, but each one stretched a claim the evidence could not fully carry.

- **Interrupted mutations are a third state, not a stale page.** `browser.py` now distinguishes `IndeterminateMutation` (the dispatch may already have crossed its mutation point) from `StalePage` (provably pre-mutation, safe to re-observe). Atomic click/fill/select evaluations that lose their result to a destroyed context — and trusted presses whose pre-release check or release dispatch fails — raise the non-retryable class; the release is still attempted under `finally` for pointer hygiene. The agent journals `action_attempted` before every dispatch and `action_confirmed`/`action_not_executed`/`action_indeterminate` after, so a crash between dispatch and acknowledgement can never erase the attempt, and indeterminate executions abort the run under `indeterminate_execution` rather than being retried by the re-perception path.
- **Causal evidence no longer generalizes across effect classes.** `classify_effect` output now flows from live actions into offered-candidate annotations, `model_choice` metadata, causal proposals, trial assignments, compact replay candidates, and scheduler/stamped-plan resolution — so the complete treatment signature (kind, effect class, role, overlap, offered rank, workflow phase) actually reaches `TrialChoiceModel`. `kind` and effect class are the signature floor: backoff can drop phase, rank, role, and overlap, but never the semantic effect — `PURCHASE` evidence cannot answer a `SEARCH`/`DELETE`/`DISCLOSURE` divergence just because both were clicks. Trial cells widen to `jev-trials/6`; legacy v4/v5 cells migrate with explicit `"unknown"` wildcard coordinates.
- **The interval is renamed to what it actually is.** `alpha_spent_delta_ci` replaces `sequential_delta_ci`: a summable α-spending schedule applied to a Newcombe-Wilson interval on self-normalized IPW estimates with Kish ESS. The spending discipline is valid; the per-look interval is approximate, so the claim is now "α-spent approximate sequential interval," not a formally anytime-valid confidence sequence.
- **Promotion actually enforces the sign test.** `CanaryGate.max_pair_sign_p` defaults to 0.05 instead of `None` — promotion requires *demonstrated* improvement on `(task_family, instance_id)` pairs, which in practice demands more than the 12-run floor. The `promote(..., allow_unbound_metrics=True)` escape hatch is additionally feature-gated behind `JEV_ALLOW_UNBOUND_METRICS=1`, so unbound promotion can no longer be reached by a flag alone.
- **Store appends are O(1) again.** Qualification scale testing found `_truncate_torn_tail_unlocked` reading the whole log on every append — O(file) per append, O(n²) overall, ~4.5 minutes for 100k events. It now checks the last byte and scans backward only within the tail fragment: 100k events append in ~15s with fsync per record, and full-chain verification stays linear (~0.8s). Torn-tail recovery semantics are unchanged.
- **Qualification coverage widened.** The offline suite gains the cross-effect isolation matrix (SEARCH↔PURCHASE↔DELETE etc.), the explicit signature-backoff matrix, exact sign-test probability tables, serialization-fuzz rejection, 16-thread concurrent appends, and bounded null/power Monte-Carlo runs; `scripts/simulate_causal.py` is the heavier statistical harness (400-sequence grid: 0 false activations under every null, honest power curve — δ=0.30 at n=30/arm establishes ~4%, δ=0.60 ~82%). The live Chrome suite grows to 40 checks covering `onclick`/`oninput`/`onchange`/`mousedown` synchronous navigation and `document.write` replacement — each asserting *confirmed or indeterminate, never retryable-stale*.
- **`jev-dream trials` inspects the causal layer.** The treatment-signature evidence the remediation built had no read-side CLI: `trials EXPERIENCE` reports the fitted cell inventory (version, cells, duplicates), `--cells` decodes every cell's stratum/signature/arm counters, and any signature flag (`--model-effect`, `--proposal-kind`, `--phase`, …) resolves that one divergence through `CounterfactualTrials.resolve()` — printing which stratum and backoff mask answered, the per-arm counts, and the α-spent verdict, or `null` when no evidence can honestly answer. It shares the `verify`/`improve` verification flags and mutates nothing.
- **`scripts/update_manifest.py` is the manifest's canonical generator.** `MANIFEST.sha256` was previously regenerated by hand; the script covers tracked plus untracked-non-ignored files in a worktree (a not-yet-staged release file can no longer be silently dropped), falls back to the manifest's own file list in packaged trees without `.git`, and `--check` is the portable byte-exact equivalent of `sha256sum -c` for hosts without coreutils.
- **Qualification volume executed.** The Monte-Carlo grid ran at the full 100,000 sequences/point (false-beneficial ≤0.02% under every null; multiplicity probe shows 0/2000 sequences with a false `beneficial` at 100 simultaneous null divergences); the evidence store was driven to 1.57M events under two concurrent writers with a clean verify; `scripts/crash_check.py` adds a SIGKILL-during-append recovery probe (30 kills, every post-crash state verified); and the live guard suite reaches 47 checks — adding fill-onchange navigation, document.write teardown during fill/select, onpointerdown navigation, native form-submit navigation, history.replaceState, and a continuous-DOM-churn TOCTOU block that produced 25 correct-target clicks with zero wrong-target mutations.

### qualification volume

- **`scripts/race_check.py`** runs the interrupted-mutation loop at volume (1,000 iterations: 1000 executed, zero stale) and the TOCTOU continuous-churn loop at 10,000 iterations (4,890 correct-target, 5,110 rejected, zero wrong-target mutations).
- **`scripts/crash_check.py`** is the kill-during-write probe; **`scripts/bench.py`** captures the §50 percentile baselines; **`scripts/env_digest.py`** emits the §3 environment digest (OS/kernel/Python/Node/Chrome/commit/tree/lock/suite).
- **32-process fan-in**: 64,000 events appended by 32 concurrent writers — zero lost, zero duplicates, chain verifies.
- **Property tests** (`tests/test_properties.py`, Hypothesis): sensitive-context authority monotonicity, classifier totality, cross-effect signature floor, and mid-chain corruption rejection across ~900 generated cases.
- **Failure-mode tests**: signer failure (KMS/HSM) never falls back to unsigned; mid-chain signer death preserves the existing chain; read-only store refuses the append; a backward clock cannot corrupt the hash-linked chain; replay fits are digest-deterministic.
- **`scripts/fuzz_check.py`** runs 1,000,000 structured cases over the parse surfaces: joined treatment signatures can never inject the `|` delimiter, event serialization is injective under hostile payloads, and every line-level chain mutation is either fail-closed or a strict prefix of the original.
- **Crash injection at volume**: `scripts/crash_check.py --segment` rotates stores so per-kill verification stays bounded; the 1,000-kill run recovered to a verified chain after every SIGKILL (2.78M events across 40 segments).
- **`scripts/mutate_check.py`** is the bounded §55 sweep: 12 curated TCB mutants, each must turn the suite red — the first pass exposed four real gaps (required-signing enforcement, `prev_hash` link integrity, reserved-key forgery, anchor-digest tamper), now covered by dedicated tests; the sweep is 12/12 killed.
- **`scripts/sign_manifest.py`** signs/verifies the manifest digest under a domain-separated Ed25519 key (`jev-manifest-sig/1`), and consecutive `uv build` runs are byte-identical — the signed-reproducible-release artifacts exist.
- **`scripts/qualify.py`** is the staged §64 pipeline (Q0→Q4, `jev-qualify/1` report; unrunnable legs report `skipped`, never `passed`) and **`scripts/soak_check.py`** runs the probes under a wall-clock deadline for the 24h gate. The suite also passes under Python 3.12.9, matching 3.13.
- **`scripts/e2e_check.py`** is a real end-to-end leg: the full agent loop on real Chrome (scripted backend, real evidence store) — verified-done clean run with journaled attempts and an approval-gated privileged click, plus a decide-time DOM-churn fault leg that aborts censored with zero wrong-element mutation. CI `live-guards` runs it; `offline` is now an ubuntu+macos × 3.12+3.13 matrix.
- **`scripts/keydrill_check.py`** rehearses the §61 compromise drill: forged and stolen-key evidence rejected under rotation, the retired chain forensically readable, a fresh epoch continuing under the new key — rotation is a new epoch, never a rewrite.
- A committed `[tool.mutmut]` config makes the exhaustive §55 tool pass `uv run --with mutmut mutmut run`; the completed full-package run evaluated 17,180 mutants (9,245 killed, 7,157 survived, 753 untested, 25 timeout — ~55% score) and survivor triage shows the residue is equivalent mutants, diagnostic fields, and non-TCB plumbing, with the curated sweep covering the fail-closed gates specifically.
- CI `live-guards` runs on a Chrome-version matrix — the runner's stable Chrome plus pinned Chrome 140 via `browser-actions/setup-chrome` — so CDP and atomic-mutation behavior are verified across major versions.
- **Coverage floor**: the offline suite now runs under `--cov-fail-under=80` (suite at 86%; the browser-execution paths it can't reach are covered by the live Chrome guards). `hypothesis` and `pytest-cov` join the dev group.

## v0.9.1 — causal-evidence hardening

Post-release adversarial review found and fixed four evidence-integrity edges: assignment metas can no longer claim an unrelated execution step on absent-id coincidence, loose (runless) `experiment_assigned` records deduplicate like run-scoped ones, and trial-context coordinates are sanitized against the `|`-join delimiter so hostile page metadata cannot merge two distinct cells in reporting. Active `CausalChoicePolicy` overrides are now tagged `causal_override` on the transition (evidence TCB `0.14`) so an overridden step is recorded as scheduler influence — not an on-policy choice the observational priors would fit — and the query side of `TrialChoiceModel` actually exercises the stored signature coordinates (role, offered rank, workflow phase) instead of leaving them as wildcards.

## v0.9.0 — the causal-validity milestone

The causal layer no longer conflates *having enough samples* with *having established the effect*: `CounterfactualTrials` (`jev-trials/5`) reports `support_sufficient` (the weighted-sample floor) separately from `effect_status` — `beneficial`/`harmful` require the *sequentially valid* interval to clear the practical threshold entirely, and a supported interval that crosses zero stays `unresolved` and schedules more evidence, so `TrialChoiceModel.choose` (`jev-causal/2`) proposes only established-beneficial divergences and `refuted` is true only for established-harmful ones. Establishment is anytime-valid per hypothesis (a summable α-spending sequence over looks, documented with its multiplicity caveat) instead of a fixed-sample interval re-peeked after every batch. Censoring became a first-class causal concern: `run_finished` records a structured reason (`operator_cancel`, `browser_crash`, `agent_exception`, `network_failure`, `timeout`, `recorder_shutdown`, `unverified_claim`, `unknown_abort`), per-arm censor rates and reason breakdowns are reported, and extreme or sharply imbalanced censoring refuses to establish an effect — eight surviving successes cannot hide forty candidate-arm crashes. Task family and site are stored as separate coordinates (a family-carrying trial keeps its site stratum, so `family+site → site → family → pooled` backoff genuinely works), and each arm is indexed by a bounded *treatment signature* — kind, effect class, element role, overlap, offered rank, workflow phase — that resolution walks hierarchically (kind is a floor: click evidence never answers a fill proposal). Stamped plans now bind `family_key` at execution, go stale when a live causal model refutes them or when `require_current_model` finds the originating model missing, and can carry a domain-separated Ed25519 `authority` block (`jev-dream/experiment-plan/v1:`) — digest integrity is distinguished from provenance, and with `JEV_EXPERIMENT_VERIFY_KEYS` configured unsigned or forged plans fail closed. `ExperimentScheduler` ranks unresolved hypotheses by practical importance × remaining uncertainty ÷ evidence coverage and drops settled ones; `UtilityWeights` scores arms over success minus latency/token/approval/censor costs (safety constraints stay outside the utility); and `CausalChoicePolicy` combines the observational and randomized channels under operator-gated `shadow` → `canary` → `active` modes, tagging every entry `observational`, `randomized`, or `pooled_randomized`. None of this changes the authority plane: everything the causal layer prefers still passes effect classification, payload review, approvals, and browser guards, and only real executions qualify anything. Evidence is `jev-ultrafast-tcb/0.14` (0.13 under v0.9.0; see v0.9.1 for the `causal_override` tag); the causal adversarial suite (CI-crossing-zero, differential censoring, family/site fallback, cross-family and signed plans, unseen-action generalization, optional stopping, duplicate assignments, aborted runs, adversarial context keys, scheduler-influence tagging) runs offline in the 368-test suite.

## v0.8.3 — causal and contextual learned layer

The new `TrialChoiceModel` (`jev-causal/1`) is a separate decision prior built *only* from randomized assignments — it proposes the offered candidate whose arm measured a reliable positive intention-to-treat delta and treats a reliably non-positive divergence as *refuted*, so a hypothesis the experiment layer already settled is never re-stamped. Replay proposal selection is causal-first: measured evidence proposes, the observational `ChoiceModel` only fills in where trials are silent and not refuted. Learning is now hierarchical rather than one global posterior: `ChoiceModel` (`jev-choice/5`) carries a task-family stratum that wins when it has its own support and backs off to the pooled cell otherwise; `CounterfactualTrials` (`jev-trials/4`) keys each assignment on a five-part context — scope (`f:<family>`/`s:<site>`/`*`), model-choice kind and overlap, proposal kind and overlap — and `resolve()` walks family → site → pooled until a stratum meets the reliability floor, so a two-assignment family cell can no longer shadow a settled pooled refutation. Experiment-tagged transitions are excluded from `ChoiceModel` fits entirely: scheduled trial actions are the scheduler's, not the policy's, and mixing them in would launder randomized evidence into the correlational prior. Assignment records carry `site` alongside `task_family`, and the stamped plan's originator digest now binds whichever prior — observational or causal — generated the hypothesis.

## v0.8.2 — causal-bookkeeping corrections

`CounterfactualTrials` now analyzes `experiment_assigned` events — the randomized assignment is the unit of analysis, not the executed transition — so a trial denied by policy, vetoed by the operator, or never reached by the browser still counts toward its arm under the run's real outcome (intention-to-treat); post-randomization selection can no longer make a losing arm silently vanish. Runs that never produced a measured outcome (`aborted`, interrupted recording, unverifiable `claimed_done`) are *censored* — counted but excluded from the endpoint and from `ChoiceModel` trajectory labeling — while measured failures (blocked, denied, unverified `done` claims) remain negative evidence. Estimates carry `assigned`/`executed`/`censored` counts, Wilson intervals on effective sample size, and a raised `MIN_ESS = 8` reliability floor. Stamped proposals are now genuine `jev-experiment-plan/1` objects: the digest binds task key, state fingerprint, model choice, offered catalogue, policy behavior, originating model, and pool/head lineage, and the agent re-verifies every binding before executing — a stale or tampered plan is discarded and suppresses ad-hoc substitution for the step rather than silently running a different hypothesis. `scripts/local_backend.py` gains the demo server's loopback request hardening: Host allowlist, cross-site Origin rejection, endpoint-path allowlist, bounded bodies, and an optional `JEV_LOCAL_TOKEN` bearer check.

## v0.8.0 — audit corrections completed

Proposals must belong to the *offered* catalogue — the policy-filtered set the model saw — not merely `page["actions"]`. Randomization records `experiment_assigned` **before** the authority plane, so a denied/vetoed/stale trial still marks the run experimental and out of canary; the assignment is an immutable plan binding the offered-catalogue, policy-behavior, model, and proposal digests plus the prior's predicted delta and uncertainty. `CounterfactualTrials` now estimates `p_success` — the run's final independently verified outcome — as the primary endpoint (`p_page_changed` is secondary telemetry only, fixing an estimator that could report the exact opposite of reality). `ChoiceModel` is back on trajectory-success labeling: every step of a run takes the run's verified outcome, so page churn inside failed runs teaches the prior nothing. Trace propensity fields are honest: `selection_mode: argmax`, `behavior_propensity: null` for deterministic selection (model scores live under `selected_model_score`/`joint_model_score`), and only a randomized trial's `assignment_probability` is a real propensity. `Agent(experiment={"proposals": [...]})` now consumes the stamped `DreamReport.experiment_proposals` themselves. The v0.6.2 external registry head anchor returns as `PolicyRegistry(anchor_path=...)` + `jev-dream registry-reanchor`, the approval UI shows the exact payload being granted, and `MANIFEST.sha256` is verified in CI again.

## v0.7.1 — trial-integrity fixes

Approving a deviated trial now executes the *approved* proposal (previously the grant silently resumed the model's original choice and dropped the experiment tag), trial contexts key on the model choice so both arms share a cell, and a stripped signed registry can no longer be laundered into a legacy file by downgrading `schema` under configured trust keys.

## v0.7.0 — closing the counterfactual loop, carefully

`improve` now emits `experiment_proposals`: stamped hypotheses of the form "in this state the prior prefers B over the recorded A", ranked by expected delta. `Agent(experiment={"model": choice_model, "rate": ε})` turns those hypotheses into *real* trials: on an eligible step the scheduler executes the proposal (candidate arm) or the model's own choice (control arm) with a recorded `assignment_probability`, capped at one trial per run so outcomes stay attributable. The deviated action passes through the full authority plane — policy assessment, approvals, payload review — so an experiment can never execute what the agent couldn't otherwise do. Trial transitions carry an `experiment` block in signed evidence; `CounterfactualTrials.fit` turns them into self-normalized inverse-propensity estimates of verified progress per arm, with effective sample size so thin support can't masquerade as confidence. Trials never qualify promotion: runs containing an experiment arm are excluded from canary metrics, so a hypothesis must still earn its way through the replay + bound-canary path like everything else.

Two authority gaps from the v0.6.1 audit also close in v0.7.0: approval grants now bind the *generated payload* (`payload_digest`), so approving "fill field X" cannot silently authorize a different text than the one pending — and a `jev-dream/4` registry file without its chained state head fails closed instead of downgrading to legacy format (which would have unprotected mutable fields like `suspended`).

## v0.6.1 — audit correction pass

Traces now record `selected_observed_rank` and `selected_offered_rank` as separate coordinates — v0.6 trained the prior on the observed index while replay queried the offered one — and replay world-building fails closed on `offered_count` or offered-rank mismatches. The `ChoiceModel` target is verified progress rather than `P(page_changed)`: a run's terminal transition counts only when the run ended verifier-confirmed `done`, so a page change that ends in failure does not teach the prior the action was good — the model remains an advisory counterfactual instrument, not a learned decision policy. Unknown mutation kinds now fail closed (`UNKNOWN_COMMIT` → approval) instead of passing as autonomous, and `TYPE_TEXT` runs a second authority pass on the *generated payload* (`classify_payload`) before the browser mutation, so a field that describes itself as plain text cannot launder a card number or credential. Promotion attestations are bound to the complete active record — parent, replay digests, the whole canary block including the health-check reference metrics, timestamp, and revision — and every registry transition is a signed, hash-chained state commit: post-write tampering of even mutable fields like `suspended` fails the state head, and suspend/resume/rollback carry transition attestations chaining to the head they replace.

## v0.6.0 — first counterfactual learned layer

`ChoiceModel` (`--choice-model`) is an uncertainty-aware choice prior over the same `(kind, overlap, rank)` cells: it evaluates recorded actions under each *candidate* policy's own recomputed offered rank, carries posterior standard deviation and an explicit abstention flag on sparse cells, and proposes which offered action its upper-confidence score prefers — annotated per candidate as `divergence_rate`/`mean_uncertainty`/`confident_fraction`. A proposal is a hypothesis, never an outcome or evidence: counterfactual predictions cannot open a replay gate, and anything it prefers must still qualify through real executions and bound live canaries.

## v0.5.1 — residual authority-plane gaps from an independent audit

The classifier is now strictly monotonic — sensitive target fields (`ctx.field`: credential, card, email, messaging) escalate `fill`/`select`/toggle edits to `DISCLOSURE`/`FINANCIAL`/`AUTHENTICATE` instead of short-circuiting to `FORM_EDIT`, and high-risk labels evaluate before any structural shortcut. Canary evidence pairs by `(task_family, instance_id)` so identical instance ids in different families can never contaminate sign-test statistics. Policy activation is signed authority: every promotion records a domain-separated attestation binding candidate/parent digests, replay and canary evidence digests, the chain head, and the registry revision — `active_policy()` fails closed under configured verification keys when the attestation is missing, forged, or signed by an unexpected key; rollbacks re-verify the original attestation and sign the transition. The store supports a signed chain-head anchor (`--anchor`, `JEV`-independent checkpoint file): a log that no longer reaches the anchored head fails on read *and* on append, so a signed tail disguised as a crash-torn write is detectable rather than silently truncated. Candidate-catalogue digests now bind every replay-relevant field (node, role, option index/label, current value, checked, ctx, overlap), and action `label`/`option_label` text is redacted before model-boundary use. Execution guarantees are explicit: `DOM_ATOMIC` (validate + mutate in one isolated-world turn — default for click/fill/select) versus `TRUSTED_INPUT_NONTRANSACTIONAL` (real CDP mouse input with pre-press/pre-release revalidation; needed for isTrusted-gated elements and non-transactional by nature).

## v0.5 — authority integrity

A deterministic effect classifier maps every candidate to a semantic class (`NAVIGATE`, `FORM_EDIT`, `SUBMISSION`, `PURCHASE`, `UNKNOWN_COMMIT`, …) from bounded DOM context — a bare "Continue" is an `UNKNOWN_COMMIT`, and no learned adviser can downgrade a deterministic high-risk classification. Click execution re-verifies page identity, node identity, geometry, and hit-test *between* `mousePressed` and `mouseReleased`; text insertion runs atomically inside the isolated world. Evidence signatures are domain-separated with verification-key rotation, and a crash-torn final JSONL line is recovered without weakening hash/signature checks.

## v0.4 — qualification control plane

v0.2's isolated-world browser executor remains fixed and v0.3's bounded replay policy remains the only self-improvable surface. v0.4 adds cross-process hash-chain and policy-registry serialization, replay-evidence binding, train/validation/holdout candidate selection, matched live-canary evidence bound to post-staging runs and the staged parent digest, latency/action/token regression gates, policy lineage checks, suspension, health monitoring, and rollback. A replay winner still cannot activate itself.
