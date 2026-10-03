# DREAM-Jev v0.4: evidence-bound replay improvement

Jev Ultrafast v0.4 keeps the v0.3 DREAM-Jev idea—use realized browser history as an empirical replay simulator—but hardens the path from historical evidence to an active exploration policy.

The conceptual source is **Dream-RSI: Recursive Self-Improvement through Evolving Worlds** (arXiv:2609.14858v1, 2026): online exploration creates structured history, the history becomes a replay world, candidate exploration policies are evaluated cheaply offline, and an improved policy is redeployed. DREAM-Jev preserves the important separation: the browser executor, safety policy, verifier, replay semantics, and promotion machinery stay fixed while only a bounded data-only `ExplorationPolicy` can change.

## Trust boundary

```text
                     USER GOAL
                        │
                        ▼
                ┌───────────────┐
                │ Jev Agent     │
                │ active π      │
                └──────┬────────┘
                       │
             finite legal actions
                       │
                       ▼
                ┌───────────────┐
                │ Policy gate   │  fixed TCB
                └──────┬────────┘
                       ▼
                ┌───────────────┐
                │ Browser       │  fixed TCB
                │ isolated world│
                └──────┬────────┘
                       │
                  real outcome
                       │
                       ▼
          cross-process hash-chained trace
                       │
                       ▼
              empirical ReplayWorlds
                       │
                 bounded π variants
                       │
                       ▼
          train / validation / holdout replay
                       │
                 replay gates pass?
                       │
                       ▼
                 STAGED π' + lineage
                       │
             matched live baseline/candidate
                       │
                       ▼
              bound CanaryEvidence
                       │
                 live gates pass?
                       │
                       ▼
                    ACTIVE π'
                       │
                  health monitor
                   /          \
               healthy      drift
                 │            │
                 ▼            ▼
              continue      suspend
                              │
                           rollback
```

## Self-improvable surface

`ExplorationPolicy` is the complete self-improvable surface. It contains bounded values for:

- goal-overlap ranking weight and overlap curvature (`overlap_exponent`);
- click/fill/select ranking bonuses;
- model-visible action budget;
- per-operation candidate quotas;
- per-DOM-node fan-out bound (`duplicate_node_cap`);
- minimum goal-overlap admission floor (`min_goal_overlap`);
- maximum browser-action budget;
- no-progress stopping patience.

All of these remain candidate-allocation parameters, so every mutation is still replay-qualifiable: the live selector and the replay filter share scoring and bounding semantics, and tests compare their catalogues identically. A learned reordering of which offered action the decision model *chooses* remains out of scope — replay cannot verify a different choice without fabricating the model's response.

A promoted policy is JSON data with a stable SHA-256 digest. The replay path cannot promote Python, JavaScript, selectors, coordinates, shell commands, browser permissions, credentials, approval rules, verifier logic, or executor code.

## Fixed trusted computing base

The following remain outside recursive improvement:

- CDP isolated-world node ownership;
- stale-state and semantic target guards, including pre-press and pre-release revalidation;
- browser mutation implementation;
- deterministic effect classification and the consequential-action approval/deny policy;
- privacy/redaction rules;
- completion-verifier contract;
- experience schema and hash validation;
- replay transition semantics and objective;
- replay/canary/health gates;
- policy registry and rollback logic.

A candidate may learn *how to allocate search effort*. It cannot learn itself more authority.

### Authority integrity (v0.5)

The approval boundary was strengthened in two places. `snapshot.js` now attaches a bounded `ctx` to every observed action — derived flags only (enclosing form, form method, submit membership, scoped field kinds such as password/money/file/search, the target's own field class, modal scope, external or messaging destination, download semantics), never free text, so the authority surface adds no privacy surface. `policy.py` classifies each mutation into a deterministic `Effect` and maps it to an authority floor; structure wins over labels, labels may only escalate, and commit-shaped controls whose effect cannot be determined classify `UNKNOWN_COMMIT → require_approval`. An optional adviser hook may escalate an `allow` to `require_approval`/`deny` but can never downgrade.

As of v0.5.1 the classifier is strictly monotonic: instead of early returns it collects every applicable effect and keeps the highest-authority one. This matters because input is itself a disclosure — page JavaScript observes every `input` event before any submit — so a `fill`/`select`/toggle whose target is a credential (`auth`/`otp`), payment (`money`), file, email/phone, or messaging field escalates to `AUTHENTICATE`/`FINANCIAL`/`SUBMISSION`/`DISCLOSURE`/`EXTERNAL_MESSAGE` rather than stopping at `FORM_EDIT`, and high-risk labels are evaluated before benign structural shortcuts on every kind.

The executor exposes two explicit execution guarantees rather than pretending physical input is transactional. `DOM_ATOMIC` — the default for click/fill/select — runs validation and mutation inside a single isolated-world evaluation: an atomic `e.click()`, or focus-verify + select-all + `execCommand('insertText')` + landed-value check for fills. No page JavaScript can interleave, so there is no press/release window to exploit. `TRUSTED_INPUT_NONTRANSACTIONAL` keeps real CDP mouse input for elements that need `isTrusted` activation; the isolated-world guard, geometry, and `elementFromPoint` hit-test are re-verified immediately before `mousePressed` and again before `mouseReleased`, a mid-press mutation still dispatches the release for pointer hygiene under `finally`, then fails closed as `IndeterminateMutation` — the click may already have landed, so the outcome is recorded as unknown and never retried. Elements that cannot be driven programmatically return `unsupported` before any mutation and escalate to the trusted path automatically. The residual risk is explicit in the model: trusted press/release is observable to page code mid-gesture, so high-authority effects are detected-and-aborted rather than transactionally prevented on that path.

## Durable experience store

`ExperienceStore` is append-only JSONL. v0.4 adds an operating-system lock file in addition to the in-process mutex. Two Jev processes writing the same store therefore serialize the append operation instead of reading the same chain head and creating a fork.

Every event contains:

```text
schema
TCB version
previous event hash
current event hash
run id / task key / sequence
run or transition payload
```

`load()` verifies the entire chain. `verify()` reports event count, head hash, schemas, TCB versions, and signature statistics. Current stores read `jev-dream/1`–`jev-dream/3` events; new events use `jev-dream/4` and `jev-ultrafast-tcb/0.12`. Older traces remain readable, but replay improvement refuses a pool that mixes TCB generations — the tokenizer used for recorded `goal_overlap` changed between them, the 0.10 generation adds experiment-tagged transitions with canary exclusion, 0.11 relabels propensity semantics plus adds `experiment_assigned` events, and 0.12 makes stamped experiment plans immutable and bound plus enriches assignments with the pre-treatment context the intention-to-treat estimator groups on, so pre- and post-change worlds are never scored together.

The bare chain is tamper-evident, not a digital signature: a party able to rewrite the whole file can recompute hashes. v0.4.1 therefore adds optional Ed25519 authenticity — each `event_hash` can be signed by a signer exposing `key_id`/`sign_hex` (`JEV_EVIDENCE_SIGNING_KEY`, or an injected signer so private material stays outside the agent process). As of `jev-ultrafast-tcb/0.7` signatures are domain-separated: the signed message is `jev-dream/evidence-event/v1:<event_hash>`, never the bare digest. Readers supply a trusted-key *set* (`JEV_EVIDENCE_VERIFY_KEYS`, `JEV_EVIDENCE_VERIFY_KEY`, or `--verify-keys`); each signed event carries its `key_id` and is verified under that key, so rotation keeps older segments valid while unknown keys fail closed, as do forged signatures and unsigned events under `require_signatures`/`JEV_REQUIRE_SIGNED_EVIDENCE`. Without a signer configured the store still works — signatures are an opt-in authenticity layer, and `verify()` reports how many events were signed and whether they were actually checked.

Crash recovery is deliberately narrow: a writer that dies mid-append leaves a final line that is a recognizable event-object prefix with no terminating newline. `load()` reports it (`verify()` sets `torn_tail_recovered`) and the next `append()` truncates it before continuing the chain — a torn fragment can never be a complete valid record, so truncation cannot destroy real evidence. Anything else at the tail — garbage, a non-object JSON value, or a parseable record whose hash or signature is invalid — remains fail-closed.

Signature authenticity still leaves one gap: event signatures prove the surviving records are genuine, but an attacker who can rewrite the file can replace a signed tail event with a torn-looking fragment and let recovery discard it — valid `A`, forged `{"…`, then valid `C` re-anchored by the next append. The optional **chain-head anchor** (`anchor_path=`/`--anchor`) closes it: every append writes a signed checkpoint (`jev-dream/chain-head-anchor/v1:` domain) recording the head and sequence after fsync. Reads — and appends, checked before writing so a writer cannot silently re-anchor over a gap — fail closed when the anchor is missing while the store holds events, the anchor digest/signature fails, or the anchored head disagrees with the recomputed log head. A crash exactly between event-fsync and anchor-write is the one ambiguity; it resolves to "anchor stale" and the operator re-anchors explicitly with `reanchor()` — never silently. Pointing `anchor_path` at separate storage (or syncing the file off-box) gives the external-anchor form; a local anchor already converts deletion-without-the-key into a detectable event.

Policy activation is signed the same way. A registry with a promotion signer (`JEV_PROMOTION_SIGNING_KEY`, falling back to the evidence key) attaches a `promotion_attestation` to the active record — a `jev-dream/promotion-attestation/v1:`-framed signature over the candidate digest, behavior digest, parent digest, replay-report/world-pool/split digests, canary evidence digest, evidence chain head, and registry revision. Since `jev-dream/4` the signature is additionally *bound to the record*: loading re-derives every signed field from the record itself — parent digest, replay digests, the whole canary block (`canary_digest`, which covers the health-check reference metrics), the promotion timestamp, and the revision — so the attestation authenticates this exact record, not merely a promotion statement floating beside it. A reader configured with verification keys (`JEV_PROMOTION_VERIFY_KEYS`, or the evidence keys) fails closed on any active record whose attestation is absent, malformed, unbound, or signed by an unexpected key, so a hand-written registry entry cannot bypass replay → staging → canaries → promotion. Rollback re-verifies the restored record's original attestation and, when a signer is present, attaches a signed `rollback_attestation` covering the transition. With no keys configured the registry remains digest-checked only — the explicit unsigned-compatibility mode.

Every registry mutation — stage, promote, suspend, resume, rollback — is itself a signed state transition under `jev-dream/4` and an explicit edge in the declared lifecycle (`POLICY_TRANSITIONS`): records carry a `status` (`staged`, `active`, `suspended`, `retired`), each mutation asserts the move is a legal edge, and a record whose declared status contradicts its slot fails closed at load — no edge mints authority outside `staged → active`. Each write chains `prev_state_digest → state_digest` over the entire payload, and under a signer the head carries a `jev-dream/registry-state/v1:` signature. Post-write edits to any field (including mutable ones like `suspended`) fail the digest, and a forged state cannot be resealed without the key. Suspend/resume/rollback additionally attach a per-record transition attestation chaining the signed statement to the head it replaces. Ed25519 proves authenticity, not freshness — so v0.8.0 restores the external freshness checkpoint: `PolicyRegistry(anchor_path=...)` checkpoints the signed state head after every write (`jev-dream/registry-head-anchor/v1:`-signed when a signer is configured), and every load fails closed when the file no longer reaches the anchored head — a wholesale restore of an older complete signed snapshot, a deleted registry, a missing anchor, and a forged unsigned anchor under configured keys all raise instead of silently rewinding. `registry.reanchor()` / `jev-dream registry-reanchor --registry R --registry-anchor A` is the explicit operator repair after a confirmed gap; nothing re-anchors silently.

## Transition evidence

A live transition records a privacy-bounded candidate catalogue and now binds four pieces of evidence:

- `candidates`/`candidate_digest`: the **pre-policy observed** catalogue in original order plus its SHA-256;
- `offered_digest`/`offered_count`: the post-policy catalogue actually shown to the model;
- `selected_observed_rank`/`selected_offered_rank`: the position of the action actually selected online in each catalogue — two different coordinates, both recorded explicitly since `jev-ultrafast-tcb/0.9` (the v0.6 `selected_rank` was the observed index and is never read as an offered rank);
- `selection_mode`/`behavior_propensity` plus the model-score fields `selected_model_score`/`joint_model_score`/`operation_probability`/`target_probability`/`decision_confidence` and the offered `action_probabilities`/`operation_probabilities` distributions. The executor is deterministic argmax, so `behavior_propensity` is `None` — a model score is not a sampling propensity and must never be used as an IPW denominator. The only real propensity a transition can carry is a randomized trial's `assignment_probability`. (v0.7.x wrote `selected_propensity` — a model score mislabeled as a propensity; `ReplayWorld` still reads it for backward compatibility but treats it only as a score.)
- `node`: the isolated-world node identity per candidate, so per-node bounds replay identically;
- `experiment` (since `jev-ultrafast-tcb/0.10`, on trial transitions and — since 0.11 — on `experiment_assigned`, `approval_required`, and `policy_denied` events): the counterfactual trial metadata — arm (`candidate`/`control`), `assignment_probability`, `state`, the recorded-vs-model action ids and kinds, and the frozen provenance binding (`proposal_digest`, `stamped_digest`, `choice_model_digest`, `policy_behavior_digest`, `offered_catalogue_digest`, `proposal_p_progress`, `proposal_uncertainty`, `expected_delta`, `experiment_id`). Since `jev-ultrafast-tcb/0.12` the assignment additionally freezes the pre-treatment context the intention-to-treat estimator groups on — `task_key`, `task_family`, `instance_id`, and each side's `goal_overlap`/`offered_rank` — so an assigned-but-never-executed trial is self-describing. `experiment_assigned` is recorded at randomization, *before* policy/approval/browser execution, so a trial that never produces a transition still marks the run experimental, is excluded from canary metrics, and still enters the trial estimator.

`ReplayWorld.from_events()` recomputes both digests — re-deriving the offered catalogue from the observed one under the recorded policy — and rejects mismatches in the digest, the offered count, or the recorded offered rank. Recording both catalogues removes the one-directional information bottleneck: a candidate can be replayed for *expansion* as well as contraction because the evidence preserves what was observed before the old policy filtered it. This does not make replay counterfactual; it makes the historical input to replay auditable.

## Conservative replay semantics

Each live run becomes one `ReplayWorld`. Runs are never stitched together into a synthetic trajectory.

For each recorded transition, a proposed `ExplorationPolicy` is applied to the *historical* candidate catalogue. Replay advances only if the action actually taken in the real run would still be offered. If the candidate policy would remove that action, the trajectory ends with a coverage miss.

DREAM-Jev therefore does **not** assert:

```text
"If D disappeared, the fixed decision model would have chosen C."
```

It only asserts:

```text
"D remained available, so the recorded D → next-state outcome can still be reused."
```

That is narrower than a learned latent world model, but it avoids manufacturing browser/model evidence.

## Evidence-bound replay report

A v0.4 `DreamReport` carries:

- replay-world-pool digest;
- deterministic split-manifest digest;
- current experience-store head hash when invoked through the CLI;
- TCB versions represented by the replay worlds;
- policy digests and per-split metrics for every bounded candidate.

A staged policy therefore records which historical pool and chain head supported the replay decision.

## Candidate selection across all splits

v0.3 selected the best training candidate and then checked it. v0.4 evaluates every bounded policy mutation on all available splits and selects only among candidates that pass the replay gates.

The gate requires:

- positive normalized training-objective improvement;
- minimum replay coverage;
- no verified-success regression;
- no risk-event regression;
- no normalized objective regression on validation or holdout when those splits exist.

Selection then prefers the candidate with the strongest worst-split gain, breaking ties with average gain. This makes an overfit training winner less likely to hide a slightly smaller but generalizing improvement.

All runs for a normalized task key remain in one partition to avoid task-family leakage.

## Replay objective

The default objective remains multi-objective:

```text
score =
    + 100 × verified task success
    + 20  × verified completion
    + 0.5 × page-progress transitions
    - 0.25 × seconds of model latency
    - 0.15 × browser actions
    - 0.10 × model calls
    - 0.02 × thousand tokens
    - 0.002 × candidate controls exposed
    - 2.0 × stale/failure events
    - 5.0 × risk events
    - 4.0 × coverage misses
```

The safety boundary is not represented by this scalar objective. A high score cannot override browser guards or approval requirements.

## Staging lineage

`PolicyRegistry.stage()` binds the candidate to the policy that was active when replay was performed. If the active policy changes before staging, the replay report is stale and staging fails. A staged record contains:

```text
candidate digest
parent policy digest
replay report hash
world-pool digest
split-manifest digest
experience head hash
TCB versions
```

Promotion checks the parent again. A staged policy therefore cannot silently leap over an intervening deployment.

## Bound live canaries

Production promotion is through:

```bash
uv run jev-dream promote .jev/experience.jsonl \
  --registry .jev/policy-registry.json
```

`promote_from_store()` derives `CanaryEvidence` from the hash-verified event store. Baseline and candidate runs are grouped by task key so the gate can inspect matched task families rather than only aggregate success.

Default v0.4 gates require:

- at least 12 baseline runs;
- at least 12 candidate runs;
- at least four candidate task families;
- at least four paired baseline/candidate task families;
- at least 80% of candidate task families represented in the paired set;
- no aggregate verified-success regression;
- no failure-rate regression;
- no extra risk events;
- the candidate must not lose more paired task families than it wins;
- an exact two-sided sign test over the paired instances at `max_pair_sign_p` (default 0.05) — since a sign test ignores ties, this bound needs at least six non-tied pairs all won by the candidate, so the qualifying canary must exceed the raw run minimums;
- no more than 25% regression in average live model latency, action count, or token use.

These defaults are now a significance check, not only a guardrail — but a p ≤ 0.05 sign test over paired runs is still a minimum. Higher-risk deployments should increase sample size and tighten the bound. `max_pair_sign_p=None` explicitly waives the significance floor for controlled testing; leaving it unset keeps it on.

The stored activation record includes the event-store head hash, paired-task counts, run IDs inside the metrics, and an `evidence_digest` over the canary evidence.

## Unbound metrics are disabled by default

`PolicyRegistry.promote(baseline_metrics, candidate_metrics)` remains only as an explicit compatibility escape hatch, feature-gated twice: it raises unless `allow_unbound_metrics=True` is supplied **and** `JEV_ALLOW_UNBOUND_METRICS=1` is present in the environment, so no code path reaches unbound promotion silently. Normal activation should use `promote_from_store()` so the evidence is tied to concrete hash-verified runs.

## Health monitoring and suspension

A policy that passed canaries can still regress as websites drift. v0.4 stores the candidate canary metrics as the health reference and adds:

```bash
uv run jev-dream health .jev/experience.jsonl \
  --registry .jev/policy-registry.json \
  --recent-tasks 20 \
  --suspend-on-fail
```

`HealthGate` compares recent active-policy traces with the promotion reference. By default it waits for 20 recent tasks (`sufficient=false` distinguishes "not enough evidence yet" from a clean pass), allows at most a 10-percentage-point verified-success regression, a 15-point failure-rate regression, and no increase in risk-event rate, with latency/action/token drift monitors at 1.5× the reference averages and a hard zero-verified-success collapse check. Attrition is evaluated *before* task sufficiency: runs the policy started but never finished (`abandoned_runs`) and the candidate-attributable subset (`crash_runs` — browser crashes, agent exceptions, indeterminate executions) are counted under the same membership filter the promotion path uses, so a policy dying on every start cannot hide behind "insufficient data".

If `--suspend-on-fail` is used, the active learned policy is marked suspended. `Agent(policy_registry=...)` then falls back to the baseline `ExplorationPolicy` without changing the browser executor or deleting the learned policy record.

## Rollback

Every superseded active policy is retained in registry history. Roll back with:

```bash
uv run jev-dream rollback --registry .jev/policy-registry.json
```

or select a specific historical digest with `--digest`. Rollback also clears any staged policy so stale qualification cannot subsequently activate over the restored parent.

## CLI

```bash
# Replay + report; optionally stage
uv run jev-dream improve EXPERIENCE --report REPORT --registry REGISTRY --stage

# v0.3 compatibility: treated as "improve"
uv run jev-dream EXPERIENCE --report REPORT --registry REGISTRY --stage

# Verify event chain
uv run jev-dream verify EXPERIENCE

# Promote from bound live evidence
uv run jev-dream promote EXPERIENCE --registry REGISTRY

# Inspect registry
uv run jev-dream status --registry REGISTRY

# Post-promotion health
uv run jev-dream health EXPERIENCE --registry REGISTRY --recent-tasks 20 --suspend-on-fail

# Manual circuit breaker / recovery
uv run jev-dream suspend --registry REGISTRY --reason "site drift"
uv run jev-dream resume --registry REGISTRY
uv run jev-dream rollback --registry REGISTRY [--digest SHA256]

# Registry head anchor: every registry command accepts --registry-anchor PATH
# to checkpoint/verify the signed state head; after a confirmed gap, repair
# explicitly with:
uv run jev-dream registry-reanchor --registry REGISTRY --registry-anchor PATH

# Randomized causal evidence: cell inventory, or one divergence resolved
# against the treatment signature (read-only; the same verification flags
# as verify/improve apply)
uv run jev-dream trials EXPERIENCE [--cells]
uv run jev-dream trials EXPERIENCE \
  --task-family F --site HOST \
  --model-kind KIND --model-effect EFFECT --model-role ROLE \
  --model-overlap N --model-rank N \
  --proposal-kind KIND --proposal-effect EFFECT --proposal-role ROLE \
  --proposal-overlap N --proposal-rank N \
  --phase N [--min-effect D]
```

`trials` is the read side of the causal layer. With no signature flags it reports the fitted cell count, evidence version, and duplicate assignments; `--cells` additionally decodes every cell's stratum, signature, arm, and assignment/censoring counters. With any signature flag it runs `CounterfactualTrials.resolve()` and prints the winning stratum (`level`), the backoff mask that answered (`signature_level`), per-arm counts, and the α-spent effect verdict — `null` when no stratum can answer. Omitted coordinates are wildcards; `kind` and `effect` are floors that stored `unknown` values never satisfy, so an unanswered question prints `null` rather than borrowing unrelated evidence.

## The causal decision-learning plane (v0.9.0)

v0.8.3 built the randomized channel; v0.9.0 makes it *epistemically correct*. The loop is unchanged in authority — predictions may suggest, only empirical results authorize — but the causal layer no longer confuses "we have samples" with "we know the intervention helps":

```text
normal decision model
        │
        ▼
   model chooses A
        │
        ├────────── observational ChoiceModel (source: "observational")
        │
        └────────── causal TrialChoiceModel (source: "randomized")
                          │
                          ▼
                 ExperimentPlan A vs B
              (bound, family-bound, optionally signed)
                          │
                   random assignment
                   /              \
             control A         candidate B
                   \              /
                    REAL browser
                         │
                  authority plane (unchanged)
                         │
                independent verifier
                         │
                         ▼
                 final trial outcome
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
    success          failure          censored + reason
        │                                 (operator_cancel, timeout, …)
        ▼
   causal estimator
        │
   hierarchical context (family+site → site → family → pooled)
        │                (treatment signature backoff: phase → rank → role → overlap; kind and effect are floors)
        ▼
   effect status: BENEFICIAL / HARMFUL / UNRESOLVED / INSUFFICIENT_DATA
        │
        ▼
   ExperimentScheduler → next real experiment
```

- **Effect status is not support.** `support_sufficient` is the weighted-sample floor; `effect_status` is the sign the data actually established, decided by `delta_cs` — an anytime-valid confidence sequence: per-arm KL confidence bounds over the self-normalized IPW statistics, union-bounded across looks by a summable α-spending schedule so coverage holds at every sample size simultaneously under unlimited peeking. The family's error budget is Bonferroni-split across the *registered* hypothesis family (`hypothesis_count`) before per-look spending: since `jev-trials/10`, every parseable `experiment_assigned` record registers its hypothesis context *before* outcomes — including declarations whose units were later quarantined (`invalid_units`), rejected as malformed, or fully censored — and the family is the registered contexts unioned with observed cells, so a declared-but-unanalyzable test still pays its share of α rather than vanishing from the denominator. Each contrast also reports `hypothesis_registered` (both arms declared the answering context — every merged context, when backoff folds several together) as preregistration provenance, and `CounterfactualTrials.hypothesis_summary` exposes `declared` / `registered_comparisons` / `observed` / `declared_unobserved` / `unregistered_observed` / `family_size`. Stores written before registration existed keep the cell-derived family and honestly report `hypothesis_registered: False`. A supported interval that crosses zero is `unresolved` — it schedules more evidence and proposes nothing. Only `beneficial` feeds `TrialChoiceModel.choose`; only `harmful` feeds `refuted`. The fixed-sample Newcombe `delta_ci` is retained for reporting.
- **Censoring is causal.** Aborts carry structured reasons, per-arm censor rates are reported, and extreme or sharply imbalanced censoring refuses to establish an effect — the surviving subset of a differentially censored arm is a biased sample, not a smaller unbiased one. Censored mass is also *bounded rather than assumed ignorable*: each arm carries its censored weight, estimates emit `delta_bounds` (every censored unit re-counted as failure, then as success), and establishment requires that bound to agree with the sequence — a nominal `beneficial` that could be erased by unfavorable censoring reports `unresolved: censoring_bounds_cross_threshold`.
- **Generalization is bounded and named.** An estimate applies to a treatment *signature* — kind, effect class, role, overlap, offered rank, workflow phase — inside a task/site context; the answer reports which stratum and which backoff mask produced it. It is never presented as an exact-action measurement, and operation kind is never generalized across. Since `jev-trials/9` each resolved contrast additionally reports *exact-action* provenance: `fit` indexes every randomized proposal's `action_key` (a stable hash of kind + effect class + role + the same `redact_text`-normalized label the evidence store records; `None` when the identity is unrecoverable, never invented), and `resolve` emits `generalization_level` — `exact` (this very action identity was the candidate arm inside the answering stratum under the enforced signature), `same_context_class` (context-specific class evidence only), `cross_context_class` (the action was randomized somewhere, but only pooled class evidence answers here), `pooled_class` (never randomized anywhere, pooled class evidence), or `none` (the query supplied no provable action identity — distinct from provably class-only evidence) — plus `exact_action_randomized` (mirrors `exact` exactly: deployable-precision evidence only), `action_randomized_anywhere`, `action_randomized_in_context` (the action's own randomized presence inside the answering context — the evidence-derived contextual-canary record), `context_randomized`, `treatment_class_randomized`, and `action_key`. Exactness is proven, never assumed.
- **Plans are bound, fresh, and can be signed.** The live agent re-verifies every stamped binding including `family_key`, discards plans that newer randomized evidence refutes, and — when trusted keys are configured — requires a domain-separated Ed25519 signature, so digest integrity is never mistaken for provenance.
- **The policy layer is operator-gated.** `CausalChoicePolicy` runs `shadow` (annotate only) → `canary` (randomized assignment) → `active` (deterministic override, but only for randomized-established beneficial proposals, still through the authority plane, recorded as `causal_policy_applied`, and excluded from policy-canary qualification). Mode changes are configuration decisions, never learned ones. A proposal's implied success probability is propagated from the estimation layer as *measured control-arm rate + measured delta* (clamped to [0, 1]) — `TrialChoiceModel` entries carry `control_p`/`p_progress`/`uncertainty`/`delta_ci` and the policy consumes them verbatim; no consumer reconstructs a probability from an assumed baseline, and a supported contrast that lacks a valid finite control probability yields `p_progress: null` rather than a fabricated value. Causal and observational probabilities stay on the proposal under separate names (`causal_p_progress` / `observational_p_progress`) — never mixed.
- **Active authority is stratum-gated.** The modes draw an explicit evidence hierarchy: `shadow` may inspect every evidence level; `canary` may let *pooled* randomized evidence nominate a candidate — but only through the recorded randomized assignment (a canary is how pooled evidence earns context-specific support); `active` requires an established beneficial estimate from a context-specific stratum — `family+site`, `site`, or `family` by default (`active_trial_levels`), so pooled evidence alone can never drive a deterministic substitution in a context it did not measure — *and* evidence whose `generalization_level` the policy accepts for that action identity (`active_generalization_levels`, default `("exact",)`). An action whose exact identity was never randomized cannot deploy on class-level evidence alone: the bridge is a completed contextual canary of *that action* — derived from the trial store itself (`action_randomized_in_context`: the action was the randomized arm inside the answering context's scope) or, for confirmations evidenced outside this model, listed in `confirmed_action_keys`. Executable requires *all* of: supported randomized evidence, `beneficial` status, `expected_delta` above `min_causal_delta`, a valid causal probability, an allowed trial level, an allowed or canary-confirmed generalization level, and membership in the offered catalogue. Every proposal entry names why it cannot execute — `no_causal_evidence`, `insufficient_support`, `unresolved`, `below_effect_threshold`, `missing_probability`, `pooled_only`, `unsupported_stratum`, `unverified_generalization`, or `harmful` (an established-harmful divergence is refused, never re-nominated) — and unknown names in either allow-list fail closed at construction. Scores are separated by purpose — `causal_score`, `observational_score`, `experiment_priority_score`, `deployment_score` — and `active` ordering uses causal terms only, so an observational prior can annotate but never outrank superior causal evidence; provenance is reported as independent dimensions (`evidence_origin`, `trial_level`, `support_status`, `effect_status`, `generalization_level`), not a single overloaded label — an unsupported pooled contrast is `randomized`/`pooled`/`insufficient`, never silently "randomized" as if context-specific. The intended ladder: pooled evidence → hypothesis → randomized contextual canary → context-specific evidence → active use.

Hard safety constraints remain outside every utility function: no weight combination can trade away an effect classification, a payload review, an approval, or a browser guard.

## What this still is not

DREAM-Jev remains empirical replay, not a latent browser-dynamics model. It cannot predict unseen DOM transitions, infer the outcome of an action never executed, or prove that historical page behavior still holds today. Live canary qualification is therefore mandatory by design.

v0.4.1 introduces the first two lower-trust learned layers, both deliberately outside the evidence boundary:

- `CostModel` (Level 1) learns how tokens and latency scale with offered-candidate count. Replay reports anchored `estimated_*` metrics — the recorded measurement plus the learned marginal delta — which prioritize *which replay-approved candidate gets staged first*. The estimates never enter a gate: a coverage-missing or regressing candidate stays rejected no matter how efficient it looks.
- `OutcomeModel` (Level 2) learns bucketed `P(page_changed)` priors over `(kind, overlap, rank)` cells and annotates candidates with expected progress, useful for designing live experiments.
- `ChoiceModel` (Level 3, v0.6.0; corrected in v0.6.1) is the first *counterfactual* layer: an uncertainty-aware prior over the same cells that proposes which offered action it would have preferred (`choose` scores posterior mean + an exploration bonus × posterior stddev, so promising-but-sparse actions stay reachable, while the `confident` flag abstains on cells without support). Two corrections landed in v0.6.1: the model trains on the action's rank in the catalogue the policy *offered* (`selected_offered_rank`), not the pre-policy observed index — the v0.6 recorder wrote the observed coordinate under `selected_rank`, so training and replay were on different bases — and the learned target is the run's verified outcome, not raw page activity. As of v0.8.0 (`jev-choice/3`) the labeling is trajectory-success again: every step of a run takes the run's final independently verified outcome, so intermediate `page_changed` churn inside a failed run is a negative association — correlational trajectory credit, not per-action causality. Since v0.8.2 (`jev-choice/4`) runs with no measured outcome — `aborted`, `claimed_done` without a verifier, torn tails with no `run_finished` — are censored out of the fit entirely rather than counted as failures; a run that was never measured is missing data, not evidence its actions were bad. During replay the prior is queried under the candidate policy's own recomputed offered rank, so its annotations are policy-dependent rather than a descriptive prior over history. It reports per-candidate `divergence_rate` (how often it would have chosen differently) with the two subjects kept separate — `selected_mean_uncertainty`/`selected_confident_fraction` for the recorded action's prior and `proposal_mean_uncertainty`/`proposal_confident_fraction` for the proposal's. It is still a correlational prior over *selected* actions, not a causal success model — outcomes for rejected candidates are not in the data — and its proposals remain annotation metadata: never an outcome, never evidence, never a gate input. Since v0.8.3 (`jev-choice/5`) the prior is hierarchical: a task-family stratum over the same cells answers when it meets `MIN_CONFIDENT`, otherwise the pooled cell does — unrelated task families no longer share one tiny posterior — and experiment-tagged transitions are excluded from the fit, since a scheduler-assigned action is not the recorded policy's own choice.
- `CounterfactualTrials` (Level 4, v0.7.0; endpoint corrected in v0.8.0/`jev-trials/2`, assignment-based in v0.8.2/`jev-trials/3`) closes the loop between "the prior prefers B" and "B has actually been tried". `DreamReport.experiment_proposals` surfaces divergent replay annotations as stamped `jev-experiment-plan/1` hypotheses ranked by expected delta; `Agent(experiment={"proposals": [...], "rate", "rng"})` consumes those stamped plans directly, while `Agent(experiment={"model", "rate", "rng"})` asks a live `ChoiceModel` — either way the proposal must be a member of the policy-filtered offered catalogue, at most one deviation per run so outcomes stay attributable, and the assigned action passes through the unchanged authority plane (policy assessment, approvals, payload review). A stamped plan is *immutable*: its digest binds the hypothesis plus the task key, state fingerprint, model choice, offered catalogue, policy behavior, originating model, and pool/head lineage — the agent re-verifies every binding before executing, and a state-matching plan that fails any binding is discarded as stale (recorded `experiment_plan_stale`) and suppresses ad-hoc substitution for that step rather than silently running a different hypothesis. Randomization records `experiment_assigned` *before* the authority plane — a denied, vetoed, or stale trial still marks the run experimental — and trial transitions carry the frozen assignment binding (arm, `assignment_probability`, proposal/model/policy/catalogue digests, predicted delta and uncertainty).

  `CounterfactualTrials.fit` analyzes *assignments*, not just executed transitions — intention-to-treat: an arm that was assigned but never executed still counts under its run's real terminal outcome, so post-randomization selection cannot drop the losing half of a trial. The primary endpoint `p_success` is the run's final independently verified outcome — a mid-run page move inside a failed run is a failure, not progress (`p_page_changed` remains secondary telemetry over executed trials only); runs with no measured outcome (`aborted`, interrupted, unverifiable `claimed_done`) are *censored* — counted and excluded rather than treated as failures. Estimates report assigned/analyzed/executed/censored counts (with structured termination reasons since v0.9.0), Wilson intervals on effective sample size, and `support_sufficient` against a raised `MIN_ESS = 8` floor; context keys on the model choice's pre-randomization features so both arms share a cell. Since v0.9.0 the context stores task family and site as separate coordinates plus a bounded treatment signature (`jev-trials/7` additionally carries per-arm censored weight for the conservative bounds; `jev-trials/8` appends a `jev-outcome-vector/1` block — verified-transition/recovery/steps secondary rates, per-run authority-touch and guard-failure safety rates — where secondary contrasts are diagnostics pinned out of the establishment machinery and a measured safety excess is a vetoing regression, and cells migrated from ≤7 report `coverage: 0`/`None` rather than fabricated zeros; legacy cells migrate with `"unknown"` wildcards and an imputed censor mass), and `resolve` walks `family+site → site → family → pooled` and backs off over signature coordinates (phase → rank → role → overlap, with kind *and* effect class as floors — evidence never generalizes across semantic effects), returning the first supported stratum so a thin specific stratum cannot shadow a settled broader refutation — and an estimate that had to generalize says so via `signature_level`. `jev-trials/9` enforces the experimental-unit invariant rather than trusting it: at most one *distinct* randomized assignment per run (identical rewrites of one assignment still deduplicate under `duplicates`; two different decisions quarantine the whole run under `invalid_units`, as does an assignment ordered after the run's terminal event; malformed arms or propensities outside (0, 1] are `rejected` before they can shape a cell), and every candidate-arm assignment indexes its proposal's `action_key` so `resolve` can report exact-action generalization provenance (see above) — assignments stamped by the agent carry `proposal_action_key` directly, and older evidence recovers the identity from the executed transition's catalogue when it can. `jev-trials/10` then treats every parseable assignment as a *registration*: the declared hypothesis family — including quarantined, rejected, and fully censored declarations — bounds the multiplicity denominator (see the error-budget paragraph above), and each contrast reports whether it was a preregistered comparison (`hypothesis_registered`). Trials are excluded from `CanaryEvidence` — a trial never counts toward its own promotion — so the loop stays honest: hypotheses generate real data, real data sharpens the prior, and anything the prior prefers must still qualify through replay + bound live canary.
- `TrialChoiceModel` (Level 4.5, v0.8.3/`jev-causal/1`; v0.9.0/`jev-causal/2`) is the causal decision prior built *only* from `CounterfactualTrials` evidence — never observational transitions. Given the model choice as premise, `choose` proposes the offered candidate whose arm *established* a beneficial intention-to-treat effect (confidence sequence clear of the practical threshold, with censoring bounds in agreement); `refuted` marks a divergence whose effect is established harmful. A supported interval that crosses zero is `unresolved` — not a proposal, not a refutation. Replay proposal selection is causal-first: measured evidence proposes, the observational prior fills in only where trials are silent and not refuted, and the stamped plan's originator digest binds whichever model produced the hypothesis. Same advisory contract as every learned layer — proposals are hypotheses, never evidence, never gate input.

A future checkpointed branch orchestrator or learned world model could propose new exploration hypotheses. Such components remain hypothesis generators and are never allowed to satisfy execution, verification, or promotion gates with synthetic outcomes alone.
