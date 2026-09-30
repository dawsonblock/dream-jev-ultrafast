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

The approval boundary was strengthened in two places. `snapshot.js` now attaches a bounded `ctx` to every observed action — derived flags only (enclosing form, form method, submit membership, scoped field kinds such as password/money/file/search, modal scope, external or messaging destination, download semantics), never free text, so the authority surface adds no privacy surface. `policy.py` classifies each mutation into a deterministic `Effect` and maps it to an authority floor; structure wins over labels, labels may only escalate, and commit-shaped controls whose effect cannot be determined classify `UNKNOWN_COMMIT → require_approval`. An optional adviser hook may escalate an `allow` to `require_approval`/`deny` but can never downgrade.

In the executor, input is bound to the same semantic target that passed authorization: the isolated-world guard, geometry, and `elementFromPoint` hit-test are re-verified immediately before `mousePressed` and again before `mouseReleased`. A mid-press mutation still dispatches the release for pointer-state hygiene, then fails closed to re-perception. Fills verify focus and run `execCommand('insertText')` inside one isolated-world evaluation, and a landed value that differs from the authorized text is a non-retryable error rather than a silent retry.

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

`load()` verifies the entire chain. `verify()` reports event count, head hash, schemas, TCB versions, and signature statistics. Current stores read `jev-dream/1`–`jev-dream/2` events; new events use `jev-dream/3` and `jev-ultrafast-tcb/0.7`. Older traces remain readable, but replay improvement refuses a pool that mixes TCB generations — the tokenizer used for recorded `goal_overlap` changed between them, so pre- and post-unification worlds are never scored together.

The bare chain is tamper-evident, not a digital signature: a party able to rewrite the whole file can recompute hashes. v0.4.1 therefore adds optional Ed25519 authenticity — each `event_hash` can be signed by a signer exposing `key_id`/`sign_hex` (`JEV_EVIDENCE_SIGNING_KEY`, or an injected signer so private material stays outside the agent process). As of `jev-ultrafast-tcb/0.7` signatures are domain-separated: the signed message is `jev-dream/evidence-event/v1:<event_hash>`, never the bare digest. Readers supply a trusted-key *set* (`JEV_EVIDENCE_VERIFY_KEYS`, `JEV_EVIDENCE_VERIFY_KEY`, or `--verify-keys`); each signed event carries its `key_id` and is verified under that key, so rotation keeps older segments valid while unknown keys fail closed, as do forged signatures and unsigned events under `require_signatures`/`JEV_REQUIRE_SIGNED_EVIDENCE`. Without a signer configured the store still works — signatures are an opt-in authenticity layer, and `verify()` reports how many events were signed and whether they were actually checked.

Crash recovery is deliberately narrow: a writer that dies mid-append leaves a final line that is a recognizable event-object prefix with no terminating newline. `load()` reports it (`verify()` sets `torn_tail_recovered`) and the next `append()` truncates it before continuing the chain — a torn fragment can never be a complete valid record, so truncation cannot destroy real evidence. Anything else at the tail — garbage, a non-object JSON value, or a parseable record whose hash or signature is invalid — remains fail-closed.

## Transition evidence

A live transition records a privacy-bounded candidate catalogue and now binds four pieces of evidence:

- `candidates`/`candidate_digest`: the **pre-policy observed** catalogue in original order plus its SHA-256;
- `offered_digest`/`offered_count`: the post-policy catalogue actually shown to the model;
- `selected_rank`: the position of the action actually selected online;
- `node`: the isolated-world node identity per candidate, so per-node bounds replay identically.

`ReplayWorld.from_events()` recomputes both digests — re-deriving the offered catalogue from the observed one under the recorded policy — and rejects mismatches. Recording both catalogues removes the one-directional information bottleneck: a candidate can be replayed for *expansion* as well as contraction because the evidence preserves what was observed before the old policy filtered it. This does not make replay counterfactual; it makes the historical input to replay auditable.

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
- no more than 25% regression in average live model latency, action count, or token use.

These defaults are a guardrail, not statistical proof. Higher-risk deployments should increase sample size and require domain-specific confidence criteria.

The stored activation record includes the event-store head hash, paired-task counts, run IDs inside the metrics, and an `evidence_digest` over the canary evidence.

## Unbound metrics are disabled by default

`PolicyRegistry.promote(baseline_metrics, candidate_metrics)` remains only as an explicit compatibility escape hatch. It raises unless `allow_unbound_metrics=True` is supplied. Normal activation should use `promote_from_store()` so the evidence is tied to concrete hash-verified runs.

## Health monitoring and suspension

A policy that passed canaries can still regress as websites drift. v0.4 stores the candidate canary metrics as the health reference and adds:

```bash
uv run jev-dream health .jev/experience.jsonl \
  --registry .jev/policy-registry.json \
  --recent-tasks 20 \
  --suspend-on-fail
```

`HealthGate` compares recent active-policy traces with the promotion reference. By default it waits for 20 recent tasks (`sufficient=false` distinguishes "not enough evidence yet" from a clean pass), allows at most a 10-percentage-point verified-success regression, and permits no increase in risk-event rate.

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
```

## What this still is not

DREAM-Jev remains empirical replay, not a latent browser-dynamics model. It cannot predict unseen DOM transitions, infer the outcome of an action never executed, or prove that historical page behavior still holds today. Live canary qualification is therefore mandatory by design.

v0.4.1 introduces the first two lower-trust learned layers, both deliberately outside the evidence boundary:

- `CostModel` (Level 1) learns how tokens and latency scale with offered-candidate count. Replay reports anchored `estimated_*` metrics — the recorded measurement plus the learned marginal delta — which prioritize *which replay-approved candidate gets staged first*. The estimates never enter a gate: a coverage-missing or regressing candidate stays rejected no matter how efficient it looks.
- `OutcomeModel` (Level 2) learns bucketed `P(page_changed)` priors over `(kind, overlap, rank)` cells and annotates candidates with expected progress, useful for designing live experiments.

A future checkpointed branch orchestrator or learned world model could propose new exploration hypotheses. Such components remain hypothesis generators and are never allowed to satisfy execution, verification, or promotion gates with synthetic outcomes alone.
