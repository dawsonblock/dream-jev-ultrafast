# Validation — Jev Ultrafast v0.4.1 DREAM-Jev

Validation date: 2026-09-29.

## Reproduced in this build environment

- `uv run pytest`: **114 passed** against the project environment with `browser-harness==0.1.13` installed; browser/CDP calls in unit tests remain mocked by the tests themselves.
- `uv run python scripts/check_guards.py` against a dedicated Chrome 152 instance (`BU_CDP_URL=http://127.0.0.1:9222`): **all 23 live browser guard checks passed**. The first live run exposed a latent defect: the guarded act/select/fill and post-input wait expressions concatenated their JSON argument without call parentheses, so every mutation path raised a `SyntaxError` that surfaced as `StalePage`. Unit mocks could not see this. The expressions were corrected to invoke their argument, `fresh()` now treats an unreachable/destroyed guard context as stale, and the live suite was re-run to green.
- `uv run ruff check .`: passed.
- `uv build`: passed (`dist/jev_ultrafast-0.4.1.tar.gz`, `dist/jev_ultrafast-0.4.1-py3-none-any.whl`).
- `python -m compileall -q jev_ultrafast tests examples scripts`: passed.
- `node --check jev_ultrafast/snapshot.js`: passed.
- `node --check jev_ultrafast/static/app.js`: passed.
- `shasum -a 256 -c MANIFEST.sha256`: all 55 tracked files verify.
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
- replay improvement refuses pools that mix DREAM TCB generations (new events are `jev-ultrafast-tcb/0.5`; pre-unification traces recorded `goal_overlap` with the legacy tokenizer);
- health decisions mark insufficient observed coverage (`sufficient=false`) separately from drift outcomes;
- promotion requires non-empty bound evidence digests in every path;
- isolated-world `(fn)(arg)` expression templates verified to invoke their argument;
- `fresh()` reports stale rather than raising when the guard context is destroyed or unreachable;
- `CanaryMetrics.from_events` treats a non-positive `max_runs` as "no runs" rather than slicing to the whole history;
- the select execution guard normalizes option values exactly as `snapshot.js` records them;
- registry writes are fsynced before the atomic rename; StalePage messages carry the underlying JS error.

## Deployment gate

Before production activation:

1. Run `uv sync --frozen` with network/package cache available.
2. Run `uv run ruff check .` and `uv run pytest` (both pass in this environment).
3. Run `uv run python scripts/check_guards.py` against the intended Chrome/Browser Harness installation.
4. Re-run representative browser benchmarks; the bundled v0.1 speed evidence is historical, not a fresh v0.4 latency claim.
5. Collect matched baseline/candidate canary runs across the intended site/task distribution. The built-in 12-run/four-family thresholds are minimum gates, not statistical proof.
6. Activate only through `jev-dream promote` / `PolicyRegistry.promote_from_store(...)`.
7. Run periodic `jev-dream health`; use suspension or rollback on meaningful post-promotion drift.
