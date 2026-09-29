# Validation — Jev Ultrafast v0.4.0 DREAM-Jev

Validation date: 2026-09-29.

## Reproduced in this build environment

- `pytest`: **68 passed** using a temporary external `browser_harness` import stub because Browser Harness is not installed in this offline environment. The stub is outside this repository; browser/CDP calls in unit tests remain mocked by the tests themselves.
- `python -m compileall -q jev_ultrafast tests examples scripts`: passed.
- `node --check jev_ultrafast/snapshot.js`: passed.
- `node --check jev_ultrafast/static/app.js`: passed.
- TOML parse for `pyproject.toml`: passed.
- JSON parse for bundled JSON evidence files: passed.
- DREAM CLI smoke: `verify` and `improve --stage` completed against synthetic hash-chained evidence; the staged record bound the parent policy, world-pool digest, split-manifest digest, evidence head, and TCB versions.
- Cross-process experience-store test: four spawned processes appended 40 events to one JSONL store and the resulting chain verified.

## Offline environment limitations

- `uv lock --check --offline`: could not resolve because `browser-harness==0.1.13` was not present in the local uv package cache. `uv.lock` itself has been updated to the local project version `0.4.0`; this is not a passing dependency-resolution result.
- `uv sync --frozen --offline`: failed because `typing-extensions==4.16.0` was absent from the local cache and network access is disabled.
- `uv build --offline`: failed because `hatchling` was absent from the local cache.
- `uv run ruff check .`: not reproduced because Ruff is not installed in the environment and offline dependency resolution is unavailable.
- The real Browser Harness / Chrome integration suite (`scripts/check_guards.py`) was not executed because Browser Harness is unavailable here.

These are environment limitations, not passing results. Run the full dependency, lint, and live-browser checks on the deployment machine before unattended use.

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
- end-to-end Agent → trace → replay verified-success integration with a fake browser boundary.

## Deployment gate

Before production activation:

1. Run `uv sync --frozen` with network/package cache available.
2. Run `uv run ruff check .`.
3. Run `uv run pytest` without the temporary import stub.
4. Run `uv run python scripts/check_guards.py` against the intended Chrome/Browser Harness installation.
5. Re-run representative browser benchmarks; the bundled v0.1 speed evidence is historical, not a fresh v0.4 latency claim.
6. Collect matched baseline/candidate canary runs across the intended site/task distribution. The built-in 12-run/four-family thresholds are minimum gates, not statistical proof.
7. Activate only through `jev-dream promote` / `PolicyRegistry.promote_from_store(...)`.
8. Run periodic `jev-dream health`; use suspension or rollback on meaningful post-promotion drift.
