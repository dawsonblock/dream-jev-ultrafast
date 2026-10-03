# Baseline Test Results — Phase 0 snapshot

All commands actually executed on this host (Darwin arm64, python
3.12.9 via `uv`, node v24.16.0) at commit `60f35f0` — the post-split
tree. Counts are copied from real output; nothing is inferred.

## Results

| check | command | result |
| --- | --- | --- |
| byte-compile | `uv run python -m compileall -q jev_ultrafast tests` | exit 0 |
| lint | `uv run ruff check .` | exit 0 — after `--fix` repaired 2 pre-existing test-file import issues (I001 unsorted import block, F401 unused `_REASON_BASE` import) |
| unit/integration suite | `uv run pytest -q` | **490 passed, 0 failed, 0 skipped** in ~16 s; coverage **87.44%** (floor 80%) |
| manifest integrity | `uv run python scripts/update_manifest.py --check` | exit 0 — **100 entries verified** |
| JS syntax | `node --check jev_ultrafast/static/app.js` | exit 0 |
| JS syntax | `node --check jev_ultrafast/snapshot.js` | exit 0 |
| env digest | `uv run python scripts/env_digest.py` | exit 0 — digests recorded in `BASELINE_ARCHITECTURE.md` |

## Unavailable on this host

| check | status | reason |
| --- | --- | --- |
| `scripts/check_guards.py` | UNAVAILABLE | no CDP endpoint reachable (no Chrome with remote debugging) |
| `scripts/e2e_check.py` | UNAVAILABLE | no CDP endpoint reachable |
| `scripts/race_check.py` | UNAVAILABLE | no CDP endpoint reachable |
| `--full` release qualification | UNAVAILABLE | `MANIFEST.sig` absent; no Chrome; full Q3/Q4 volumes not run |

## Coverage detail (from pytest run)

`jev_ultrafast/` total: **87.44%** — 4283 statements, 538 missed.
Weakest: `demo.py` (0%), `browser.py` (56%), `dream_cli.py` (72%) —
all require a live browser/backend to exercise.

## Notes

- `git status` was clean at snapshot except the two ruff fixes listed
  above (`tests/test_causal_validity.py` import block only).
- `MANIFEST.sig` is absent: the tree is unsigned, so
  `provenance_status` would report `unsigned` and `release_qualified`
  would be `false` under `--full`.
