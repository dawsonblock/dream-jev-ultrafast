# Jev Ultrafast v0.3.0 — DREAM-Jev

v0.3.0 keeps the v0.2 execution-integrity architecture and adds bounded recursive improvement of exploration strategy through historical replay.

## Added

- `ExplorationPolicy`: data-only, validated exploration parameters.
- Goal-aware live candidate selection driven by the active exploration policy.
- `ExperienceStore`: append-only, SHA-256 hash-chained JSONL evidence.
- `DreamTraceRecorder`: privacy-bounded task and transition logging.
- `ReplayWorld` / `ReplaySimulator`: one world per recorded run; proposed policies may filter historical candidate catalogues but replay never invents a different model choice or unseen browser outcome.
- Multi-objective replay scoring covering verified success, cost, latency, failures, risk, and replay coverage.
- Deterministic policy mutation constrained to whitelisted knobs.
- Deterministic train/validation/holdout world splitting.
- `PromotionGate`: replay score, verified-success, coverage, and risk gates.
- `PolicyRegistry`: atomic staging/activation of data-only policies with digest verification.
- `CanaryGate`: real-execution requirement before a staged policy becomes active.
- `jev-dream` CLI for replay evaluation and staging.
- Additional privacy handling for bare `token=` URL query parameters.
- DREAM-Jev design documentation and regression tests.

## Deliberately unchanged

DREAM-Jev cannot modify browser execution code, isolated-world identity, freshness guards, approval rules, redaction rules, verifier semantics, or promotion code. These remain part of the fixed trusted computing base.

## Evidence boundary

Historical replay only reveals recorded outcomes. If a candidate policy chooses an action for which no historical transition exists, the trajectory ends with a coverage miss. v0.3.0 does not claim a learned browser world model.
