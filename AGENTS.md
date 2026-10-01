# Jev Ultrafast

Read README.md before editing. Keep the loop small: page -> indexed elements -> operation + target -> execution.

- The input is one natural-language goal. Do not add site-specific plans or hardcoded field values.
- TypeSafe chooses an operation and operation-specific target heads in one request. Consume only the selected operation's target.
- Targets must map to observed elements and supported operations. Never let the model emit selectors or executable code.
- TYPE_TEXT invokes the text LLM. Cache a stale retry's value only while its entire helper input is identical.
- Never retry a browser mutation. Log execution before observing its result.
- Screenshots are optional; the model does not consume them. Keep demonstration footage at its original speed.
- Keep credentials server-side and .env ignored. Tests must not call paid APIs.
- Verify actual final outcomes independently. A DONE choice is not proof of success.
- Keep examples, README claims, raw evidence, and model-call counts consistent.
- Do not commit or push unless the user requests it.

DREAM-Jev invariants:
- Recursive improvement may change only validated `ExplorationPolicy` data. Never self-modify browser execution, approval, privacy, verifier, replay scoring, evidence validation, or promotion code.
- Replay is empirical. A proposed policy may filter a recorded candidate catalogue, but must not fabricate a different model choice or browser outcome.
- Learned models (`CostModel`, `OutcomeModel`, `ChoiceModel`, `TrialChoiceModel`) annotate and prioritize only. A counterfactual choice proposal is a hypothesis — never an outcome, never evidence, never a gate input. Anything a learned model prefers must still qualify through real executions and bound live canaries.
- `TrialChoiceModel` is the causal channel: it reads only randomized `experiment_assigned` evidence through `CounterfactualTrials`, never the observational trace. Keep the observational and causal priors separate — never mix trial transitions into `ChoiceModel` fits.
- Learning context is hierarchical: task-family/site strata answer before pooled estimates, and a thin specific stratum may defer to a reliable broader one — never pool unrelated task families into one cell as if they were interchangeable.
- Keep runs as separate replay worlds. Keep all runs for one task key in the same train/validation/holdout partition.
- Replay approval is provisional. A changed policy must be staged and separately pass bound live-canary evidence before activation.
- Experience-store appends must remain cross-process serialized; never weaken the hash-chain or candidate-catalogue digest checks.
- Staged policies are bound to their active parent digest and replay-evidence digests. Reject stale lineage instead of rebasing silently.
- Normal activation must use `promote_from_store()` and paired task-family evidence. Unbound metric promotion is an explicit compatibility escape hatch only.
- Preserve suspension and rollback. A drifted learned policy must be able to fall back to the baseline without changing browser authority.
- Do not remove hash-chain verification from experience stores or policy-digest verification from the registry.
- Effect classification is monotonic: structural floors, sensitive target fields, and label signals collect and the highest authority wins; no early return may bypass escalation.
- Canary pair identity is `(task_family, instance_id)`; cross-family instance collisions are evidence corruption, not a pair.
- When promotion verification keys are configured, active policy records must carry a valid domain-separated attestation bound to candidate/parent digests and the qualifying evidence digests; unsigned or forged authority fails closed.
- A configured chain-head anchor must agree with the log on every read and append; a gap is resolved only by explicit `reanchor()`, never silently.
- Browser execution guarantees are `atomic` (single isolated-world validate+mutate turn, the default) and `trusted` (non-transactional CDP input with pre-press/pre-release revalidation); never describe trusted input as transactional.
- Stamped experiment plans (`jev-experiment-plan/1`) are immutable: their digest binds task, state, model choice, offered catalogue, policy behavior, and originating model. A plan whose bindings no longer hold is stale — discard it; never rebase or reinterpret it.
- Trial analysis is intention-to-treat over recorded `experiment_assigned` events: an assigned-but-unexecuted arm still counts under its run's real outcome. Aborted/interrupted/unverifiable runs are censored — missing data, never negative examples — while measured failures remain negative.

Checks: uv run ruff check ., uv run pytest, node --check jev_ultrafast/static/app.js, uv build.
