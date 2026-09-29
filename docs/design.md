# Dynamic finite choice + isolated execution

The input is a natural-language goal. Every page observation builds an indexed table of accessible elements and their current values. One DOM node receives one internal identity even when it supports multiple operations such as click and text entry.

One finite-choice request asks which operation to perform and which target would be appropriate for each available operation. The executor consumes only the target head corresponding to the selected operation. Target heads are speculative and independent. Model output cannot become selectors, coordinates, JavaScript, shell commands, or raw CDP methods.

`TYPE_TEXT` is the only generative branch. It sends the goal, selected field, bounded/redacted visible page context, and recent actions to a small text model. Output must be a JSON object containing exactly one non-empty `text` string. A generated value may survive a stale retry only while the complete helper input remains identical.

## Trust boundary

The page's main JavaScript world is untrusted. Jev creates a named CDP isolated world for the main frame and evaluates `snapshot.js` there. The isolated world owns:

- `WeakMap<Element, node_id>`
- `Map<node_id, Element>` live references
- semantic `pageKey()`
- per-target `guard()`
- the structured DOM snapshot

Main-world code can still change the DOM—that is the website—but it cannot directly replace Jev's registry, guard functions, or node-to-ID mapping. Navigation destroys the isolated execution context; the Browser recreates it and treats old decisions as stale.

Before a click, fill, or select, the Python layer performs a freshness preflight. The actual execution evaluation also receives the exact observed page key and target guard and re-compares them inside the isolated world. The target must still be connected, enabled, visible, inside the viewport, and the hit-tested element at its current center.

Mouse input still necessarily crosses a CDP boundary after geometry resolution, so arbitrary DOM mutation can never be made physically impossible. The important invariant is that page JavaScript cannot rewrite Jev's identity table or cause a model-generated selector/coordinate to become executable. For fills, a second check after the click verifies that the observed element still owns focus before any text insertion is sent.

## Dropdown semantics

Native `<select>` elements are exposed only when they are single-select. Every offered option records its original option index, normalized label, and value. Immediately before mutation the executor verifies that the option at that exact index still matches both label and value and is enabled. It then sets `selectedIndex` and dispatches `input`/`change`. Duplicate `value` attributes are therefore safe. Native multi-select is intentionally unsupported in v0.2 rather than emulated incorrectly.

## Candidate and context budgets

The browser-side raw action catalogue is capped at 1,200 controls before it is included in the semantic marker. The model receives at most 250 actions. Candidate selection is goal-aware and initially reserves capacity across click, fill, and select operations before using spare capacity, so a very large dropdown cannot evict the only search field or submit button.

Strings are bounded at the observation boundary. The model request has a hard serialized byte ceiling. `JEV_MODEL_PRIVACY=basic` additionally redacts incidental email, account/card-like numeric, secret-token patterns, and values associated with obviously sensitive labels. Applications with stronger data-loss requirements should replace or extend this layer.

## Policy boundary

Finite choices prevent arbitrary generated browser commands, but they do not make every legitimate click safe. A malicious page can still label a real button persuasively. `DefaultActionPolicy` therefore intercepts obvious consequential commit actions (purchase/payment, movement of money, destructive account changes, external communication/submission) and places the run in `approval_required` before browser input. Approval executes the exact pending decision only if its original page fingerprint and semantic guards remain valid.

The default policy is a narrow backstop, not a complete semantic security system. Production applications should inject a policy aligned with their own capabilities and consequence model.

## Completion

`DONE` is a model claim, not independent evidence. With no verifier configured, Jev stops at `claimed_done` and records `verified=false`. A caller can supply `verifier(page) -> bool | {"passed": bool, ...}`. Only a passing verifier produces `done` and `verified=true`. Failed verification is recorded and the agent returns to `ready` so the next decision can continue working toward the postcondition.

## Backends

`SystemOneBackend` is the default finite-choice adapter and remains compatible with TypeSafe/Jev. Its URL, API key, and model may be redirected to a compatible local gateway with `JEV_DECISION_BASE_URL`, `JEV_DECISION_API_KEY`, and `JEV_DECISION_MODEL`. `Agent(decision_backend=...)` accepts an in-process implementation exposing `decide(body)`.

The text helper remains separately configurable through the OpenAI-compatible `TEXT_MODEL_*` environment variables.

## Remaining browser coverage limits

The reader covers common HTML and ARIA controls but not the browser's full accessibility algorithm. Cross-origin/same-origin frames, shadow roots, canvas-only interfaces, nested scrolling, pop-up tabs, uploads, browser dialogs, drag/drop, arbitrary keyboard widgets, and native multi-select controls remain outside v0.4. Those are coverage gaps, not reasons to weaken the isolated-world or verification invariants.


## v0.4 DREAM-Jev qualification boundary

v0.4 retains the same bounded `ExplorationPolicy` self-improvement surface but hardens evidence and deployment. Live traces are now serialized across processes with an OS lock; transition catalogues carry a digest and selected rank; replay reports bind the world pool, split manifest, evidence head, and TCB versions.

`DreamImprover` evaluates every bounded candidate across available train/validation/holdout splits. `PromotionGate` requires training objective gain and forbids success, risk, coverage, or normalized-objective regressions on validation/holdout. The selected replay winner is only staged.

`PolicyRegistry` binds a staged policy to its active parent digest. Normal activation uses `promote_from_store()`, which builds matched `CanaryEvidence` from hash-verified baseline/candidate runs. The default gate requires 12 baseline and candidate runs, four task families, four paired families, no success/risk/failure regression, and bounded live latency/action/token regression. Unbound metric promotion is disabled unless explicitly requested.

Promoted policies retain their canary reference metrics. `HealthGate` can detect post-promotion success/risk drift and optionally suspend the learned policy, causing `Agent(policy_registry=...)` to fall back to baseline. Registry history supports explicit rollback.

The recursive-improvement boundary still excludes isolated-world execution, browser guards, action authorization, privacy code, verifier semantics, replay scoring, evidence validation, promotion logic, and policy-registry code.
