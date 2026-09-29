# Jev Ultrafast v0.2 — execution-integrity release

This release keeps the finite operation/target architecture and changes the trust boundaries around it.

## Security and correctness invariants

1. **The webpage does not own Jev's node registry.** `snapshot.js` is evaluated in a named CDP isolated world. The live `WeakMap<Element, id>`, `Map<id, Element>`, freshness functions, and guards are therefore separate from main-world page JavaScript.
2. **A decision is bound to what was observed.** Browser actions carry the observed page key and target guard into the execution evaluation. If either differs, execution fails as stale before input.
3. **Text is inserted only into the observed focused field.** After the mouse click, the executor checks that `document.activeElement` is the observed field (or its descendant) and that it remains editable before sending select-all and text insertion commands.
4. **Native single-select options are exact.** Every option carries its original `option_index`, label, and value. The executor verifies all three immediately before selecting by index. Duplicate option values are no longer ambiguous. `<select multiple>` remains unsupported rather than receiving incorrect single-select semantics.
5. **Mutation uncertainty is not retried blindly.** An interrupted native-select evaluation remains an ambiguous execution error because its `change` event may already have fired.
6. **Model context is bounded.** Browser observations cap labels, values, nearby scope text, page text, raw action count, and total serialized model request size.
7. **Large control sets cannot crowd out every other operation.** The browser exposes a bounded raw catalogue and the model layer applies goal-aware, operation-balanced candidate selection before the 250-action model budget.
8. **Incidental sensitive data is reduced before model calls.** Basic privacy mode redacts common email, long account/card-like numeric, API-secret, and sensitive-field-value patterns. This is intentionally described as reduction, not comprehensive DLP.
9. **Consequential commit actions are not ordinary clicks.** The default policy pauses obvious purchases, money transfers, destructive account actions, publication/message sends, and similar commits in `approval_required`. The caller must explicitly approve or reject the exact pending decision. A stricter application policy can be injected.
10. **`DONE` is not success by itself.** Without a verifier, the terminal state is `claimed_done` and `verified=false`. With a verifier, only a passing independent postcondition produces `done` and `verified=true`; a failed verifier returns the agent to `ready` with failure evidence in history.

## Backend boundary

`SystemOneBackend` preserves the current TypeSafe/Jev protocol, but endpoint, API key, and model are configurable through `JEV_DECISION_*` variables. `Agent(decision_backend=...)` accepts an injected backend implementing `decide(body) -> result`, so a local Jev/AnyJev gateway can be used without modifying the agent loop.

## Validation performed for this source package

The offline suite contains 45 tests. It covers the original finite-choice contracts plus isolated-context binding, fail-closed missing guards, focus-loss protection, duplicate dropdown values, candidate crowding, privacy redaction, policy gating, and independent completion verification.

Validation run for this package:

```text
pytest:                       45 passed
python compileall:            passed
node --check snapshot.js:     passed
node --check static/app.js:   passed
```

The environment used to assemble this package did not have the `browser-harness` distribution cached and did not permit fetching dependencies, so the live Browser Harness integration suite was not re-run here. Offline tests used a non-executing import stub only to make the unavailable package importable; browser-protocol behavior is still exercised through mocked CDP contracts. Run `uv sync --frozen`, `uv run pytest`, and `uv run python scripts/check_guards.py` on a machine with Browser Harness before release.

## Deliberately still out of scope

The following are not papered over as “supported”: cross-frame control, shadow DOM traversal, nested scroll containers, pop-up/new-tab ownership, canvas-only controls, drag/drop, file upload, browser permission dialogs, arbitrary keyboard widgets, and native multi-select semantics. These should be added after the execution-integrity invariants above remain green in live browser tests.
