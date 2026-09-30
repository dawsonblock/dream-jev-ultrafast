<img src="docs/banner.svg" alt="Jev Ultrafast · Browser Use × TypeSafe" width="100%" />

# Jev Ultrafast ⚡

> [!IMPORTANT]
> **The Browser Use Cloud waitlist is open.** Get early access to ultrafast browser agents in the cloud.
> **[Join the waitlist →](https://browser-use.com/ultrafast?utm_source=github&utm_medium=readme&utm_campaign=jev-ultrafast)**

**A browser agent with a dynamic, indexed action space.**

Give it one goal. A finite-choice decision backend picks an operation and an observed element. A small LLM writes text only when the operation is `TYPE_TEXT`. The default backend remains TypeSafe SystemOne/Jev, and any compatible endpoint or injected `DecisionBackend` can be used.

> [!NOTE]
> **v0.4 hardens DREAM-Jev into a qualification control plane.** v0.2's isolated-world browser executor remains fixed and v0.3's bounded replay policy remains the only self-improvable surface. v0.4 adds cross-process hash-chain and policy-registry serialization, replay-evidence binding, train/validation/holdout candidate selection, matched live-canary evidence bound to post-staging runs and the staged parent digest, latency/action/token regression gates, policy lineage checks, suspension, health monitoring, and rollback. A replay winner still cannot activate itself.

> [!NOTE]
> **v0.5 adds authority integrity.** A deterministic effect classifier maps every candidate to a semantic class (`NAVIGATE`, `FORM_EDIT`, `SUBMISSION`, `PURCHASE`, `UNKNOWN_COMMIT`, …) from bounded DOM context — a bare "Continue" is an `UNKNOWN_COMMIT`, and no learned adviser can downgrade a deterministic high-risk classification. Click execution re-verifies page identity, node identity, geometry, and hit-test *between* `mousePressed` and `mouseReleased`; text insertion runs atomically inside the isolated world. Evidence signatures are domain-separated with verification-key rotation, and a crash-torn final JSONL line is recovered without weakening hash/signature checks.

**Zürich → London on Google Flights in 7.1 seconds.** One natural-language goal, actual text generation, and loading waits included.

<a href="docs/demo.mp4"><img src="docs/demo.gif" alt="A real Google Flights search at 1× speed, with generated city names and dynamic operation/target decisions" width="100%" /></a>

[Watch the MP4](docs/demo.mp4) · [Measurements](docs/performance.md) · [Read the loop](jev_ultrafast/agent.py)

## The action space

Every observation produces a new element table:

```text
[1] button    Change ticket type · Round trip
[2] combobox  Where from?        · San Francisco
[3] combobox  Where to?          · empty
[4] textbox   Departure          · empty
...
```

The operations are `CLICK`, `TYPE_TEXT`, `SELECT`, `SCROLL_UP`, `SCROLL_DOWN`, `WAIT`, `DONE`, and `BLOCKED`. Only supported operations and targets are offered.

```text
                      one TypeSafe request
                     ┌───────────────────────────┐
page → element table → operation                 │
                     │ click_target              │
                     │ type_text_target          │
                     │ select_target, if present │
                     └─────────────┬─────────────┘
                         use the matching target
                                   │
                    CLICK [7] ─────┤──→ browser
                TYPE_TEXT [3] ─────┘
                          ↓
                   small LLM → text → browser
```

Target questions are speculative. If the operation is `CLICK`, only `click_target` can execute. Two decisions, **one network round trip**. Each target head contains only compatible elements. Native single-select dropdown choices carry the exact observed option index, label, and value; duplicate option values therefore remain distinguishable. Large pages are reduced through a goal-aware 250-action model budget so one large dropdown cannot hide unrelated controls.

There are no site-specific action scripts or prepared field strings in the policy. The Flights example supplies a goal and independently verifies the outcome. The screenshot renderer adds labels afterward; it does not drive the browser.

## Try it

```bash
git clone https://github.com/dawsonblock/dream-jev-ultrafast.git
cd dream-jev-ultrafast
uv sync
cp .env.example .env
# Add TYPESAFE_API_KEY and TEXT_MODEL_API_KEY.
uv run jev
```

Open **http://127.0.0.1:8766** and click **Start demo → Run automatically**. The inspector shows numbered elements, operation probabilities, target probabilities, and executed actions. **Choose next** pauses before execution.

Chrome connects through [Browser Harness](https://github.com/browser-use/browser-harness), installed by `uv sync`. Run `uv run browser-harness --doctor` if it needs connecting. Allow remote debugging in Chrome when prompted.

`TEXT_MODEL_API_KEY` is an OpenRouter key in the example configuration. The current example uses `inception/mercury-2.5` with reasoning disabled. Gemini, GLM, and DeepSeek can also use the OpenAI-compatible text helper. The finite-choice backend can be redirected with `JEV_DECISION_BASE_URL`, `JEV_DECISION_API_KEY`, and `JEV_DECISION_MODEL`, or replaced in-process through the `decision_backend=` argument.

`JEV_MODEL_PRIVACY=basic` is the default. It bounds serialized strings and redacts incidental email addresses, card/account-like long numbers, API-secret patterns, common credential query parameters (API keys, access/refresh/ID tokens, client secrets, session and CSRF values), and values from obviously sensitive fields before page observations are sent to a model. This is a useful reduction layer, not a complete DLP system.

## Use the library

```python
from jev_ultrafast import Agent

def verify(page):
    # Application-specific independent postcondition.
    return {"passed": "/travel/flights/search" in page["url"] and bool(page["text"])}

with Agent(
    "https://www.google.com/travel/flights?hl=en",
    "Find one-way flights from Zurich to London on September 20, 2026, "
    "for one adult in economy. Stop when matching flight options are visible.",
    verifier=verify,
) as agent:
    for state in agent.run():
        print(state["elapsed_ms"], state["status"], state["verified"])
```

Run with `uv run --env-file .env python your_script.py`. The same policy can run a different task:

```bash
uv run --env-file .env python examples/run.py \
  --url https://en.wikipedia.org/wiki/Main_Page \
  --goal 'Find and open the Wikipedia article about Gödel’s incompleteness theorems.'
```

`uv run --env-file .env python examples/flights.py --keep-open` performs the flight search, checks the actual route/date/results, and saves its trace. It does not select or book a flight.


## DREAM-Jev: learn how to search without self-modifying the executor

Opt into experience capture with `dream_store=`:

```python
from jev_ultrafast import Agent

with Agent(
    "https://example.com",
    "Find the requested item. Do not purchase anything.",
    verifier=verify,
    dream_store=".jev/experience.jsonl",
) as agent:
    for state in agent.run():
        ...
```

The store is privacy-bounded and SHA-256 hash-chained. Each transition records the **pre-policy observed catalogue** plus a digest of the catalogue actually offered to the model, so replay can evaluate both contraction and expansion of the candidate space rather than only pruning an already-filtered set. Optional `task_family`/`instance_id` metadata keeps semantically equivalent goals in one train/validation/holdout partition and pairs canary evidence per task instance. Offline, `jev-dream improve` turns completed histories into empirical replay worlds, evaluates bounded `ExplorationPolicy` variants across deterministic splits, binds the report to the replay-world pool and current hash-chain head, and can stage a replay-approved policy:

```bash
uv run jev-dream improve .jev/experience.jsonl \
  --report .jev/dream-report.json \
  --registry .jev/policy-registry.json \
  --cost-model --outcome-model \
  --stage
```

`--cost-model` fits a linear `CostModel` of tokens/latency against offered-candidate count on the same store; candidates keep their empirical gate results, and the learned estimate only re-orders which *passing* candidate is selected for staging (reported as `estimated_*` metrics alongside the real measurements). `--outcome-model` fits a bucketed, Beta-smoothed prior of `P(page_changed)` per `(kind, overlap, rank)` cell and annotates candidates with predicted progress. Both are hypothesis-prioritization layers: their digests are recorded in the report, but they cannot open a replay gate or count as promotion evidence.

For authenticity on top of the hash chain, the store accepts an Ed25519 signer (`JEV_EVIDENCE_SIGNING_KEY`, or any injected signer exposing `key_id`/`sign_hex` so private material can stay outside the agent process). Signatures are domain-separated over `jev-dream/evidence-event/v1:<event_hash>`. Readers pass trusted verification keys — `JEV_EVIDENCE_VERIFY_KEY`, `JEV_EVIDENCE_VERIFY_KEYS`, or `--verify-key`/`--verify-keys` — and every signed event carries its `key_id`, so key rotation keeps older signed segments verifiable under a trusted-key set. Signed events fail closed on unexpected keys or forged signatures, and `require_signatures`/`JEV_REQUIRE_SIGNED_EVIDENCE` rejects unsigned events outright. A crash-torn final line (a recognizable event prefix without its newline) is recovered on read and truncated on the next append; a parseable record with a bad hash or signature stays fail-closed:

```bash
uv run jev-dream verify .jev/experience.jsonl --verify-keys <hex-pubkey>,<hex-pubkey> --require-signatures
```

The old v0.3 command form (`jev-dream EXPERIENCE ...`) still maps to `improve`. Staging is not activation. Run matched baseline/candidate canaries, then promote from the same hash-verified trace store. Only candidate runs recorded after staging count toward promotion, and the paired baseline digest must match the staged policy's parent digest:

```bash
uv run jev-dream verify .jev/experience.jsonl
uv run jev-dream promote .jev/experience.jsonl --registry .jev/policy-registry.json
uv run jev-dream health .jev/experience.jsonl --registry .jev/policy-registry.json --recent-tasks 20
```

Default activation gates require at least 12 baseline and 12 candidate canary tasks across at least four candidate task families, at least four paired task families, no verified-success/risk regression, and no more than 25% live regression in average latency, action count, or token use. Canary evidence is paired per task instance (`instance_id`, falling back to the goal-hash task key) and reports an exact two-sided sign-test p-value; `CanaryGate(max_pair_sign_p=...)` can enforce a confidence bound on top of the count gates, and `--baseline-since-ms` bounds baseline evidence to a matched time window. Runs abandoned before a terminal decision (`aborted`) do not count as task outcomes. A failing active policy can be suspended so `Agent(policy_registry=...)` falls back to the baseline policy, and prior active policies can be restored with `jev-dream rollback`. Replay never fabricates a browser/model counterfactual: it follows the action actually recorded only if the proposed candidate-allocation policy would still have offered that action; otherwise the trajectory ends as a coverage miss. See [DREAM-Jev design](docs/dream-rsi-integration.md).

## Why it moves

- **One request per decision cycle.** Operation and target heads share the same observed state.
- **No screenshots in the default agent loop.** Jev consumes structured state. The inspector opts into screenshots; the video uses a separate continuous screencast.
- **Isolated-world identity.** Node IDs, live node references, page keys, and target guards live in a named CDP isolated world. Main-world page JavaScript cannot replace Jev's registry or guard functions.
- **One steady-state browser read per snapshot.** Read visible controls, their names, values, and text together. Isolated-world recreation adds a control-plane call only when the execution context changes.
- **Validate the selected target continuously.** A preflight freshness check happens before execution; the page/target guard, geometry, and hit-test are then re-verified inside the isolated world immediately before `mousePressed` *and again* before `mouseReleased`. A mid-press mutation (overlay, node swap, geometry drift, page change) aborts the operation — the release is still dispatched for pointer hygiene — and control returns to perception rather than retrying.
- **Insert text atomically.** After the trusted click, focus verification and `execCommand('insertText')` run inside a single isolated-world evaluation, so focus can never be redirected between check and insertion. A landed value that differs from the authorized text is a non-retryable error.
- **Wait for useful state.** After typing into a combobox, wait for visible suggestions, capped at 200 ms. Other interactions get at most two animation frames or 50 ms. These reads happen after execution is logged.
- **Keep hidden tabs rendering.** Focus emulation prevents background animation throttling without switching Chrome's visible tab.
- **Send visible text.** Offscreen article bodies and footers do not fill the model context.
- **Reuse an interrupted text request.** A generated value survives a stale-page retry only if the entire text-helper input is unchanged.

Every executed target is resolved from an observed node owned by the isolated-world registry. The executor rechecks semantic guards, current geometry, click occlusion, and—when typing—post-click focus. Model output never becomes selectors, coordinates, shell commands, or executable JavaScript. Text-helper output must parse as a small JSON object before typing. Every mutation is classified into a deterministic semantic effect (`NAVIGATE`, `SEARCH`, `FORM_EDIT`, `AUTHENTICATE`, `SUBMISSION`, `EXTERNAL_MESSAGE`, `PURCHASE`, `FINANCIAL`, `DELETE`, `PERMISSION_CHANGE`, `UNKNOWN_COMMIT`) from the bounded structural context each snapshot attaches to a candidate — form membership and method, submit semantics, scoped field inventory, modal/dialog scope, outbound destination — plus label signals that can only escalate, never downgrade. High-authority effects pause in `approval_required`; a commit-shaped control whose effect cannot be determined is itself an `UNKNOWN_COMMIT` and also requires approval. `act()` accepts no caller-asserted approval: only `approve()` issues a one-shot grant bound to the pending action and page fingerprint, and `reject()` consumes the pending decision. Applications should inject a stricter policy where their risk model demands it.

## Small enough to read

| File | Job |
| --- | --- |
| [agent.py](jev_ultrafast/agent.py) | The complete loop and text-helper handoff |
| [snapshot.js](jev_ultrafast/snapshot.js) | Atomic DOM snapshot, indexed controls, freshness guards |
| [browser.py](jev_ultrafast/browser.py) | Browser connection, current geometry, execution |
| [model.py](jev_ultrafast/model.py) | Candidate budgeting, finite-choice backend interface, validation, text generation |
| [policy.py](jev_ultrafast/policy.py) | Deterministic effect classification + approval/deny boundary for consequential mutations |
| [privacy.py](jev_ultrafast/privacy.py) | Bounded/redacted model observation layer |
| [dream.py](jev_ultrafast/dream.py) | Replay worlds, evidence-bound policy improvement, canary/health gates, lineage and rollback |
| [dreamlearn.py](jev_ultrafast/dreamlearn.py) | Learned cost/outcome priors for candidate prioritization — never qualification evidence |
| [signing.py](jev_ultrafast/signing.py) | Domain-separated Ed25519 authenticity for evidence event hashes, injectable signer boundary, key rotation |
| [trace.py](jev_ultrafast/trace.py) | Privacy-bounded, hash-chained live experience capture with candidate-catalogue evidence |
| [questions.py](jev_ultrafast/questions.py) | Model instructions |
| [demo.py](jev_ultrafast/demo.py) | Local inspector with approval controls |

## Evidence and limits

The bundled **7,073 ms** Google Flights video and paired performance artifacts were recorded against the v0.1 fast path. They remain useful historical evidence for the architecture, but they are **not claimed as fresh v0.4 measurements** because v0.2 added isolated-world/guard checks, v0.3 added optional exploration-policy/trace paths, and v0.4 adds qualification/evidence bookkeeping. Re-benchmark before publishing a new latency number for this release.

In six alternating runs with identical models and settings, both versions passed **3/3**. Median task time went from **9.450 s → 7.092 s**, a **25% reduction**; median browser protocol calls went from **1,092 → 101**. This is three repeats of one task on one browser profile, not a general reliability benchmark.

The same policy opened the requested Wikipedia article in **2.798 s** and passed a local hotel search/filter task in **1.896 s**. Runs, failures, source hashes, and measurement boundaries are in [performance.md](docs/performance.md).

A `DONE` choice is now represented as `claimed_done` unless a caller-supplied verifier passes; successful verification produces `done` with `verified=true`. The DOM reader handles common HTML and ARIA controls, not the full accessible-name specification. Shadow roots, frames, canvas, uploads, pop-up tabs, nested scrolling, multi-select widgets, and arbitrary keyboard widgets remain outside this release. Browser Harness still controls the profile/session boundary; v0.4 does not claim that the Chrome profile itself is isolated from the user's normal browser profile. DREAM-Jev is an empirical replay system, not a learned latent world model: the auxiliary `CostModel`/`OutcomeModel` only prioritize among empirically evaluated candidates and can never count as qualification evidence. Its usefulness depends on historical branch coverage and website freshness. The default canary thresholds are a deployment gate, not statistical proof of general reliability; production users should increase task count and diversity for their domain.

## Development

```bash
uv run ruff check .
uv run pytest
node --check jev_ultrafast/static/app.js
node --check jev_ultrafast/snapshot.js
uv build
```

Unit tests are offline; CI additionally runs `scripts/check_guards.py` in a `live-guards` job against headless Chrome over CDP, because a past release shipped a mutation-path defect that passed mocked tests and failed only in a real browser. Run the same suite locally against a dedicated automation Chrome — the harness does not see Chrome running under a non-default profile or without remote debugging:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-port=9222 --user-data-dir=/tmp/jev-guard-chrome \
  --no-first-run --no-default-browser-check about:blank &
BU_CDP_URL=http://127.0.0.1:9222 uv run python scripts/check_guards.py
```

Live examples and recording scripts make paid API calls. `scripts/record_flights.py <new-folder>` captures original browser timestamps; `scripts/render_demo.py <recording-folder>` renders that verified run at 1× and crops out the Google account strip. Credentials and raw traces stay ignored.

---

[Browser Use](https://github.com/browser-use/browser-use) · [Browser Harness](https://github.com/browser-use/browser-harness) · [TypeSafe speculative fan-out](https://docs.typesafe.ai/patterns/fan-out)
