<img src="docs/banner.svg" alt="Jev Ultrafast · Browser Use × TypeSafe" width="100%" />

<div align="center">

# Jev Ultrafast

**A browser agent that chooses instead of generating.**

[![ci](https://github.com/dawsonblock/dream-jev-ultrafast/actions/workflows/ci.yml/badge.svg)](https://github.com/dawsonblock/dream-jev-ultrafast/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/dawsonblock/dream-jev-ultrafast)](https://github.com/dawsonblock/dream-jev-ultrafast/releases)
[![python](https://img.shields.io/badge/python-%E2%89%A53.12-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[Quick start](#quick-start) · [How it works](#how-it-works) · [DREAM-Jev](#dream-jev-self-improvement-without-self-modification) · [Docs](#documentation) · [Changelog](CHANGELOG.md)

</div>

Give it one goal. Every observation produces an indexed table of the page's actual controls; a finite-choice decision backend picks an **operation** and an **observed element** in a single request. A small LLM writes text only when the operation is `TYPE_TEXT`. Model output can never become a selector, a coordinate, or a script — and every mutation passes a deterministic authority plane before it touches the page.

> [!IMPORTANT]
> **The Browser Use Cloud waitlist is open.** Get early access to ultrafast browser agents in the cloud.
> **[Join the waitlist →](https://browser-use.com/ultrafast?utm_source=github&utm_medium=readme&utm_campaign=jev-ultrafast)**

<a href="docs/demo.mp4"><img src="docs/demo.gif" alt="A real Google Flights search at 1× speed, with generated city names and dynamic operation/target decisions" width="100%" /></a>

**Zürich → London on Google Flights in 7.1 seconds.** One natural-language goal, actual text generation, and loading waits included — recorded on the v0.1 fast path and labeled as [historical evidence](#evidence-and-limits), not a fresh benchmark.

[Watch the MP4](docs/demo.mp4) · [Measurements](docs/performance.md) · [Read the loop](jev_ultrafast/agent.py)

> [!NOTE]
> **v0.9.x makes the learning layer statistically honest.** Randomized in-browser trials are analyzed intention-to-treat with structured censoring; effects are called `beneficial`/`harmful` only when an *anytime-valid confidence sequence* clears a practical threshold *and* conservative censoring bounds agree — "we observed it" and "we established it" are different claims. Experiment plans are digested, signed, and stale-proof. See the [changelog](CHANGELOG.md) and [VALIDATION.md](VALIDATION.md) for the full history.

## How it works

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

### Execution guarantees

Every executed target is resolved from an observed node owned by a named CDP **isolated world** — main-world page JavaScript cannot replace Jev's registry or guard functions. Mutations run under one of two explicit guarantees:

| Guarantee | Mechanism | When |
| --- | --- | --- |
| `DOM_ATOMIC` | Validation and the mutation (`e.click()`, or focus + insert + landed-value check) in a *single* isolated-world turn — no page JS can interleave | Default for click/fill/select |
| `TRUSTED_INPUT_NONTRANSACTIONAL` | Real CDP mouse input for `isTrusted`-gated elements, with page identity, node guard, geometry, and hit-test re-verified before `mousePressed` *and again* before `mouseReleased` | Automatic escalation only after the atomic path provably did not mutate |

Before any mutation, a deterministic **effect classifier** maps the candidate to a semantic class (`NAVIGATE`, `FORM_EDIT`, `SUBMISSION`, `PURCHASE`, `DISCLOSURE`, `AUTHENTICATE`, `UNKNOWN_COMMIT`, …) from bounded structural context — form membership, submit semantics, scoped field inventory, the target's own field class, modal scope, outbound destination, label signals. Classification is monotonic: every signal can only escalate, so a credit-card `fill` or a "Share publicly" switch can never short-circuit to `FORM_EDIT`. High-authority effects pause in `approval_required`; a commit-shaped control whose effect cannot be determined is itself an `UNKNOWN_COMMIT`. `act()` accepts no caller-asserted approval — only `approve()` issues a one-shot grant bound to the pending action, its *generated payload digest*, and the page fingerprint.

That classification context is also *bound into the execution transaction*: the pre-mutation guard embeds the same derived signals the classifier consumed, so a page that flips `type`, `autocomplete`, form association/method/action, sibling field inventory, `href` scheme, `target`, or `download` between decision and dispatch is a stale target — not a mutation executing under a stale authorization.

## Quick start

```bash
git clone https://github.com/dawsonblock/dream-jev-ultrafast.git
cd dream-jev-ultrafast
uv sync
cp .env.example .env       # add TYPESAFE_API_KEY and TEXT_MODEL_API_KEY
uv run jev
```

Open **http://127.0.0.1:8766** and click **Start demo → Run automatically**. The inspector shows numbered elements, operation probabilities, target probabilities, and executed actions. **Choose next** pauses before execution.

Chrome connects through [Browser Harness](https://github.com/browser-use/browser-harness), installed by `uv sync`. Run `uv run browser-harness --doctor` if it needs connecting. Allow remote debugging in Chrome when prompted.

`TEXT_MODEL_API_KEY` is an OpenRouter key in the example configuration (`inception/mercury-2.5`, reasoning disabled). Gemini, GLM, and DeepSeek work through the same OpenAI-compatible helper. The finite-choice backend can be redirected with `JEV_DECISION_BASE_URL` / `JEV_DECISION_API_KEY` / `JEV_DECISION_MODEL`, or replaced in-process through `decision_backend=`.

<details>
<summary><strong>Fully local runs — no API keys</strong></summary>

Install [Ollama](https://ollama.com), then:

```bash
ollama pull qwen3:8b && ollama serve
OLLAMA_MODEL=qwen3:8b uv run python scripts/local_backend.py
```

Point the agent at the loopback shim:

```bash
JEV_DECISION_BASE_URL=http://127.0.0.1:9000/v1/systemone
JEV_MODEL_TIMEOUT=240                    # local inference is slower than the 25s default
TEXT_MODEL_BASE_URL=http://127.0.0.1:11434/v1
TEXT_MODEL=qwen3:8b
TEXT_MODEL_REASONING=none                # Ollama rejects reasoning fields
```

The shim binds `127.0.0.1` and applies the demo server's request hardening (Host/Origin/path allowlists, bounded bodies); set `JEV_LOCAL_TOKEN` on the shim and `JEV_DECISION_API_KEY` on the agent for a shared-secret bearer check. An 8B model won't plan like a frontier model — but every trace it produces is real DREAM experience, and the replay/experiment layer exists precisely to improve a weak policy from its own outcomes.

</details>

`JEV_MODEL_PRIVACY=basic` (the default) bounds serialized strings and redacts incidental emails, card-like numbers, API-secret patterns, JWTs, credential `key=value` parameters in URL queries, paths, and fragments, URL userinfo, token-shaped URL path segments, and sensitive field values before observations reach a model. A useful reduction layer — not a complete DLP system.

`JEV_ACTION_KEY` optionally supplies a persistent 32-byte secret as 64 hexadecimal characters for HMAC-based exact-action fingerprints in DREAM evidence. Keep it private and consistent across runs; without it, exact-action fingerprints are omitted rather than stored as guessable hashes.

`JEV_MODEL_ROUTING` binds the run's model traffic to a security level before any inference starts: `local-only` refuses every non-loopback model endpoint outright (the goal and page never leave the machine), `sanitized` (the default) allows remote endpoints but sends the task goal through the same redaction pass as page content, and `public` declares the task non-sensitive and sends the goal verbatim. The levels are security invariants: `sanitized` cannot be combined with `JEV_MODEL_PRIVACY=off` — the pair fails closed rather than silently downgrading to raw transmission (use `public` to declare that intent). Regex redaction is a floor, not a guarantee — declare `local-only` for runs whose objective contains secrets. Every non-loopback endpoint also requires `https` — plaintext remote HTTP would carry the API key and model context in cleartext (`JEV_ALLOW_INSECURE_TRANSPORT=1` exists only for local development).

`JEV_INPUT_GUARANTEE=trusted` (or `Agent(input_guarantee="trusted")`) routes clicks/fills through the real-CDP-input path for sites that ignore synthetic `isTrusted=false` events. It is a declaration, not a fallback: a silent no-op synthetic click cannot be distinguished from one that landed, so the system escalates automatically only when the atomic path provably did not mutate.

`JEV_SECURITY_PROFILE` bounds how the security knobs compose. `standard` (the default) is the status quo. `strict` is the hardened posture: `JEV_MODEL_ROUTING=public` and `JEV_MODEL_PRIVACY=off` are refused under any routing level, `JEV_ALLOW_INSECURE_TRANSPORT` fails closed even where it would apply, `trusted` input guarantee is refused (atomic execution only), the `JEV_ALLOW_UNBOUND_METRICS` promotion escape hatch stays closed, and signed evidence is required as if `JEV_REQUIRE_SIGNED_EVIDENCE=1`. `qualified` applies every strict refusal *and* requires release provenance *and* qualification evidence: the agent re-verifies the installed tree at construction — `MANIFEST.sig` under `JEV_MANIFEST_VERIFY_KEYS`, the full release-set hashes, the trusted-base boundary, and `RELEASE_QUALIFICATION.json`, a `jev-qualify/*` report signed under `JEV_QUALIFY_VERIFY_KEYS` declaring `release_qualified: true` and bound to the current manifest digest. `JEV_EXPECT_TCB_VERSION` and `JEV_EXPECT_MANIFEST_DIGEST` optionally pin the approved release so a validly-signed older tree cannot roll back — unsigned, unpinned, unqualified, or drifted trees fail closed. A knob weakening below the declared profile fails closed naming the profile — it is never silently ignored; use `standard` explicitly to keep a weakening flag.

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

## DREAM-Jev: self-improvement without self-modification

The agent's executor and authority plane are immutable. The only thing that learns is a bounded, validated `ExplorationPolicy` — and it earns activation through evidence, not preference.

```text
real runs ──► hash-chained experience store ──► replay worlds ──► improve
                                                            │
                  ┌─────────────────────────────────────────┤
                  ▼                                         ▼
        observational prior                        randomized trials
        (ChoiceModel — advisory)            (CounterfactualTrials → TrialChoiceModel)
                  │                                         │
                  └───────────────► stamped hypotheses ◄────┘
                                    (jev-experiment-plan/1, signed)
                                            │
                                    live experiment arms ──► ITT estimates
                                            │
                                    replay approval → bound live canaries
                                            │
                                    signed promotion → canary gates → active
```

Opt into experience capture with `dream_store=`:

```python
with Agent(
    "https://example.com",
    "Find the requested item. Do not purchase anything.",
    verifier=verify,
    dream_store=".jev/experience.jsonl",
) as agent:
    for state in agent.run():
        ...
```

The store is privacy-bounded and SHA-256 hash-chained. Each transition records the **pre-policy observed catalogue** plus a digest of the catalogue actually offered to the model, so replay can evaluate both contraction and expansion of the candidate space. Offline, `jev-dream improve` turns histories into replay worlds, evaluates bounded `ExplorationPolicy` variants across deterministic train/validation/holdout splits, and can stage a replay-approved policy:

```bash
uv run jev-dream improve .jev/experience.jsonl \
  --report .jev/dream-report.json \
  --registry .jev/policy-registry.json \
  --cost-model --outcome-model --choice-model \
  --stage
```

- **`--cost-model`** fits a linear `CostModel` of tokens/latency against offered-candidate count — it only re-orders which *passing* candidate gets staged.
- **`--outcome-model`** fits a bucketed, Beta-smoothed `P(page_changed)` prior per `(kind, overlap, offered-rank)` cell.
- **`--choice-model`** fits an uncertainty-aware counterfactual prior labeled by each run's *verified terminal outcome* — page churn inside failed runs teaches it nothing — with posterior stddev and explicit abstention on sparse cells.
- **`CounterfactualTrials` + `TrialChoiceModel`** (always fitted) consume only randomized `experiment_assigned` evidence: intention-to-treat arms, structured censoring with an imbalance gate plus Manski worst/best-case bounds, an anytime-valid confidence sequence over a practical threshold (Bonferroni-split across the tracked hypothesis family), and hierarchical `family+site → site → family → pooled` resolution against a bounded treatment signature (`kind` and effect class are floors — click evidence never answers a fill proposal, and `PURCHASE` evidence never answers a `SEARCH`/`DELETE` divergence).
- **`ExperimentScheduler`** ranks which unresolved hypothesis is worth the next real run; **`CausalChoicePolicy`** exposes `shadow` → `canary` → `active` modes under operator control, tagging every entry `observational` / `randomized` / `pooled_randomized`.

Learned models annotate and prioritize only. A counterfactual proposal is a hypothesis — never an outcome, never evidence, never a gate input. Anything a model prefers must still qualify through real executions and bound live canaries.

### Evidence integrity

- **Signed store**: optional Ed25519 event signatures (`JEV_EVIDENCE_SIGNING_KEY`, injectable signer so key material stays outside the agent), domain-separated over `jev-dream/evidence-event/v1:<event_hash>`, with trusted-key sets, rotation, and fail-closed `require_signatures` mode. A signed **chain-head anchor** (`--anchor PATH`) closes the forged-tail gap — point it at storage the log attacker cannot also roll back.
- **Signed plans**: stamped `jev-experiment-plan/1` objects bind task, family, state, model choice, proposal, offered catalogue, policy behavior, and originating model. Digest integrity is distinguished from Ed25519 provenance (`jev-dream/experiment-plan/v1:`); under `JEV_EXPERIMENT_VERIFY_KEYS`, unsigned or forged plans fail closed. A stale plan is discarded — never rebased or reinterpreted.
- **Signed promotion**: activation attaches a domain-separated attestation binding candidate/parent digests, replay and canary evidence, the chain head, and the registry revision. Every registry write is a hash-chained, signed state transition — post-write tampering fails the digest on the next read. Suspension and rollback re-verify the original attestation.

```bash
uv run jev-dream verify .jev/experience.jsonl --verify-keys <hex-pubkey>,<hex-pubkey> --require-signatures
uv run jev-dream promote .jev/experience.jsonl --registry .jev/policy-registry.json
uv run jev-dream health  .jev/experience.jsonl --registry .jev/policy-registry.json --recent-tasks 20
# Read-only causal inspection: cell inventory, or one divergence resolved
# through the treatment-signature hierarchy (null when nothing can answer)
uv run jev-dream trials  .jev/experience.jsonl --cells
uv run jev-dream trials  .jev/experience.jsonl --model-kind click --proposal-effect navigate
```

Staging is not activation: promotion requires matched baseline/candidate canaries (≥12 runs each across ≥4 task families by default), no verified-success/risk regression, ≤25% latency/action/token regression, and an exact two-sided sign-test over `(task_family, instance_id)` pairs at p ≤ 0.05 by default — which in practice demands more than the minimum canary, since four pairs cannot reach significance. See [DREAM-Jev design](docs/dream-rsi-integration.md).

## Why it moves

- **One request per decision cycle** — operation and target heads share the same observed state.
- **No screenshots in the default loop** — structured state only; the inspector opts in, and demo video uses a separate screencast.
- **One steady-state browser read per snapshot** — visible controls, names, values, and text together.
- **Wait for useful state** — ≤200 ms for combobox suggestions, ≤2 frames elsewhere, always after execution is logged.
- **Send visible text** — offscreen bodies and footers stay out of model context.
- **Reuse an interrupted text request** — a generated value survives a stale-page retry only if the entire helper input is unchanged.

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
| [dreamlearn.py](jev_ultrafast/dreamlearn.py) | Learned cost/outcome/choice priors and the causal trial layer — never qualification evidence |
| [signing.py](jev_ultrafast/signing.py) | Domain-separated Ed25519 authenticity for evidence event hashes, injectable signer boundary, key rotation |
| [trace.py](jev_ultrafast/trace.py) | Privacy-bounded, hash-chained live experience capture with candidate-catalogue evidence |
| [questions.py](jev_ultrafast/questions.py) | Model instructions |
| [demo.py](jev_ultrafast/demo.py) | Local inspector with approval controls |

## Evidence and limits

The bundled **7,073 ms** Google Flights video and paired performance artifacts were recorded against the v0.1 fast path — useful historical evidence for the architecture, **not** a fresh v0.9 measurement (isolated-world guards, the exploration/trace path, and the qualification layer all landed afterward). Re-benchmark before citing a latency number for this release.

In six alternating runs with identical models and settings, both versions passed **3/3**; median task time went **9.450 s → 7.092 s** (−25%) and median browser protocol calls **1,092 → 101**. Three repeats of one task on one profile — not a general reliability benchmark. The same policy opened the requested Wikipedia article in **2.798 s** and passed a local hotel search/filter in **1.896 s**. Runs, failures, source hashes, and measurement boundaries are in [performance.md](docs/performance.md).

A `DONE` choice is `claimed_done` unless a caller-supplied verifier passes. The DOM reader handles common HTML/ARIA controls plus same-origin frames, open shadow roots, nested scroll containers, keyboard scroll, allowlisted file upload (`Agent(uploads=[...])` / `JEV_UPLOADS` — declared files are snapshotted to private staging and the approval binds the staged content digest, so a post-approval file swap cannot change the bytes that were reviewed), and popup adoption — cross-origin frames, closed shadow roots, canvas, browser dialogs, drag/drop, arbitrary keyboard widgets, and multi-select remain out of scope. DREAM-Jev is an empirical replay system, not a latent world model: learned priors prioritize among empirically evaluated candidates and can never count as qualification evidence. The default canary thresholds are a deployment gate, not statistical proof — raise task count and diversity for your domain.

## Development

```bash
uv run ruff check .
uv run pytest                                   # 446 tests, offline
node --check jev_ultrafast/static/app.js
node --check jev_ultrafast/snapshot.js
uv build
```

CI additionally runs `scripts/check_guards.py` in a `live-guards` job against headless Chrome over CDP — a past release shipped a mutation-path defect that passed mocked tests and failed only in a real browser. Run the same suite locally against a dedicated automation Chrome:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-port=9222 --user-data-dir=/tmp/jev-guard-chrome \
  --no-first-run --no-default-browser-check about:blank &
BU_CDP_URL=http://127.0.0.1:9222 uv run python scripts/check_guards.py
```

Live examples and recording scripts make paid API calls. `scripts/record_flights.py <new-folder>` captures original browser timestamps; `scripts/render_demo.py <recording-folder>` renders that verified run at 1×. Credentials and raw traces stay ignored.

## Documentation

| Doc | Contents |
| --- | --- |
| [CHANGELOG.md](CHANGELOG.md) | Full release history, v0.4 → v0.9.2 |
| [VALIDATION.md](VALIDATION.md) | Reproduced verification results per release |
| [docs/performance.md](docs/performance.md) | Measurements, boundaries, raw evidence |
| [docs/design.md](docs/design.md) | Architecture decisions |
| [docs/dream-rsi-integration.md](docs/dream-rsi-integration.md) | DREAM-Jev replay/canary/promotion design |

---

[Browser Use](https://github.com/browser-use/browser-use) · [Browser Harness](https://github.com/browser-use/browser-harness) · [TypeSafe speculative fan-out](https://docs.typesafe.ai/patterns/fan-out)
