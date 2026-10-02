"""Finite-choice agent loop with policy approval and independent completion verification."""

from __future__ import annotations

import base64
import copy
import hashlib
import inspect
import json
import os
import random
import time
from pathlib import Path
from urllib.parse import urlsplit

from .browser import Browser, BrowserError, IndeterminateMutation, StalePage
from .dream import (
    ExperienceStore,
    ExplorationPolicy,
    PolicyRegistry,
    candidate_catalog_digest,
    experiment_plan_authority,
    experiment_plan_digest,
    task_key,
)
from .model import (
    ModelConnectionError,
    ModelTimeoutError,
    action_space,
    candidate_actions,
    choose,
    field_context,
    field_text,
)
from .policy import DefaultActionPolicy, assess_payload, classify_effect
from .privacy import action_goal_overlap, redact_text, sanitize_url, tokenize
from .questions import MAX_STEPS
from .signing import verify_keys_from_env
from .trace import DreamTraceRecorder, compact_candidate

TERMINAL_STATUSES = {"claimed_done", "done", "blocked"}

# Minimum randomized-allocation share for each experiment arm. A candidate-
# vs-control contrast needs nonzero support on *both* arms to identify an
# effect at all: rate=1.0 (or 0.0) makes every assignment the same arm, which
# produces trial-looking evidence that can never estimate a treatment effect.
# The small floor additionally keeps the thinner arm's IPW weights bounded
# (propensity >= 0.05 ⇒ weight <= 20) so the confidence sequence stays sane.
MIN_EXPERIMENT_ARM_RATE = 0.05


def _abort_reason(exc: BaseException) -> str:
    """Structured termination reason for an exception that ended a run.

    Deliberately closed and coarse: only the failure shapes the runtime can
    actually distinguish get their own reason. Everything else is
    ``agent_exception`` — an unattributable interruption, censored as missing
    data rather than scored as a task failure.
    """
    if isinstance(exc, ModelTimeoutError):
        return "timeout"
    if isinstance(exc, ModelConnectionError):
        return "network_failure"
    if isinstance(exc, BrowserError):
        return "browser_crash"
    if isinstance(exc, IndeterminateMutation):
        # The browser may have executed the mutation before the run died —
        # neither a clean "not executed" nor a measured failure. Its own
        # reason keeps that distinguishable in censoring analysis.
        return "indeterminate_execution"
    return "agent_exception"


def _page_site(page: dict) -> str | None:
    """The run's site stratum — the sanitized host of the observed page URL."""
    try:
        return urlsplit(sanitize_url(page.get("url") or "")).hostname or None
    except ValueError:
        return None


def _model_call(fn, *args, **kwargs):
    """Call a learned-model method with only the kwargs it declares.

    ``experiment.model`` may be any fitted prior — ChoiceModel, the causal
    TrialChoiceModel, or an operator-provided object — so context kwargs the
    callee does not declare are dropped rather than TypeErroring.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    return fn(*args, **{k: v for k, v in kwargs.items() if k in params})


class Agent:
    def __init__(
        self,
        url,
        goals,
        *,
        record_dir=None,
        screenshots=False,
        decision_backend=None,
        verifier=None,
        policy=None,
        exploration_policy=None,
        policy_registry=None,
        dream_store=None,
        run_id=None,
        task_family=None,
        instance_id=None,
        experiment=None,
        experiment_verify_keys=None,
        input_guarantee=None,
    ):
        task = goals.strip() if isinstance(goals, str) else "\n".join(goals).strip()
        if not task:
            raise ValueError("Supply a task")
        plan = [task]
        self.pending_text = None
        # Execution guarantee for browser input. "atomic" (default) validates
        # and mutates in one isolated-world turn; "trusted" dispatches real CDP
        # input under pre-press/pre-release revalidation. Automatic escalation
        # happens ONLY after the atomic path provably did not mutate — a silent
        # no-op synthetic click on an isTrusted-gated site is indistinguishable
        # from one that landed, so it is never retried; the operator selects
        # trusted input up front for those sites (JEV_INPUT_GUARANTEE).
        guarantee = input_guarantee if input_guarantee is not None else os.environ.get(
            "JEV_INPUT_GUARANTEE", "atomic"
        )
        if guarantee not in {"atomic", "trusted"}:
            raise ValueError(
                "input_guarantee must be 'atomic' or 'trusted'"
            )
        self.input_guarantee = guarantee
        self.decision_backend = decision_backend
        self.verifier = verifier
        self.policy = DefaultActionPolicy() if policy is None else policy
        if exploration_policy is not None and policy_registry is not None:
            raise ValueError("Supply exploration_policy or policy_registry, not both")
        if policy_registry is not None:
            registry = (
                policy_registry if isinstance(policy_registry, PolicyRegistry) else PolicyRegistry(policy_registry)
            )
            exploration_policy = registry.active_policy()
        self.exploration_policy = exploration_policy or ExplorationPolicy()
        # Optional on-policy experiment config: {"model": ChoiceModel, "rate": ε,
        # "rng": random.Random(...)} — or {"proposals": [...], "rate": ε} to run
        # the stamped DreamReport.experiment_proposals plans themselves (matched
        # to a live state by fingerprint). When set, eligible steps execute the
        # prior's divergent proposal — or the model's own choice as the control
        # arm — with a recorded assignment probability. At most one trial per
        # run so outcomes stay attributable. Validated here so a malformed
        # config fails before the browser starts, not mid-run.
        if experiment is not None:
            if not isinstance(experiment, dict):
                raise ValueError("experiment must be a config dict")
            try:
                rate = float(experiment.get("rate", 0.0) or 0.0)
            except (TypeError, ValueError):
                raise ValueError("experiment rate must be a number") from None
            if not MIN_EXPERIMENT_ARM_RATE <= rate <= 1.0 - MIN_EXPERIMENT_ARM_RATE:
                raise ValueError(
                    "experiment rate must keep both arms supported: "
                    f"{MIN_EXPERIMENT_ARM_RATE} <= rate <= "
                    f"{1.0 - MIN_EXPERIMENT_ARM_RATE}. A degenerate allocation "
                    "(rate=1.0 assigns every run to the candidate) produces "
                    "trial evidence that cannot estimate a causal effect."
                )
            proposals = experiment.get("proposals")
            if proposals is not None and not isinstance(proposals, (list, tuple)):
                raise ValueError("experiment proposals must be a list of stamped plans")
            if experiment.get("model") is None and not proposals:
                raise ValueError("experiment requires a fitted model or stamped proposals")
            causal_policy = experiment.get("causal_policy")
            if causal_policy is not None:
                if getattr(causal_policy, "mode", None) not in {"shadow", "canary", "active"}:
                    raise ValueError(
                        "experiment causal_policy must expose a mode of shadow/canary/active"
                    )
                if not callable(getattr(causal_policy, "rank", None)):
                    raise ValueError("experiment causal_policy must expose rank()")
        self.experiment = experiment
        # Trusted Ed25519 keys for experiment-plan provenance. With keys
        # configured, a stamped plan must carry a valid domain-separated
        # signature or it is stale; without them the agent stays in explicit
        # unsigned-compatibility mode (digest integrity only).
        if isinstance(experiment_verify_keys, str):
            experiment_verify_keys = experiment_verify_keys.replace(",", " ").split()
        self.experiment_verify_keys = {
            str(key) for key in (experiment_verify_keys or ()) if key
        }
        # Task identity for experiment-plan binding and assignment records —
        # normalized exactly as the trace recorder stores them.
        self.task_family = str(task_family).strip()[:256] if task_family else None
        self.instance_id = str(instance_id).strip()[:256] if instance_id else None
        if isinstance(dream_store, (str, Path)):
            dream_store = ExperienceStore(dream_store)
        self.dream_recorder = (
            DreamTraceRecorder(
                dream_store,
                goal=task,
                run_id=run_id,
                task_family=task_family,
                instance_id=instance_id,
            )
            if dream_store is not None
            else None
        )
        self.browser = Browser(url)
        self.record_dir = Path(record_dir) if record_dir else None
        self.screenshots = screenshots or bool(record_dir)
        try:
            page = self.browser.observe(screenshot=self.screenshots)
        except Exception:
            self.browser.close()
            raise
        self.state = dict(
            browser=self.browser,
            goal="\n".join(plan),
            page=page,
            decision=None,
            history=[],
            status="ready",
            plan=plan,
            plan_index=0,
            decisions=[],
            text_calls=[],
            elapsed_ms=0,
            started_at=None,
            record=bool(self.record_dir),
            pending_approval=None,
            granted_approval=None,
            experiment_spent=False,
            experiment_assignment=None,
            verification=None,
            verification_failures=[],
            verified=False,
            exploration_policy={
                "name": self.exploration_policy.name,
                "version": self.exploration_policy.version,
                "digest": self.exploration_policy.digest,
            },
            dream_run_id=self.dream_recorder.run_id if self.dream_recorder else None,
        )
        if self.record_dir:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            (self.record_dir / "000000.jpg").write_bytes(base64.b64decode(page["screenshot"]))
        if self.dream_recorder:
            self.dream_recorder.start(page, self.exploration_policy)

    def _perception(self):
        """Candidate catalogue + action space for the current page, computed once.

        ``predict``/``choose``, the experiment scheduler, the recorder's offered
        catalogue, and the inspector snapshot all consume the same reduction of
        the observed action list — re-running it per consumer re-sanitizes
        ~250 actions several times per step. The cache key is the actions-list
        object itself: a fresh ``observe()`` installs a new list, and keeping
        the old list referenced here means its ``id`` can never alias.
        """
        page = self.state["page"]
        policy = getattr(self, "exploration_policy", None)
        cache = getattr(self, "_perception_cache", None)
        if (
            cache is not None
            and cache[0] is page["actions"]
            and cache[1] == self.state["goal"]
            and cache[2] is policy
        ):
            return cache[3:]
        candidates, omitted = candidate_actions(
            page["actions"], self.state["goal"], exploration_policy=policy
        )
        elements, targets, controls = action_space(candidates)
        result = (candidates, omitted, elements, targets, controls)
        self._perception_cache = (page["actions"], self.state["goal"], policy, *result)
        return result

    def snapshot(self):
        candidates, omitted, elements, _targets, _controls = self._perception()
        ordered_nodes = []
        for action in candidates:
            node = action.get("node")
            if action.get("kind") in {"click", "fill", "select"} and node not in ordered_nodes:
                ordered_nodes.append(node)
        # Annotate copies — the cached elements are shared with the model
        # request body and must never grow inspector-only keys.
        elements = [
            {**element, "node": node}  # local inspector metadata; never sent to a decision backend
            for element, node in zip(elements, ordered_nodes, strict=True)
        ]
        return {
            # Deep copy: a snapshot consumer must not hold references into
            # live agent state (page, history, decisions, pending approval).
            **copy.deepcopy({k: v for k, v in self.state.items() if k != "browser"}),
            "elements": elements,
            "candidate_metadata": {
                "offered_actions": len(candidates),
                "omitted_for_model": omitted + int(self.state["page"].get("omitted_actions", 0)),
                "raw_action_count": self.state["page"].get("raw_action_count", len(self.state["page"]["actions"])),
            },
        }

    def _elapsed(self):
        if self.state.get("started_at") is None:
            return 0
        return round((time.perf_counter() - self.state["started_at"]) * 1000)

    def _verify(self, page):
        # Dispatch on the declared signature so an internal TypeError raised by a
        # two-argument verifier is never retried as a one-argument call.
        try:
            params = list(inspect.signature(self.verifier).parameters.values())
            inspectable = True
        except (TypeError, ValueError):
            params = []
            inspectable = False
        positional_slots = sum(
            p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params
        )
        accepts_two = (
            not inspectable
            or positional_slots >= 2
            or any(p.kind is p.VAR_POSITIONAL for p in params)
        )
        result = self.verifier(page, self.snapshot()) if accepts_two else self.verifier(page)
        if isinstance(result, bool):
            result = {"passed": result}
        if not isinstance(result, dict) or type(result.get("passed")) is not bool:
            raise ValueError("Verifier must return bool or a dict containing passed: bool")
        return result

    def _experiment_verify_keys(self) -> set[str]:
        """Trusted experiment-plan verification keys.

        Configured on the agent, in the experiment config, or through
        ``JEV_EXPERIMENT_VERIFY_KEYS``/``JEV_EXPERIMENT_VERIFY_KEY``. With no
        keys the agent stays in explicit unsigned-compatibility mode: digest
        integrity is still enforced, and a *present* signature must still
        verify against its own key, but provenance is not established.
        """
        keys = set(getattr(self, "experiment_verify_keys", None) or ())
        experiment = getattr(self, "experiment", None) or {}
        configured = experiment.get("verify_keys")
        if isinstance(configured, str):
            keys.update(part for part in configured.replace(",", " ").split() if part)
        elif configured:
            keys.update(str(key) for key in configured)
        keys |= verify_keys_from_env(
            "JEV_EXPERIMENT_VERIFY_KEYS", "JEV_EXPERIMENT_VERIFY_KEY"
        )
        return keys

    def _annotated_offered(self):
        """Offered catalogue with goal_overlap on the redacted-goal basis.

        The priors were trained on the trace's sanitized features; raw page
        actions carry no goal_overlap, so annotate it — otherwise every live
        proposal is scored on a zeroed overlap the model never saw in
        training. The token basis is the redacted goal, matching what the
        recorder stored.
        """
        goal_tokens = set(tokenize(redact_text(self.state["goal"], 4096)))
        # Each offered candidate carries its deterministic effect class — the
        # causal layer keys treatment signatures on it, so a click that
        # purchases can never be answered by evidence about a click that
        # navigates. Classification happens here, on the raw action, while the
        # full structural context (role/ctx) is still present.
        offered = [
            {
                **c,
                "goal_overlap": action_goal_overlap(c, goal_tokens),
                "effect": classify_effect(c).value,
            }
            for c in self._perception()[0]
        ]
        return offered, goal_tokens

    def _live_proposal(self, *, trial_model, causal_policy, offered_now, selected, action, page):
        """The live (unstamped) experimental proposal for this step.

        A canary-mode ``CausalChoicePolicy`` speaks first — its combined
        ranking may propose only under randomized assignment — then the
        standalone trial prior. A ``shadow`` or ``active`` policy contributes
        nothing here: shadow observes, and active overrides the choice
        directly (a different code path) rather than entering the randomized
        assignment.
        """
        model_choice = {
            "id": selected,
            "kind": action.get("kind"),
            "role": action.get("role"),
            "effect": classify_effect(action).value,
            "goal_overlap": next(
                (c.get("goal_overlap") for c in offered_now if c.get("id") == selected),
                None,
            ),
        }
        phase = len(
            [
                h
                for h in self.state["history"]
                if h.get("kind") not in {"verification", "approval"}
            ]
        )
        if causal_policy is not None and getattr(causal_policy, "mode", "shadow") == "canary":
            proposal = _model_call(
                causal_policy.proposal,
                offered_now,
                model_choice=model_choice,
                task_family=getattr(self, "task_family", None),
                site=_page_site(page),
                phase=phase,
            )
            if proposal is not None:
                return proposal
        if trial_model is None:
            return None
        return _model_call(
            trial_model.choose,
            offered_now,
            model_choice=model_choice,
            task_family=getattr(self, "task_family", None),
            site=_page_site(page),
            phase=phase,
        )

    def _match_stamped_plan(self, proposals, *, page, offered_now, model_choice_id, goal_tokens):
        """Match a stamped DreamReport experiment plan against the live step.

        Returns ``(plan, stale_reasons)``. A stamped plan is an immutable
        hypothesis — "when the model picks A in this state under this policy,
        try B" — so every binding recorded at stamping time must still hold:
        the plan digest verifies over its own content, the task key matches
        this run's goal, the **task family** matches the run's family (a
        hypothesis generated under one family must not execute under another,
        even when goal text and state happen to collide), the state
        fingerprint matches, the live model choice equals the recorded
        historical choice, the proposal is still offered, and the offered
        catalogue and policy behavior digests still hold. Freshness is
        enforced too: a live causal model that now *refutes* the plan's
        proposal makes the plan stale — an old hypothesis does not survive
        newer randomized evidence. With trusted verification keys configured,
        the plan must additionally carry a valid domain-separated signature
        (``authority_signature_valid``), so a hand-written plan cannot
        masquerade as one stamped by the improvement authority.

        A plan that claims this state but fails any binding is *stale*: it is
        discarded, never reinterpreted, and its staleness suppresses an ad-hoc
        substitution for the same step — the experiment that runs must be the
        one that was stamped, or none.
        """
        stale: list[str] = []
        policy = getattr(self, "exploration_policy", None)
        live_policy_digest = getattr(policy, "behavior_digest", None)
        live_task_key = task_key(self.state["goal"])
        # Family identity is ``normalized task_family or task_key`` — the same
        # derivation the replay worlds use for ``family_key``.
        live_family = str(getattr(self, "task_family", None) or "").strip().lower() or live_task_key
        offered_ids = {a["id"] for a in offered_now}
        live_offered_digest = candidate_catalog_digest(
            [compact_candidate(candidate, goal_tokens) for candidate in offered_now]
        )
        experiment = getattr(self, "experiment", None) or {}
        trial_model = experiment.get("model")
        trusted_keys = self._experiment_verify_keys()
        require_model = bool(experiment.get("require_current_model"))
        for plan in proposals:
            if not isinstance(plan, dict) or plan.get("state") != page.get("fingerprint"):
                continue
            reasons = []
            if plan.get("digest") != experiment_plan_digest(plan):
                reasons.append("digest")
            if not plan.get("task_key") or plan.get("task_key") != live_task_key:
                reasons.append("task_key")
            if plan.get("family_key") != live_family:
                reasons.append("task_family")
            historical = plan.get("historical") or {}
            if historical.get("id") != model_choice_id:
                reasons.append("model_choice")
            proposal = plan.get("proposal")
            if not isinstance(proposal, dict) or proposal.get("id") not in offered_ids:
                reasons.append("proposal_not_offered")
            if plan.get("offered_catalogue_digest") != live_offered_digest:
                reasons.append("offered_catalogue")
            if plan.get("policy_behavior_digest") != live_policy_digest:
                reasons.append("policy_behavior")
            if plan.get("choice_model_digest") is not None:
                if trial_model is None:
                    # "Available and current": with require_current_model the
                    # originating model must be supplied and match; otherwise
                    # proposals-only execution stays possible in explicit
                    # unsigned mode.
                    if require_model:
                        reasons.append("choice_model")
                elif getattr(trial_model, "digest", None) != plan["choice_model_digest"]:
                    reasons.append("choice_model")
            # Freshness is checked against the *live* signature: the offered
            # catalogue digest already binds the candidates, so their live
            # effect/role/overlap classification is the honest basis for the
            # refuted query. Stamped values only fill what the live catalogue
            # cannot supply (a plan may outlive the fields it was stamped with).
            live_choice = next(
                (a for a in offered_now if a.get("id") == model_choice_id), None
            ) or {}
            live_proposal = next(
                (a for a in offered_now if a.get("id") == (proposal or {}).get("id")),
                None,
            ) or {}
            if (
                trial_model is not None
                and isinstance(proposal, dict)
                and callable(getattr(trial_model, "refuted", None))
                and _model_call(
                    trial_model.refuted,
                    kind=str(proposal.get("kind") or "unknown"),
                    goal_overlap=live_proposal.get("goal_overlap"),
                    model_kind=str(historical.get("kind") or "unknown"),
                    model_overlap=historical.get("goal_overlap"),
                    model_effect=live_choice.get("effect", historical.get("effect")),
                    model_role=live_choice.get("role", historical.get("role")),
                    model_rank=historical.get("offered_rank"),
                    proposal_effect=live_proposal.get("effect", proposal.get("effect")),
                    proposal_role=live_proposal.get("role", proposal.get("role")),
                    proposal_rank=plan.get("proposal_offered_rank"),
                    phase=plan.get("step"),
                    task_family=getattr(self, "task_family", None),
                    site=_page_site(page),
                )
            ):
                reasons.append("trial_refuted")
            authority = experiment_plan_authority(plan, trusted_keys)
            if trusted_keys:
                if not authority["trusted"]:
                    reasons.append(
                        "plan_signature" if authority["present"] else "plan_unsigned"
                    )
            elif authority["present"] and not authority["signature_valid"]:
                reasons.append("plan_signature")
            if not reasons:
                return plan, []
            stale.extend(reasons)
        return None, sorted(set(stale))

    def command(self, name, body=None):
        body = body or {}
        state = self.state

        if name == "tick":
            if state["status"] == "approval_required":
                raise ValueError("This action requires explicit approval or rejection.")
            try:
                self.command("predict", {})
                return self.command("act", {"fingerprint": state["page"]["fingerprint"]})
            except StalePage:
                state["decision"] = None
                state["status"] = "ready"
                state["page"] = state["browser"].observe(screenshot=self.screenshots)
                state["elapsed_ms"] = self._elapsed()
                return self.snapshot()

        if name == "predict":
            if not state["browser"]:
                raise ValueError("Start a demo first")
            if state["started_at"] is None:
                state["started_at"] = time.perf_counter()
            if not state["browser"].fresh(state["page"]):
                state["page"] = state["browser"].observe(screenshot=self.screenshots)
            state["decision"] = None
            # A spent or orphaned approval grant never survives into a fresh
            # decision; only approve() can create one, bound to one act() call.
            # The same goes for a trial assignment left over from an approval
            # that never executed — it dies with the decision it belonged to.
            state["granted_approval"] = None
            state["experiment_assignment"] = None
            state["causal_shadow"] = None
            state["causal_override"] = None
            if state["status"] in TERMINAL_STATUSES:
                raise ValueError("This run has stopped. Start a fresh demo.")
            if state["status"] == "approval_required":
                raise ValueError("Resolve the pending approval before predicting again.")
            model_call_budget = getattr(getattr(self, "exploration_policy", None), "max_actions", MAX_STEPS) * 2
            if len(state["decisions"]) >= model_call_budget:
                raise ValueError("Reached the model-call budget")
            state["decision"] = choose(
                state["page"],
                state["goal"],
                state["history"],
                backend=getattr(self, "decision_backend", None),
                exploration_policy=getattr(self, "exploration_policy", None),
                perception=self._perception(),
            )
            state["decisions"].append(
                {
                    **state["decision"],
                    "fingerprint": state["page"]["fingerprint"],
                    "elapsed_ms": self._elapsed(),
                }
            )
            # Shadow mode: annotate what the combined causal policy *would*
            # prefer, with provenance, without letting it influence anything.
            causal_policy = (getattr(self, "experiment", None) or {}).get("causal_policy")
            if causal_policy is not None and getattr(causal_policy, "mode", "shadow") == "shadow":
                offered_now, _goal_tokens = self._annotated_offered()
                choice_id = state["decision"]["choice"]
                state["causal_shadow"] = _model_call(
                    causal_policy.rank,
                    offered_now,
                    model_choice={
                        "id": choice_id,
                        "kind": next(
                            (a.get("kind") for a in offered_now if a.get("id") == choice_id),
                            None,
                        ),
                        "role": next(
                            (a.get("role") for a in offered_now if a.get("id") == choice_id),
                            None,
                        ),
                        "effect": next(
                            (a.get("effect") for a in offered_now if a.get("id") == choice_id),
                            None,
                        ),
                        "goal_overlap": next(
                            (a.get("goal_overlap") for a in offered_now if a.get("id") == choice_id),
                            None,
                        ),
                    },
                    task_family=getattr(self, "task_family", None),
                    site=_page_site(state["page"]),
                    phase=len(
                        [
                            h
                            for h in state["history"]
                            if h.get("kind") not in {"verification", "approval"}
                        ]
                    ),
                )
            state["status"] = "predicted"
            return self.snapshot()

        if name == "approve":
            pending = state.get("pending_approval")
            if state["status"] != "approval_required" or not pending:
                raise ValueError("There is no pending action to approve.")
            state["pending_approval"] = None
            # Approval is a server-side capability bound to the pending action on
            # the pending page. act() never derives authority from caller input;
            # the grant is consumed exactly once by the act() it enables.
            state["granted_approval"] = {
                "action_id": pending["action"]["id"],
                "fingerprint": pending["fingerprint"],
                # One-shot authority bound to the exact payload reviewed: for a
                # TYPE_TEXT fill the grant covers only the generated text that
                # was pending, never whatever a regenerated helper would emit.
                "payload_digest": pending.get("payload_digest"),
            }
            # Execute the action that was actually presented for approval — for
            # an experiment-deviated step that is the proposal, not the model's
            # original decision choice, which act() would otherwise re-derive.
            state["decision"] = {
                **pending["decision"],
                "choice": pending["action"]["id"],
            }
            state["experiment_assignment"] = pending.get("experiment")
            state["status"] = "predicted"
            return self.command("act", {"fingerprint": pending["fingerprint"]})

        if name == "reject":
            if state["status"] != "approval_required" or not state.get("pending_approval"):
                raise ValueError("There is no pending action to reject.")
            pending = state["pending_approval"]
            state["pending_approval"] = None
            state["granted_approval"] = None
            state["status"] = "ready"
            state["history"].append(
                {
                    "step": len(state["history"]) + 1,
                    "action": pending["action"]["label"],
                    "kind": "approval",
                    "page_changed": False,
                    "approval": "rejected",
                    "reason": pending["reason"],
                    "elapsed_ms": self._elapsed(),
                }
            )
            return self.snapshot()

        if name == "act":
            if body.get("approved") is not None:
                raise ValueError(
                    "act() does not accept caller-supplied approval; use the approve command."
                )
            decision, page = state["decision"], state["page"]
            if not decision or body.get("fingerprint") != page["fingerprint"]:
                raise ValueError("Observe and choose before acting")
            state["decision"] = None  # consume exactly once before model calls or mutations
            selected = decision["choice"]

            if selected in {"DONE", "BLOCKED"}:
                if not state["browser"].fresh(page):
                    state["status"] = "ready"
                    raise StalePage("Page changed since the decision. Choose again.")
                if selected == "BLOCKED":
                    state["status"] = "blocked"
                    if getattr(self, "dream_recorder", None):
                        self.dream_recorder.finish(status="blocked", verified=False)
                elif getattr(self, "verifier", None) is None:
                    state["status"] = "claimed_done"
                    state["verified"] = False
                    state["plan_index"] = 1
                else:
                    verification = self._verify(page)
                    state["verification"] = verification
                    if verification["passed"]:
                        state["status"] = "done"
                        state["verified"] = True
                        state["plan_index"] = 1
                    else:
                        state["verification_failures"].append(verification)
                        state["history"].append(
                            {
                                "step": len(state["history"]) + 1,
                                "action": "DONE rejected by verifier",
                                "kind": "verification",
                                "choice": "DONE",
                                "page_changed": False,
                                "verification": verification,
                                "elapsed_ms": self._elapsed(),
                            }
                        )
                        state["status"] = "ready"
                if state["status"] in {"done", "claimed_done", "blocked"} and getattr(self, "dream_recorder", None):
                    self.dream_recorder.finish(status=state["status"], verified=state.get("verified", False))
                state["elapsed_ms"] = self._elapsed()
                return self.snapshot()

            action = next((a for a in page["actions"] if a["id"] == selected), None)
            if action is None:
                state["status"] = "ready"
                raise StalePage("Selected action is no longer present. Observe again.")

            # A trial assigned before an approval pause resumes with its arm —
            # approve() restores it so the executed action and its provenance
            # survive the round trip instead of silently reverting.
            experiment_meta = state.pop("experiment_assignment", None)
            experiment = getattr(self, "experiment", None)
            causal_policy = (experiment or {}).get("causal_policy")
            if (
                experiment_meta is None
                and causal_policy is not None
                and getattr(causal_policy, "mode", "shadow") == "active"
                and not state.get("experiment_spent")
            ):
                # Active mode (operator-gated): a randomized-established
                # beneficial proposal overrides the model's own choice. The
                # action stays bounded to the offered catalogue and still
                # passes the full authority plane below; the override is
                # recorded as provenance and keeps the run out of canary
                # qualification, because the executed action was the causal
                # policy's, not the exploration policy's.
                offered_now, _goal_tokens = self._annotated_offered()
                override = _model_call(
                    causal_policy.proposal,
                    offered_now,
                    model_choice={
                        "id": selected,
                        "kind": action.get("kind"),
                        "role": action.get("role"),
                        "effect": classify_effect(action).value,
                        "goal_overlap": next(
                            (c.get("goal_overlap") for c in offered_now if c.get("id") == selected),
                            None,
                        ),
                    },
                    task_family=getattr(self, "task_family", None),
                    site=_page_site(page),
                    phase=len(
                        [
                            h
                            for h in state["history"]
                            if h.get("kind") not in {"verification", "approval"}
                        ]
                    ),
                )
                if (
                    override is not None
                    and override.get("id") is not None
                    and override["id"] != selected
                    and override["id"] in {a["id"] for a in offered_now}
                ):
                    state["experiment_spent"] = True
                    state["causal_override"] = override
                    if getattr(self, "dream_recorder", None):
                        self.dream_recorder.event(
                            "causal_policy_applied",
                            state=page.get("fingerprint"),
                            selected=override["id"],
                            proposal=override,
                        )
                    selected = override["id"]
                    action = next(a for a in offered_now if a["id"] == selected)
            if experiment_meta is None and experiment is not None and not state.get("experiment_spent"):
                # On-policy counterfactual trial: when the fitted choice prior
                # prefers a different *offered* action than the model picked,
                # the scheduler assigns an arm with a recorded probability and
                # executes it for real. The executed action still passes
                # through every authority check below — a trial can never
                # bypass policy assessment, approval, or payload review.
                trial_model = experiment.get("model")
                causal_policy = experiment.get("causal_policy")
                rate = float(experiment.get("rate", 0.0) or 0.0)
                if MIN_EXPERIMENT_ARM_RATE <= rate <= 1.0 - MIN_EXPERIMENT_ARM_RATE:
                    offered_now, goal_tokens = self._annotated_offered()
                    # A stamped DreamReport experiment plan is immutable: it
                    # executes only when every stamped binding still verifies
                    # (digest, task, state, model choice, offered catalogue,
                    # policy behavior). A state-matching plan that fails is
                    # stale — discarded, and it suppresses an ad-hoc
                    # substitution for this step rather than silently running
                    # a different hypothesis under its identity. States no
                    # plan claims fall through to the live prior, if provided.
                    stamped, stale_reasons = self._match_stamped_plan(
                        experiment.get("proposals") or (),
                        page=page,
                        offered_now=offered_now,
                        model_choice_id=selected,
                        goal_tokens=goal_tokens,
                    )
                    if stale_reasons and getattr(self, "dream_recorder", None):
                        self.dream_recorder.event(
                            "experiment_plan_stale",
                            state=page.get("fingerprint"),
                            reasons=stale_reasons,
                        )
                    proposal = (
                        dict(stamped.get("proposal") or {})
                        if stamped is not None
                        else (
                            None
                            if stale_reasons
                            else self._live_proposal(
                                trial_model=trial_model,
                                causal_policy=causal_policy,
                                offered_now=offered_now,
                                selected=selected,
                                action=action,
                                page=page,
                            )
                        )
                    )
                    # Hard boundary: a proposal is eligible only if it is one of
                    # the actions the exploration policy actually offered the
                    # model — page["actions"] is the pre-policy catalogue and
                    # must never widen the experimental surface.
                    offered_ids = {a["id"] for a in offered_now}
                    if (
                        proposal is not None
                        and proposal.get("id") is not None
                        and proposal["id"] != selected
                        and proposal["id"] in offered_ids
                    ):
                        state["experiment_spent"] = True
                        rng = experiment.get("rng")
                        roll = (rng.random() if rng is not None else random.random())
                        arm = "candidate" if roll < rate else "control"
                        selected_rank = next(
                            (i for i, a in enumerate(offered_now) if a["id"] == selected),
                            None,
                        )
                        proposal_rank = next(
                            (i for i, a in enumerate(offered_now) if a["id"] == proposal["id"]),
                            None,
                        )
                        expected_delta = None
                        if stamped is not None:
                            expected_delta = stamped.get("expected_delta")
                        elif proposal.get("expected_delta") is not None:
                            # A causal-prior proposal carries its measured
                            # effect directly.
                            expected_delta = proposal["expected_delta"]
                        elif selected_rank is not None and callable(getattr(trial_model, "predict", None)):
                            baseline_pred = _model_call(
                                trial_model.predict,
                                kind=action.get("kind"),
                                goal_overlap=offered_now[selected_rank].get("goal_overlap", 0),
                                rank=selected_rank,
                                proposal_effect=classify_effect(action).value,
                                proposal_role=action.get("role"),
                                task_family=getattr(self, "task_family", None),
                                site=_page_site(page),
                            )
                            if (
                                isinstance(baseline_pred, dict)
                                and isinstance(baseline_pred.get("p_progress"), (int, float))
                                and isinstance(proposal.get("p_progress"), (int, float))
                            ):
                                expected_delta = float(proposal["p_progress"]) - float(
                                    baseline_pred["p_progress"]
                                )
                        policy = getattr(self, "exploration_policy", None)
                        policy_digest = getattr(policy, "behavior_digest", None)
                        model_digest = getattr(trial_model, "digest", None)
                        proposal_digest = hashlib.sha256(
                            json.dumps(
                                {
                                    "state": page.get("fingerprint"),
                                    "model_choice_id": selected,
                                    "proposal_id": proposal["id"],
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        ).hexdigest()
                        # Immutable assignment: everything a later audit needs
                        # to re-derive what was proposed and why is frozen here
                        # — the offered catalogue, the policy behavior digest,
                        # the model's identity, the prior's own predicted delta
                        # and uncertainty, and the pre-treatment context the
                        # intention-to-treat estimator groups on (task
                        # identity, each side's overlap and offered rank). The
                        # assignment must be self-contained: an assigned-but-
                        # never-executed trial leaves no transition to recover
                        # this context from.
                        # The bounded treatment signature of both arms of the
                        # divergence, frozen at assignment time: kind, effect
                        # class, element role, overlap, offered rank, and the
                        # workflow phase (browser-action count so far). Trial
                        # analysis keys on these coordinates so a measured
                        # delta generalizes over a treatment *class* with
                        # provenance, never silently onto an arbitrary action
                        # ID that was never randomized.
                        proposal_action = next(
                            (a for a in offered_now if a["id"] == proposal["id"]),
                            None,
                        )
                        phase = len(
                            [
                                h
                                for h in state["history"]
                                if h.get("kind") not in {"verification", "approval"}
                            ]
                        )
                        experiment_meta = {
                            "state": page.get("fingerprint"),
                            "arm": arm,
                            "assignment_probability": (
                                rate if arm == "candidate" else 1.0 - rate
                            ),
                            "proposal_id": proposal["id"],
                            "proposal_kind": proposal.get("kind"),
                            "proposal_effect": (
                                classify_effect(proposal_action).value
                                if proposal_action is not None
                                else None
                            ),
                            "proposal_role": (
                                proposal_action.get("role")
                                if proposal_action is not None
                                else None
                            ),
                            "proposal_overlap": (
                                offered_now[proposal_rank].get("goal_overlap", 0)
                                if proposal_rank is not None
                                else None
                            ),
                            "proposal_offered_rank": proposal_rank,
                            "model_choice_id": selected,
                            "model_choice_kind": action.get("kind"),
                            "model_choice_effect": classify_effect(action).value,
                            "model_choice_role": action.get("role"),
                            "model_choice_overlap": (
                                offered_now[selected_rank].get("goal_overlap", 0)
                                if selected_rank is not None
                                else None
                            ),
                            "model_choice_offered_rank": selected_rank,
                            "phase": phase,
                            "task_key": task_key(state["goal"]),
                            "task_family": getattr(self, "task_family", None),
                            "site": _page_site(page),
                            "instance_id": getattr(self, "instance_id", None),
                            "proposal_digest": proposal_digest,
                            "stamped_digest": stamped.get("digest") if stamped else None,
                            "choice_model_digest": (
                                model_digest if isinstance(model_digest, str) else None
                            ),
                            "policy_behavior_digest": (
                                policy_digest if isinstance(policy_digest, str) else None
                            ),
                            "offered_catalogue_digest": candidate_catalog_digest(
                                [
                                    compact_candidate(candidate, goal_tokens)
                                    for candidate in offered_now
                                ]
                            ),
                            "proposal_p_progress": proposal.get("p_progress"),
                            "proposal_uncertainty": proposal.get("uncertainty"),
                            "expected_delta": expected_delta,
                            "experiment_id": (
                                stamped["digest"][:16]
                                if stamped and stamped.get("digest")
                                else proposal_digest[:16]
                            ),
                        }
                        # Randomization is evidence in its own right: record it
                        # BEFORE the authority plane so a denied, rejected,
                        # stale, or aborted trial still marks this run as
                        # experimental and can never qualify as a canary.
                        if getattr(self, "dream_recorder", None):
                            self.dream_recorder.event(
                                "experiment_assigned",
                                state=page.get("fingerprint"),
                                experiment=experiment_meta,
                            )
                        if arm == "candidate":
                            selected = proposal["id"]
                            action = next(
                                a for a in offered_now if a["id"] == selected
                            )

            browser_actions = [h for h in state["history"] if h.get("kind") not in {"verification", "approval"}]
            max_actions = getattr(getattr(self, "exploration_policy", None), "max_actions", MAX_STEPS)
            if len(browser_actions) >= max_actions:
                state["status"] = "blocked"
                if getattr(self, "dream_recorder", None):
                    self.dream_recorder.finish(status="blocked", verified=False)
                raise ValueError(f"Stopped at the {max_actions}-action budget")

            policy_result = None
            if getattr(self, "policy", None):
                policy_result = self.policy.assess(action, page=page, goal=state["goal"])
                if policy_result.level == "deny":
                    state["status"] = "blocked"
                    if getattr(self, "dream_recorder", None):
                        self.dream_recorder.event(
                            "policy_denied",
                            state=page.get("fingerprint"),
                            selected=action.get("id"),
                            # A denied trial action ends the run — tag it so the
                            # scheduler-caused block is never scored as the
                            # policy's own failure in canary.
                            experiment=experiment_meta,
                            reason=policy_result.reason,
                        )
                        self.dream_recorder.finish(status="blocked", verified=False)
                    raise ValueError(f"Action denied by policy: {policy_result.reason}")

            text, helper = None, None
            if action["kind"] == "fill":
                if not state["browser"].fresh(page):
                    raise StalePage("Page changed before text generation. Choose again.")
                # The payload is generated *before* the approval decision: the
                # granted authority binds the exact text being inserted, so an
                # operator approving "fill field X" cannot silently authorize a
                # payload nobody reviewed.
                context = field_context(state["goal"], action, page, state["history"])
                if self.pending_text and self.pending_text[0] == context:
                    _, text, helper = self.pending_text
                else:
                    text, helper = field_text(context)
                    self.pending_text = (context, text, helper)
                    state["text_calls"].append({**helper, "field": action["label"], "value": text})

            payload_result = None
            payload_digest = None
            if text is not None:
                payload_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if getattr(self, "policy", None):
                    # The target pass ran on page-supplied metadata, which a
                    # hostile page controls. The generated payload is a second,
                    # independent signal: a card number or credential typed into
                    # a field that claimed to be plain text escalates the fill.
                    payload_result = assess_payload(text)
                    if payload_result.level == "deny":
                        state["status"] = "blocked"
                        if getattr(self, "dream_recorder", None):
                            self.dream_recorder.event(
                                "policy_denied",
                                state=page.get("fingerprint"),
                                selected=action.get("id"),
                                experiment=experiment_meta,
                                reason=payload_result.reason,
                            )
                            self.dream_recorder.finish(status="blocked", verified=False)
                        raise ValueError(f"Action denied by policy: {payload_result.reason}")

            approval_consumed = False
            requires_approval = (
                (policy_result is not None and policy_result.level == "require_approval")
                or (payload_result is not None and payload_result.level == "require_approval")
            )
            if requires_approval:
                granted = state.pop("granted_approval", None)
                approval_consumed = bool(
                    granted
                    and granted.get("action_id") == action["id"]
                    and granted.get("fingerprint") == page["fingerprint"]
                    and granted.get("payload_digest") == payload_digest
                )
                if not approval_consumed:
                    reason = (
                        payload_result.reason
                        if payload_result is not None and payload_result.level == "require_approval"
                        else policy_result.reason
                    )
                    state["pending_approval"] = {
                        "decision": decision,
                        "action": action,
                        "fingerprint": page["fingerprint"],
                        "payload_digest": payload_digest,
                        # The exact generated value under review, so the
                        # approval screen can show precisely what the grant
                        # covers — the digest alone is opaque to a human.
                        "payload_preview": text,
                        # A trial assigned this step keeps its provenance across
                        # the approval pause; reject still drops the assignment.
                        "experiment": experiment_meta,
                        "reason": reason,
                    }
                    state["status"] = "approval_required"
                    state["elapsed_ms"] = self._elapsed()
                    if getattr(self, "dream_recorder", None):
                        self.dream_recorder.event(
                            "approval_required",
                            state=page.get("fingerprint"),
                            selected=action.get("id"),
                            # Tag the run even if the trial never executes —
                            # an operator veto is still scheduler influence.
                            experiment=experiment_meta,
                            reason=reason,
                        )
                    return self.snapshot()

            observed_candidates = offered_candidates = None
            if getattr(self, "dream_recorder", None):
                # The observed catalogue is pre-policy evidence: later replay can
                # evaluate candidates the active policy had discarded as well as
                # ones it offered. The offered catalogue is what the model saw.
                observed_candidates = page["actions"]
                offered_candidates = self._perception()[0]
                # Journal the dispatch BEFORE the mutation crosses into the
                # page: a crash between dispatch and acknowledgement must never
                # erase the fact that a mutation was attempted at all.
                self.dream_recorder.event(
                    "action_attempted",
                    state=page.get("fingerprint"),
                    selected=action.get("id"),
                    kind=action.get("kind"),
                    effect=classify_effect(action).value,
                    guarantee=self.input_guarantee,
                    experiment=experiment_meta,
                )
            try:
                state["browser"].act(
                    action, page, text=text, guarantee=self.input_guarantee
                )
            except StalePage:
                # Provably pre-mutation: safe to re-perceive — say so in the
                # journal instead of leaving the attempt dangling.
                if getattr(self, "dream_recorder", None):
                    self.dream_recorder.event(
                        "action_not_executed",
                        state=page.get("fingerprint"),
                        selected=action.get("id"),
                        experiment=experiment_meta,
                    )
                raise
            except IndeterminateMutation as exc:
                # The mutation may already have run. This is a terminal,
                # non-retryable outcome — deliberately NOT a StalePage, so the
                # tick loop cannot re-perceive and silently retry a click that
                # may have landed. The run aborts as censored evidence.
                state["elapsed_ms"] = self._elapsed()
                state["history"].append(
                    {
                        "step": len(state["history"]) + 1,
                        "action": action.get("label"),
                        "kind": action["kind"],
                        "choice": selected,
                        "probability": decision["probabilities"].get(selected),
                        "confidence": decision["confidence"],
                        "latency_ms": decision["latency_ms"],
                        "text": text,
                        "text_helper": helper["model"] if helper else None,
                        "text_latency_ms": helper["latency_ms"] if helper else 0,
                        "operation": decision["operation"],
                        "target": decision["target"],
                        "page_changed": None,
                        "execution": "indeterminate",
                        "url": page["url"],
                        "usage": decision["usage"],
                        "executed_ms": state["elapsed_ms"],
                        "elapsed_ms": state["elapsed_ms"],
                    }
                )
                if getattr(self, "dream_recorder", None):
                    self.dream_recorder.event(
                        "action_indeterminate",
                        state=page.get("fingerprint"),
                        selected=action.get("id"),
                        kind=action.get("kind"),
                        effect=classify_effect(action).value,
                        experiment=experiment_meta,
                        reason=str(exc),
                    )
                # Terminal for this run: close() records the aborted run under
                # its own censoring reason — the step is neither a measured
                # failure nor a clean retry.
                state["abort_reason"] = "indeterminate_execution"
                raise
            except (BrowserError, OSError, ConnectionError) as exc:
                # The transport died around dispatch: the mutation's fate is
                # unknowable (it may have been delivered), so the journal must
                # not claim a clean outcome.
                if getattr(self, "dream_recorder", None):
                    self.dream_recorder.event(
                        "action_indeterminate",
                        state=page.get("fingerprint"),
                        selected=action.get("id"),
                        kind=action.get("kind"),
                        effect=classify_effect(action).value,
                        experiment=experiment_meta,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                raise
            if getattr(self, "dream_recorder", None):
                self.dream_recorder.event(
                    "action_confirmed",
                    state=page.get("fingerprint"),
                    selected=action.get("id"),
                    kind=action.get("kind"),
                    experiment=experiment_meta,
                )
            self.pending_text = None
            state["elapsed_ms"] = self._elapsed()
            state["history"].append(
                {
                    "step": len(state["history"]) + 1,
                    "action": action["label"],
                    "kind": action["kind"],
                    "choice": selected,
                    "probability": decision["probabilities"].get(selected),
                    "confidence": decision["confidence"],
                    "latency_ms": decision["latency_ms"],
                    "text": text,
                    "text_helper": helper["model"] if helper else None,
                    "text_latency_ms": helper["latency_ms"] if helper else 0,
                    "operation": decision["operation"],
                    "target": decision["target"],
                    "page_changed": None,
                    "execution": "executed",
                    "url": page["url"],
                    "usage": decision["usage"],
                    "executed_ms": self._elapsed(),
                    "elapsed_ms": state["elapsed_ms"],
                }
            )
            state["page"] = state["browser"].observe(screenshot=self.screenshots)
            state["elapsed_ms"] = self._elapsed()
            state["history"][-1].update(
                page_changed=state["page"]["fingerprint"] != page["fingerprint"],
                url=state["page"]["url"],
                elapsed_ms=state["elapsed_ms"],
            )
            if state["record"]:
                (self.record_dir / f"{state['elapsed_ms']:06d}.jpg").write_bytes(
                    base64.b64decode(state["page"]["screenshot"])
                )
            if getattr(self, "dream_recorder", None):
                self.dream_recorder.transition(
                    before=page,
                    after=state["page"],
                    action=action,
                    decision=decision,
                    candidates=observed_candidates or [],
                    offered=offered_candidates,
                    helper=helper,
                    risk_events=int(approval_consumed),
                    experiment=experiment_meta,
                    causal_override=state.get("causal_override"),
                )
            browser_history = [h for h in state["history"] if h.get("kind") not in {"verification", "approval"}]
            no_progress_window = getattr(getattr(self, "exploration_policy", None), "no_progress_window", 3)
            repeated = browser_history[-no_progress_window:]
            state["status"] = (
                "blocked"
                if len(repeated) == no_progress_window
                and all(h["page_changed"] is False and h["kind"] != "wait" for h in repeated)
                else "ready"
            )
            if state["status"] == "blocked" and getattr(self, "dream_recorder", None):
                self.dream_recorder.finish(status="blocked", verified=False)
            return self.snapshot()

        raise ValueError("Unknown command")

    def run(self):
        while self.state["status"] not in TERMINAL_STATUSES | {"approval_required"}:
            try:
                yield self.command("tick")
            except Exception as exc:
                # A run that dies mid-flight records *why* before close()
                # writes its aborted run_finished: the censoring taxonomy
                # must be able to tell an operator cancel from a crash the
                # candidate arm may have caused.
                self.state["abort_reason"] = _abort_reason(exc)
                raise

    def close(self, reason: str | None = None):
        try:
            if getattr(self, "dream_recorder", None) and not self.dream_recorder.finished:
                status = self.state.get("status", "closed")
                if status in TERMINAL_STATUSES:
                    self.dream_recorder.finish(status=status, verified=self.state.get("verified", False))
                else:
                    self.dream_recorder.finish(
                        status="aborted",
                        verified=False,
                        reason=(
                            reason
                            or self.state.get("abort_reason")
                            or "operator_cancel"
                        ),
                    )
        finally:
            self.browser.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
