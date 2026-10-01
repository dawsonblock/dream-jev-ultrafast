"""Finite-choice agent loop with policy approval and independent completion verification."""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import random
import time
from pathlib import Path

from .browser import Browser, StalePage
from .dream import ExperienceStore, ExplorationPolicy, PolicyRegistry, candidate_catalog_digest
from .model import action_space, candidate_actions, choose, field_context, field_text
from .policy import DefaultActionPolicy, assess_payload
from .privacy import action_goal_overlap, tokenize
from .questions import MAX_STEPS
from .trace import DreamTraceRecorder

TERMINAL_STATUSES = {"claimed_done", "done", "blocked"}


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
    ):
        task = goals.strip() if isinstance(goals, str) else "\n".join(goals).strip()
        if not task:
            raise ValueError("Supply a task")
        plan = [task]
        self.pending_text = None
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
            if not 0.0 < rate <= 1.0:
                raise ValueError("experiment rate must be in (0, 1]")
            proposals = experiment.get("proposals")
            if proposals is not None and not isinstance(proposals, (list, tuple)):
                raise ValueError("experiment proposals must be a list of stamped plans")
            if experiment.get("model") is None and not proposals:
                raise ValueError("experiment requires a fitted model or stamped proposals")
        self.experiment = experiment
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
            **{k: v for k, v in self.state.items() if k != "browser"},
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
            if experiment_meta is None and experiment is not None and not state.get("experiment_spent"):
                # On-policy counterfactual trial: when the fitted choice prior
                # prefers a different *offered* action than the model picked,
                # the scheduler assigns an arm with a recorded probability and
                # executes it for real. The executed action still passes
                # through every authority check below — a trial can never
                # bypass policy assessment, approval, or payload review.
                trial_model = experiment.get("model")
                rate = float(experiment.get("rate", 0.0) or 0.0)
                if 0.0 < rate <= 1.0:
                    offered_now = self._perception()[0]
                    # The prior was trained on the trace's sanitized features;
                    # raw page actions carry no goal_overlap, so annotate it —
                    # otherwise every live proposal is scored on a zeroed
                    # overlap the model never saw in training.
                    goal_tokens = set(tokenize(state["goal"]))
                    offered_now = [
                        {**c, "goal_overlap": action_goal_overlap(c, goal_tokens)}
                        for c in offered_now
                    ]
                    # A stamped DreamReport experiment proposal is matched to
                    # this state by fingerprint; otherwise the live model
                    # chooses from the same bounded catalogue.
                    stamped = next(
                        (
                            p for p in experiment.get("proposals") or ()
                            if isinstance(p, dict)
                            and p.get("state") == page.get("fingerprint")
                        ),
                        None,
                    )
                    proposal = (
                        dict(stamped.get("proposal") or {})
                        if stamped is not None
                        else (trial_model.choose(offered_now) if trial_model is not None else None)
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
                        expected_delta = None
                        if stamped is not None:
                            expected_delta = stamped.get("expected_delta")
                        elif selected_rank is not None and callable(getattr(trial_model, "predict", None)):
                            baseline_pred = trial_model.predict(
                                kind=action.get("kind"),
                                goal_overlap=offered_now[selected_rank].get("goal_overlap", 0),
                                rank=selected_rank,
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
                        # the model's identity, and the prior's own predicted
                        # delta and uncertainty.
                        experiment_meta = {
                            "arm": arm,
                            "assignment_probability": (
                                rate if arm == "candidate" else 1.0 - rate
                            ),
                            "proposal_id": proposal["id"],
                            "proposal_kind": proposal.get("kind"),
                            "model_choice_id": selected,
                            "model_choice_kind": action.get("kind"),
                            "proposal_digest": proposal_digest,
                            "stamped_digest": stamped.get("digest") if stamped else None,
                            "choice_model_digest": (
                                model_digest if isinstance(model_digest, str) else None
                            ),
                            "policy_behavior_digest": (
                                policy_digest if isinstance(policy_digest, str) else None
                            ),
                            "offered_catalogue_digest": candidate_catalog_digest(offered_now),
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
            state["browser"].act(action, page, text=text)
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
            yield self.command("tick")

    def close(self):
        try:
            if getattr(self, "dream_recorder", None) and not self.dream_recorder.finished:
                status = self.state.get("status", "closed")
                self.dream_recorder.finish(
                    status=status if status in TERMINAL_STATUSES else "aborted",
                    verified=self.state.get("verified", False),
                )
        finally:
            self.browser.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
