"""Finite-choice agent loop with policy approval and independent completion verification."""

from __future__ import annotations

import base64
import inspect
import time
from pathlib import Path

from .browser import Browser, StalePage
from .dream import ExperienceStore, ExplorationPolicy, PolicyRegistry
from .model import action_space, candidate_actions, choose, field_context, field_text
from .policy import DefaultActionPolicy
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
        if isinstance(dream_store, (str, Path)):
            dream_store = ExperienceStore(dream_store)
        self.dream_recorder = (
            DreamTraceRecorder(dream_store, goal=task, run_id=run_id) if dream_store is not None else None
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

    def snapshot(self):
        candidates, omitted = candidate_actions(
            self.state["page"]["actions"],
            self.state["goal"],
            exploration_policy=getattr(self, "exploration_policy", None),
        )
        elements = action_space(candidates)[0]
        ordered_nodes = []
        for action in candidates:
            node = action.get("node")
            if action.get("kind") in {"click", "fill", "select"} and node not in ordered_nodes:
                ordered_nodes.append(node)
        for element, node in zip(elements, ordered_nodes, strict=True):
            element["node"] = node  # local inspector metadata; never sent to a decision backend
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
            state["decision"] = pending["decision"]
            state["status"] = "predicted"
            return self.command(
                "act",
                {"fingerprint": pending["fingerprint"], "approved": True},
            )

        if name == "reject":
            if state["status"] != "approval_required" or not state.get("pending_approval"):
                raise ValueError("There is no pending action to reject.")
            pending = state["pending_approval"]
            state["pending_approval"] = None
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
            browser_actions = [h for h in state["history"] if h.get("kind") not in {"verification", "approval"}]
            max_actions = getattr(getattr(self, "exploration_policy", None), "max_actions", MAX_STEPS)
            if len(browser_actions) >= max_actions:
                state["status"] = "blocked"
                if getattr(self, "dream_recorder", None):
                    self.dream_recorder.finish(status="blocked", verified=False)
                raise ValueError(f"Stopped at the {max_actions}-action budget")

            if not body.get("approved") and getattr(self, "policy", None):
                policy_result = self.policy.assess(action, page=page, goal=state["goal"])
                if policy_result.level == "deny":
                    state["status"] = "blocked"
                    if getattr(self, "dream_recorder", None):
                        self.dream_recorder.event(
                            "policy_denied",
                            state=page.get("fingerprint"),
                            selected=action.get("id"),
                            reason=policy_result.reason,
                        )
                        self.dream_recorder.finish(status="blocked", verified=False)
                    raise ValueError(f"Action denied by policy: {policy_result.reason}")
                if policy_result.level == "require_approval":
                    state["pending_approval"] = {
                        "decision": decision,
                        "action": action,
                        "fingerprint": page["fingerprint"],
                        "reason": policy_result.reason,
                    }
                    state["status"] = "approval_required"
                    state["elapsed_ms"] = self._elapsed()
                    if getattr(self, "dream_recorder", None):
                        self.dream_recorder.event(
                            "approval_required",
                            state=page.get("fingerprint"),
                            selected=action.get("id"),
                            reason=policy_result.reason,
                        )
                    return self.snapshot()

            text, helper = None, None
            if action["kind"] == "fill":
                if not state["browser"].fresh(page):
                    raise StalePage("Page changed before text generation. Choose again.")
                context = field_context(state["goal"], action, page, state["history"])
                if self.pending_text and self.pending_text[0] == context:
                    _, text, helper = self.pending_text
                else:
                    text, helper = field_text(context)
                    self.pending_text = (context, text, helper)
                    state["text_calls"].append({**helper, "field": action["label"], "value": text})

            trace_candidates = None
            if getattr(self, "dream_recorder", None):
                trace_candidates, _trace_omitted = candidate_actions(
                    page["actions"],
                    state["goal"],
                    exploration_policy=getattr(self, "exploration_policy", None),
                )
            state["browser"].act(action, page, text=text)
            self.pending_text = None
            state["elapsed_ms"] = self._elapsed()
            state["history"].append(
                {
                    "step": len(state["history"]) + 1,
                    "action": action["label"],
                    "kind": action["kind"],
                    "choice": selected,
                    "probability": decision["probabilities"][selected],
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
                    candidates=trace_candidates or [],
                    helper=helper,
                    risk_events=int(bool(body.get("approved"))),
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
