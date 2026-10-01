"""Trace helpers that connect live Jev execution to DREAM-Jev replay."""

from __future__ import annotations

from .dream import ExperienceStore, candidate_catalog_digest, new_run_id, summarize_usage, task_key
from .policy import classify_effect
from .privacy import action_goal_overlap, redact_text, sanitize_action, sanitize_url, tokenize

_RESERVED_EVENT_KEYS = {"run_id", "task_key", "sequence"}


class DreamTraceRecorder:
    def __init__(
        self,
        store: ExperienceStore,
        *,
        goal: str,
        run_id: str | None = None,
        task_family: str | None = None,
        instance_id: str | None = None,
    ):
        self.store = store
        self.goal = redact_text(goal, 4096)
        self.run_id = run_id or new_run_id()
        self.task_key = task_key(goal)
        self.task_family = str(task_family).strip()[:256] if task_family else None
        self.instance_id = str(instance_id).strip()[:256] if instance_id else None
        self.sequence = 0
        self.started = False
        self.finished = False

    def _append(self, event: dict):
        if _RESERVED_EVENT_KEYS & event.keys():
            raise ValueError(f"Trace event cannot set reserved keys: {sorted(_RESERVED_EVENT_KEYS & event.keys())}")
        self.sequence += 1
        self.store.append({
            "run_id": self.run_id,
            "task_key": self.task_key,
            "sequence": self.sequence,
            **event,
        })

    def start(self, page: dict, policy):
        if self.started:
            return
        self.started = True
        self._append({
            "event": "run_started",
            "goal": self.goal,
            "state": page.get("fingerprint", "ROOT"),
            "url": sanitize_url(page.get("url", "")),
            "policy": policy.to_dict() if policy else None,
            "policy_digest": policy.digest if policy else None,
            "task_family": self.task_family,
            "instance_id": self.instance_id,
        })

    def transition(
        self,
        *,
        before: dict,
        after: dict,
        action: dict,
        decision: dict,
        candidates: list[dict],
        offered: list[dict] | None = None,
        helper: dict | None = None,
        risk_events=0,
        experiment: dict | None = None,
    ):
        selected = sanitize_action(action)
        goal_tokens = set(tokenize(self.goal))

        def compact(actions):
            entries = []
            for candidate in actions:
                clean = sanitize_action(candidate)
                entries.append({
                    "id": candidate.get("id"),
                    "kind": candidate.get("kind"),
                    "node": candidate.get("node"),
                    "label": clean.get("label", ""),
                    "value": clean.get("value", clean.get("current_value", "")),
                    "goal_overlap": action_goal_overlap(candidate, goal_tokens, clean=clean),
                })
            return entries

        compact_candidates = compact(candidates)
        compact_offered = compact(offered) if offered is not None else None
        # The two rank coordinates are different evidence: the observed rank is
        # the action's position in the pre-policy catalogue, the offered rank
        # its position in what the model was actually shown. Learned models
        # must train on the offered basis — replay queries them there.
        selected_observed_rank = next(
            (index for index, candidate in enumerate(compact_candidates) if candidate.get("id") == action.get("id")),
            None,
        )
        selected_offered_rank = (
            next(
                (index for index, candidate in enumerate(compact_offered) if candidate.get("id") == action.get("id")),
                None,
            )
            if compact_offered is not None
            else None
        )
        # The executor selects by deterministic argmax over the model's heads,
        # so head probabilities are model *scores*, not sampling propensities —
        # treating one as a propensity would let an off-policy estimator
        # weight by a denominator that never existed. The only real behavior
        # propensity is an experiment's recorded ``assignment_probability``;
        # deterministic selection records ``behavior_propensity = None``.
        probabilities = decision.get("probabilities") or {}
        operation_probabilities = decision.get("operation_probabilities") or {}
        target_probabilities = decision.get("target_probabilities") or {}
        target = decision.get("target")
        operation_p = operation_probabilities.get(decision.get("operation"))
        target_p = target_probabilities.get(target) if target is not None else None
        if operation_p is not None and target_p is not None:
            joint_model_score = operation_p * target_p
        elif operation_p is not None:
            joint_model_score = operation_p
        else:
            joint_model_score = target_p
        assignment_propensity = None
        if isinstance(experiment, dict) and experiment.get("assignment_probability") is not None:
            assignment_propensity = float(experiment["assignment_probability"])
        self._append({
            "event": "transition",
            "state": before.get("fingerprint", ""),
            "next_state": after.get("fingerprint", before.get("fingerprint", "")),
            "selected": {
                "id": action.get("id"),
                "kind": action.get("kind"),
                "label": selected.get("label", ""),
                # Deterministic effect class — auditable evidence of which
                # authority tier the executed mutation was classified under.
                "effect": classify_effect(action).value,
            },
            "candidates": compact_candidates,
            "candidate_digest": candidate_catalog_digest(compact_candidates),
            "catalog": "observed" if compact_offered is not None else "recorded",
            "offered_digest": (
                candidate_catalog_digest(compact_offered) if compact_offered is not None else None
            ),
            "offered_count": len(compact_offered) if compact_offered is not None else None,
            "selected_observed_rank": selected_observed_rank,
            "selected_offered_rank": selected_offered_rank,
            # selection_mode is "argmax": deterministic execution has no
            # behavior propensity, so behavior_propensity stays None. The
            # model's own scores are kept under honest names — a score is
            # evidence of model preference, not of assignment probability.
            "selection_mode": "argmax",
            "behavior_propensity": assignment_propensity,
            "selected_model_score": probabilities.get(action.get("id")),
            "joint_model_score": joint_model_score,
            "operation_probability": operation_p,
            "target_probability": target_p,
            "decision_confidence": decision.get("confidence"),
            "action_probabilities": {
                str(k): round(float(v), 6) for k, v in probabilities.items()
            },
            "operation_probabilities": {
                str(k): round(float(v), 6) for k, v in operation_probabilities.items()
            },
            # Experiment provenance (present only on assigned trials): which
            # arm executed, the scheduler's assignment probability — the
            # propensity any off-policy correction is weighted by — and the
            # model choice the trial deviated from or confirmed.
            "experiment": experiment,
            "page_changed": after.get("fingerprint") != before.get("fingerprint"),
            "latency_ms": int(decision.get("latency_ms", 0)) + int((helper or {}).get("latency_ms", 0)),
            "model_calls": 1 + int(helper is not None),
            "tokens": summarize_usage(decision.get("usage")) + summarize_usage((helper or {}).get("usage")),
            "stale_or_failure": 0,
            "risk_events": int(bool(risk_events)),
            "url": sanitize_url(after.get("url", "")),
        })

    def event(self, name: str, **payload):
        self._append({"event": name, **payload})

    def finish(self, *, status: str, verified: bool):
        if self.finished:
            return
        self.finished = True
        self._append({"event": "run_finished", "status": status, "verified": bool(verified)})
