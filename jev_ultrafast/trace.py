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
                    "goal_overlap": action_goal_overlap(candidate, goal_tokens),
                })
            return entries

        compact_candidates = compact(candidates)
        compact_offered = compact(offered) if offered is not None else None
        selected_rank = next(
            (index for index, candidate in enumerate(compact_candidates) if candidate.get("id") == action.get("id")),
            None,
        )
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
            "selected_rank": selected_rank,
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
