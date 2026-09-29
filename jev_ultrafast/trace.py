"""Trace helpers that connect live Jev execution to DREAM-Jev replay."""

from __future__ import annotations

from .dream import ExperienceStore, candidate_catalog_digest, new_run_id, summarize_usage, task_key
from .privacy import redact_text, sanitize_action, sanitize_url


class DreamTraceRecorder:
    def __init__(self, store: ExperienceStore, *, goal: str, run_id: str | None = None):
        self.store = store
        self.goal = redact_text(goal, 4096)
        self.run_id = run_id or new_run_id()
        self.task_key = task_key(goal)
        self.sequence = 0
        self.started = False
        self.finished = False

    def _append(self, event: dict):
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
        })

    def transition(
        self,
        *,
        before: dict,
        after: dict,
        action: dict,
        decision: dict,
        candidates: list[dict],
        helper: dict | None = None,
        risk_events=0,
    ):
        selected = sanitize_action(action)
        compact_candidates = []
        goal_tokens = set(_tokens(self.goal))
        for candidate in candidates:
            clean = sanitize_action(candidate)
            searchable = " ".join(str(clean.get(k, "")) for k in ("label", "value", "current_value", "option_label"))
            compact_candidates.append({
                "id": candidate.get("id"),
                "kind": candidate.get("kind"),
                "label": clean.get("label", ""),
                "value": clean.get("value", clean.get("current_value", "")),
                "goal_overlap": len(goal_tokens & set(_tokens(searchable))),
            })
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
            },
            "candidates": compact_candidates,
            "candidate_digest": candidate_catalog_digest(compact_candidates),
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


def _tokens(value: str):
    current = []
    for char in value.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            word = "".join(current)
            if len(word) >= 2:
                yield word
            current = []
    if current:
        word = "".join(current)
        if len(word) >= 2:
            yield word
