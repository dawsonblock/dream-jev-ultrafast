"""Finite-choice decision backends plus a bounded OpenAI-compatible text helper."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from .privacy import (
    action_goal_overlap,
    assert_endpoint_allowed,
    assert_transport_secure,
    loopback_endpoint,
    outbound_text,
    redact_text,
    sanitize_action,
    sanitize_page,
    tokenize,
)
from .questions import NEXT_ACTION, TARGET, TEXT_VALUE

CLIENT = httpx.Client(http2=True, timeout=float(os.environ.get("JEV_MODEL_TIMEOUT", "25")))
MODEL_ACTION_LIMIT = 250
MODEL_BODY_LIMIT = 256 * 1024


class DecisionBackend(Protocol):
    def decide(self, body: dict) -> dict: ...


class ModelConnectionError(RuntimeError):
    """The decision/text model endpoint could not be reached."""


class ModelTimeoutError(ModelConnectionError):
    """The decision/text model endpoint did not answer in time.

    A distinct subclass so run termination reasons can separate ``timeout``
    from ``network_failure``: the censoring taxonomy needs to tell an
    unrelated infrastructure hiccup from a hang the candidate arm may have
    caused.
    """


@dataclass
class SystemOneBackend:
    """TypeSafe/SystemOne-compatible backend. The endpoint may be local or remote."""

    url: str | None = None
    api_key: str | None = None

    def decide(self, body):
        url = self.url or os.environ.get("JEV_DECISION_BASE_URL", "https://api.typesafe.ai/v1/systemone")
        # Routing level binds before the request is even assembled: a
        # local-only run refuses a remote endpoint outright, and every
        # non-loopback endpoint requires TLS — the Authorization header and
        # model context must never travel in cleartext.
        assert_endpoint_allowed(url)
        assert_transport_secure(url)
        key = self.api_key or os.environ.get("JEV_DECISION_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
        if not key and not loopback_endpoint(url):
            raise ValueError(
                "Remote decision backend needs JEV_DECISION_API_KEY or TYPESAFE_API_KEY; no action executed."
            )
        return post_json(url, key, body)


def post_json(url, key, body):
    encoded = json.dumps(body, separators=(",", ":")).encode()
    if len(encoded) > int(os.environ.get("JEV_MODEL_BODY_LIMIT", MODEL_BODY_LIMIT)):
        raise ValueError("Model request exceeded the configured context budget; no action executed.")
    for attempt in range(3):
        try:
            headers = {"Content-Type": "application/json"}
            if key:
                headers["Authorization"] = f"Bearer {key}"
            response = CLIENT.post(url, content=encoded, headers=headers)
        except httpx.TimeoutException:
            raise ModelTimeoutError("Model connection timed out; no action executed.") from None
        except httpx.HTTPError:
            raise ModelConnectionError("Model connection failed; no action executed.") from None
        if response.status_code in {429, 529, 503} and attempt < 2:
            time.sleep(0.5 * 2**attempt)
            continue
        if response.is_error:
            raise RuntimeError(f"Model provider returned HTTP {response.status_code}; no action executed.")
        return response.json()
    raise RuntimeError("Model unavailable")


def validate_choice(answer, ids):
    try:
        probabilities = answer["probabilities"]
        numbers = [*probabilities.values(), answer["confidence"]]
        valid = (
            answer["choice"] in ids
            and set(probabilities) == set(ids)
            and all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 1 for n in numbers)
            and abs(sum(probabilities.values()) - 1) < 0.02
            and probabilities[answer["choice"]] >= max(probabilities.values()) - 1e-6
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Invalid TypeSafe/finite-choice response; no action executed.")
    return answer


def _goal_tokens(goal):
    return set(tokenize(goal))


def _candidate_score(action, goal_tokens, order, exploration_policy=None, overlap=None):
    if exploration_policy is not None:
        return exploration_policy.candidate_score(action, goal_tokens, order, overlap=overlap)
    if overlap is None:
        overlap = action_goal_overlap(action, goal_tokens)
    kind_bonus = {"fill": 1.5, "select": 1.0, "click": 0.5}.get(action.get("kind"), 0)
    return overlap * 10 + kind_bonus - order / 100000


def candidate_actions(actions, goal, limit=None, exploration_policy=None):
    """Goal-aware, operation-balanced candidate selection.

    System controls (scroll/wait) are retained. Per-kind quotas prevent one large
    native dropdown from consuming the entire model action budget. A learned
    ``ExplorationPolicy`` may additionally bound per-node fan-out and require a
    minimum goal overlap; both stay replayable because trace candidates carry
    the same ``node`` identity and sanitized overlap.
    """
    limit = int(limit or getattr(exploration_policy, "model_action_limit", MODEL_ACTION_LIMIT))
    min_overlap = int(getattr(exploration_policy, "min_goal_overlap", 0) or 0)
    node_cap = int(getattr(exploration_policy, "duplicate_node_cap", 250) or 250)
    controls = [a for a in actions if a.get("kind") not in {"click", "fill", "select"}]
    tokens = _goal_tokens(goal)
    # Overlap is computed once per action and shared by the min-overlap filter
    # and the scorer — scoring an unfiltered catalogue would otherwise re-run
    # the sanitize+tokenize pass inside every comparison key.
    regular = []
    overlaps = {}
    for i, a in enumerate(actions):
        if a.get("kind") not in {"click", "fill", "select"}:
            continue
        overlap = action_goal_overlap(a, tokens)
        if min_overlap and overlap < min_overlap:
            continue
        overlaps[id(a)] = overlap
        regular.append((i, a))
    ranked = sorted(
        regular,
        key=lambda pair: _candidate_score(
            pair[1], tokens, pair[0], exploration_policy, overlap=overlaps[id(pair[1])]
        ),
        reverse=True,
    )
    quotas = {
        "click": getattr(exploration_policy, "click_quota", 150),
        "fill": getattr(exploration_policy, "fill_quota", 45),
        "select": getattr(exploration_policy, "select_quota", 55),
    }
    selected, counts, used, node_counts = [], {k: 0 for k in quotas}, set(), {}
    regular_budget = max(0, limit - len(controls))
    for index, action in ranked:
        kind = action["kind"]
        node = action.get("node")
        group = ("n", node) if node is not None else ("~", index)
        if (
            counts[kind] >= quotas[kind]
            or len(selected) >= regular_budget
            or node_counts.get(group, 0) >= node_cap
        ):
            continue
        selected.append((index, action))
        used.add(index)
        counts[kind] += 1
        node_counts[group] = node_counts.get(group, 0) + 1
    if len(selected) < regular_budget:
        for index, action in ranked:
            if index in used:
                continue
            node = action.get("node")
            group = ("n", node) if node is not None else ("~", index)
            if node_counts.get(group, 0) >= node_cap:
                continue
            selected.append((index, action))
            used.add(index)
            node_counts[group] = node_counts.get(group, 0) + 1
            if len(selected) >= regular_budget:
                break
    # Preserve document order after relevance selection. This keeps element indices
    # stable and easy to inspect while still choosing which controls enter the budget.
    result = [a for _, a in sorted(selected, key=lambda pair: pair[0])]
    result.extend(controls)
    return result[:limit], max(0, len(actions) - len(result[:limit]))


def action_space(actions):
    """One index per observed element; each operation has its own valid target choices."""
    elements, indices, targets, controls = [], {}, {}, {}
    operations = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}
    for action in actions:
        kind = action["kind"]
        if kind not in operations:
            controls[action["id"].upper()] = action
            continue
        node = action["node"]
        clean = sanitize_action(action)
        if node not in indices:
            index = str(len(elements) + 1)
            indices[node] = index
            element = {k: clean[k] for k in ("role", "value", "checked", "selected", "expanded") if k in clean}
            element.update(index=index, label=clean["label"].split(" → ")[0], operations=[])
            if kind == "select":
                element["value"] = clean.get("current_value", "")
                element["options"] = []
            elements.append(element)
        index = indices[node]
        operation = operations[kind]
        group = targets.setdefault(operation, {})
        element = elements[int(index) - 1]
        if operation not in element["operations"]:
            element["operations"].append(operation)
        target = index
        if kind == "select":
            # Key the option by its stable DOM option_index, not its position
            # in the filtered list — the currently-selected option is excluded
            # from observations, so positional keys re-map the same target id
            # to a different option every time the value changes.
            option_index = action.get("option_index")
            try:
                target = f"{index}:{int(option_index) + 1}"
            except (TypeError, ValueError):
                target = f"{index}:{len(element['options']) + 1}"
            element["options"].append(
                {
                    "index": target,
                    "label": clean["label"],
                    "value": clean.get("value", ""),
                    "option_index": action.get("option_index"),
                }
            )
        group[target] = action
    return elements, targets, controls


def choose(state, goal, history, backend=None, exploration_policy=None, perception=None):
    if perception is not None:
        # Caller (the agent loop) already reduced this exact page's actions —
        # reuse the same candidates/action space instead of re-sanitizing.
        candidates, omitted_for_model, elements, targets, controls = perception
    else:
        candidates, omitted_for_model = candidate_actions(
            state["actions"], goal, exploration_policy=exploration_policy
        )
        elements, targets, controls = action_space(candidates)
    labels = {
        "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
        "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value from the goal.",
        "SELECT": "Select one observed option from a native single-select dropdown.",
    }
    operations = {key: labels[key] for key in targets}
    operations.update({key: value["label"] for key, value in controls.items()})
    operations.update(DONE="Every requirement is visibly satisfied.", BLOCKED="No supported operation can progress.")
    # The goal leaves the machine only at the routing level's discretion:
    # local scoring uses the raw text, the request body carries the bound copy.
    sent_goal = outbound_text(goal)
    questions = {
        "operation": {
            "type": "choice",
            "criteria": operations,
            "instructions": {"goal": sent_goal, "rules": NEXT_ACTION},
        }
    }
    for operation, candidates_for_operation in targets.items():
        criteria = {}
        for index, a in candidates_for_operation.items():
            clean = sanitize_action(a)
            criteria[index] = {
                "element": f"[{index}] {clean['label']}",
                "current_value": clean.get("current_value", clean.get("value", "")),
                **{k: a[k] for k in ("role", "checked", "selected", "expanded") if k in a},
            }
        questions[operation.lower() + "_target"] = {
            "type": "choice",
            "criteria": criteria,
            "instructions": {"goal": sent_goal, "operation": operation, "rules": [NEXT_ACTION, TARGET]},
        }
    body = {
        "model": os.environ.get("JEV_DECISION_MODEL", os.environ.get("TYPESAFE_MODEL", "jev-latest")),
        "state": {
            "page": sanitize_page(state),
            "elements": elements,
            "candidate_metadata": {
                "visible_actions": len(state["actions"]),
                "offered_actions": len(candidates),
                "omitted_for_model": omitted_for_model + int(state.get("omitted_actions", 0)),
            },
            "recent_actions": [
                {
                    **{k: h.get(k) for k in ("kind", "page_changed", "verification", "approval")},
                    "action": redact_text(h.get("action"), 256),
                    "text": redact_text(h.get("text"), 512),
                }
                for h in history[-10:]
            ],
        },
        "questions": questions,
    }
    started = time.perf_counter()
    decision_backend = backend or SystemOneBackend()
    result = decision_backend.decide(body)
    operation_answer = validate_choice(result["answers"].get("operation", {}), operations)
    operation = operation_answer["choice"]
    target = None
    target_answer = None
    probabilities = {}
    if operation in targets:
        target_answer = validate_choice(result["answers"].get(operation.lower() + "_target", {}), targets[operation])
        target = target_answer["choice"]
        choice = targets[operation][target]["id"]
        probabilities = {
            a["id"]: target_answer["probabilities"][index] for index, a in targets[operation].items()
        }
    else:
        choice = controls[operation]["id"] if operation in controls else operation
        probabilities[choice] = operation_answer["probabilities"][operation]
    return {
        "choice": choice,
        "operation": operation,
        "target": target,
        "confidence": operation_answer["confidence"],
        "probabilities": probabilities,
        "operation_probabilities": operation_answer["probabilities"],
        "target_probabilities": target_answer["probabilities"] if target_answer else {},
        "target_confidence": target_answer["confidence"] if target_answer else None,
        "raw_answers": result["answers"],
        "model": result.get("model", body["model"]),
        "usage": result.get("usage", {}),
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "request": body,
        "candidate_count": len(candidates),
        "omitted_candidates": omitted_for_model + int(state.get("omitted_actions", 0)),
        "exploration_policy": (
            {
                "name": exploration_policy.name,
                "version": exploration_policy.version,
                "digest": exploration_policy.digest,
            }
            if exploration_policy is not None else None
        ),
    }


def field_context(goal, action, page, history):
    clean = sanitize_action(action)
    return {
        "goal": outbound_text(goal),
        "field": {k: clean.get(k) for k in ("label", "role", "value")},
        "page": sanitize_page(page),
        "recent_actions": [
            {"action": redact_text(h.get("action"), 256), "text": redact_text(h.get("text"), 512)}
            for h in history[-6:]
        ],
    }


def field_text(context):
    base = os.environ.get("TEXT_MODEL_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    assert_endpoint_allowed(base)
    assert_transport_secure(base)
    key = os.environ.get("TEXT_MODEL_API_KEY")
    if not key and not loopback_endpoint(base):
        raise ValueError(
            "Remote TYPE_TEXT backend needs TEXT_MODEL_API_KEY; no text is hardcoded or guessed by the executor."
        )
    model = os.environ.get("TEXT_MODEL", "deepseek-chat")
    reasoning = {"thinking": {"type": "disabled"}} if "api.deepseek.com/" in base else {"reasoning": {"effort": "low"}}
    if os.environ.get("TEXT_MODEL_REASONING") == "none":
        # Omit the field entirely — "none" means send no reasoning hint.
        # Strict OpenAI-compatible servers (e.g. Ollama /v1) reject unknown
        # reasoning keys outright, so a disabled-looking object still fails.
        reasoning = {}
    started = time.perf_counter()
    # Small local models occasionally emit malformed JSON or the {"text": null}
    # escape hatch; one retry rescues a flaky generation without ever typing
    # an invalid value. Transport failures propagate — post_json has its own
    # retry policy for those.
    result, usage = None, {}
    for _ in range(2):
        result = post_json(
            base + "/chat/completions",
            key,
            {
                "model": model,
                "max_tokens": 1024,
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
                **reasoning,
                "messages": [
                    {"role": "system", "content": TEXT_VALUE},
                    {"role": "user", "content": json.dumps(context)},
                ],
            },
        )
        for k, v in (result.get("usage") or {}).items():
            if isinstance(v, (int, float)):
                usage[k] = usage.get(k, 0) + v
        try:
            output = json.loads(result["choices"][0]["message"]["content"])
            value = output["text"]
            if set(output) != {"text"} or not isinstance(value, str) or not value.strip() or len(value) > 2000:
                raise ValueError()
            break
        except (ValueError, KeyError, TypeError):
            result = None
    if result is None:
        raise ValueError("Text helper returned no valid field value; nothing typed.")
    return value, {
        "model": model,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "usage": usage,
    }
