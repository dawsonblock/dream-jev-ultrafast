"""Local finite-choice decision backend for Jev Ultrafast.

Serves the TypeSafe/SystemOne-compatible finite-choice protocol on loopback and
answers each question with a small local model via Ollama. No API keys, no paid
calls, no external network — the agent's authority plane (policy, approvals,
payload review, verifier) is unchanged; only the decision scoring is local.

Usage:

    ollama serve                          # terminal 1 (or the Ollama app)
    ollama pull qwen3:8b                  # once
    uv run python scripts/local_backend.py

    # agent process:
    export JEV_DECISION_BASE_URL=http://127.0.0.1:9000/v1/systemone
    export JEV_DECISION_MODEL=jev-local
    export JEV_MODEL_TIMEOUT=240          # local inference exceeds the 25s default
    export TEXT_MODEL_BASE_URL=http://127.0.0.1:11434/v1   # TYPE_TEXT helper
    export TEXT_MODEL=qwen3:8b
    export TEXT_MODEL_REASONING=none      # required — Ollama rejects reasoning fields
    uv run python -m jev_ultrafast.demo                    # or scripts/smoke.py

Configuration:

    JEV_LOCAL_PORT   listen port (default 9000)
    OLLAMA_BASE      Ollama server (default http://127.0.0.1:11434)
    OLLAMA_MODEL     model tag (default qwen3:8b)

The protocol: the agent POSTs {model, state, questions} where each question is
{type: "choice", criteria: {key: description}, instructions}; the backend must
return {answers: {<question>: {choice, probabilities, confidence}}}. The agent
only reads "operation" plus the selected operation's "<op>_target", so this shim
answers exactly those two questions per request — operation first, then the
target question for the operation the model actually chose.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from jev_ultrafast.privacy import tokenize

PORT = int(os.environ.get("JEV_LOCAL_PORT", "9000"))
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
# A 250-candidate catalogue plus page text needs more than Ollama's default
# 4k context window.
NUM_CTX = int(os.environ.get("JEV_LOCAL_NUM_CTX", "16384"))


def ollama_chat(prompt: str, *, num_predict: int = 128) -> tuple[dict, dict]:
    """One JSON-mode chat call. Returns (parsed_json, usage)."""
    request = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat",
        data=json.dumps(
            {
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "format": "json",
                # Hybrid-reasoning models (qwen3) must not burn the token
                # budget on a <think> block — the shim does its own terse CoT.
                "think": False,
                "options": {"temperature": 0.1, "num_ctx": NUM_CTX, "num_predict": num_predict},
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Ollama returned HTTP {error.code}: {error.read()[:200]!r}") from None
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"Ollama unreachable at {OLLAMA_BASE}: {error}") from None
    usage = {
        "prompt_tokens": payload.get("prompt_eval_count", 0),
        "completion_tokens": payload.get("eval_count", 0),
    }
    try:
        return json.loads(payload["message"]["content"]), usage
    except (KeyError, TypeError, json.JSONDecodeError):
        raise RuntimeError(f"Ollama returned non-JSON content: {payload.get('message', {})!r:.300}") from None


def describe(key: str, criterion) -> str:
    if isinstance(criterion, dict):
        element = criterion.get("element", "")
        extras = {
            k: v for k, v in criterion.items() if k != "element" and v not in ("", None)
        }
        suffix = f" {json.dumps(extras, ensure_ascii=False)}" if extras else ""
        return f"{key}: {element}{suffix}"
    return f"{key}: {criterion}"


def ask(state: dict, question: dict, *, context: str, previews: dict | None = None) -> tuple[str, float, dict]:
    """Ask the local model to pick exactly one criterion key.

    Returns (choice, confidence, usage). Raises RuntimeError when the model
    cannot produce a valid answer — the shim never fabricates a choice.
    """
    criteria = question.get("criteria") or {}
    keys = list(criteria)
    instructions = question.get("instructions") or {}
    page = state.get("page") or {}
    recent = state.get("recent_actions") or []
    listing = "\n".join(describe(k, criteria[k]) for k in keys)
    if previews:
        preview_lines = "\n".join(
            f"  {k} can act on: {v}" for k, v in previews.items() if k in criteria
        )
        if preview_lines:
            listing += f"\n\nWhat each option can act on:\n{preview_lines}"
    history = "\n".join(
        f"- {h.get('action', '')} ({h.get('kind', '')})"
        + (f" typed={h.get('text')!r}" if h.get("text") else "")
        + (" [page changed]" if h.get("page_changed") else " [no change]")
        for h in recent[-6:]
    )
    # A control whose current value already matches the goal is satisfied —
    # small models otherwise re-select the same dropdown forever.
    current_values = sorted(
        {
            str(c.get("current_value"))
            for c in criteria.values()
            if isinstance(c, dict) and c.get("current_value") not in (None, "")
        }
    )
    satisfied = (
        f"Current value of this control: {', '.join(current_values)}. "
        "If that already satisfies the goal for this control, say so in reason "
        "and prefer not to re-operate it.\n"
        if current_values
        else ""
    )
    page_text = str(page.get("text", ""))[:2500]
    prompt = (
        "You are the decision head of a bounded browser agent. Answer ONLY with "
        'JSON: {"remaining": ["<goal parts not yet satisfied>"], '
        '"reason": "<one sentence>", "choice": "<key>", '
        '"confidence": <number between 0 and 1>}. List the remaining goal '
        "parts first — the choice must serve one of them.\n\n"
        f"Goal: {instructions.get('goal', '')}\n"
        f"Rules: {instructions.get('rules', '')}\n"
        f"Operation context: {instructions.get('operation', context)}\n"
        f"Page: {page.get('url', '')} — {page.get('title', '')}\n"
        f"Visible page text (truncated): {page_text}\n\n"
        + satisfied
        + (f"Actions already taken:\n{history}\n" if history else "")
        + "Do not repeat an action that already ran unless it clearly remains the next step. "
        "A control whose current value already matches the goal is done — "
        "move on to an unsatisfied part of the goal.\n\n"
        + f"Options:\n{listing}\n"
    )
    last_error = ""
    for attempt in range(2):
        try:
            output, usage = ollama_chat(prompt)
        except RuntimeError as error:
            last_error = str(error)
            continue
        if not isinstance(output, dict):
            last_error = f"model returned non-object JSON: {output!r:.100}"
            continue
        choice = str(output.get("choice", "")).strip()
        if choice in criteria:
            try:
                confidence = float(output.get("confidence", 0.6))
            except (TypeError, ValueError):
                confidence = 0.6
            print(
                f"[{context}] choice={choice!r} conf={confidence:.2f} "
                f"reason={output.get('reason', '')!r}",
                file=sys.stderr,
                flush=True,
            )
            return choice, min(max(confidence, 0.0), 1.0), usage
        last_error = f"model picked {choice!r}, not an option key"
        prompt += (
            "\nYour previous answer was not one of the option keys. Respond ONLY "
            'with {"remaining": ["..."], "reason": "<one sentence>", "choice": '
            '"<exact option key>", "confidence": <0..1>}.'
        )
    raise RuntimeError(f"Local model could not answer the choice question: {last_error}")


def answer(choice: str, confidence: float, criteria: dict) -> dict:
    """Build a validate_choice-compatible answer.

    The local model supplies (choice, confidence); a calibrated per-option
    distribution is not something a 3B chat model produces reliably, so the
    remainder is spread uniformly and labelled by construction — these are
    model *scores*, never sampling propensities.
    """
    keys = list(criteria)
    n = len(keys)
    if n == 1:
        probabilities = {keys[0]: 1.0}
    else:
        top = min(max(confidence, 0.51), 0.99)
        rest = round((1.0 - top) / (n - 1), 6)
        probabilities = {k: rest for k in keys}
        probabilities[choice] = round(top, 6)
    return {"choice": choice, "probabilities": probabilities, "confidence": confidence}


def decide(body: dict) -> dict:
    state = body.get("state") or {}
    questions = body.get("questions") or {}
    operation_q = questions.get("operation")
    if not isinstance(operation_q, dict) or not isinstance(operation_q.get("criteria"), dict):
        raise ValueError("Missing operation choice question")
    # Ground the operation choice: a small model cannot pick CLICK vs
    # TYPE_TEXT from descriptions alone — show the first few elements each
    # operation would actually target, drawn from the sibling questions.
    goal_tokens = set(tokenize(str((operation_q.get("instructions") or {}).get("goal", ""))))
    previews = {}
    for name, question in questions.items():
        if not name.endswith("_target"):
            continue
        operation = name[: -len("_target")].upper()
        criteria = question.get("criteria") or {}
        labels = []
        current = None
        for k, c in list(criteria.items())[:6]:
            if isinstance(c, dict):
                current = current or c.get("current_value")
                labels.append(str(c.get("element") or c.get("label") or k)[:60])
            else:
                labels.append(str(c)[:60])
        if labels:
            suffix = ""
            if current:
                suffix = f" (currently set to {current!r}"
                if set(tokenize(str(current))) & goal_tokens:
                    suffix += " — already satisfies the goal"
                suffix += ")"
            previews[operation] = ", ".join(labels) + suffix
    operation, op_confidence, usage = ask(
        state, operation_q, context="operation", previews=previews
    )
    answers = {"operation": answer(operation, op_confidence, operation_q["criteria"])}
    target_q = questions.get(f"{operation.lower()}_target")
    if isinstance(target_q, dict) and isinstance(target_q.get("criteria"), dict):
        target, t_confidence, t_usage = ask(state, target_q, context=operation)
        answers[f"{operation.lower()}_target"] = answer(target, t_confidence, target_q["criteria"])
        usage = {
            "prompt_tokens": usage["prompt_tokens"] + t_usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"] + t_usage["completion_tokens"],
        }
    return {"answers": answers, "model": OLLAMA_MODEL, "usage": usage}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            # Decision bodies carry the candidate catalogue (~256 KB ceiling
            # upstream) — bound the read, never block on a negative length.
            if not 0 < length <= 4 * 1024 * 1024:
                raise ValueError("Invalid request size")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Request body must be a JSON object")
            self._send(200, decide(body))
        except (ValueError, RuntimeError) as error:
            self._send(502, {"error": str(error)})
        except Exception as error:  # malformed input must not wedge the agent loop
            self._send(500, {"error": f"local backend failure: {error}"})

    def _send(self, status: int, payload: dict):
        content = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, *_args):
        pass


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(
        f"Jev local decision backend: http://127.0.0.1:{PORT}/v1/systemone "
        f"→ {OLLAMA_BASE} ({OLLAMA_MODEL})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    sys.exit(main())
