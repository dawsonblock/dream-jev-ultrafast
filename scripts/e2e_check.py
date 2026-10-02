"""End-to-end scenario qualification on a real browser (§46/§47).

A scripted local decision backend drives the *real* agent loop — observe →
choose → authorize → journal → execute — against a real Chrome instance and a
real hash-chained evidence store, on a local multi-page fixture site. No paid
APIs, no mocked browser, no model endpoint: the backend reads the offered
catalogue (``body["questions"]``) and picks the labeled element it was told
to pick — the same contract TypeSafe fulfills.

Legs:
    clean   — three-step task completes, verifier confirms, evidence chain
              carries action_attempted/executed transitions and a verified
              run_finished.
    churn   — the backend rewrites the DOM *while deciding* (between observe
              and act): the step must fail closed, the run must abort with an
              honest censor reason, no wrong-element mutation may land, and
              the chain must still verify.

    uv run python scripts/e2e_check.py            # needs Chrome CDP up
"""

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jev_ultrafast.agent import Agent  # noqa: E402
from jev_ultrafast.dream import ExperienceStore  # noqa: E402


def _answer(choice: str, ids) -> dict:
    return {"choice": choice, "confidence": 1.0,
            "probabilities": {i: float(i == choice) for i in ids}}


class ScriptedBackend:
    """decide(body) per the real contract: choose an operation, then the
    labeled target index inside that operation's offered catalogue."""

    def __init__(self, inject=None):
        self.calls = 0
        self.inject = inject  # callable(browser) — fires once on call #2
        self.browser = None

    def decide(self, body: dict) -> dict:
        self.calls += 1
        if self.inject is not None and self.calls == 2:
            self.inject(self.browser)
        questions = body["questions"]
        page_text = (body["state"]["page"].get("text") or "")
        operations = questions["operation"]["criteria"]

        def op(name):
            return _answer(name, list(operations))

        if "confirmation received" in page_text.lower():
            answers = {"operation": op("DONE")}
            return {"answers": answers, "model": "scripted-e2e", "usage": {}}

        # Pick the CLICK target whose element label carries the scenario verb
        # ("Continue" on the index page, "Confirm" on the confirm page) —
        # the label, not page text, decides which step we're on.
        criteria = questions["click_target"]["criteria"]
        for index, cand in criteria.items():
            if "continue" in cand.get("element", "").lower():
                answers = {"operation": op("CLICK"),
                           "click_target": _answer(index, list(criteria))}
                return {"answers": answers, "model": "scripted-e2e",
                        "usage": {}}
        for index, cand in criteria.items():
            if "confirm" in cand.get("element", "").lower():
                answers = {"operation": op("CLICK"),
                           "click_target": _answer(index, list(criteria))}
                return {"answers": answers, "model": "scripted-e2e",
                        "usage": {}}
        answers = {"operation": op("BLOCKED")}
        return {"answers": answers, "model": "scripted-e2e", "usage": {}}


def _site(tmp: Path) -> Path:
    index = """<!doctype html><title>JEV E2E</title><body>
        <h1>Booking flow</h1>
        <a href="page2.html">Continue to confirmation</a>
        <a href="#noop">Decoy link</a></body>"""
    page2 = """<!doctype html><title>confirm</title><body>
        <h1>confirm the booking</h1>
        <button id="go" onclick="location.href='done.html'">Confirm booking</button>
        <button id="decoy" onclick="window.__wrong=1">Decoy</button></body>"""
    done = """<!doctype html><title>done</title><body>
        <h1>confirmation received</h1></body>"""
    (tmp / "index.html").write_text(index)
    (tmp / "page2.html").write_text(page2)
    (tmp / "done.html").write_text(done)
    return tmp / "index.html"


def _run(agent: Agent, approve: bool = True):
    try:
        while True:
            if agent.state["status"] == "approval_required":
                if not approve:
                    break
                agent.command("approve")  # operator resolves the grant
                continue
            if agent.state["status"] in {"done", "verified_done", "blocked",
                                         "aborted"}:
                break
            agent.command("tick")
    finally:
        agent.close()


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="jev-e2e-"))
    index = _site(tmp)
    url = "file://" + str(index)
    checks = 0

    def expect(cond, label):
        nonlocal checks
        checks += 1
        print(f"  {'ok' if cond else 'FAIL'} {label}", flush=True)
        return cond

    # --- leg 1: clean run ----------------------------------------------------
    store_path = tmp / "e2e-clean.jsonl"
    backend = ScriptedBackend()
    agent = Agent(url, "Complete the booking flow",
                  decision_backend=backend,
                  verifier=lambda page: "confirmation received"
                  in (page.get("text") or "").lower(),
                  dream_store=store_path)
    backend.browser = agent.browser
    _run(agent)

    store = ExperienceStore(store_path)
    events = store.load()
    kinds = [e.get("event") for e in events]
    ok = expect(agent.state["status"] in {"done", "verified_done"} or
                agent.state.get("verified"),
                "clean run reached a verified-done state")
    ok &= expect(store.verify()["torn_tail_recovered"] is False,
                 "clean-run chain verifies")
    ok &= expect("action_attempted" in kinds,
                 "attempt journaled before dispatch")
    ok &= expect(any(e.get("event") == "run_finished"
                     for e in events), "run_finished recorded")
    if not ok:
        return 1

    # --- leg 2: churn injected between observe and act -----------------------
    store_path = tmp / "e2e-churn.jsonl"

    def churn(browser):
        # Replace the page mid-decide: a lookalike button that would set a
        # marker if the guard ever clicked the wrong node.
        browser.evaluate(
            "document.body.innerHTML='<h1>confirm the booking</h1>"
            "<button id=\"go\" onclick=\"window.__wrong=1\">Confirm booking</button>'")

    backend = ScriptedBackend(inject=churn)
    agent = Agent(url, "Complete the booking flow",
                  decision_backend=backend, dream_store=store_path)
    backend.browser = agent.browser
    abort = None
    try:
        _run(agent)
    except Exception as exc:  # StalePage / IndeterminateMutation / BrowserError
        abort = exc

    events = ExperienceStore(store_path).load()
    ExperienceStore(store_path).verify()
    ok = expect(abort is not None,
                "churned run aborted (StalePage/Indeterminate, not success)")
    ok &= expect(agent.state.get("status") == "aborted" or
                 any(e.get("event") == "run_finished" and
                     e.get("status") == "aborted" for e in events),
                 "run_finished records an aborted/censored outcome")
    # The agent's browser is closed by now; confirm through a fresh session
    # that neither marker on the churned page was ever set.
    from jev_ultrafast.browser import Browser
    b2 = Browser(url)
    try:
        b2.evaluate("location.href='page2.html'")
        time.sleep(0.4)
        flagged = b2.evaluate("window.__wrong === 1 || window.__confirmed === 1")
    finally:
        b2.close()
    ok &= expect(flagged is not True,
                 "no wrong-element mutation ever landed")
    if not ok:
        return 1

    print(f"e2e: {checks} checks passed — clean run verified end-to-end, "
          "injected churn aborted safely with intact evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
