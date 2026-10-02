"""High-iteration race probes for browser execution (qualification §12/§44).

Two families, against a real Chrome instance via BU_CDP_URL:

- ``--nav-iterations N``: atomic clicks whose onclick synchronously
  navigates. Every iteration must resolve as executed or indeterminate —
  never a retryable stale state (T-403).
- ``--churn-iterations N``: continuous DOM regeneration between observe
  and act; every click must land on the element the action described or
  refuse pre-mutation — never mutate a different element (T-44x).

    BU_CDP_URL=http://127.0.0.1:9222 \
      uv run python scripts/race_check.py --nav-iterations 1000 --churn-iterations 10000
"""

import argparse
import os
import time

from jev_ultrafast.browser import Browser, IndeterminateMutation, StalePage

_NAV_HTML = (
    '<button id="adv" onclick="window.hits=(window.hits||0)+1;'
    "location.replace('about:blank')\">Continue</button>"
)
_CHURN_HTML = '<div id="box"></div>'
_CHURN_JS = """
window.clicked=null;
window.__churn=setInterval(()=>{
  const b=document.getElementById('box');
  if(!b) return;
  b.innerHTML=[0,1,2,3].map(i=>
    `<button onclick="window.clicked='b'+${i}">B${i}</button>`).join('');
},3)
"""


def _inject(browser, html, extra_js=""):
    """Settle-aware injection: navigation from the previous iteration may
    commit after a successful evaluate and wipe the fixture — inject until
    an observe actually sees it."""
    for _ in range(200):
        try:
            browser.evaluate("document.body.innerHTML=" + repr(html))
            if extra_js:
                browser.evaluate(extra_js)
            page = browser.observe(screenshot=False)
            if page["actions"]:
                return page
        except Exception:
            pass
        time.sleep(0.02)
    raise AssertionError("fixture never survived navigation settle")


def nav_race(browser, iterations):
    executed = indeterminate = stale = 0
    for _ in range(iterations):
        page = _inject(browser, _NAV_HTML)
        action = next(
            (a for a in page["actions"] if a["kind"] == "click"), None
        )
        if action is None:
            continue
        try:
            browser.act(action, page)
            executed += 1
        except IndeterminateMutation:
            indeterminate += 1
        except StalePage:
            stale += 1
    return executed, indeterminate, stale


def churn(browser, iterations):
    correct = rejected = wrong = 0
    # One churn interval for the whole run — installing it per iteration
    # would stack intervals until the DOM re-renders near-continuously.
    _inject(browser, _CHURN_HTML, _CHURN_JS)
    for _ in range(iterations):
        page = browser.observe(screenshot=False)
        target = next(
            (a for a in page["actions"]
             if a["kind"] == "click" and a["label"] == "B0"),
            None,
        )
        if target is None:
            continue
        try:
            browser.act(target, page)
        except (StalePage, IndeterminateMutation):
            rejected += 1
            continue
        clicked = browser.evaluate("window.clicked")
        if clicked == "b0":
            correct += 1
        else:
            wrong += 1
        browser.evaluate("window.clicked=null")
    browser.evaluate("clearInterval(window.__churn)")
    return correct, rejected, wrong


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nav-iterations", type=int, default=1000)
    parser.add_argument("--churn-iterations", type=int, default=10000)
    args = parser.parse_args()

    browser = Browser(os.environ.get("BU_CDP_URL", "http://127.0.0.1:9222"))
    browser.evaluate("document.body.innerHTML=''")

    if args.nav_iterations:
        t = time.time()
        executed, indeterminate, stale = nav_race(browser, args.nav_iterations)
        dt = time.time() - t
        ok = stale == 0
        print(f"nav-race: {args.nav_iterations} iterations in {dt:.0f}s — "
              f"{executed} executed, {indeterminate} indeterminate, "
              f"{stale} stale {'PASS' if ok else 'FAIL'}")
        if not ok:
            return 1

    if args.churn_iterations:
        t = time.time()
        correct, rejected, wrong = churn(browser, args.churn_iterations)
        dt = time.time() - t
        ok = wrong == 0 and correct > 0
        print(f"churn: {args.churn_iterations} iterations in {dt:.0f}s — "
              f"{correct} correct-target, {rejected} rejected, "
              f"{wrong} wrong-target {'PASS' if ok else 'FAIL'}")
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
