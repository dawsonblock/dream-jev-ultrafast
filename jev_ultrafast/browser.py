"""Observed actions through Browser Harness with isolated-world execution state."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from browser_harness.admin import ensure_daemon
from browser_harness.helpers import cdp

READ_STATE = Path(__file__).with_name("snapshot.js").read_text()
MARKER = f"(() => {{ const state={READ_STATE}; return state?.marker ?? null; }})()"
WORLD_NAME = "jev-ultrafast-v2"


class StalePage(ValueError):
    """A decision no longer refers to the observed page."""


class Browser:
    def __init__(self, url):
        self.target = None
        self.session = None
        self.world_context = None
        self.world_frame = None
        try:
            ensure_daemon()
            self.target = cdp("Target.createTarget", url="about:blank", background=True)["targetId"]
            self.session = cdp("Target.attachToTarget", targetId=self.target, flatten=True)["sessionId"]
            self.call("Emulation.setDeviceMetricsOverride", width=1120, height=780, deviceScaleFactor=1, mobile=False)
            self.call("Emulation.setFocusEmulationEnabled", enabled=True)
            self.call("Page.navigate", url=url)
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if self.evaluate("document.readyState") == "complete":
                    break
                time.sleep(0.02)
            self._ensure_world()
        except Exception:
            self.close()
            raise

    def call(self, method, **params):
        if not self.session:
            raise RuntimeError("Browser session is closed")
        return cdp(method, session_id=self.session, **params)

    def _runtime_evaluate(self, expression, *, context_id=None, await_promise=False):
        params = {"expression": expression, "returnByValue": True}
        if context_id is not None:
            params["contextId"] = context_id
        if await_promise:
            params["awaitPromise"] = True
        try:
            response = self.call("Runtime.evaluate", **params)
        except RuntimeError as exc:
            message = str(exc).lower()
            if context_id is not None and any(word in message for word in ("context", "destroyed", "navigation")):
                self.world_context = None
                raise StalePage("Isolated execution context changed") from exc
            raise
        if response.get("exceptionDetails"):
            if context_id is not None:
                self.world_context = None
            details = response["exceptionDetails"]
            reason = (details.get("exception") or {}).get("description") or details.get("text") or "unknown"
            raise StalePage(f"Document changed during evaluation: {reason.splitlines()[0][:200]}")
        return response.get("result", {}).get("value")

    def evaluate(self, expression):
        """Evaluate in the page's main world. Intended for callers/tests, not Jev internals."""
        return self._runtime_evaluate(expression)

    def _main_frame_id(self):
        return self.call("Page.getFrameTree")["frameTree"]["frame"]["id"]

    def _ensure_world(self):
        if self.world_frame is None:
            self.world_frame = self._main_frame_id()
        if self.world_context is None:
            try:
                result = self.call("Page.createIsolatedWorld", frameId=self.world_frame, worldName=WORLD_NAME)
            except RuntimeError:
                # Main-frame IDs normally survive navigation. Refresh only if Chrome
                # reports that the remembered frame can no longer host the world.
                self.world_frame = self._main_frame_id()
                result = self.call("Page.createIsolatedWorld", frameId=self.world_frame, worldName=WORLD_NAME)
            self.world_context = result["executionContextId"]
        return self.world_context

    def _isolated(self, expression, *, await_promise=False):
        return self._runtime_evaluate(
            expression,
            context_id=self._ensure_world(),
            await_promise=await_promise,
        )

    def observe(self, screenshot=True):
        if getattr(self, "after_input", None):
            action, self.after_input = self.after_input, None
            try:
                self._isolated(
                    """(action => new Promise(resolve => {
                      const field=globalThis.__jevFastV2?.nodes.get(action.node);
                      const autocomplete=action.kind==='fill' && field?.getAttribute('role')==='combobox';
                      let frames=0, stopped=false;
                      const finish=()=>{stopped=true;resolve()};
                      setTimeout(finish,autocomplete ? 200 : 50);
                      const ready=()=>{
                        if (stopped) return;
                        const ids=(field?.getAttribute('aria-controls')||field?.getAttribute('aria-owns')||'')
                          .split(/\\s+/).filter(Boolean);
                        const roots=ids.length ? ids.map(id=>document.getElementById(id)).filter(Boolean) : [document];
                        const options=roots.flatMap(root=>[...root.querySelectorAll('[role="option"]')]);
                        if (++frames>=2 && (!autocomplete || options.some(e=>{
                          const r=e.getBoundingClientRect();
                          return r.width && r.height && r.bottom>0 && r.top<innerHeight &&
                            e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
                        }))) finish();
                        else requestAnimationFrame(ready);
                      };
                      requestAnimationFrame(ready);
                    }))(""" + json.dumps(action) + ")",
                    await_promise=True,
                )
            except (RuntimeError, StalePage):
                pass
        for attempt in range(10):
            try:
                context_id = self._ensure_world()
                return browser_operation(
                    {
                        "operation": "observe",
                        "session": self.session,
                        "context_id": context_id,
                        "screenshot": screenshot,
                    }
                )
            except StalePage:
                self.world_context = None
                if attempt == 9:
                    raise
                time.sleep(0.02)
        raise StalePage("Page did not settle")

    def fresh(self, page, action=None):
        try:
            if action is not None and action.get("kind") in {"click", "select", "fill"}:
                node = action.get("node")
                if type(node) is not int:
                    return False
                current = self._isolated(
                    "(() => { const c=globalThis.__jevFastV2; "
                    f"return c ? [c.pageKey(),c.guard(c.nodes.get({node}))] : null; }})()"
                )
                return current == [page["page_key"], page["guards"].get(str(node))]
            return self._isolated(MARKER) == page["marker"]
        except StalePage:
            return False

    def act(self, action, page, text=None):
        if not self.fresh(page, action):
            raise StalePage("Page changed since this decision. Observe again.")
        if action["kind"] == "wait":
            time.sleep(0.1)
        expected = None
        if type(action.get("node")) is int:
            expected = {
                "page_key": page["page_key"],
                "guard": page["guards"].get(str(action["node"])),
            }
        result = browser_operation(
            {
                "operation": "act",
                "session": self.session,
                "context_id": self._ensure_world(),
                "action": action,
                "expected": expected,
                "text": text,
            }
        )
        self.after_input = action if action["kind"] != "wait" else None
        return result

    def close(self):
        # Best-effort teardown: a dead daemon must not mask the error that
        # triggered cleanup or leak through __exit__/__init__ failure paths.
        if self.target:
            try:
                cdp("Target.closeTarget", targetId=self.target)
            except Exception:
                pass
            finally:
                self.target = None
                self.session = None
                self.world_context = None
                self.world_frame = None


def fingerprint(state):
    content = {k: state[k] for k in ("url", "text", "actions", "scroll")}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def browser_operation(request):
    operation = request["operation"]
    session = request["session"]
    context_id = request.get("context_id")

    def call(method, **params):
        return cdp(method, session_id=session, **params)

    def evaluate(expression):
        params = {"expression": expression, "returnByValue": True}
        if context_id is not None:
            params["contextId"] = context_id
        try:
            result = call("Runtime.evaluate", **params)
        except RuntimeError as exc:
            if operation == "act" and request.get("action", {}).get("kind") == "select":
                raise RuntimeError("Dropdown execution was interrupted; inspect before retrying.") from exc
            message = str(exc).lower()
            if context_id is not None and any(word in message for word in ("context", "destroyed", "navigation")):
                raise StalePage("Document changed during evaluation") from exc
            raise
        if result.get("exceptionDetails"):
            if operation == "act" and request["action"]["kind"] == "select":
                raise RuntimeError("Dropdown execution was interrupted; inspect before retrying.")
            details = result["exceptionDetails"]
            reason = (details.get("exception") or {}).get("description") or details.get("text") or "unknown"
            raise StalePage(f"Document changed during evaluation: {reason.splitlines()[0][:200]}")
        return result.get("result", {}).get("value")

    if operation == "act":
        action = request["action"]
        kind = action["kind"]
        if kind == "scroll":
            call("Input.dispatchMouseEvent", type="mouseWheel", x=550, y=650, deltaX=0, deltaY=action["delta"])
        elif kind != "wait":
            if type(action.get("node")) is not int:
                raise ValueError("Invalid observed node")
            expected = request.get("expected")
            if not expected or expected.get("guard") is None:
                raise StalePage("Missing execution guard. Observe again.")
            target = evaluate(
                """((input) => {
                  const {action,expected}=input, c=globalThis.__jevFastV2;
                  const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
                  if (!c || !same(c.pageKey(),expected.page_key)) return {error:'stale-page'};
                  const e=c.nodes.get(action.node);
                  if (!e?.isConnected || !same(c.guard(e),expected.guard)) return {error:'stale-target'};
                  if (e.matches(':disabled') || e.closest('[aria-disabled="true"],[inert]') ||
                      !e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true})) return {error:'unavailable'};
                  if (action.kind==='fill' && (e.readOnly || e.getAttribute('aria-readonly')==='true'))
                    return {error:'readonly'};
                  const r=e.getBoundingClientRect(), x=r.x+r.width/2, y=r.y+r.height/2;
                  if (!r.width || !r.height || x<0 || y<0 || x>=innerWidth || y>=innerHeight)
                    return {error:'offscreen'};
                  const hit=document.elementFromPoint(x,y);
                  if (!hit || !(hit===e || e.contains(hit))) return {error:'covered'};
                  if (action.kind==='select') {
                    if (e.tagName!=='SELECT' || e.multiple || !Number.isInteger(action.option_index))
                      return {error:'unsupported-select'};
                    const o=e.options[action.option_index];
                    if (!o || o.disabled || o.closest('optgroup[disabled]') ||
                        String(o.value).replace(/\\s+/g,' ').trim().slice(0,512)!==String(action.value) ||
                        String(o.label).replace(/\\s+/g,' ').trim().slice(0,256)!==String(action.option_label))
                      return {error:'stale-option'};
                    e.selectedIndex=action.option_index;
                    e.dispatchEvent(new Event('input',{bubbles:true}));
                    e.dispatchEvent(new Event('change',{bubbles:true}));
                    return {x,y,selected_index:e.selectedIndex};
                  }
                  return {x,y};
                })(""" + json.dumps({"action": action, "expected": expected}) + ")"
            )
            if not target or target.get("error"):
                # The guarded script only mutates in its final statement, so every
                # explicit {error: ...} return is provably pre-mutation and safe to
                # retry after re-observing. A missing result or an interrupted
                # evaluation (handled in evaluate()) remains non-retryable.
                if kind == "select" and target is None:
                    raise RuntimeError("Dropdown execution returned no result; inspect before retrying.")
                raise StalePage("Target changed, became unavailable, or is covered. Observe again.")
            if kind != "select":
                x, y = target["x"], target["y"]
                # Authority integrity: the physical input must land on the same
                # semantic target that passed validation. Re-check page identity,
                # element guard, geometry, and hit-test immediately before press
                # and again before release — validation and dispatch are separate
                # CDP calls, so the page could mutate in between.
                precheck = (
                    """((input) => {
                      const {action,expected,x,y}=input, c=globalThis.__jevFastV2;
                      const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
                      if (!c || !same(c.pageKey(),expected.page_key)) return false;
                      const e=c.nodes.get(action.node);
                      if (!e?.isConnected || !same(c.guard(e),expected.guard)) return false;
                      if (e.matches(':disabled') || e.closest('[aria-disabled="true"],[inert]') ||
                          !e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true})) return false;
                      const r=e.getBoundingClientRect();
                      if (!r.width || !r.height || Math.abs(r.x+r.width/2-x)>2 || Math.abs(r.y+r.height/2-y)>2)
                        return false;
                      const hit=document.elementFromPoint(x,y);
                      return !!(hit && (hit===e || e.contains(hit)));
                    })(""" + json.dumps({"action": action, "expected": expected, "x": x, "y": y}) + ")"
                )
                if not evaluate(precheck):
                    raise StalePage("Target failed pre-press identity check. Observe again.")
                call("Input.dispatchMouseEvent", type="mousePressed", x=x, y=y, button="left", clickCount=1)
                if not evaluate(precheck):
                    # A press already left the process; restore pointer state at
                    # the same point, then fail closed to re-perception.
                    call("Input.dispatchMouseEvent", type="mouseReleased", x=x, y=y, button="left", clickCount=1)
                    raise StalePage("Target changed between press and release; click aborted.")
                call("Input.dispatchMouseEvent", type="mouseReleased", x=x, y=y, button="left", clickCount=1)
                if kind == "fill":
                    # Focus verification and text insertion in one evaluation:
                    # Input.insertText cannot be bound to the target, so typing
                    # happens in-world, atomically with the focus check.
                    inserted = evaluate(
                        """((input) => {
                          const {action,expected,text}=input, c=globalThis.__jevFastV2;
                          const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
                          if (!c || !same(c.pageKey(),expected.page_key)) return {error:'stale-page'};
                          const e=c.nodes.get(action.node);
                          if (!e?.isConnected || !same(c.guard(e),expected.guard)) return {error:'stale-target'};
                          if (e.readOnly || e.getAttribute('aria-readonly')==='true' ||
                              e.matches(':disabled') || e.closest('[aria-disabled="true"],[inert]'))
                            return {error:'readonly'};
                          let active=document.activeElement;
                          if (!(active && (active===e || e.contains(active)))) {
                            if (e.focus) e.focus();
                            active=document.activeElement;
                          }
                          if (!(active && (active===e || e.contains(active)))) return {error:'focus'};
                          try {
                            if ('value' in e && e.setSelectionRange) e.setSelectionRange(0, String(e.value).length);
                            else { const s=getSelection(), r=document.createRange();
                                   r.selectNodeContents(e); s.removeAllRanges(); s.addRange(r); }
                          } catch (_) {}
                          if (!document.execCommand || !document.execCommand('insertText', false, text))
                            return {error:'insert'};
                          const value = 'value' in e ? String(e.value) : String(e.innerText || '');
                          return {inserted:true, matched:value===text, value:value.slice(0,1024)};
                        })(""" + json.dumps({"action": action, "expected": expected, "text": request["text"]}) + ")"
                    )
                    if not inserted or inserted.get("error"):
                        raise StalePage(f"Text insertion failed before mutation: {(inserted or {}).get('error')}")
                    if not inserted.get("matched"):
                        # The insert already ran — a mismatch means the mutation
                        # did not land as authorized. This is not retryable.
                        raise RuntimeError(
                            "Text insertion landed differently than authorized; inspect before retrying."
                        )
        return {"executed": action["id"]}

    info = evaluate(READ_STATE)
    if info is None:
        raise StalePage("Document is navigating")
    info["fingerprint"] = fingerprint(info)
    if request.get("screenshot", True):
        info["screenshot"] = call("Page.captureScreenshot", format="jpeg", quality=72)["data"]
    return info
