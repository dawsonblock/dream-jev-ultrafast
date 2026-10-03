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


class IndeterminateMutation(RuntimeError):
    """A mutation was dispatched but its outcome cannot be confirmed.

    Raised only when execution may already have crossed the mutation point —
    e.g. ``e.click()`` fired a ``click`` handler that synchronously navigated
    and destroyed the evaluation context before the result returned. This is
    the third outcome beside "provably not executed" (``StalePage`` — safe to
    re-perceive and retry) and confirmed success: the caller must treat it as
    a terminal, non-retryable state, because retrying could double-apply the
    mutation. It deliberately does not subclass ``StalePage``, so the
    recoverable re-perception path cannot swallow it.
    """


class BrowserError(RuntimeError):
    """The browser connection or CDP session itself failed.

    Distinct from ``StalePage`` (a recoverable re-perception) and from page
    script errors: a run that dies here never reached a measured outcome, and
    the censoring taxonomy records it as ``browser_crash`` rather than
    pretending it was a task failure.
    """


class Browser:
    def __init__(self, url):
        self.target = None
        self.session = None
        self.world_context = None
        self.world_frame = None
        # Owned tab lineage: the created target plus every popup whose opener
        # chain leads back to it (transitively). observe() adopts the newest
        # owned tab — the page a click produced is the page the user now sees —
        # and falls back to the surviving opener when a popup closes. act()
        # never syncs: a decision always executes on the tab it was made on.
        self.tab_order = []
        try:
            ensure_daemon()
            target = cdp("Target.createTarget", url="about:blank", background=True)["targetId"]
            self.tab_order = [target]
            self._attach(target)
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

    def _attach(self, target):
        self.target = target
        self.session = cdp("Target.attachToTarget", targetId=target, flatten=True)["sessionId"]
        self.call("Emulation.setDeviceMetricsOverride", width=1120, height=780, deviceScaleFactor=1, mobile=False)
        self.call("Emulation.setFocusEmulationEnabled", enabled=True)
        self.world_context = None
        self.world_frame = None

    def _sync_targets(self):
        """Adopt the newest owned tab, or fall back to a surviving opener."""
        infos = (cdp("Target.getTargets") or {}).get("targetInfos") or []
        pages = {i["targetId"]: i for i in infos if i.get("type") == "page" and i.get("targetId")}
        owned = set(self.tab_order)
        order = [t for t in self.tab_order if t in pages]
        # Popups claim ownership transitively: a popup opened by our popup is
        # still part of this session's tab cluster.
        changed = True
        while changed:
            changed = False
            for info in infos:
                tid = info.get("targetId")
                if tid in pages and tid not in owned and info.get("openerId") in owned:
                    owned.add(tid)
                    order.append(tid)
                    changed = True
        self.tab_order = order
        if not order:
            raise BrowserError("Every owned tab is closed")
        if order[-1] != self.target:
            self._attach(order[-1])

    def call(self, method, **params):
        if not self.session:
            raise BrowserError("Browser session is closed")
        try:
            return cdp(method, session_id=self.session, **params)
        except (RuntimeError, OSError, ConnectionError) as exc:
            # Normalize transport-level failures to BrowserError so the run's
            # termination reason is honest (browser_crash, not agent_exception)
            # while every existing RuntimeError handler keeps working.
            raise BrowserError(str(exc)) from exc

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
        # Tab ownership lives at the observe boundary only: a click that opened
        # a popup means the *next* perception must describe the page the user
        # now sees, while a pending decision always executes on the tab it was
        # made on.
        self._sync_targets()
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
                        const scope=(field&&field.getRootNode&&field.getRootNode())||document;
                        const byId=id=>scope.getElementById?scope.getElementById(id):null;
                        const roots=ids.length ? ids.map(byId).filter(Boolean) : [scope];
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
            if action is not None and type(action.get("node")) is int and action.get("kind") != "wait":
                node = action.get("node")
                current = self._isolated(
                    "(() => { const c=globalThis.__jevFastV2; "
                    f"return c ? [c.pageKey(),c.guard(c.nodes.get({node}))] : null; }})()"
                )
                return current == [page["page_key"], page["guards"].get(str(node))]
            return self._isolated(MARKER) == page["marker"]
        except StalePage:
            return False

    def act(self, action, page, text=None, guarantee=None, file_path=None):
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
                "guarantee": guarantee or "atomic",
                "file_path": file_path,
            }
        )
        self.after_input = action if action["kind"] != "wait" else None
        return result

    def close(self):
        # Best-effort teardown: a dead daemon must not mask the error that
        # triggered cleanup or leak through __exit__/__init__ failure paths.
        # Every owned tab closes — a popup we adopted is still ours.
        targets = list(dict.fromkeys(getattr(self, "tab_order", None) or ([self.target] if self.target else [])))
        for target in targets:
            try:
                cdp("Target.closeTarget", targetId=target)
            except Exception:
                pass
        self.target = None
        self.session = None
        self.world_context = None
        self.world_frame = None
        self.tab_order = []


def fingerprint(state):
    content = {k: state[k] for k in ("url", "text", "actions", "scroll")}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def browser_operation(request):
    operation = request["operation"]
    session = request["session"]
    context_id = request.get("context_id")

    def call(method, **params):
        return cdp(method, session_id=session, **params)

    def evaluate(expression, *, mutating=False):
        params = {"expression": expression, "returnByValue": True}
        if context_id is not None:
            params["contextId"] = context_id
        try:
            result = call("Runtime.evaluate", **params)
        except RuntimeError as exc:
            # A mutating evaluation that never returned its acknowledgement is
            # NOT a StalePage: the page may already have executed the mutation
            # and then navigated, destroying the context on the way back. The
            # outcome is indeterminate — retrying could double-apply it.
            if operation == "act" and request.get("action", {}).get("kind") == "select":
                raise IndeterminateMutation(
                    "Dropdown execution was interrupted; inspect before retrying."
                ) from exc
            if mutating:
                raise IndeterminateMutation(
                    "Mutation evaluation was interrupted after the mutation may "
                    "already have run; the outcome is unknown and must not be "
                    "retried. Re-observe and reconcile the page state."
                ) from exc
            message = str(exc).lower()
            if context_id is not None and any(word in message for word in ("context", "destroyed", "navigation")):
                raise StalePage("Document changed during evaluation") from exc
            raise
        if result.get("exceptionDetails"):
            if operation == "act" and request["action"]["kind"] == "select":
                raise IndeterminateMutation("Dropdown execution was interrupted; inspect before retrying.")
            details = result["exceptionDetails"]
            reason = (details.get("exception") or {}).get("description") or details.get("text") or "unknown"
            if mutating:
                raise IndeterminateMutation(
                    "Mutation evaluation was interrupted after the mutation may "
                    f"already have run ({reason.splitlines()[0][:200]}); the "
                    "outcome is unknown and must not be retried."
                )
            raise StalePage(f"Document changed during evaluation: {reason.splitlines()[0][:200]}")
        return result.get("result", {}).get("value")

    # Keyboard-scroll keys: a bounded whitelist executed through real CDP key
    # events. Anything outside this set is not a scroll — and never offered.
    KEY_VK = {"PageDown": 34, "PageUp": 33, "End": 35, "Home": 36}

    if operation == "act":
        action = request["action"]
        kind = action["kind"]
        if kind == "scroll" and action.get("node") is None:
            call("Input.dispatchMouseEvent", type="mouseWheel", x=550, y=650, deltaX=0, deltaY=action["delta"])
        elif kind == "key":
            key = action.get("key")
            if key not in KEY_VK:
                raise ValueError("Unsupported key action")
            # A scroll key landing on editable focus moves the caret or changes
            # a value — an edit, not a scroll. Focus can live inside a frame's
            # document, so descend the activeElement chain through every
            # same-origin frame before dispatching.
            editable_focus = evaluate(
                "(()=>{let d=document,a=d.activeElement,depth=0;"
                "while(a&&(a.tagName==='IFRAME'||a.tagName==='FRAME')&&depth++<8){"
                "try{d=a.contentDocument;if(!d)break;a=d.activeElement;}catch(_){break;}}"
                "return !!(a&&(a.isContentEditable||/^(INPUT|TEXTAREA|SELECT)$/.test(a.tagName)));})()"
            )
            if editable_focus:
                raise StalePage(
                    "An editable element holds keyboard focus; a scroll key "
                    "would edit, not scroll. Observe again."
                )
            vk = KEY_VK[key]
            call("Input.dispatchKeyEvent", type="rawKeyDown", key=key, code=key,
                 windowsVirtualKeyCode=vk, nativeVirtualKeyCode=vk)
            call("Input.dispatchKeyEvent", type="keyUp", key=key, code=key,
                 windowsVirtualKeyCode=vk, nativeVirtualKeyCode=vk)
        elif kind == "upload":
            if type(action.get("node")) is not int:
                raise ValueError("Invalid observed node")
            expected = request.get("expected")
            if not expected or expected.get("guard") is None:
                raise StalePage("Missing execution guard. Observe again.")
            file_path = request.get("file_path")
            if not file_path:
                raise ValueError("upload requires an operator-resolved file path")
            # Upload is non-transactional by construction: input.files can only
            # be set through DOM.setFileInputFiles (page JS has no in-world
            # setter), so the guard runs in-world immediately before the
            # browser-level mutation — the same trust class as trusted input,
            # under whichever guarantee was configured.
            probe = (
                """((input) => {
                  const {action,expected}=input, c=globalThis.__jevFastV2;
                  const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
                  if (!c || !same(c.pageKey(),expected.page_key)) return null;
                  const e=c.nodes.get(action.node);
                  if (!e?.isConnected || !same(c.guard(e),expected.guard)) return null;
                  if (e.tagName!=='INPUT' || String(e.type).toLowerCase()!=='file' ||
                      e.matches(':disabled') || e.closest('[aria-disabled="true"],[inert]')) return null;
                  return e;
                })(""" + json.dumps({"action": action, "expected": expected}) + ")"
            )
            try:
                response = call("Runtime.evaluate", expression=probe,
                                contextId=context_id, returnByValue=False)
            except RuntimeError as exc:
                raise StalePage("Upload target validation was interrupted. Observe again.") from exc
            if response.get("exceptionDetails"):
                raise StalePage("Upload target validation failed in-page. Observe again.")
            remote = response.get("result") or {}
            if remote.get("subtype") != "node" or not remote.get("objectId"):
                raise StalePage("Upload target changed or is no longer a file input. Observe again.")
            try:
                call("DOM.setFileInputFiles", files=[file_path], objectId=remote["objectId"])
            except RuntimeError as exc:
                # A lost acknowledgement cannot be distinguished from an
                # applied file list — the upload may already have crossed into
                # the page, so the outcome is indeterminate, never retried.
                raise IndeterminateMutation(
                    "File-upload dispatch was not acknowledged; the outcome is "
                    "unknown and must not be retried."
                ) from exc
        elif kind != "wait":
            if type(action.get("node")) is not int:
                raise ValueError("Invalid observed node")
            expected = request.get("expected")
            if not expected or expected.get("guard") is None:
                raise StalePage("Missing execution guard. Observe again.")
            # Two execution guarantees: "atomic" validates and mutates inside a
            # single isolated-world turn (no IPC gap — the strongest authority
            # bound), while "trusted" dispatches real CDP input with pre-press
            # and pre-release revalidation (needed for isTrusted-gated sites,
            # non-transactional by nature).
            guarantee = request.get("guarantee") or "atomic"
            # Geometry note: elements may live in a same-origin frame document
            # or shadow root. The inner hit-test runs in the element's OWN
            # document (its rect is in that viewport); the offscreen check,
            # outer hit-test, and returned trusted-input coordinates use
            # viewRect's frame-chain translation into TOP viewport space —
            # which is the coordinate space Input.dispatchMouseEvent consumes.
            # For shadow-DOM targets elementFromPoint retargets to the host, so
            # the host counts as the element's own surface.
            guarded = (
                """((input) => {
                  const {action,expected,guarantee,text}=input, c=globalThis.__jevFastV2;
                  const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
                  if (!c || !same(c.pageKey(),expected.page_key)) return {error:'stale-page'};
                  const e=c.nodes.get(action.node);
                  if (!e?.isConnected || !same(c.guard(e),expected.guard)) return {error:'stale-target'};
                  if (e.matches(':disabled') || e.closest('[aria-disabled="true"],[inert]') ||
                      !e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true})) return {error:'unavailable'};
                  if (action.kind==='fill' && (e.readOnly || e.getAttribute('aria-readonly')==='true'))
                    return {error:'readonly'};
                  const doc=e.ownerDocument||document, win=doc.defaultView||window;
                  const root=e.getRootNode&&e.getRootNode(), host=root&&root.host?root.host:null;
                  const r=e.getBoundingClientRect(), rx=r.x+r.width/2, ry=r.y+r.height/2;
                  const v=typeof c.viewRect==='function'?c.viewRect(e):{x:r.x,y:r.y,w:r.width,h:r.height};
                  const x=v.x+v.w/2, y=v.y+v.h/2;
                  if (!r.width || !r.height || x<0 || y<0 || x>=innerWidth || y>=innerHeight)
                    return {error:'offscreen'};
                  const hit=doc.elementFromPoint(rx,ry);
                  if (!hit || !(hit===e || e.contains(hit) || (host && (hit===host || host.contains(hit)))))
                    return {error:'covered'};
                  let frame=null, w=win;
                  while (w && w!==w.parent) { frame=w.frameElement; w=w.parent; }
                  if (frame) {
                    const outer=document.elementFromPoint(x,y);
                    if (!outer || !(outer===frame || frame.contains(outer))) return {error:'covered'};
                  }
                  if (action.kind==='scroll') {
                    if (typeof e.scrollBy!=='function') return {error:'unsupported'};
                    e.scrollBy({top:Number(action.delta)||0,left:0,behavior:'instant'});
                    return {x,y,scrolled:true,scrollTop:e.scrollTop};
                  }
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
                  if (guarantee==='atomic') {
                    if (action.kind==='click') {
                      if (typeof e.click !== 'function') return {error:'unsupported'};
                      e.click();
                      return {x,y,clicked:true};
                    }
                    if (action.kind==='fill') {
                      let active=doc.activeElement;
                      if (!(active && (active===e || e.contains(active)))) {
                        if (typeof e.focus !== 'function') return {error:'unsupported'};
                        e.focus(); active=doc.activeElement;
                      }
                      if (!(active && (active===e || e.contains(active)))) return {error:'focus'};
                      try {
                        if ('value' in e && e.setSelectionRange) e.setSelectionRange(0, String(e.value).length);
                        else { const s=win.getSelection(), rg=doc.createRange();
                               rg.selectNodeContents(e); s.removeAllRanges(); s.addRange(rg); }
                      } catch (_) {}
                      if (!doc.execCommand || !doc.execCommand('insertText', false, text))
                        return {error:'insert'};
                      const v='value' in e ? String(e.value) : String(e.innerText||'');
                      return {x,y,inserted:true,matched:v===text,value:v.slice(0,1024)};
                    }
                  }
                  return {x,y};
                })("""
            )
            payload = {
                "action": action, "expected": expected,
                "guarantee": guarantee, "text": request.get("text") or "",
            }
            # The guarded script mutates only at the end: select writes the
            # option and scroll runs scrollBy unconditionally — in-world
            # execution is their only form — while click/fill mutate under
            # "atomic" inside the same turn. Under "trusted" a click/fill
            # script only validates and reports coordinates — no mutation —
            # so interruption stays a recoverable StalePage there.
            target = evaluate(
                guarded + json.dumps(payload) + ")",
                mutating=kind in ("select", "scroll") or guarantee == "atomic",
            )
            if not isinstance(target, dict) or not target or target.get("error"):
                # Every explicit {error: ...} return is provably pre-mutation
                # and safe to retry after re-observing. A *missing or shapeless*
                # result is different: on a mutating call nothing distinguishes
                # "validated and returned nothing" from "mutated, then the
                # result was lost" — so it is indeterminate, not stale. On a
                # validation-only (trusted) call the script provably never
                # mutated, so a missing result is just a failed check.
                if not isinstance(target, dict) or not target.get("error"):
                    if kind in ("select", "scroll") or guarantee == "atomic":
                        raise IndeterminateMutation(
                            "Dropdown execution returned no result; inspect before retrying."
                            if kind == "select"
                            else "Mutation evaluation returned no result; the outcome is "
                                 "unknown and must not be retried."
                        )
                    raise StalePage("Target validation returned no result. Observe again.")
                if guarantee == "atomic" and target.get("error") == "unsupported":
                    # Programmatic control is unavailable; nothing was mutated.
                    # Escalate to the trusted-input path for this element.
                    guarantee = "trusted"
                    payload["guarantee"] = "trusted"
                    target = evaluate(
                        guarded + json.dumps(payload) + ")",
                        mutating=kind == "scroll",
                    )
                    if not isinstance(target, dict) or not target or target.get("error"):
                        raise StalePage("Target changed, became unavailable, or is covered. Observe again.")
                else:
                    raise StalePage("Target changed, became unavailable, or is covered. Observe again.")
            elif kind == "scroll":
                # scrollBy already ran inside the guarded turn — an element
                # scroll has no physical-input form, so the in-world scroll IS
                # the execution under either guarantee, the same trust class as
                # upload. It must never reach the trusted click dispatch below:
                # an OBSERVE-classified scroll is not a press+release.
                if target.get("scrolled"):
                    return {"executed": action["id"]}
                raise StalePage("Scroll execution returned no result. Observe again.")
            elif kind != "select" and guarantee == "atomic":
                if target.get("clicked"):
                    return {"executed": action["id"]}
                if target.get("inserted"):
                    # The insert already ran — a mismatch means the mutation
                    # did not land as authorized. This is not retryable.
                    if not target.get("matched"):
                        raise RuntimeError(
                            "Text insertion landed differently than authorized; inspect before retrying."
                        )
                    return {"executed": action["id"]}
                raise StalePage("Atomic execution returned no mutation result. Observe again.")
            if kind not in ("select", "scroll") and guarantee == "trusted":
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
                      const doc=e.ownerDocument||document;
                      const root=e.getRootNode&&e.getRootNode(), host=root&&root.host?root.host:null;
                      const r=e.getBoundingClientRect();
                      const v=typeof c.viewRect==='function'?c.viewRect(e):{x:r.x,y:r.y};
                      if (!r.width || !r.height || Math.abs(v.x+r.width/2-x)>2 || Math.abs(v.y+r.height/2-y)>2)
                        return false;
                      // x,y are top-viewport coordinates: hit-test in the top
                      // document against the outermost frame element when the
                      // target lives inside a frame, otherwise directly.
                      let frame=null, w=doc.defaultView;
                      while (w && w!==w.parent) { frame=w.frameElement; w=w.parent; }
                      const hit=document.elementFromPoint(x,y);
                      if (frame) return !!(hit && (hit===frame || frame.contains(hit)));
                      return !!(hit && (hit===e || e.contains(hit) || (host && (hit===host || host.contains(hit)))));
                    })(""" + json.dumps({"action": action, "expected": expected, "x": x, "y": y}) + ")"
                )
                if not evaluate(precheck):
                    raise StalePage("Target failed pre-press identity check. Observe again.")
                try:
                    call("Input.dispatchMouseEvent", type="mousePressed", x=x, y=y, button="left", clickCount=1)
                except Exception as exc:
                    # The dispatch may already have reached the browser before
                    # the acknowledgement was lost — mousedown is itself a
                    # mutation point, so a lost answer is indeterminate, and a
                    # release is still attempted for pointer hygiene.
                    try:
                        call("Input.dispatchMouseEvent", type="mouseReleased", x=x, y=y, button="left", clickCount=1)
                    except Exception:
                        pass
                    raise IndeterminateMutation(
                        "Trusted-input press dispatch was not acknowledged; the "
                        "press may already have landed — the outcome is unknown "
                        "and must not be retried."
                    ) from exc
                # The press already left the process — mousedown handlers may
                # have run. From here on nothing can be called "not executed":
                # a failed or interrupted pre-release check, a lost release,
                # or a destroyed context all leave the click outcome unknown,
                # so every post-press failure is indeterminate. The release is
                # dispatched for pointer hygiene under finally so an exception
                # in the re-check can never strand the mouse button.
                try:
                    recheck = evaluate(precheck) is True
                except Exception:
                    recheck = False
                finally:
                    try:
                        call("Input.dispatchMouseEvent", type="mouseReleased", x=x, y=y, button="left", clickCount=1)
                    except Exception:
                        recheck = False
                if not recheck:
                    raise IndeterminateMutation(
                        "Trusted-input press was dispatched but the click outcome "
                        "could not be confirmed; the mutation is indeterminate and "
                        "must not be retried."
                    )
                if kind == "fill":
                    # Focus verification and text insertion in one evaluation:
                    # Input.insertText cannot be bound to the target, so typing
                    # happens in-world, atomically with the focus check.
                    # execCommand('insertText') may fire input handlers that
                    # navigate before the result returns — mutating=True keeps
                    # an interrupted evaluation indeterminate, never stale.
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
                          const doc=e.ownerDocument||document, win=doc.defaultView||window;
                          let active=doc.activeElement;
                          if (!(active && (active===e || e.contains(active)))) {
                            if (e.focus) e.focus();
                            active=doc.activeElement;
                          }
                          if (!(active && (active===e || e.contains(active)))) return {error:'focus'};
                          try {
                            if ('value' in e && e.setSelectionRange) e.setSelectionRange(0, String(e.value).length);
                            else { const s=win.getSelection(), r=doc.createRange();
                                   r.selectNodeContents(e); s.removeAllRanges(); s.addRange(r); }
                          } catch (_) {}
                          if (!doc.execCommand || !doc.execCommand('insertText', false, text))
                            return {error:'insert'};
                          const value = 'value' in e ? String(e.value) : String(e.innerText || '');
                          return {inserted:true, matched:value===text, value:value.slice(0,1024)};
                        })(""" + json.dumps({"action": action, "expected": expected, "text": request["text"]}) + ")",
                        mutating=True,
                    )
                    if inserted is None:
                        raise IndeterminateMutation(
                            "Text insertion evaluation returned no result; the "
                            "outcome is unknown and must not be retried."
                        )
                    if inserted.get("error"):
                        raise StalePage(f"Text insertion failed before mutation: {inserted.get('error')}")
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
