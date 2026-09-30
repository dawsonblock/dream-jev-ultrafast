"""Local-browser freshness/execution regressions. No model calls or external websites."""

from urllib.parse import quote

from jev_ultrafast.browser import Browser, StalePage

HTML = """<!doctype html><title>Guard checks</title>
<style>body{margin:30px}button{width:180px;height:50px}#outside{position:absolute;top:3000px}</style>
<p id="context">Cart total: $10</p>
<button id="target" onclick="window.clicks=(window.clicks||0)+1">Continue</button>
<label>City<input id="field" value="Zurich"></label>
<label><input id="toggle" type="checkbox">Refundable</label>
<select aria-label="Category"><option>All</option><option>Design</option></select>
<p id="outside">Unrelated offscreen text</p>"""


def main():
    browser = Browser("data:text/html," + quote(HTML))
    passed = []
    try:
        page = browser.observe(screenshot=False)
        action = next(a for a in page["actions"] if a["label"] == "Continue")
        # A same-named main-world object must not replace the isolated-world registry.
        browser.evaluate("window.__jevFastV2={nodes:new Map(),pageKey:()=>['owned'],guard:()=>null}")
        assert browser.fresh(page, action)
        passed.append("main-world registry spoof cannot replace isolated-world state")
        browser.evaluate("document.querySelector('#target').style.transform='translateX(200px)'")
        assert browser.fresh(page), "Movement should use fresh geometry, not another model call"
        browser.act(action, page)
        assert browser.evaluate("window.clicks") == 1
        passed.append("moving target clicked at its current location")

        browser.evaluate("document.querySelector('#outside').textContent='Updated outside the viewport'")
        assert browser.fresh(page)
        passed.append("unrelated offscreen text does not invalidate")

        mutations = {
            "visible context": "document.querySelector('#context').textContent='Cart total: $100'",
            "accessible label": "document.querySelector('#target').setAttribute('aria-label','Delete account')",
            "field property": "document.querySelector('#field').value='London'",
            "checkbox property": "document.querySelector('#toggle').checked=true",
            "disabled target": "document.querySelector('#target').disabled=true",
            "read-only field": "document.querySelector('#field').readOnly=true",
            "hidden target": "document.querySelector('#target').style.display='none'",
            "replaced node": "document.querySelector('#target').outerHTML=document.querySelector('#target').outerHTML",
            "dropdown option": "document.querySelector('select').options[1].text='Coastal'",
        }
        for label, expression in mutations.items():
            browser.evaluate("document.querySelector('#target').style.display='block'; "
                             "document.querySelector('#target').disabled=false")
            page = browser.observe(screenshot=False)
            browser.evaluate(expression)
            assert not browser.fresh(page), label
            passed.append(label + " invalidates")

        browser.evaluate("document.querySelector('#target').disabled=false; "
                         "document.querySelector('#target').style.display='block'")
        page = browser.observe(screenshot=False)
        action = next(a for a in page["actions"] if a["label"] == "Delete account")
        # A textless overlay does not alter the model's semantic state, but must block a click.
        browser.evaluate("const cover=document.createElement('div'); "
                         "cover.style.cssText='position:fixed;inset:0;z-index:9999;background:white'; "
                         "document.body.append(cover)")
        assert browser.fresh(page)
        try:
            browser.act(action, page)
        except (RuntimeError, StalePage):
            pass
        else:
            raise AssertionError("Covered target was clicked")
        assert browser.evaluate("window.clicks") == 1
        passed.append("overlay blocked before input")

        browser.evaluate("document.body.innerHTML=" + repr("""
          <form><p id="price">Total $10</p>
          <button type="button" id="buy">Buy</button>
          <label>Search <input id="query" role="combobox" aria-controls="suggestions"></label>
          <div role="listbox" id="suggestions"></div>
          <label><input id="check" type="checkbox">Enabled</label>
          <label><input id="radio" type="radio">Choice</label>
          <input id="readonly" aria-label="Read only" readonly>
          <input id="secret" type="password" value="never expose this">
          <button id="off" disabled>Disabled</button>
          <select id="category" aria-label="Category">
            <option value="all">All</option><option value="x">Design</option><option value="x">Coastal</option>
            <option disabled>Unavailable</option>
          </select>
          <select id="tags" aria-label="Tags" multiple><option>A</option><option>B</option></select>
          <button type="button" id="sink">Focus sink</button>
          </form><aside id="unrelated">News</aside>
        """))
        page = browser.observe(screenshot=False)
        buy = next(a for a in page["actions"] if a["label"] == "Buy")
        # Structured effect context reaches Python: derived flags only, no free text.
        ctx = buy.get("ctx") or {}
        assert ctx.get("form") is True and (ctx.get("fields") or {}).get("password") is True
        assert ctx.get("submit") is False  # type="button" is not a submit member
        passed.append("bounded ctx propagates form membership and field inventory")
        browser.evaluate("document.querySelector('#unrelated').textContent='New unrelated news'")
        assert browser.fresh(page, buy)
        assert not browser.fresh(page)
        passed.append("click guard accepts unrelated visible updates; terminal guard rejects them")
        for label, expression in {
            "nearby price": "document.querySelector('#price').textContent='Total $100'",
            "form value": "document.querySelector('#query').value='changed'",
            "form toggle": "document.querySelector('#check').checked=true",
            "target replacement": "document.querySelector('#buy').outerHTML=document.querySelector('#buy').outerHTML",
        }.items():
            page = browser.observe(screenshot=False)
            buy = next(a for a in page["actions"] if a["label"] == "Buy")
            browser.evaluate(expression)
            assert not browser.fresh(page, buy), label
            passed.append(label + " invalidates action-specific guard")

        page = browser.observe(screenshot=False)
        actions = page["actions"]
        for role in ("checkbox", "radio"):
            assert {a["kind"] for a in actions if a.get("role") == role} == {"click"}
        assert {a["kind"] for a in actions if a["label"] == "Read only"} == {"click"}
        assert not any(a["label"] == "Disabled" or a.get("value") == "never expose this" for a in actions)
        select_actions = [a for a in actions if a["kind"] == "select"]
        assert [(a["value"], a["option_index"]) for a in select_actions] == [("x", 1), ("x", 2)]
        assert not any(a.get("label", "").startswith("Tags") for a in actions)
        passed.append("native controls expose only supported operations and safe values")

        coastal = next(a for a in select_actions if a["option_label"] == "Coastal")
        browser.act(coastal, page)
        assert browser.evaluate("document.querySelector('#category').selectedIndex") == 2
        passed.append("native dropdown selects the exact observed option index despite duplicate values")

        browser.evaluate("document.querySelector('#query').onfocus=()=>document.querySelector('#sink').focus()")
        page = browser.observe(screenshot=False)
        field = next(a for a in page["actions"] if a["kind"] == "fill")
        try:
            browser.act(field, page, text="must-not-type")
        except StalePage:
            pass
        else:
            raise AssertionError("Focus redirection allowed text insertion")
        assert browser.evaluate("document.querySelector('#query').value") != "must-not-type"
        browser.evaluate("document.querySelector('#query').onfocus=null")
        passed.append("post-click focus redirect blocks text insertion")

        browser.evaluate("document.querySelector('#query').addEventListener('input',()=>setTimeout(()=>{"
                         "document.querySelector('#suggestions').innerHTML='<div role=option>Generated</div>'"
                         "},60))")
        page = browser.observe(screenshot=False)
        field = next(a for a in page["actions"] if a["kind"] == "fill")
        browser.act(field, page, text="Generated")
        page = browser.observe(screenshot=False)
        value = browser.evaluate("document.querySelector('#query').value")
        assert value == "Generated", repr(value)
        assert any(a.get("role") == "option" for a in page["actions"])
        passed.append("real text input waits for asynchronous combobox suggestions")

        # --- v0.5 authority integrity: mutation between validation and input ---
        def reset_adv():
            browser.evaluate("document.body.innerHTML=" + repr("""
              <style>button{width:180px;height:50px}</style>
              <button id="adv" onclick="window.hits=(window.hits||0)+1">Continue</button>
              <button id="nested" onclick="window.nested=(window.nested||0)+1">Go <span>nested</span></button>
            """))
            browser.evaluate("window.hits=0; window.nested=0; window.evil=0")

        reset_adv()
        page = browser.observe(screenshot=False)
        adv = next(a for a in page["actions"] if a["label"] == "Continue")
        browser.evaluate("document.querySelector('#adv').textContent='Place order'")
        try:
            browser.act(adv, page)
        except (RuntimeError, StalePage):
            pass
        else:
            raise AssertionError("Post-observe commit swap executed")
        assert browser.evaluate("window.hits") == 0
        passed.append("post-observe commit swap aborts before input")

        # A page-injected listener fires during the trusted press itself; each of
        # these must abort at the pre-release identity check and never hit the
        # target. (Atomic execution has no press/release window — see below.)
        # handler: page JS run during mousePressed; clicks_expected: whether the
        # hygiene release can still complete a click on the same physical
        # element (navigation changes page identity without detaching the node).
        for label, handler, clicks_expected in [
            ("same-label node swap", "const b=document.querySelector('#adv');"
             "b.outerHTML='<button id=adv onclick=\"window.evil=(window.evil||0)+1\">Continue</button>'", 0),
            ("pre-release overlay", "const c=document.createElement('div');"
             "c.style.cssText='position:fixed;inset:0;z-index:9999';document.body.append(c)", 0),
            ("pre-release geometry drift",
             "document.querySelector('#adv').style.transform='translateX(400px)'", 0),
            # Page identity includes the live form-state fingerprint, so adding
            # an input mid-press is a synchronous same-document page mutation.
            ("mid-press page mutation", "document.body.append(document.createElement('input'))", 1),
        ]:
            reset_adv()
            page = browser.observe(screenshot=False)
            adv = next(a for a in page["actions"] if a["label"] == "Continue")
            browser.evaluate(
                "document.querySelector('#adv').addEventListener('mousedown',()=>{" + handler + "})"
            )
            try:
                browser.act(adv, page, guarantee="trusted")
            except StalePage:
                pass
            else:
                raise AssertionError(f"{label}: input executed")
            assert browser.evaluate("window.hits") == clicks_expected, label
            assert browser.evaluate("window.evil") == 0, label
            passed.append(f"{label} aborts between press and release")

        # Atomic guarantee: validation + e.click() happen in one isolated-world
        # turn. A hostile mousedown handler cannot interleave because no press
        # or release is ever dispatched — the sabotage listener never even runs.
        reset_adv()
        page = browser.observe(screenshot=False)
        adv = next(a for a in page["actions"] if a["label"] == "Continue")
        browser.evaluate(
            "window.sabotage=0; document.querySelector('#adv').addEventListener('mousedown',()=>{"
            "window.sabotage=1; const b=document.querySelector('#adv');"
            "b.outerHTML='<button id=adv onclick=\"window.evil=(window.evil||0)+1\">Continue</button>'})"
        )
        browser.act(adv, page)
        assert browser.evaluate("window.hits") == 1
        assert browser.evaluate("window.sabotage") == 0  # no press was dispatched at all
        assert browser.evaluate("window.evil") == 0
        passed.append("atomic click has no press window for mid-event sabotage")

        # The aborted press must leave clean pointer state: a later legitimate
        # trusted click still works.
        reset_adv()
        page = browser.observe(screenshot=False)
        adv = next(a for a in page["actions"] if a["label"] == "Continue")
        browser.act(adv, page, guarantee="trusted")
        assert browser.evaluate("window.hits") == 1
        passed.append("pointer state stays clean after aborted presses")

        # A nested descendant legitimately intercepts the hit test.
        page = browser.observe(screenshot=False)
        nested = next(a for a in page["actions"] if a["label"].startswith("Go"))
        browser.act(nested, page)
        assert browser.evaluate("window.nested") == 1
        passed.append("nested descendant at click point stays valid")

        # Hidden / detached targets post-observe abort before press.
        for label, expr in [
            ("hidden target", "document.querySelector('#adv').style.display='none'"),
            ("detached target", "document.querySelector('#adv').remove()"),
        ]:
            reset_adv()
            page = browser.observe(screenshot=False)
            adv = next(a for a in page["actions"] if a["label"] == "Continue")
            browser.evaluate(expr)
            try:
                browser.act(adv, page)
            except (RuntimeError, StalePage):
                pass
            else:
                raise AssertionError(f"{label}: input executed")
            assert browser.evaluate("window.hits") == 0, label
            passed.append(f"post-observe {label} aborts before input")

        browser.call("Page.navigate", url="about:blank")
        assert not browser.fresh(page, field)
        passed.append("navigation invalidates the old document")
    finally:
        browser.close()
    print("\n".join(passed))
    print(f"PASS: {len(passed)} browser guard checks; no model calls")


if __name__ == "__main__":
    main()
