(() => {
  if (!document.body) return null;

  // This cache lives in Jev's CDP isolated world. The page's main-world JavaScript
  // cannot read or replace it. browser.py is responsible for always evaluating this
  // file in that isolated execution context.
  const cache = globalThis.__jevFastV2 ||= {ids:new WeakMap(), nodes:new Map(), next:1};
  const clip = (value, limit) => String(value ?? '').replace(/\s+/g, ' ').trim().slice(0, limit);
  const identity = e => {
    if (!cache.ids.has(e)) cache.ids.set(e,cache.next++);
    const id=cache.ids.get(e); cache.nodes.set(id,e); return id;
  };
  for (const [id,e] of cache.nodes) if (!e.isConnected) cache.nodes.delete(id);

  const safe = e => !['password','file','hidden'].includes(String(e.type || '').toLowerCase());
  // Catalogue membership is a separate question from value capture: file inputs
  // may be *offered* (upload is a supported operation) while their value stays
  // excluded from every recorded field by safe() above. Password and hidden
  // inputs are never offered at all.
  const offerable = e => !['password','hidden'].includes(String(e.type || '').toLowerCase());
  const visible = e => !e.closest('[aria-hidden="true"],[inert]') &&
    e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
  // Accessible-name resolution is scope-relative: aria-labelledby ids resolve in
  // the element's own root (document or shadow root), not the top document.
  const name = (e,seen=new Set()) => {
    if (!e || seen.has(e)) return '';
    seen.add(e);
    const root=e.getRootNode && e.getRootNode();
    const byId=id=>{try{return (root&&root.getElementById)?root.getElementById(id):document.getElementById(id);}catch(_){return null;}};
    const referenced=(e.getAttribute?.('aria-labelledby')||'').split(/\s+/)
      .map(id=>name(byId(id),seen)).filter(Boolean).join(' ');
    const result = referenced || e.getAttribute?.('aria-label') ||
      [...(e.labels||[])].map(l=>name(l,seen)).filter(Boolean).join(' ') ||
      (['button','submit','reset'].includes(e.type) ? e.value : '') || e.getAttribute?.('alt') ||
      (e.tagName==='INPUT' ? '' : [...(e.childNodes||[])].map(n=>n.nodeType===3 ? n.textContent :
        n.nodeType===1 && n.getAttribute('aria-hidden')!=='true' ? name(n,seen) : '').join(' ').trim()) ||
      e.getAttribute?.('title') || e.getAttribute?.('placeholder') || '';
    return clip(result, 256);
  };

  const roles=['button','link','checkbox','radio','switch','tab','menuitem','menuitemradio',
    'option','gridcell','combobox','textbox','searchbox','spinbutton'];
  const selector='a[href],button,input,textarea,select,summary,'+
    '[contenteditable]:not([contenteditable="false"]),'+
    roles.map(role=>'[role="'+role+'"]').join(',');
  const role = e => {
    const explicit=e.getAttribute('role');
    if (roles.includes(explicit)) return explicit;
    if (e.tagName==='BUTTON' || e.tagName==='SUMMARY') return 'button';
    if (e.tagName==='A') return 'link';
    if (e.tagName==='SELECT') return 'combobox';
    if (e.tagName==='TEXTAREA' || e.isContentEditable) return 'textbox';
    if (e.tagName==='INPUT') {
      if (['checkbox','radio'].includes(e.type)) return e.type;
      if (['button','submit','reset','image','file'].includes(e.type)) return 'button';
      if (e.type==='search') return 'searchbox';
      if (e.type==='number') return 'spinbutton';
      if (['text','email','url','tel'].includes(e.type)) return 'textbox';
    }
    return null;
  };

  // Deep reachability: the isolated world reads same-origin frame DOMs and open
  // shadow roots through ordinary DOM references, so a single top-frame world
  // can observe and guard elements inside them. Cross-origin frames stay opaque
  // (counted, never traversed) and closed shadow roots are indistinguishable
  // from "no shadow root" — both are honest coverage gaps, not silent failures.
  // Depth and breadth caps keep a hostile or cyclic frame tree from making the
  // snapshot itself unbounded.
  let unreachable_frames = 0;
  const all_elements = [];
  const deep_visit = (root, depth) => {
    if (depth > 8 || all_elements.length > 60000) return;
    for (const el of root.querySelectorAll('*')) {
      all_elements.push(el);
      if (el.shadowRoot) {
        deep_visit(el.shadowRoot, depth + 1);
      } else if (el.tagName === 'IFRAME' || el.tagName === 'FRAME') {
        let doc = null;
        try { doc = el.contentDocument; } catch (_) { doc = null; }
        // Only traverse frames that are themselves on screen — a display:none
        // frame's contents cannot be acted on and must not be offered.
        if (doc && visible(el)) deep_visit(doc, depth + 1);
        else if (!doc) unreachable_frames++;
      }
    }
  };
  deep_visit(document, 0);
  const query_deep = sel => all_elements.filter(e => { try { return e.matches(sel); } catch (_) { return false; } });
  // pageKey must see the DOM as it is RIGHT NOW — a node appended between
  // observe and dispatch changes page identity and must fail the guard, which
  // filtering the captured all_elements could never do. Re-walk every document
  // that could host a matching element, live, on every call.
  const deep_query_live = (sel, cap = 60000) => {
    const out = [];
    const visit = (root, depth) => {
      if (depth > 8 || out.length > cap) return;
      for (const e of root.querySelectorAll(sel)) out.push(e);
      for (const e of root.querySelectorAll('iframe,frame')) {
        try { if (e.contentDocument) visit(e.contentDocument, depth + 1); } catch (_) {}
      }
      for (const e of root.querySelectorAll('*')) {
        if (e.shadowRoot) visit(e.shadowRoot, depth + 1);
      }
    };
    visit(document, 0);
    return out;
  };
  cache.reachableDocuments = () => {
    const docs = new Set();
    for (const e of all_elements) if (e.ownerDocument) docs.add(e.ownerDocument);
    return docs;
  };

  // Top-viewport coordinates for an element in any reachable document: the
  // element's own getBoundingClientRect (relative to its document's viewport)
  // plus every ancestor frame element's content-box offset up to the top.
  cache.viewRect = e => {
    const r = e.getBoundingClientRect();
    let x = r.x, y = r.y, win = e.ownerDocument && e.ownerDocument.defaultView, hops = 0;
    while (win && win !== win.parent && hops++ < 8) {
      const fe = win.frameElement;
      if (!fe) break;
      const fr = fe.getBoundingClientRect();
      x += fr.x + (fe.clientLeft || 0);
      y += fr.y + (fe.clientTop || 0);
      win = win.parent;
    }
    return {x, y, w: r.width, h: r.height};
  };

  cache.pageKey=()=>[performance.timeOrigin,location.href,scrollX,scrollY,innerWidth,innerHeight,
    deep_query_live('input,textarea,select').filter(safe)
      .map(e=>[identity(e),clip(e.value,512),Boolean(e.checked),e.selectedIndex,Boolean(e.disabled),Boolean(e.readOnly)])];

  // Structural effect context for the deterministic authority classifier in
  // policy.py. Deliberately derived flags only — no free text leaves this
  // function — so the authority surface adds no privacy surface. Origin checks
  // bind the element's OWN document location: a form inside a same-origin
  // iframe that posts cross-origin must still classify external.
  const ctxOf = e => {
    const doc = e.ownerDocument || document;
    const docLoc = (doc.defaultView && doc.defaultView.location) || location;
    const form = e.form || e.closest('form');
    const scope = e.closest('dialog,[role="dialog"],[aria-modal="true"],form,article,li,tr,[role="row"],[role="search"]');
    const fieldScope = form || scope;
    const q = sel => !!(fieldScope && fieldScope.querySelector && fieldScope.querySelector(sel));
    const fields = fieldScope ? {
      password: q('input[type="password"],input[autocomplete="current-password"],input[autocomplete="new-password"]'),
      file: q('input[type="file"]'),
      money: q('input[name*="amount" i],input[name*="price" i],input[name*="card" i],input[name*="cvv" i],input[id*="card" i],input[id*="cvv" i],input[name*="routing" i],input[name*="iban" i]'),
      email: q('input[type="email"]'),
      search: q('input[type="search"],[role="searchbox"]') || fieldScope.getAttribute('role')==='search',
    } : {};
    let submit = false;
    if (form) {
      const t = String(e.type||'').toLowerCase();
      if (e.tagName==='BUTTON') submit = t!=='button' && t!=='reset';
      if (e.tagName==='INPUT') submit = ['submit','image'].includes(t);
    }
    let external = false, messaging = false;
    const href = e.getAttribute && e.getAttribute('href');
    if (href) {
      const scheme = href.trim().split(':')[0].toLowerCase();
      if (['mailto','tel','sms'].includes(scheme)) messaging = true;
      else { try { external = new URL(href, docLoc.href).origin !== docLoc.origin; } catch (_) {} }
    }
    if (e.target === '_blank') external = true;
    let method = '', same_origin = null;
    if (form) {
      const m = String(form.getAttribute('method')||'get').toLowerCase();
      method = ['get','post','dialog'].includes(m) ? m : 'post';
      try { same_origin = new URL(form.getAttribute('action')||docLoc.href, docLoc.href).origin === docLoc.origin; }
      catch (_) { same_origin = null; }
    }
    // The target's own data class — typing into a sensitive field is itself a
    // disclosure because page JS observes every input event before any submit.
    let field = '';
    if (e.matches && e.matches('input,textarea,select,[contenteditable=""],[contenteditable="true"]')) {
      const t = String(e.type||'').toLowerCase();
      const ac = String(e.getAttribute('autocomplete')||'').toLowerCase();
      const nm = `${e.name||''} ${e.id||''} ${e.getAttribute('aria-label')||''}`.toLowerCase();
      if (t==='password' || /password|username/.test(ac)) field = 'auth';
      else if (ac==='one-time-code' || /\botp\b|2fa|verification.code/.test(nm)) field = 'otp';
      else if (t==='file') field = 'file';
      else if (t==='search' || e.getAttribute('role')==='searchbox') field = 'search';
      else if (t==='email' || ac.includes('email')) field = 'email';
      else if (t==='tel' || ac.includes('tel')) field = 'tel';
      else if (/cc[-_ ]|card|cvv|cvc|routing|iban|payment|billing|amount|price|acct/.test(ac+' '+nm)) field = 'money';
      else if (/comment|message|bio|review|post|tweet/.test(nm)) field = 'message';
      else field = 'text';
    }
    return {
      form: !!form, submit, method, same_origin, fields, field,
      modal: !!e.closest('dialog,[role="dialog"],[aria-modal="true"]'),
      row: !!e.closest('tr,li,[role="row"],[role="listitem"],article'),
      external, messaging,
      download: !!(e.hasAttribute && e.hasAttribute('download')),
    };
  };
  // The guard binds BOTH element identity and the full authority context
  // classify_effect() consumed at decision time. Mutating authority-relevant
  // attributes — type, autocomplete, form association/method/action, sibling
  // field inventory, href scheme, target, download, modal scope — between
  // observation and dispatch must change this value or the same action could
  // execute under a weaker authority than its true semantics.
  cache.guard=e=>{
    if (!e?.isConnected || !visible(e)) return null;
    const scope=e.closest('form,dialog,[role="dialog"],article,li,tr,[role="row"]') || e.parentElement;
    return [identity(e),role(e),name(e),clip(e.value,512),e.checked??null,e.selectedIndex??null,
      e.readOnly??null,e.matches(':disabled'),e.getAttribute('aria-disabled'),
      e.getAttribute('aria-expanded'),e.getAttribute('aria-checked'),e.getAttribute('aria-selected'),
      clip(e.getAttribute('href'),1024),clip(scope?.innerText,2000),ctxOf(e)];
  };

  // Actions are collected per observed node, then merged round-robin across
  // nodes. A native <select> expands into one action per option, so a single
  // giant dropdown in DOM order could otherwise consume the entire bounded
  // catalogue before later controls are ever represented — the 1200-cap below
  // would truncate them away before Python's goal-aware ranking could see
  // them. Interleaving guarantees every node contributes its first action
  // before any node contributes its second, so the cap starves a node's
  // deep alternative list instead of whole controls behind it.
  const buckets=[];
  for (const e of all_elements) {
    if (!e.matches(selector) || !offerable(e) || !visible(e) || e.matches(':disabled') || e.closest('[aria-disabled="true"]')) continue;
    const r=cache.viewRect(e), x=r.x+r.w/2, y=r.y+r.h/2, rname=role(e);
    if (!rname || r.w<=0 || r.h<=0 || x<0 || y<0 || x>=innerWidth || y>=innerHeight) continue;
    if (rname==='gridcell' && e.querySelector('button,[role="button"]')) continue;
    const base={node:identity(e),role:rname,label:name(e)||rname,
      rect:{x:r.x,y:r.y,w:r.w,h:r.h},ctx:ctxOf(e)};
    for (const key of ['checked','selected','expanded']) {
      const value=e.getAttribute('aria-'+key);
      if (value!==null) base[key]=clip(value,32);
    }
    if (['checkbox','radio'].includes(e.type)) base.checked=String(e.checked);

    if (e.tagName==='INPUT' && String(e.type).toLowerCase()==='file') {
      // Upload is an observed operation like any other: the catalogue names the
      // input, never a path. The operator-declared file set arrives later via
      // the UPLOAD target question and resolves server-side at dispatch.
      buckets.push([{...base,kind:'upload',role:'button',
        label:clip('Choose file — '+(base.label||'file input'),256)}]);
    } else if (e.tagName==='SELECT' && !e.multiple) {
      const current=[...e.selectedOptions].map(o=>clip(o.label,256)).join(', ');
      const bucket=[];
      for (let optionIndex=0; optionIndex<e.options.length; optionIndex++) {
        const o=e.options[optionIndex];
        if (o.selected || o.disabled || o.closest('optgroup[disabled]')) continue;
        const optionLabel=clip(o.label,256), optionValue=clip(o.value,512);
        bucket.push({...base,kind:'select',value:optionValue,option_index:optionIndex,
          option_label:optionLabel,current_value:clip(current,512),label:clip(base.label+' → '+optionLabel,256)});
      }
      if (bucket.length) buckets.push(bucket);
    } else if (e.tagName!=='SELECT') {
      const editable=!e.readOnly && e.getAttribute('aria-readonly')!=='true' &&
        (['textbox','searchbox','spinbutton'].includes(rname) ||
          (rname==='combobox' && ['INPUT','TEXTAREA'].includes(e.tagName)));
      const value='value' in e && safe(e) ? clip(e.value,512) :
        e.isContentEditable || rname==='combobox' ? clip(e.innerText,512) : '';
      const bucket=[{...base,kind:editable?'fill':'click',value}];
      if (editable) bucket.push({...base,kind:'click',value,label:clip('Open '+base.label,256)});
      buckets.push(bucket);
    }
  }

  // Nested scroll containers: any visible element that can actually scroll and
  // holds actionable content earns guarded scroll actions. In-world scrollBy
  // keeps execution atomic — the same validate-and-mutate turn as click/fill —
  // and reaches scrollers a fixed-point mouse wheel cannot (overflow divs,
  // listboxes, frame documents) without trusting physical input.
  const scrollable=[];
  for (const e of all_elements) {
    if (scrollable.length>=8) break;
    try {
      if (!visible(e)) continue;
      const doc=e.ownerDocument;
      if (e===doc.documentElement || e===doc.body || e===doc.scrollingElement) continue;
      const cs=doc.defaultView.getComputedStyle(e);
      if (!/(auto|scroll)/.test(cs.overflowY)) continue;
      if (e.clientHeight<40 || e.scrollHeight<=e.clientHeight+8) continue;
      if (!e.querySelector(selector)) continue;
      scrollable.push(e);
    } catch (_) {}
  }
  for (const e of scrollable) {
    const label=clip(name(e)||e.getAttribute('aria-label')||e.getAttribute('role')||e.tagName.toLowerCase(),200);
    const step=Math.max(80,Math.round(e.clientHeight*0.8));
    const bucket=[];
    if (e.scrollTop+e.clientHeight<e.scrollHeight-2)
      bucket.push({node:identity(e),kind:'scroll',delta:step,ctx:ctxOf(e),label:clip('Scroll down inside '+label,256)});
    if (e.scrollTop>0)
      bucket.push({node:identity(e),kind:'scroll',delta:-step,ctx:ctxOf(e),label:clip('Scroll up inside '+label,256)});
    if (bucket.length) buckets.push(bucket);
  }

  const actions=[];
  const heads=buckets.map(()=>0);
  for (let active=true; active;) {
    active=false;
    for (let b=0; b<buckets.length; b++) {
      if (heads[b]<buckets[b].length) { actions.push(buckets[b][heads[b]++]); active=true; }
    }
  }

  // Text from every reachable document, in traversal order, sharing the same
  // 6000-codepoint budget — frame and shadow content is part of the page the
  // model reads and of the marker that binds decisions to it.
  const text_roots=[document];
  for (const el of all_elements) {
    if (el.tagName==='IFRAME'||el.tagName==='FRAME') {
      try { if (el.contentDocument && visible(el)) text_roots.push(el.contentDocument); } catch (_) {}
    } else if (el.shadowRoot) text_roots.push(el.shadowRoot);
  }
  const text=(function(){
    const range=document.createRange(), kept=[];
    let used=0;
    for (const root of text_roots) {
      const win=root.defaultView || (root.ownerDocument && root.ownerDocument.defaultView);
      const vw=win?win.innerWidth:innerWidth, vh=win?win.innerHeight:innerHeight;
      const walker=document.createTreeWalker(root.body || root,NodeFilter.SHOW_TEXT);
      let n;
      while ((n=walker.nextNode()) && used<6000) {
        const value=clip(n.textContent,1000), parent=n.parentElement;
        if (!value || !parent || parent.closest('script,style,noscript,template') || !visible(parent)) continue;
        range.selectNodeContents(n); const r=range.getBoundingClientRect();
        if (r.width>0 && r.height>0 && r.bottom>0 && r.top<vh && r.right>0 && r.left<vw) {
          kept.push(value); used+=value.length;
        }
      }
    }
    return kept.join('\n').slice(0,6000);
  })();
  const height=document.documentElement.scrollHeight;

  // Bound the browser-to-Python catalogue before it becomes part of the semantic
  // marker. The round-robin merge above keeps the cut *node-fair*: truncation now
  // removes a node's deepest alternatives (e.g. a 1500-option dropdown's tail)
  // rather than controls that merely appeared after it. Goal-aware ranking and
  // the 250-model-candidate budget still happen in Python.
  const raw_total=actions.length;
  actions.splice(1200);
  const page_key=cache.pageKey(), guards={};
  for (const a of actions) if (!(a.node in guards)) guards[a.node]=cache.guard(cache.nodes.get(a.node));
  const semantics=actions.map(({rect,...action})=>action);
  const marker=[performance.timeOrigin,location.href,scrollX,scrollY,innerWidth,innerHeight,
    clip(document.title,256),text,semantics,page_key[6]];
  actions.forEach((a,i)=>a.id='e'+(i+1));
  if (scrollY+innerHeight<height-2) actions.push({id:'scroll_down',kind:'scroll',label:'Scroll down',delta:560});
  if (scrollY>0) actions.push({id:'scroll_up',kind:'scroll',label:'Scroll up',delta:-560});
  // Keyboard scroll complements the wheel: pages that hijack wheel events still
  // answer PageDown/PageUp, and End/Home reach positions one wheel delta can't.
  // Kind 'key' is whitelisted to scroll keys — browser.py refuses anything else.
  actions.push({id:'key_pagedown',kind:'key',key:'PageDown',label:'Press Page Down (scroll one screen down)'});
  actions.push({id:'key_pageup',kind:'key',key:'PageUp',label:'Press Page Up (scroll one screen up)'});
  if (scrollY+innerHeight<height-2) actions.push({id:'key_end',kind:'key',key:'End',label:'Press End (scroll to the bottom)'});
  if (scrollY>0) actions.push({id:'key_home',kind:'key',key:'Home',label:'Press Home (scroll back to the top)'});
  actions.push({id:'wait',kind:'wait',label:'Wait for the page to update'});
  return {url:clip(location.href,4096),title:clip(document.title,256),w:innerWidth,h:innerHeight,text,
    scroll:{y:scrollY,height},actions,marker,page_key,guards,
    unreachable_frames,documents:cache.reachableDocuments().size,
    omitted_actions:Math.max(0,raw_total-1200),raw_action_count:raw_total};
})()
