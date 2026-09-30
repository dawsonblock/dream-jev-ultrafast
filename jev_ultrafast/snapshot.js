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
  const visible = e => !e.closest('[aria-hidden="true"],[inert]') &&
    e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
  const name = (e,seen=new Set()) => {
    if (!e || seen.has(e)) return '';
    seen.add(e);
    const referenced=(e.getAttribute?.('aria-labelledby')||'').split(/\s+/)
      .map(id=>name(document.getElementById(id),seen)).filter(Boolean).join(' ');
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
      if (['button','submit','reset','image'].includes(e.type)) return 'button';
      if (e.type==='search') return 'searchbox';
      if (e.type==='number') return 'spinbutton';
      if (['text','email','url','tel'].includes(e.type)) return 'textbox';
    }
    return null;
  };

  cache.pageKey=()=>[performance.timeOrigin,location.href,scrollX,scrollY,innerWidth,innerHeight,
    [...document.querySelectorAll('input,textarea,select')].filter(safe)
      .map(e=>[identity(e),clip(e.value,512),Boolean(e.checked),e.selectedIndex,Boolean(e.disabled),Boolean(e.readOnly)])];
  cache.guard=e=>{
    if (!e?.isConnected || !visible(e)) return null;
    const scope=e.closest('form,dialog,[role="dialog"],article,li,tr,[role="row"]') || e.parentElement;
    return [identity(e),role(e),name(e),clip(e.value,512),e.checked??null,e.selectedIndex??null,
      e.readOnly??null,e.matches(':disabled'),e.getAttribute('aria-disabled'),
      e.getAttribute('aria-expanded'),e.getAttribute('aria-checked'),e.getAttribute('aria-selected'),
      clip(e.getAttribute('href'),1024),clip(scope?.innerText,2000)];
  };

  const actions=[];
  for (const e of document.querySelectorAll(selector)) {
    if (!safe(e) || !visible(e) || e.matches(':disabled') || e.closest('[aria-disabled="true"]')) continue;
    const r=e.getBoundingClientRect(), x=r.x+r.width/2, y=r.y+r.height/2, rname=role(e);
    if (!rname || r.width<=0 || r.height<=0 || x<0 || y<0 || x>=innerWidth || y>=innerHeight) continue;
    if (rname==='gridcell' && e.querySelector('button,[role="button"]')) continue;
    const base={node:identity(e),role:rname,label:name(e)||rname,
      rect:{x:r.x,y:r.y,w:r.width,h:r.height}};
    for (const key of ['checked','selected','expanded']) {
      const value=e.getAttribute('aria-'+key);
      if (value!==null) base[key]=clip(value,32);
    }
    if (['checkbox','radio'].includes(e.type)) base.checked=String(e.checked);

    if (e.tagName==='SELECT' && !e.multiple) {
      const current=[...e.selectedOptions].map(o=>clip(o.label,256)).join(', ');
      for (let optionIndex=0; optionIndex<e.options.length; optionIndex++) {
        const o=e.options[optionIndex];
        if (o.selected || o.disabled || o.closest('optgroup[disabled]')) continue;
        const optionLabel=clip(o.label,256), optionValue=clip(o.value,512);
        actions.push({...base,kind:'select',value:optionValue,option_index:optionIndex,
          option_label:optionLabel,current_value:clip(current,512),label:clip(base.label+' → '+optionLabel,256)});
      }
    } else if (e.tagName!=='SELECT') {
      const editable=!e.readOnly && e.getAttribute('aria-readonly')!=='true' &&
        (['textbox','searchbox','spinbutton'].includes(rname) ||
          (rname==='combobox' && ['INPUT','TEXTAREA'].includes(e.tagName)));
      const value='value' in e ? clip(e.value,512) :
        e.isContentEditable || rname==='combobox' ? clip(e.innerText,512) : '';
      actions.push({...base,kind:editable?'fill':'click',value});
      if (editable) actions.push({...base,kind:'click',value,label:clip('Open '+base.label,256)});
    }
  }

  const words=[], walker=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);
  const range=document.createRange(); let node,length=0;
  while ((node=walker.nextNode()) && length<6000) {
    const value=clip(node.textContent,1000), parent=node.parentElement;
    if (!value || !parent || parent.closest('script,style,noscript,template') || !visible(parent)) continue;
    range.selectNodeContents(node); const r=range.getBoundingClientRect();
    if (r.width>0 && r.height>0 && r.bottom>0 && r.top<innerHeight && r.right>0 && r.left<innerWidth) {
      words.push(value); length+=value.length;
    }
  }

  const text=words.join('\n').slice(0,6000), height=document.documentElement.scrollHeight;

  // Bound the browser-to-Python catalogue before it becomes part of the semantic
  // marker. Goal-aware ranking and the 250-model-candidate budget happen in Python.
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
  actions.push({id:'wait',kind:'wait',label:'Wait for the page to update'});
  return {url:clip(location.href,4096),title:clip(document.title,256),w:innerWidth,h:innerHeight,text,
    scroll:{y:scrollY,height},actions,marker,page_key,guards,
    omitted_actions:Math.max(0,raw_total-1200),raw_action_count:raw_total};
})()
