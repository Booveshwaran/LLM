/* ═══════════════════════════════════════════════════
   HealthcareAI — App Logic: SSE, Agent State, UI
   ═══════════════════════════════════════════════════ */
const API='http://localhost:8000';
const AGENTS=['planner','researcher','critic','refiner','solver'];
const $=id=>document.getElementById(id);
const agentStates={};AGENTS.forEach(a=>agentStates[a]='waiting');
let timerInterval=null,startTime=0,waveforms={},stopPreloader=null;

/* ── Init ─────────────────────────────────────────── */
window.addEventListener('load',()=>{
  const pc=document.querySelector('#preloader canvas');
  if(pc) stopPreloader=window._canvas.initPreloader(pc);
  const bg=$('bg-canvas');
  if(bg) window._canvas.initBackground(bg);
  window._canvas.initCursor();
  window._canvas.initNav();
  // Init waveforms
  AGENTS.forEach(a=>{
    const c=document.querySelector(`#card-${a} .waveform`);
    if(c) waveforms[a]=window._canvas.initWaveform(c,a,ag=>agentStates[ag]);
  });
  setTimeout(()=>{
    if(stopPreloader)stopPreloader();
    $('preloader').classList.add('hidden');
    animateHero();
  },2200);
});

/* ── GSAP Hero ────────────────────────────────────── */
function animateHero(){
  if(typeof gsap==='undefined')return;
  const tl=gsap.timeline({defaults:{ease:'power3.out'}});
  tl.to('.hero-badge',{opacity:1,y:0,duration:.6})
    .to('.hero-title .word',{opacity:1,y:0,duration:.5,stagger:.07},'-=.3')
    .to('.hero-sub',{opacity:1,y:0,duration:.6},'-=.2')
    .to('.hero-stats',{opacity:1,y:0,duration:.6},'-=.3')
    .to('.query-panel',{opacity:1,y:0,duration:.5},'-=.2');
  // Count-up numbers
  document.querySelectorAll('.hero-stat .num').forEach(el=>{
    const target=el.dataset.value||el.textContent;
    const isPercent=target.includes('%');
    const num=parseInt(target);
    gsap.from(el,{textContent:0,duration:1.5,ease:'power2.out',snap:{textContent:1},
      onUpdate:function(){el.textContent=Math.round(gsap.getProperty(el,'textContent'))+(isPercent?'%':'+');}
    });
  });
}

/* ── Card States ──────────────────────────────────── */
function setCardState(agent,state,preview){
  agentStates[agent]=state;
  const card=$('card-'+agent),badge=$('badge-'+agent),prev=$('preview-'+agent);
  card.classList.remove('running','done');
  badge.className='agent-badge';
  if(state==='waiting'){badge.classList.add('badge-waiting');badge.textContent='WAITING';prev.textContent='';}
  else if(state==='running'){badge.classList.add('badge-running');badge.textContent='RUNNING';card.classList.add('running');prev.textContent='';if(waveforms[agent])waveforms[agent].restart();}
  else if(state==='done'){badge.classList.add('badge-done');badge.textContent='DONE';card.classList.add('done');if(preview)typeText(prev,preview.length>80?preview.substring(0,80)+'…':preview,10);}
  else if(state==='skipped'){badge.classList.add('badge-skipped');badge.textContent='SKIPPED';prev.textContent=preview||'Not needed';card.style.opacity='.4';}
}
function resetAllCards(){AGENTS.forEach(a=>{setCardState(a,'waiting');$('card-'+a).style.opacity='1';});}

/* ── Typewriter ───────────────────────────────────── */
function typeText(el,text,speed){
  el.textContent='';let i=0;
  (function t(){if(i<text.length){el.textContent+=text[i++];setTimeout(t,speed);}})();
}
function typeAnswer(text){
  const el=$('answer-text');el.innerHTML='';
  // Split disclaimer
  const parts=text.split(/⚕️\s*Disclaimer:/i);
  const main=parts[0]||text;
  let i=0;
  (function t(){
    if(i<main.length){el.textContent+=main[i++];setTimeout(t,10);}
    else if(parts.length>1){
      const d=document.createElement('span');d.className='disclaimer';
      d.textContent='⚕️ Disclaimer:'+parts[1];el.appendChild(d);
    }
  })();
}

/* ── Timer ─────────────────────────────────────────── */
function startTimer(){startTime=Date.now();$('elapsed-timer').classList.add('visible');timerInterval=setInterval(()=>{$('elapsed-timer').textContent='⏱ '+((Date.now()-startTime)/1000).toFixed(1)+'s';},100);}
function stopTimer(){if(timerInterval){clearInterval(timerInterval);timerInterval=null;}$('elapsed-timer').textContent='⏱ '+((Date.now()-startTime)/1000).toFixed(1)+'s';}

/* ── Trace ─────────────────────────────────────────── */
function toggleTrace(){
  const tc=$('trace-content'),tb=$('trace-toggle');
  tc.classList.toggle('visible');tb.classList.toggle('expanded');
}
function expandTrace(){$('trace-content').classList.add('visible');$('trace-toggle').classList.add('expanded');}

/* ── Error ─────────────────────────────────────────── */
function showError(msg){$('error-msg').textContent=msg;$('error-banner').classList.add('visible');}
function hideError(){$('error-banner').classList.remove('visible');}

/* ── Loading ───────────────────────────────────────── */
function setLoading(on){
  $('run-btn').disabled=on;$('run-btn').classList.toggle('loading',on);
  $('spinner').classList.toggle('visible',on);
  $('btn-text').textContent=on?'ANALYZING…':'▶ ANALYZE';
}
function hideResults(){
  $('final-answer').classList.remove('visible');
  $('cache-stats').classList.remove('visible');
  $('latency-tag').classList.remove('visible');
  $('trace-content').classList.remove('visible');
  $('trace-toggle').classList.remove('expanded');
  $('trace-content').innerHTML='';
}

/* ── Copy ──────────────────────────────────────────── */
function copyAnswer(){
  const text=$('answer-text').textContent;
  navigator.clipboard.writeText(text).then(()=>{
    const btn=$('copy-btn');btn.textContent='✓ Copied';
    setTimeout(()=>{btn.textContent='Copy';},2000);
  });
}

/* ── Show Result ───────────────────────────────────── */
function showResult(data){
  stopTimer();
  let answer=data.final_answer||'No answer generated.';
  try{const p=JSON.parse(answer);if(p&&p.answer)answer=p.answer;}catch(e){}
  // Dedupe disclaimers
  const dp=answer.split(/⚕️\s*Disclaimer:/i);
  if(dp.length>2)answer=dp[0]+'⚕️ Disclaimer:'+dp[1];
  typeAnswer(answer);
  $('final-answer').classList.add('visible');
  // Trace
  const tc=$('trace-content');tc.innerHTML='';
  const trace=data.reasoning_trace||[];
  trace.forEach((entry,i)=>{
    const div=document.createElement('div');div.className='trace-step';
    div.style.animationDelay=i*60+'ms';
    const l=entry.toLowerCase();
    if(l.includes('planner'))div.dataset.agent='planner';
    else if(l.includes('researcher'))div.dataset.agent='researcher';
    else if(l.includes('critic')||l.includes('reviewer'))div.dataset.agent='critic';
    else if(l.includes('refiner'))div.dataset.agent='refiner';
    else if(l.includes('solver')||l.includes('advisor'))div.dataset.agent='solver';
    div.textContent=(i+1)+'. '+entry;
    tc.appendChild(div);
  });
  const sc=$('trace-toggle').querySelector('.step-count');
  if(sc) sc.textContent=trace.length+' steps';
  expandTrace();
  mapTraceToAgents(trace);
  // Stats
  const s=data.token_stats||{};
  $('stat-hits').textContent=s.cache_hits!==undefined?s.cache_hits:'0';
  $('stat-misses').textContent=s.cache_misses!==undefined?s.cache_misses:'0';
  $('stat-tokens').textContent='~'+(s.estimated_tokens_saved!==undefined?s.estimated_tokens_saved:'0');
  $('cache-stats').classList.add('visible');
  const elapsed=((Date.now()-startTime)/1000).toFixed(1);
  $('latency-tag').textContent='⏱ Completed in '+elapsed+'s · Retries: '+(data.retry_count||0);
  $('latency-tag').classList.add('visible');
  setLoading(false);
}
function mapTraceToAgents(trace){
  const map={};let refUsed=false;
  trace.forEach(e=>{const l=e.toLowerCase();
    if(l.includes('planner'))map.planner=e;
    else if(l.includes('researcher'))map.researcher=e;
    else if(l.includes('critic')||l.includes('reviewer'))map.critic=e;
    else if(l.includes('refiner')){map.refiner=e;refUsed=true;}
    else if(l.includes('solver')||l.includes('advisor'))map.solver=e;
  });
  AGENTS.forEach(a=>{
    if(map[a])setCardState(a,'done',map[a]);
    else if(a==='refiner'&&!refUsed)setCardState(a,'skipped','Not needed');
    else setCardState(a,'done','Completed');
  });
}

/* ── Main Run (SSE) ────────────────────────────────── */
function handleRun(){
  hideError();
  const query=$('query-input').value.trim();
  if(!query){$('query-input').classList.add('shake');setTimeout(()=>$('query-input').classList.remove('shake'),500);return;}
  setLoading(true);resetAllCards();hideResults();startTimer();
  const url=API+'/query/stream?q='+encodeURIComponent(query)+'&mock='+$('mock-toggle').checked;
  let evtSource;
  try{evtSource=new EventSource(url);}catch(e){fallbackFetch(query,$('mock-toggle').checked);return;}
  evtSource.onmessage=function(event){
    try{
      const d=JSON.parse(event.data);
      if(d.type==='agent_update')setCardState(d.agent,d.status,d.detail||'');
      else if(d.type==='result'){evtSource.close();showResult(d);}
      else if(d.type==='error'){evtSource.close();stopTimer();showError(d.message||'Unknown error');resetAllCards();setLoading(false);}
    }catch(e){}
  };
  evtSource.onerror=function(){evtSource.close();stopTimer();showError('Stream connection failed');resetAllCards();setLoading(false);};
}
function fallbackFetch(query,mock){
  fetch(API+'/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query,mock})})
  .then(r=>{if(!r.ok)return r.json().then(d=>{throw new Error(d.detail||'HTTP '+r.status);});return r.json();})
  .then(data=>showResult(data))
  .catch(err=>{stopTimer();showError(err.message||'Network error');resetAllCards();setLoading(false);});
}
$('query-input').addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){e.preventDefault();handleRun();}});
