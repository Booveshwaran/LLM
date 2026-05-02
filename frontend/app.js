/* ═══════════════════════════════════════════════════════════════
   Healthcare AI — Immersive Frontend Logic
   GSAP animations, custom cursor, SSE streaming, agent pipeline
   ═══════════════════════════════════════════════════════════════ */

const API_URL = "http://localhost:8000";
const AGENTS = ["planner","researcher","critic","refiner","solver"];

/* ── DOM refs ───────────────────────────────────────── */
const $ = id => document.getElementById(id);
const queryInput = $("query-input"), runBtn = $("run-btn"), btnText = $("btn-text"),
      spinner = $("spinner"), mockToggle = $("mock-toggle"),
      finalAnswer = $("final-answer"), answerText = $("answer-text"),
      traceToggle = $("trace-toggle"), traceContent = $("trace-content"),
      cacheStats = $("cache-stats"), errorBanner = $("error-banner"),
      errorMsg = $("error-msg"), latencyTag = $("latency-tag"),
      elapsedTimer = $("elapsed-timer"), preloader = $("preloader");

let timerInterval = null, startTime = 0;

/* ── Preloader ──────────────────────────────────────── */
window.addEventListener("load", () => {
  setTimeout(() => {
    preloader.classList.add("hidden");
    animateHero();
  }, 1200);
});

/* ── GSAP Hero Animation ────────────────────────────── */
function animateHero() {
  if (typeof gsap === "undefined") return;
  const tl = gsap.timeline({ defaults: { ease: "power3.out" } });
  tl.to(".hero-badge", { opacity: 1, y: 0, duration: 0.6 })
    .to(".hero-title .word", { opacity: 1, y: 0, duration: 0.5, stagger: 0.06 }, "-=0.3")
    .to(".hero-sub", { opacity: 1, y: 0, duration: 0.6 }, "-=0.2")
    .to(".hero-stats", { opacity: 1, y: 0, duration: 0.6 }, "-=0.3")
    .to(".query-panel", { opacity: 1, y: 0, duration: 0.5 }, "-=0.2");
}

/* ── Custom Cursor ──────────────────────────────────── */
(function initCursor() {
  const dot = document.querySelector(".cursor-dot");
  const ring = document.querySelector(".cursor-ring");
  if (!dot || !ring || !matchMedia("(hover:hover)").matches) return;
  let mx = 0, my = 0, rx = 0, ry = 0;
  document.addEventListener("mousemove", e => { mx = e.clientX; my = e.clientY; dot.style.left = mx+"px"; dot.style.top = my+"px"; });
  (function lerp() { rx += (mx-rx)*0.12; ry += (my-ry)*0.12; ring.style.left = rx+"px"; ring.style.top = ry+"px"; requestAnimationFrame(lerp); })();
  document.querySelectorAll("button,a,.agent-card,.toggle-label").forEach(el => {
    el.addEventListener("mouseenter", () => ring.classList.add("hover"));
    el.addEventListener("mouseleave", () => ring.classList.remove("hover"));
  });
})();

/* ── Sticky Nav (hide on scroll down, show on scroll up) ── */
(function initNav() {
  const nav = document.querySelector(".nav");
  let lastY = 0;
  window.addEventListener("scroll", () => {
    const y = window.scrollY;
    nav.classList.toggle("hidden", y > lastY && y > 100);
    lastY = y;
  }, { passive: true });
})();

/* ── Card State ─────────────────────────────────────── */
function setCardState(agent, state, preview) {
  const card = $("card-"+agent), badge = $("badge-"+agent), prev = $("preview-"+agent);
  card.classList.remove("running","done");
  badge.className = "agent-badge";
  if (state === "waiting") { badge.classList.add("badge-waiting"); badge.textContent = "WAITING"; prev.textContent = ""; }
  else if (state === "running") { badge.classList.add("badge-running"); badge.textContent = "RUNNING"; card.classList.add("running"); prev.textContent = ""; }
  else if (state === "done") { badge.classList.add("badge-done"); badge.textContent = "DONE"; card.classList.add("done"); if (preview) prev.textContent = preview.length>80 ? preview.substring(0,80)+"…" : preview; }
  else if (state === "skipped") { badge.classList.add("badge-skipped"); badge.textContent = "SKIPPED"; prev.textContent = preview||""; }
}
function resetAllCards() { AGENTS.forEach(a => setCardState(a,"waiting")); }

/* ── Timer ──────────────────────────────────────────── */
function startTimer() {
  startTime = Date.now();
  elapsedTimer.classList.add("visible");
  elapsedTimer.textContent = "⏱ 0.0s";
  timerInterval = setInterval(() => { elapsedTimer.textContent = "⏱ "+((Date.now()-startTime)/1000).toFixed(1)+"s"; }, 100);
}
function stopTimer() {
  if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
  elapsedTimer.textContent = "⏱ "+((Date.now()-startTime)/1000).toFixed(1)+"s";
}

/* ── Trace ──────────────────────────────────────────── */
function toggleTrace() {
  const expanded = traceContent.classList.contains("visible");
  traceContent.classList.toggle("visible",!expanded);
  traceToggle.classList.toggle("expanded",!expanded);
}
function expandTrace() { traceContent.classList.add("visible"); traceToggle.classList.add("expanded"); }
function collapseTrace() { traceContent.classList.remove("visible"); traceToggle.classList.remove("expanded"); }

/* ── Error ──────────────────────────────────────────── */
function showError(msg) { errorMsg.textContent = msg+". Is the API running at localhost:8000?"; errorBanner.classList.add("visible"); }
function hideError() { errorBanner.classList.remove("visible"); }

/* ── Loading ────────────────────────────────────────── */
function setLoading(on) {
  runBtn.disabled = on;
  spinner.classList.toggle("visible",on);
  btnText.textContent = on ? "Analyzing…" : "▶ Analyze";
}
function hideResults() {
  finalAnswer.classList.remove("visible");
  cacheStats.classList.remove("visible");
  latencyTag.classList.remove("visible");
  collapseTrace();
  traceContent.innerHTML = "";
}

/* ── Show Result ────────────────────────────────────── */
function showResult(data) {
  stopTimer();
  let answer = data.final_answer || "No answer generated.";
  try { const p = JSON.parse(answer); if (p&&p.answer) answer = p.answer; } catch(e){}
  const parts = answer.split(/⚕️\s*Disclaimer:/i);
  if (parts.length > 2) answer = parts[0]+"⚕️ Disclaimer:"+parts[1];
  answerText.textContent = answer;
  finalAnswer.classList.add("visible");

  traceContent.innerHTML = "";
  (data.reasoning_trace||[]).forEach((entry,i) => {
    const div = document.createElement("div"); div.className = "trace-step";
    div.textContent = (i+1)+". "+entry; traceContent.appendChild(div);
  });
  expandTrace();
  mapTraceToAgents(data.reasoning_trace||[]);

  const stats = data.token_stats||{};
  $("stat-hits").textContent = stats.cache_hits!==undefined?stats.cache_hits:"0";
  $("stat-misses").textContent = stats.cache_misses!==undefined?stats.cache_misses:"0";
  $("stat-tokens").textContent = "~"+(stats.estimated_tokens_saved!==undefined?stats.estimated_tokens_saved:"0");
  cacheStats.classList.add("visible");

  const elapsed = ((Date.now()-startTime)/1000).toFixed(1);
  latencyTag.textContent = "⏱ "+elapsed+"s | Retries: "+(data.retry_count||0);
  latencyTag.classList.add("visible");
  setLoading(false);
}

function mapTraceToAgents(trace) {
  const map = {}; let refUsed = false;
  trace.forEach(e => {
    const l = e.toLowerCase();
    if (l.includes("planner")) map.planner = e;
    else if (l.includes("researcher")) map.researcher = e;
    else if (l.includes("critic")||l.includes("reviewer")) map.critic = e;
    else if (l.includes("refiner")) { map.refiner = e; refUsed = true; }
    else if (l.includes("solver")||l.includes("advisor")) map.solver = e;
  });
  AGENTS.forEach(a => {
    if (map[a]) setCardState(a,"done",map[a]);
    else if (a==="refiner"&&!refUsed) setCardState(a,"skipped","Not needed");
    else setCardState(a,"done","Completed");
  });
}

/* ── Main Run (SSE) ─────────────────────────────────── */
function handleRun() {
  hideError();
  const query = queryInput.value.trim();
  if (!query) { queryInput.classList.add("shake"); setTimeout(()=>queryInput.classList.remove("shake"),500); return; }
  setLoading(true); resetAllCards(); hideResults(); startTimer();
  const url = API_URL+"/query/stream?q="+encodeURIComponent(query)+"&mock="+mockToggle.checked;
  let evtSource;
  try { evtSource = new EventSource(url); } catch(e) { fallbackFetch(query,mockToggle.checked); return; }
  evtSource.onmessage = function(event) {
    try {
      const data = JSON.parse(event.data);
      if (data.type==="agent_update") setCardState(data.agent,data.status,data.detail||"");
      else if (data.type==="result") { evtSource.close(); showResult(data); }
      else if (data.type==="error") { evtSource.close(); stopTimer(); showError(data.message||"Unknown error"); resetAllCards(); setLoading(false); }
    } catch(e){}
  };
  evtSource.onerror = function() { evtSource.close(); stopTimer(); showError("Stream connection failed"); resetAllCards(); setLoading(false); };
}

function fallbackFetch(query,mock) {
  fetch(API_URL+"/query",{ method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({query,mock}) })
  .then(r => { if(!r.ok) return r.json().then(d=>{throw new Error(d.detail||"HTTP "+r.status);}); return r.json(); })
  .then(data => showResult(data))
  .catch(err => { stopTimer(); showError(err.message||"Network error"); resetAllCards(); setLoading(false); });
}

queryInput.addEventListener("keydown", e => { if ((e.ctrlKey||e.metaKey)&&e.key==="Enter") { e.preventDefault(); handleRun(); } });
