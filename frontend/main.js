/* ═══════════════════════════════════════════════════
   HealthcareAI — Lusion-style Main JS
   Three.js · GSAP · SSE · Custom Cursor
   ═══════════════════════════════════════════════════ */
const API='http://localhost:8000';
const $=id=>document.getElementById(id);
const AGENTS=['planner','researcher','critic','refiner','solver'];
let timerInt=null,t0=0,sparkData=[];

/* ── Three.js WebGL Background ────────────────────── */
(function initWebGL(){
  const canvas=$('webgl-canvas');
  if(window.innerWidth<768||!window.THREE){canvas.style.background='radial-gradient(ellipse at 30% 50%,#061a2e,#020810)';return;}
  const scene=new THREE.Scene();
  const camera=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,.1,1000);
  camera.position.z=300;
  const renderer=new THREE.WebGLRenderer({canvas,alpha:true,antialias:true});
  renderer.setSize(innerWidth,innerHeight);
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  // Particle cloud
  const count=3000,positions=new Float32Array(count*3),colors=new Float32Array(count*3);
  const palette=[[0,.83,1],[0,1,.62],[.55,.36,.96]];
  for(let i=0;i<count;i++){
    const i3=i*3,r=150+Math.random()*200,theta=Math.random()*Math.PI*2,phi=Math.acos(2*Math.random()-1);
    positions[i3]=r*Math.sin(phi)*Math.cos(theta);
    positions[i3+1]=r*Math.sin(phi)*Math.sin(theta);
    positions[i3+2]=r*Math.cos(phi);
    const c=palette[Math.floor(Math.random()*3)];
    colors[i3]=c[0];colors[i3+1]=c[1];colors[i3+2]=c[2];
  }
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position',new THREE.BufferAttribute(positions,3));
  geo.setAttribute('color',new THREE.BufferAttribute(colors,3));
  const mat=new THREE.PointsMaterial({size:1.8,vertexColors:true,transparent:true,opacity:.6,blending:THREE.AdditiveBlending,depthWrite:false});
  const points=new THREE.Points(geo,mat);
  scene.add(points);
  // Central mesh
  const icoGeo=new THREE.IcosahedronGeometry(40,2);
  const icoMat=new THREE.MeshBasicMaterial({color:0x00d4ff,wireframe:true,transparent:true,opacity:.08});
  const ico=new THREE.Mesh(icoGeo,icoMat);
  scene.add(ico);
  let mx=0,my=0;
  document.addEventListener('mousemove',e=>{mx=(e.clientX/innerWidth-.5)*2;my=(e.clientY/innerHeight-.5)*2;});
  window.addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
  (function animate(){
    requestAnimationFrame(animate);
    const t=Date.now()*.0003;
    points.rotation.y=t*.3+mx*.15;points.rotation.x=t*.15+my*.1;
    ico.rotation.y=t*.5;ico.rotation.x=Math.sin(t)*.3;
    camera.position.x+=(mx*30-camera.position.x)*.02;
    camera.position.y+=(-my*20-camera.position.y)*.02;
    camera.lookAt(0,0,0);
    renderer.render(scene,camera);
  })();
})();

/* ── Custom Cursor ────────────────────────────────── */
(function initCursor(){
  if(!matchMedia('(hover:hover)').matches)return;
  const dot=$('cursor-dot'),ring=$('cursor-ring');
  if(!dot||!ring)return;
  let mx=-100,my=-100,rx=-100,ry=-100;
  // Move dot instantly, ring follows with lerp
  document.addEventListener('mousemove',e=>{
    mx=e.clientX;my=e.clientY;
    dot.style.left=mx+'px';dot.style.top=my+'px';
  });
  // Smooth ring follow
  (function lp(){
    rx+=(mx-rx)*.12;ry+=(my-ry)*.12;
    ring.style.left=rx+'px';ring.style.top=ry+'px';
    requestAnimationFrame(lp);
  })();
  // Click animation — dot shrinks then springs back
  document.addEventListener('mousedown',()=>dot.classList.add('click'));
  document.addEventListener('mouseup',()=>dot.classList.remove('click'));
  // Hover expansion on interactive elements
  document.querySelectorAll('a,button,.agent-card,.spec-tile,.toggle,.stats-refresh,.toggle-wrap').forEach(el=>{
    el.addEventListener('mouseenter',()=>ring.classList.add('hover'));
    el.addEventListener('mouseleave',()=>ring.classList.remove('hover'));
  });
  // Magnetic pull on buttons
  document.querySelectorAll('#analyze-btn,.btn-primary,.btn-ghost,.stats-refresh').forEach(btn=>{
    btn.addEventListener('mousemove',e=>{
      const r=btn.getBoundingClientRect();
      const dx=(e.clientX-r.left-r.width/2)*.15;
      const dy=(e.clientY-r.top-r.height/2)*.15;
      btn.style.transform=`translate(${dx}px,${dy}px)`;
    });
    btn.addEventListener('mouseleave',()=>{btn.style.transform='';});
  });
})();

/* ── GSAP Hero Animation ──────────────────────────── */
window.addEventListener('load',()=>{
  if(typeof gsap==='undefined')return;
  const tl=gsap.timeline({defaults:{ease:'power3.out'}});
  tl.to('.hero-eyebrow',{opacity:1,duration:.6,delay:.3})
    .to('.hero h1 .line span',{y:0,duration:.8,stagger:.12},'-=.3')
    .to('.hero-sub',{opacity:1,duration:.6},'-=.4')
    .to('.hero-btns',{opacity:1,y:0,duration:.5},'-=.3')
    .to('.scroll-indicator',{opacity:1,duration:.5},'-=.2');
  // Scroll-triggered reveals
  const io=new IntersectionObserver((entries)=>{
    entries.forEach(en=>{if(en.isIntersecting){
      const el=en.target;
      gsap.to(el,{opacity:1,y:0,duration:.6,ease:'power3.out',stagger:.08,
        onComplete:()=>io.unobserve(el)});
    }});
  },{threshold:.15});
  document.querySelectorAll('.query-card,.agent-card,.stat-card,.spec-tile,.specialties-section h2,.arch-section h2,#arch-svg').forEach(el=>io.observe(el));
  // Nav hide on scroll
  let lastY=0;const nav=document.querySelector('.hero-eyebrow');
  // Animate arch paths
  document.querySelectorAll('.arch-path').forEach(p=>{
    const len=p.getTotalLength?p.getTotalLength():100;
    p.style.strokeDasharray=len;p.style.strokeDashoffset=len;
    const aio=new IntersectionObserver(e=>{if(e[0].isIntersecting){
      gsap.to(p,{strokeDashoffset:0,duration:1.5,ease:'power2.out'});aio.unobserve(p);}});
    aio.observe(p);
  });
});

/* ── Specialties Grid ─────────────────────────────── */
(function buildSpecialties(){
  const specs=[
    {icon:'❤️',name:'Cardiology',count:6},{icon:'🫁',name:'Respiratory',count:5},
    {icon:'🧪',name:'Endocrinology',count:6},{icon:'🧠',name:'Neurology',count:5},
    {icon:'🫃',name:'Gastroenterology',count:5},{icon:'🫘',name:'Nephrology',count:3},
    {icon:'🦠',name:'Infectious Disease',count:5},{icon:'🎗️',name:'Oncology',count:4},
    {icon:'👶',name:'Pediatrics',count:4},{icon:'🚑',name:'Emergency',count:4},
    {icon:'🧘',name:'Psychiatry',count:5},{icon:'🦴',name:'Orthopedics',count:4},
    {icon:'🩹',name:'Dermatology',count:4},{icon:'🤰',name:'OB/GYN',count:4},
    {icon:'🩸',name:'Hematology',count:4},{icon:'💊',name:'Pharmacology',count:5},
    {icon:'🔢',name:'Lab Values',count:8},{icon:'📋',name:'Clinical Reasoning',count:8}
  ];
  const grid=$('spec-grid');
  specs.forEach(s=>{
    const d=document.createElement('div');d.className='spec-tile glass';
    d.innerHTML=`<div class="spec-icon">${s.icon}</div><div class="spec-name">${s.name}</div><div class="spec-count">${s.count} docs</div>`;
    grid.appendChild(d);
  });
})();

/* ── Agent Card State ─────────────────────────────── */
function setAgent(agent,status,detail){
  const card=$('ac-'+agent),badge=$('ab-'+agent),meta=$('am-'+agent);
  card.classList.remove('active','done');badge.className='badge';
  if(status==='running'||status==='thinking'){
    badge.className='badge badge-thinking';badge.textContent='THINKING';card.classList.add('active');meta.textContent='';
  }else if(status==='done'){
    badge.className='badge badge-done';badge.textContent='DONE';card.classList.add('done');
    if(detail)typeInto(meta,detail.length>60?detail.slice(0,60)+'…':detail,12);
  }else if(status==='rejected'){
    badge.className='badge badge-rejected';badge.textContent='REJECTED';
    if(detail)meta.textContent=detail;
  }else if(status==='skipped'){
    badge.className='badge badge-pending';badge.textContent='SKIPPED';card.style.opacity='.35';meta.textContent='Not needed';
  }else{
    badge.className='badge badge-pending';badge.textContent='PENDING';meta.textContent='';
  }
}
function resetAgents(){AGENTS.forEach(a=>{setAgent(a,'pending');$('ac-'+a).style.opacity='1';});}

/* ── Typewriter ───────────────────────────────────── */
function typeInto(el,text,speed){el.textContent='';let i=0;(function t(){if(i<text.length){el.textContent+=text[i++];setTimeout(t,speed);}})();}

/* ── Timer ─────────────────────────────────────────── */
function startTimer(){t0=Date.now();$('timer').classList.add('on');timerInt=setInterval(()=>{$('timer').textContent=((Date.now()-t0)/1000).toFixed(1)+'s';},100);}
function stopTimer(){if(timerInt){clearInterval(timerInt);timerInt=null;}$('timer').textContent=((Date.now()-t0)/1000).toFixed(1)+'s';}

/* ── Show Answer ───────────────────────────────────── */
function showAnswer(data){
  stopTimer();
  let answer=data.final_answer||'';
  try{const p=JSON.parse(answer);if(p&&p.answer)answer=p.answer;}catch(e){}
  // Dedupe disclaimers
  const dp=answer.split(/⚕️\s*Disclaimer:/i);
  if(dp.length>2)answer=dp[0]+'⚕️ Disclaimer:'+dp[1];
  const el=$('answer-text');el.innerHTML='';
  const mainText=dp[0]||answer;
  // Typewriter
  let i=0;
  (function t(){
    if(i<mainText.length){el.textContent+=mainText[i++];setTimeout(t,8);}
    else{
      const disc=document.createElement('div');disc.className='disclaimer';
      disc.textContent='⚕️ This information is for educational purposes only. Always consult a qualified healthcare professional for medical advice.';
      el.appendChild(disc);
    }
  })();
  $('answer-card').classList.add('visible');
  // Map trace to agents
  const trace=data.reasoning_trace||[];
  const map={};let refUsed=false;
  trace.forEach(e=>{const l=e.toLowerCase();
    if(l.includes('planner'))map.planner=e;
    else if(l.includes('researcher'))map.researcher=e;
    else if(l.includes('critic')||l.includes('reviewer'))map.critic=e;
    else if(l.includes('refiner')){map.refiner=e;refUsed=true;}
    else if(l.includes('solver')||l.includes('advisor'))map.solver=e;
  });
  AGENTS.forEach(a=>{
    if(map[a])setAgent(a,'done',map[a]);
    else if(a==='refiner'&&!refUsed)setAgent(a,'skipped');
    else setAgent(a,'done','Completed');
  });
  setLoading(false);
  // Refresh CAG stats after every query
  setTimeout(fetchStats, 500);
}

/* ── Loading ───────────────────────────────────────── */
function setLoading(on){
  const btn=$('analyze-btn');btn.disabled=on;btn.classList.toggle('loading',on);
  btn.querySelector('.label').textContent=on?'Analyzing…':'Analyze';
  btn.querySelector('.spinner').style.display=on?'inline-block':'none';
}

/* ── Main Run ──────────────────────────────────────── */
function handleRun(){
  const q=$('query-input').value.trim();
  if(!q){$('query-input').style.animation='none';void $('query-input').offsetWidth;$('query-input').style.animation='shake .4s ease';return;}
  $('query-input').style.animation='none';
  setLoading(true);resetAgents();$('answer-card').classList.remove('visible');startTimer();
  const mock=$('mock-toggle').checked;
  const stream=$('stream-toggle').checked;
  if(stream){
    const url=API+'/query/stream?q='+encodeURIComponent(q)+'&mock='+mock;
    let es;
    try{es=new EventSource(url);}catch(e){doFetch(q,mock);return;}
    es.onmessage=function(ev){
      try{
        const d=JSON.parse(ev.data);
        if(d.type==='agent_update')setAgent(d.agent,d.status,d.detail||'');
        else if(d.type==='result'){es.close();showAnswer(d);}
        else if(d.type==='error'){es.close();stopTimer();showErr(d.message);resetAgents();setLoading(false);}
      }catch(e){}
    };
    es.onerror=function(){es.close();stopTimer();showErr('SSE stream failed — is the API running?');resetAgents();setLoading(false);};
  }else{doFetch(q,mock);}
}
function doFetch(q,mock){
  fetch(API+'/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q,mock:mock})})
  .then(r=>{if(!r.ok)return r.json().then(d=>{throw new Error(d.detail||r.status);});return r.json();})
  .then(d=>showAnswer(d))
  .catch(e=>{stopTimer();showErr(e.message);resetAgents();setLoading(false);});
}
function showErr(m){$('error-msg').textContent=m;$('error-bar').classList.add('on');}

/* ── CAG Stats ─────────────────────────────────────── */
function fetchStats(){
  fetch(API+'/cag-stats').then(r=>r.json()).then(d=>{
    // API returns cag_-prefixed keys: cag_hit_rate, cag_cache_size, cag_hits, etc.
    const hitRate = d.cag_hit_rate !== undefined ? d.cag_hit_rate : (d.hit_rate || 0);
    const hr = (hitRate * 100).toFixed(1) + '%';
    $('sv-hitrate').textContent = hr;
    $('sv-entries').textContent = d.cag_cache_size || d.cache_size || '0';
    // Show threshold as similarity reference
    const sim = d.similarity_threshold || d.avg_similarity;
    $('sv-similarity').textContent = sim !== undefined ? (sim * 100).toFixed(0) + '%' : '—';
    // Sparkline
    sparkData.push(parseFloat(hr)); if(sparkData.length > 8) sparkData.shift();
    drawSparkline();
  }).catch(()=>{
    // Fallback: try /cache-stats (KV-Cache) if /cag-stats fails
    fetch(API+'/cache-stats').then(r=>r.json()).then(d=>{
      const total = (d.cache_hits||0) + (d.cache_misses||0);
      const hr = total > 0 ? ((d.cache_hits/total)*100).toFixed(1)+'%' : '0%';
      $('sv-hitrate').textContent = hr;
      $('sv-entries').textContent = d.cache_size || '0';
      $('sv-similarity').textContent = '—';
      sparkData.push(parseFloat(hr)); if(sparkData.length > 8) sparkData.shift();
      drawSparkline();
    }).catch(()=>{});
  });
}
function drawSparkline(){
  const svg=$('sparkline');svg.innerHTML='';
  if(sparkData.length<2)return;
  const max=Math.max(...sparkData,1),pts=sparkData.map((v,i)=>`${i*(120/(sparkData.length-1))},${30-v/max*28}`).join(' ');
  svg.innerHTML=`<polyline points="${pts}" fill="none" stroke="var(--primary)" stroke-width="1.5" stroke-linecap="round"/>`;
  sparkData.forEach((v,i)=>{
    svg.innerHTML+=`<circle cx="${i*(120/(sparkData.length-1))}" cy="${30-v/max*28}" r="2.5" fill="var(--primary)"/>`;
  });
}
fetchStats();

/* ── Keyboard shortcut ─────────────────────────────── */
$('query-input').addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){e.preventDefault();handleRun();}});
/* Shake animation for empty input */
const style=document.createElement('style');
style.textContent='@keyframes shake{0%,100%{transform:translateX(0)}20%{transform:translateX(-6px)}40%{transform:translateX(6px)}60%{transform:translateX(-3px)}80%{transform:translateX(3px)}}';
document.head.appendChild(style);
