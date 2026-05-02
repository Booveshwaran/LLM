/* ═══════════════════════════════════════════════════
   Canvas Systems: Preloader, Background Particles,
   Agent Waveforms, Custom Cursor
   ═══════════════════════════════════════════════════ */

const COLORS={planner:'#00FFD1',researcher:'#3B82F6',critic:'#FF6B6B',refiner:'#F59E0B',solver:'#8B5CF6'};

/* ── Preloader Canvas ─────────────────────────────── */
function initPreloader(canvas){
  const ctx=canvas.getContext('2d'),W=canvas.width=window.innerWidth,H=canvas.height=window.innerHeight;
  const cx=W/2,cy=H/2,particles=[];
  for(let i=0;i<50;i++) particles.push({a:Math.random()*Math.PI*2,r:80+Math.random()*140,s:1+Math.random()*2,o:.2+Math.random()*.4,v:.01+Math.random()*.02});
  let phase=0,t0=Date.now(),raf;
  function draw(){
    const t=(Date.now()-t0)/1000;phase=t;
    ctx.clearRect(0,0,W,H);
    // Particles orbiting
    ctx.globalAlpha=1;
    particles.forEach(p=>{
      p.a+=p.v;
      const x=cx+Math.cos(p.a)*p.r,y=cy+Math.sin(p.a)*p.r;
      ctx.beginPath();ctx.arc(x,y,p.s,0,Math.PI*2);
      ctx.fillStyle=`rgba(0,255,209,${p.o*(.5+.5*Math.sin(t+p.a))})`;ctx.fill();
    });
    // Central nucleus
    const pulse=1+.3*Math.sin(t*2);
    const grad=ctx.createRadialGradient(cx,cy,0,cx,cy,30*pulse);
    grad.addColorStop(0,'rgba(0,255,209,.9)');grad.addColorStop(.5,'rgba(0,255,209,.2)');grad.addColorStop(1,'transparent');
    ctx.beginPath();ctx.arc(cx,cy,30*pulse,0,Math.PI*2);ctx.fillStyle=grad;ctx.fill();
    // Sonar rings
    for(let i=0;i<3;i++){
      const rPhase=(t*.4+i*.33)%1;
      ctx.beginPath();ctx.arc(cx,cy,20+rPhase*80,0,Math.PI*2);
      ctx.strokeStyle=`rgba(0,255,209,${(1-rPhase)*.5})`;ctx.lineWidth=1.5;ctx.stroke();
    }
    // Hex structure
    if(t>.6){
      const hexR=Math.min(60,(t-.6)*150);
      ctx.beginPath();
      for(let i=0;i<6;i++){
        const a=Math.PI/3*i-Math.PI/6;
        ctx[i?'lineTo':'moveTo'](cx+Math.cos(a)*hexR,cy+Math.sin(a)*hexR);
      }
      ctx.closePath();ctx.strokeStyle=`rgba(0,255,209,${Math.min(.5,(t-.6))})`;ctx.lineWidth=1;ctx.stroke();
      // Inner lines
      if(t>1){
        const io=Math.min(1,(t-1)*2);
        for(let i=0;i<6;i++){
          const a=Math.PI/3*i-Math.PI/6;
          ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(cx+Math.cos(a)*hexR*io,cy+Math.sin(a)*hexR*io);
          ctx.strokeStyle=`rgba(0,255,209,${io*.3})`;ctx.stroke();
        }
      }
    }
    raf=requestAnimationFrame(draw);
  }
  draw();
  return ()=>{cancelAnimationFrame(raf)};
}

/* ── Background Particles ─────────────────────────── */
function initBackground(canvas){
  const ctx=canvas.getContext('2d');
  let W,H;
  const colors=['rgba(0,255,209,','rgba(0,128,255,','rgba(139,92,246,','rgba(255,255,255,'];
  const weights=[.6,.25,.1,.05];
  let particles=[];
  function pickColor(){let r=Math.random(),s=0;for(let i=0;i<weights.length;i++){s+=weights[i];if(r<s)return colors[i];}return colors[0];}
  function resize(){W=canvas.width=window.innerWidth;H=canvas.height=window.innerHeight;initParticles();}
  function initParticles(){
    const count=window.innerWidth<768?150:600;
    particles=[];
    for(let i=0;i<count;i++) particles.push({
      x:Math.random()*W,y:Math.random()*H,
      z:.1+Math.random()*.9,
      phase:Math.random()*Math.PI*2,
      speed:.2+Math.random()*.8,
      color:pickColor()
    });
  }
  resize();
  window.addEventListener('resize',resize);
  let last=0;
  function draw(now){
    const dt=Math.min((now-last)/1000,.05);last=now;
    ctx.clearRect(0,0,W,H);
    const t=now/1000;
    particles.forEach(p=>{
      p.y-=p.speed*dt*18;
      p.x+=Math.sin(t*.5+p.phase)*.3;
      if(p.y<-10){p.y=H+10;p.x=Math.random()*W;}
      const sz=p.z*1.8+.5;
      const op=p.z*.5;
      ctx.beginPath();ctx.arc(p.x,p.y,sz,0,Math.PI*2);
      ctx.fillStyle=p.color+op+')';
      ctx.shadowBlur=sz*4;ctx.shadowColor=p.color+'0.4)';
      ctx.fill();
    });
    ctx.shadowBlur=0;
    requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);
}

/* ── Agent Waveforms ──────────────────────────────── */
function initWaveform(canvas,agent,getState){
  const ctx=canvas.getContext('2d');
  const W=canvas.width=canvas.parentElement.offsetWidth;
  const H=canvas.height=50;
  const color=COLORS[agent]||COLORS.planner;
  function draw(now){
    const t=now/1000;
    const state=getState(agent);
    ctx.clearRect(0,0,W,H);
    const amp=state==='running'?18:state==='done'?0:6;
    const freq=state==='running'?3:1.5;
    const speed=state==='running'?3:state==='done'?0:.8;
    ctx.beginPath();ctx.moveTo(0,H);
    for(let x=0;x<=W;x++){
      let y=H/2;
      if(state==='done'){
        // Crystalline sawtooth
        y=H/2+(((x*7+t*20)%20)-10)*.6;
      } else {
        y=H/2+Math.sin(x*freq/W*Math.PI*2+t*speed)*amp;
        if(agent==='researcher') y+=Math.sin(x*freq*1.5/W*Math.PI*2-t*speed*.7)*amp*.4;
        if(agent==='critic') y=H/2+((Math.sin(x*freq/W*Math.PI*2+t*speed)>0?1:-1)*amp*.7);
        if(agent==='refiner'){const p=x/W;y=H/2+(p<.5?((x*3+t*40)%16-8)*.8:Math.sin(x*freq/W*Math.PI*2+t*speed)*amp*.7);}
      }
      ctx.lineTo(x,y);
    }
    ctx.lineTo(W,H);ctx.closePath();
    const grad=ctx.createLinearGradient(0,0,0,H);
    const op=state==='running'?.5:state==='done'?.25:.15;
    grad.addColorStop(0,color.replace(')',`,${op})`).replace('#','')); // fallback
    grad.addColorStop(0,`${color}${Math.round(op*255).toString(16).padStart(2,'0')}`);
    grad.addColorStop(1,'transparent');
    ctx.fillStyle=grad;ctx.fill();
    ctx.beginPath();ctx.moveTo(0,H/2);
    for(let x=0;x<=W;x++){
      let y=H/2;
      if(state==='done') y=H/2+(((x*7+t*20)%20)-10)*.6;
      else{
        y=H/2+Math.sin(x*freq/W*Math.PI*2+t*speed)*amp;
        if(agent==='researcher') y+=Math.sin(x*freq*1.5/W*Math.PI*2-t*speed*.7)*amp*.4;
        if(agent==='critic') y=H/2+((Math.sin(x*freq/W*Math.PI*2+t*speed)>0?1:-1)*amp*.7);
        if(agent==='refiner'){const p=x/W;y=H/2+(p<.5?((x*3+t*40)%16-8)*.8:Math.sin(x*freq/W*Math.PI*2+t*speed)*amp*.7);}
      }
      ctx.lineTo(x,y);
    }
    ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.globalAlpha=state==='skipped'?.15:state==='done'?.4:.7;ctx.stroke();ctx.globalAlpha=1;
    if(state!=='skipped') requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);
  return{restart:()=>requestAnimationFrame(draw)};
}

/* ── Custom Cursor ────────────────────────────────── */
function initCursor(){
  if(!matchMedia('(hover:hover)').matches)return;
  const dot=document.querySelector('.cursor-dot'),ring=document.querySelector('.cursor-ring');
  if(!dot||!ring)return;
  let mx=0,my=0,rx=0,ry=0;
  document.addEventListener('mousemove',e=>{mx=e.clientX;my=e.clientY;dot.style.left=mx+'px';dot.style.top=my+'px';});
  (function lerp(){rx+=(mx-rx)*.1;ry+=(my-ry)*.1;ring.style.left=rx+'px';ring.style.top=ry+'px';requestAnimationFrame(lerp);})();
  document.querySelectorAll('button,a,.agent-card,.toggle-label,.nav-pill').forEach(el=>{
    el.addEventListener('mouseenter',()=>ring.classList.add('hover'));
    el.addEventListener('mouseleave',()=>ring.classList.remove('hover'));
  });
  // Magnetic effect on buttons
  document.querySelectorAll('#run-btn,.nav-pill').forEach(btn=>{
    btn.addEventListener('mousemove',e=>{
      const r=btn.getBoundingClientRect();
      const dx=(e.clientX-r.left-r.width/2)*.12;
      const dy=(e.clientY-r.top-r.height/2)*.12;
      btn.style.transform=`translate(${dx}px,${dy}px)`;
    });
    btn.addEventListener('mouseleave',()=>{btn.style.transform='';});
  });
}

/* ── Scroll Nav Hide ──────────────────────────────── */
function initNav(){
  const nav=document.querySelector('.nav');
  let lastY=0;
  window.addEventListener('scroll',()=>{
    const y=window.scrollY;
    nav.classList.toggle('hidden',y>lastY&&y>80);
    lastY=y;
  },{passive:true});
}

window._canvas={initPreloader,initBackground,initWaveform,initCursor,initNav,COLORS};
