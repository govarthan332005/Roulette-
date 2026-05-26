/* ============================================================
   ROULETTE AI PREDICTOR — Enhanced Edition
   White & Blue theme • Tap-based keypad • Sound effects • Lag-free
   ============================================================ */

/* ----------- Roulette constants ----------- */
const RED_SET = new Set([1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36]);
const colorOf  = n => n === 0 ? 'green' : (RED_SET.has(n) ? 'red' : 'black');
const parityOf = n => n === 0 ? 'zero'  : (n % 2 === 0 ? 'even' : 'odd');
const dozenOf  = n => n === 0 ? '-'     : (n <= 12 ? '1st' : n <= 24 ? '2nd' : '3rd');
const rangeOf  = n => n === 0 ? '-'     : (n <= 18 ? 'Low' : 'High');

/* ----------- State ----------- */
const STORE = {
  TRAIN: 'rai_train_v1',
  MODEL: 'rai_model_v1',
  HIST:  'rai_hist_v1',
  SOUND: 'rai_sound_v1',
};
let trainData = loadArr(STORE.TRAIN);
let model     = loadObj(STORE.MODEL);
let history   = loadArr(STORE.HIST);
let lastPrediction = null;
let soundOn   = localStorage.getItem(STORE.SOUND) !== '0';
let activeSlotIdx = 0;
let seqValues = new Array(10).fill('');
let trainInputBuffer = '';

function loadArr(k){ try{ return JSON.parse(localStorage.getItem(k)||'[]'); }catch(e){return [];} }
function loadObj(k){ try{ return JSON.parse(localStorage.getItem(k)||'null'); }catch(e){return null;} }
function save(k,v){ try{ localStorage.setItem(k, JSON.stringify(v)); }catch(e){} }

/* ============================================================
   SOUND ENGINE (Web Audio API — zero file load, zero lag)
   ============================================================ */
let audioCtx = null;
function getCtx(){
  if(!audioCtx){
    const AC = window.AudioContext || window.webkitAudioContext;
    if(AC) audioCtx = new AC();
  }
  if(audioCtx && audioCtx.state === 'suspended') audioCtx.resume();
  return audioCtx;
}

function beep(opts){
  if(!soundOn) return;
  const ctx = getCtx(); if(!ctx) return;
  const {freq=600, dur=0.08, type='sine', vol=0.12, slide=null, delay=0} = opts || {};
  const t0 = ctx.currentTime + delay;
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(freq, t0);
  if(slide !== null) osc.frequency.exponentialRampToValueAtTime(Math.max(40, slide), t0 + dur);
  gain.gain.setValueAtTime(0, t0);
  gain.gain.linearRampToValueAtTime(vol, t0 + 0.005);
  gain.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  osc.connect(gain).connect(ctx.destination);
  osc.start(t0);
  osc.stop(t0 + dur + 0.02);
}

const SFX = {
  tap:     () => beep({freq: 720, dur: 0.05, type:'square', vol:0.06}),
  key:     () => beep({freq: 880, dur: 0.04, type:'triangle', vol:0.07}),
  tabSwitch: () => beep({freq: 520, dur: 0.07, type:'sine', vol:0.08, slide: 720}),
  success: () => { beep({freq: 660, dur: 0.10, type:'sine', vol:0.10}); beep({freq: 880, dur: 0.14, type:'sine', vol:0.10, delay:0.10}); },
  predict: () => { beep({freq: 440, dur: 0.08, type:'sine', vol:0.10}); beep({freq: 660, dur: 0.10, type:'sine', vol:0.10, delay:0.08}); beep({freq: 880, dur: 0.16, type:'sine', vol:0.10, delay:0.18}); },
  error:   () => beep({freq: 200, dur: 0.18, type:'sawtooth', vol:0.10, slide: 120}),
  hit:     () => { beep({freq: 660, dur:0.10, vol:0.10}); beep({freq: 880, dur:0.10, vol:0.10, delay:0.10}); beep({freq:1320, dur:0.18, vol:0.10, delay:0.20}); },
  delete:  () => beep({freq: 300, dur: 0.05, type:'square', vol:0.07}),
};

/* Haptic feedback (tactile feel on Android) */
function haptic(ms){
  if(navigator.vibrate) try { navigator.vibrate(ms || 8); } catch(e){}
}

/* ============================================================
   PREDICTION ENGINE (unchanged math, same accuracy)
   ============================================================ */
function buildModel(data){
  const m = {
    n: data.length,
    freq: new Array(37).fill(0),
    m1: Array.from({length:37},()=>new Array(37).fill(0)),
    m2: {},
    m3: {},
    colorTrans:  Array.from({length:3},()=>new Array(3).fill(0)),
    parityTrans: Array.from({length:3},()=>new Array(3).fill(0)),
    dozenTrans:  Array.from({length:4},()=>new Array(4).fill(0)),
    rangeTrans:  Array.from({length:3},()=>new Array(3).fill(0)),
  };
  const cIdx = c => c==='red'?0:c==='black'?1:2;
  const pIdx = p => p==='odd'?0:p==='even'?1:2;
  const dIdx = d => d==='1st'?1:d==='2nd'?2:d==='3rd'?3:0;
  const rIdx = r => r==='Low'?0:r==='High'?1:2;

  for(let i=0;i<data.length;i++){
    const cur = data[i];
    if(cur<0||cur>36) continue;
    m.freq[cur]++;
    if(i>=1){
      const a = data[i-1];
      m.m1[a][cur]++;
      m.colorTrans [cIdx(colorOf(a))]  [cIdx(colorOf(cur))]++;
      m.parityTrans[pIdx(parityOf(a))] [pIdx(parityOf(cur))]++;
      m.dozenTrans [dIdx(dozenOf(a))]  [dIdx(dozenOf(cur))]++;
      m.rangeTrans [rIdx(rangeOf(a))]  [rIdx(rangeOf(cur))]++;
    }
    if(i>=2){
      const key = data[i-2]+','+data[i-1];
      if(!m.m2[key]) m.m2[key] = new Array(37).fill(0);
      m.m2[key][cur]++;
    }
    if(i>=3){
      const key = data[i-3]+','+data[i-2]+','+data[i-1];
      if(!m.m3[key]) m.m3[key] = new Array(37).fill(0);
      m.m3[key][cur]++;
    }
  }
  return m;
}

function normalize(arr){
  const s = arr.reduce((a,b)=>a+b,0);
  if(s===0) return arr.map(_=>1/arr.length);
  return arr.map(x=>x/s);
}

function predictNext(seq, m){
  if(!m || m.n < 2){
    return { dist:new Array(37).fill(1/37), top:Array.from({length:37},(_,i)=>({n:i,p:1/37})).slice(0,5), conf:0 };
  }
  const last1 = seq[seq.length-1];
  const last2 = seq.length>=2 ? seq[seq.length-2]+','+seq[seq.length-1] : null;
  const last3 = seq.length>=3 ? seq[seq.length-3]+','+seq[seq.length-2]+','+seq[seq.length-1] : null;

  const freqDist  = normalize(m.freq.slice());
  const m1Row     = (last1>=0 && last1<=36) ? normalize(m.m1[last1].slice()) : freqDist;
  const m2Row     = (last2 && m.m2[last2])  ? normalize(m.m2[last2].slice()) : null;
  const m3Row     = (last3 && m.m3[last3])  ? normalize(m.m3[last3].slice()) : null;

  let w1=0.30, w2=0.25, w3=0.20, wF=0.10, wC=0.15;
  if(!m2Row){ w1+=w2; w2=0; }
  if(!m3Row){ w1+=w3; w3=0; }

  const cIdx = c => c==='red'?0:c==='black'?1:2;
  const pIdx = p => p==='odd'?0:p==='even'?1:2;
  const dIdx = d => d==='1st'?1:d==='2nd'?2:d==='3rd'?3:0;
  const rIdx = r => r==='Low'?0:r==='High'?1:2;

  const colorRow  = normalize(m.colorTrans [cIdx(colorOf (last1))]);
  const parityRow = normalize(m.parityTrans[pIdx(parityOf(last1))]);
  const dozenRow  = normalize(m.dozenTrans [dIdx(dozenOf (last1))]);
  const rangeRow  = normalize(m.rangeTrans [rIdx(rangeOf (last1))]);

  const dist = new Array(37).fill(0);
  for(let n=0;n<=36;n++){
    let p = w1*(m1Row[n]||0)
          + (m2Row?w2*m2Row[n]:0)
          + (m3Row?w3*m3Row[n]:0)
          + wF*freqDist[n];
    const cBoost = colorRow [cIdx(colorOf (n))];
    const pBoost = parityRow[pIdx(parityOf(n))];
    const dBoost = dozenRow [dIdx(dozenOf (n))];
    const rBoost = rangeRow [rIdx(rangeOf (n))];
    const patternMix = (cBoost+pBoost+dBoost+rBoost)/4;
    p += wC * patternMix;
    dist[n] = p;
  }
  for(let i=Math.max(0,seq.length-6); i<seq.length; i++){
    const n = seq[i];
    if(n>=0 && n<=36) dist[n] *= 1.03;
  }
  const final = normalize(dist);
  const ranked = final.map((p,n)=>({n,p})).sort((a,b)=>b.p-a.p);
  const top = ranked.slice(0,5);
  const conf = Math.min(99, Math.round(ranked[0].p*100*4));
  return { dist:final, top, conf };
}

/* ============================================================
   RIPPLE EFFECT (Material-style, lightweight)
   ============================================================ */
function attachRipples(){
  document.addEventListener('pointerdown', e=>{
    const el = e.target.closest('.ripple');
    if(!el) return;
    const rect = el.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const ink = document.createElement('span');
    ink.className = 'ripple-ink';
    ink.style.width = ink.style.height = size + 'px';
    ink.style.left = (e.clientX - rect.left - size/2) + 'px';
    ink.style.top  = (e.clientY - rect.top  - size/2) + 'px';
    el.appendChild(ink);
    setTimeout(()=>ink.remove(), 500);
  }, {passive:true});
}

/* ============================================================
   SPLASH
   ============================================================ */
window.addEventListener('load', () => {
  setTimeout(()=>document.getElementById('splash').classList.add('hide'), 1200);
});

/* ============================================================
   TABS
   ============================================================ */
document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    SFX.tabSwitch(); haptic(6);
    document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-'+btn.dataset.tab).classList.add('active');
    if(btn.dataset.tab==='history') renderHistory();
    if(btn.dataset.tab==='train')   renderStats();
  });
});

/* ============================================================
   MENU
   ============================================================ */
const menu = document.getElementById('menu');
document.getElementById('menuBtn').addEventListener('click', e=>{
  e.stopPropagation();
  SFX.tap(); haptic(5);
  menu.classList.toggle('show');
});
document.addEventListener('click', ()=>menu.classList.remove('show'));
menu.addEventListener('click', e=>{
  const a = e.target.dataset.action;
  if(!a) return;
  SFX.tap();
  menu.classList.remove('show');
  if(a==='export') exportModel();
  if(a==='import') importModel();
  if(a==='reset')  confirmModal('Reset all data?','This will erase training data, model and history.', resetAll);
  if(a==='about')  infoModal('About Roulette AI',
     'A modern roulette prediction tool that learns from your uploaded training data using a hybrid Markov + pattern engine. Predictions are entirely on-device and based ONLY on your training data. Use responsibly — no model can guarantee outcomes of a fair random game.');
});

/* ============================================================
   SOUND TOGGLE
   ============================================================ */
const soundBtn = document.getElementById('soundBtn');
function updateSoundBtn(){
  soundBtn.textContent = soundOn ? '🔊' : '🔇';
  soundBtn.classList.toggle('muted', !soundOn);
}
updateSoundBtn();
soundBtn.addEventListener('click', e=>{
  e.stopPropagation();
  soundOn = !soundOn;
  localStorage.setItem(STORE.SOUND, soundOn ? '1' : '0');
  updateSoundBtn();
  if(soundOn){ SFX.success(); haptic(10); }
});

/* ============================================================
   SEQUENCE GRID (tap-to-select slot)
   ============================================================ */
const seqGrid = document.getElementById('seqGrid');
const seqCells = [];
for(let i=0;i<10;i++){
  const cell = document.createElement('div');
  cell.className = 'seq-cell';
  cell.dataset.i = i;
  cell.innerHTML = `<div class="idx">R-${10-i}</div><div class="val empty">--</div>`;
  cell.addEventListener('click', ()=>{
    setActiveSlot(i);
    showKeypad(true);
    SFX.tap(); haptic(5);
  });
  seqGrid.appendChild(cell);
  seqCells.push(cell);
}

function setActiveSlot(i){
  if(i < 0) i = 0;
  if(i > 9) i = 9;
  activeSlotIdx = i;
  seqCells.forEach((c,k)=>c.classList.toggle('active', k===i));
  const lbl = document.getElementById('activeSlotLbl');
  if(lbl) lbl.textContent = 'R-' + (10 - i);
}

function setSlotValue(i, val){
  seqValues[i] = val;
  const valEl = seqCells[i].querySelector('.val');
  if(val === '' || val === null || val === undefined){
    valEl.textContent = '--';
    valEl.className = 'val empty';
  } else {
    const n = +val;
    valEl.textContent = val;
    valEl.className = 'val ' + colorOf(n);
  }
}

function nextSlot(){
  if(activeSlotIdx < 9){
    setActiveSlot(activeSlotIdx + 1);
  } else {
    showKeypad(false);
  }
}

/* ============================================================
   KEYPAD (the speed-input system)
   ============================================================ */
const keypadGrid = document.getElementById('keypadGrid');
const keypad = document.getElementById('keypad');

function buildKeypad(target, onPress){
  target.innerHTML = '';
  // 0 (green, full width)
  const zero = document.createElement('button');
  zero.className = 'key green ripple';
  zero.textContent = '0';
  zero.type = 'button';
  zero.addEventListener('click', ()=>onPress(0));
  target.appendChild(zero);
  // 1-36
  for(let n=1; n<=36; n++){
    const k = document.createElement('button');
    k.className = 'key ripple ' + colorOf(n);
    k.textContent = n;
    k.type = 'button';
    k.addEventListener('click', ()=>onPress(n));
    target.appendChild(k);
  }
}

function showKeypad(show){
  keypad.classList.toggle('show', show);
}

buildKeypad(keypadGrid, (n)=>{
  setSlotValue(activeSlotIdx, String(n));
  SFX.key(); haptic(4);
  // auto-advance to next slot
  setTimeout(()=>nextSlot(), 60);
});

document.getElementById('kpBack').addEventListener('click', ()=>{
  SFX.delete(); haptic(5);
  if(seqValues[activeSlotIdx] !== ''){
    setSlotValue(activeSlotIdx, '');
  } else if(activeSlotIdx > 0){
    setActiveSlot(activeSlotIdx - 1);
    setSlotValue(activeSlotIdx, '');
  }
});

document.getElementById('kpClear').addEventListener('click', ()=>{
  SFX.delete(); haptic(8);
  for(let i=0;i<10;i++) setSlotValue(i, '');
  setActiveSlot(0);
});

document.getElementById('kpNext').addEventListener('click', ()=>{
  SFX.tap(); haptic(5);
  nextSlot();
});

document.getElementById('closeKeypad').addEventListener('click', ()=>{
  SFX.tap(); haptic(4);
  showKeypad(false);
});

document.getElementById('toggleKeypad').addEventListener('click', ()=>{
  SFX.tap(); haptic(5);
  showKeypad(!keypad.classList.contains('show'));
});

document.getElementById('clearSeq').addEventListener('click', ()=>{
  SFX.delete(); haptic(8);
  for(let i=0;i<10;i++) setSlotValue(i, '');
  setActiveSlot(0);
});

document.getElementById('fillRecent').addEventListener('click', ()=>{
  const recent = trainData.slice(-10);
  if(recent.length<1){ SFX.error(); return toast('No training data yet'); }
  SFX.success(); haptic(12);
  for(let i=0;i<10;i++) setSlotValue(i, '');
  const start = 10 - recent.length;
  recent.forEach((n,k)=> setSlotValue(start+k, String(n)));
  setActiveSlot(Math.min(9, start + recent.length));
  toast('Filled with recent history');
});

// Start with first slot active
setActiveSlot(0);
showKeypad(true);

/* ============================================================
   PREDICT
   ============================================================ */
document.getElementById('predictBtn').addEventListener('click', ()=>{
  const seq = seqValues.filter(v=>v!=='').map(v=>+v);
  if(seq.length<3){ SFX.error(); haptic(20); return toast('Enter at least 3 recent numbers'); }
  if(seq.some(v=>v<0||v>36||isNaN(v))){ SFX.error(); return toast('Numbers must be 0–36'); }

  let m = model;
  if(!m || m.n !== trainData.length){
    if(trainData.length < 5){ SFX.error(); return toast('Need at least 5 training rounds. Go to Train tab.'); }
    m = buildModel(trainData);
    model = m; save(STORE.MODEL, m);
  }
  SFX.predict(); haptic(18);
  const res = predictNext(seq, m);
  // Use requestAnimationFrame for smooth UI update
  requestAnimationFrame(()=>showResult(res, seq));
});

function showResult(res, seq){
  const card = document.getElementById('resultCard');
  card.classList.remove('hidden');
  const top1 = res.top[0];
  const n = top1.n;
  const col = colorOf(n);

  const numEl = document.getElementById('predNumber');
  numEl.textContent = n;
  numEl.className = 'pred-number ' + col;
  document.getElementById('predColorTag').textContent = col.toUpperCase();

  document.getElementById('predConf').textContent  = res.conf+'%';
  document.getElementById('predColor').textContent = col.toUpperCase();
  document.getElementById('predParity').textContent= parityOf(n).toUpperCase();
  document.getElementById('predDozen').textContent = dozenOf(n);
  document.getElementById('predRange').textContent = rangeOf(n);

  // Animate fill bar
  const fill = document.getElementById('confFill');
  fill.style.width = '0%';
  requestAnimationFrame(()=>{
    setTimeout(()=>{ fill.style.width = res.conf + '%'; }, 50);
  });

  const top5 = document.getElementById('top5');
  top5.innerHTML = res.top.map(t => {
    const c = colorOf(t.n);
    return `<div class="t5">
      <div class="t5-num ${c}">${t.n}</div>
      <div class="t5-prob">${(t.p*100).toFixed(1)}%</div>
    </div>`;
  }).join('');

  lastPrediction = {
    ts: Date.now(),
    seq,
    predicted: n,
    top5: res.top.map(t=>t.n),
    conf: res.conf,
    color: col, parity: parityOf(n), dozen: dozenOf(n), range: rangeOf(n),
    actual: null
  };
  setTimeout(()=>card.scrollIntoView({behavior:'smooth', block:'nearest'}), 100);
}

/* ============================================================
   RECORD ACTUAL — uses keypad modal instead of prompt
   ============================================================ */
document.getElementById('saveActualBtn').addEventListener('click', ()=>{
  if(!lastPrediction){ SFX.error(); return toast('Make a prediction first'); }
  SFX.tap();
  keypadModal('Record Actual Result','Tap the actual winning number:', val=>{
    const n = parseInt(val,10);
    if(isNaN(n)||n<0||n>36){ SFX.error(); return toast('Invalid number'); }
    lastPrediction.actual = n;
    // Check if hit
    if(n === lastPrediction.predicted){ SFX.hit(); haptic([20,40,30]); }
    else { SFX.success(); haptic(12); }
    history.unshift(lastPrediction);
    save(STORE.HIST, history);
    trainData.push(n); save(STORE.TRAIN, trainData);
    model = buildModel(trainData); save(STORE.MODEL, model);
    updateModelStatus();
    toast(n === lastPrediction.predicted ? '🎉 Exact hit! Model updated' : 'Result saved & model retrained');
    lastPrediction = null;
    renderHistory();
  });
});

/* ============================================================
   TRAIN TAB
   ============================================================ */
document.getElementById('uploadZone').addEventListener('click', ()=>{
  SFX.tap(); haptic(5);
  document.getElementById('fileInput').click();
});
document.getElementById('fileInput').addEventListener('change', e=>{
  const f = e.target.files[0]; if(!f) return;
  const r = new FileReader();
  r.onload = ()=>{
    const nums = parseNumbers(r.result);
    if(nums.length===0){ SFX.error(); return toast('No valid numbers found'); }
    trainData = trainData.concat(nums);
    save(STORE.TRAIN, trainData);
    // Defer heavy model build for UI smoothness
    setTimeout(()=>{
      model = buildModel(trainData); save(STORE.MODEL, model);
      updateModelStatus(); renderStats();
    }, 30);
    SFX.success(); haptic(15);
    toast('Added '+nums.length+' rounds from file');
  };
  r.readAsText(f);
  e.target.value = '';
});

document.getElementById('addPasteBtn').addEventListener('click', ()=>{
  const txt = document.getElementById('pasteBox').value;
  const nums = parseNumbers(txt);
  if(nums.length===0){ SFX.error(); return toast('No valid numbers detected'); }
  trainData = trainData.concat(nums);
  save(STORE.TRAIN, trainData);
  setTimeout(()=>{
    model = buildModel(trainData); save(STORE.MODEL, model);
    updateModelStatus(); renderStats();
  }, 30);
  document.getElementById('pasteBox').value='';
  SFX.success(); haptic(15);
  toast('Added '+nums.length+' rounds');
});

/* Train keypad (mini, for fast manual entry) */
const trainKeypadGrid = document.getElementById('trainKeypadGrid');
const trainInputDisplay = document.getElementById('trainInputDisplay');

function updateTrainDisplay(){
  if(trainInputBuffer === ''){
    trainInputDisplay.innerHTML = '<span class="placeholder">Tap numbers below…</span>';
  } else {
    trainInputDisplay.textContent = trainInputBuffer;
  }
}

buildKeypad(trainKeypadGrid, (n)=>{
  SFX.key(); haptic(4);
  // Immediate single-number add for fastest input
  addTrainingNumber(n);
});

function addTrainingNumber(n){
  trainData.push(n);
  save(STORE.TRAIN, trainData);
  // Show in display briefly
  trainInputBuffer = String(n);
  updateTrainDisplay();
  // Update model in background (debounced)
  scheduleModelRebuild();
  updateModelStatus();
  renderStatsLight();
}

let modelRebuildTimer = null;
function scheduleModelRebuild(){
  if(modelRebuildTimer) clearTimeout(modelRebuildTimer);
  modelRebuildTimer = setTimeout(()=>{
    model = buildModel(trainData);
    save(STORE.MODEL, model);
    renderStats();
  }, 350);
}

document.getElementById('trainBack').addEventListener('click', ()=>{
  if(trainData.length === 0){ SFX.error(); return; }
  SFX.delete(); haptic(6);
  trainData.pop();
  save(STORE.TRAIN, trainData);
  trainInputBuffer = '';
  updateTrainDisplay();
  scheduleModelRebuild();
  updateModelStatus();
  renderStatsLight();
  toast('Removed last number');
});

document.getElementById('trainClear').addEventListener('click', ()=>{
  SFX.tap();
  trainInputBuffer = '';
  updateTrainDisplay();
});

document.getElementById('trainAddBtn').addEventListener('click', ()=>{
  SFX.success(); haptic(10);
  toast('All taps already saved — keep going!');
});

document.getElementById('trainBtn').addEventListener('click', ()=>{
  if(trainData.length<5){ SFX.error(); return toast('Need at least 5 rounds'); }
  SFX.predict(); haptic(20);
  // Defer heavy work
  setTimeout(()=>{
    model = buildModel(trainData);
    save(STORE.MODEL, model);
    updateModelStatus();
    toast('✅ Model trained on '+trainData.length+' rounds');
  }, 30);
});

document.getElementById('clearTrainBtn').addEventListener('click', ()=>{
  confirmModal('Clear training data?','All training samples will be erased.', ()=>{
    trainData = []; model = null;
    save(STORE.TRAIN, trainData); localStorage.removeItem(STORE.MODEL);
    updateModelStatus(); renderStats();
    SFX.delete(); haptic(15);
    toast('Training data cleared');
  });
});

/* History */
document.getElementById('clearHistBtn').addEventListener('click', ()=>{
  confirmModal('Clear history?','All saved predictions will be erased.', ()=>{
    history = []; save(STORE.HIST, history); renderHistory();
    SFX.delete(); toast('History cleared');
  });
});

function parseNumbers(txt){
  return txt.split(/[\s,;\n\r\t]+/)
    .map(s=>s.trim()).filter(Boolean)
    .map(s=>parseInt(s,10))
    .filter(n=>!isNaN(n) && n>=0 && n<=36);
}

/* ============================================================
   RENDERERS (optimized)
   ============================================================ */
function updateModelStatus(){
  document.getElementById('dataSize').textContent = trainData.length+' rounds';
  document.getElementById('trainCount').textContent= trainData.length+' samples';
  const el = document.getElementById('modelStatus');
  if(!model || model.n<5) el.textContent='No model trained';
  else el.textContent='Model: '+model.n+' rounds';
}

function renderStatsLight(){
  // Light update — only counters
  document.getElementById('statTotal').textContent = trainData.length;
  let r=0,b=0,z=0;
  for(const n of trainData){
    if(n===0) z++;
    else if(RED_SET.has(n)) r++;
    else b++;
  }
  document.getElementById('statRed').textContent = r;
  document.getElementById('statBlack').textContent = b;
  document.getElementById('statZero').textContent = z;
}

function renderStats(){
  renderStatsLight();
  const counts = new Array(37).fill(0);
  trainData.forEach(n=>{ if(n>=0&&n<=36) counts[n]++; });
  const max = Math.max(1,...counts);
  const grid = document.getElementById('heatGrid');
  // Build via DocumentFragment for performance
  const frag = document.createDocumentFragment();
  for(let i=0;i<=36;i++){
    const intensity = counts[i]/max;
    const c = colorOf(i);
    const base = c==='red'?[229,57,53]:c==='black'?[55,71,79]:[46,125,50];
    const bg = `rgba(${base[0]},${base[1]},${base[2]},${0.08+intensity*0.55})`;
    const div = document.createElement('div');
    div.className = 'heat-cell';
    div.style.background = bg;
    div.title = `${i}: ${counts[i]} times`;
    div.textContent = i;
    if(intensity > 0.5) div.style.color = '#fff';
    frag.appendChild(div);
  }
  grid.innerHTML = '';
  grid.appendChild(frag);
}

function renderHistory(){
  const list = document.getElementById('histList');
  const recent = document.getElementById('recentStrip');

  if(history.length===0){
    list.innerHTML = '<div class="empty">No predictions yet.</div>';
    document.getElementById('accuracyPill').textContent = '--% acc';
  } else {
    let hits=0, partial=0, total=0;
    const items = history.slice(0,40);
    const parts = [];
    for(const h of items){
      let badge='', label='Pending';
      if(h.actual!==null && h.actual!==undefined){
        total++;
        if(h.actual===h.predicted){ badge='hit'; label='✓ Exact'; hits++; }
        else if((h.top5||[]).includes(h.actual)){ badge='partial'; label='~ In Top 5'; partial++; }
        else if(colorOf(h.actual)===h.color){ badge='partial'; label='~ Color'; partial++; }
        else { badge='miss'; label='✗ Miss'; }
      }
      const pCol = colorOf(h.predicted);
      const aCol = h.actual!==null && h.actual!==undefined ? colorOf(h.actual) : 'pending';
      const date = new Date(h.ts).toLocaleString();
      parts.push(`<div class="hist-item">
        <div class="hist-pred ${pCol}">${h.predicted}</div>
        <div class="hist-mid">
          <b>Predicted ${h.predicted} (${h.conf}%)</b>
          <small>${date}</small>
        </div>
        <div class="hist-actual ${aCol}">${h.actual!==null && h.actual!==undefined ? h.actual : '?'}</div>
        ${badge?`<div class="hist-badge ${badge}">${label}</div>`:''}
      </div>`);
    }
    list.innerHTML = parts.join('');
    const accPill = document.getElementById('accuracyPill');
    if(total>0){
      const acc = Math.round((hits + partial*0.4)/total*100);
      accPill.textContent = acc+'% acc';
    } else accPill.textContent='--% acc';
  }

  const last = trainData.slice(-30).reverse();
  recent.innerHTML = last.length===0
    ? '<div class="empty" style="padding:10px">No data yet.</div>'
    : last.map(n=>`<div class="r-num ${colorOf(n)}">${n}</div>`).join('');
}

/* ============================================================
   EXPORT / IMPORT
   ============================================================ */
function exportModel(){
  const pkg = { trainData, model, history, exportedAt:new Date().toISOString() };
  const blob = new Blob([JSON.stringify(pkg,null,2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'roulette-ai-model.json';
  a.click();
  SFX.success();
  toast('Model exported');
}
function importModel(){
  const inp = document.createElement('input');
  inp.type='file'; inp.accept='application/json,.json';
  inp.onchange = e=>{
    const f = e.target.files[0]; if(!f) return;
    const r = new FileReader();
    r.onload = ()=>{
      try{
        const pkg = JSON.parse(r.result);
        if(Array.isArray(pkg.trainData)) trainData = pkg.trainData.filter(n=>n>=0&&n<=36);
        if(pkg.model) model = pkg.model;
        if(Array.isArray(pkg.history)) history = pkg.history;
        save(STORE.TRAIN,trainData); save(STORE.MODEL,model); save(STORE.HIST,history);
        updateModelStatus(); renderStats(); renderHistory();
        SFX.success(); toast('Model imported');
      }catch(err){ SFX.error(); toast('Invalid file'); }
    };
    r.readAsText(f);
  };
  inp.click();
}
function resetAll(){
  trainData=[]; model=null; history=[];
  localStorage.removeItem(STORE.TRAIN);
  localStorage.removeItem(STORE.MODEL);
  localStorage.removeItem(STORE.HIST);
  updateModelStatus(); renderStats(); renderHistory();
  document.getElementById('resultCard').classList.add('hidden');
  for(let i=0;i<10;i++) setSlotValue(i, '');
  setActiveSlot(0);
  SFX.delete(); haptic(20);
  toast('All data reset');
}

/* ============================================================
   TOAST & MODALS
   ============================================================ */
let toastTimer=null;
function toast(msg){
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(()=>t.classList.remove('show'), 2000);
}
const modal = document.getElementById('modal');
const mTitle = document.getElementById('modalTitle');
const mBody  = document.getElementById('modalBody');
const mOk    = document.getElementById('modalOk');
const mCancel= document.getElementById('modalCancel');
function closeModal(){ modal.classList.add('hidden'); }
mCancel.addEventListener('click', ()=>{ SFX.tap(); closeModal(); });
modal.addEventListener('click', e=>{ if(e.target===modal) closeModal(); });

function confirmModal(title, msg, onOk){
  mTitle.textContent=title; mBody.textContent=msg;
  mOk.textContent='Confirm'; mCancel.style.display='inline-block';
  mOk.onclick = ()=>{ SFX.tap(); closeModal(); onOk&&onOk(); };
  modal.classList.remove('hidden');
}
function infoModal(title, msg){
  mTitle.textContent=title; mBody.textContent=msg;
  mOk.textContent='OK'; mCancel.style.display='none';
  mOk.onclick = ()=>{ SFX.tap(); closeModal(); };
  modal.classList.remove('hidden');
}

/* Keypad-style modal for entering a number (replaces prompt) */
function keypadModal(title, msg, cb){
  mTitle.textContent = title;
  mBody.innerHTML = `
    <div style="margin-bottom:10px;">${msg}</div>
    <div class="train-input-display" id="kpModalDisp"><span class="placeholder">Tap a number…</span></div>
    <div class="keypad-grid mini" id="kpModalGrid"></div>
  `;
  mOk.textContent='Save'; mCancel.style.display='inline-block';

  let value = '';
  const disp = document.getElementById('kpModalDisp');
  const grid = document.getElementById('kpModalGrid');
  buildKeypad(grid, (n)=>{
    value = String(n);
    disp.textContent = value;
    SFX.key(); haptic(4);
  });

  mOk.onclick = ()=>{
    if(value === ''){ SFX.error(); return toast('Pick a number'); }
    SFX.tap();
    closeModal();
    cb(value);
  };
  modal.classList.remove('hidden');
}

/* ============================================================
   PERFORMANCE: passive scroll listeners + viewport lock
   ============================================================ */
// Prevent pinch-zoom
document.addEventListener('gesturestart', e => e.preventDefault());
document.addEventListener('gesturechange', e => e.preventDefault());
// Prevent double-tap zoom
let lastTouchEnd = 0;
document.addEventListener('touchend', e => {
  const now = Date.now();
  if(now - lastTouchEnd <= 300) e.preventDefault();
  lastTouchEnd = now;
}, {passive:false});

// Resume audio context on first user interaction (mobile requirement)
const resumeAudio = () => {
  getCtx();
  document.removeEventListener('touchstart', resumeAudio);
  document.removeEventListener('click', resumeAudio);
};
document.addEventListener('touchstart', resumeAudio, {passive:true, once:true});
document.addEventListener('click', resumeAudio, {once:true});

/* ============================================================
   BOOT
   ============================================================ */
attachRipples();
updateModelStatus();
renderStats();
renderHistory();
updateTrainDisplay();
