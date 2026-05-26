/* ============================================================
   ROULETTE AI PREDICTOR — Pure JS, fully offline, ML on device
   ============================================================ */

/* ----------- Roulette constants ----------- */
const RED_SET = new Set([1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36]);
const colorOf = n => n === 0 ? 'green' : (RED_SET.has(n) ? 'red' : 'black');
const parityOf = n => n === 0 ? 'zero' : (n % 2 === 0 ? 'even' : 'odd');
const dozenOf  = n => n === 0 ? '-' : (n <= 12 ? '1st' : n <= 24 ? '2nd' : '3rd');
const rangeOf  = n => n === 0 ? '-' : (n <= 18 ? 'Low' : 'High');

/* ----------- State ----------- */
const STORE = {
  TRAIN: 'rai_train_v1',
  MODEL: 'rai_model_v1',
  HIST:  'rai_hist_v1',
};
let trainData = loadArr(STORE.TRAIN);     // array of numbers (0-36)
let model     = loadObj(STORE.MODEL);     // markov-like model
let history   = loadArr(STORE.HIST);      // {ts, predicted, top5, actual}
let lastPrediction = null;

function loadArr(k){ try{ return JSON.parse(localStorage.getItem(k)||'[]'); }catch(e){return [];} }
function loadObj(k){ try{ return JSON.parse(localStorage.getItem(k)||'null'); }catch(e){return null;} }
function save(k,v){ localStorage.setItem(k, JSON.stringify(v)); }

/* ============================================================
   PREDICTION ENGINE
   ----------------------------------------------------------------
   Hybrid model that learns ONLY from your uploaded training data:
     • Order-1, Order-2, Order-3 Markov transition tables
     • Global frequency prior
     • Color / parity / dozen / range pattern tables
     • Recency-weighted blending
   Given last N rounds, computes a probability distribution over 0–36.
   ============================================================ */

function buildModel(data){
  const m = {
    n: data.length,
    freq: new Array(37).fill(0),
    m1:   Array.from({length:37},()=>new Array(37).fill(0)),         // p(next | n-1)
    m2:   {},   // key "a,b" -> array(37)
    m3:   {},   // key "a,b,c" -> array(37)
    colorTrans:  Array.from({length:3},()=>new Array(3).fill(0)),    // r,b,g
    parityTrans: Array.from({length:3},()=>new Array(3).fill(0)),    // odd,even,zero
    dozenTrans:  Array.from({length:4},()=>new Array(4).fill(0)),    // 0,1,2,3
    rangeTrans:  Array.from({length:3},()=>new Array(3).fill(0)),    // low,high,zero
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
    // not enough data — return uniform
    return { dist:new Array(37).fill(1/37), top:Array.from({length:37},(_,i)=>({n:i,p:1/37})).slice(0,5), conf:0 };
  }

  const last1 = seq[seq.length-1];
  const last2 = seq.length>=2 ? seq[seq.length-2]+','+seq[seq.length-1] : null;
  const last3 = seq.length>=3 ? seq[seq.length-3]+','+seq[seq.length-2]+','+seq[seq.length-1] : null;

  const freqDist  = normalize(m.freq.slice());
  const m1Row     = (last1>=0 && last1<=36) ? normalize(m.m1[last1].slice()) : freqDist;
  const m2Row     = (last2 && m.m2[last2])  ? normalize(m.m2[last2].slice()) : null;
  const m3Row     = (last3 && m.m3[last3])  ? normalize(m.m3[last3].slice()) : null;

  // Weights tuned: prefer higher-order chains when available + always blend recency & frequency
  let w1=0.30, w2=0.25, w3=0.20, wF=0.10, wC=0.15;
  if(!m2Row){ w1+=w2; w2=0; }
  if(!m3Row){ w1+=w3; w3=0; }

  // Pattern features from last1
  const cIdx = c => c==='red'?0:c==='black'?1:2;
  const pIdx = p => p==='odd'?0:p==='even'?1:2;
  const dIdx = d => d==='1st'?1:d==='2nd'?2:d==='3rd'?3:0;
  const rIdx = r => r==='Low'?0:r==='High'?1:2;

  const colorRow  = normalize(m.colorTrans [cIdx(colorOf (last1))]);
  const parityRow = normalize(m.parityTrans[pIdx(parityOf(last1))]);
  const dozenRow  = normalize(m.dozenTrans [dIdx(dozenOf (last1))]);
  const rangeRow  = normalize(m.rangeTrans [rIdx(rangeOf (last1))]);

  // Build final distribution
  const dist = new Array(37).fill(0);
  for(let n=0;n<=36;n++){
    let p = w1*(m1Row[n]||0)
          + (m2Row?w2*m2Row[n]:0)
          + (m3Row?w3*m3Row[n]:0)
          + wF*freqDist[n];

    // pattern boost
    const cBoost = colorRow [cIdx(colorOf (n))];
    const pBoost = parityRow[pIdx(parityOf(n))];
    const dBoost = dozenRow [dIdx(dozenOf (n))];
    const rBoost = rangeRow [rIdx(rangeOf (n))];
    const patternMix = (cBoost+pBoost+dBoost+rBoost)/4;
    p += wC * patternMix;

    dist[n] = p;
  }

  // Recency bonus from the input sequence itself (very small)
  for(let i=Math.max(0,seq.length-6); i<seq.length; i++){
    const n = seq[i];
    if(n>=0 && n<=36) dist[n] *= 1.03;
  }

  const final = normalize(dist);
  const ranked = final.map((p,n)=>({n,p})).sort((a,b)=>b.p-a.p);
  const top = ranked.slice(0,5);
  const conf = Math.min(99, Math.round(ranked[0].p*100*4)); // scaled
  return { dist:final, top, conf };
}

/* ============================================================
   UI WIRING
   ============================================================ */

/* ---- Splash ---- */
window.addEventListener('load', () => {
  setTimeout(()=>document.getElementById('splash').classList.add('hide'), 1600);
});

/* ---- Tabs ---- */
document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-'+btn.dataset.tab).classList.add('active');
    if(btn.dataset.tab==='history') renderHistory();
    if(btn.dataset.tab==='train')   renderStats();
  });
});

/* ---- Menu ---- */
const menu = document.getElementById('menu');
document.getElementById('menuBtn').addEventListener('click', e=>{
  e.stopPropagation();
  menu.classList.toggle('show');
});
document.addEventListener('click', ()=>menu.classList.remove('show'));
menu.addEventListener('click', e=>{
  const a = e.target.dataset.action;
  if(!a) return;
  menu.classList.remove('show');
  if(a==='export') exportModel();
  if(a==='import') importModel();
  if(a==='reset')  confirmModal('Reset all data?','This will erase training data, model and history.', resetAll);
  if(a==='about')  infoModal('About Roulette AI',
     'A modern roulette prediction tool that learns from your uploaded training data using a hybrid Markov + pattern engine. Predictions are entirely on-device and based ONLY on your training data. Use responsibly — no model can guarantee outcomes of a fair random game.');
});

/* ---- Build sequence grid ---- */
const seqGrid = document.getElementById('seqGrid');
for(let i=0;i<10;i++){
  const cell = document.createElement('div');
  cell.className = 'seq-cell';
  cell.innerHTML = `<div class="idx">R-${10-i}</div>
    <input type="number" min="0" max="36" inputmode="numeric" placeholder="--" data-i="${i}">`;
  seqGrid.appendChild(cell);
}
const seqInputs = [...seqGrid.querySelectorAll('input')];
seqInputs.forEach((inp,i)=>{
  inp.addEventListener('input', ()=>{
    let v = inp.value.replace(/[^\d]/g,'');
    if(v!=='' && +v>36) v = '36';
    inp.value = v;
    if(v!=='' && i<seqInputs.length-1) seqInputs[i+1].focus();
  });
  inp.addEventListener('keydown', e=>{
    if(e.key==='Backspace' && inp.value==='' && i>0) seqInputs[i-1].focus();
  });
});

document.getElementById('clearSeq').addEventListener('click', ()=>{
  seqInputs.forEach(i=>i.value='');
  seqInputs[0].focus();
});
document.getElementById('fillRecent').addEventListener('click', ()=>{
  const recent = trainData.slice(-10);
  if(recent.length<1) return toast('No training data yet');
  seqInputs.forEach(i=>i.value='');
  const start = 10 - recent.length;
  recent.forEach((n,k)=> seqInputs[start+k].value = n);
  toast('Filled with recent training history');
});

/* ---- Predict ---- */
document.getElementById('predictBtn').addEventListener('click', ()=>{
  const seq = seqInputs.map(i=>i.value).filter(v=>v!=='').map(v=>+v);
  if(seq.length<3) return toast('Enter at least 3 recent numbers');
  if(seq.some(v=>v<0||v>36||isNaN(v))) return toast('Numbers must be 0–36');

  let m = model;
  if(!m || m.n !== trainData.length){
    if(trainData.length < 5) return toast('Need at least 5 training rounds. Go to Train tab.');
    m = buildModel(trainData);
    model = m; save(STORE.MODEL, m);
  }
  const res = predictNext(seq, m);
  showResult(res, seq);
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

  setTimeout(()=>{
    document.getElementById('confFill').style.width = res.conf + '%';
  }, 100);

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
  card.scrollIntoView({behavior:'smooth', block:'nearest'});
}

/* ---- Record actual ---- */
document.getElementById('saveActualBtn').addEventListener('click', ()=>{
  if(!lastPrediction) return toast('Make a prediction first');
  promptModal('Record Actual Result','Enter the actual winning number (0–36):', val=>{
    const n = parseInt(val,10);
    if(isNaN(n)||n<0||n>36) return toast('Invalid number');
    lastPrediction.actual = n;
    history.unshift(lastPrediction);
    save(STORE.HIST, history);
    // also add to training data automatically
    trainData.push(n); save(STORE.TRAIN, trainData);
    model = buildModel(trainData); save(STORE.MODEL, model);
    updateModelStatus();
    toast('Result saved & model retrained');
    lastPrediction = null;
    renderHistory();
  });
});

/* ---- Train tab actions ---- */
document.getElementById('uploadZone').addEventListener('click', ()=>document.getElementById('fileInput').click());
document.getElementById('fileInput').addEventListener('change', e=>{
  const f = e.target.files[0]; if(!f) return;
  const r = new FileReader();
  r.onload = ()=>{
    const nums = parseNumbers(r.result);
    if(nums.length===0) return toast('No valid numbers found');
    trainData = trainData.concat(nums);
    save(STORE.TRAIN, trainData);
    model = buildModel(trainData); save(STORE.MODEL, model);
    updateModelStatus(); renderStats();
    toast('Added '+nums.length+' rounds from file');
  };
  r.readAsText(f);
  e.target.value = '';
});

document.getElementById('addPasteBtn').addEventListener('click', ()=>{
  const txt = document.getElementById('pasteBox').value;
  const nums = parseNumbers(txt);
  if(nums.length===0) return toast('No valid numbers detected');
  trainData = trainData.concat(nums);
  save(STORE.TRAIN, trainData);
  model = buildModel(trainData); save(STORE.MODEL, model);
  document.getElementById('pasteBox').value='';
  updateModelStatus(); renderStats();
  toast('Added '+nums.length+' rounds');
});

document.getElementById('addSingleBtn').addEventListener('click', ()=>{
  const inp = document.getElementById('singleNum');
  const v = parseInt(inp.value,10);
  if(isNaN(v)||v<0||v>36) return toast('Enter 0–36');
  trainData.push(v); save(STORE.TRAIN, trainData);
  model = buildModel(trainData); save(STORE.MODEL, model);
  inp.value=''; inp.focus();
  updateModelStatus(); renderStats();
});

document.getElementById('trainBtn').addEventListener('click', ()=>{
  if(trainData.length<5) return toast('Need at least 5 rounds');
  model = buildModel(trainData);
  save(STORE.MODEL, model);
  updateModelStatus();
  toast('✅ Model trained on '+trainData.length+' rounds');
});

document.getElementById('clearTrainBtn').addEventListener('click', ()=>{
  confirmModal('Clear training data?','All training samples will be erased.', ()=>{
    trainData = []; model = null;
    save(STORE.TRAIN, trainData); localStorage.removeItem(STORE.MODEL);
    updateModelStatus(); renderStats();
    toast('Training data cleared');
  });
});

/* ---- History ---- */
document.getElementById('clearHistBtn').addEventListener('click', ()=>{
  confirmModal('Clear history?','All saved predictions will be erased.', ()=>{
    history = []; save(STORE.HIST, history); renderHistory(); toast('History cleared');
  });
});

function parseNumbers(txt){
  return txt.split(/[\s,;\n\r\t]+/)
    .map(s=>s.trim()).filter(Boolean)
    .map(s=>parseInt(s,10))
    .filter(n=>!isNaN(n) && n>=0 && n<=36);
}

/* ---- Renderers ---- */
function updateModelStatus(){
  document.getElementById('dataSize').textContent = trainData.length+' rounds';
  document.getElementById('trainCount').textContent= trainData.length+' samples';
  const el = document.getElementById('modelStatus');
  if(!model || model.n<5) el.textContent='No model trained';
  else el.textContent='Model: '+model.n+' rounds';
}

function renderStats(){
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

  // Heat grid
  const counts = new Array(37).fill(0);
  trainData.forEach(n=>{ if(n>=0&&n<=36) counts[n]++; });
  const max = Math.max(1,...counts);
  const grid = document.getElementById('heatGrid');
  grid.innerHTML = '';
  for(let i=0;i<=36;i++){
    const intensity = counts[i]/max;
    const c = colorOf(i);
    const base = c==='red'?[239,64,96]:c==='black'?[80,80,120]:[33,192,122];
    const bg = `rgba(${base[0]},${base[1]},${base[2]},${0.15+intensity*0.7})`;
    const div = document.createElement('div');
    div.className = 'heat-cell';
    div.style.background = bg;
    div.title = `${i}: ${counts[i]} times`;
    div.textContent = i;
    grid.appendChild(div);
  }
}

function renderHistory(){
  const list = document.getElementById('histList');
  const recent = document.getElementById('recentStrip');

  if(history.length===0){
    list.innerHTML = '<div class="empty">No predictions yet.</div>';
  } else {
    let hits=0, partial=0, total=0;
    list.innerHTML = history.slice(0,40).map(h=>{
      let badge='', label='Pending';
      if(h.actual!==null && h.actual!==undefined){
        total++;
        if(h.actual===h.predicted){ badge='hit'; label='✓ Exact'; hits++; }
        else if((h.top5||[]).includes(h.actual)){ badge='partial'; label='~ In Top 5'; partial++; }
        else if(colorOf(h.actual)===h.color){ badge='partial'; label='~ Color match'; partial++; }
        else { badge='miss'; label='✗ Miss'; }
      }
      const pCol = colorOf(h.predicted);
      const aCol = h.actual!==null && h.actual!==undefined ? colorOf(h.actual) : 'pending';
      const date = new Date(h.ts).toLocaleString();
      return `<div class="hist-item">
        <div class="hist-pred ${pCol}">${h.predicted}</div>
        <div class="hist-mid">
          <b>Predicted ${h.predicted} (${h.conf}%)</b>
          <small>${date}</small>
        </div>
        <div class="hist-actual ${aCol}">${h.actual!==null && h.actual!==undefined ? h.actual : '?'}</div>
        ${badge?`<div class="hist-badge ${badge}">${label}</div>`:''}
      </div>`;
    }).join('');
    const accPill = document.getElementById('accuracyPill');
    if(total>0){
      const acc = Math.round((hits + partial*0.4)/total*100);
      accPill.textContent = acc+'% acc';
    } else accPill.textContent='--% acc';
  }

  // recent strip = last 30 training numbers
  const last = trainData.slice(-30).reverse();
  recent.innerHTML = last.length===0
    ? '<div class="empty" style="padding:10px">No data yet.</div>'
    : last.map(n=>`<div class="r-num ${colorOf(n)}">${n}</div>`).join('');
}

/* ---- Export / Import ---- */
function exportModel(){
  const pkg = { trainData, model, history, exportedAt:new Date().toISOString() };
  const blob = new Blob([JSON.stringify(pkg,null,2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'roulette-ai-model.json';
  a.click();
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
        toast('Model imported');
      }catch(err){ toast('Invalid file'); }
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
  toast('All data reset');
}

/* ---- Toast & Modal ---- */
let toastTimer=null;
function toast(msg){
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(()=>t.classList.remove('show'), 2200);
}
const modal = document.getElementById('modal');
const mTitle = document.getElementById('modalTitle');
const mBody  = document.getElementById('modalBody');
const mOk    = document.getElementById('modalOk');
const mCancel= document.getElementById('modalCancel');
function closeModal(){ modal.classList.add('hidden'); }
mCancel.addEventListener('click', closeModal);
modal.addEventListener('click', e=>{ if(e.target===modal) closeModal(); });
function confirmModal(title, msg, onOk){
  mTitle.textContent=title; mBody.textContent=msg;
  mOk.textContent='Confirm'; mCancel.style.display='inline-block';
  mOk.onclick = ()=>{ closeModal(); onOk&&onOk(); };
  modal.classList.remove('hidden');
}
function infoModal(title, msg){
  mTitle.textContent=title; mBody.textContent=msg;
  mOk.textContent='OK'; mCancel.style.display='none';
  mOk.onclick = closeModal;
  modal.classList.remove('hidden');
}
function promptModal(title, msg, cb){
  mTitle.textContent=title;
  mBody.innerHTML = `<div>${msg}</div><input type="number" id="promptInp" min="0" max="36" placeholder="0-36">`;
  mOk.textContent='Save'; mCancel.style.display='inline-block';
  setTimeout(()=>document.getElementById('promptInp').focus(),50);
  mOk.onclick = ()=>{
    const v = document.getElementById('promptInp').value;
    closeModal();
    cb(v);
  };
  modal.classList.remove('hidden');
}

/* ---- Boot ---- */
updateModelStatus();
renderStats();
renderHistory();
