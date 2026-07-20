let stream, recording = false;
let _chunkCount = 0;

window.GROQ_EVERY   = 1;     // show text every N chunks (1 = every chunk)
window.GROQ_HOLD_MS = 6000;  // ms before text fades

async function toggleMic(){
  const btn = document.getElementById('micBtn');
  if(!recording){
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio:true, video:false });
      recording = true;
      btn.textContent = 'stop';
      btn.classList.add('active');
      log('listening...');
      initVisualizer(stream);
      startChunking();
    } catch(e){
      log('mic denied: ' + e.message);
    }
  } else {
    recording = false;
    btn.textContent = 'start';
    btn.classList.remove('active');
    stopVisualizer();
    if(stream) stream.getTracks().forEach(t => t.stop());
    log('stopped');
  }
}

// Recording loop is fully decoupled from the fetch — each recorder fires and
// immediately starts the next one. Sends run in parallel; back-pressure from
// the server never delays the next recording window.
function startChunking(){
  function nextChunk(){
    if(!recording) return;
    const recorder = new MediaRecorder(stream);
    const chunks   = [];
    recorder.ondataavailable = e => { if(e.data.size > 0) chunks.push(e.data); };
    recorder.onstop = () => {
      // start next recording immediately — don't wait for the fetch
      nextChunk();
      if(chunks.length > 0){
        sendChunk(new Blob(chunks, { type:'audio/webm' }));  // fire and forget
      }
    };
    recorder.start();
    setTimeout(() => {
      if(recorder.state === 'recording') recorder.stop();
    }, CONFIG.CHUNK_MS);
  }
  nextChunk();
}

async function sendChunk(blob){
  const form = new FormData();
  form.append('audio', blob, 'chunk.webm');
  try {
    const res  = await fetch(`${CONFIG.SERVER}/search`, { method:'POST', body:form });
    const data = await res.json();
    if(data.error){ log('server error: ' + data.error); return; }

    window._lastServerResponse = data;
    window._lastServerTime = Date.now();

    console.log('[emotion]', data.emotion, data.probs, 'vad:', data.vad);

    if(data.probs) window.updateFromEmotion(data.probs);

    if(data.vad && window.vadTarget){
      window.vadTarget.v = data.vad[0];
      window.vadTarget.a = data.vad[1];
    }

    _chunkCount++;
    const every = (window.GROQ_EVERY > 0) ? window.GROQ_EVERY : 1;
    if(data.question && (_chunkCount % every === 0)) showQuestion(data.question);

    const top = data.emotion_word || data.emotion || '?';
    const probStr = data.probs
      ? Object.entries(data.probs).map(([k,v]) => `${k}:${v.toFixed(2)}`).join(' ')
      : '';
    log(`${top} · ${probStr}`);
  } catch(e){
    log('error: ' + e.message);
    console.error('[sendChunk]', e);
  }
}

// ── typewriter text display — canvas-based, zero DOM reflow ──────────────────
// Text is drawn onto a 2D canvas overlay. No DOM textContent writes,
// no layout invalidation, no reflow. Opacity is simulated via globalAlpha.
const TYPE_CHAR_MS = 48;
const TEXT_COLOR       = 'rgba(255,255,255,0.90)';
const TEXT_ATTR_COLOR  = 'rgba(255,255,255,0.50)';
const TEXT_SHADOW_BLUR  = 18;
const TEXT_SHADOW_COLOR = 'rgba(0,0,0,0.7)';

// live text style — mutated by UI controls
window.textStyle = {
  size:   28,
  family: 'Georgia',
};

const FONT_FAMILIES = {
  'Georgia':     '"Georgia", "Times New Roman", serif',
  'Palatino':    '"Palatino Linotype", "Palatino", serif',
  'Garamond':    '"EB Garamond", "Garamond", serif',
  'Didot':       '"Didot", "Bodoni MT", serif',
  'Futura':      '"Futura", "Century Gothic", sans-serif',
  'Helvetica':   '"Helvetica Neue", "Helvetica", sans-serif',
  'Courier':     '"Courier New", monospace',
  'Baskerville': '"Baskerville", "Libre Baskerville", serif',
};

function _bodyFont()  { return `italic 300 ${window.textStyle.size}px ${FONT_FAMILIES[window.textStyle.family] || window.textStyle.family}`; }
function _attrFont()  { return `300 ${Math.round(window.textStyle.size * 0.54)}px ${FONT_FAMILIES[window.textStyle.family] || window.textStyle.family}`; }

let _typeBusy    = false;
let _typePending = null;
let _typeText    = '';
let _typePos     = 0;
let _typeAccumMs = 0;
let _typePauseMs = 0;
let _typeLastTs  = 0;
let _typePhase   = 'idle';   // 'typing' | 'holding' | 'fading' | 'idle'
let _typeHoldEnd = 0;
let _typeFadeEnd = 0;
let _typeAlpha   = 1;
let _textCanvas  = null;
let _textCtx     = null;

function _getCtx(){
  if(_textCtx) return _textCtx;
  _textCanvas = document.getElementById('textCanvas');
  if(!_textCanvas) return null;
  _textCtx = _textCanvas.getContext('2d');
  return _textCtx;
}

function _textDraw(alpha){
  const ctx = _getCtx();
  if(!ctx || !_textCanvas) return;

  // match canvas pixel size to display size
  const W = _textCanvas.offsetWidth  || window.innerWidth;
  const H = _textCanvas.offsetHeight || window.innerHeight;
  if(_textCanvas.width !== W || _textCanvas.height !== H){
    _textCanvas.width  = W;
    _textCanvas.height = H;
  }

  ctx.clearRect(0, 0, W, H);
  if(!_typeText || _typePos === 0 || alpha <= 0) return;

  const visible = _typeText.slice(0, _typePos);

  // split "quote text — Attribution" into body + attribution
  const dashIdx = visible.lastIndexOf(' — ');
  const body    = dashIdx > 0 ? visible.slice(0, dashIdx).trim() : visible.trim();
  const attr    = dashIdx > 0 ? '— ' + visible.slice(dashIdx + 3).trim() : '';

  ctx.globalAlpha  = alpha;
  ctx.textBaseline = 'top';
  ctx.shadowBlur   = TEXT_SHADOW_BLUR;
  ctx.shadowColor  = TEXT_SHADOW_COLOR;

  const sz      = window.textStyle.size;
  const bodyF   = _bodyFont();
  const attrF   = _attrFont();
  const lineH   = Math.round(sz * 1.55);
  const attrSz  = Math.round(sz * 0.54);
  const maxW    = Math.min(680, W * 0.72);

  function wordWrap(text, font, mW) {
    ctx.font = font;
    const words = text.split(' ');
    const lines = [];
    let line = '';
    for(const w of words){
      const test = line ? line + ' ' + w : w;
      if(ctx.measureText(test).width > mW && line){ lines.push(line); line = w; }
      else line = test;
    }
    if(line) lines.push(line);
    return lines;
  }

  const bodyLines = wordWrap(body, bodyF, maxW);
  const attrLines = attr ? wordWrap(attr, attrF, maxW) : [];

  const totalH = bodyLines.length * lineH + (attrLines.length ? 12 + attrLines.length * (attrSz * 1.6) : 0);
  const blockX = (W - maxW) / 2;
  const startY = H * 0.76 - totalH / 2;

  // draw body
  ctx.font      = bodyF;
  ctx.fillStyle = TEXT_COLOR;
  ctx.textAlign = 'left';
  for(let i = 0; i < bodyLines.length; i++){
    ctx.fillText(bodyLines[i], blockX, startY + i * lineH);
  }

  // draw attribution
  if(attrLines.length){
    ctx.font      = attrF;
    ctx.fillStyle = TEXT_ATTR_COLOR;
    ctx.shadowBlur = 8;
    const attrY = startY + bodyLines.length * lineH + 14;
    for(let i = 0; i < attrLines.length; i++){
      ctx.fillText(attrLines[i], blockX, attrY + i * (attrSz * 1.6));
    }
  }

  ctx.globalAlpha = 1;
}

// driven by rAF loop in transition.js
window._onTypeFrame = function(ts){
  if(_typePhase === 'idle') return;

  const dt = _typeLastTs ? Math.min(ts - _typeLastTs, 100) : 16;
  _typeLastTs = ts;

  if(_typePhase === 'typing'){
    _typeAccumMs += dt;
    const charMs = _typePauseMs > 0 ? _typePauseMs : TYPE_CHAR_MS;
    if(_typeAccumMs >= charMs){
      _typeAccumMs = 0;
      _typePauseMs = 0;
      if(_typePos < _typeText.length){
        const ch = _typeText[_typePos++];
        if(ch === '.' || ch === ':') _typePauseMs = TYPE_CHAR_MS * 3;
        else if(ch === ',')          _typePauseMs = TYPE_CHAR_MS * 1.5;
      } else {
        const holdMs = (window.GROQ_HOLD_MS > 0) ? window.GROQ_HOLD_MS : 6000;
        _typeHoldEnd = ts + holdMs;
        _typePhase   = 'holding';
      }
    }
    _typeAlpha = 1;
    _textDraw(_typeAlpha);

  } else if(_typePhase === 'holding'){
    if(ts >= _typeHoldEnd){
      _typeFadeEnd = ts + 1000;
      _typePhase   = 'fading';
    }

  } else if(_typePhase === 'fading'){
    const progress = (ts - (_typeFadeEnd - 1000)) / 1000;
    _typeAlpha = Math.max(0, 1 - progress);
    _textDraw(_typeAlpha);
    if(ts >= _typeFadeEnd){
      _textDraw(0);
      _typePhase = 'idle';
      _typeBusy  = false;
      if(_typePending){ const t = _typePending; _typePending = null; _beginText(t); }
    }
  }
};

function _beginText(text){
  _typeBusy    = true;
  _typeText    = text;
  _typePos     = 0;
  _typeAccumMs = 0;
  _typePauseMs = 0;
  _typeLastTs  = 0;
  _typeAlpha   = 1;
  _typePhase   = 'typing';
}

function showQuestion(text){
  if(!text) return;
  if(_typeBusy){ _typePending = text; return; }
  _beginText(text);
}

function toggleKnobs(){
  const el  = document.getElementById('knobs');
  const app = document.getElementById('app');
  el.classList.toggle('hidden');
  app.classList.toggle('panel-open', !el.classList.contains('hidden'));
}

function log(msg){
  const el = document.getElementById('log');
  if(el) el.textContent = msg;
}

fetch(`${CONFIG.SERVER}/health`)
  .then(r  => r.json())
  .then(() => log('ready'))
  .catch(() => log('server not reachable'));
