// ── FFT parameter extraction — Web Audio API ──────────────────────────────────

let analyser    = null;
let freqData    = null;
let prevFreq    = null;
let prevVolume  = 0;
let timeData    = null;
let fftAnimId   = null;
let vizCanvas   = null;
let vizCtx      = null;

// ── 12 smoothed signals ───────────────────────────────────────────────────────
const fftSmooth = {
  volume:0.2,   bass:0.5,      midrange:0.4,  presence:0.3,
  brightness:0.5, tone:0.3,    texture:0.2,   movement:0.1,
  spread:0.5,   f0:0.3,        f0delta:0.0,   f0confidence:0.3,
  tilt:0.5,     spectralSpread:0.3,
};
let FFT_alpha = 0.08;
Object.defineProperty(window, 'FFT_alpha', { get:()=>FFT_alpha, set:v=>{ FFT_alpha=v; } });

// ── all signals that appear in the routing panel ──────────────────────────────
const SIGNALS = [
  'volume','bass','midrange','presence',
  'brightness','tone','texture','movement',
  'spread','f0','f0delta','f0confidence',
  'tilt','spectralSpread',
];

// ── all graphical targets (shape + mood + colour) in one flat pool ────────────
const SHAPE_KEYS      = ['brightness','tone','movement','texture','volume','bass','spread'];
const MOOD_KEYS_POOL  = ['colorTemp','speed','blobSize','fog','saturation'];
const COLOUR_KEYS_POOL= ['colorRadius','hueShift','colorBand'];
const ALL_GFX_TARGETS = [...SHAPE_KEYS, ...MOOD_KEYS_POOL, ...COLOUR_KEYS_POOL];

// ── default routing — each signal maps to the gfx target of the same name or none
const fftRoute = {
  volume:        { dst:'blobSize',    mul:0.85 },
  bass:          { dst:'bass',        mul:-1.00 },
  midrange:      { dst:'speed',       mul:1.00 },
  presence:      { dst:'colorRadius', mul:1.90 },
  brightness:    { dst:'brightness',  mul:1.00 },
  tone:          { dst:'tone',        mul:0.40 },
  texture:       { dst:'texture',     mul:1.00 },
  movement:      { dst:'speed',       mul:1.00 },
  spread:        { dst:'none',        mul:1.00 },
  f0:            { dst:'none',        mul:1.00 },
  f0delta:       { dst:'none',        mul:1.00 },
  f0confidence:  { dst:'none',        mul:1.00 },
  tilt:          { dst:'none',        mul:1.00 },
  spectralSpread:{ dst:'none',        mul:1.00 },
};

function initVisualizer(stream){
  vizCanvas = document.getElementById('vizCanvas');
  vizCtx    = vizCanvas.getContext('2d');

  const audioCtx = new AudioContext();
  const source   = audioCtx.createMediaStreamSource(stream);
  analyser       = audioCtx.createAnalyser();
  analyser.fftSize               = 2048;
  analyser.smoothingTimeConstant = 0.6;
  source.connect(analyser);

  freqData = new Float32Array(analyser.frequencyBinCount);
  timeData = new Float32Array(analyser.fftSize);
  prevFreq = new Float32Array(analyser.frequencyBinCount);

  fftLoop();
}

function stopVisualizer(){
  if(fftAnimId) cancelAnimationFrame(fftAnimId);
  fftAnimId = null;
  analyser  = null;
  if(vizCtx && vizCanvas) vizCtx.clearRect(0, 0, vizCanvas.width, vizCanvas.height);
}

let _fftLastTs = 0;
function fftLoop(ts){
  if(!analyser) return;
  fftAnimId = requestAnimationFrame(fftLoop);
  if(ts - _fftLastTs < 33) return;  // cap at ~30fps
  _fftLastTs = ts;

  analyser.getFloatFrequencyData(freqData);
  analyser.getFloatTimeDomainData(timeData);

  const binCount = freqData.length;
  const nyquist  = 22050;
  const binHz    = nyquist / binCount;
  const N        = timeData.length;

  // linear power per bin
  const power = new Float32Array(binCount);
  let totalPower = 0;
  for(let i = 0; i < binCount; i++){
    power[i]    = Math.pow(10, Math.max(freqData[i], -90) / 10);
    totalPower += power[i];
  }
  const safe = totalPower > 1e-10;

  // band boundaries
  const b = {
    sub0:   Math.floor(20  / binHz),
    sub1:   Math.floor(80  / binHz),
    bass0:  Math.floor(80  / binHz),
    bass1:  Math.floor(300 / binHz),
    mid0:   Math.floor(300 / binHz),
    mid1:   Math.floor(2000/ binHz),
    pres0:  Math.floor(2000/ binHz),
    pres1:  Math.floor(5000/ binHz),
    hi0:    Math.floor(5000/ binHz),
    hi1:    Math.floor(10000/binHz),
    air0:   Math.floor(10000/binHz),
  };

  function bandEnergy(lo, hi){
    let e = 0;
    for(let i = lo; i <= Math.min(hi, binCount-1); i++) e += power[i];
    return e;
  }

  const bassE = bandEnergy(b.bass0, b.bass1);
  const midE  = bandEnergy(b.mid0,  b.mid1);
  const presE = bandEnergy(b.pres0, b.pres1);
  const airE  = bandEnergy(b.air0,  binCount-1);

  // ── volume — RMS ─────────────────────────────────────────────────────────────
  let rmsSum = 0;
  for(let i = 0; i < N; i++) rmsSum += timeData[i] * timeData[i];
  const rmsRaw = Math.sqrt(rmsSum / N);
  const volume = Math.min(rmsRaw * 8.0, 1.0);

  // noise gate: ratio-based signals zero out below this floor
  const NOISE_FLOOR = 0.008;
  const voiced = rmsRaw > NOISE_FLOOR;

  // ── bass — low body (80–300Hz) ───────────────────────────────────────────────
  const bass = (safe && voiced) ? Math.min(bassE / totalPower * 6.0, 1.0) : 0;

  // ── midrange — voice body (300Hz–2kHz) ───────────────────────────────────────
  const midrange = (safe && voiced) ? Math.min(midE / totalPower * 2.5, 1.0) : 0;

  // ── presence — projection (2–5kHz) ───────────────────────────────────────────
  const presence = (safe && voiced) ? Math.min(presE / totalPower * 8.0, 1.0) : 0;

  // ── brightness — spectral centroid ───────────────────────────────────────────
  let wSum = 0;
  for(let i = 0; i < binCount; i++) wSum += power[i] * i * binHz;
  const brightness = (safe && voiced) ? Math.min(wSum / (totalPower * nyquist), 1.0) : 0;

  // ── tone — spectral flatness (pitched vs noise) ───────────────────────────────
  let logSum = 0, linSum2 = 0, vBins = 0;
  for(let i = 1; i < binCount; i++){
    if(power[i] > 1e-12){ logSum += Math.log(power[i]); linSum2 += power[i]; vBins++; }
  }
  const tone = (voiced && vBins > 0 && linSum2 > 0)
    ? Math.min(Math.exp(logSum / vBins) / (linSum2 / vBins) * 10.0, 1.0) : 0;

  // ── texture — zero crossing rate ─────────────────────────────────────────────
  let zcCount = 0;
  for(let i = 1; i < N; i++){
    if((timeData[i] >= 0) !== (timeData[i-1] >= 0)) zcCount++;
  }
  const texture = voiced ? Math.min(zcCount / N * 10.0, 1.0) : 0;

  // ── movement — spectral flux ──────────────────────────────────────────────────
  let fluxSum = 0;
  for(let i = 0; i < binCount; i++){
    const d = power[i] - prevFreq[i];
    fluxSum += d * d;
    prevFreq[i] = power[i];
  }
  const movement = (safe && voiced)
    ? Math.min(Math.sqrt(fluxSum / binCount) / (totalPower * 0.01 + 1e-10) * 0.3, 1.0) : 0;

  // ── spread — air / breath (10kHz+) ───────────────────────────────────────────
  const spread = (safe && voiced) ? Math.min(airE / totalPower * 20.0, 1.0) : 0;

  // ── F0 — fundamental frequency via autocorrelation (80–800Hz) ────────────────
  // returns 0 when silent or no clear pitch found
  const AC_WIN = 512;
  const acLo   = Math.floor(44100 / 800);
  const acHi   = Math.min(Math.floor(44100 / 80), AC_WIN - 1);
  let bestLag = 0, bestCorr = -Infinity;
  for(let lag = acLo; lag <= acHi; lag++){
    let c = 0;
    for(let i = 0; i < AC_WIN - lag; i++) c += timeData[i] * timeData[i + lag];
    if(c > bestCorr){ bestCorr = c; bestLag = lag; }
  }
  // confidence: ratio of best correlation to zero-lag (self-correlation)
  let zeroLag = 0;
  for(let i = 0; i < AC_WIN; i++) zeroLag += timeData[i] * timeData[i];
  const f0confidence = (voiced && zeroLag > 1e-10)
    ? Math.min(bestCorr / zeroLag, 1.0) : 0;
  const f0raw = (voiced && bestLag > 0 && f0confidence > 0.15) ? 44100 / bestLag : 0;
  // normalise F0 to 0-1 over 80–800Hz range on a log scale (perceptually linear)
  const f0 = f0raw > 0
    ? Math.min(Math.max((Math.log2(f0raw) - Math.log2(80)) / Math.log2(800/80), 0), 1) : 0;

  // ── F0 delta — how fast pitch is moving frame-to-frame ───────────────────────
  const f0deltaRaw = Math.abs(f0 - (fftSmooth.f0 || 0));
  const f0delta = voiced ? Math.min(f0deltaRaw * 20.0, 1.0) : 0;

  // ── spectral tilt — slope of energy from low to high ─────────────────────────
  // negative = dark (bass-heavy), positive = bright; remap to 0-1 around 0.5
  const lowE  = bandEnergy(b.bass0, b.mid0);   // 80–300Hz
  const highE = bandEnergy(b.pres0, b.hi1);    // 2–10kHz
  const tiltRaw = (safe && voiced && (lowE + highE) > 1e-10)
    ? highE / (lowE + highE) : 0.5;
  const tilt = voiced ? tiltRaw : 0;

  // ── spectral spread — how wide energy is distributed around the centroid ──────
  const centroid = (safe && voiced) ? wSum / totalPower : 0;
  let spreadSum = 0;
  for(let i = 0; i < binCount; i++) spreadSum += power[i] * Math.pow(i * binHz - centroid, 2);
  const spectralSpread = (safe && voiced)
    ? Math.min(Math.sqrt(spreadSum / (totalPower + 1e-10)) / 4000.0, 1.0) : 0;

  // ── EMA smooth all signals ────────────────────────────────────────────────────
  const raw = {
    volume, bass, midrange, presence, brightness, tone, texture, movement,
    spread, f0, f0delta, f0confidence, tilt, spectralSpread,
  };
  for(const k of SIGNALS){
    fftSmooth[k] = FFT_alpha * raw[k] + (1 - FFT_alpha) * fftSmooth[k];
  }

  // ── write into targets via routing table ──────────────────────────────────────
  const clamp01 = v => Math.min(1, Math.max(0, v));
  const shapAccum   = {};
  const moodAccum   = {};
  const colourAccum = {};

  for(const sig of SIGNALS){
    const { dst, mul } = fftRoute[sig];
    if(!dst || dst === 'none') continue;
    // negative mul inverts: high signal → low target, scaled by |mul|
    const val = mul < 0
      ? (1 - fftSmooth[sig]) * (-mul)
      : fftSmooth[sig] * mul;
    if(SHAPE_KEYS.includes(dst)){
      shapAccum[dst] = (shapAccum[dst] ?? 0) + val;
    } else if(MOOD_KEYS_POOL.includes(dst)){
      moodAccum[dst] = (moodAccum[dst] ?? 0) + val;
    } else if(COLOUR_KEYS_POOL.includes(dst)){
      colourAccum[dst] = (colourAccum[dst] ?? 0) + val;
    }
  }

  const lock = window.gfxLock || {};
  // merge emotion→shape contributions so they don't race against FFT writes
  const emotShape = window._emotionShapeAccum || {};
  for(const k of SHAPE_KEYS){
    if(lock[k]) continue;
    const fftVal   = shapAccum[k];
    const emotVal  = emotShape[k];
    if(fftVal !== undefined || emotVal !== undefined){
      fftTarget[k] = clamp01((fftVal ?? 0) + (emotVal ?? 0));
    }
  }
  // strip locked keys before handing accum to _globalTick
  for(const k of Object.keys(moodAccum))   { if(lock[k]) delete moodAccum[k]; }
  for(const k of Object.keys(colourAccum)) { if(lock[k]) delete colourAccum[k]; }
  window._fftMoodAccum = { ...moodAccum, ...colourAccum };

  // ── waveform bar — only paint when knobs panel is open ───────────────────────
  const _knobs = document.getElementById('knobs');
  if(vizCtx && vizCanvas && _knobs && !_knobs.classList.contains('hidden')){
    const W = vizCanvas.clientWidth  || vizCanvas.width;
    const H = vizCanvas.clientHeight || vizCanvas.height;
    vizCanvas.width  = W;
    vizCanvas.height = H;
    vizCtx.fillStyle = 'rgba(0,0,0,0.3)';
    vizCtx.fillRect(0, 0, W, H);
    vizCtx.beginPath();
    vizCtx.strokeStyle = `rgba(255,255,255,${0.3 + volume * 0.5})`;
    vizCtx.lineWidth   = 1.2;
    const step = W / timeData.length;
    for(let i = 0; i < timeData.length; i++){
      const x = i * step;
      const y = H / 2 + timeData[i] * H * 0.45;
      i === 0 ? vizCtx.moveTo(x, y) : vizCtx.lineTo(x, y);
    }
    vizCtx.stroke();
  }
}
