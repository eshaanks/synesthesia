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
  volume:0.2,   bass:0.5,   sub:0.3,      midrange:0.4,
  presence:0.3, brightness:0.5, tone:0.3, clarity:0.3,
  texture:0.2,  movement:0.1,   attack:0.0, spread:0.5,
};
let FFT_alpha = 0.08;

// ── all signals that appear in the routing panel ──────────────────────────────
const SIGNALS = [
  'volume','bass','sub','midrange',
  'presence','brightness','tone','clarity',
  'texture','movement','attack','spread',
];

// ── all graphical targets (shape + mood) in one flat pool ────────────────────
// shape targets live in fftTarget; mood targets live in moodTarget
// the routing write loop handles both
const SHAPE_KEYS = ['brightness','tone','movement','texture','volume','bass','spread'];
const MOOD_KEYS_POOL = ['colorTemp','speed','blobSize','fog','saturation'];
const ALL_GFX_TARGETS = [...SHAPE_KEYS, ...MOOD_KEYS_POOL];

// ── default routing — each signal maps to the gfx target of the same name or none
const fftRoute = {
  volume:     { dst:'volume',     mul:1 },
  bass:       { dst:'bass',       mul:1 },
  sub:        { dst:'none',       mul:1 },
  midrange:   { dst:'none',       mul:1 },
  presence:   { dst:'none',       mul:1 },
  brightness: { dst:'brightness', mul:1 },
  tone:       { dst:'tone',       mul:1 },
  clarity:    { dst:'none',       mul:1 },
  texture:    { dst:'texture',    mul:1 },
  movement:   { dst:'movement',   mul:1 },
  attack:     { dst:'none',       mul:1 },
  spread:     { dst:'spread',     mul:1 },
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

function fftLoop(){
  if(!analyser) return;
  fftAnimId = requestAnimationFrame(fftLoop);

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

  const subE  = bandEnergy(b.sub0,  b.sub1);
  const bassE = bandEnergy(b.bass0, b.bass1);
  const midE  = bandEnergy(b.mid0,  b.mid1);
  const presE = bandEnergy(b.pres0, b.pres1);
  bandEnergy(b.hi0, b.hi1); // hi band reserved for future use
  const airE  = bandEnergy(b.air0,  binCount-1);

  // ── volume — RMS ─────────────────────────────────────────────────────────────
  let rmsSum = 0;
  for(let i = 0; i < N; i++) rmsSum += timeData[i] * timeData[i];
  const volume = Math.min(Math.sqrt(rmsSum / N) * 8.0, 1.0);

  // ── sub — deep rumble (20-80Hz) ───────────────────────────────────────────────
  const sub = safe ? Math.min(subE / totalPower * 12.0, 1.0) : 0;

  // ── bass — low body (80-300Hz) ────────────────────────────────────────────────
  const bass = safe ? Math.min(bassE / totalPower * 6.0, 1.0) : 0.5;

  // ── midrange — voice body (300-2kHz) ─────────────────────────────────────────
  const midrange = safe ? Math.min(midE / totalPower * 2.5, 1.0) : 0.4;

  // ── presence — attack/consonants (2-5kHz) ────────────────────────────────────
  const presence = safe ? Math.min(presE / totalPower * 8.0, 1.0) : 0.3;

  // ── brightness — spectral centroid (where the energy lives) ──────────────────
  let wSum = 0;
  for(let i = 0; i < binCount; i++) wSum += power[i] * i * binHz;
  const brightness = safe ? Math.min(wSum / (totalPower * nyquist), 1.0) : 0.5;

  // ── tone — spectral flatness (pitched vs noise) ───────────────────────────────
  let logSum = 0, linSum2 = 0, vBins = 0;
  for(let i = 1; i < binCount; i++){
    if(power[i] > 1e-12){ logSum += Math.log(power[i]); linSum2 += power[i]; vBins++; }
  }
  const tone = (vBins > 0 && linSum2 > 0)
    ? Math.min(Math.exp(logSum / vBins) / (linSum2 / vBins) * 10.0, 1.0) : 0;

  // ── clarity — high-mid to noise ratio, how clear/defined the voice is ────────
  // presence relative to overall flatness — high when voice is clean and forward
  const clarity = safe ? Math.min((presE / (totalPower + 1e-10)) * (1.0 - tone) * 15.0, 1.0) : 0;

  // ── texture — zero crossing rate (smooth vs raspy/fricative) ─────────────────
  let zcCount = 0;
  for(let i = 1; i < N; i++){
    if((timeData[i] >= 0) !== (timeData[i-1] >= 0)) zcCount++;
  }
  const texture = Math.min(zcCount / N * 10.0, 1.0);

  // ── movement — spectral flux (rate of change, frame-to-frame) ────────────────
  let fluxSum = 0;
  for(let i = 0; i < binCount; i++){
    const d = power[i] - prevFreq[i];
    fluxSum += d * d;
    prevFreq[i] = power[i];
  }
  const movement = safe
    ? Math.min(Math.sqrt(fluxSum / binCount) / (totalPower * 0.01 + 1e-10) * 0.3, 1.0) : 0;

  // ── attack — onset sharpness (sudden volume jump) ─────────────────────────────
  const attack = Math.max(0, Math.min((volume - prevVolume) * 6.0, 1.0));
  prevVolume   = volume;

  // ── spread — air/sibilance (10kHz+) ──────────────────────────────────────────
  const spread = safe ? Math.min(airE / totalPower * 20.0, 1.0) : 0.5;

  // ── EMA smooth all signals ────────────────────────────────────────────────────
  const raw = { volume, bass, sub, midrange, presence, brightness, tone, clarity, texture, movement, attack, spread };
  for(const k of SIGNALS){
    fftSmooth[k] = FFT_alpha * raw[k] + (1 - FFT_alpha) * fftSmooth[k];
  }

  // ── write into targets via routing table ──────────────────────────────────────
  const clamp01 = v => Math.min(1, Math.max(0, v));
  const shapAccum = {};
  const moodAccum = {};

  for(const sig of SIGNALS){
    const { dst, mul } = fftRoute[sig];
    if(!dst || dst === 'none') continue;
    const val = fftSmooth[sig] * mul;
    if(SHAPE_KEYS.includes(dst)){
      shapAccum[dst] = (shapAccum[dst] ?? 0) + val;
    } else if(MOOD_KEYS_POOL.includes(dst)){
      moodAccum[dst] = (moodAccum[dst] ?? 0) + val;
    }
  }

  for(const k of SHAPE_KEYS){
    if(shapAccum[k] !== undefined) fftTarget[k] = clamp01(shapAccum[k]);
  }
  // mood accumulations are picked up by _globalTick in index.html
  window._fftMoodAccum = moodAccum;

  // ── waveform bar at bottom ────────────────────────────────────────────────────
  if(vizCtx && vizCanvas){
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
