// ── all uniform state lives here — targets written by sources, lerped each frame
// emotion targets — set by server chunks
let targetValence   = 0.0, targetArousal   = 0.0, targetDominance = 0.0;
let currentValence  = 0.0, currentArousal  = 0.0, currentDominance = 0.0;

// FFT shape targets — set by visualizer.js each frame
window.fftTarget  = { brightness:0.5, tone:0.3, movement:0.1, texture:0.2, volume:0.2, bass:0.5, spread:0.5 };
window.fftCurrent = { brightness:0.5, tone:0.3, movement:0.1, texture:0.2, volume:0.2, bass:0.5, spread:0.5 };

// mood targets — set by model routing or sliders
window.moodTarget  = { colorTemp:0.5, speed:1.0, blobSize:2.2, fog:0.0, saturation:1.0 };
window.moodCurrent = { colorTemp:0.5, speed:1.0, blobSize:2.2, fog:0.0, saturation:1.0 };

// expose emotion current values for meters
window.emotionCurrent = { valence:0.0, arousal:0.0, dominance:0.0 };

// lerp speeds — all exposed as knobs
let EMOTION_SPEED = 0.004;   // very slow — mood changes take seconds
let FFT_LERP      = 0.12;    // fast — audio response feels instant but not jittery
let MOOD_LERP     = 0.02;    // medium — mood param changes feel deliberate

// smootherstep ease — bell-curve velocity so transitions don't feel mechanical
function slerp(a, b, t){
  const s = t * t * t * (t * (t * 6 - 15) + 10);
  return a + (b - a) * s;
}

function updateFromEmotion(valence, arousal, dominance){
  targetValence   = valence;
  targetArousal   = arousal;
  targetDominance = dominance;
}

// ── render loop ───────────────────────────────────────────────────────────────
let startTime = null;
let lastTs    = null;

const FFT_KEYS  = Object.keys(window.fftTarget);
const MOOD_KEYS = Object.keys(window.moodTarget);

function renderLoop(ts){
  if(!startTime) startTime = ts;
  const t  = (ts - startTime) * 0.001;
  const dt = lastTs ? Math.min((ts - lastTs) * 0.001, 0.05) : 0.016; // cap at 50ms
  lastTs   = ts;

  // normalise speeds to 60fps so behaviour is frame-rate independent
  // at 60fps dt=0.0167 → factor=1.0, at 30fps dt=0.033 → factor≈2 → same visual speed
  const fftFactor   = 1.0 - Math.pow(1.0 - FFT_LERP,    dt * 60);
  const moodFactor  = 1.0 - Math.pow(1.0 - MOOD_LERP,   dt * 60);
  const emotFactor  = 1.0 - Math.pow(1.0 - EMOTION_SPEED, dt * 60);

  // let sources write their targets first
  if(typeof window._globalTick === 'function') window._globalTick(ts);

  // lerp all FFT uniforms toward their targets
  for(const k of FFT_KEYS){
    window.fftCurrent[k] = slerp(window.fftCurrent[k], fftTarget[k], fftFactor);
  }

  // lerp all mood uniforms
  for(const k of MOOD_KEYS){
    window.moodCurrent[k] = slerp(window.moodCurrent[k], moodTarget[k], moodFactor);
  }

  // lerp emotion scalars
  currentValence   = slerp(currentValence,   targetValence,   emotFactor);
  currentArousal   = slerp(currentArousal,   targetArousal,   emotFactor);
  currentDominance = slerp(currentDominance, targetDominance, emotFactor);

  // expose lerped values for meters
  emotionCurrent.valence   = currentValence;
  emotionCurrent.arousal   = currentArousal;
  emotionCurrent.dominance = currentDominance;

  // push everything to GPU
  gl.useProgram(prog);
  gl.uniform1f(U.time,        t);
  gl.uniform1f(U.valence,     currentValence);
  gl.uniform1f(U.arousal,     currentArousal);
  gl.uniform1f(U.dominance,   currentDominance);

  gl.uniform1f(U.brightness,  window.fftCurrent.brightness);
  gl.uniform1f(U.tone,        window.fftCurrent.tone);
  gl.uniform1f(U.movement,    window.fftCurrent.movement);
  gl.uniform1f(U.texture,     window.fftCurrent.texture);
  gl.uniform1f(U.volume,      window.fftCurrent.volume);
  gl.uniform1f(U.bass,        window.fftCurrent.bass);
  gl.uniform1f(U.spread,      window.fftCurrent.spread);

  gl.uniform1f(U.colorTemp,   window.moodCurrent.colorTemp);
  gl.uniform1f(U.speed,       window.moodCurrent.speed);
  gl.uniform1f(U.blobSize,    window.moodCurrent.blobSize);
  gl.uniform1f(U.fog,         window.moodCurrent.fog);
  gl.uniform1f(U.saturation,  window.moodCurrent.saturation);

  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  requestAnimationFrame(renderLoop);
}

requestAnimationFrame(renderLoop);
