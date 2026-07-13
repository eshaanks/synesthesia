// ── all uniform state lives here — targets written by sources, lerped each frame

// emotion probability targets — set by server chunks (0-1 each, sum ~1)
const emotTarget  = { neu:0.25, hap:0.25, ang:0.25, sad:0.25 };
const emotCurrent = { neu:0.25, hap:0.25, ang:0.25, sad:0.25 };
window.emotionCurrent = emotCurrent;

// FFT shape targets — set by visualizer.js each frame
window.fftTarget  = { brightness:0.5, tone:0.3, movement:0.1, texture:0.2, volume:0.2, bass:0.5, spread:0.5 };
window.fftCurrent = { brightness:0.5, tone:0.3, movement:0.1, texture:0.2, volume:0.2, bass:0.5, spread:0.5 };

// mood targets — set by routing or locked sliders
window.moodTarget  = { colorTemp:0.5, speed:1.0, blobSize:2.2, fog:0.0, saturation:1.0 };
window.moodCurrent = { colorTemp:0.5, speed:1.0, blobSize:2.2, fog:0.0, saturation:1.0 };

// lerp speeds
let EMOTION_SPEED = 0.008;
let FFT_LERP      = 0.12;
let MOOD_LERP     = 0.02;

function slerp(a, b, t){
  const s = t * t * t * (t * (t * 6 - 15) + 10);
  return a + (b - a) * s;
}

// called by mic.js with new server probs — exported so index.html can wrap it
window.updateFromEmotion = function updateFromEmotion(probs){
  for(const k of Object.keys(emotTarget)){
    if(probs[k] !== undefined) emotTarget[k] = probs[k];
  }
};

// ── render loop ───────────────────────────────────────────────────────────────
let startTime = null;
let lastTs    = null;

const FFT_KEYS  = Object.keys(window.fftTarget);
const MOOD_KEYS = Object.keys(window.moodTarget);
const EMOT_KEYS = Object.keys(emotTarget);

function renderLoop(ts){
  if(!startTime) startTime = ts;
  const t  = (ts - startTime) * 0.001;
  const dt = lastTs ? Math.min((ts - lastTs) * 0.001, 0.05) : 0.016;
  lastTs   = ts;

  const fftFactor  = 1.0 - Math.pow(1.0 - FFT_LERP,      dt * 60);
  const moodFactor = 1.0 - Math.pow(1.0 - MOOD_LERP,     dt * 60);
  const emotFactor = 1.0 - Math.pow(1.0 - EMOTION_SPEED, dt * 60);

  if(typeof window._globalTick === 'function') window._globalTick(ts);

  for(const k of FFT_KEYS)  window.fftCurrent[k]  = slerp(window.fftCurrent[k],  window.fftTarget[k],  fftFactor);
  for(const k of MOOD_KEYS) window.moodCurrent[k]  = slerp(window.moodCurrent[k], window.moodTarget[k], moodFactor);
  for(const k of EMOT_KEYS) emotCurrent[k]         = slerp(emotCurrent[k],        emotTarget[k],        emotFactor);

  gl.useProgram(prog);
  gl.uniform1f(U.time,       t);

  gl.uniform1f(U.neu,        emotCurrent.neu);
  gl.uniform1f(U.hap,        emotCurrent.hap);
  gl.uniform1f(U.ang,        emotCurrent.ang);
  gl.uniform1f(U.sad,        emotCurrent.sad);

  gl.uniform1f(U.brightness, window.fftCurrent.brightness);
  gl.uniform1f(U.tone,       window.fftCurrent.tone);
  gl.uniform1f(U.movement,   window.fftCurrent.movement);
  gl.uniform1f(U.texture,    window.fftCurrent.texture);
  gl.uniform1f(U.volume,     window.fftCurrent.volume);
  gl.uniform1f(U.bass,       window.fftCurrent.bass);
  gl.uniform1f(U.spread,     window.fftCurrent.spread);

  gl.uniform1f(U.colorTemp,  window.moodCurrent.colorTemp);
  gl.uniform1f(U.speed,      window.moodCurrent.speed);
  gl.uniform1f(U.blobSize,   window.moodCurrent.blobSize);
  gl.uniform1f(U.fog,        window.moodCurrent.fog);
  gl.uniform1f(U.saturation, window.moodCurrent.saturation);

  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  requestAnimationFrame(renderLoop);
}

requestAnimationFrame(renderLoop);
