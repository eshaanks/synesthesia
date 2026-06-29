// ── emotion state — lerped slowly from server updates ─────────────────────────
let currentValence   = 0.0, targetValence   = 0.0;
let currentArousal   = 0.0, targetArousal   = 0.0;
let currentDominance = 0.0, targetDominance = 0.0;

const EMOTION_SPEED = 0.018;  // slow drift — emotion mood settles over ~3s
function lerp(a, b, t){ return a + (b - a) * t; }

// called by mic.js every ~2s when server returns new VAD values
function updateFromEmotion(valence, arousal, dominance){
  targetValence   = valence;
  targetArousal   = arousal;
  targetDominance = dominance;
}

// ── render loop ───────────────────────────────────────────────────────────────
let startTime = null;

function renderLoop(ts){
  if(!startTime) startTime = ts;
  const t = (ts - startTime) * 0.001;

  // lerp emotion scalars toward targets
  currentValence   = lerp(currentValence,   targetValence,   EMOTION_SPEED);
  currentArousal   = lerp(currentArousal,   targetArousal,   EMOTION_SPEED);
  currentDominance = lerp(currentDominance, targetDominance, EMOTION_SPEED);

  // push time + emotion uniforms — FFT uniforms pushed by fft.js each frame
  gl.useProgram(prog);
  gl.uniform1f(U.time,      t);
  gl.uniform1f(U.valence,   currentValence);
  gl.uniform1f(U.arousal,   currentArousal);
  gl.uniform1f(U.dominance, currentDominance);

  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

  requestAnimationFrame(renderLoop);
}

requestAnimationFrame(renderLoop);
