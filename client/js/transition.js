// ── all uniform state lives here — targets written by sources, lerped each frame

// emotion probability targets — set by server chunks (0-1 each, sum ~1)
const emotTarget  = { neu:0.25, hap:0.25, ang:0.25, sad:0.25 };
const emotCurrent = { neu:0.25, hap:0.25, ang:0.25, sad:0.25 };
window.emotTarget    = emotTarget;
window.emotionCurrent = emotCurrent;

// VAD position targets — drive the 2D colour wheel directly
window.vadTarget  = { v:0.5, a:0.4 };
window.vadCurrent = { v:0.5, a:0.4 };

// FFT shape targets — set by visualizer.js each frame
window.fftTarget  = { brightness:0.5, tone:0.3, movement:0.1, texture:0.2, volume:0.2, bass:0.5, spread:0.5 };
window.fftCurrent = { brightness:0.5, tone:0.3, movement:0.1, texture:0.2, volume:0.2, bass:0.5, spread:0.5 };

// mood targets — set by routing or locked sliders
window.moodTarget  = { colorTemp:0.5, speed:1.0, blobSize:2.2, fog:0.0, saturation:1.0 };
window.moodCurrent = { colorTemp:0.5, speed:1.0, blobSize:2.2, fog:0.0, saturation:1.0 };

// lerp speeds
let EMOTION_SPEED = 0.06;
let FFT_LERP      = 0.12;
let MOOD_LERP     = 0.02;
// expose so index.html slider oninput assignments reach these variables
Object.defineProperty(window, 'EMOTION_SPEED', { get:()=>EMOTION_SPEED, set:v=>{ EMOTION_SPEED=v; } });
Object.defineProperty(window, 'FFT_LERP',      { get:()=>FFT_LERP,      set:v=>{ FFT_LERP=v; } });
Object.defineProperty(window, 'MOOD_LERP',     { get:()=>MOOD_LERP,     set:v=>{ MOOD_LERP=v; } });

function slerp(a, b, t){
  const s = t * t * t * (t * (t * 6 - 15) + 10);
  return a + (b - a) * s;
}

// updateFromEmotion is defined in index.html (needs _forcedEmotion + routing)
// render loop ─────────────────────────────────────────────────────────────────
let startTime = null;
let lastTs    = null;
let _animTime = 0;   // integrated speed*dt — avoids jump when speed changes

const FFT_KEYS  = Object.keys(window.fftTarget);
const MOOD_KEYS = Object.keys(window.moodTarget);
const EMOT_KEYS = Object.keys(emotTarget);

function renderLoop(ts){
  if(!startTime) startTime = ts;
  const dt = lastTs ? Math.min((ts - lastTs) * 0.001, 0.05) : 0.016;
  lastTs   = ts;

  const emotFactor = 1.0 - Math.pow(1.0 - EMOTION_SPEED, dt * 60);

  if(typeof window._globalTick === 'function') window._globalTick(ts);

  const _lock   = window.gfxLock   || {};
  const _manual = window.gfxManual || {};
  const _lerp   = window.gfxLerp   || {};

  for(const k of FFT_KEYS){
    if(_lock[k]){
      window.fftCurrent[k] = _manual[k] ?? window.fftCurrent[k];
    } else {
      const speed = _lerp[k];
      if(speed === null){
        window.fftCurrent[k] = window.fftTarget[k];   // explicit snap
      } else {
        const f = 1.0 - Math.pow(1.0 - (speed ?? FFT_LERP), dt * 60);
        window.fftCurrent[k] = slerp(window.fftCurrent[k], window.fftTarget[k], f);
      }
    }
  }
  // mood: pushMoodUniforms writes moodTarget/moodCurrent for locked keys
  for(const k of MOOD_KEYS){
    if(!_lock[k]){
      const speed = _lerp[k] ?? MOOD_LERP;
      const f = 1.0 - Math.pow(1.0 - speed, dt * 60);
      window.moodCurrent[k] = slerp(window.moodCurrent[k], window.moodTarget[k], f);
    }
  }
  for(const k of EMOT_KEYS) emotCurrent[k] = slerp(emotCurrent[k], emotTarget[k], emotFactor);
  if(!window.vadLock){
    window.vadCurrent.v = slerp(window.vadCurrent.v, window.vadTarget.v, emotFactor);
    window.vadCurrent.a = slerp(window.vadCurrent.a, window.vadTarget.a, emotFactor);
  }

  // accumulate animation time at the current (post-lerp) speed so that
  // changing u_speed never causes a discontinuous jump in noise coordinates
  _animTime += dt * window.moodCurrent.speed;

  gl.useProgram(prog);
  gl.uniform1f(U.time,       _animTime);

  gl.uniform1f(U.valence,    window.vadCurrent.v);
  gl.uniform1f(U.arousal,   window.vadCurrent.a);

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
  if(typeof window._onTypeFrame   === 'function') window._onTypeFrame(ts);
  if(typeof window._onRenderFrame === 'function') window._onRenderFrame();
  requestAnimationFrame(renderLoop);
}

requestAnimationFrame(renderLoop);
