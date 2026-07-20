// ── WebGL setup ───────────────────────────────────────────────────────────────
const canvas   = document.getElementById('glCanvas');
let RENDER_W = 1280;
let RENDER_H = 720;
canvas.width   = RENDER_W;
canvas.height  = RENDER_H;

const gl = canvas.getContext('webgl');
gl.viewport(0, 0, RENDER_W, RENDER_H);

// called from UI to change render resolution without reload
window.setResolution = function(w, h){
  RENDER_W = w; RENDER_H = h;
  canvas.width  = w; canvas.height = h;
  gl.viewport(0, 0, w, h);
  gl.useProgram(prog);
  gl.uniform2f(U.resolution, w, h);
};

// ── vertex shader (shared) ────────────────────────────────────────────────────
const VS = `
attribute vec2 a_pos;
varying   vec2 v_uv;
void main(){
  v_uv        = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

// ── Rorschach fragment shader ─────────────────────────────────────────────────
// Layer 1 — FFT uniforms (instant, every frame from Web Audio API):
//   u_centroid   spectral centroid 0-1   → centre-of-mass shift
//   u_flatness   spectral flatness 0-1   → structured vs noisy warp
//   u_flux       frame-to-frame change   → mutation speed
//   u_zcr        zero crossing rate 0-1  → edge sharpness
//   u_rms        RMS energy 0-1          → brightness pulse
//   u_bassRatio  bass/treble ratio 0-1   → warp depth
//   u_rotSpeed   F0-derived rotation     → pattern rotation speed
//
// Layer 2 — VAD + colour wheel (every ~2s from WavLM server):
//   u_valence/u_arousal  live VAD position → samples the 4-pole colour field
//   u_r[0-5]pos/rad/col0/col1  6 draggable palette regions in VAD space

const FS_RORSCHACH = `
precision highp float;
varying vec2 v_uv;

// time
uniform float u_time;
uniform vec2  u_resolution;

// FFT layer — instant audio response
uniform float u_brightness;   // spectral centroid — thin/sharp vs warm/thick
uniform float u_tone;         // spectral flatness — singing/humming vs whispering
uniform float u_movement;     // spectral flux — how fast sound is changing
uniform float u_texture;      // zero crossing rate — smooth vs grainy/fricative
uniform float u_volume;       // RMS energy — loudness
uniform float u_bass;         // bass/treble ratio — low end vs high end
uniform float u_pitch;        // F0 estimate — fundamental frequency

// VAD position (from model)
uniform float u_valence;     // 0=negative → 1=positive
uniform float u_arousal;     // 0=calm → 1=active

// 6 emotion regions — hard-edged circles in VAD space
// each region has 3 flat colours; ink noise picks which colour is visible at each point
// pos=(valence,arousal), rad=radius in VAD space (scaled by u_colorRadius)
uniform vec2  u_r0pos; uniform float u_r0rad; uniform vec3 u_r0col0; uniform vec3 u_r0col1; uniform vec3 u_r0col2; // sad
uniform vec2  u_r1pos; uniform float u_r1rad; uniform vec3 u_r1col0; uniform vec3 u_r1col1; uniform vec3 u_r1col2; // happy
uniform vec2  u_r2pos; uniform float u_r2rad; uniform vec3 u_r2col0; uniform vec3 u_r2col1; uniform vec3 u_r2col2; // angry
uniform vec2  u_r3pos; uniform float u_r3rad; uniform vec3 u_r3col0; uniform vec3 u_r3col1; uniform vec3 u_r3col2; // happy+aroused
uniform vec2  u_r4pos; uniform float u_r4rad; uniform vec3 u_r4col0; uniform vec3 u_r4col1; uniform vec3 u_r4col2; // dominant aroused
uniform vec2  u_r5pos; uniform float u_r5rad; uniform vec3 u_r5col0; uniform vec3 u_r5col1; uniform vec3 u_r5col2; // calm

// global colour modifiers (routable to FFT)
uniform float u_colorRadius; // multiplier on all region radii
uniform float u_hueShift;    // slides noise thresholds — rotates which colour patch dominates
uniform float u_colorBand;   // 0-1: split between band0 and band1 (0.5=equal thirds)

// FFT shape param
uniform float u_spread;      // how wide/open the blob arms splay

// global mood params — driven by model scalars, pure atmosphere
uniform float u_colorTemp;   // 0=cold blue  1=warm gold
uniform float u_speed;       // ebb/flow animation speed
uniform float u_blobSize;    // overall scale / zoom
uniform float u_fog;         // edge mist / vignette fog
uniform float u_saturation;  // 0=greyscale  1=vivid

// ── noise primitives ──────────────────────────────────────────────────────────
float hash(vec2 p){
  p = fract(p * vec2(127.1, 311.7));
  p += dot(p, p + 19.19);
  return fract(p.x * p.y);
}

float vnoise(vec2 p){
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f*f*(3.0-2.0*f);
  return mix(
    mix(hash(i),             hash(i+vec2(1,0)), u.x),
    mix(hash(i+vec2(0,1)),   hash(i+vec2(1,1)), u.x),
    u.y);
}

// domain-warped noise — warp depth driven by bass, mutation by flux
// u_time is already pre-scaled by speed in JS, so no multiply here
float inkNoise(vec2 p, float warpDepth, float mutRate){
  float t  = u_time * (0.02 + mutRate * 0.5);
  vec2  q  = vec2(vnoise(p + vec2(0.0, 0.0) + t),
                  vnoise(p + vec2(5.2, 1.3) + t));
  vec2  r  = vec2(vnoise(p + warpDepth * q + vec2(1.7, 9.2) + t * 0.7),
                  vnoise(p + warpDepth * q + vec2(8.3, 2.8) + t * 0.5));
  vec2  s  = vec2(vnoise(p + warpDepth * r + vec2(0.3, 5.1) + t * 0.3),
                  vnoise(p + warpDepth * r + vec2(4.1, 3.7) + t * 0.3));
  // texture adds high-freq grain — visible range 0.0 (smooth) to 1.0 (coarse)
  float detail = vnoise(p * 6.0 + s * 1.5 + t * 0.4) * u_texture;
  return vnoise(p + 3.0*q + 2.0*r + s) + detail * 0.35;
}

// ── seamless colour field ────────────────────────────────────────────────────
// 6 VAD regions, each with 3 colours.
// n (0-1, slow coarse noise) smoothly mixes between the 3 colours — no thresholds.
// u_colorBand controls spread: 0=only col0, 1=full sweep col0→col1→col2.
// u_hueShift rotates the phase of n so the colour cycle shifts.
// Winner-takes-all on nearest VAD region — no inter-region colour blending.
vec3 colourField(float v, float a, float n){
  vec2 vadPos = vec2(v, a);

  vec2  rpos[6]; float rrad[6];
  vec3  rc0[6];  vec3  rc1[6];  vec3  rc2[6];

  rpos[0]=u_r0pos; rrad[0]=max(u_r0rad*u_colorRadius,0.01); rc0[0]=u_r0col0; rc1[0]=u_r0col1; rc2[0]=u_r0col2;
  rpos[1]=u_r1pos; rrad[1]=max(u_r1rad*u_colorRadius,0.01); rc0[1]=u_r1col0; rc1[1]=u_r1col1; rc2[1]=u_r1col2;
  rpos[2]=u_r2pos; rrad[2]=max(u_r2rad*u_colorRadius,0.01); rc0[2]=u_r2col0; rc1[2]=u_r2col1; rc2[2]=u_r2col2;
  rpos[3]=u_r3pos; rrad[3]=max(u_r3rad*u_colorRadius,0.01); rc0[3]=u_r3col0; rc1[3]=u_r3col1; rc2[3]=u_r3col2;
  rpos[4]=u_r4pos; rrad[4]=max(u_r4rad*u_colorRadius,0.01); rc0[4]=u_r4col0; rc1[4]=u_r4col1; rc2[4]=u_r4col2;
  rpos[5]=u_r5pos; rrad[5]=max(u_r5rad*u_colorRadius,0.01); rc0[5]=u_r5col0; rc1[5]=u_r5col1; rc2[5]=u_r5col2;

  // smooth 3-colour mix: t sweeps col0→col1→col2 with no hard cuts
  float t = fract(n + u_hueShift) * u_colorBand;
  #define SMIX(c0,c1,c2) (t < 0.5 ? mix(c0, c1, t*2.0) : mix(c1, c2, (t-0.5)*2.0))

  // nearest-region always wins — no hard radius cutoff so colour never goes black
  // u_colorRadius softens the distance weighting: low=sharp region edges, high=blended
  float bestScore = -1.0;
  vec3  bestCol   = vec3(0.0);
  float d; float score;

  d=length(vadPos-rpos[0]); score=1.0/(d/max(u_colorRadius,0.01)+0.001); if(score>bestScore){ bestScore=score; bestCol=SMIX(rc0[0],rc1[0],rc2[0]); }
  d=length(vadPos-rpos[1]); score=1.0/(d/max(u_colorRadius,0.01)+0.001); if(score>bestScore){ bestScore=score; bestCol=SMIX(rc0[1],rc1[1],rc2[1]); }
  d=length(vadPos-rpos[2]); score=1.0/(d/max(u_colorRadius,0.01)+0.001); if(score>bestScore){ bestScore=score; bestCol=SMIX(rc0[2],rc1[2],rc2[2]); }
  d=length(vadPos-rpos[3]); score=1.0/(d/max(u_colorRadius,0.01)+0.001); if(score>bestScore){ bestScore=score; bestCol=SMIX(rc0[3],rc1[3],rc2[3]); }
  d=length(vadPos-rpos[4]); score=1.0/(d/max(u_colorRadius,0.01)+0.001); if(score>bestScore){ bestScore=score; bestCol=SMIX(rc0[4],rc1[4],rc2[4]); }
  d=length(vadPos-rpos[5]); score=1.0/(d/max(u_colorRadius,0.01)+0.001); if(score>bestScore){ bestScore=score; bestCol=SMIX(rc0[5],rc1[5],rc2[5]); }

  #undef SMIX

  return bestCol;
}

void main(){
  float aspect = u_resolution.x / u_resolution.y;

  // centre, aspect-correct
  vec2 uv  = v_uv - 0.5;
  uv.x    *= aspect;

  // brightness shifts centre-of-mass vertically — full 0-1 sweeps half the canvas height
  uv.y += (u_brightness - 0.5) * 0.9;

  // spread: squeezes or splays the x-axis — 0=needle-thin, 1=very wide
  uv.x *= 0.2 + u_spread * 2.2;

  // ── bilateral symmetry ────────────────────────────────────────────────────
  vec2 uvL = vec2(-abs(uv.x), uv.y);

  float baseScale = max(u_blobSize, 0.1);
  vec2 pL = uvL * baseScale;

  // warp depth — 0=almost no warp (clean geometric), 1=heavily folded organic mess
  float warpDepth = 0.3 + u_bass * 9.0;

  // tone: 0=tight form, 1=slightly softer edge
  float thresh = 0.64 - u_tone * 0.18;
  float edgeW  = 0.03 + u_tone * 0.05;

  // single ink field — shape as before
  float ink = inkNoise(pL, warpDepth, u_movement);

  // colour noise: very coarse scale + very slow drift → smooth colour wash
  // across the form with no visible frequency / contour lines
  float colNoise = vnoise(pL * 0.18 + vec2(3.1, 7.6) + u_time * 0.008);
  vec3  col      = colourField(u_valence, u_arousal, colNoise);

  // soft narrow band: lit inside, black outside — like a neon tube
  float density = smoothstep(thresh - edgeW, thresh + edgeW, ink);

  // glass highlight: the very peak of the form reads slightly brighter/whiter
  float core = pow(clamp((ink - thresh) / 0.07 + 0.5, 0.0, 1.0), 3.0) * 0.4;

  // specular: single slow-moving glint across the surface
  float eps  = 0.022;
  float dx   = inkNoise(pL + vec2(eps,0.0), warpDepth, u_movement)
             - inkNoise(pL - vec2(eps,0.0), warpDepth, u_movement);
  float dy   = inkNoise(pL + vec2(0.0,eps), warpDepth, u_movement)
             - inkNoise(pL - vec2(0.0,eps), warpDepth, u_movement);
  vec2  nrm  = normalize(vec2(dx, dy) + 0.001);
  vec2  ldir = normalize(vec2(sin(u_time * 0.05), cos(u_time * 0.04)));
  float spec = pow(max(dot(nrm, ldir), 0.0), 22.0) * density * 0.35;

  vec3 lit    = col * (density + core) + vec3(0.85, 0.92, 1.0) * spec;
  vec3 inkHue = lit;

  // volume gates brightness — quiet=dim, loud=full neon, never whites out
  float bright = 0.15 + u_volume * 0.95;
  inkHue      *= bright;

  // ── vignette ──────────────────────────────────────────────────────────────
  float r2  = dot(uv*vec2(1.0/aspect,1.0), uv*vec2(1.0/aspect,1.0));
  float vig = 1.0 - r2 * 0.6;
  inkHue   *= clamp(vig, 0.0, 1.0);

  // ── saturation — applied after vignette so it acts on the full rendered image
  // 0=greyscale (full desaturate), 1=vivid colour
  float luma = dot(inkHue, vec3(0.299, 0.587, 0.114));
  inkHue     = mix(vec3(luma), inkHue, u_saturation);

  // ── fog — milky white mist over the whole canvas, stronger at edges ───────
  // u_fog=0: clear. u_fog=0.5: noticeable edge haze. u_fog=1: canvas washed out.
  float dist    = sqrt(r2);
  float radial  = smoothstep(0.0, 0.7, dist);          // 0 at centre → 1 at edge
  float fogAmt  = u_fog * (0.45 + radial * 0.55);      // flat base + radial boost
  vec3  fogCol  = vec3(0.90, 0.91, 0.96);
  inkHue = mix(inkHue, fogCol, clamp(fogAmt, 0.0, 0.97));

  gl_FragColor = vec4(inkHue, 1.0);
}`;

// ── compile & link ────────────────────────────────────────────────────────────
function compileShader(type, src){
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if(!gl.getShaderParameter(s, gl.COMPILE_STATUS))
    console.error('shader error:', gl.getShaderInfoLog(s));
  return s;
}

const prog = gl.createProgram();
gl.attachShader(prog, compileShader(gl.VERTEX_SHADER, VS));
gl.attachShader(prog, compileShader(gl.FRAGMENT_SHADER, FS_RORSCHACH));
gl.linkProgram(prog);
if(!gl.getProgramParameter(prog, gl.LINK_STATUS))
  console.error('program link error:', gl.getProgramInfoLog(prog));
gl.useProgram(prog);

// ── fullscreen quad ───────────────────────────────────────────────────────────
const buf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, buf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
const aPos = gl.getAttribLocation(prog, 'a_pos');
gl.enableVertexAttribArray(aPos);
gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

// ── uniform locations ─────────────────────────────────────────────────────────
const U = {
  time:         gl.getUniformLocation(prog, 'u_time'),
  resolution:   gl.getUniformLocation(prog, 'u_resolution'),
  // FFT layer (musical names)
  brightness:   gl.getUniformLocation(prog, 'u_brightness'),
  tone:         gl.getUniformLocation(prog, 'u_tone'),
  movement:     gl.getUniformLocation(prog, 'u_movement'),
  texture:      gl.getUniformLocation(prog, 'u_texture'),
  volume:       gl.getUniformLocation(prog, 'u_volume'),
  bass:         gl.getUniformLocation(prog, 'u_bass'),
  spread:       gl.getUniformLocation(prog, 'u_spread'),
  // VAD position
  valence:      gl.getUniformLocation(prog, 'u_valence'),
  arousal:      gl.getUniformLocation(prog, 'u_arousal'),
  // 6 palette regions — pos/rad/col0/col1 per region
  r0pos: gl.getUniformLocation(prog,'u_r0pos'), r0rad: gl.getUniformLocation(prog,'u_r0rad'),
  r0col0:gl.getUniformLocation(prog,'u_r0col0'),r0col1:gl.getUniformLocation(prog,'u_r0col1'),r0col2:gl.getUniformLocation(prog,'u_r0col2'),
  r1pos: gl.getUniformLocation(prog,'u_r1pos'), r1rad: gl.getUniformLocation(prog,'u_r1rad'),
  r1col0:gl.getUniformLocation(prog,'u_r1col0'),r1col1:gl.getUniformLocation(prog,'u_r1col1'),r1col2:gl.getUniformLocation(prog,'u_r1col2'),
  r2pos: gl.getUniformLocation(prog,'u_r2pos'), r2rad: gl.getUniformLocation(prog,'u_r2rad'),
  r2col0:gl.getUniformLocation(prog,'u_r2col0'),r2col1:gl.getUniformLocation(prog,'u_r2col1'),r2col2:gl.getUniformLocation(prog,'u_r2col2'),
  r3pos: gl.getUniformLocation(prog,'u_r3pos'), r3rad: gl.getUniformLocation(prog,'u_r3rad'),
  r3col0:gl.getUniformLocation(prog,'u_r3col0'),r3col1:gl.getUniformLocation(prog,'u_r3col1'),r3col2:gl.getUniformLocation(prog,'u_r3col2'),
  r4pos: gl.getUniformLocation(prog,'u_r4pos'), r4rad: gl.getUniformLocation(prog,'u_r4rad'),
  r4col0:gl.getUniformLocation(prog,'u_r4col0'),r4col1:gl.getUniformLocation(prog,'u_r4col1'),r4col2:gl.getUniformLocation(prog,'u_r4col2'),
  r5pos: gl.getUniformLocation(prog,'u_r5pos'), r5rad: gl.getUniformLocation(prog,'u_r5rad'),
  r5col0:gl.getUniformLocation(prog,'u_r5col0'),r5col1:gl.getUniformLocation(prog,'u_r5col1'),r5col2:gl.getUniformLocation(prog,'u_r5col2'),
  // global colour modifiers
  colorRadius:  gl.getUniformLocation(prog, 'u_colorRadius'),
  hueShift:     gl.getUniformLocation(prog, 'u_hueShift'),
  colorBand:    gl.getUniformLocation(prog, 'u_colorBand'),
  // global mood params
  colorTemp:    gl.getUniformLocation(prog, 'u_colorTemp'),
  speed:        gl.getUniformLocation(prog, 'u_speed'),
  blobSize:     gl.getUniformLocation(prog, 'u_blobSize'),
  fog:          gl.getUniformLocation(prog, 'u_fog'),
  saturation:   gl.getUniformLocation(prog, 'u_saturation'),
};

gl.uniform2f(U.resolution, RENDER_W, RENDER_H);
gl.uniform1f(U.brightness,   0.5);
gl.uniform1f(U.tone,         0.3);
gl.uniform1f(U.movement,     0.2);
gl.uniform1f(U.texture,      0.2);
gl.uniform1f(U.volume,       0.3);
gl.uniform1f(U.bass,         0.5);
gl.uniform1f(U.spread,       0.5);
// VAD position defaults
gl.uniform1f(U.valence,     0.5);
gl.uniform1f(U.arousal,     0.4);
// helper: hex #rrggbb → [r,g,b] floats 0-1
function hex3(h){ return [parseInt(h.slice(1,3),16)/255,parseInt(h.slice(3,5),16)/255,parseInt(h.slice(5,7),16)/255]; }
function setRegion(n, x,y, rad, c0,c1,c2){
  gl.uniform2f(U[`r${n}pos`], x, y);
  gl.uniform1f(U[`r${n}rad`], rad);
  const [r0,g0,b0]=hex3(c0); gl.uniform3f(U[`r${n}col0`],r0,g0,b0);
  const [r1,g1,b1]=hex3(c1); gl.uniform3f(U[`r${n}col1`],r1,g1,b1);
  const [r2,g2,b2]=hex3(c2); gl.uniform3f(U[`r${n}col2`],r2,g2,b2);
}
// R0 Sad        — neon blue / electric cyan / violet
setRegion(0, 0.15,0.20, 0.50, '#0000ff','#00eeff','#cc00ff');
// R1 Happy      — neon yellow / hot lime / gold
setRegion(1, 0.85,0.65, 0.50, '#ffee00','#aaff00','#ff9900');
// R2 Angry      — neon red / hot orange / magenta-red
setRegion(2, 0.15,0.82, 0.48, '#ff0000','#ff5500','#ff0066');
// R3 Happy+Aroused — neon orange / electric yellow / lime
setRegion(3, 0.78,0.88, 0.40, '#ff6600','#ffdd00','#88ff00');
// R4 Dominant Aroused — neon magenta / hot pink / electric purple
setRegion(4, 0.48,0.92, 0.34, '#ff00cc','#ff44ff','#8800ff');
// R5 Calm       — neon green / electric teal / cyan
setRegion(5, 0.68,0.12, 0.30, '#00ff44','#00ffcc','#00aaff');
// global colour modifiers
gl.uniform1f(U.colorRadius, 1.0);
gl.uniform1f(U.hueShift,    0.0);
gl.uniform1f(U.colorBand,   0.7);
// global mood param defaults
gl.uniform1f(U.colorTemp,   0.5);
gl.uniform1f(U.speed,       1.0);
gl.uniform1f(U.blobSize,    2.2);
gl.uniform1f(U.fog,         0.0);
gl.uniform1f(U.saturation,  1.0);
