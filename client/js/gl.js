// ── WebGL setup ───────────────────────────────────────────────────────────────
const canvas   = document.getElementById('glCanvas');
const RENDER_W = 960;
const RENDER_H = 600;
canvas.width   = RENDER_W;
canvas.height  = RENDER_H;

const gl = canvas.getContext('webgl');
gl.viewport(0, 0, RENDER_W, RENDER_H);

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
// Layer 2 — emotion uniforms (every ~2s from wav2vec2-emotion server):
//   u_valence    -1..+1  → hue (cold blue ↔ warm gold)
//   u_arousal    -1..+1  → symmetry break (calm=pure mirror, excited=broken)
//   u_dominance  -1..+1  → scale (strong=zoomed out, weak=zoomed in)

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

// emotion layer — slow mood
uniform float u_valence;
uniform float u_arousal;
uniform float u_dominance;

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

// 5-octave domain-warped noise — warp depth driven by bass, mutation by flux
float inkNoise(vec2 p, float warpDepth, float mutRate){
  float t  = u_time * u_speed * (0.04 + mutRate * 0.12);
  vec2  q  = vec2(vnoise(p + vec2(0.0, 0.0) + t),
                  vnoise(p + vec2(5.2, 1.3) + t));
  vec2  r  = vec2(vnoise(p + warpDepth * q + vec2(1.7, 9.2) + t * 0.7),
                  vnoise(p + warpDepth * q + vec2(8.3, 2.8) + t * 0.5));
  vec2  s  = vec2(vnoise(p + warpDepth * r + vec2(0.3, 5.1) + t * 0.3),
                  vnoise(p + warpDepth * r + vec2(4.1, 3.7) + t * 0.3));
  // extra octave for edge detail — driven by texture (ZCR)
  float detail = vnoise(p * 3.5 + s * u_texture * 2.0 + t * 0.2) * 0.2;
  return vnoise(p + 3.0*q + 2.0*r + s) + detail;
}

// cosine palette — phase driven by valence-mapped hue
vec3 cosPalette(float t, float hue){
  float ph = hue / 360.0;
  return 0.5 + 0.5 * cos(6.28318 * (vec3(1.0,1.0,1.0)*t + vec3(ph, ph+0.33, ph+0.67)));
}

void main(){
  float aspect = u_resolution.x / u_resolution.y;

  // centre, aspect-correct
  vec2 uv  = v_uv - 0.5;
  uv.x    *= aspect;

  // brightness shifts centre-of-mass vertically
  uv.y += (u_brightness - 0.5) * 0.22;

  // spread: squeezes or splays the x-axis — open/wide vs closed/narrow
  uv.x *= 0.6 + u_spread * 1.2;

  // ── bilateral symmetry ────────────────────────────────────────────────────
  vec2 uvL = vec2(-abs(uv.x), uv.y);
  vec2 uvR = vec2( abs(uv.x), uv.y);

  float baseScale = max(u_blobSize, 0.1);
  vec2 pL = uvL * baseScale;
  vec2 pR = uvR * baseScale;

  // warp depth driven purely by bass audio signal
  float warpDepth = 1.5 + u_bass * 4.0;

  float inkL = inkNoise(pL, warpDepth, u_movement);
  float inkR = inkNoise(pR, warpDepth, u_movement);

  // tone: singing voice = tighter threshold (structured), noise = loose (organic)
  float thresh = 0.50 - mix(-0.04, 0.06, u_tone);
  float bodyL  = smoothstep(thresh + 0.07, thresh - 0.07, inkL);
  float bodyR  = smoothstep(thresh + 0.07, thresh - 0.07, inkR);

  // satellite lobes
  vec2 pL2 = uvL * baseScale * 1.8 + vec2(2.3, 4.1);
  vec2 pR2 = uvR * baseScale * 1.8 + vec2(2.3, 4.1);
  float lobesL = smoothstep(0.58, 0.50, inkNoise(pL2, warpDepth*0.6, u_movement*0.5)) * 0.65;
  float lobesR = smoothstep(0.58, 0.50, inkNoise(pR2, warpDepth*0.6, u_movement*0.5)) * 0.65;

  float shape = clamp(bodyL + lobesL, 0.0, 1.0);
  float ink   = inkL;

  // ── gloss ─────────────────────────────────────────────────────────────────
  float eps    = 0.018;
  float dx     = inkNoise(pL + vec2(eps,0.0), warpDepth, u_movement)
               - inkNoise(pL - vec2(eps,0.0), warpDepth, u_movement);
  float dy     = inkNoise(pL + vec2(0.0,eps), warpDepth, u_movement)
               - inkNoise(pL - vec2(0.0,eps), warpDepth, u_movement);
  vec2  nrm    = normalize(vec2(dx,dy) + 0.001);
  vec2  ldir   = normalize(vec2(sin(u_time*0.07), cos(u_time*0.05)));
  float spec   = pow(max(dot(nrm, ldir), 0.0), 20.0);
  float gloss  = spec * (1.0 - clamp(abs(ink-thresh)/0.07*0.8,0.0,1.0)) * shape * 0.5;

  // ── colour — mood driven ───────────────────────────────────────────────────
  float hue    = mix(235.0, 40.0, u_colorTemp);
  vec3  inkHue = cosPalette(ink, hue) * shape * 1.2;
  inkHue      += cosPalette(ink, hue + 60.0) * (lobesL + lobesR) * 0.5;

  // saturation — lerp toward luminance for greyscale
  float luma   = dot(inkHue, vec3(0.299, 0.587, 0.114));
  inkHue       = mix(vec3(luma), inkHue, u_saturation);

  // volume pulses brightness
  float bright = 0.85 + u_volume * 0.35;
  inkHue      *= bright;

  // ── vignette ──────────────────────────────────────────────────────────────
  float r2  = dot(uv*vec2(1.0/aspect,1.0), uv*vec2(1.0/aspect,1.0));
  float vig = 1.0 - r2 * 0.6;
  inkHue   *= clamp(vig, 0.0, 1.0);

  // ── fog — radial mist, thickens from edges inward proportional to u_fog ───
  // adds a soft milky bloom at the perimeter, not a flat overlay
  float fogVig   = smoothstep(0.0, 1.4, sqrt(r2));           // 0 at centre, 1 at edge
  float fogAmt   = fogVig * fogVig * u_fog;                   // quadratic rolloff
  vec3  fogColor = vec3(0.88, 0.90, 0.95);
  inkHue = mix(inkHue, fogColor, clamp(fogAmt, 0.0, 0.85));

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
  // emotion layer
  valence:      gl.getUniformLocation(prog, 'u_valence'),
  arousal:      gl.getUniformLocation(prog, 'u_arousal'),
  dominance:    gl.getUniformLocation(prog, 'u_dominance'),
  // global mood params
  colorTemp:    gl.getUniformLocation(prog, 'u_colorTemp'),
  speed:        gl.getUniformLocation(prog, 'u_speed'),
  blobSize:     gl.getUniformLocation(prog, 'u_blobSize'),
  fog:          gl.getUniformLocation(prog, 'u_fog'),
  saturation:   gl.getUniformLocation(prog, 'u_saturation'),
};

gl.uniform2f(U.resolution,   RENDER_W, RENDER_H);
gl.uniform1f(U.brightness,   0.5);
gl.uniform1f(U.tone,         0.3);
gl.uniform1f(U.movement,     0.2);
gl.uniform1f(U.texture,      0.2);
gl.uniform1f(U.volume,       0.3);
gl.uniform1f(U.bass,         0.5);
gl.uniform1f(U.spread,       0.5);
gl.uniform1f(U.valence,     0.0);
gl.uniform1f(U.arousal,     0.0);
gl.uniform1f(U.dominance,   0.0);
// global mood param defaults
gl.uniform1f(U.colorTemp,   0.5);
gl.uniform1f(U.speed,       1.0);
gl.uniform1f(U.blobSize,    2.2);
gl.uniform1f(U.fog,         0.0);
gl.uniform1f(U.saturation,  1.0);
