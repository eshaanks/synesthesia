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
uniform float u_centroid;
uniform float u_flatness;
uniform float u_flux;
uniform float u_zcr;
uniform float u_rms;
uniform float u_bassRatio;
uniform float u_rotSpeed;

// emotion layer — slow mood
uniform float u_valence;
uniform float u_arousal;
uniform float u_dominance;

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
  float t  = u_time * (0.04 + mutRate * 0.12);
  vec2  q  = vec2(vnoise(p + vec2(0.0, 0.0) + t),
                  vnoise(p + vec2(5.2, 1.3) + t));
  vec2  r  = vec2(vnoise(p + warpDepth * q + vec2(1.7, 9.2) + t * 0.7),
                  vnoise(p + warpDepth * q + vec2(8.3, 2.8) + t * 0.5));
  vec2  s  = vec2(vnoise(p + warpDepth * r + vec2(0.3, 5.1) + t * 0.3),
                  vnoise(p + warpDepth * r + vec2(4.1, 3.7) + t * 0.3));
  // extra octave for edge detail — driven by ZCR
  float detail = vnoise(p * 3.5 + s * u_zcr * 2.0 + t * 0.2) * 0.2;
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

  // F0-driven slow rotation of the whole field
  float angle  = u_time * u_rotSpeed * 0.08;
  float ca = cos(angle), sa = sin(angle);
  uv = vec2(uv.x*ca - uv.y*sa, uv.x*sa + uv.y*ca);

  // centroid shifts centre-of-mass vertically
  uv.y += (u_centroid - 0.5) * 0.18;

  // ── bilateral symmetry (Rorschach fold) ──────────────────────────────────
  // arousal breaks symmetry: 0=perfect mirror, 1=independent halves
  float asymBreak = smoothstep(0.0, 1.0, u_arousal * 0.5 + 0.5) * 0.35;
  vec2  uvL = vec2(-abs(uv.x), uv.y);
  vec2  uvR = vec2( abs(uv.x), uv.y);
  // add small independent offset to right side when arousal is high
  uvR += vec2(asymBreak * 0.15 * sin(u_time * 0.3),
              asymBreak * 0.10 * cos(u_time * 0.23));

  // dominance → scale: high dominance = zoomed out (see big shapes)
  float baseScale = 2.2 + u_dominance * (-0.7);  // 1.5 (strong) to 2.9 (weak)
  baseScale = clamp(baseScale, 1.4, 3.2);

  vec2 pL = uvL * baseScale;
  vec2 pR = uvR * baseScale;

  // warp depth from bass ratio
  float warpDepth = 2.5 + u_bassRatio * 3.5;

  float inkL = inkNoise(pL, warpDepth, u_flux);
  float inkR = inkNoise(pR, warpDepth, u_flux);

  // flatness: tonal audio = structured threshold, noisy = organic/jagged
  float thresh = 0.50 - mix(-0.04, 0.06, u_flatness);
  float bodyL  = smoothstep(thresh + 0.07, thresh - 0.07, inkL);
  float bodyR  = smoothstep(thresh + 0.07, thresh - 0.07, inkR);

  // satellite lobes — secondary noise layer offset
  vec2 pL2 = uvL * baseScale * 1.8 + vec2(2.3, 4.1);
  vec2 pR2 = uvR * baseScale * 1.8 + vec2(2.3, 4.1);
  float lobesL = smoothstep(0.58, 0.50, inkNoise(pL2, warpDepth*0.6, u_flux*0.5)) * 0.65;
  float lobesR = smoothstep(0.58, 0.50, inkNoise(pR2, warpDepth*0.6, u_flux*0.5)) * 0.65;

  float shapeL = clamp(bodyL + lobesL, 0.0, 1.0);
  float shapeR = clamp(bodyR + lobesR, 0.0, 1.0);

  // fold left/right back together — asymBreak controls blend
  // at asymBreak=0: pure mirror (left=right), at asymBreak=1: independent
  float shape  = mix(shapeL, mix(shapeL, shapeR, 0.5), asymBreak * 2.0);
  shape        = clamp(shape, 0.0, 1.0);
  float ink    = mix(inkL, mix(inkL, inkR, 0.5), asymBreak * 2.0);

  // ── gloss (specular highlight on ink surface) ─────────────────────────────
  float eps  = 0.018;
  float dx   = inkNoise(pL + vec2(eps,0.0), warpDepth, u_flux)
             - inkNoise(pL - vec2(eps,0.0), warpDepth, u_flux);
  float dy   = inkNoise(pL + vec2(0.0,eps), warpDepth, u_flux)
             - inkNoise(pL - vec2(0.0,eps), warpDepth, u_flux);
  vec2  nrm  = normalize(vec2(dx,dy) + 0.001);
  vec2  ldir = normalize(vec2(sin(u_time*0.07), cos(u_time*0.05)));
  float spec = pow(max(dot(nrm, ldir), 0.0), 20.0);
  float edgeDist = abs(ink - thresh) / 0.07;
  float gloss = spec * (1.0 - clamp(edgeDist*0.8,0.0,1.0)) * shape * 0.5;

  // ── colour from valence ───────────────────────────────────────────────────
  // valence -1 → hue 235 (cold blue), valence +1 → hue 40 (warm gold)
  float hue     = mix(235.0, 40.0, u_valence * 0.5 + 0.5);
  vec3  inkHue  = cosPalette(ink, hue) * shape * 1.2;
  inkHue       += cosPalette(ink, hue + 60.0) * (lobesL + lobesR) * 0.5 * 0.8;

  // RMS pulses the brightness
  float bright = 0.85 + u_rms * 0.35;
  inkHue *= bright;

  // ── vignette ──────────────────────────────────────────────────────────────
  float vig = 1.0 - dot(uv*vec2(1.0/aspect, 1.0), uv*vec2(1.0/aspect, 1.0)) * 0.6;
  inkHue *= clamp(vig, 0.0, 1.0);

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
  time:       gl.getUniformLocation(prog, 'u_time'),
  resolution: gl.getUniformLocation(prog, 'u_resolution'),
  // FFT layer
  centroid:   gl.getUniformLocation(prog, 'u_centroid'),
  flatness:   gl.getUniformLocation(prog, 'u_flatness'),
  flux:       gl.getUniformLocation(prog, 'u_flux'),
  zcr:        gl.getUniformLocation(prog, 'u_zcr'),
  rms:        gl.getUniformLocation(prog, 'u_rms'),
  bassRatio:  gl.getUniformLocation(prog, 'u_bassRatio'),
  rotSpeed:   gl.getUniformLocation(prog, 'u_rotSpeed'),
  // emotion layer
  valence:    gl.getUniformLocation(prog, 'u_valence'),
  arousal:    gl.getUniformLocation(prog, 'u_arousal'),
  dominance:  gl.getUniformLocation(prog, 'u_dominance'),
};

// seed defaults so shader is never uninitialised
gl.uniform2f(U.resolution, RENDER_W, RENDER_H);
gl.uniform1f(U.centroid,  0.5);
gl.uniform1f(U.flatness,  0.3);
gl.uniform1f(U.flux,      0.2);
gl.uniform1f(U.zcr,       0.2);
gl.uniform1f(U.rms,       0.3);
gl.uniform1f(U.bassRatio, 0.5);
gl.uniform1f(U.rotSpeed,  1.0);
gl.uniform1f(U.valence,   0.0);
gl.uniform1f(U.arousal,   0.0);
gl.uniform1f(U.dominance, 0.0);
