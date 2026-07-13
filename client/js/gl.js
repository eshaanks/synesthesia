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
//   u_neu  0-1  neutral probability
//   u_hap  0-1  happy probability
//   u_ang  0-1  angry probability
//   u_sad  0-1  sad probability

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

// emotion layer — smoothed classifier probabilities 0-1 each, sum to ~1
uniform float u_neu;   // neutral
uniform float u_hap;   // happy
uniform float u_ang;   // angry
uniform float u_sad;   // sad

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
float inkNoise(vec2 p, float warpDepth, float mutRate){
  float t  = u_time * u_speed * (0.02 + mutRate * 0.5);
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

// cosine palette — a,b,c,d are the standard IQ form
// col = a + b*cos(2pi*(c*t+d))
vec3 cosPal(float t, vec3 a, vec3 b, vec3 c, vec3 d){
  return clamp(a + b * cos(6.28318 * (c * t + d)), 0.0, 1.0);
}

// ── per-emotion palettes ──────────────────────────────────────────────────────
// ANGER — reds, deep oranges, hot embers. No cool tones.
vec3 palAnger(float t){
  return cosPal(t,
    vec3(0.55, 0.15, 0.05),   // bias: deep red-orange base
    vec3(0.45, 0.15, 0.05),   // amplitude
    vec3(0.80, 1.00, 1.20),   // frequency
    vec3(0.00, 0.25, 0.50)    // phase: locks hue in red-orange band
  );
}

// SADNESS — desaturated blue-grey, slate, cold dark indigo. No warm tones.
vec3 palSad(float t){
  return cosPal(t,
    vec3(0.12, 0.14, 0.22),   // bias: dark blue-grey base
    vec3(0.10, 0.10, 0.18),   // low amplitude — stays muted
    vec3(0.60, 0.70, 0.50),
    vec3(0.55, 0.60, 0.65)    // phase: locks in cool blue-slate
  );
}

// HAPPY — bright warm yellows, coral, sky blue, mint. Full vibrancy.
vec3 palHappy(float t){
  return cosPal(t,
    vec3(0.65, 0.60, 0.45),   // bias: warm bright base
    vec3(0.35, 0.35, 0.35),   // high amplitude — lots of colour swing
    vec3(1.00, 0.80, 0.60),
    vec3(0.00, 0.15, 0.40)    // phase: sweeps yellow → coral → sky
  );
}

// NEUTRAL — balanced mid-tones, soft lavender to warm white. Unobtrusive.
vec3 palNeutral(float t){
  return cosPal(t,
    vec3(0.40, 0.38, 0.42),
    vec3(0.22, 0.20, 0.25),
    vec3(0.80, 0.90, 0.70),
    vec3(0.30, 0.35, 0.60)
  );
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
  vec2 uvR = vec2( abs(uv.x), uv.y);

  float baseScale = max(u_blobSize, 0.1);
  vec2 pL = uvL * baseScale;
  vec2 pR = uvR * baseScale;

  // warp depth — 0=almost no warp (clean geometric), 1=heavily folded organic mess
  float warpDepth = 0.3 + u_bass * 9.0;

  float inkL = inkNoise(pL, warpDepth, u_movement);
  float inkR = inkNoise(pR, warpDepth, u_movement);

  // tone: 0=tight crisp edge (structured), 1=dissolved filamentous (noisy/organic)
  // threshold shift spreads the shape from compact blob → diffuse cloud
  float thresh = 0.62 - u_tone * 0.30;
  float edge   = 0.02 + u_tone * 0.12;
  float bodyL  = smoothstep(thresh + edge, thresh - edge, inkL);
  float bodyR  = smoothstep(thresh + edge, thresh - edge, inkR);

  // satellite lobes — threshold uses same tone-driven thresh so they dissolve together
  vec2 pL2 = uvL * baseScale * 1.8 + vec2(2.3, 4.1);
  vec2 pR2 = uvR * baseScale * 1.8 + vec2(2.3, 4.1);
  float lobesL = smoothstep(thresh + edge, thresh - edge, inkNoise(pL2, warpDepth*0.6, u_movement*0.5)) * 0.65;
  float lobesR = smoothstep(thresh + edge, thresh - edge, inkNoise(pR2, warpDepth*0.6, u_movement*0.5)) * 0.65;

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

  // ── colour — blended from per-emotion palettes ────────────────────────────
  // each palette is sampled at the same noise value so texture is preserved
  // weights are the smoothed emotion probs — they sum to ~1 so blend is stable
  vec3 cAng  = palAnger (ink);
  vec3 cSad  = palSad   (ink);
  vec3 cHap  = palHappy (ink);
  vec3 cNeu  = palNeutral(ink);

  // weighted blend — emotion probs drive hue in real time
  vec3 blended = cAng  * u_ang
               + cSad  * u_sad
               + cHap  * u_hap
               + cNeu  * u_neu;

  // u_colorTemp nudges the blend slightly warm or cool on top of emotion colour
  // (keeps the manual colorTemp knob useful as a fine-tune)
  blended = mix(blended, blended * vec3(0.85, 0.90, 1.10), (1.0 - u_colorTemp) * 0.3);
  blended = mix(blended, blended * vec3(1.10, 1.00, 0.80), u_colorTemp * 0.3);

  vec3  inkHue = blended * shape * 1.2;
  inkHue      += blended * 0.6 * (lobesL + lobesR) * 0.5;

  // volume: 0=dark/dim, 1=blown out bright
  float bright = 0.3 + u_volume * 1.8;
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
  // emotion layer
  neu:          gl.getUniformLocation(prog, 'u_neu'),
  hap:          gl.getUniformLocation(prog, 'u_hap'),
  ang:          gl.getUniformLocation(prog, 'u_ang'),
  sad:          gl.getUniformLocation(prog, 'u_sad'),
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
// emotion uniform defaults — start neutral so shader has colour from frame 1
gl.uniform1f(U.neu,        1.0);
gl.uniform1f(U.hap,        0.0);
gl.uniform1f(U.ang,        0.0);
gl.uniform1f(U.sad,        0.0);
// global mood param defaults
gl.uniform1f(U.colorTemp,   0.5);
gl.uniform1f(U.speed,       1.0);
gl.uniform1f(U.blobSize,    2.2);
gl.uniform1f(U.fog,         0.0);
gl.uniform1f(U.saturation,  1.0);
