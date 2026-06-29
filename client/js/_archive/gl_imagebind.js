const canvas = document.getElementById('glCanvas');
const glowCv  = document.getElementById('glowCanvas');

const RENDER_W = 960;
const RENDER_H = 600;

canvas.width  = RENDER_W;
canvas.height = RENDER_H;
glowCv.width  = RENDER_W;
glowCv.height = RENDER_H;

const gl = canvas.getContext('webgl');
gl.viewport(0, 0, RENDER_W, RENDER_H);

const VS = `
attribute vec2 a_pos;
varying vec2 v_uv;
void main(){
  v_uv = vec2(a_pos.x*0.5+0.5, 1.0-(a_pos.y*0.5+0.5));
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

// grid — raw vector, instant response, free colour cycle
const FS_GRID = `
precision highp float;
varying vec2 v_uv;
uniform sampler2D u_vecTex;
uniform float u_time;
uniform float u_hueShift;

vec3 hsv2rgb(float h, float s, float v){
  h=mod(h,360.0)/60.0;
  float c=v*s, x=c*(1.0-abs(mod(h,2.0)-1.0)), m=v-c;
  vec3 rgb;
  if(h<1.0)      rgb=vec3(c,x,0.0);
  else if(h<2.0) rgb=vec3(x,c,0.0);
  else if(h<3.0) rgb=vec3(0.0,c,x);
  else if(h<4.0) rgb=vec3(0.0,x,c);
  else if(h<5.0) rgb=vec3(x,0.0,c);
  else           rgb=vec3(c,0.0,x);
  return rgb+m;
}

void main(){
  // sample exact cell — NEAREST so no bleed between cells
  vec2 cellUV = (floor(v_uv * 32.0) + 0.5) / 32.0;
  float val   = texture2D(u_vecTex, cellUV).r;

  // stretch clustered 0.3-0.6 values across full hue range
  float stretched = pow(val, 0.4);

  // pure value to colour — no animation, no flicker, just the vector
  float hue = stretched * 360.0;
  vec3 col  = hsv2rgb(hue, 1.0, 1.0);

  // cell gap — slightly dim edges, not black
  vec2 local = fract(v_uv * 32.0);
  float edge = smoothstep(0.0, 0.05, local.x) * smoothstep(0.0, 0.05, local.y) *
               smoothstep(0.0, 0.05, 1.0-local.x) * smoothstep(0.0, 0.05, 1.0-local.y);
  float brightness = 0.35 + edge * 0.65;

  gl_FragColor = vec4(col * brightness, 1.0);
}`;

// inkblot — domain-warped noise, bilaterally symmetric, grey/ghost ink + gloss
const FS_FRACTAL = `
precision highp float;
varying vec2 v_uv;
uniform float u_time;
uniform vec2  u_resolution;
uniform vec4  u_warp0;
uniform vec4  u_warp1;
uniform float u_spread;
uniform float u_tendrils;
uniform float u_scale;
uniform float u_inkMode;  // 0 = ghost (light on black), 1 = ink on paper
uniform float u_colour;   // 0 = B&W, 1 = colour
uniform float u_hue;      // emotion hue 0-360

float hash(vec2 p){
  p = fract(p * vec2(127.1, 311.7));
  p += dot(p, p + 19.19);
  return fract(p.x * p.y);
}

float vnoise(vec2 p){
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash(i),             hash(i + vec2(1,0)), u.x),
    mix(hash(i + vec2(0,1)), hash(i + vec2(1,1)), u.x),
    u.y);
}

vec3 cosPalette(float t, float hue){
  float phase = hue / 360.0;
  vec3 col1 = 0.5 + 0.5 * cos(6.28318 * (vec3(1.0) * t + vec3(phase, phase+0.333, phase+0.667)));
  vec3 col2 = 0.5 + 0.5 * cos(6.28318 * (vec3(1.5) * (t*2.3+0.4) + vec3(phase+0.25, phase+0.583, phase+0.917)));
  return col1 + col2 * 0.55;
}

float inkNoise(vec2 p, vec4 w0, vec4 w1, float tend){
  vec2 q = vec2(vnoise(p + w0.xy),          vnoise(p + w0.zw + vec2(5.2, 1.3)));
  vec2 r = vec2(vnoise(p + 4.0*q + w1.xy),  vnoise(p + 4.0*q + w1.zw + vec2(1.7, 9.2)));
  vec2 s = vec2(vnoise(p + 6.0*r + vec2(u_time*0.04, 0.0)),
                vnoise(p + 6.0*r + vec2(0.0, u_time*0.031)));
  return vnoise(p + 4.0*q + 2.0*r + s * tend * 2.5);
}

void main(){
  float aspect = u_resolution.x / u_resolution.y;
  vec2 uv = v_uv - 0.5;
  uv.x *= aspect;
  uv.x = abs(uv.x);
  uv += vec2(sin(u_time * 0.013) * 0.018, cos(u_time * 0.019) * 0.022);

  vec2 p  = uv * u_scale;
  vec2 p2 = uv * (u_scale * 1.9) + vec2(2.3, 4.1);

  float ink  = inkNoise(p,  u_warp0, u_warp1, u_tendrils);
  float ink2 = inkNoise(p2, u_warp1, u_warp0, u_tendrils * 0.6);

  float thresh    = 0.50 - u_spread * 0.20;
  float body      = smoothstep(thresh + 0.06, thresh - 0.06, ink);
  float lobes     = smoothstep(0.58, 0.50, ink2) * 0.7;
  float edgeDist  = abs(ink - thresh) / 0.06;
  float inkDensity = mix(0.85, 0.15, clamp(edgeDist * 0.5, 0.0, 1.0));
  float shape     = clamp(body + lobes, 0.0, 1.0);

  // --- gloss layer ---
  // fake specular: a highlight that moves slowly across the surface
  vec2 lightDir  = normalize(vec2(sin(u_time * 0.07), cos(u_time * 0.05)));
  // surface normal approximated from noise gradient
  float eps      = 0.02;
  float dx       = inkNoise(p + vec2(eps, 0.0), u_warp0, u_warp1, u_tendrils)
                 - inkNoise(p - vec2(eps, 0.0), u_warp0, u_warp1, u_tendrils);
  float dy       = inkNoise(p + vec2(0.0, eps), u_warp0, u_warp1, u_tendrils)
                 - inkNoise(p - vec2(0.0, eps), u_warp0, u_warp1, u_tendrils);
  vec2  normal2d = normalize(vec2(dx, dy) + vec2(0.001));
  float spec     = pow(max(dot(normal2d, lightDir), 0.0), 18.0);
  // gloss only appears on the ink body, brightest near edges
  float edgeRim  = 1.0 - clamp(edgeDist * 0.8, 0.0, 1.0);
  float gloss    = spec * edgeRim * shape * 0.55;

  // GHOST — white/grey ink on deep black
  float rim   = 1.0 - clamp(edgeDist, 0.0, 1.0);
  float ghost = shape * mix(0.12, 0.88, rim * rim);
  ghost = max(ghost, lobes * 0.45);
  vec3 colGhost = mix(vec3(0.03, 0.03, 0.04), vec3(ghost), ghost);
  colGhost += vec3(gloss * 1.1);

  // INK ON PAPER — near-black ink on cold grey-white
  vec3 paper    = vec3(0.91, 0.90, 0.88);
  float darkness = inkDensity * shape + lobes * 0.6;
  vec3 inkCol   = vec3(mix(0.22, 0.04, darkness));
  vec3 colPaper = mix(paper, inkCol, clamp(body * 1.2 + lobes, 0.0, 1.0));
  colPaper += vec3(gloss * 0.7);

  // smooth crossfade between ghost and paper
  vec3 col = mix(colGhost, colPaper, smoothstep(0.0, 1.0, u_inkMode));

  // colour layer — cosine palette tinted by emotion hue, multiplied onto shape
  float hShift  = u_hue + u_time * 3.0;
  vec3  inkHue  = cosPalette(ink, hShift) * shape * 1.3;
  inkHue       += cosPalette(ink2, hShift + 60.0) * lobes * 0.9;

  // blend B&W and colour — u_colour lerps between them
  col = mix(col, inkHue, smoothstep(0.0, 1.0, u_colour));

  float vig = 1.0 - dot(uv, uv) * 0.55;
  col *= clamp(vig, 0.0, 1.0);

  gl_FragColor = vec4(col, 1.0);
}`;

function compileShader(type, src){
  const s=gl.createShader(type);
  gl.shaderSource(s,src);
  gl.compileShader(s);
  if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))
    console.error('shader:',gl.getShaderInfoLog(s));
  return s;
}

function makeProgram(fs){
  const p=gl.createProgram();
  gl.attachShader(p,compileShader(gl.VERTEX_SHADER,VS));
  gl.attachShader(p,compileShader(gl.FRAGMENT_SHADER,fs));
  gl.linkProgram(p);
  return p;
}

const gridProg    = makeProgram(FS_GRID);
const fractalProg = makeProgram(FS_FRACTAL);

const quadBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
gl.bufferData(gl.ARRAY_BUFFER,
  new Float32Array([-1,-1,1,-1,-1,1,1,1]), gl.STATIC_DRAW);

function bindQuad(prog){
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
  const loc=gl.getAttribLocation(prog,'a_pos');
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc,2,gl.FLOAT,false,0,0);
}

const vecTex = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, vecTex);
gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,32,32,0,
  gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array(32*32*4).fill(128));
gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);
gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);

function uploadVectorTexture(vec1024){
  const px = new Uint8Array(32*32*4);
  for(let i=0;i<1024;i++){
    const v  = Math.min(1,Math.max(0,vec1024[i]));
    px[i*4]  = Math.floor(v*255);
    px[i*4+1]= 0;
    px[i*4+2]= 0;
    px[i*4+3]= 255;
  }
  gl.bindTexture(gl.TEXTURE_2D,vecTex);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,32,32,0,
    gl.RGBA,gl.UNSIGNED_BYTE,px);
}

const U = {
  grid:{
    vecTex:   gl.getUniformLocation(gridProg,'u_vecTex'),
    time:     gl.getUniformLocation(gridProg,'u_time'),
    hueShift: gl.getUniformLocation(gridProg,'u_hueShift'),
  },
  fractal:{
    time:       gl.getUniformLocation(fractalProg,'u_time'),
    inkMode:    gl.getUniformLocation(fractalProg,'u_inkMode'),
    colour:     gl.getUniformLocation(fractalProg,'u_colour'),
    hue:        gl.getUniformLocation(fractalProg,'u_hue'),
    resolution: gl.getUniformLocation(fractalProg,'u_resolution'),
    warp0:      gl.getUniformLocation(fractalProg,'u_warp0'),
    warp1:      gl.getUniformLocation(fractalProg,'u_warp1'),
    spread:     gl.getUniformLocation(fractalProg,'u_spread'),
    tendrils:   gl.getUniformLocation(fractalProg,'u_tendrils'),
    scale:      gl.getUniformLocation(fractalProg,'u_scale'),
  },
};
