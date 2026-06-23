// ── WebGL init and shader compilation ────────────────────────────────────

const canvas = document.getElementById('glCanvas');
const gl     = canvas.getContext('webgl');

const VS = `
attribute vec2 a_pos;
varying vec2 v_uv;
void main(){
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const FS = `
precision highp float;
varying vec2 v_uv;
uniform float u_progress;
uniform float u_time;
uniform float u_scale;
uniform float u_strength;
uniform float u_sigma;
uniform sampler2D u_texA;
uniform sampler2D u_texB;

vec2 hash(vec2 p){
  p = vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3)));
  return fract(sin(p)*43758.5453)*2.0-1.0;
}

float vnoise(vec2 p){
  vec2 i=floor(p),f=fract(p);
  vec2 u=f*f*(3.0-2.0*f);
  return mix(
    mix(dot(hash(i),f),dot(hash(i+vec2(1,0)),f-vec2(1,0)),u.x),
    mix(dot(hash(i+vec2(0,1)),f-vec2(0,1)),dot(hash(i+vec2(1,1)),f-vec2(1,1)),u.x),
    u.y);
}

float fbm(vec2 p){
  float v=0.0,amp=0.5;
  mat2 rot=mat2(0.8,-0.6,0.6,0.8);
  for(int i=0;i<6;i++){
    v+=amp*vnoise(p);
    p=rot*p*2.1;
    amp*=0.48;
  }
  return v;
}

vec2 domainDisplace(vec2 uv, float strength){
  float t=u_time*0.15;
  vec2 p=uv*u_scale;
  vec2 q=vec2(fbm(p+vec2(0.0,0.0)+t),fbm(p+vec2(5.2,1.3)+t));
  vec2 r=vec2(fbm(p+4.0*q+vec2(1.7,9.2)+t*0.7),fbm(p+4.0*q+vec2(8.3,2.8)+t*0.5));
  vec2 d=vec2(fbm(p+4.0*r+vec2(0.0,0.0)+t*0.3),fbm(p+4.0*r+vec2(4.1,3.7)+t*0.3));
  return d*strength;
}

float gaussian(float x, float mean, float sigma){
  return exp(-pow(x-mean,2.0)/(2.0*sigma*sigma));
}

void main(){
  float warpEnv = gaussian(u_progress, 0.5, u_sigma);
  vec2 dispA    = domainDisplace(v_uv, u_strength * warpEnv);
  vec2 dispB    = domainDisplace(v_uv, u_strength * warpEnv);
  vec2 uvA      = clamp(v_uv + dispA * 0.5, 0.001, 0.999);
  vec2 uvB      = clamp(v_uv - dispB * 0.5, 0.001, 0.999);
  vec4 colA     = texture2D(u_texA, uvA);
  vec4 colB     = texture2D(u_texB, uvB);
  gl_FragColor  = mix(colA, colB, u_progress);
}`;

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
gl.attachShader(prog, compileShader(gl.FRAGMENT_SHADER, FS));
gl.linkProgram(prog);
gl.useProgram(prog);

const buf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, buf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,1,1]), gl.STATIC_DRAW);
const aPos = gl.getAttribLocation(prog, 'a_pos');
gl.enableVertexAttribArray(aPos);
gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

// uniform locations
const U = {
  progress: gl.getUniformLocation(prog,'u_progress'),
  time:     gl.getUniformLocation(prog,'u_time'),
  scale:    gl.getUniformLocation(prog,'u_scale'),
  strength: gl.getUniformLocation(prog,'u_strength'),
  sigma:    gl.getUniformLocation(prog,'u_sigma'),
  texA:     gl.getUniformLocation(prog,'u_texA'),
  texB:     gl.getUniformLocation(prog,'u_texB'),
};
gl.uniform1i(U.texA, 0);
gl.uniform1i(U.texB, 1);
