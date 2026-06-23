// ── texture management and transition state ───────────────────────────────
// depends on: gl.js (gl, U), config.js (CONFIG)

let texA, texB;
let progress     = 1.0;   // start at 1.0 so first image shows immediately
let startTime    = null;
let transitioning = false;

// 1x1 black placeholder texture
function blackTex(){
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,
    new Uint8Array([0,0,0,255]));
  return tex;
}

// load an image URL into a WebGL texture
function loadTexFromURL(url){
  return new Promise((res, rej) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,img);
      gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
      res(tex);
    };
    img.onerror = rej;
    img.src = url;
  });
}

// call this whenever the server returns a new top match
async function showImage(url){
  try {
    const newTex = await loadTexFromURL(url);
    // current B becomes new A — new image becomes B
    texA = texB;
    texB = newTex;
    // restart transition from 0
    progress = 0.0;
    transitioning = true;
  } catch(e) {
    console.error('failed to load image:', url, e);
  }
}

// render loop — called every frame
function renderLoop(ts){
  if(!startTime) startTime = ts;
  const t = (ts - startTime) * 0.001;

  if(transitioning){
    progress = Math.min(1.0, progress + CONFIG.WARP.speed);
    if(progress >= 1.0) transitioning = false;
  }

  gl.uniform1f(U.progress,  progress);
  gl.uniform1f(U.time,      t);
  gl.uniform1f(U.scale,     CONFIG.WARP.scale);
  gl.uniform1f(U.strength,  CONFIG.WARP.strength);
  gl.uniform1f(U.sigma,     CONFIG.WARP.sigma);

  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texA);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, texB);

  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  requestAnimationFrame(renderLoop);
}

// initialise textures and start loop
texA = blackTex();
texB = blackTex();
requestAnimationFrame(renderLoop);
