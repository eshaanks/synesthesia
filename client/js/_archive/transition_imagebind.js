const glowCanvas = document.getElementById('glowCanvas');
const glowCtx    = glowCanvas.getContext('2d');

let currentMode = 'grid';
let startTime   = null;

// grid — raw, no lerp
let currentVec  = new Float32Array(1024).fill(0.5);
let targetVec   = new Float32Array(1024).fill(0.5);

// inkblot params — all lerped smoothly
let currentColour   = 0,   targetColour   = 0;   // 0=B&W, 1=colour
let currentHue      = 200, targetHue      = 200;
let currentInkMode  = 0,   targetInkMode  = 0;   // 0=ghost, 1=ink-on-paper
let currentSpread   = 0.5, targetSpread   = 0.5;
let currentTendrils = 0.5, targetTendrils = 0.5;
let currentScale    = 2.8, targetScale    = 2.8;
// warp offsets: 8 floats driving domain warp per octave
let currentWarp = new Float32Array(8).fill(3.0);
let targetWarp  = new Float32Array(8).fill(3.0);

const SPEED = 0.025;
function lerp(a,b,t){ return a+(b-a)*t; }

function updateFromVector(vec){
  targetVec = new Float32Array(vec);
  const avg = (s, e) => vec.slice(s, e).reduce((a, b) => a + b, 0) / (e - s);

  // warp offsets span [0, 12] — wide enough that different slices land in
  // completely different noise regions, giving visually distinct blob shapes
  for(let i = 0; i < 8; i++){
    targetWarp[i] = avg(i * 128, (i + 1) * 128) * 7.0;
  }

  // spread: how much of the frame is ink
  targetSpread = avg(0, 256);

  // tendrils: edge complexity from mid-range frequencies
  targetTendrils = avg(256, 512);

  // scale: zooms the noise field — different magnifications look structurally different
  // range 1.8 (big organic blobs) to 4.5 (fine tendrils, more lobes)
  targetScale = 1.8 + avg(768, 1024) * 2.7;
}

const DARK_EMOTIONS = new Set([
  'grief','despair','dread','void','hollow','frozen','collapse','fractured',
  'melancholy','numbness','coldness','dissolve','rage','tension','turbulence',
  'restlessness','heavy','dense','ancient','infinite','vast','boundless','trembling'
]);

function updateFromEmotions(emotions){
  if(!emotions || !emotions.length) return;
  targetInkMode = DARK_EMOTIONS.has(emotions[0]) ? 0 : 1;
  const h = CONFIG.EMOTION_HUES[emotions[0]];
  if(h !== undefined) targetHue = h;
}

function toggleColour(){
  targetColour = targetColour > 0.5 ? 0 : 1;
  document.getElementById('colourBtn').textContent =
    targetColour > 0.5 ? 'B&W' : 'colour';
}

function switchMode(){
  currentMode = currentMode === 'grid' ? 'fractal' : 'grid';
  document.getElementById('modeBtn').textContent =
    currentMode === 'grid' ? 'switch to fractal' : 'switch to grid';
}

function updateGlow(){
  glowCtx.drawImage(canvas, 0, 0, glowCanvas.width, glowCanvas.height);
}

function renderLoop(ts){
  if(!startTime) startTime = ts;
  const t = (ts - startTime) * 0.001;

  // lerp all inkblot params
  for(let i = 0; i < 8; i++){
    currentWarp[i] = lerp(currentWarp[i], targetWarp[i], SPEED);
  }
  currentInkMode  = lerp(currentInkMode,  targetInkMode,  SPEED * 0.15);
  currentColour   = lerp(currentColour,   targetColour,   SPEED * 0.2);
  const hueDiff   = ((targetHue - currentHue + 540) % 360) - 180;
  currentHue      = (currentHue + hueDiff * SPEED + 360) % 360;
  currentSpread   = lerp(currentSpread,   targetSpread,   SPEED);
  currentTendrils = lerp(currentTendrils, targetTendrils, SPEED);
  currentScale    = lerp(currentScale,    targetScale,    SPEED);

  if(currentMode === 'grid'){
    currentVec = new Float32Array(targetVec);
    uploadVectorTexture(currentVec);

    gl.useProgram(gridProg);
    bindQuad(gridProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, vecTex);
    gl.uniform1i(U.grid.vecTex,   0);
    gl.uniform1f(U.grid.time,     t);
    gl.uniform1f(U.grid.hueShift, t * 15.0 % 360);
  } else {
    gl.useProgram(fractalProg);
    bindQuad(fractalProg);
    gl.uniform1f(U.fractal.time,     t);
    gl.uniform1f(U.fractal.inkMode,  currentInkMode);
    gl.uniform1f(U.fractal.colour,   currentColour);
    gl.uniform1f(U.fractal.hue,      currentHue);
    gl.uniform1f(U.fractal.spread,   currentSpread);
    gl.uniform1f(U.fractal.tendrils, currentTendrils);
    gl.uniform1f(U.fractal.scale,    currentScale);
    gl.uniform4f(U.fractal.warp0,    currentWarp[0], currentWarp[1], currentWarp[2], currentWarp[3]);
    gl.uniform4f(U.fractal.warp1,    currentWarp[4], currentWarp[5], currentWarp[6], currentWarp[7]);
    gl.uniform2f(U.fractal.resolution, canvas.width, canvas.height);
  }

  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  updateGlow();
  requestAnimationFrame(renderLoop);
}

requestAnimationFrame(renderLoop);
