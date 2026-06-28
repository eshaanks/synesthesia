// ── dual visualizer — mic input + image memory playback ───────────────────

// ── mic visualizer ────────────────────────────────────────────────────────
let micAnalyser, micDataArray, micAnimId;

function initMicVisualizer(stream){
  const audioCtx = new AudioContext();
  const source   = audioCtx.createMediaStreamSource(stream);
  micAnalyser    = audioCtx.createAnalyser();
  micAnalyser.fftSize = 256;
  micAnalyser.smoothingTimeConstant = 0.8;
  source.connect(micAnalyser);
  micDataArray = new Uint8Array(micAnalyser.frequencyBinCount);
  drawMicVisualizer();
}

function stopMicVisualizer(){
  if(micAnimId) cancelAnimationFrame(micAnimId);
  const canvas = document.getElementById('micCanvas');
  const ctx    = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawMicVisualizer(){
  const canvas = document.getElementById('micCanvas');
  const ctx    = canvas.getContext('2d');
  const W      = canvas.width;
  const H      = canvas.height;

  micAnimId = requestAnimationFrame(drawMicVisualizer);
  if(!micAnalyser) return;
  micAnalyser.getByteFrequencyData(micDataArray);
  ctx.clearRect(0, 0, W, H);

  const barWidth = (W / micDataArray.length) * 2.5;
  let x = 0;
  for(let i = 0; i < micDataArray.length; i++){
    const barHeight = (micDataArray[i] / 255) * H;
    const alpha     = 0.3 + (micDataArray[i] / 255) * 0.7;
    ctx.fillStyle   = `rgba(255,255,255,${alpha})`;
    ctx.fillRect(x, H - barHeight, barWidth - 1, barHeight);
    x += barWidth;
    if(x > W) break;
  }
}

// ── playback visualizer — single shared AudioContext ─────────────────────
// reuse one context to avoid "insufficient resources" from too many instances
const sharedPlaybackCtx  = new AudioContext();
let playbackAnalyser     = sharedPlaybackCtx.createAnalyser();
let playbackData         = new Uint8Array(playbackAnalyser.frequencyBinCount);
let playbackAnimId       = null;
let isPlaying            = false;
let currentSource        = null;

playbackAnalyser.fftSize = 256;
playbackAnalyser.smoothingTimeConstant = 0.8;

async function playImageMemory(blob){
  // stop current playback if any
  if(currentSource){
    try { currentSource.stop(); } catch(e){}
    currentSource = null;
  }
  if(playbackAnimId) cancelAnimationFrame(playbackAnimId);

  try {
    // resume context if suspended (browser autoplay policy)
    if(sharedPlaybackCtx.state === 'suspended'){
      await sharedPlaybackCtx.resume();
    }

    const arrayBuffer = await blob.arrayBuffer();
    const audioBuffer = await sharedPlaybackCtx.decodeAudioData(arrayBuffer);

    currentSource        = sharedPlaybackCtx.createBufferSource();
    currentSource.buffer = audioBuffer;

    const gain         = sharedPlaybackCtx.createGain();
    gain.gain.value    = 0.15;

    currentSource.connect(playbackAnalyser);
    playbackAnalyser.connect(gain);
    gain.connect(sharedPlaybackCtx.destination);

    currentSource.start();
    isPlaying              = true;
    currentSource.onended  = () => { isPlaying = false; };

    drawPlaybackVisualizer();
  } catch(e){
    console.error('playback error:', e);
  }
}

function drawPlaybackVisualizer(){
  const canvas = document.getElementById('playbackCanvas');
  const ctx    = canvas.getContext('2d');
  const W      = canvas.width;
  const H      = canvas.height;

  playbackAnimId = requestAnimationFrame(drawPlaybackVisualizer);
  playbackAnalyser.getByteFrequencyData(playbackData);
  ctx.clearRect(0, 0, W, H);

  ctx.beginPath();
  ctx.strokeStyle = 'rgba(255,255,255,0.5)';
  ctx.lineWidth   = 1.5;

  const sliceWidth = W / playbackData.length;
  let x = 0;
  for(let i = 0; i < playbackData.length; i++){
    const y = H - (playbackData[i] / 255) * H;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    x += sliceWidth;
  }
  ctx.stroke();
}
