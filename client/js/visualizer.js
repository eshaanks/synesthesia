// ── FFT parameter extraction — Web Audio API ──────────────────────────────────
// Extracts 7 parameters every animation frame and pushes them as WebGL uniforms.
// These drive the Rorschach pattern instantly with zero server latency.
//
//   centroid   — spectral centroid (0-1)   → vertical centre-of-mass shift
//   flatness   — spectral flatness (0-1)   → structured vs noisy warp character
//   rotSpeed   — F0 estimate (0-1)         → rotation speed of whole field
//   flux       — frame-to-frame change (0-1) → mutation rate / animation speed
//   zcr        — zero crossing rate (0-1)  → edge sharpness
//   rms        — RMS energy (0-1)          → overall brightness pulse
//   bassRatio  — bass/treble energy ratio  → warp depth

let analyser    = null;
let freqData    = null;
let prevFreq    = null;
let timeData    = null;
let fftAnimId   = null;
let vizCanvas   = null;
let vizCtx      = null;

function initVisualizer(stream){
  vizCanvas = document.getElementById('vizCanvas');
  vizCtx    = vizCanvas.getContext('2d');

  const audioCtx = new AudioContext();
  const source   = audioCtx.createMediaStreamSource(stream);
  analyser       = audioCtx.createAnalyser();
  analyser.fftSize                = 2048;
  analyser.smoothingTimeConstant  = 0.6;
  source.connect(analyser);

  freqData = new Float32Array(analyser.frequencyBinCount);
  timeData = new Float32Array(analyser.fftSize);
  prevFreq = new Float32Array(analyser.frequencyBinCount);

  fftLoop();
}

function stopVisualizer(){
  if(fftAnimId) cancelAnimationFrame(fftAnimId);
  fftAnimId = null;
  analyser  = null;
  if(vizCtx && vizCanvas) vizCtx.clearRect(0, 0, vizCanvas.width, vizCanvas.height);
}

function fftLoop(){
  if(!analyser) return;
  fftAnimId = requestAnimationFrame(fftLoop);

  analyser.getFloatFrequencyData(freqData);
  analyser.getFloatTimeDomainData(timeData);

  const binCount  = freqData.length;
  const nyquist   = 22050;
  const binHz     = nyquist / binCount;

  // convert dB to linear power, clamp silence floor
  const power = new Float32Array(binCount);
  let   totalPower = 0;
  for(let i = 0; i < binCount; i++){
    power[i]   = Math.pow(10, Math.max(freqData[i], -90) / 10);
    totalPower += power[i];
  }

  // ── spectral centroid (0-1 normalized to nyquist) ─────────────────────────
  let weightedSum = 0;
  for(let i = 0; i < binCount; i++) weightedSum += power[i] * i * binHz;
  const centroid = totalPower > 1e-10
    ? Math.min(weightedSum / (totalPower * nyquist), 1.0)
    : 0.5;

  // ── spectral flatness (geometric/arithmetic mean ratio → 0-1) ────────────
  let logSum = 0, linSum = 0, validBins = 0;
  for(let i = 1; i < binCount; i++){
    if(power[i] > 1e-12){
      logSum += Math.log(power[i]);
      linSum += power[i];
      validBins++;
    }
  }
  const flatness = validBins > 0 && linSum > 0
    ? Math.min(Math.exp(logSum / validBins) / (linSum / validBins) * 10.0, 1.0)
    : 0;

  // ── F0 estimate via autocorrelation → rotSpeed ────────────────────────────
  // look for strongest periodicity in 80-800 Hz range
  const minLag = Math.floor(44100 / 800);
  const maxLag = Math.floor(44100 / 80);
  let bestCorr = 0, bestLag = minLag;
  const N = timeData.length;
  for(let lag = minLag; lag <= Math.min(maxLag, N - 1); lag++){
    let corr = 0;
    for(let i = 0; i < N - lag; i++) corr += timeData[i] * timeData[i + lag];
    if(Math.abs(corr) > Math.abs(bestCorr)){ bestCorr = corr; bestLag = lag; }
  }
  const f0Hz     = bestCorr > 0.01 ? 44100 / bestLag : 0;
  const rotSpeed = Math.min(f0Hz / 500.0, 1.0);

  // ── spectral flux (frame-to-frame change) ─────────────────────────────────
  let fluxSum = 0;
  for(let i = 0; i < binCount; i++){
    const diff = power[i] - prevFreq[i];
    fluxSum   += diff * diff;
    prevFreq[i] = power[i];
  }
  const flux = Math.min(Math.sqrt(fluxSum / binCount) * 80.0, 1.0);

  // ── zero crossing rate ────────────────────────────────────────────────────
  let zcCount = 0;
  for(let i = 1; i < N; i++){
    if((timeData[i] >= 0) !== (timeData[i-1] >= 0)) zcCount++;
  }
  const zcr = Math.min(zcCount / N * 10.0, 1.0);

  // ── RMS energy ────────────────────────────────────────────────────────────
  let rmsSum = 0;
  for(let i = 0; i < N; i++) rmsSum += timeData[i] * timeData[i];
  const rms = Math.min(Math.sqrt(rmsSum / N) * 8.0, 1.0);

  // ── bass/treble ratio ─────────────────────────────────────────────────────
  // bass: 20-300Hz bins, treble: 2000-8000Hz bins
  const bassCutBin   = Math.floor(300  / binHz);
  const trebleLoB    = Math.floor(2000 / binHz);
  const trebleHiB    = Math.floor(8000 / binHz);
  let bassEnergy = 0, trebleEnergy = 0;
  for(let i = 1; i <= bassCutBin; i++)                  bassEnergy   += power[i];
  for(let i = trebleLoB; i <= trebleHiB; i++)           trebleEnergy += power[i];
  const bassRatio = trebleEnergy > 1e-10
    ? Math.min(bassEnergy / (bassEnergy + trebleEnergy), 1.0)
    : 0.5;

  // ── push to WebGL uniforms ────────────────────────────────────────────────
  if(typeof U !== 'undefined'){
    gl.uniform1f(U.centroid,  centroid);
    gl.uniform1f(U.flatness,  flatness);
    gl.uniform1f(U.rotSpeed,  rotSpeed);
    gl.uniform1f(U.flux,      flux);
    gl.uniform1f(U.zcr,       zcr);
    gl.uniform1f(U.rms,       rms);
    gl.uniform1f(U.bassRatio, bassRatio);
  }

  // ── waveform visualizer bar at bottom ────────────────────────────────────
  if(vizCtx && vizCanvas){
    const W = vizCanvas.clientWidth  || vizCanvas.width;
    const H = vizCanvas.clientHeight || vizCanvas.height;
    vizCanvas.width  = W;
    vizCanvas.height = H;

    vizCtx.fillStyle = 'rgba(0,0,0,0.3)';
    vizCtx.fillRect(0, 0, W, H);

    vizCtx.beginPath();
    vizCtx.strokeStyle = `rgba(255,255,255,${0.3 + rms * 0.5})`;
    vizCtx.lineWidth   = 1.2;
    const step = W / timeData.length;
    for(let i = 0; i < timeData.length; i++){
      const x = i * step;
      const y = H / 2 + timeData[i] * H * 0.45;
      i === 0 ? vizCtx.moveTo(x, y) : vizCtx.lineTo(x, y);
    }
    vizCtx.stroke();
  }
}
