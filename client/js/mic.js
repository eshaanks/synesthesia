let stream, recording = false;
let questionActive = false;

async function toggleMic(){
  const btn = document.getElementById('micBtn');
  if(!recording){
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio:true, video:false });
      recording = true;
      btn.textContent = 'stop';
      btn.classList.add('active');
      log('listening...');
      initVisualizer(stream);
      startChunking();
    } catch(e){
      log('mic denied: ' + e.message);
    }
  } else {
    recording = false;
    btn.textContent = 'start';
    btn.classList.remove('active');
    stopVisualizer();
    if(stream) stream.getTracks().forEach(t => t.stop());
    log('stopped');
  }
}

function startChunking(){
  function recordAndSend(){
    if(!recording) return;
    const recorder = new MediaRecorder(stream);
    const chunks   = [];
    recorder.ondataavailable = e => { if(e.data.size > 0) chunks.push(e.data); };
    recorder.onstop = async () => {
      if(chunks.length > 0){
        await sendChunk(new Blob(chunks, { type:'audio/webm' }));
      }
      if(recording) recordAndSend();
    };
    recorder.start();
    setTimeout(() => {
      if(recorder.state === 'recording') recorder.stop();
    }, CONFIG.CHUNK_MS);
  }
  recordAndSend();
}

async function sendChunk(blob){
  const form = new FormData();
  form.append('audio', blob, 'chunk.webm');
  try {
    const res  = await fetch(`${CONFIG.SERVER}/search`, { method:'POST', body:form });
    const data = await res.json();
    if(data.error){ log('server error: ' + data.error); return; }

    // store raw response for debug bar
    window._lastServerResponse = data;
    window._lastServerTime = Date.now();

    console.log('[emotion]', data.emotion, data.probs, 'vad:', data.vad);

    // update emotion probabilities — index.html routes these to visuals
    if(data.probs) window.updateFromEmotion(data.probs);

    // show question from server if present
    if(data.question) showQuestion(data.question);

    const top = data.emotion_word || data.emotion || '?';
    const probStr = data.probs
      ? Object.entries(data.probs).map(([k,v]) => `${k}:${v.toFixed(2)}`).join(' ')
      : '';
    log(`${top} · ${probStr}`);
  } catch(e){
    log('error: ' + e.message);
    console.error('[sendChunk]', e);
  }
}

let _questionTimer = null;
function showQuestion(text){
  const el = document.getElementById('question');
  if(!el) return;
  // fade out, swap text, fade in
  el.style.transition = 'opacity 0.8s ease';
  el.style.opacity    = '0';
  clearTimeout(_questionTimer);
  _questionTimer = setTimeout(() => {
    el.textContent      = text;
    el.style.opacity    = '1';
    // hold for 6s then fade out
    clearTimeout(_questionTimer);
    _questionTimer = setTimeout(() => {
      el.style.opacity = '0';
    }, 6000);
  }, 500);
}

function toggleKnobs(){
  const el  = document.getElementById('knobs');
  const app = document.getElementById('app');
  el.classList.toggle('hidden');
  app.classList.toggle('panel-open', !el.classList.contains('hidden'));
}

function log(msg){
  const el = document.getElementById('log');
  if(el) el.textContent = msg;
}

fetch(`${CONFIG.SERVER}/health`)
  .then(r  => r.json())
  .then(() => log('ready'))
  .catch(() => log('server not reachable'));
