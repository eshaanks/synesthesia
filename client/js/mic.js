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

    // update emotion targets — transition.js lerps these slowly
    updateFromEmotion(data.valence, data.arousal, data.dominance);

    // show question if one isn't already on screen
    if(!questionActive && data.question){
      showQuestion(data.question);
    }

    log(`${data.emotion_word} (v:${data.valence.toFixed(2)} a:${data.arousal.toFixed(2)})`);
  } catch(e){
    log('error: ' + e.message);
  }
}

function showQuestion(text){
  const el = document.getElementById('question');
  if(!el) return;
  questionActive = true;

  el.style.transition = 'none';
  el.style.opacity    = '0';
  el.textContent      = text;
  el.offsetHeight;

  setTimeout(() => {
    el.style.transition = 'opacity 0.4s ease-in';
    el.style.opacity    = '1';

    setTimeout(() => {
      el.style.transition = 'opacity 0.7s ease-out';
      el.style.opacity    = '0';
      setTimeout(() => { questionActive = false; }, 700);
    }, 1500);
  }, 400);
}

function log(msg){
  const el = document.getElementById('log');
  if(el) el.textContent = msg;
}

fetch(`${CONFIG.SERVER}/health`)
  .then(r  => r.json())
  .then(() => log('ready'))
  .catch(() => log('server not reachable'));
