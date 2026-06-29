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
      startQuestionCycle();
    } catch(e) {
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
      // schedule next chunk only after response — no overlap, no blocking
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
    const res  = await fetch(`${CONFIG.SERVER}/search`, {method:'POST', body:form});
    const data = await res.json();
    if(data.error){ log('error: '+data.error); return; }

    // shape only — colour locked to question word, not live chunks
    updateFromVector(data.vector);

    log((data.emotions||[])[0] || '—');
  } catch(e) {
    log('error: '+e.message);
  }
}

function startQuestionCycle(){
  setInterval(() => {
    if(!questionActive) showQuestion();
  }, CONFIG.EMOTION_EVERY_MS);
}

function emotionFromCurrentHue(){
  const hues = CONFIG.EMOTION_HUES;
  let best = null, bestDist = Infinity;
  for(const [word, h] of Object.entries(hues)){
    const dist = Math.abs(((currentHue - h + 540) % 360) - 180);
    if(dist < bestDist){ bestDist = dist; best = word; }
  }
  return best;
}

function showQuestion(){
  const word = emotionFromCurrentHue();
  if(!word) return;
  questionActive = true;

  // colour + ink mode transition NOW, locked to this question's word
  const hue = CONFIG.EMOTION_HUES[word];
  if(hue !== undefined) targetHue = hue;
  targetInkMode = DARK_EMOTIONS.has(word) ? 0 : 1;

  const template = CONFIG.QUESTIONS[Math.floor(Math.random() * CONFIG.QUESTIONS.length)];
  const el       = document.getElementById('question');

  console.log(`[question] "${word}" — hue ${hue}°`);

  el.style.transition = 'none';
  el.style.opacity    = '0';
  el.textContent      = template.replace('{w}', word);
  el.offsetHeight;

  el.style.transition = 'opacity 0.8s ease-in';
  el.style.opacity    = '1';

  setTimeout(() => {
    el.style.transition = 'opacity 0.8s ease-out';
    el.style.opacity    = '0';
    setTimeout(() => { questionActive = false; }, 800);
  }, 1500);
}

function log(msg){
  document.getElementById('log').textContent = msg;
}

fetch(`${CONFIG.SERVER}/health`)
  .then(r => r.json())
  .then(d => log(`ready — ${d.emotions} emotions`))
  .catch(() => log('server not reachable'));
