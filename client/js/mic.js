let stream, recording = false, intervalId;
let lastBlob = null;

async function toggleMic(){
  const btn = document.getElementById('btn');
  if(!recording){
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio:true, video:false });
      recording = true;
      btn.textContent = 'stop mic';
      btn.classList.add('active');
      log('mic open');
      initMicVisualizer(stream);
      startChunking();
    } catch(e) {
      log('mic denied: ' + e.message);
    }
  } else {
    recording = false;
    btn.textContent = 'start mic';
    btn.classList.remove('active');
    clearInterval(intervalId);
    stopMicVisualizer();
    if(stream) stream.getTracks().forEach(t => t.stop());
    log('mic closed');
  }
}

function startChunking(){
  intervalId = setInterval(() => {
    if(!recording) return;
    const recorder = new MediaRecorder(stream);
    const chunks   = [];
    recorder.ondataavailable = e => { if(e.data.size > 0) chunks.push(e.data); };
    recorder.onstop = async () => {
      if(chunks.length === 0) return;
      const blob = new Blob(chunks, { type:'audio/webm' });
      lastBlob   = blob;
      await sendChunk(blob);
    };
    recorder.start();
    setTimeout(() => {
      if(recorder.state === 'recording') recorder.stop();
    }, CONFIG.CHUNK_MS);
  }, CONFIG.CHUNK_MS + 200);
}

async function sendChunk(blob){
  const form = new FormData();
  form.append('audio', blob, 'chunk.webm');

  try {
    const res  = await fetch(`${CONFIG.SERVER}/search`, { method:'POST', body:form });
    const json = await res.json();

    if(!json.matches || json.matches.length === 0) return;

    const top = json.matches[0];

    // ── pipeline logging ──────────────────────────────────────────────
    console.group('── ImageBind pipeline ──');
    console.log('raw score:',        top.raw_score);
    console.log('rank score:',       top.rank_score);
    console.log('winning category:', json.winning_category);
    console.log('confidence:',       top.category_confidence);
    console.group('category ranking:');
    json.category_ranking.forEach(([cat, score], i) => {
      console.log(`${i+1}. ${cat}: ${score.toFixed(6)}`);
    });
    console.groupEnd();
    console.group('top matches:');
    json.matches.forEach((m, i) => {
      console.log(`${i+1}. ${m.filename} | raw: ${m.raw_score} | rank: ${m.rank_score}`);
    });
    console.groupEnd();
    console.groupEnd();
    // ─────────────────────────────────────────────────────────────────

    document.getElementById('info').textContent  = json.winning_category;
    document.getElementById('score').textContent = (top.category_confidence * 100).toFixed(1) + '%';
    log(`→ ${json.winning_category} (${(top.category_confidence * 100).toFixed(1)}%)`);

    showImage(`${CONFIG.SERVER}/image/${top.path}`);
    if(lastBlob) playImageMemory(lastBlob);

  } catch(e) {
    log('error: ' + e.message);
    console.error(e);
  }
}

function log(msg){
  document.getElementById('log').textContent = msg;
  console.log(msg);
}

fetch(`${CONFIG.SERVER}/health`)
  .then(r  => r.json())
  .then(d  => log(`server ready — ${d.images} images, ${d.categories} categories`))
  .catch(() => log('server not reachable — start server.py first'));
