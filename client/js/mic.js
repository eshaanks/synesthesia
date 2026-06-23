// ── mic capture, chunking, server communication ───────────────────────────
// depends on: config.js (CONFIG), transition.js (showImage)

let stream, intervalId, recording = false;

async function toggleMic(){
  const btn = document.getElementById('btn');
  if(!recording){
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio:true, video:false });
      recording = true;
      btn.textContent = 'stop mic';
      btn.classList.add('active');
      log('mic open — chunking every ' + CONFIG.CHUNK_MS + 'ms');
      startChunking();
    } catch(e) {
      log('mic denied: ' + e.message);
    }
  } else {
    recording = false;
    btn.textContent = 'start mic';
    btn.classList.remove('active');
    clearInterval(intervalId);
    stream.getTracks().forEach(t => t.stop());
    log('mic closed');
  }
}

function startChunking(){
  intervalId = setInterval(() => {
    if(!recording) return;

    const recorder = new MediaRecorder(stream);
    const chunks   = [];

    recorder.ondataavailable = e => chunks.push(e.data);
    recorder.onstop = async () => {
      const blob = new Blob(chunks, { type:'audio/webm' });
      await sendChunk(blob);
    };

    recorder.start();
    setTimeout(() => recorder.stop(), CONFIG.CHUNK_MS);
  }, CONFIG.CHUNK_MS);
}

async function sendChunk(blob){
  log('sending chunk...');
  const form = new FormData();
  form.append('audio', blob, 'chunk.webm');

  try {
    const res  = await fetch(`${CONFIG.SERVER}/search`, { method:'POST', body:form });
    const data = await res.json();

    if(data.matches && data.matches.length > 0){
      const top = data.matches[0];
      // update UI
      document.getElementById('score').textContent = top.score.toFixed(4);
      document.getElementById('info').textContent  = top.filename;
      log(`matched: ${top.filename} (${top.score.toFixed(4)})`);
      // trigger WebGL transition
      showImage(`${CONFIG.SERVER}/image/${top.filename}`);
    } else {
      log('no matches returned');
    }
  } catch(e) {
    log('server error: ' + e.message);
  }
}

function log(msg){
  document.getElementById('log').textContent = msg;
  console.log(msg);
}

// check server health on load
fetch(`${CONFIG.SERVER}/health`)
  .then(r  => r.json())
  .then(d  => log(`server ready — ${d.images} images in database`))
  .catch(() => log('server not reachable — start server.py first'));
