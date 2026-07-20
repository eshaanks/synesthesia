const { app, BrowserWindow, shell } = require('electron');
const { exec, execSync, spawn }     = require('child_process');
const path  = require('path');
const http  = require('http');
const fs    = require('fs');

const SERVER_PORT    = 5001;
const CONTAINER_NAME = 'synesthesia-server';
const IMAGE_NAME     = 'synesthesia:latest';
const HEALTH_URL     = `http://localhost:${SERVER_PORT}/health`;
const MAX_WAIT_MS    = 60000;   // 60s for model to load
const POLL_MS        = 800;

let win = null;

// ── find docker binary ─────────────────────────────────────────────────────────
function dockerBin(){
  const candidates = [
    '/usr/local/bin/docker',
    '/opt/homebrew/bin/docker',
    '/usr/bin/docker',
  ];
  for(const p of candidates) if(fs.existsSync(p)) return p;
  return 'docker';
}

// ── run a shell command, return stdout ────────────────────────────────────────
function run(cmd){
  try { return execSync(cmd, { encoding:'utf8' }).trim(); }
  catch(e){ return ''; }
}

// ── check if Docker daemon is running ────────────────────────────────────────
function dockerRunning(){
  const out = run(`${dockerBin()} info 2>&1`);
  return !out.includes('Cannot connect') && !out.includes('Error');
}

// ── start the server container ────────────────────────────────────────────────
function startContainer(){
  // stop any stale container from a previous session
  run(`${dockerBin()} rm -f ${CONTAINER_NAME} 2>/dev/null`);

  spawn(dockerBin(), [
    'run', '--rm',
    '--name', CONTAINER_NAME,
    '-p', `${SERVER_PORT}:${SERVER_PORT}`,
    IMAGE_NAME,
  ], { detached: true, stdio: 'ignore' }).unref();
}

// ── poll until server responds or timeout ─────────────────────────────────────
function waitForServer(timeoutMs){
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + timeoutMs;
    function poll(){
      http.get(HEALTH_URL, res => {
        if(res.statusCode === 200) resolve();
        else retry();
      }).on('error', retry);
      function retry(){
        if(Date.now() > deadline) reject(new Error('server timeout'));
        else setTimeout(poll, POLL_MS);
      }
    }
    poll();
  });
}

// ── create the app window ──────────────────────────────────────────────────────
function createWindow(){
  win = new BrowserWindow({
    width:  1920,
    height: 1080,
    backgroundColor: '#000000',
    title: 'Synesthesia',
    // hide menu bar — this is an installation, not a dev tool
    autoHideMenuBar: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  // load client from bundled files
  const clientPath = path.join(__dirname, '..', 'client', 'index.html');
  win.loadFile(clientPath);

  // open in fullscreen for installation mode
  win.setFullScreen(true);

  win.on('closed', () => { win = null; });
}

// ── show a loading splash while model starts ──────────────────────────────────
function createSplash(){
  const splash = new BrowserWindow({
    width: 480, height: 260,
    frame: false,
    backgroundColor: '#0a0c10',
    alwaysOnTop: true,
    center: true,
    webPreferences: { nodeIntegration: false },
  });

  splash.loadURL(`data:text/html,
    <style>
      body { margin:0; background:#0a0c10; display:flex; flex-direction:column;
             align-items:center; justify-content:center; height:100vh;
             font-family:-apple-system,sans-serif; color:#00e5ff; }
      h1   { font-size:22px; font-weight:300; letter-spacing:.2em;
             text-transform:uppercase; margin:0 0 12px; }
      p    { font-size:12px; color:#4a5568; letter-spacing:.08em; margin:0; }
      .dot { animation:blink 1.2s infinite; }
      .dot:nth-child(2){ animation-delay:.2s; }
      .dot:nth-child(3){ animation-delay:.4s; }
      @keyframes blink { 0%,80%,100%{opacity:0} 40%{opacity:1} }
    </style>
    <h1>Synesthesia</h1>
    <p>loading model<span class=dot>.</span><span class=dot>.</span><span class=dot>.</span></p>
  `);

  return splash;
}

// ── main startup sequence ─────────────────────────────────────────────────────
app.whenReady().then(async () => {
  if(!dockerRunning()){
    const { dialog } = require('electron');
    dialog.showErrorBox(
      'Docker not running',
      'Please start Docker Desktop and relaunch Synesthesia.'
    );
    app.quit();
    return;
  }

  const splash = createSplash();
  startContainer();

  try {
    await waitForServer(MAX_WAIT_MS);
    splash.close();
    createWindow();
  } catch(e) {
    splash.close();
    const { dialog } = require('electron');
    dialog.showErrorBox('Startup failed', 'The emotion server did not start in time. Try relaunching.');
    app.quit();
  }
});

app.on('window-all-closed', () => {
  // stop the container when the app closes
  run(`${dockerBin()} stop ${CONTAINER_NAME} 2>/dev/null`);
  app.quit();
});
