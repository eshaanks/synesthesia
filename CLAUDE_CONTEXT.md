# Synesthesia — Art Installation Project
# Full context for Claude Code

## Project location
/Users/aiswaryashybu/ART+CODE/Proj/synesthesia

## How to run
Terminal 1: cd /Users/aiswaryashybu/ART+CODE/Proj/synesthesia && source ../ImageBind/imagebind-env/bin/activate && python server/server.py
Terminal 2: cd /Users/aiswaryashybu/ART+CODE/Proj/synesthesia && python3 -m http.server 8080
Browser: http://localhost:8080/client/index.html

## What this is
Live voice installation. Person speaks into mic → system visualises the emotional content of their voice as a pattern + displays the model's emotional interpretation as text questions.

## Current stack
- Flask server (port 5001): server/server.py
- ImageBind model at: ../ImageBind/ — encodes audio to 1024-d vector
- Frontend: client/index.html + client/js/ + client/css/
- Python env: ../ImageBind/imagebind-env/

## Current visualization (working but needs replacing)
Two modes via button:
1. GRID — 32x32 grid, each cell = one dimension of 1024-d ImageBind vector, mapped to HSV hue
2. FRACTAL — Julia set driven by vector slice averages, emotion hue drives colour

## What needs to be built (new feature branch: feature/rorschach-emotion)
Replace current visualization with:

### PATTERN (WebGL, client side)
Bilaterally symmetric Rorschach-style domain-warped pattern.
Driven by TWO layers:

LAYER 1 — Raw audio FFT (instant, zero latency, Web Audio API):
- spectral centroid → spatial distribution / centre of mass of pattern
- spectral flatness → structured vs noisy character (tonal voice = structured, whisper = noisy)
- fundamental frequency F0 → rotation speed of pattern
- flux (frame-to-frame spectral change) → mutation rate
- zero crossing rate → edge sharpness
- RMS energy → overall brightness pulse
- bass/treble ratio → warp depth
These update every animation frame — pattern reacts instantly to every phoneme and breath.

LAYER 2 — Emotion model (every 2 seconds, server side):
- valence (-1 to +1) → hue (negative=blue/cold hue 200-270, positive=warm/gold hue 20-60)
- arousal (-1 to +1) → symmetry order (calm=high symmetry, excited=broken symmetry)
- dominance (-1 to +1) → zoom/scale (strong=zoomed out, weak=zoomed in tight)
These update slowly, setting the emotional character/mood of the pattern.

Pattern must be:
- Bilaterally symmetric (Rorschach ink blot style — mirrored left/right)
- Always moving, never static
- Visually beautiful — think domain-warped fluid simulation with colour
- Responsive enough that different voices produce visibly different patterns

### EMOTION MODEL (server side)
Replace ImageBind emotion detection with:
Model: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
Available on HuggingFace, runs on M1 CPU
Outputs: valence, arousal, dominance (each -1 to +1)
Install: pip install transformers torch

Keep ImageBind running for the 1024-d vector (still used for pattern variation).
Run wav2vec2-emotion in parallel for emotion dimensions.

Server /search endpoint should return:
{
  "vector": [...1024 floats 0-1...],     # from ImageBind, drives pattern variation
  "valence": float,                       # from wav2vec2-emotion
  "arousal": float,                       # from wav2vec2-emotion  
  "dominance": float,                     # from wav2vec2-emotion
  "emotion_word": string,                 # derived from v/a/d dimensions
  "question": string                      # pre-generated question string
}

### EMOTION WORD + QUESTION (server side)
Derive emotion word from valence/arousal/dominance:
- high valence + high arousal → "euphoria"
- high valence + low arousal → "serenity"  
- low valence + high arousal → "anguish"
- low valence + low arousal → "grief"
- high dominance + low valence → "rage"
- low dominance + low valence → "despair"
- neutral valence + high arousal → "tension"
- neutral all → "stillness"
etc — build a full mapping

Generate question from emotion word using templates:
- "i feel a hint of {w} here."
- "something like {w}..."
- "is there {w} in this?"
- "the model reads this as {w}."
- "...{w}. or something close."

### QUESTION DISPLAY (client side)
- Fades in over pattern 400ms after emotion update
- Holds for 1.5 seconds  
- Fades out over 700ms
- Only one question visible at a time
- Position: lower third of screen, centred
- Font: light weight, large, white, letter-spaced

### AUDIO VISUALIZER
Keep existing mic waveform visualizer at bottom of screen.

## File structure
server/
  server.py              ← needs wav2vec2-emotion added alongside ImageBind
client/
  index.html
  css/style.css
  js/config.js           ← EMOTION_HUES, QUESTIONS, CONFIG params
  js/gl.js               ← WebGL shaders — needs complete rewrite for Rorschach
  js/transition.js       ← render loop, state management — needs rewrite
  js/mic.js              ← mic capture, chunking, question display
  js/visualizer.js       ← Web Audio API FFT extraction — needs expansion

## Git workflow
Current branch: feature/rorschach-emotion (already created and pushed)
Commit messages should be descriptive e.g.:
  "feat: add wav2vec2-emotion to server pipeline"
  "feat: Rorschach WebGL shader with bilateral symmetry"
  "feat: raw audio FFT parameters driving pattern"

## Known issues in current code to be aware of
- Green flash in Julia fractal when cx/cy values go out of interesting range
- Canvas needs to fill full viewport (100vw x 100vh)
- Question display sometimes shows multiple overlapping questions

## Dependencies already installed in imagebind-env
torch, flask, flask-cors, numpy, imagebind (custom), transformers

## M1 Mac constraints
- No GPU — MPS available but memory limited (8GB shared)
- Run inference on CPU for wav2vec2-emotion to avoid MPS memory issues
- ImageBind already configured for MPS or CPU
- Keep models lean — load once at server startup, never reload per request
