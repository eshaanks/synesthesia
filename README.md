# Synesthesia

Live voice art installation. Sings back the emotional texture of a voice as colour, form, and language.

---

## What it does

Listens to a microphone in 2-second chunks, runs a WavLM emotion model to extract valence, arousal, and dominance, and drives a WebGL shader visualisation in real time. A Groq LLM generates quotes matched to the emotional state.

---

## Running from USB (Mac)

### Requirements
- macOS 12 (Monterey) or later
- Docker Desktop installed and running (whale icon in menu bar)
- 8 GB free disk space
- Microphone (built-in or external)
- Chrome or Firefox

### First time setup
1. Install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop/) if not already installed
2. Open Docker Desktop and wait for the whale icon to appear in the menu bar
3. Create a `.env` file next to `install.command` containing:
   ```
   GROQ_API_KEY=your_key_here
   ```
4. Double-click `install.command`
5. The first run loads the image (~2 minutes). Subsequent runs start in seconds.
6. Browser opens automatically at `http://localhost:5001`

### To stop
Press `Ctrl+C` in the terminal window that opened, or quit Docker Desktop.

### USB contents
```
synesthesia.tar.gz   — Docker image
install.command      — Mac launcher (double-click this)
.env                 — API key (create this yourself, never share)
README.md            — this file
```

---

## Running in development (no Docker)

Requires Python 3.10+ with deps installed in a virtualenv.

```bash
# update VENV path in start.sh to match your environment
./start.sh
```

Server runs at `http://localhost:5001`.

---

## Controls

| Key | Action |
|-----|--------|
| `H` | Hide/show UI |
| Click **start** | Begin listening |
| Click **knobs** | Open parameter panel |

**Knobs panel:**
- **Routing** — maps FFT signals (pitch, volume, spectral spread etc) to visual parameters
- **Graphics** — manual overrides for hue, colour radius, speed, blob size
- **Text** — quote hold duration, font family and size
- **Transient** — sensitivity threshold, palette spread, bloom flash

---

## Architecture

```
Mic → 2s audio chunks → Flask (port 5001) → WavLM → VAD [valence, arousal, dominance]
                                           → Groq LLM → quote text
VAD + FFT → WebGL shader → live visualisation
```

- **Model:** `tiantiaf/wavlm-large-msp-podcast-emotion-dim` (fine-tuned on MSP-Podcast)
- **Backbone:** `microsoft/wavlm-large` (300M parameter speech transformer)
- **LLM:** `llama-3.3-70b-versatile` → fallback `llama-3.1-8b-instant` via Groq
- **Quote rate:** ~15 second cooldown, 8-quote diversity buffer

---

## API key

Get a free Groq API key at [console.groq.com](https://console.groq.com). Without it the visuals work but no quotes appear. Create a separate key per installation so individual keys can be revoked.
