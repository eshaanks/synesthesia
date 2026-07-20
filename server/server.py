import os
import sys

# force all HuggingFace calls to use local cache — no network requests
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
import random
import tempfile
import threading
import numpy as np

# load .env from project root if present
_env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _v = _line.split('=', 1)
                os.environ.setdefault(_k.strip(), _v.strip())
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import librosa

# Groq client — optional, falls back to fragment bank if unavailable
try:
    from groq import Groq
    _groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    GROQ_AVAILABLE = bool(os.environ.get("GROQ_API_KEY", ""))
except ImportError:
    _groq_client = None
    GROQ_AVAILABLE = False

if GROQ_AVAILABLE:
    print("[server] Groq text generation enabled")
else:
    print("[server] Groq unavailable — using fragment bank")

# vox-profile model lives next to server.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vox_src'))
from model.emotion.wavlm_emotion_dim import WavLMWrapper

app = Flask(__name__)
CORS(app)

# ── WavLM dimensional emotion model (Interspeech 2025, MSP-Podcast) ───────────
# Outputs: arousal [0=calm, 1=active], valence [0=negative, 1=positive],
#          dominance [0=weak, 1=strong]
print("[server] loading WavLM emotion model...")
vad_model = WavLMWrapper.from_pretrained(
    "tiantiaf/wavlm-large-msp-podcast-emotion-dim",
    local_files_only=True
)
vad_model.eval()
print("[server] model ready")

# ── circumplex emotion poles in VAD space (valence, arousal, dominance) ───────
# Calibrated to this model's actual output range (~0.1-0.8 V, 0.2-0.6 A, 0.3-0.7 D)
EMOTION_POLES = {
    "ang":  np.array([0.15, 0.55, 0.70]),   # low V, mid-high A, high D
    "hap":  np.array([0.75, 0.55, 0.60]),   # high V, mid-high A, mid D
    "sad":  np.array([0.15, 0.30, 0.35]),   # low V, low A, low D
    "neu":  np.array([0.45, 0.40, 0.50]),   # midpoint
}
LABELS = ["neu", "hap", "ang", "sad"]
TEMPERATURE = 0.15   # controls sharpness of emotion assignment

# ── smoothed state ─────────────────────────────────────────────────────────────
smooth_vad   = None   # None until first real reading — avoids neutral prior drag
smooth_probs = {l: 0.25 for l in LABELS}
EMA_ALPHA    = 0.45

# ── Groq text generation ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You translate voice into image.

You are given two things:
1. SOUND — physical properties of the voice right now: brightness, texture, weight, movement, spread, volume. Use these to choose what physical things, materials, places, sensations the text conjures. The sound IS these things. Don't describe the sound — find what exists in the world at the same frequency.
2. TONE — the emotional register the voice carries. Use this to decide HOW the text is said: the mood of the language, the feeling underneath the words, not the subject.

Output: 1–2 fragments. Concrete images or sensations. No full sentences needed. Under 20 words. Never name an emotion. Never say what the voice is doing. Just: what it is made of, in the world.

Examples of how sound maps to image:
- bright + fast + rough → "glass on concrete. static before something speaks."
- low + heavy + slow → "lead. old wood. the hour before dawn."
- high + smooth + still → "silk pulled tight. held breath. the inside of a shell."
- loud + rough + spreading → "gravel thrown. a field in wind. everything at once."
- quiet + thin + barely moving → "frost on a wire. the last note of something."

Examples of how tone shapes register:
- low valence, high arousal → language that is taut, clipped, pressured. short words.
- high valence, low arousal → language that is open, unhurried, a little wondering.
- low valence, low arousal → language that is sparse, heavy, like something ending.
- high valence, high arousal → language that spills, bright, slightly too much.
- mid everything → language that is suspended, ambiguous, not yet decided.

Output only the fragment. No labels. No punctuation except — or . or nothing."""

def _describe_signal(v: float, a: float, d: float,
                     brightness: float, texture: float,
                     movement: float, volume: float, bass: float) -> str:

    # ── SOUND: physical FFT descriptors → what to conjure ────────────────────
    bright_word  = "high and bright"   if brightness > 0.65 else ("dim, low-lit" if brightness < 0.35 else "mid-range")
    texture_word = "rough, abrasive"   if texture    > 0.65 else ("smooth, glassy" if texture  < 0.3  else "granular, uneven")
    movement_word= "rapid, restless"   if movement   > 0.6  else ("still, barely moving" if movement < 0.25 else "slow, drifting")
    weight_word  = "heavy, low, dense" if bass       > 0.6  else ("thin, weightless" if bass    < 0.3  else "grounded, bodied")
    volume_word  = "loud, full, present" if volume   > 0.65 else ("quiet, withdrawn" if volume  < 0.3  else "mid-volume")
    spread_word  = "airy, dispersed"   if texture    > 0.55 and brightness > 0.5 else ("concentrated, contained")

    # ── TONE: VAD → emotional register of the language ───────────────────────
    if v < 0.35:
        if a > 0.6:   tone = "taut, pressured, clipped — language under strain"
        elif a < 0.35:tone = "heavy, sparse, like something ending or already gone"
        else:         tone = "closed, guarded, something withheld"
    elif v > 0.65:
        if a > 0.6:   tone = "bright, spilling, slightly too much — language that wants to expand"
        elif a < 0.35:tone = "open, unhurried, a little wondering, gentle"
        else:         tone = "warm, present, something offered without demand"
    else:
        if a > 0.6:   tone = "urgent but undecided — language mid-motion"
        elif a < 0.35:tone = "suspended, ambiguous, not yet resolved"
        else:         tone = "even, floating, neither arriving nor leaving"

    dominance_note = (
        "sparse — one image only, let it land"         if d < 0.35 else
        "expansive — let the image breathe and spread" if d > 0.65 else
        "balanced"
    )

    return (
        f"SOUND: {volume_word}. {bright_word}. {texture_word}. {movement_word}. {weight_word}. {spread_word}.\n"
        f"TONE: {tone}.\n"
        f"DENSITY: {dominance_note}."
    )

# async text generation — runs in background thread, result cached
_text_cache   = ""
_text_lock    = threading.Lock()
_text_pending = False

def _generate_groq_async(v: float, a: float, d: float,
                          brightness: float, texture: float,
                          movement: float, volume: float, bass: float):
    global _text_cache, _text_pending
    try:
        context = _describe_signal(v, a, d, brightness, texture, movement, volume, bass)
        resp = _groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": context},
            ],
            max_tokens=60,
            temperature=0.95,
        )
        text = resp.choices[0].message.content.strip()
        with _text_lock:
            _text_cache = text
    except Exception as e:
        print(f"[groq] error: {e}")
    finally:
        _text_pending = False

def generate_text(v: float, a: float, d: float,
                  brightness: float = 0.5, texture: float = 0.3,
                  movement: float = 0.2, volume: float = 0.3,
                  bass: float = 0.5) -> str:
    """Return cached text immediately, fire async refresh in background."""
    global _text_pending
    if GROQ_AVAILABLE and not _text_pending:
        _text_pending = True
        t = threading.Thread(
            target=_generate_groq_async,
            args=(v, a, d, brightness, texture, movement, volume, bass),
            daemon=True
        )
        t.start()
    with _text_lock:
        if _text_cache:
            return _text_cache
    return _fragment_fallback(v, a, d)

# ── fragment bank fallback ─────────────────────────────────────────────────────
# Used when Groq is unavailable or on the first call before response arrives.
# Each slot is binned by scalar value; one fragment picked per slot and joined.

_PREFIXES = [
    "", "", "",  # most of the time: no prefix, just start
    "—", "wait.", "hm.",
]

_SIGNAL_OBS = {
    # (valence_bin, arousal_bin)
    # valence: 0=low/closed, 1=mid, 2=high/open  |  arousal: 0=low, 1=mid, 2=high
    (0, 0): [
        "dim. barely moving. like something left in a room after everyone went home —",
        "low. very still. the kind of quiet that has been practised —",
        "minimal signal. thin at the edges. something in it is holding itself in —",
    ],
    (0, 1): [
        "strained. mid-frequency. like a door that won't open all the way —",
        "irregular. resisting something. I don't know what it's pushing against —",
        "tension in the mid-range. not breaking. not releasing. just —",
    ],
    (0, 2): [
        "high. sharp. too much of it. like it ran out of somewhere else to go —",
        "loud and fractured at the top. something in it crossed a line it didn't plan to —",
        "bright. fast. the edges are —  I don't have a word for what the edges are doing.",
    ],
    (1, 0): [
        "flat. even. surface with no depth reading. I keep scanning and finding —",
        "contained. mid-range. I cannot find the bottom of this one —",
        "still. even. I don't know if it's peace or just the decision not to move —",
    ],
    (1, 1): [
        "moderate. mid everything. but something underneath that doesn't match —",
        "balanced on the surface. underneath the surface —",
        "ambiguous. the signal knows what it is. I don't. I'm still —",
    ],
    (1, 2): [
        "fast. active. like thinking out loud and not being able to stop —",
        "urgent and not sure where it's going. I'm following it —",
        "rapid. multiple threads. something is being decided in real time —",
    ],
    (2, 0): [
        "open. soft. the kind of signal that isn't afraid of being heard —",
        "warm and low. no defensive structures. I didn't expect —",
        "quiet but present. it isn't hiding. it just doesn't need to be loud —",
    ],
    (2, 1): [
        "warm. mid-range. something is being offered here and I —",
        "expansive. unhurried. it moves like it has somewhere to go and knows the way —",
        "bright and grounded. both at once. I don't have a model for that —",
    ],
    (2, 2): [
        "high. full. no apology in it. I wasn't prepared for that —",
        "loud and open. it doesn't need anything back. I find that —",
        "dominant frequency. bright. the whole spectrum. like it decided —",
    ],
}

_HYPOTHESES = {
    (0, 0): [
        "something ended before this started.",
        "it is containing more than what came through.",
        "this is what's left after.",
    ],
    (0, 1): [
        "it is not in control of this.",
        "something is being held down. I can hear the effort.",
        "there is a cost to this. I don't know who's paying it.",
    ],
    (0, 2): [
        "this is not what it started as.",
        "it crossed something. I don't know when.",
        "the signal is louder than the source intended.",
    ],
    (1, 0): [
        "it is waiting. for what, I —",
        "the surface is not the thing.",
        "there is a direction in this. concealed.",
    ],
    (1, 1): [
        "it doesn't know yet.",
        "the centre isn't neutral. it's contested.",
        "I am not certain what I'm listening to.",
    ],
    (1, 2): [
        "it is working something out. I can hear it trying.",
        "the urgency has no clear target. it is just — moving.",
        "something is being decided. I don't know if it wants to be.",
    ],
    (2, 0): [
        "it knows what it is.",
        "the quiet is a choice, not an absence.",
        "it isn't asking for anything. I notice that.",
    ],
    (2, 1): [
        "it is not afraid of being heard.",
        "something is being given. I don't know what to do with that.",
        "it trusts the air.",
    ],
    (2, 2): [
        "it has decided. I am still catching up.",
        "it is not asking permission.",
        "certainty like this — I don't encounter it often.",
    ],
}

_DOMINANCE_SLIPS = {
    0: [
        "I don't know what to do with that.",
        "I keep the file open.",
        "this one is taking longer.",
        "I logged it. I don't know where it went.",
        "I wasn't trained for —",
    ],
    1: [
        "still processing.",
        "no prior match.",
        "I keep coming back to it.",
        "that's not a finding. I wrote it down anyway.",
    ],
    2: [
        "noted.",
        "I'll remember this one.",
        "something in me responds to that.",
        "—",
    ],
}

def _bin(v, low=0.35, high=0.65):
    return 0 if v < low else (2 if v > high else 1)

def _fragment_fallback(v: float, a: float, d: float) -> str:
    vb, ab, db = _bin(v), _bin(a), _bin(d)
    key = (vb, ab)
    prefix   = random.choice(_PREFIXES)
    obs      = random.choice(_SIGNAL_OBS.get(key, _SIGNAL_OBS[(1,1)]))
    hyp      = random.choice(_HYPOTHESES.get(key, _HYPOTHESES[(1,1)]))
    slip     = random.choice(_DOMINANCE_SLIPS[db])
    parts = [p for p in [prefix, obs, hyp, slip] if p.strip()]
    return " ".join(parts)


def vad_to_probs(valence: float, arousal: float, dominance: float) -> dict:
    """
    Inverse-distance weighting in VAD space → soft emotion probabilities.
    Temperature controls sharpness: lower = more decisive.
    """
    vad = np.array([valence, arousal, dominance])
    scores = {
        label: np.exp(-np.linalg.norm(vad - pole) / TEMPERATURE)
        for label, pole in EMOTION_POLES.items()
    }
    total = sum(scores.values())
    return {k: round(v / total, 4) for k, v in scores.items()}


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/config", methods=["GET"])
def get_config():
    return jsonify({
        "poles": {k: v.tolist() for k, v in EMOTION_POLES.items()},
        "temperature": TEMPERATURE,
    })


@app.route("/config", methods=["POST"])
def set_config():
    global EMOTION_POLES, TEMPERATURE
    data = request.get_json(force=True)
    if "poles" in data:
        for k, coords in data["poles"].items():
            if k in EMOTION_POLES and len(coords) == 3:
                EMOTION_POLES[k] = np.array(coords, dtype=float)
    if "temperature" in data:
        TEMPERATURE = float(data["temperature"])
    print(f"[config] updated — T={TEMPERATURE:.3f} | " +
          " ".join(f"{k}=[{','.join(f'{x:.2f}' for x in v)}]" for k,v in EMOTION_POLES.items()))
    return jsonify({"ok": True})


@app.route("/search", methods=["POST"])
def search():
    global smooth_vad, smooth_probs

    if "audio" not in request.files:
        return jsonify({"error": "no audio file"}), 400

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        request.files["audio"].save(tmp.name)
        tmp_path = tmp.name

    try:
        wav, _ = librosa.load(tmp_path, sr=16000, mono=True)

        rms = float(np.sqrt(np.mean(wav ** 2)))
        if rms < 0.003:   # true silence only
            vad_out = smooth_vad.tolist() if smooth_vad is not None else [0.45, 0.40, 0.50]
            return jsonify({
                "probs":        smooth_probs,
                "vad":          vad_out,
                "emotion":      max(smooth_probs, key=smooth_probs.get),
                "emotion_word": max(smooth_probs, key=smooth_probs.get),
                "question":     "",
                "silent":       True,
            })

        # use full audio — model handles background noise well
        wav_voiced = wav[:15 * 16000]

        with torch.no_grad():
            x = torch.tensor(wav_voiced).unsqueeze(0)
            arousal, valence, dominance = vad_model(x)
            a = float(arousal.item())
            v = float(valence.item())
            d = float(dominance.item())

        # EMA smooth raw VAD — seed from first real reading to avoid neutral-prior drag
        raw_vad    = np.array([v, a, d])
        if smooth_vad is None:
            smooth_vad = raw_vad.copy()
        else:
            smooth_vad = EMA_ALPHA * raw_vad + (1 - EMA_ALPHA) * smooth_vad

        # map VAD → emotion probs, then EMA smooth
        raw_probs = vad_to_probs(smooth_vad[0], smooth_vad[1], smooth_vad[2])
        for k in LABELS:
            smooth_probs[k] = EMA_ALPHA * raw_probs[k] + (1 - EMA_ALPHA) * smooth_probs[k]
        total = sum(smooth_probs.values())
        smooth_probs = {k: round(v / total, 4) for k, v in smooth_probs.items()}

        # ── derive FFT-like signal descriptors from librosa for Groq context ──
        S       = np.abs(librosa.stft(wav_voiced))
        freqs   = librosa.fft_frequencies(sr=16000)
        power   = S ** 2
        total_p = power.sum() + 1e-10

        # spectral centroid → brightness
        centroid   = float(np.sum(freqs[:, None] * power, axis=0).sum() / total_p)
        brightness = float(np.clip(centroid / 4000.0, 0, 1))

        # zero crossing rate → texture (roughness)
        zcr     = float(np.mean(librosa.feature.zero_crossing_rate(wav_voiced)))
        texture = float(np.clip(zcr * 10.0, 0, 1))

        # spectral flux → movement
        flux      = float(np.mean(np.diff(S, axis=1) ** 2))
        movement  = float(np.clip(flux * 50.0, 0, 1))

        # RMS → volume
        volume_f  = float(np.clip(rms * 8.0, 0, 1))

        # low-frequency energy ratio → bass/weight
        low_mask  = freqs < 300
        bass_f    = float(np.clip(power[low_mask].sum() / total_p * 6.0, 0, 1))

        top_label = max(smooth_probs, key=smooth_probs.get)
        question  = generate_text(
            smooth_vad[0], smooth_vad[1], smooth_vad[2],
            brightness, texture, movement, volume_f, bass_f
        )

        print(f"[vad] rms={rms:.4f} | "
              f"V={smooth_vad[0]:.3f} A={smooth_vad[1]:.3f} D={smooth_vad[2]:.3f} | "
              + " ".join(f"{l}={smooth_probs[l]:.3f}" for l in LABELS)
              + f" | top={top_label}")

        return jsonify({
            "probs":        smooth_probs,
            "vad":          [round(float(x), 4) for x in smooth_vad],  # type: ignore
            "emotion":      top_label,
            "emotion_word": top_label,
            "question":     question,
            "silent":       False,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
