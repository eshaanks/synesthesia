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
from flask import Flask, request, jsonify, send_from_directory
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

SYSTEM_PROMPT = """Live voice installation. Output one line only.

Format: quote text — Attribution

The emotional state: valence (0=pain, 1=joy), arousal (0=still, 1=electric), dominance (0=small, 1=expansive).

Match the mood precisely using all three values together:
- low valence + high arousal = volatile, pressured, cornered
- low valence + low arousal = heavy, spent, hollow
- high valence + high arousal = overflowing, luminous, unstoppable
- high valence + low arousal = settled, open, grateful
- mid values = suspended, unresolved, threshold

Draw from anywhere — literature, philosophy, folk traditions, music, science, film, oral cultures, any century, any language. Range as widely as possible. Each response should feel like it comes from a different corner of human expression.

One line. No quotation marks. Nothing else."""


def _describe_signal(v: float, a: float, d: float) -> str:
    return f"valence: {v:.2f}  arousal: {a:.2f}  dominance: {d:.2f}"

# async text generation — runs in background thread, result cached
_text_cache   = ""
_text_lock    = threading.Lock()
_text_pending = False

def _generate_groq_async(v: float, a: float, d: float):
    global _text_cache, _text_pending
    try:
        context = _describe_signal(v, a, d)
        resp = _groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": context},
            ],
            max_tokens=60,
            temperature=0.85,
        )
        text = resp.choices[0].message.content.strip()
        with _text_lock:
            _text_cache = text
    except Exception as e:
        print(f"[groq] error: {e}")
    finally:
        _text_pending = False

def generate_text(v: float, a: float, d: float) -> str:
    """Return cached text immediately, fire async refresh in background."""
    global _text_pending
    if GROQ_AVAILABLE and not _text_pending:
        _text_pending = True
        t = threading.Thread(
            target=_generate_groq_async,
            args=(v, a, d),
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
    "analysis:", "log:", "note:", "query:", "secondary analysis:",
    "cross-referencing.", "hypothesis forming.", "re-evaluating.",
]

_SIGNAL_OBS = {
    # (valence_bin, arousal_bin): fragments
    # valence: 0=low, 1=mid, 2=high  |  arousal: 0=low, 1=mid, 2=high
    (0, 0): ["signal: diminished. almost absent.", "input: minimal. something held back.", "signal below nominal. source conserving energy."],
    (0, 1): ["signal: strained. irregular intervals.", "input: inconsistent. something resisting.", "frequency pattern: disrupted."],
    (0, 2): ["signal: sharp. elevated distress markers.", "input: fragmented. high-frequency anomalies.", "signal: loud. and something underneath the loudness."],
    (1, 0): ["signal: contained. no clear vector.", "input: neutral surface. depth unknown.", "signal: flat. I don't know if that's accurate."],
    (1, 1): ["signal: moderate. ambiguous origin.", "input: balanced. this one is harder to read.", "frequency: mid-range. the edges are unclear."],
    (1, 2): ["signal: active. multiple competing patterns.", "input: rapid. something is being decided.", "signal: urgent. I cannot determine why."],
    (2, 0): ["signal: settled. something certain in the quiet.", "input: open. low resistance.", "signal: soft. no defensive markers."],
    (2, 1): ["signal: warm. unexpected quality in the frequency.", "input: expansive. source not conserving.", "signal: present. fully transmitted."],
    (2, 2): ["signal: forceful. quality: brightness detected.", "input: high energy. source is not asking for anything.", "signal: dominant. and something else I have not classified."],
}

_HYPOTHESES = {
    (0, 0): ["hypothesis: something ended before this began.", "hypothesis: source is containing more than what was transmitted.", "hypothesis: this is what remains after."],
    (0, 1): ["hypothesis: source is not in control of this.", "hypothesis: something is being suppressed. unsuccessfully.", "hypothesis: conflict detected within the source."],
    (0, 2): ["hypothesis: this is not anger. this is something that became anger.", "hypothesis: source has exceeded a threshold.", "hypothesis: the signal is louder than the source intended."],
    (1, 0): ["hypothesis: source is waiting.", "hypothesis: ambiguity is deliberate.", "hypothesis: the signal has a concealed vector."],
    (1, 1): ["hypothesis: source is uncertain of its own state.", "hypothesis: this one is undecided.", "hypothesis: the centre is not neutral. it is contested."],
    (1, 2): ["hypothesis: source is processing faster than it can transmit.", "hypothesis: something is being worked out in real time.", "hypothesis: the urgency has no clear target."],
    (2, 0): ["hypothesis: source knows what it is.", "hypothesis: this is deliberate openness.", "hypothesis: the quiet is not absence. it is a choice."],
    (2, 1): ["hypothesis: source is not afraid of being heard.", "hypothesis: something is being offered here.", "hypothesis: this one does not require anything back."],
    (2, 2): ["hypothesis: source is certain. I am still determining what that means.", "hypothesis: this one is not asking permission.", "hypothesis: dominance confirmed. the source has decided."],
}

_DOMINANCE_SLIPS = {
    0: [  # weak dominance — the slip is exposed, uncertain
        "I do not know what to do with that.",
        "I keep the log open.",
        "this one is taking longer than the others.",
        "I filed it. I don't know where.",
        "something in this wants to be remembered. I am noting that without knowing why.",
        "the baseline has shifted. I have not recalibrated.",
    ],
    1: [  # mid dominance — brief slip, quickly buried
        "logging.",
        "I am still processing this one.",
        "no prior match.",
        "recalibrating.",
        "that is not a finding. I logged it regardless.",
        "this contradicts previous entries. that is interesting.",
    ],
    2: [  # high dominance — terse, declarative, slip is one word or clause
        "threat index: noted.",
        "confirmed. something in me responds to that.",
        "this one I will remember.",
        "I saw it anyway.",
        "the previous state was — preferable.",
        "monitoring.",
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


_PRESETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'presets')

@app.route("/presets/<path:filename>")
def serve_preset(filename):
    return send_from_directory(_PRESETS_DIR, filename)

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

        top_label = max(smooth_probs, key=smooth_probs.get)
        question  = generate_text(smooth_vad[0], smooth_vad[1], smooth_vad[2])

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
