import os
import sys
import random
import tempfile
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import librosa

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

TEMPLATES = [
    "something like {w}...",
    "is there {w} in this?",
    "the shape of {w}.",
    "closer to {w} than anything.",
    "what {w} sounds like.",
    "...{w}.",
    "a trace of {w}.",
]

EMOTION_LABELS = {
    "neu": "stillness",
    "hap": "warmth",
    "ang": "tension",
    "sad": "melancholy",
}


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
                "emotion_word": EMOTION_LABELS.get(max(smooth_probs, key=smooth_probs.get)),
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
        word      = EMOTION_LABELS.get(top_label, top_label)
        question  = random.choice(TEMPLATES).replace("{w}", word)

        print(f"[vad] rms={rms:.4f} | "
              f"V={smooth_vad[0]:.3f} A={smooth_vad[1]:.3f} D={smooth_vad[2]:.3f} | "
              + " ".join(f"{l}={smooth_probs[l]:.3f}" for l in LABELS)
              + f" | top={top_label}")

        return jsonify({
            "probs":        smooth_probs,
            "vad":          [round(float(x), 4) for x in smooth_vad],  # type: ignore
            "emotion":      top_label,
            "emotion_word": word,
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
