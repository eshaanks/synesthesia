import os
import random
import tempfile
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import librosa
from transformers import AutoProcessor, AutoModelForAudioClassification

app = Flask(__name__)
CORS(app)

# ── wav2vec2-emotion — load once at startup, CPU only ────────────────────────
print("[server] loading wav2vec2-emotion...")
MODEL_ID      = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
processor     = AutoProcessor.from_pretrained(MODEL_ID)
emotion_model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
emotion_model.eval()
print("[server] ready")

# ── VAD → emotion word ────────────────────────────────────────────────────────
def vad_to_word(v, a, d):
    if   v >  0.3 and a >  0.3:             return "euphoria"
    elif v >  0.3 and a < -0.3:             return "serenity"
    elif v < -0.3 and a >  0.3:             return "anguish"
    elif v < -0.3 and a < -0.3:             return "grief"
    elif v < -0.3 and d >  0.3:             return "rage"
    elif v < -0.3 and d < -0.3:             return "despair"
    elif v >  0.3 and d >  0.3:             return "joy"
    elif v >  0.3 and d < -0.3:             return "tenderness"
    elif abs(v) < 0.2 and a >  0.4:         return "tension"
    elif abs(v) < 0.2 and a < -0.4:         return "stillness"
    elif abs(v) < 0.2 and abs(a) < 0.2:     return "drift"
    elif d >  0.4:                           return "surge"
    elif d < -0.4:                           return "fragile"
    elif v >  0.1:                           return "warmth"
    elif v < -0.1:                           return "melancholy"
    else:                                    return "stillness"

TEMPLATES = [
    "i feel a hint of {w} here.",
    "something like {w}...",
    "is there {w} in this?",
    "the model reads this as {w}.",
    "...{w}. or something close.",
    "closer to {w} than anything.",
    "what {w} sounds like.",
    "the shape of {w}.",
    "somewhere between {w} and silence.",
]


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/search", methods=["POST"])
def search():
    if "audio" not in request.files:
        return jsonify({"error": "no audio file"}), 400

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        request.files["audio"].save(tmp.name)
        tmp_path = tmp.name

    try:
        wav, _ = librosa.load(tmp_path, sr=16000, mono=True)

        inputs = processor(wav, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = emotion_model(**inputs).logits[0].cpu().numpy()

        # model outputs [arousal, dominance, valence]
        arousal   = float(np.tanh(logits[0]))
        dominance = float(np.tanh(logits[1]))
        valence   = float(np.tanh(logits[2]))

        word     = vad_to_word(valence, arousal, dominance)
        question = random.choice(TEMPLATES).replace("{w}", word)

        return jsonify({
            "valence":      valence,
            "arousal":      arousal,
            "dominance":    dominance,
            "emotion_word": word,
            "question":     question,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
