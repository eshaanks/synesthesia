import os
import random
import tempfile
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
import librosa
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

app = Flask(__name__)
CORS(app)

# ── superb emotion classifier — 4 classes, loads from local cache ─────────────
print("[server] loading emotion classifier...")
MODEL_ID      = "superb/wav2vec2-base-superb-er"
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, local_files_only=True)
emotion_model     = AutoModelForAudioClassification.from_pretrained(MODEL_ID, local_files_only=True)
emotion_model.eval()
LABELS = [emotion_model.config.id2label[i] for i in range(len(emotion_model.config.id2label))]
# LABELS = ['neu', 'hap', 'ang', 'sad']
print(f"[server] ready — labels: {LABELS}")

# ── smoothed probability state ─────────────────────────────────────────────────
# EMA over raw softmax probs so transitions are gradual not jumpy
smooth_probs = {l: 1.0 / len(LABELS) for l in LABELS}
EMA_ALPHA    = 0.25   # faster than before — classification is more stable than regression

TEMPLATES = [
    "something like {w}...",
    "is there {w} in this?",
    "the shape of {w}.",
    "closer to {w} than anything.",
    "what {w} sounds like.",
    "somewhere between {w} and silence.",
    "...{w}.",
    "a trace of {w}.",
]

EMOTION_LABELS = {
    "neu": "stillness",
    "hap": "warmth",
    "ang": "tension",
    "sad": "melancholy",
}


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

        rms = float(np.sqrt(np.mean(wav ** 2)))
        if rms < 0.01:
            return jsonify({
                "probs":        {l: round(smooth_probs[l], 4) for l in LABELS},
                "emotion":      max(smooth_probs, key=smooth_probs.get),
                "emotion_word": EMOTION_LABELS.get(max(smooth_probs, key=smooth_probs.get), "stillness"),
                "question":     "",
                "silent":       True,
            })

        # voice activity gate — only keep voiced frames
        frame_len = 512
        frames    = librosa.util.frame(wav, frame_length=frame_len, hop_length=frame_len).T
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        voiced    = frames[frame_rms > 0.008]
        if len(voiced) < 2:
            return jsonify({
                "probs":        {l: round(smooth_probs[l], 4) for l in LABELS},
                "emotion":      max(smooth_probs, key=smooth_probs.get),
                "emotion_word": EMOTION_LABELS.get(max(smooth_probs, key=smooth_probs.get), "stillness"),
                "question":     "",
                "silent":       True,
            })
        wav_voiced = voiced.flatten()

        inputs = feature_extractor(wav_voiced, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = emotion_model(**inputs).logits[0]
        probs = F.softmax(logits, dim=-1).cpu().numpy()

        # EMA smooth each probability
        for i, label in enumerate(LABELS):
            smooth_probs[label] = EMA_ALPHA * float(probs[i]) + (1 - EMA_ALPHA) * smooth_probs[label]

        top_label = max(smooth_probs, key=smooth_probs.get)
        word      = EMOTION_LABELS.get(top_label, top_label)
        question  = random.choice(TEMPLATES).replace("{w}", word)

        print(f"[emotion] rms={rms:.4f} | " +
              " ".join(f"{l}={smooth_probs[l]:.3f}" for l in LABELS) +
              f" | top={top_label}")

        return jsonify({
            "probs":        {l: round(smooth_probs[l], 4) for l in LABELS},
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
