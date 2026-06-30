import os
import random
import tempfile
import numpy as np
from scipy.stats import norm
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

app = Flask(__name__)
CORS(app)

# ── wav2vec2-emotion — load once at startup, CPU only ────────────────────────
print("[server] loading wav2vec2-emotion...")
MODEL_ID      = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
processor     = Wav2Vec2Processor.from_pretrained(MODEL_ID)
emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)
emotion_model.eval()
print("[server] ready")

# ── smoothed state — persists across requests ─────────────────────────────────
smooth_state  = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0, "word": "stillness"}
# seed history with small variance so stretch works from chunk 1
logit_history = {"v": list(np.linspace(-0.5, 0.5, 10)),
                 "a": list(np.linspace(-0.5, 0.5, 10)),
                 "d": list(np.linspace(-0.5, 0.5, 10))}

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

        # silence gate — skip inference if chunk is mostly silence
        rms = float(np.sqrt(np.mean(wav ** 2)))
        if rms < 0.01:
            return jsonify({
                "valence":      smooth_state["valence"],
                "arousal":      smooth_state["arousal"],
                "dominance":    smooth_state["dominance"],
                "emotion_word": smooth_state["word"],
                "question":     "",
                "silent":       True,
            })

        # voice activity — keep only frames above energy threshold
        frame_len = 512
        frames    = librosa.util.frame(wav, frame_length=frame_len, hop_length=frame_len).T
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        voiced    = frames[frame_rms > 0.008]
        if len(voiced) < 2:
            # not enough voiced frames — return smoothed state
            return jsonify({
                "valence":      smooth_state["valence"],
                "arousal":      smooth_state["arousal"],
                "dominance":    smooth_state["dominance"],
                "emotion_word": smooth_state["word"],
                "question":     "",
                "silent":       True,
            })
        wav_voiced = voiced.flatten()

        inputs = processor(wav_voiced, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = emotion_model(**inputs).logits[0].cpu().numpy()

        # model outputs [arousal, dominance, valence] as raw logits
        # stretch each scalar independently to [-1, 1] using a running
        # min/max window so the full range is used over time
        raw_a, raw_d, raw_v = logits[0], logits[1], logits[2]

        for key, val in [("a", raw_a), ("d", raw_d), ("v", raw_v)]:
            logit_history[key].append(float(val))
            if len(logit_history[key]) > 60:   # ~2min window at 2s chunks
                logit_history[key].pop(0)

        def gaussian_spread(val, key):
            # fit Gaussian to history, map val through CDF → uniform [0,1] → [-1,1]
            hist = np.array(logit_history[key])
            mu, sigma = float(hist.mean()), float(hist.std()) + 1e-6
            cdf = float(norm.cdf(val, loc=mu, scale=sigma))  # 0-1, Gaussian distributed
            return float(cdf * 2.0 - 1.0)                    # rescale to -1..+1

        raw_v = gaussian_spread(raw_v, "v")
        raw_a = gaussian_spread(raw_a, "a")
        raw_d = gaussian_spread(raw_d, "d")
        print(f"[raw logits] v={logits[2]:.3f} a={logits[0]:.3f} d={logits[1]:.3f} → spread v={raw_v:.3f} a={raw_a:.3f} d={raw_d:.3f}")

        # heavy EMA — alpha=0.1 means each new reading is only 10% weight
        # takes ~20 chunks (~40s) to fully settle, eliminates jitter
        α = 0.1
        smooth_state["valence"]   = α * raw_v + (1 - α) * smooth_state["valence"]
        smooth_state["arousal"]   = α * raw_a + (1 - α) * smooth_state["arousal"]
        smooth_state["dominance"] = α * raw_d + (1 - α) * smooth_state["dominance"]

        valence   = smooth_state["valence"]
        arousal   = smooth_state["arousal"]
        dominance = smooth_state["dominance"]

        word     = vad_to_word(valence, arousal, dominance)
        question = random.choice(TEMPLATES).replace("{w}", word)
        smooth_state["word"] = word

        print(f"[emotion] raw v={raw_v:.3f} a={raw_a:.3f} d={raw_d:.3f} | smoothed v={valence:.3f} a={arousal:.3f} d={dominance:.3f} | rms={rms:.4f} | voiced_frames={len(voiced)}")

        return jsonify({
            "valence":      round(valence, 4),
            "arousal":      round(arousal, 4),
            "dominance":    round(dominance, 4),
            "emotion_word": word,
            "question":     question,
            "silent":       False,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
