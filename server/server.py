import os
import json
import tempfile
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

# ── app setup ────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings", "image_database.json")

# ── load model once at startup ───────────────────────────────
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[server] loading ImageBind on {device}...")
os.chdir(os.path.join(BASE_DIR, "..", "ImageBind"))
model = imagebind_model.imagebind_huge(pretrained=True)
os.chdir(BASE_DIR)
model.eval()
model.to(device)
print("[server] model ready")

# ── load image database once at startup ──────────────────────
with open(EMBEDDINGS_PATH, "r") as f:
    database = json.load(f)
db_vectors = np.array([entry["vector"] for entry in database])
db_filenames = [entry["filename"] for entry in database]
print(f"[server] database loaded — {len(database)} images")

# ── routes ───────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok", "images": len(database)})

@app.route("/search", methods=["POST"])
def search():
    if "audio" not in request.files:
        return jsonify({"error": "no audio file"}), 400

    audio_file = request.files["audio"]

    # save incoming audio chunk to a temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # encode audio with ImageBind
        inputs = {
            ModalityType.AUDIO: data.load_and_transform_audio_data([tmp_path], device),
        }
        with torch.no_grad():
            embeddings = model(inputs)

        audio_vector = embeddings[ModalityType.AUDIO].cpu().numpy()[0]

        # cosine similarity against every image in database
        audio_norm = np.linalg.norm(audio_vector)
        db_norms = np.linalg.norm(db_vectors, axis=1)
        similarities = (db_vectors @ audio_vector) / (db_norms * audio_norm)

        # return top 5
        top5_idx = np.argsort(similarities)[::-1][:5]
        results = [
            {
                "filename": db_filenames[i],
                "score": float(similarities[i])
            }
            for i in top5_idx
        ]

        return jsonify({"matches": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        os.unlink(tmp_path)

@app.route("/image/<filename>")
def serve_image(filename):
    return send_from_directory(DATASET_DIR, filename)

# ── run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
