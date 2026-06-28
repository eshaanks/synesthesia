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

app = Flask(__name__)
CORS(app)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "full_dataset")
EMBEDDINGS  = os.path.join(BASE_DIR, "embeddings", "full_database.json")

os.chdir(os.path.join(BASE_DIR, "..", "ImageBind"))

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[server] loading model on {device}...")
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)
print("[server] model ready")

with open(EMBEDDINGS, "r") as f:
    database = json.load(f)

db_vectors    = np.array([e["vector"] for e in database])
db_filenames  = [e["filename"] for e in database]
db_paths      = [e["path"] for e in database]
db_categories = [e["category"] for e in database]

category_indices = {}
for i, cat in enumerate(db_categories):
    category_indices.setdefault(cat, []).append(i)

print(f"[server] {len(database)} images across {len(category_indices)} categories")

def rank_normalize(similarities):
    n = len(similarities)
    ranks = np.argsort(np.argsort(similarities)).astype(float)
    return ranks / (n - 1)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "images": len(database), "categories": len(category_indices)})

@app.route("/search", methods=["POST"])
def search():
    if "audio" not in request.files:
        return jsonify({"error": "no audio file"}), 400

    audio_file = request.files["audio"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        inputs = {
            ModalityType.AUDIO: data.load_and_transform_audio_data([tmp_path], device),
        }
        with torch.no_grad():
            embeddings = model(inputs)

        audio_vector  = embeddings[ModalityType.AUDIO].cpu().numpy()[0]
        audio_norm    = np.linalg.norm(audio_vector)
        db_norms      = np.linalg.norm(db_vectors, axis=1)
        raw_sim       = (db_vectors @ audio_vector) / (db_norms * audio_norm)

        # rank normalize — histogram equalization on similarity scores
        rank_scores   = rank_normalize(raw_sim)

        # category histogram — average rank score per category
        category_scores = {}
        for cat, indices in category_indices.items():
            category_scores[cat] = float(np.mean(rank_scores[indices]))

        winning_category = max(category_scores, key=category_scores.get)

        # top 5 from winning category
        winning_indices = category_indices[winning_category]
        winning_scores  = [(i, float(rank_scores[i])) for i in winning_indices]
        winning_scores.sort(key=lambda x: x[1], reverse=True)
        top5 = winning_scores[:5]

        results = [
            {
                "filename":           db_filenames[i],
                "category":           db_categories[i],
                "path":               db_paths[i],
                "raw_score":          float(raw_sim[i]),
                "rank_score":         score,
                "winning_category":   winning_category,
                "category_confidence": category_scores[winning_category],
            }
            for i, score in top5
        ]

        cat_ranking = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)

        return jsonify({
            "matches":          results,
            "winning_category": winning_category,
            "category_ranking": cat_ranking[:8],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)

@app.route("/image/<path:filepath>")
def serve_image(filepath):
    directory = os.path.join(DATASET_DIR, os.path.dirname(filepath))
    filename  = os.path.basename(filepath)
    return send_from_directory(directory, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
