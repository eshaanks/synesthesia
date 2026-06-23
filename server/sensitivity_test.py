import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'ImageBind'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'ImageBind'))

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[test] loading model on {device}...")
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)
print("[test] model ready\n")

BASE      = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'test_dataset')
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'ImageBind', '.assets')

audio_files = {
    "bird": os.path.join(AUDIO_DIR, "bird_audio.wav"),
    "car":  os.path.join(AUDIO_DIR, "car_audio.wav"),
    "dog":  os.path.join(AUDIO_DIR, "dog_audio.wav"),
}

image_files = sorted([f for f in os.listdir(BASE) if f.lower().endswith(('.jpg','.jpeg','.png'))])
image_paths = [os.path.join(BASE, f) for f in image_files]
print(f"[test] found {len(image_files)} images\n")

inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device)}
with torch.no_grad():
    embeddings = model(inputs)
image_vectors = embeddings[ModalityType.VISION].cpu().numpy()

def category(fname):
    p = fname[:3]
    return {
        'BOD': 'bright open desert',
        'CDE': 'cracked dry earth',
        'DCC': 'dense city crowd',
        'SMF': 'soft moss forest',
    }.get(p, 'dark stormy ocean' if fname.startswith('DS') else p)

for label, audio_path in audio_files.items():
    print(f"{'='*50}")
    print(f"AUDIO: {label}")
    print(f"{'='*50}")

    inputs = {ModalityType.AUDIO: data.load_and_transform_audio_data([audio_path], device)}
    with torch.no_grad():
        embeddings = model(inputs)
    audio_vec = embeddings[ModalityType.AUDIO].cpu().numpy()[0]

    scores = []
    for i, img_vec in enumerate(image_vectors):
        sim = np.dot(audio_vec, img_vec) / (np.linalg.norm(audio_vec) * np.linalg.norm(img_vec))
        scores.append((image_files[i], float(sim)))
    scores.sort(key=lambda x: x[1], reverse=True)

    print("Top 5:")
    for fname, score in scores[:5]:
        print(f"  {score:.4f}  {fname:<20}  ({category(fname)})")

    print("\nCategory ranking:")
    cat_scores = {}
    for fname, score in scores:
        cat = category(fname)
        cat_scores.setdefault(cat, []).append(score)
    cat_avg = sorted({k: np.mean(v) for k,v in cat_scores.items()}.items(), key=lambda x: x[1], reverse=True)
    for cat, avg in cat_avg:
        bar = '█' * int(avg * 300)
        print(f"  {avg:.4f}  {cat:<25} {bar}")
    print()
