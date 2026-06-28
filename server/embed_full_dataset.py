import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'ImageBind'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'ImageBind'))

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[embed] loading model on {device}...")
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)
print("[embed] model ready\n")

DATASET_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'full_dataset')
OUTPUT_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'embeddings', 'full_database.json')
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# collect all images across all category subfolders
all_images = []
for category in sorted(os.listdir(DATASET_DIR)):
    cat_path = os.path.join(DATASET_DIR, category)
    if not os.path.isdir(cat_path):
        continue
    for fname in sorted(os.listdir(cat_path)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            all_images.append({
                'filename': fname,
                'category': category,
                'path': os.path.join(cat_path, fname)
            })

print(f"[embed] found {len(all_images)} images across {len(os.listdir(DATASET_DIR))} categories\n")

# embed in batches to avoid memory issues on M1
BATCH_SIZE = 10
database   = []
failed     = 0

for i in range(0, len(all_images), BATCH_SIZE):
    batch = all_images[i:i+BATCH_SIZE]
    paths = [img['path'] for img in batch]

    try:
        inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data(paths, device)
        }
        with torch.no_grad():
            embeddings = model(inputs)
        vectors = embeddings[ModalityType.VISION].cpu().numpy()

        for j, img in enumerate(batch):
            database.append({
                'filename': img['filename'],
                'category': img['category'],
                'path':     f"{img['category']}/{img['filename']}",
                'vector':   vectors[j].tolist()
            })

        print(f"  embedded {min(i+BATCH_SIZE, len(all_images))}/{len(all_images)}")

    except Exception as e:
        print(f"  batch {i} failed: {e}")
        failed += len(batch)

with open(OUTPUT_PATH, 'w') as f:
    json.dump(database, f)

print(f"\n[embed] done — {len(database)} images embedded, {failed} failed")
print(f"[embed] saved to {OUTPUT_PATH}")
