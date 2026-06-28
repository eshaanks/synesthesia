import os
from PIL import Image, ExifTags

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'full_dataset')

fixed = 0
skipped = 0

for category in sorted(os.listdir(DATASET_DIR)):
    cat_path = os.path.join(DATASET_DIR, category)
    if not os.path.isdir(cat_path):
        continue
    for fname in os.listdir(cat_path):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        fpath = os.path.join(cat_path, fname)
        try:
            img = Image.open(fpath)
            exif = img._getexif()
            if exif:
                for tag, value in exif.items():
                    if ExifTags.TAGS.get(tag) == 'Orientation':
                        if value == 3:
                            img = img.rotate(180, expand=True)
                        elif value == 6:
                            img = img.rotate(270, expand=True)
                        elif value == 8:
                            img = img.rotate(90, expand=True)
                        break
            # save without EXIF so WebGL always sees correct orientation
            img_no_exif = Image.new(img.mode, img.size)
            img_no_exif.putdata(list(img.getdata()))
            img_no_exif.save(fpath, quality=95)
            fixed += 1
        except Exception as e:
            skipped += 1

print(f"fixed {fixed} images, skipped {skipped}")
