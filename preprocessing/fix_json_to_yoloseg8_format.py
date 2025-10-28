import os
import cv2
import json
import numpy as np
import random
from pathlib import Path

# Paths
image_dir = Path("./data/images")
json_dir = Path("./data/jsons")   # your converted jsons
base_yolo_dir = Path("dataset")

# Train/val directories
yolo_images_train = base_yolo_dir / "images" / "train"
yolo_images_val   = base_yolo_dir / "images" / "val"
yolo_images_test   = base_yolo_dir / "images" / "test"
yolo_labels_train = base_yolo_dir / "labels" / "train"
yolo_labels_val   = base_yolo_dir / "labels" / "val"
yolo_labels_test   = base_yolo_dir / "labels" / "test"

# Create directories
for d in [yolo_images_train, yolo_images_val, yolo_images_test, yolo_labels_train, yolo_labels_val, yolo_labels_test]:
    d.mkdir(parents=True, exist_ok=True)

# Collect relevant image names
names = [
    name.split('.')[0]
    for name in sorted(os.listdir(image_dir))
    if name.endswith('.png') or name.endswith('.jpg')
]

# Shuffle & split
random.seed(42)  # reproducible split
random.shuffle(names)
split_train = int(0.7 * len(names))
split_val = int(0.9 * len(names))  # 0.7 + 0.2 = 0.9

train_names = names[:split_train]
val_names = names[split_train:split_val]
test_names = names[split_val:]

def process_and_save(name, img_dir, lbl_dir):
    img_path = image_dir / f"{name}.png"
    json_path = json_dir / f"{name}.json"

    if not json_path.exists():
        print(f"⚠️ Missing JSON for {name}, skipping")
        return

    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️ Could not read {img_path}, skipping")
        return
    h, w = img.shape[:2]

    # Read contours from JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    contours = data.get("points", [])

    # Save image
    cv2.imwrite(str(img_dir / f"{name}.png"), img)

    # Save YOLO segmentation labels
    label_file = lbl_dir / f"{name}.txt"
    with open(label_file, "w") as f:
        for contour in contours:
            contour = np.array(contour)

            if contour.ndim == 3 and contour.shape[1] == 1:  # (n,1,2)
                contour = contour[:, 0, :]

            if contour.ndim != 2 or contour.shape[1] != 2:
                print(f"⚠️ Bad contour shape {contour.shape} in {name}, skipping object")
                continue

            # Normalize
            contour_norm = contour.astype(np.float32)
            contour_norm[:, 0] /= w
            contour_norm[:, 1] /= h

            # Flatten & format
            coords_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in contour_norm])
            f.write(f"0 {coords_str}\n")  # class 0 = "cell"

    print(f"✅ Processed {name}, {len(contours)} objects → {img_dir}")

# Process train set
for name in train_names:
    process_and_save(name, yolo_images_train, yolo_labels_train)

# Process val set
for name in val_names:
    process_and_save(name, yolo_images_val, yolo_labels_val)

for name in test_names:
    process_and_save(name, yolo_images_test, yolo_labels_test)

print(f"📊 Done! Train: {len(train_names)} images, Val: {len(val_names)} images, Test: {len(test_names)}")
