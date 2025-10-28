import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

image_dir = Path("./data/images")
pickle_dir = Path("./data/pickles")
yolo_images_dir = Path("dataset/images/train")
yolo_labels_dir = Path("dataset/labels/train")

yolo_images_dir.mkdir(parents=True, exist_ok=True)
yolo_labels_dir.mkdir(parents=True, exist_ok=True)

names = [name.split('.')[0] for name in sorted(os.listdir(image_dir)) if name.endswith('.png') and name.startswith('samp4_3_')]

for name in names:
    img_path = image_dir / f"{name}.png"
    pkl_path = pickle_dir / f"{name}.pkl"

    if not pkl_path.exists():
        continue

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    contours = pd.read_pickle(pkl_path)

    cv2.imwrite(str(yolo_images_dir / f"{name}.png"), img)

    label_file = yolo_labels_dir / f"{name}.txt"
    with open(label_file, "w") as f:
        for contour in contours:
            if contour.ndim == 3:  
                contour = contour[:, 0, :]
            contour_norm = contour.astype(np.float32)
            contour_norm[:, 0] /= w
            contour_norm[:, 1] /= h
            # Reformat to YOLO format
            coords_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in contour_norm])
            f.write(f"0 {coords_str}\n")  # class 0 = "cell"
    print(f"Processed {name}, {len(contours)} objects")
