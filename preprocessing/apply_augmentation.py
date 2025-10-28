import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import albumentations as A

input_dir = "dataset/images"
labels_dir = "dataset/labels"
output_img_dir = "dataset_aug/images"
output_lbl_dir = "dataset_aug/labels"

os.makedirs(f"{output_img_dir}/train", exist_ok=True)
os.makedirs(f"{output_lbl_dir}/train", exist_ok=True)
os.makedirs(f"{output_img_dir}/val", exist_ok=True)
os.makedirs(f"{output_lbl_dir}/val", exist_ok=True)
os.makedirs(f"{output_img_dir}/test", exist_ok=True)
os.makedirs(f"{output_lbl_dir}/test", exist_ok=True)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.4),   # image-only
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # image-only
    A.ElasticTransform(alpha=30, sigma=5, alpha_affine=5, p=0.3),
])

def read_segmentation_label(path):
    """
    Expect YOLO-style polygons per line:
      <cls> x1 y1 x2 y2 ... (normalized to [0,1])
    Returns:
      polys: List[List[Tuple[x,y]]]
      classes: List[int]
    """
    polys, classes = [], []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            # guard against odd-length coordinate lists
            if len(coords) < 6 or len(coords) % 2 != 0:
                continue
            poly = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            if len(poly) >= 3:
                polys.append(poly)
                classes.append(cls)
    return polys, classes

def save_segmentation_label(path, polygons, classes):
    """
    Save polygons back in YOLO-style normalized format.
    polygons and classes must be aligned.
    """
    with open(path, "w") as f:
        for cls, poly in zip(classes, polygons):
            if len(poly) < 3:
                continue
            flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
            f.write(f"{cls} {flat}\n")


def poly_to_binary_mask(poly, h, w):
    """
    poly: List[(x_norm, y_norm)]
    Returns HxW uint8 mask for a single polygon instance.
    """
    pts = np.array(
        [[int(round(x * w)), int(round(y * h))] for x, y in poly],
        dtype=np.int32
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 255)
    return mask

def polys_to_instance_masks(polygons, h, w):
    """
    Returns list of per-instance masks corresponding to 'polygons'.
    """
    return [poly_to_binary_mask(poly, h, w) for poly in polygons]

def mask_to_polygons_list(mask, min_area_px=5.0):
    """
    Convert a binary mask to a list of polygons (normalized later by caller).
    Uses external contours (YOLO segments typically store outer boundary).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if len(cnt) >= 3:
            area = cv2.contourArea(cnt)
            if area >= min_area_px:
                poly = [(float(x), float(y)) for [[x, y]] in cnt]
                polys.append(poly)
    return polys

def normalize_polygon(poly_px, w, h):
    return [(x / w, y / h) for (x, y) in poly_px]

if __name__ == "__main__":
    for img_path in tqdm(glob(f"{input_dir}/*/*.png")):
        if "train" in img_path:
            status = "train"
        elif "val" in img_path:
            status = "val"
        elif "test" in img_path:
            status = "test"
        else:
            status = "unknown"
        label_path = img_path.replace("images", "labels").replace(".png", ".txt")
        if not os.path.exists(label_path):
            continue
    
        image = cv2.imread(img_path)  
        if image is None:
            continue
        h, w = image.shape[:2]
    
        polygons, classes = read_segmentation_label(label_path)
        if not polygons:
            continue
    
        instance_masks = polys_to_instance_masks(polygons, h, w)
    
        subdir = os.path.basename(os.path.dirname(img_path))
        base = os.path.splitext(os.path.basename(img_path))[0]
    
        for i in range(5):
            transformed = transform(image=image, masks=instance_masks)
            aug_img = transformed["image"]
            aug_masks = transformed["masks"]  
    
            new_h, new_w = aug_img.shape[:2]

            out_polys_norm = []
            out_classes = []
            for cls, m in zip(classes, aug_masks):
                polys_px = mask_to_polygons_list(m, min_area_px=5.0)
                if not polys_px:
                    continue
                for poly_px in polys_px:
                    poly_norm = normalize_polygon(poly_px, new_w, new_h)
                    if len(poly_norm) >= 3:
                        out_polys_norm.append(poly_norm)
                        out_classes.append(cls)
    
            if not out_polys_norm:
                continue  # nothing valid after augmentation
    
            out_img_path = os.path.join(
                output_img_dir, f"{status}/{subdir}_{base}_aug{i}.jpg"
            )
            out_lbl_path = os.path.join(
                output_lbl_dir, f"{status}/{subdir}_{base}_aug{i}.txt"
            )
    
            cv2.imwrite(out_img_path, aug_img)
            save_segmentation_label(out_lbl_path, out_polys_norm, out_classes)
