import os
import random
import cv2
import numpy as np
from ultralytics import YOLO
import csv


def draw_contours(vis_image, contours, alpha=0.5):
    vis_image_copy = vis_image.copy()
    for contour in contours:
        cv2.drawContours(
            vis_image_copy,
            [contour],
            -1,
            [random.randint(0, 255) for _ in range(3)],
            -1
        )
    return cv2.addWeighted(vis_image, 1 - alpha, vis_image_copy, alpha, 0)


def create_mask(vis_image, contours):
    mask = np.zeros(vis_image.shape[:2], dtype=np.uint8)
    for contour in contours:
        cv2.drawContours(mask, [contour], -1, 255, -1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.medianBlur(mask, 3)
    return mask


def create_contour_mask(vis_image, contours):
    mask = np.zeros(vis_image.shape[:2], dtype=np.uint8)
    for contour in contours:
        cv2.drawContours(mask, [contour], -1, 255, -1)
    result = np.zeros_like(vis_image)
    result[mask == 255] = vis_image[mask == 255]
    return result


def create_background_mask(vis_image, contours):
    result = vis_image.copy()
    for contour in contours:
        cv2.drawContours(result, [contour], -1, [255, 255, 255], -1)
    return result


def load_gt_contours(label_file, img_width, img_height):
    if not os.path.exists(label_file):
        return []
    contours = []
    with open(label_file, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            coords = list(map(float, parts[1:]))  # skip class_id
            points = np.array(coords, dtype=np.float32).reshape(-1, 2)
            points[:, 0] *= img_width
            points[:, 1] *= img_height
            contour = np.int32(points.reshape(-1, 1, 2))
            contours.append(contour)
    return contours


def mask_from_contour(contour, shape):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask


def center_of_contour(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy


def match_contours_center(pred_contours, gt_contours, shape):
    tp = []
    fp = []
    fn = []

    # Precompute GT masks
    gt_masks = [mask_from_contour(g, shape) for g in gt_contours]

    for p in pred_contours:
        center = center_of_contour(p)
        if center is None:
            fp.append(p)
            continue

        matched = False
        x, y = center
        for gt_mask in gt_masks:
            if gt_mask[y, x] > 0:  # center inside GT
                tp.append(p)
                matched = True
                break
        if not matched:
            fp.append(p)

    for i, g in enumerate(gt_contours):
        matched = False
        g_mask = gt_masks[i]
        for t in tp:
            c = center_of_contour(t)
            if c is not None:
                x, y = c
                if g_mask[y, x] > 0:
                    matched = True
                    break
        if not matched:
            fn.append(g)

    return tp, fp, fn


def draw_eval(image, tp, fp, fn, box_size=20):
    """
    Draws colored bounding boxes (not contours) around contour centers.
    box_size: half the width/height of each box.
    """
    vis = image.copy()

    def draw_center_box(vis_img, contour_list, color):
        for contour in contour_list:
            c = center_of_contour(contour)
            if c is None:
                continue
            cx, cy = c
            x1 = max(0, cx - box_size)
            y1 = max(0, cy - box_size)
            x2 = min(vis_img.shape[1] - 1, cx + box_size)
            y2 = min(vis_img.shape[0] - 1, cy + box_size)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            cv2.circle(vis_img, (cx, cy), 2, color, -1)

    # TP – green
    draw_center_box(vis, tp, (0, 255, 0))
    # FP – red
    draw_center_box(vis, fp, (0, 0, 255))
    # FN – orange
    draw_center_box(vis, fn, (0, 165, 255))

    return vis

def compute_overall_metrics(results_list):
    total_tp = sum(r[1] for r in results_list)
    total_fp = sum(r[2] for r in results_list)
    total_fn = sum(r[3] for r in results_list)

    if total_tp + total_fp > 0:
        precision = total_tp / (total_tp + total_fp)
    else:
        precision = 0.0

    if total_tp + total_fn > 0:
        recall = total_tp / (total_tp + total_fn)
    else:
        recall = 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    print("\n=== Overall Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    return precision, recall, f1


def run_inference():
    image_files = [f for f in sorted(os.listdir(image_path)) if f.lower().endswith((".png", ".jpg"))]
    results_list = []

    for name in image_files:
        img_path = os.path.join(image_path, name)
        label_path = os.path.join(labels_path, name.replace(".png", ".txt").replace(".jpg", ".txt"))
        image = cv2.imread(img_path)
        if image is None:
            continue

        results = model.predict(img_path, conf=conf_value, iou=iou_value, verbose=False)

        contours_all = []
        for r in results:
            if r.masks is None:
                continue
            for m in r.masks.xy:  # list of [N,2] arrays
                contour = np.array(m, dtype=np.int32).reshape(-1, 1, 2)
                contours_all.append(contour)


        if not contours_all:
            print(f"No objects in {name}")
            results_list.append([name, 0, 0, 0])
            continue


        gt_contours = load_gt_contours(label_path, image.shape[1], image.shape[0])
        if gt_contours:
            cv2.imwrite(os.path.join(outputs_path, f"{name.replace('.png', '')}_gt.png"),
                        draw_contours(image, gt_contours, alpha=0.5))

            tp, fp, fn = match_contours_center(contours_all, gt_contours, image.shape)

            eval_img = draw_eval(image, tp, fp, fn)
            cv2.imwrite(os.path.join(outputs_path, f"{name.replace('.png', '')}_eval.png"), eval_img)
            results_list.append([name, len(tp), len(fp), len(fn)])
        else:
            results_list.append([name, 0, len(contours_all), 0])
    return results_list

if __name__ == "__main__":
    model_path = "spore_yolov8m_aug/cells4/weights/best.pt"  # path to your trained YOLOv8 model
    image_path = "dataset/images/test"  # test images dir
    labels_path = "dataset/labels/test"  # test labels dir (GT)
    conf_values = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    iou_values = [0.4, 0.45,0.5, 0.55, 0.6, 0.65, 0.7]


    model = YOLO(model_path)
    metrics_list = []
    for conf_value in conf_values:
        for iou_value in iou_values:
            outputs_path = f"final_validation/conf_{conf_value}_iou_{iou_value}"

            os.makedirs(outputs_path, exist_ok=True)
            results_list = run_inference()
            precision, recall, f1 = compute_overall_metrics(results_list)
            metrics_list.append([iou_value, conf_value, precision, recall, f1])
            csv_path = os.path.join(outputs_path, f"evaluation_results_{precision:.2f}_{recall:.2f}_{f1:.2f}.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["image_name", "TP", "FP", "FN"])
                writer.writerows(results_list)

            print(f"\nSaved evaluation results to: {csv_path}")

    with open(f"final_validation/total_evaluation_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iou", "conf", "precision", "recall", "f1"])
        writer.writerows(metrics_list)

    print(f"\nSaved evaluation results to: {csv_path}")
