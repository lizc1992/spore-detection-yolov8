import os
import random
import cv2
import numpy as np
import csv
import yaml
import logging
from ultralytics import YOLO


def setup_logging(level="INFO"):
    logging.basicConfig(level=getattr(logging, level.upper()),
                        format="%(asctime)s - %(levelname)s - %(message)s")


def load_config(path="config/evaluation.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def draw_contours(vis_image, contours, alpha=0.5):
    vis_image_copy = vis_image.copy()
    for contour in contours:
        cv2.drawContours(vis_image_copy, [contour], -1,
                         [random.randint(0, 255) for _ in range(3)], -1)
    return cv2.addWeighted(vis_image, 1 - alpha, vis_image_copy, alpha, 0)


def load_gt_contours(label_file, img_width, img_height):
    if not os.path.exists(label_file):
        return []
    contours = []
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            coords = list(map(float, parts[1:]))
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
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


def match_contours_center(pred_contours, gt_contours, shape):
    tp, fp, fn = [], [], []

    gt_masks = [mask_from_contour(g, shape) for g in gt_contours]

    for p in pred_contours:
        c = center_of_contour(p)
        if c is None:
            fp.append(p)
            continue
        x, y = c
        if any(gt_mask[y, x] > 0 for gt_mask in gt_masks):
            tp.append(p)
        else:
            fp.append(p)

    for i, g in enumerate(gt_contours):
        matched = False
        g_mask = gt_masks[i]
        for t in tp:
            c = center_of_contour(t)
            if c and g_mask[c[1], c[0]] > 0:
                matched = True
                break
        if not matched:
            fn.append(g)

    return tp, fp, fn


def draw_eval(image, tp, fp, fn, box_size=20):
    vis = image.copy()

    def draw_center_box(list_contours, color):
        for ctr in list_contours:
            c = center_of_contour(ctr)
            if not c:
                continue
            cx, cy = c
            x1, y1 = max(0, cx-box_size), max(0, cy-box_size)
            x2, y2 = min(vis.shape[1]-1, cx+box_size), min(vis.shape[0]-1, cy+box_size)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.circle(vis, (cx, cy), 2, color, -1)

    draw_center_box(tp, (0, 255, 0))
    draw_center_box(fp, (0, 0, 255))
    draw_center_box(fn, (0, 165, 255))

    return vis


def compute_overall_metrics(results_list):
    total_tp = sum(r[1] for r in results_list)
    total_fp = sum(r[2] for r in results_list)
    total_fn = sum(r[3] for r in results_list)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if precision+recall else 0

    logging.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return precision, recall, f1


def run_eval(model, cfg, conf_value, iou_value):
    image_path = cfg["data"]["image_path"]
    labels_path = cfg["data"]["labels_path"]
    outputs_path = cfg["output"]["results_dir"]

    image_files = sorted([f for f in os.listdir(image_path)
                          if f.lower().endswith((".png", ".jpg"))])

    results_list = []

    for name in image_files:
        img_file = os.path.join(image_path, name)
        label_file = os.path.join(labels_path, name.rsplit(".", 1)[0] + ".txt")
        image = cv2.imread(img_file)
        if image is None:
            continue

        results = model.predict(img_file, conf=conf_value, iou=iou_value, verbose=False)

        contours_all = []
        for r in results:
            if r.masks is not None:
                for m in r.masks.xy:
                    contours_all.append(np.array(m, dtype=np.int32).reshape(-1, 1, 2))

        if not contours_all:
            results_list.append([name, 0, 0, 0])
            continue

        gt_contours = load_gt_contours(label_file, image.shape[1], image.shape[0])

        if gt_contours:
            tp, fp, fn = match_contours_center(contours_all, gt_contours, image.shape)
            eval_img = draw_eval(image, tp, fp, fn)
            cv2.imwrite(os.path.join(outputs_path, f"{name}_eval.png"), eval_img)
            results_list.append([name, len(tp), len(fp), len(fn)])
        else:
            results_list.append([name, 0, len(contours_all), 0])

    return results_list


if __name__ == "__main__":
    cfg = load_config()
    setup_logging(cfg["logging"]["level"])

    model = YOLO(cfg["model"]["checkpoint_path"])

    os.makedirs(cfg["output"]["results_dir"], exist_ok=True)
    summary_csv = os.path.join(cfg["output"]["results_dir"], "total_results.csv")

    metrics_list = []

    for conf_value in cfg["inference"]["conf_values"]:
        for iou_value in cfg["inference"]["iou_values"]:
            logging.info(f"Running: conf={conf_value}, iou={iou_value}")
            results_list = run_eval(model, cfg, conf_value, iou_value)
            precision, recall, f1 = compute_overall_metrics(results_list)

            csv_path = os.path.join(
                cfg["output"]["results_dir"],
                f"eval_conf{conf_value}_iou{iou_value}.csv"
            )
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["image_name", "TP", "FP", "FN"])
                writer.writerows(results_list)

            metrics_list.append([iou_value, conf_value, precision, recall, f1])

    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iou", "conf", "precision", "recall", "f1"])
        writer.writerows(metrics_list)

    logging.info("✅ Evaluation Complete.")
    logging.info(f"Saved summary: {summary_csv}")
