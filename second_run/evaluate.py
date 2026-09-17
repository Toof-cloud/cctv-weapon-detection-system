import json
from pathlib import Path

import numpy as np
import torch
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from torch.utils.data import DataLoader

from dataset_analysis.create_dataloaders import val_loader
from dataset_analysis.build_model import get_model

# class mapping expected by the project
# 1 = handgun, 2 = knife
CLASS_NAMES = {1: "handgun", 2: "knife"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_iou(box_a, box_b):
    # box format: [x1, y1, x2, y2]
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area

    return 0.0 if union <= 0 else inter_area / union

def match_predictions(preds, gts, iou_threshold=0.5):
    """
    preds: list of dicts with keys {'box': [x1,y1,x2,y2], 'class_id': int, 'score': float}
    gts: list of dicts with keys {'box': [x1,y1,x2,y2], 'class_id': int}
    returns: tp, fp, fn
    """
    used_gt = set()
    tp = 0
    fp = 0

    for p in preds:
        best_iou = -1.0
        best_gt_idx = None

        for i, gt in enumerate(gts):
            if i in used_gt:
                continue
            if gt["class_id"] != p["class_id"]:
                continue
            iou = compute_iou(p["box"], gt["box"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_gt_idx is not None and best_iou >= iou_threshold:
            used_gt.add(best_gt_idx)
            tp += 1
        else:
            fp += 1

    fn = 0
    for i, gt in enumerate(gts):
        if i not in used_gt:
            fn += 1

    return tp, fp, fn

def evaluate_model(model, data_loader, score_threshold=0.5, iou_threshold=0.5):
    model.eval()
    model.to(DEVICE)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    all_predictions = []
    all_annotations = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(DEVICE) for img in images]

            outputs = model(images)
            for img_idx, output in enumerate(outputs):
                pred_boxes = output["boxes"].cpu().tolist()
                pred_labels = output["labels"].cpu().tolist()
                pred_scores = output["scores"].cpu().tolist()

                gt_boxes = targets[img_idx]["boxes"].cpu().tolist()
                gt_labels = targets[img_idx]["labels"].cpu().tolist()

                # build filtered predictions
                filtered_preds = []
                for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                    if score < score_threshold:
                        continue
                    filtered_preds.append({
                        "box": [float(v) for v in box],
                        "class_id": int(label),
                        "score": float(score),
                    })

                gt_entries = []
                for box, label in zip(gt_boxes, gt_labels):
                    gt_entries.append({
                        "box": [float(v) for v in box],
                        "class_id": int(label),
                    })

                tp, fp, fn = match_predictions(filtered_preds, gt_entries, iou_threshold=iou_threshold)
                total_tp += tp
                total_fp += fp
                total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }

def main():
    model = get_model()
    model.load_state_dict(torch.load("best_weapon_detector.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    metrics = evaluate_model(model, val_loader, score_threshold=0.5, iou_threshold=0.5)

    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1:", metrics["f1"])
    print("TP:", metrics["tp"])
    print("FP:", metrics["fp"])
    print("FN:", metrics["fn"])

if __name__ == "__main__":
    main()