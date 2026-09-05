import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_analysis.build_model import get_model
from training.fourth_model_dataset import FourthModelDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = ROOT / "best_weapon_detector_fourth_model.pth"
CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50
BATCH_SIZE = 2


def collate_fn(batch):
    return tuple(zip(*batch))


def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return 0.0 if union <= 0 else inter / union


def compute_ap_for_class(predictions, gt_by_image, class_id, iou_threshold=0.50):
    class_preds = [p for p in predictions if p["class_id"] == class_id]
    total_gts = sum(len([g for g in gts if g["class_id"] == class_id]) for gts in gt_by_image)

    if total_gts == 0:
        return 0.0

    if not class_preds:
        return 0.0

    class_preds.sort(key=lambda x: x["score"], reverse=True)

    tp_list = []
    fp_list = []
    matched_gts = set()

    for pred in class_preds:
        img_idx = pred["image_idx"]
        gts = [g for g in gt_by_image[img_idx] if g["class_id"] == class_id]

        best_iou = -1.0
        best_gt_id = None

        for idx, gt in enumerate(gts):
            gt_key = (img_idx, idx)
            if gt_key in matched_gts:
                continue
            iou = compute_iou(pred["box"], gt["box"])
            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_key

        if best_gt_id is not None and best_iou >= iou_threshold:
            tp_list.append(1)
            fp_list.append(0)
            matched_gts.add(best_gt_id)
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)

    precisions = tp_cum / (tp_cum + fp_cum)
    recalls = tp_cum / total_gts

    precisions = np.concatenate(([0.0], precisions, [0.0]))
    recalls = np.concatenate(([0.0], recalls, [1.0]))

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return float(ap)


def evaluate():
    test_dataset = FourthModelDataset("test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model = get_model(num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_predictions = []
    gt_by_image = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            for img_idx, output in enumerate(outputs):
                global_img_idx = batch_idx * BATCH_SIZE + img_idx
                gt_boxes = targets[img_idx]["boxes"].cpu().tolist()
                gt_labels = targets[img_idx]["labels"].cpu().tolist()

                gts = [
                    {"box": b, "class_id": int(l)}
                    for b, l in zip(gt_boxes, gt_labels)
                    if int(l) in {1, 2}
                ]
                gt_by_image.append(gts)

                pred_boxes = output["boxes"].cpu().tolist()
                pred_labels = output["labels"].cpu().tolist()
                pred_scores = output["scores"].cpu().tolist()

                filtered_preds = []
                for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                    if score >= CONFIDENCE_THRESHOLD and int(label) in {1, 2}:
                        p = {
                            "image_idx": global_img_idx,
                            "box": box,
                            "class_id": int(label),
                            "score": float(score),
                        }
                        filtered_preds.append(p)
                        all_predictions.append(p)

                # Match for overall TP, FP, FN
                used_gt = set()
                for p in filtered_preds:
                    best_iou = -1.0
                    best_gt_idx = None
                    for g_idx, g in enumerate(gts):
                        if g_idx in used_gt or g["class_id"] != p["class_id"]:
                            continue
                        iou = compute_iou(p["box"], g["box"])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = g_idx
                    if best_gt_idx is not None and best_iou >= IOU_THRESHOLD:
                        total_tp += 1
                        used_gt.add(best_gt_idx)
                    else:
                        total_fp += 1

                for g_idx in range(len(gts)):
                    if g_idx not in used_gt:
                        total_fn += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    ap_handgun = compute_ap_for_class(all_predictions, gt_by_image, class_id=1, iou_threshold=IOU_THRESHOLD)
    ap_knife = compute_ap_for_class(all_predictions, gt_by_image, class_id=2, iou_threshold=IOU_THRESHOLD)
    mAP_05 = (ap_handgun + ap_knife) / 2.0

    print("=" * 60)
    print("MODEL 4 EVALUATION RESULTS (Test Set: 333 images)")
    print("=" * 60)
    print(f"Checkpoint:           {MODEL_PATH.name}")
    print(f"Test images:          {len(test_dataset)}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.2f}")
    print(f"IoU threshold:        {IOU_THRESHOLD:.2f}")
    print(f"Precision:            {precision:.6f}")
    print(f"Recall:               {recall:.6f}")
    print(f"F1:                   {f1:.6f}")
    print(f"AP(Handgun):          {ap_handgun:.6f}")
    print(f"AP(Knife):            {ap_knife:.6f}")
    print(f"mAP@0.5:              {mAP_05:.6f}")
    print(f"TP:                   {total_tp}")
    print(f"FP:                   {total_fp}")
    print(f"FN:                   {total_fn}")
    print("=" * 60)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap_handgun": ap_handgun,
        "ap_knife": ap_knife,
        "mAP": mAP_05,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


if __name__ == "__main__":
    evaluate()
