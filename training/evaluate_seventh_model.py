import os
import sys
from pathlib import Path

# Thread guards for CPU stability
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_analysis.build_model import get_model
from training.seventh_model_dataset import SeventhModelDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = ROOT / "best_weapon_detector_seventh_model.pth"
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

    matched_gt = {img_idx: set() for img_idx in range(len(gt_by_image))}

    for pred in class_preds:
        img_idx = pred["img_idx"]
        pred_box = pred["box"]

        gts = [
            (gt_idx, gt["box"])
            for gt_idx, gt in enumerate(gt_by_image[img_idx])
            if gt["class_id"] == class_id
        ]

        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_box in gts:
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx not in matched_gt[img_idx]:
            tp_list.append(1)
            fp_list.append(0)
            matched_gt[img_idx].add(best_gt_idx)
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)

    recalls = tp_cumsum / total_gts
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    recall_change_indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum(
        (recalls[recall_change_indices + 1] - recalls[recall_change_indices])
        * precisions[recall_change_indices + 1]
    )

    return float(ap)


def main():
    print("=" * 70)
    print("        EVALUATING MODEL 7 ON HELD-OUT TEST SPLIT (397 IMAGES)       ")
    print(f"Model Checkpoint: {MODEL_PATH}")
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}\n"
            "Please train Model 7 first by running: python training/train_seventh_model.py"
        )

    test_dataset = SeventhModelDataset("test", augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model = get_model(num_classes=3, small_anchors=True).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    all_predictions = []
    all_ground_truths = []
    total_samples = 0
    negative_frame_count = 0
    clean_negatives_count = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = [image.to(DEVICE) for image in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                img_idx = total_samples
                total_samples += 1

                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()
                img_gts = [
                    {"box": box, "class_id": int(label)}
                    for box, label in zip(gt_boxes, gt_labels)
                    if int(label) in (1, 2)
                ]
                all_ground_truths.append(img_gts)

                is_negative_image = (len(img_gts) == 0)
                if is_negative_image:
                    negative_frame_count += 1

                pred_boxes = output["boxes"].cpu().numpy()
                pred_scores = output["scores"].cpu().numpy()
                pred_labels = output["labels"].cpu().numpy()

                image_has_fp_on_negative = False
                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    if score >= CONFIDENCE_THRESHOLD and int(label) in (1, 2):
                        all_predictions.append({
                            "img_idx": img_idx,
                            "box": box,
                            "score": float(score),
                            "class_id": int(label),
                        })
                        if is_negative_image:
                            image_has_fp_on_negative = True

                if is_negative_image and not image_has_fp_on_negative:
                    clean_negatives_count += 1

    tp = 0
    fp = 0
    total_gt = sum(len(gts) for gts in all_ground_truths)

    matched_gt = {img_idx: set() for img_idx in range(len(all_ground_truths))}
    sorted_preds = sorted(all_predictions, key=lambda x: x["score"], reverse=True)

    for pred in sorted_preds:
        img_idx = pred["img_idx"]
        pred_box = pred["box"]
        pred_class = pred["class_id"]

        gts = [
            (gt_idx, gt["box"])
            for gt_idx, gt in enumerate(all_ground_truths[img_idx])
            if gt["class_id"] == pred_class
        ]

        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_box in gts:
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= IOU_THRESHOLD and best_gt_idx not in matched_gt[img_idx]:
            tp += 1
            matched_gt[img_idx].add(best_gt_idx)
        else:
            fp += 1

    fn = total_gt - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    ap_handgun = compute_ap_for_class(all_predictions, all_ground_truths, class_id=1, iou_threshold=IOU_THRESHOLD)
    ap_knife = compute_ap_for_class(all_predictions, all_ground_truths, class_id=2, iou_threshold=IOU_THRESHOLD)
    mAP = (ap_handgun + ap_knife) / 2.0

    neg_specificity = (clean_negatives_count / negative_frame_count * 100) if negative_frame_count > 0 else 100.0

    print("\n" + "=" * 70)
    print("                  MODEL 7 TEST SET EVALUATION REPORT                  ")
    print("=" * 70)
    print(f"Checkpoint:                   {MODEL_PATH.name}")
    print(f"Test Images Evaluated:        {total_samples}")
    print(f"  - Weapon Images:            {total_samples - negative_frame_count}")
    print(f"  - Hard Negative Images:     {negative_frame_count}")
    print(f"Confidence Threshold:         {CONFIDENCE_THRESHOLD:.2f}")
    print(f"IoU Threshold:                {IOU_THRESHOLD:.2f}")
    print("-" * 70)
    print(f"Precision:                    {precision:.6f} ({precision * 100:.2f}%)")
    print(f"Recall:                       {recall:.6f} ({recall * 100:.2f}%)")
    print(f"F1 Score:                     {f1:.6f} ({f1 * 100:.2f}%)")
    print(f"AP (Handgun):                 {ap_handgun:.6f} ({ap_handgun * 100:.2f}%)")
    print(f"AP (Knife):                   {ap_knife:.6f} ({ap_knife * 100:.2f}%)")
    print(f"mAP@0.5:                      {mAP:.6f} ({mAP * 100:.2f}%)")
    print("-" * 70)
    print(f"True Positives (TP):          {tp}")
    print(f"False Positives (FP):         {fp}")
    print(f"False Negatives (FN):         {fn}")
    print(f"Negative Image Rejection Rate: {clean_negatives_count}/{negative_frame_count} ({neg_specificity:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
