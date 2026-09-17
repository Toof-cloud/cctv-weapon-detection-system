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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_analysis.build_model import get_model
from training.model_8.eighth_model_dataset import EighthModelDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = ROOT / "best_weapon_detector_eighth_model.pth"
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
    ap = np.sum((recalls[recall_change_indices + 1] - recalls[recall_change_indices]) * precisions[recall_change_indices + 1])

    return float(ap)


def evaluate_model():
    print("=" * 76)
    print("      MODEL 8 INDEPENDENT EVALUATION: TEST SPLIT (265 HELD-OUT IMAGES)")
    print("=" * 76)
    print(f"Evaluation Checkpoint: {MODEL_PATH}")
    print(f"Compute Device:        {DEVICE}")
    print(f"Confidence Threshold:  {CONFIDENCE_THRESHOLD:.2f}")
    print(f"IoU Threshold:         {IOU_THRESHOLD:.2f}")
    print(f"RPN Anchor Generator:  Balanced Scales ((16,), (32,), (64,), (128,), (256,))")
    print("=" * 76)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")

    # Load Model 8 architecture with balanced anchors
    model = get_model(num_classes=3, anchor_scales=(16, 32, 64, 128, 256))
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    test_dataset = EighthModelDataset("test", augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    print(f"\nRunning inference on {len(test_dataset)} test split images...")

    predictions = []
    gt_by_image = []

    tp_total = 0
    fp_total = 0
    fn_total = 0

    negative_images_count = 0
    negative_images_zero_fp = 0

    image_counter = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()

                gts = [
                    {"box": b.tolist(), "class_id": int(l)}
                    for b, l in zip(gt_boxes, gt_labels)
                ]
                gt_by_image.append(gts)

                is_negative_image = (len(gts) == 0)
                if is_negative_image:
                    negative_images_count += 1

                pred_boxes = output["boxes"].cpu().numpy()
                pred_labels = output["labels"].cpu().numpy()
                pred_scores = output["scores"].cpu().numpy()

                image_fps_found = False

                for b, l, s in zip(pred_boxes, pred_labels, pred_scores):
                    if float(s) >= CONFIDENCE_THRESHOLD:
                        predictions.append({
                            "img_idx": image_counter,
                            "box": b.tolist(),
                            "class_id": int(l),
                            "score": float(s),
                        })
                        if is_negative_image:
                            image_fps_found = True

                if is_negative_image and not image_fps_found:
                    negative_images_zero_fp += 1

                # Calculate TP/FP/FN for this image
                active_preds = [
                    {"box": b.tolist(), "class_id": int(l), "score": float(s)}
                    for b, l, s in zip(pred_boxes, pred_labels, pred_scores)
                    if float(s) >= CONFIDENCE_THRESHOLD
                ]

                matched_gt_indices = set()
                img_tps = 0
                img_fps = 0

                for pred in active_preds:
                    matched = False
                    for gt_idx, gt in enumerate(gts):
                        if gt_idx not in matched_gt_indices and pred["class_id"] == gt["class_id"]:
                            iou = compute_iou(pred["box"], gt["box"])
                            if iou >= IOU_THRESHOLD:
                                matched = True
                                matched_gt_indices.add(gt_idx)
                                break
                    if matched:
                        img_tps += 1
                    else:
                        img_fps += 1

                img_fns = len(gts) - len(matched_gt_indices)

                tp_total += img_tps
                fp_total += img_fps
                fn_total += img_fns

                image_counter += 1

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    ap_handgun = compute_ap_for_class(predictions, gt_by_image, class_id=1, iou_threshold=IOU_THRESHOLD)
    ap_knife = compute_ap_for_class(predictions, gt_by_image, class_id=2, iou_threshold=IOU_THRESHOLD)
    mAP = (ap_handgun + ap_knife) / 2.0

    neg_specificity = (
        (negative_images_zero_fp / negative_images_count * 100.0)
        if negative_images_count > 0
        else 100.0
    )

    print("\n" + "=" * 76)
    print("                  MODEL 8 TEST SET EVALUATION RESULTS")
    print("=" * 76)
    print(f"Total Test Images Analyzed:        {len(test_dataset)}")
    print(f"Handgun Ground Truths:             {sum(len([g for g in gts if g['class_id'] == 1]) for gts in gt_by_image)}")
    print(f"Knife Ground Truths:               {sum(len([g for g in gts if g['class_id'] == 2]) for gts in gt_by_image)}")
    print(f"Negative (Zero-Box) Test Images:   {negative_images_count}")
    print("-" * 76)
    print(f"True Positives (TP):               {tp_total}")
    print(f"False Positives (FP):              {fp_total}")
    print(f"False Negatives (FN):              {fn_total}")
    print("-" * 76)
    print(f"Precision:                         {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:                            {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-Score:                          {f1_score:.4f} ({f1_score * 100:.2f}%)")
    print("-" * 76)
    print(f"Average Precision (Handgun):       {ap_handgun:.4f} ({ap_handgun * 100:.2f}%)")
    print(f"Average Precision (Knife):         {ap_knife:.4f} ({ap_knife * 100:.2f}%)")
    print(f"Mean Average Precision (mAP@0.5):  {mAP:.4f} ({mAP * 100:.2f}%)")
    print("-" * 76)
    print(f"Negative Specificity (Zero-Alarm): {negative_images_zero_fp}/{negative_images_count} ({neg_specificity:.2f}%)")
    print("=" * 76)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "ap_handgun": ap_handgun,
        "ap_knife": ap_knife,
        "map": mAP,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "neg_specificity": neg_specificity,
    }


if __name__ == "__main__":
    evaluate_model()
