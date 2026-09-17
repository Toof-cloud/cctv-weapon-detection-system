"""
Dedicated Evaluation Pipeline for Research Models.
Computes mAP@0.50, Per-Class AP, Precision, Recall, and Hard Negative Rejection.
Directly compares metrics against Model 9 Baseline Matrix.
"""
import os
import sys
import json
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.model_improvement.training.model_builder import build_research_model
from research.model_improvement.dataset.enhanced_dataset import EnhancedWeaponDataset

DEFAULT_CHECKPOINT = ROOT / "research" / "model_improvement" / "checkpoints" / "best_candidate_model_v3.pth"
BASELINE_MATRIX = ROOT / "research" / "model_improvement" / "evaluations" / "baseline_model9_matrix.json"
OUTPUT_METRICS = ROOT / "research" / "model_improvement" / "evaluations" / "candidate_model_v3_metrics.json"


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
    if total_gts == 0 or not class_preds:
        return 0.0, 0.0, 0.0, 0.0, 0, 0, total_gts

    class_preds.sort(key=lambda x: x["score"], reverse=True)
    tp_list, fp_list = [], []
    matched_gt = {img_idx: set() for img_idx in range(len(gt_by_image))}

    for pred in class_preds:
        img_idx = pred["img_idx"]
        pred_box = pred["box"]
        gts = [(idx, gt["box"]) for idx, gt in enumerate(gt_by_image[img_idx]) if gt["class_id"] == class_id]

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

    tp_count = sum(tp_list)
    fp_count = sum(fp_list)
    fn_count = total_gts - tp_count

    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    recall = tp_count / total_gts if total_gts > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # 11-point interpolation AP calculation
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)
    recalls = tp_cumsum / total_gts
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])

    return ap, precision, recall, f1, tp_count, fp_count, fn_count


def evaluate_model(checkpoint_path: Path = None, output_path: Path = None, conf_threshold: float = 0.50):
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT
    if output_path is None:
        output_path = OUTPUT_METRICS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    anchor_sizes = ckpt.get("anchor_sizes", ((12,), (20,), (36,), (64,), (96,)))
    aspect_ratios = ckpt.get("aspect_ratios") or ckpt.get("anchor_aspect_ratios", (0.5, 0.7, 1.0, 1.4, 2.0))

    model = build_research_model(
        num_classes=3,
        anchor_sizes=anchor_sizes,
        aspect_ratios=aspect_ratios,
        pretrained_backbone=False,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    val_dataset = EnhancedWeaponDataset("val", augment=False)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    print(f"Evaluating {len(val_dataset)} validation images on {device}...")
    predictions = []
    gt_by_image = []

    clean_negatives = 0
    total_negatives = 0

    with torch.no_grad():
        img_idx = 0
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()
                gt_records = [{"box": box, "class_id": label} for box, label in zip(gt_boxes, gt_labels)]
                gt_by_image.append(gt_records)

                is_negative = (len(gt_records) == 0)
                if is_negative:
                    total_negatives += 1

                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                has_fp = False
                for box, score, label in zip(boxes, scores, labels):
                    if score >= conf_threshold:
                        predictions.append({"img_idx": img_idx, "box": box, "score": score, "class_id": label})
                        if is_negative:
                            has_fp = True

                if is_negative and not has_fp:
                    clean_negatives += 1

                img_idx += 1

    # Compute metrics
    hg_ap, hg_pr, hg_rc, hg_f1, hg_tp, hg_fp, hg_fn = compute_ap_for_class(predictions, gt_by_image, 1)
    kn_ap, kn_pr, kn_rc, kn_f1, kn_tp, kn_fp, kn_fn = compute_ap_for_class(predictions, gt_by_image, 2)
    mAP = (hg_ap + kn_ap) / 2.0
    neg_rejection = (clean_negatives / total_negatives * 100.0) if total_negatives > 0 else 0.0

    print("\n" + "=" * 76)
    print("           CANDIDATE MODEL EVALUATION RESULTS vs. MODEL 9")
    print("=" * 76)
    print(f"Overall mAP@0.50:       {mAP*100:.2f}%")
    print(f"Handgun AP@0.50:        {hg_ap*100:.2f}% (Precision: {hg_pr*100:.2f}%, Recall: {hg_rc*100:.2f}%)")
    print(f"Knife AP@0.50:          {kn_ap*100:.2f}% (Precision: {kn_pr*100:.2f}%, Recall: {kn_rc*100:.2f}%)")
    print(f"Negative Rejection Rate: {neg_rejection:.2f}% ({clean_negatives}/{total_negatives} clean)")

    # Compare with Baseline
    if BASELINE_MATRIX.exists():
        base = json.loads(BASELINE_MATRIX.read_text())["test_split"]
        d_map = (mAP * 100) - base["mAP_50"]
        d_hg = (hg_ap * 100) - base["handgun"]["ap_50"]
        d_kn = (kn_ap * 100) - base["knife"]["ap_50"]
        print("-" * 76)
        print("COMPARISON DELTA TO MODEL 9 BASELINE:")
        print(f"  Delta mAP:     {d_map:+.2f}% (Base: {base['mAP_50']:.2f}% -> Candidate: {mAP*100:.2f}%)")
        print(f"  Delta Handgun: {d_hg:+.2f}% (Base: {base['handgun']['ap_50']:.2f}% -> Candidate: {hg_ap*100:.2f}%)")
        print(f"  Delta Knife:   {d_kn:+.2f}% (Base: {base['knife']['ap_50']:.2f}% -> Candidate: {kn_ap*100:.2f}%)")
        print("=" * 76 + "\n")

    results = {
        "checkpoint": str(checkpoint_path),
        "mAP_50": round(mAP * 100, 2),
        "handgun": {
            "ap_50": round(hg_ap * 100, 2), "precision": round(hg_pr * 100, 2),
            "recall": round(hg_rc * 100, 2), "f1": round(hg_f1 * 100, 2),
            "tp": hg_tp, "fp": hg_fp, "fn": hg_fn,
        },
        "knife": {
            "ap_50": round(kn_ap * 100, 2), "precision": round(kn_pr * 100, 2),
            "recall": round(kn_rc * 100, 2), "f1": round(kn_f1 * 100, 2),
            "tp": kn_tp, "fp": kn_fp, "fn": kn_fn,
        },
        "hard_negative_rejection_rate": round(neg_rejection, 2),
    }
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {output_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.50)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINT
    out = Path(args.output) if args.output else OUTPUT_METRICS
    evaluate_model(checkpoint_path=ckpt, output_path=out, conf_threshold=args.conf)
