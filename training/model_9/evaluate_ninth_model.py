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
from training.model_9.ninth_model_dataset import NinthModelDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = ROOT / "best_weapon_detector_ninth_model.pth"
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
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

    return float(ap)


def evaluate_model():
    print("=" * 76)
    print("        EVALUATING MODEL 9 ON INDEPENDENT TEST SET")
    print(f"Checkpoint:           {MODEL_PATH}")
    print(f"Device:               {DEVICE}")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"IoU Threshold:        {IOU_THRESHOLD}")
    print("=" * 76)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    test_dataset = NinthModelDataset("test", augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Total Test Images: {len(test_dataset)} ({len(test_loader)} batches)")

    model = get_model(num_classes=3, anchor_scales=(16, 32, 64, 128, 256))
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    all_predictions = []
    gt_by_image = []

    global_img_idx = 0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader, start=1):
            images_gpu = [img.to(DEVICE) for img in images]
            outputs = model(images_gpu)

            for target, output in zip(targets, outputs):
                gts = []
                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()
                for b, l in zip(gt_boxes, gt_labels):
                    gts.append({"box": b.tolist(), "class_id": int(l)})
                gt_by_image.append(gts)

                out_boxes = output["boxes"].cpu().numpy()
                out_labels = output["labels"].cpu().numpy()
                out_scores = output["scores"].cpu().numpy()

                for b, l, s in zip(out_boxes, out_labels, out_scores):
                    if s >= CONFIDENCE_THRESHOLD:
                        all_predictions.append({
                            "img_idx": global_img_idx,
                            "box": b.tolist(),
                            "class_id": int(l),
                            "score": float(s),
                        })

                global_img_idx += 1

    print("\nComputing Evaluation Metrics...")
    ap_handgun = compute_ap_for_class(all_predictions, gt_by_image, class_id=1, iou_threshold=IOU_THRESHOLD)
    ap_knife = compute_ap_for_class(all_predictions, gt_by_image, class_id=2, iou_threshold=IOU_THRESHOLD)
    mAP = (ap_handgun + ap_knife) / 2.0

    # Summary metrics per class
    classes = [("Handgun", 1), ("Knife", 2)]
    metrics = {}

    for name, cid in classes:
        c_preds = [p for p in all_predictions if p["class_id"] == cid]
        c_gts = sum(len([g for g in gts if g["class_id"] == cid]) for gts in gt_by_image)

        matched_gt = {i: set() for i in range(len(gt_by_image))}
        tp = 0
        fp = 0

        for p in sorted(c_preds, key=lambda x: x["score"], reverse=True):
            i_idx = p["img_idx"]
            p_box = p["box"]
            gts = [(g_idx, g["box"]) for g_idx, g in enumerate(gt_by_image[i_idx]) if g["class_id"] == cid]

            best_iou = 0.0
            best_g_idx = -1
            for g_idx, g_box in gts:
                iou = compute_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx

            if best_iou >= IOU_THRESHOLD and best_g_idx not in matched_gt[i_idx]:
                tp += 1
                matched_gt[i_idx].add(best_g_idx)
            else:
                fp += 1

        fn = c_gts - tp
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        metrics[name] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Total GT": c_gts,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "AP@0.5": ap_handgun if cid == 1 else ap_knife,
        }

    # Evaluate Negative Image Rejection Rate (Class 0 specificity)
    negative_images = [i for i, gts in enumerate(gt_by_image) if len(gts) == 0]
    neg_fps = 0
    for img_idx in negative_images:
        preds_in_neg = [p for p in all_predictions if p["img_idx"] == img_idx]
        if preds_in_neg:
            neg_fps += 1

    clean_neg_rate = ((len(negative_images) - neg_fps) / len(negative_images)) if negative_images else 1.0

    print("\n" + "=" * 76)
    print("                 MODEL 9 INDEPENDENT TEST SET RESULTS")
    print("=" * 76)
    print(f"{'Metric':<20} | {'Handgun (Class 1)':<24} | {'Knife (Class 2)':<24}")
    print("-" * 76)
    print(f"{'AP@0.50':<20} | {metrics['Handgun']['AP@0.5']:.2%}                   | {metrics['Knife']['AP@0.5']:.2%}")
    print(f"{'Precision':<20} | {metrics['Handgun']['Precision']:.2%}                   | {metrics['Knife']['Precision']:.2%}")
    print(f"{'Recall':<20} | {metrics['Handgun']['Recall']:.2%}                   | {metrics['Knife']['Recall']:.2%}")
    print(f"{'F1-Score':<20} | {metrics['Handgun']['F1']:.2%}                   | {metrics['Knife']['F1']:.2%}")
    print(f"{'True Positives (TP)':<20} | {metrics['Handgun']['TP']:<24} | {metrics['Knife']['TP']:<24}")
    print(f"{'False Positives (FP)':<20} | {metrics['Handgun']['FP']:<24} | {metrics['Knife']['FP']:<24}")
    print(f"{'False Negatives (FN)':<20} | {metrics['Handgun']['FN']:<24} | {metrics['Knife']['FN']:<24}")
    print(f"{'Total Ground Truth':<20} | {metrics['Handgun']['Total GT']:<24} | {metrics['Knife']['Total GT']:<24}")
    print("=" * 76)
    print(f"Overall Mean Average Precision (mAP@0.50): {mAP:.2%}")
    print(f"Hard Negative Scene Rejection Rate:       {clean_neg_rate:.2%} ({len(negative_images) - neg_fps}/{len(negative_images)} clean)")
    print("=" * 76)

    return {
        "mAP": mAP,
        "metrics": metrics,
        "clean_neg_rate": clean_neg_rate,
    }


if __name__ == "__main__":
    evaluate_model()
