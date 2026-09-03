import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset_analysis"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DATASET_DIR))

import torch
from dataset_analysis.create_dataloaders import val_loader
from dataset_analysis.build_model import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = ROOT / "best_weapon_detector.pth"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

def compute_iou(box_a, box_b):
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

def compute_ap_for_class(model, data_loader, class_id, score_threshold=0.5, iou_threshold=0.5):
    predictions = []
    gt_by_image = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            for img_idx, output in enumerate(outputs):
                gt_boxes = targets[img_idx]["boxes"].cpu().tolist()
                gt_labels = targets[img_idx]["labels"].cpu().tolist()

                gt_objs = []
                for box, label in zip(gt_boxes, gt_labels):
                    if int(label) == class_id:
                        gt_objs.append({
                            "box": [float(v) for v in box],
                            "class_id": int(label),
                        })

                gt_by_image.append(gt_objs)

                pred_boxes = output["boxes"].cpu().tolist()
                pred_labels = output["labels"].cpu().tolist()
                pred_scores = output["scores"].cpu().tolist()

                for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                    if score < score_threshold:
                        continue
                    if int(label) != class_id:
                        continue
                    predictions.append({
                        "image_idx": batch_idx * 2 + img_idx,
                        "box": [float(v) for v in box],
                        "class_id": int(label),
                        "score": float(score),
                    })

    total_gt = sum(len(gts) for gts in gt_by_image)
    if total_gt == 0:
        return 0.0, 0, 0

    predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)
    tp_list = []
    fp_list = []
    matched = set()

    for pred in predictions:
        img_idx = pred["image_idx"]
        gt_list = gt_by_image[img_idx]

        best_iou = -1.0
        best_gt_idx = None

        for gt_idx, gt in enumerate(gt_list):
            key = (img_idx, gt_idx)
            if key in matched:
                continue
            if gt["class_id"] != class_id:
                continue
            iou = compute_iou(pred["box"], gt["box"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is not None and best_iou >= iou_threshold:
            matched.add((img_idx, best_gt_idx))
            tp_list.append(1)
            fp_list.append(0)
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cum = []
    fp_cum = []
    running_tp = 0
    running_fp = 0

    for tp, fp in zip(tp_list, fp_list):
        running_tp += tp
        running_fp += fp
        tp_cum.append(running_tp)
        fp_cum.append(running_fp)

    recall = [t / total_gt for t in tp_cum]
    precision = [t / (t + f) if (t + f) > 0 else 0.0 for t, f in zip(tp_cum, fp_cum)]

    ap = 0.0
    prev_recall = 0.0
    for r, p in zip(recall, precision):
        ap += max(0.0, r - prev_recall) * p
        prev_recall = r

    return ap, total_gt, len(predictions)

def evaluate_overall(model, data_loader, score_threshold=0.5, iou_threshold=0.5):
    total_tp = 0
    total_fp = 0
    total_fn = 0

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
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    overall = evaluate_overall(model, val_loader, score_threshold=CONFIDENCE_THRESHOLD, iou_threshold=IOU_THRESHOLD)

    ap_handgun, _, _ = compute_ap_for_class(model, val_loader, class_id=1, score_threshold=CONFIDENCE_THRESHOLD, iou_threshold=IOU_THRESHOLD)
    ap_knife, _, _ = compute_ap_for_class(model, val_loader, class_id=2, score_threshold=CONFIDENCE_THRESHOLD, iou_threshold=IOU_THRESHOLD)
    mAP_05 = (ap_handgun + ap_knife) / 2.0

    print("Precision:", round(overall["precision"], 6))
    print("Recall:", round(overall["recall"], 6))
    print("F1:", round(overall["f1"], 6))
    print("AP(Handgun):", round(ap_handgun, 6))
    print("AP(Knife):", round(ap_knife, 6))
    print("mAP@0.5:", round(mAP_05, 6))
    print("FP:", overall["fp"])
    print("FN:", overall["fn"])

if __name__ == "__main__":
    main()