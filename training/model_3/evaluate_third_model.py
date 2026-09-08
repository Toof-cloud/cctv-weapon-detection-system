import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_analysis.build_model import get_model
from training.model_3.third_model_dataset import ThirdModelDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = ROOT / "best_weapon_detector_third_model.pth"
OUTPUT_PATH = ROOT / "third_run" / "THIRD_MODEL_METRICS.txt"
SCORE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50


def collate_fn(batch):
    return tuple(zip(*batch))


def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_predictions(predictions, ground_truths):
    matched = set()
    true_positive = 0
    false_positive = 0

    for prediction in predictions:
        best_index = None
        best_iou = 0.0
        for index, ground_truth in enumerate(ground_truths):
            if index in matched or prediction["class_id"] != ground_truth["class_id"]:
                continue
            iou = compute_iou(prediction["box"], ground_truth["box"])
            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index is not None and best_iou >= IOU_THRESHOLD:
            matched.add(best_index)
            true_positive += 1
        else:
            false_positive += 1

    false_negative = len(ground_truths) - len(matched)
    return true_positive, false_positive, false_negative


def collect_predictions(model, data_loader):
    predictions_by_image = []
    ground_truths_by_image = []

    with torch.inference_mode():
        for images, targets in data_loader:
            outputs = model([image.to(DEVICE) for image in images])
            for output, target in zip(outputs, targets):
                predictions = []
                for box, label, score in zip(
                    output["boxes"].cpu().tolist(),
                    output["labels"].cpu().tolist(),
                    output["scores"].cpu().tolist(),
                ):
                    if score >= SCORE_THRESHOLD and int(label) in (1, 2):
                        predictions.append({
                            "box": box,
                            "class_id": int(label),
                            "score": float(score),
                        })

                ground_truths = [
                    {"box": box, "class_id": int(label)}
                    for box, label in zip(
                        target["boxes"].tolist(),
                        target["labels"].tolist(),
                    )
                    if int(label) in (1, 2)
                ]
                predictions_by_image.append(predictions)
                ground_truths_by_image.append(ground_truths)

    return predictions_by_image, ground_truths_by_image


def compute_ap(predictions_by_image, ground_truths_by_image, class_id):
    predictions = []
    total_ground_truths = 0

    for image_index, (image_predictions, image_ground_truths) in enumerate(
        zip(predictions_by_image, ground_truths_by_image)
    ):
        total_ground_truths += sum(
            ground_truth["class_id"] == class_id
            for ground_truth in image_ground_truths
        )
        for prediction in image_predictions:
            if prediction["class_id"] == class_id:
                predictions.append({
                    **prediction,
                    "image_index": image_index,
                })

    if total_ground_truths == 0:
        return 0.0

    predictions.sort(key=lambda prediction: prediction["score"], reverse=True)
    matched = set()
    true_positive = []
    false_positive = []

    for prediction in predictions:
        image_index = prediction["image_index"]
        image_ground_truths = ground_truths_by_image[image_index]
        best_index = None
        best_iou = 0.0

        for ground_truth_index, ground_truth in enumerate(image_ground_truths):
            if (image_index, ground_truth_index) in matched:
                continue
            if ground_truth["class_id"] != class_id:
                continue
            iou = compute_iou(prediction["box"], ground_truth["box"])
            if iou > best_iou:
                best_iou = iou
                best_index = ground_truth_index

        if best_index is not None and best_iou >= IOU_THRESHOLD:
            matched.add((image_index, best_index))
            true_positive.append(1)
            false_positive.append(0)
        else:
            true_positive.append(0)
            false_positive.append(1)

    cumulative_tp = 0
    cumulative_fp = 0
    previous_recall = 0.0
    average_precision = 0.0

    for tp, fp in zip(true_positive, false_positive):
        cumulative_tp += tp
        cumulative_fp += fp
        recall = cumulative_tp / total_ground_truths
        precision = cumulative_tp / (cumulative_tp + cumulative_fp)
        average_precision += max(0.0, recall - previous_recall) * precision
        previous_recall = recall

    return average_precision


def evaluate_overall(predictions_by_image, ground_truths_by_image):
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for predictions, ground_truths in zip(
        predictions_by_image,
        ground_truths_by_image,
    ):
        tp, fp, fn = match_predictions(predictions, ground_truths)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, total_tp, total_fp, total_fn


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")

    test_dataset = ThirdModelDataset("test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
    )
    model = get_model(num_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    predictions, ground_truths = collect_predictions(model, test_loader)
    precision, recall, f1, tp, fp, fn = evaluate_overall(
        predictions,
        ground_truths,
    )
    ap_handgun = compute_ap(predictions, ground_truths, class_id=1)
    ap_knife = compute_ap(predictions, ground_truths, class_id=2)
    map_05 = (ap_handgun + ap_knife) / 2.0
    lines = [
        "Third Model Evaluation",
        "=======================",
        f"Checkpoint: {MODEL_PATH.name}",
        f"Evaluation split: test ({len(test_dataset)} images)",
        f"Confidence threshold: {SCORE_THRESHOLD}",
        f"IoU threshold: {IOU_THRESHOLD}",
        f"Precision: {precision:.6f}",
        f"Recall: {recall:.6f}",
        f"F1: {f1:.6f}",
        f"AP(Handgun): {ap_handgun:.6f}",
        f"AP(Knife): {ap_knife:.6f}",
        f"mAP@0.5: {map_05:.6f}",
        f"TP: {tp}",
        f"FP: {fp}",
        f"FN: {fn}",
    ]
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Metrics saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()