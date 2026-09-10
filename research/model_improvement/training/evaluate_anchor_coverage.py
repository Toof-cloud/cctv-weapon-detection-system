import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = ROOT / "training" / "model_9" / "ninth_model_data" / "annotations.csv"

def compute_box_anchor_iou(w, h, anchor_w, anchor_h):
    # Centered intersection over union
    inter_w = np.minimum(w, anchor_w)
    inter_h = np.minimum(h, anchor_h)
    inter_area = inter_w * inter_h
    union_area = (w * h) + (anchor_w * anchor_h) - inter_area
    return inter_area / np.maximum(union_area, 1e-6)

def evaluate_anchors():
    df = pd.read_csv(CSV_PATH)
    pos = df[(df["xmax"] > df["xmin"]) & (df["ymax"] > df["ymin"])].copy()
    pos["w"] = pos["xmax"] - pos["xmin"]
    pos["h"] = pos["ymax"] - pos["ymin"]

    # Config 1: Model 9 Default (scales: 16, 32, 64, 128, 256; ratios: 0.5, 1.0, 2.0)
    scales_default = [16, 32, 64, 128, 256]
    ratios_default = [0.5, 1.0, 2.0]
    anchors_default = []
    for s in scales_default:
        for r in ratios_default:
            # area = s^2, w/h = r => w = s * sqrt(r), h = s / sqrt(r)
            anchors_default.append((s * np.sqrt(r), s / np.sqrt(r)))

    # Config 2: Proposed Elongated + Small Anchors (scales: 12, 24, 48, 96, 192, 384; ratios: 0.25, 0.5, 1.0, 2.0, 4.0)
    scales_proposed = [12, 24, 48, 96, 192, 384]
    ratios_proposed = [0.25, 0.5, 1.0, 2.0, 4.0]
    anchors_proposed = []
    for s in scales_proposed:
        for r in ratios_proposed:
            anchors_proposed.append((s * np.sqrt(r), s / np.sqrt(r)))

    # Config 3: Balanced 5-ratio (scales: 16, 32, 64, 128, 256; ratios: 0.33, 0.5, 1.0, 2.0, 3.0)
    scales_c3 = [16, 32, 64, 128, 256]
    ratios_c3 = [0.33, 0.5, 1.0, 2.0, 3.0]
    anchors_c3 = []
    for s in scales_c3:
        for r in ratios_c3:
            anchors_c3.append((s * np.sqrt(r), s / np.sqrt(r)))

    for label, cid in [("HANDGUN", 1), ("KNIFE", 2), ("ALL WEAPONS", None)]:
        sub = pos if cid is None else pos[pos["class_id"] == cid]
        ws = sub["w"].values[:, None]
        hs = sub["h"].values[:, None]

        # Calculate max IoU for each box with anchor set
        def get_max_ious(anchors):
            aw = np.array([a[0] for a in anchors])[None, :]
            ah = np.array([a[1] for a in anchors])[None, :]
            ious = compute_box_anchor_iou(ws, hs, aw, ah)
            return ious.max(axis=1)

        iou_def = get_max_ious(anchors_default)
        iou_c3 = get_max_ious(anchors_c3)
        iou_prop = get_max_ious(anchors_proposed)

        print(f"\n=======================================================")
        print(f"ANCHOR COVERAGE EVALUATION: {label} (N = {len(sub)})")
        print(f"=======================================================")
        print(f"Default Model 9 Anchors (3 ratios): Mean Max-IoU = {iou_def.mean():.4f}, Recall@0.50 IoU = {(iou_def >= 0.5).mean()*100:.2f}%")
        print(f"Config 3 Balanced (5 ratios, 0.33-3.0): Mean Max-IoU = {iou_c3.mean():.4f}, Recall@0.50 IoU = {(iou_c3 >= 0.5).mean()*100:.2f}%")
        print(f"Proposed Multi-scale (5 ratios, 0.25-4.0, scale 12): Mean Max-IoU = {iou_prop.mean():.4f}, Recall@0.50 IoU = {(iou_prop >= 0.5).mean()*100:.2f}%")

if __name__ == "__main__":
    evaluate_anchors()
