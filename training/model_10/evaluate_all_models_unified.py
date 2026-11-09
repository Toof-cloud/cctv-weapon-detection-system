"""
Unified External Benchmark Evaluation Across Models 1 through 10 (Adviser Requirement).
Evaluates every trained checkpoint (Model 1 to Model 10) on the identical, strictly unseen
Unified External Benchmark Test Split (350 images: 134 Handgun, 116 Knife, 100 Hard Negatives
including 60 Held Non-Weapon Confounders), with 0% overlap with any Model 1-10 training/val split.
Also evaluates Model 9 vs Model 10 on the 279-image Model 9 Test Split and the 32 Held Non-Weapon Grid Cells.
"""
import os
import sys
import json
import math
import csv
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import cv2
import numpy as np
import torch
torch.set_num_threads(8)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_analysis.build_model import get_model
from training.model_10.tenth_model_dataset import TenthModelDataset
from training.model_10.train_tenth_model import evaluate_on_loader
from training.model_9.evaluate_ninth_model import collate_fn


MODELS_CONFIG = [
    ("Model 1", ROOT / "best_weapon_detector.pth", None, False, None),
    ("Model 2", ROOT / "best_weapon_detector_retrained.pth", None, False, None),
    ("Model 3", ROOT / "best_weapon_detector_third_model.pth", None, False, None),
    ("Model 4", ROOT / "best_weapon_detector_fourth_model.pth", None, False, None),
    ("Model 5", ROOT / "best_weapon_detector_fifth_model.pth", None, False, None),
    ("Model 6", ROOT / "best_weapon_detector_sixth_model.pth", None, False, None),
    ("Model 7", ROOT / "best_weapon_detector_seventh_model.pth", None, True, None),
    ("Model 8", ROOT / "best_weapon_detector_eighth_model.pth", (16, 32, 64, 128, 256), False, None),
    ("Model 9", ROOT / "best_weapon_detector_ninth_model.pth", (8, 16, 32, 64, 128), False, math.log(1.4)),
    ("Model 10", ROOT / "best_weapon_detector_tenth_model.pth", (16, 32, 64, 128, 256), False, math.log(2.2)),
]


def load_checkpoint_model(ckpt_path: Path, anchor_scales, small_anchors: bool, clip_val, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "anchor_scales" in ckpt:
        anchor_scales = ckpt["anchor_scales"]
    if isinstance(ckpt, dict) and "rpn_bbox_xform_clip" in ckpt:
        clip_val = ckpt["rpn_bbox_xform_clip"]

    model = get_model(
        num_classes=3,
        small_anchors=small_anchors,
        anchor_scales=anchor_scales,
        pretrained=False,
    )
    if clip_val is not None:
        model.rpn.box_coder.bbox_xform_clip = clip_val
        model.roi_heads.box_coder.bbox_xform_clip = clip_val

    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


def evaluate_no_weapon_grid_cells(model, device, conf_thresh=0.50):
    """Tests how many of the 32 CostiCatargiu no_weapon_grid1/2 cells trigger false detections."""
    # Note: Some cells in no_weapon_grid1/2 contain both a real weapon AND a no_weapon box,
    # whereas 21 of the 32 cells are pure held non-weapons (mugs, bottles, wallets, phones, umbrellas, brooms, selfie sticks).
    pure_noweapon_cells = [
        (1, 0, 0, "Man holding coffee mug"),
        (1, 0, 2, "Man holding glass beer bottle"),
        (1, 1, 1, "Couple holding sandwich & coffee cup"),
        (1, 1, 2, "Hand holding grey smartphone/card"),
        (1, 1, 3, "Family holding garden rakes"),
        (1, 2, 0, "Worker holding window squeegee"),
        (1, 2, 2, "Man on treadmill holding water bottle"),
        (1, 2, 3, "Hand holding clear water bottle"),
        (1, 3, 3, "Hand holding selfie stick & tripod"),
        (2, 0, 0, "Hand gripping umbrella handle"),
        (2, 0, 2, "Man sweeping street with broom"),
        (2, 1, 1, "Man holding selfie stick"),
        (2, 1, 3, "Gloved hand scrubbing floor with brush"),
        (2, 2, 1, "Woman holding leather clutch purse"),
        (2, 2, 3, "Man holding smartphone in both hands"),
        (2, 3, 0, "Woman holding floor broom"),
        (2, 3, 1, "Hands holding two metal water bottles"),
        (2, 3, 2, "Couple raking leaves in yard"),
    ]
    triggered = []
    for grid_idx, r, c, desc in pure_noweapon_cells:
        grid_name = f"no_weapon_grid{grid_idx}.png"
        img_bgr = cv2.imread(str(ROOT / "data" / "costi_yolov12" / "repo" / "DatasetExamples" / grid_name))
        H, W, _ = img_bgr.shape
        top_offset = int(H * 0.055)
        grid_h = H - top_offset
        cell_h, cell_w = grid_h // 4, W // 4
        y1 = top_offset + r * cell_h + 8
        y2 = top_offset + (r + 1) * cell_h - 8
        x1 = c * cell_w + 8
        x2 = (c + 1) * cell_w - 8
        cell_rgb = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(cell_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
        with torch.no_grad():
            out = model([tensor])[0]
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        high = [(int(l), float(s)) for l, s in zip(labels, scores) if s >= conf_thresh]
        if high:
            triggered.append({"grid": grid_idx, "row": r + 1, "col": c + 1, "desc": desc, "detections": high})

    return {
        "total_pure_noweapon_cells": len(pure_noweapon_cells),
        "hallucinated_cells": len(triggered),
        "clean_cells": len(pure_noweapon_cells) - len(triggered),
        "specificity": (len(pure_noweapon_cells) - len(triggered)) / len(pure_noweapon_cells),
        "details": triggered,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 100)
    print("   UNIFIED EXTERNAL BENCHMARK EVALUATION ACROSS MODELS 1 TO 10 (ZERO DATA LEAKAGE)   ")
    print("=" * 100)

    ext_bench_dataset = TenthModelDataset(
        split_name="test",
        augment=False,
        include_hard_negatives=False,
        data_root=ROOT / "training" / "unified_external_benchmark",
    )
    ext_loader = DataLoader(ext_bench_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)

    internal_unseen_dataset = TenthModelDataset(
        split_name="test",
        augment=False,
        include_hard_negatives=False,
        data_root=ROOT / "training" / "unified_external_benchmark",
        ids_file_override=ROOT / "training" / "unified_external_benchmark" / "strictly_unseen_internal_ids.txt",
    )
    internal_unseen_loader = DataLoader(internal_unseen_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)

    m9_test_dataset = TenthModelDataset(
        split_name="test",
        augment=False,
        include_hard_negatives=False,
        data_root=ROOT / "training" / "model_10" / "tenth_model_data",
    )
    m9_loader = DataLoader(m9_test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)

    unified_results = {}
    csv_rows = []

    print(f"\n[*] Evaluating Models 1 to 10 on Unified External Benchmark ({len(ext_bench_dataset)} images) & Unseen Internal CCTV ({len(internal_unseen_dataset)} images)...")
    print("-" * 130)
    print(f"{'Model':<10} | {'Ext350 P':<8} | {'Ext350 R':<8} | {'Ext350 F1':<9} | {'Ext350 mAP':<10} | {'Ext NegRej':<14} || {'Unseen185 F1':<12} | {'Unseen185 mAP':<13} | {'Unseen185 TP/FP/FN'}")
    print("-" * 130)

    for m_name, ckpt_path, scales, small_anc, clip_val in MODELS_CONFIG:
        if not ckpt_path.exists():
            print(f"{m_name:<10} | MISSING CHECKPOINT: {ckpt_path.name}")
            continue
        model = load_checkpoint_model(ckpt_path, scales, small_anc, clip_val, device)
        ext_res = evaluate_on_loader(model, ext_loader, device)
        int_res = evaluate_on_loader(model, internal_unseen_loader, device)

        m9_res = None
        grid_res = None
        if m_name in ("Model 8", "Model 9", "Model 10"):
            m9_res = evaluate_on_loader(model, m9_loader, device)
            grid_res = evaluate_no_weapon_grid_cells(model, device)

        unified_results[m_name] = {
            "checkpoint": ckpt_path.name,
            "external_benchmark_350": ext_res,
            "unseen_internal_cctv_185": int_res,
            "m9_benchmark_279": m9_res,
            "no_weapon_grid_audit": grid_res,
        }

        neg_str = f"{ext_res['neg_clean']}/{ext_res['neg_total']} ({ext_res['neg_rate']:.0%})"
        int_tpfpfn = f"{int_res['TP']}/{int_res['FP']}/{int_res['FN']} (Neg {int_res['neg_clean']}/{int_res['neg_total']})"
        print(
            f"{m_name:<10} | {ext_res['Precision']:.2%}  | {ext_res['Recall']:.2%}  | {ext_res['F1']:.2%}   | "
            f"{ext_res['mAP']:.2%}     | {neg_str:<14} || {int_res['F1']:.2%}       | {int_res['mAP']:.2%}        | {int_tpfpfn}",
            flush=True,
        )

        csv_rows.append({
            "Model": m_name,
            "Checkpoint": ckpt_path.name,
            "Ext350_Precision": round(ext_res["Precision"], 6),
            "Ext350_Recall": round(ext_res["Recall"], 6),
            "Ext350_F1": round(ext_res["F1"], 6),
            "Ext350_AP_Handgun": round(ext_res["AP_Handgun"], 6),
            "Ext350_AP_Knife": round(ext_res["AP_Knife"], 6),
            "Ext350_mAP_50": round(ext_res["mAP"], 6),
            "Ext350_TP": ext_res["TP"],
            "Ext350_FP": ext_res["FP"],
            "Ext350_FN": ext_res["FN"],
            "Ext350_Neg_Clean": ext_res["neg_clean"],
            "Ext350_Neg_Total": ext_res["neg_total"],
            "Ext350_Neg_Reject_Rate": round(ext_res["neg_rate"], 6),
            "Unseen185_Precision": round(int_res["Precision"], 6),
            "Unseen185_Recall": round(int_res["Recall"], 6),
            "Unseen185_F1": round(int_res["F1"], 6),
            "Unseen185_AP_Handgun": round(int_res["AP_Handgun"], 6),
            "Unseen185_AP_Knife": round(int_res["AP_Knife"], 6),
            "Unseen185_mAP_50": round(int_res["mAP"], 6),
            "Unseen185_TP": int_res["TP"],
            "Unseen185_FP": int_res["FP"],
            "Unseen185_FN": int_res["FN"],
            "Unseen185_Neg_Clean": int_res["neg_clean"],
            "Unseen185_Neg_Total": int_res["neg_total"],
            "Unseen185_Neg_Reject_Rate": round(int_res["neg_rate"], 6),
        })

    print("-" * 115)

    out_dir = ROOT / "training" / "unified_external_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "unified_models_1_to_10_results.json"
    csv_path = out_dir / "unified_models_1_to_10_results.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(unified_results, f, indent=2)

    if csv_rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"\n[*] Saved Unified External Benchmark JSON: {json_path}")
    print(f"[*] Saved Unified External Benchmark CSV:  {csv_path}")

    for m_name in ("Model 9", "Model 10"):
        if m_name in unified_results and unified_results[m_name]["m9_benchmark_279"]:
            r = unified_results[m_name]["m9_benchmark_279"]
            g = unified_results[m_name]["no_weapon_grid_audit"]
            print(f"\n=== {m_name} on 279-Image M9 Held-Out Test Split ===")
            print(f"  Precision: {r['Precision']:.2%} | Recall: {r['Recall']:.2%} | F1: {r['F1']:.2%} | mAP@0.50: {r['mAP']:.2%}")
            print(f"  AP(Handgun): {r['AP_Handgun']:.2%} | AP(Knife): {r['AP_Knife']:.2%}")
            print(f"  TP: {r['TP']} | FP: {r['FP']} | FN: {r['FN']} -> {r['per_class']}")
            print(f"  Negative Rejection: {r['neg_clean']}/{r['neg_total']} ({r['neg_rate']:.2%})")
            print(f"  Pure Held Non-Weapon Grid Cells (18 pure cells): {g['clean_cells']}/{g['total_pure_noweapon_cells']} clean ({g['specificity']:.2%}), Hallucinated={g['hallucinated_cells']}")


if __name__ == "__main__":
    main()

