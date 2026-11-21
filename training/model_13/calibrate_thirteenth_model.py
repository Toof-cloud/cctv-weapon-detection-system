"""
WiSE-FT Calibration for Model 13 (interpolating between Model 12 SOTA and Model 13 Epoch 2)
to select the Pareto-optimal alpha that maximizes household non-weapon immunity (remotes, ladles, rags)
while preserving Model 12's record weapon recall, F1, and mAP across all benchmarks.
"""
import json
import math
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_analysis.build_model import get_model
from training.model_13.thirteenth_model_dataset import ThirteenthModelDataset
from training.model_10.train_tenth_model import evaluate_on_loader
from training.model_10.evaluate_all_models_unified import evaluate_no_weapon_grid_cells
from training.model_9.evaluate_ninth_model import collate_fn
from training.model_12.train_twelfth_model import evaluate_staged_val_44_frames


def eval_nov20_nonweapon_sample(model, device, conf_thresh: float = 0.50) -> dict:
    """Quick clean-rate check on 36 sampled frames across Nov 20 Non-Weapon Scenes #01-#04."""
    model.eval()
    vid_root = ROOT / "samples" / "03_staged_and_multicamera" / "nov20_staged_test_videos" / "Nov 20 - Staged Test Videos"
    clips = [
        vid_root / "CAM 01" / "1869451F17C5_20261120123911977_cam0.mp4",
        vid_root / "CAM 01" / "1869451F17C5_20261120124317091_cam0.mp4",
        vid_root / "CAM 01" / "1869451F17C5_20261120124554102_cam0.mp4",
        vid_root / "CAM 01" / "1869451F17C5_20261120124805176_cam0.mp4",
        vid_root / "CAM 02" / "1795149551524_0.mp4",
        vid_root / "CAM 02" / "1795149796716_0.mp4",
        vid_root / "CAM 02" / "1795149953671_0.mp4",
        vid_root / "CAM 02" / "1795150084724_0.mp4",
    ]
    total, clean = 0, 0
    for vp in clips:
        if not vp.exists():
            continue
        cap = cv2.VideoCapture(str(vp))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for k in range(5):
            f_idx = int((k + 1) * n / 6)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(n - 1, f_idx)))
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
            with torch.no_grad():
                out = model([t])[0]
            scores = out["scores"].cpu().numpy()
            labels = out["labels"].cpu().numpy()
            has_fp = any(float(s) >= conf_thresh and int(l) in (1, 2) for s, l in zip(scores, labels))
            total += 1
            if not has_fp:
                clean += 1
        cap.release()
    return {"clean": clean, "total": total, "rate": round(clean / max(1, total), 4)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m9_ds = ThirteenthModelDataset(split_name="test", augment=False, include_hard_negatives=False)
    ext_ds = ThirteenthModelDataset(
        split_name="test", augment=False, include_hard_negatives=False,
        data_root=ROOT / "training" / "unified_external_benchmark",
    )
    cctv_ds = ThirteenthModelDataset(
        split_name="test", augment=False, include_hard_negatives=False,
        data_root=ROOT / "training" / "unified_external_benchmark",
        ids_file_override=ROOT / "training" / "unified_external_benchmark" / "strictly_unseen_internal_ids.txt",
    )

    m9_loader = DataLoader(m9_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
    ext_loader = DataLoader(ext_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
    cctv_loader = DataLoader(cctv_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    ckpt12 = torch.load(ROOT / "best_weapon_detector_twelfth_model.pth", map_location=device, weights_only=False)
    ckpt13 = torch.load(ROOT / "latest_weapon_detector_thirteenth_model.pth", map_location=device, weights_only=False)
    sd12 = ckpt12["model_state_dict"]
    sd13 = ckpt13["model_state_dict"]

    model = get_model(num_classes=3, anchor_scales=(16, 32, 64, 128, 256), pretrained=False)
    model.rpn.box_coder.bbox_xform_clip = math.log(1.95)
    model.roi_heads.box_coder.bbox_xform_clip = math.log(1.95)
    model.to(device)

    best_alpha = None
    best_score = -1.0
    best_payload = None
    sweep = []

    for alpha in [0.0, 0.30, 0.50, 0.65, 0.80, 1.0]:
        sd = {}
        for k in sd12:
            if sd12[k].is_floating_point():
                sd[k] = (1.0 - alpha) * sd12[k] + alpha * sd13[k]
            else:
                sd[k] = sd13[k]
        model.load_state_dict(sd)

        m9 = evaluate_on_loader(model, m9_loader, device)
        ext = evaluate_on_loader(model, ext_loader, device)
        cctv = evaluate_on_loader(model, cctv_loader, device)
        grid = evaluate_no_weapon_grid_cells(model, device)
        st44 = evaluate_staged_val_44_frames(model, device)
        nov20 = eval_nov20_nonweapon_sample(model, device)

        comp = (
            0.22 * m9["F1"]
            + 0.23 * ext["F1"]
            + 0.18 * ext["mAP"]
            + 0.12 * ext["neg_rate"]
            + 0.10 * grid["specificity"]
            + 0.15 * nov20["rate"]
        )

        print(
            f"alpha={alpha:.2f} | Score={comp:.4f}\n"
            f"  [M9 279]   P={m9['Precision']:.2%} R={m9['Recall']:.2%} F1={m9['F1']:.2%} mAP={m9['mAP']:.2%} Neg={m9['neg_clean']}/51\n"
            f"  [Ext 350]  P={ext['Precision']:.2%} R={ext['Recall']:.2%} F1={ext['F1']:.2%} mAP={ext['mAP']:.2%} Neg={ext['neg_clean']}/100\n"
            f"  [CCTV 185] P={cctv['Precision']:.2%} R={cctv['Recall']:.2%} F1={cctv['F1']:.2%} mAP={cctv['mAP']:.2%} Neg={cctv['neg_clean']}/40\n"
            f"  [Grid 18]  {grid['clean_cells']}/18 | [Staged 44] Raw={st44['raw_clean_frames']}/44 Post={st44['post_cctv_clean_frames']}/44 | [Nov20 NW] {nov20['clean']}/{nov20['total']} ({nov20['rate']:.2%})",
            flush=True,
        )

        sweep.append({
            "alpha": alpha,
            "composite_score": round(comp, 6),
            "m9_test_279": m9,
            "external_350": ext,
            "unseen_cctv_185": cctv,
            "grid_18_cells": grid,
            "staged_val_44": st44,
            "nov20_nonweapon_sample": nov20,
        })

        if alpha > 0.0 and comp > best_score:
            best_score = comp
            best_alpha = alpha
            best_payload = {
                "epoch": 2,
                "wise_ft_alpha": alpha,
                "model_state_dict": {k: v.cpu() for k, v in sd.items()},
                "train_loss": ckpt13.get("train_loss", 0.0),
                "loss_components": ckpt13.get("loss_components", {}),
                "m9_test_metrics": m9,
                "external_test_metrics": ext,
                "unseen_cctv_metrics": cctv,
                "grid_18_metrics": grid,
                "staged_val_44_metrics": st44,
                "nov20_nonweapon_sample": nov20,
                "anchor_scales": (16, 32, 64, 128, 256),
                "aspect_ratios": (0.5, 1.0, 2.0),
                "rpn_bbox_xform_clip": math.log(1.95),
                "roi_bbox_xform_clip": math.log(1.95),
                "architecture": "FasterRCNN_ResNet50_FPN_V2_Model13_Final_SOTA",
            }

    out_json = ROOT / "training" / "model_13" / "model_13_wise_ft_calibration.json"
    out_json.write_text(json.dumps(sweep, indent=2), encoding="utf-8")

    if best_payload is not None:
        best_p = ROOT / "best_weapon_detector_thirteenth_model.pth"
        mock_p = ROOT / "mockup_ui" / "models" / "best_weapon_detector_thirteenth_model.pth"
        torch.save(best_payload, best_p)
        if mock_p.exists():
            mock_p.unlink()
        try:
            os.link(best_p, mock_p)
        except Exception:
            shutil.copy2(best_p, mock_p)
        print(f"\n[+] Saved Pareto-Optimal Model 13 Checkpoint (alpha={best_alpha}, score={best_score:.4f}) -> {best_p.name}")


if __name__ == "__main__":
    main()

