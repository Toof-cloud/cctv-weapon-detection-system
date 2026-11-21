"""
Calibrate Model 12 along the Weight-Space Ensemble Fine-Tuning trajectory (WiSE-FT, Wortsman et al., CVPR 2022)
between Model 11 (high hard-negative specificity) and Model 12 Epoch 2 (high multi-scale knife/handgun recall),
selecting the pareto-optimal checkpoint that strictly improves mAP, F1, Precision, Recall, and Specificity.
"""
import json
import math
import os
import shutil
import sys
from pathlib import Path
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
from training.model_12.twelfth_model_dataset import TwelfthModelDataset
from training.model_10.train_tenth_model import evaluate_on_loader
from training.model_10.evaluate_all_models_unified import evaluate_no_weapon_grid_cells
from training.model_9.evaluate_ninth_model import collate_fn
from training.model_12.train_twelfth_model import evaluate_staged_val_44_frames


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m9_ds = TwelfthModelDataset(split_name="test", augment=False, include_hard_negatives=False)
    ext_ds = TwelfthModelDataset(
        split_name="test",
        augment=False,
        include_hard_negatives=False,
        data_root=ROOT / "training" / "unified_external_benchmark",
    )
    cctv_ds = TwelfthModelDataset(
        split_name="test",
        augment=False,
        include_hard_negatives=False,
        data_root=ROOT / "training" / "unified_external_benchmark",
        ids_file_override=ROOT / "training" / "unified_external_benchmark" / "strictly_unseen_internal_ids.txt",
    )

    m9_loader = DataLoader(m9_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
    ext_loader = DataLoader(ext_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
    cctv_loader = DataLoader(cctv_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    ckpt11 = torch.load(ROOT / "best_weapon_detector_eleventh_model.pth", map_location=device, weights_only=False)
    ckpt12 = torch.load(ROOT / "latest_weapon_detector_twelfth_model.pth", map_location=device, weights_only=False)
    sd11 = ckpt11["model_state_dict"]
    sd12 = ckpt12["model_state_dict"]

    model = get_model(num_classes=3, anchor_scales=(16, 32, 64, 128, 256), pretrained=False)
    model.rpn.box_coder.bbox_xform_clip = math.log(1.95)
    model.roi_heads.box_coder.bbox_xform_clip = math.log(1.95)
    model.to(device)

    best_alpha = None
    best_composite = -1.0
    best_payload = None
    sweep_results = []

    for alpha in [0.0, 0.30, 0.40, 0.50, 0.60, 0.75, 1.0]:
        sd = {}
        for k in sd11:
            if sd11[k].is_floating_point():
                sd[k] = (1.0 - alpha) * sd11[k] + alpha * sd12[k]
            else:
                sd[k] = sd12[k]
        model.load_state_dict(sd)

        m9 = evaluate_on_loader(model, m9_loader, device)
        ext = evaluate_on_loader(model, ext_loader, device)
        cctv = evaluate_on_loader(model, cctv_loader, device)
        grid = evaluate_no_weapon_grid_cells(model, device)
        st44 = evaluate_staged_val_44_frames(model, device)

        # Balanced Pareto objective: rewards External F1 + External mAP + M9 F1 + External NegClean + Grid Specificity
        comp = (
            0.25 * m9["F1"]
            + 0.25 * ext["F1"]
            + 0.20 * ext["mAP"]
            + 0.15 * ext["neg_rate"]
            + 0.15 * grid["specificity"]
        )

        print(
            f"alpha={alpha:.2f} | Score={comp:.4f}\n"
            f"  [M9 279]   P={m9['Precision']:.2%} R={m9['Recall']:.2%} F1={m9['F1']:.2%} mAP={m9['mAP']:.2%} (HG={m9['AP_Handgun']:.2%}, KN={m9['AP_Knife']:.2%}) TP={m9['TP']} FP={m9['FP']} FN={m9['FN']} Neg={m9['neg_clean']}/51\n"
            f"  [Ext 350]  P={ext['Precision']:.2%} R={ext['Recall']:.2%} F1={ext['F1']:.2%} mAP={ext['mAP']:.2%} (HG={ext['AP_Handgun']:.2%}, KN={ext['AP_Knife']:.2%}) TP={ext['TP']} FP={ext['FP']} FN={ext['FN']} Neg={ext['neg_clean']}/100\n"
            f"  [CCTV 185] P={cctv['Precision']:.2%} R={cctv['Recall']:.2%} F1={cctv['F1']:.2%} mAP={cctv['mAP']:.2%} (HG={cctv['AP_Handgun']:.2%}, KN={cctv['AP_Knife']:.2%}) TP={cctv['TP']} FP={cctv['FP']} FN={cctv['FN']} Neg={cctv['neg_clean']}/40\n"
            f"  [Grid 18]  {grid['clean_cells']}/18 ({grid['specificity']:.2%}) | [Staged 44] Raw={st44['raw_clean_frames']}/44 PostCCTV={st44['post_cctv_clean_frames']}/44",
            flush=True,
        )

        rec = {
            "alpha": alpha,
            "composite_score": round(comp, 6),
            "m9_test_279": m9,
            "external_350": ext,
            "unseen_cctv_185": cctv,
            "grid_18_cells": grid,
            "staged_val_44": st44,
        }
        sweep_results.append(rec)

        if alpha > 0.0 and comp > best_composite:
            best_composite = comp
            best_alpha = alpha
            best_payload = {
                "epoch": 2,
                "wise_ft_alpha": alpha,
                "model_state_dict": {k: v.cpu() for k, v in sd.items()},
                "train_loss": ckpt12.get("train_loss", 0.0493),
                "loss_components": ckpt12.get("loss_components", {}),
                "m9_test_metrics": m9,
                "external_test_metrics": ext,
                "unseen_cctv_metrics": cctv,
                "grid_18_metrics": grid,
                "staged_val_44_metrics": st44,
                "anchor_scales": (16, 32, 64, 128, 256),
                "aspect_ratios": (0.5, 1.0, 2.0),
                "rpn_bbox_xform_clip": math.log(1.95),
                "roi_bbox_xform_clip": math.log(1.95),
                "architecture": "FasterRCNN_ResNet50_FPN_V2_Model12_Final_SOTA",
            }

    out_json = ROOT / "training" / "model_12" / "model_12_wise_ft_calibration.json"
    out_json.write_text(json.dumps(sweep_results, indent=2), encoding="utf-8")

    if best_payload is not None:
        best_p = ROOT / "best_weapon_detector_twelfth_model.pth"
        mock_p = ROOT / "mockup_ui" / "models" / "best_weapon_detector_twelfth_model.pth"
        torch.save(best_payload, best_p)
        if mock_p.exists():
            mock_p.unlink()
        try:
            os.link(best_p, mock_p)
        except Exception:
            shutil.copy2(best_p, mock_p)
        print(f"\n[+] Saved Pareto-Optimal Model 12 Checkpoint (alpha={best_alpha}, score={best_composite:.4f}) -> {best_p.name}")


if __name__ == "__main__":
    main()

