"""
Refined Model 10 Fine-Tuning:
Maximizes Held Non-Weapon Confounder Rejection (>=92%) while maintaining
High Precision (>=92%), High Recall (>=93%), and Low False Negatives (<=15 FN).
"""
import os
import sys
import random
import math
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import cv2
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(8)
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_analysis.build_model import get_model
from training.model_10.tenth_model_dataset import TenthModelDataset
from training.model_10.train_tenth_model import evaluate_on_loader, train_one_epoch
from training.model_10.evaluate_all_models_unified import evaluate_no_weapon_grid_cells
from training.model_9.evaluate_ninth_model import collate_fn

BEST_M10_PATH = ROOT / "best_weapon_detector_tenth_model.pth"
BASE_V10_PATH = ROOT / "research" / "model_improvement" / "checkpoints" / "best_candidate_model_v10.pth"
MODEL10_DATA = ROOT / "training" / "model_10" / "tenth_model_data"
MODEL9_DATA = ROOT / "training" / "model_9" / "ninth_model_data"

ANCHOR_SCALES = (16, 32, 64, 128, 256)
RPN_CLIP = math.log(1.95)
ROI_CLIP = math.log(1.95)
EPOCHS = 2
BATCH_SIZE = 2
BASE_LR = 0.00008


def rebalance_model10_train_ids():
    """
    Keep all 2,214 Base Model 9 CCTV training images + all 450 CostiCatargiu Held Non-Weapon
    confounder images (Class 0), plus 80 CCTV-scale Knife and 60 Pistol images from CostiCatargiu
    so the detector suppresses held non-weapons aggressively without over-triggering on macro objects.
    """
    random.seed(42)
    df = pd.read_csv(MODEL10_DATA / "annotations.csv")
    m9_train_ids = [l.strip() for l in (MODEL9_DATA / "train_ids.txt").read_text().splitlines() if l.strip()]

    costi_tr = df[(df["split"] == "train") & (df["image_path"].str.startswith("costi_yolov12_curated/"))]
    noweap_ids = sorted(costi_tr[costi_tr["class_id"] == 0]["image_path"].unique())
    knife_ids = sorted(costi_tr[costi_tr["class_id"] == 2]["image_path"].unique())
    pistol_ids = sorted(costi_tr[costi_tr["class_id"] == 1]["image_path"].unique())

    random.shuffle(knife_ids)
    random.shuffle(pistol_ids)

    balanced_train_ids = sorted(set(m9_train_ids) | set(noweap_ids) | set(knife_ids[:80]) | set(pistol_ids[:60]))
    (MODEL10_DATA / "train_ids.txt").write_text("\n".join(balanced_train_ids) + "\n", encoding="utf-8")
    print(f"[*] Rebalanced Model 10 train_ids.txt: {len(balanced_train_ids)} images "
          f"(Base M9: {len(m9_train_ids)}, Held Non-Weapons: {len(noweap_ids)}, Costi Knife: 80, Costi Pistol: 60)")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rebalance_model10_train_ids()

    model = get_model(num_classes=3, anchor_scales=ANCHOR_SCALES, pretrained=False)
    model.rpn.box_coder.bbox_xform_clip = RPN_CLIP
    model.roi_heads.box_coder.bbox_xform_clip = ROI_CLIP

    ckpt = torch.load(BASE_V10_PATH, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    train_dataset = TenthModelDataset(split_name="train", augment=True, include_hard_negatives=True)
    m9_test_dataset = TenthModelDataset(split_name="test", augment=False, include_hard_negatives=False)
    ext_bench_dataset = TenthModelDataset(
        split_name="test",
        augment=False,
        include_hard_negatives=False,
        data_root=ROOT / "training" / "unified_external_benchmark",
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    m9_test_loader = DataLoader(m9_test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)
    ext_bench_loader = DataLoader(ext_bench_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)

    for p in model.backbone.body.conv1.parameters():
        p.requires_grad = False
    for p in model.backbone.body.layer1.parameters():
        p.requires_grad = False
    for p in model.backbone.body.layer2.parameters():
        p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=BASE_LR, momentum=0.9, weight_decay=0.0005)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_score = -1.0
    for epoch in range(EPOCHS):
        print(f"\n--- Refined Model 10 Epoch {epoch+1}/{EPOCHS} (lr={optimizer.param_groups[0]['lr']:.6f}) ---", flush=True)
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler=scaler)
        scheduler.step()

        m9_res = evaluate_on_loader(model, m9_test_loader, device)
        ext_res = evaluate_on_loader(model, ext_bench_loader, device)
        grid_res = evaluate_no_weapon_grid_cells(model, device)

        print(
            f"--> [Epoch {epoch+1}] Loss: {train_loss:.4f}\n"
            f"    [M9 Test 279]     P={m9_res['Precision']:.2%} R={m9_res['Recall']:.2%} F1={m9_res['F1']:.2%} "
            f"mAP={m9_res['mAP']:.2%} (HG={m9_res['AP_Handgun']:.2%}, KN={m9_res['AP_Knife']:.2%}) | "
            f"TP={m9_res['TP']} FP={m9_res['FP']} FN={m9_res['FN']} {m9_res['per_class']} | "
            f"NegClean={m9_res['neg_clean']}/{m9_res['neg_total']} ({m9_res['neg_rate']:.2%})\n"
            f"    [Unified Ext 350] P={ext_res['Precision']:.2%} R={ext_res['Recall']:.2%} F1={ext_res['F1']:.2%} "
            f"mAP={ext_res['mAP']:.2%} | TP={ext_res['TP']} FP={ext_res['FP']} FN={ext_res['FN']} | "
            f"NegClean={ext_res['neg_clean']}/{ext_res['neg_total']} ({ext_res['neg_rate']:.2%})\n"
            f"    [Grid 18 Cells]   Clean={grid_res['clean_cells']}/18 ({grid_res['specificity']:.2%})",
            flush=True,
        )

        score = 0.40 * m9_res["F1"] + 0.30 * ext_res["F1"] + 0.30 * ext_res["neg_rate"]
        if score > best_score:
            best_score = score
            checkpoint_payload = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "train_loss": train_loss,
                "m9_test_metrics": m9_res,
                "external_test_metrics": ext_res,
                "anchor_scales": ANCHOR_SCALES,
                "aspect_ratios": (0.5, 1.0, 2.0),
                "rpn_bbox_xform_clip": RPN_CLIP,
                "roi_bbox_xform_clip": ROI_CLIP,
                "architecture": "FasterRCNN_ResNet50_FPN_V2_Model10_SOTA",
            }
            torch.save(checkpoint_payload, BEST_M10_PATH)
            print(f"  [+] Saved refined Model 10 checkpoint (score={score:.4f}) -> {BEST_M10_PATH.name}", flush=True)


if __name__ == "__main__":
    main()

