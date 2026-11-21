"""
Training Pipeline for Model 13 (Final Production SOTA — Model 12 + Household Look-Alike Hard-Negative Immunity:
TV Remotes, Metallic Kitchen Ladles, Rolled White Rags/Towels, Smartphones & Wallets).
"""
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import torch
torch.set_num_threads(8)
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_analysis.build_model import get_model
from training.model_13.thirteenth_model_dataset import ThirteenthModelDataset
from training.model_10.train_tenth_model import evaluate_on_loader
from training.model_10.evaluate_all_models_unified import evaluate_no_weapon_grid_cells
from training.model_9.evaluate_ninth_model import collate_fn
from training.model_12.train_twelfth_model import evaluate_staged_val_44_frames

BASE_M12_PATH = ROOT / "best_weapon_detector_twelfth_model.pth"
BEST_M13_PATH = ROOT / "best_weapon_detector_thirteenth_model.pth"
LATEST_M13_PATH = ROOT / "latest_weapon_detector_thirteenth_model.pth"
MOCKUP_M13_PATH = ROOT / "mockup_ui" / "models" / "best_weapon_detector_thirteenth_model.pth"
HISTORY_JSON_PATH = ROOT / "training" / "model_13" / "model_13_training_history.json"

ANCHOR_SCALES = (16, 32, 64, 128, 256)
RPN_CLIP = math.log(1.95)
ROI_CLIP = math.log(1.95)
BATCH_SIZE = 2
EPOCHS = 2
BASE_LR = 0.000035


def build_curated_epoch_subset(dataset: ThirteenthModelDataset, epoch: int) -> Subset:
    rng = random.Random(130 + epoch)
    base_and_costi = []
    staged_prev = []
    nov20_full = []
    usrt_hg = []
    surv_kn = []
    crops = []

    for idx, item in enumerate(dataset.items):
        if item["type"] == "negative_crop":
            crops.append(idx)
            continue
        p = item["path"]
        if p.startswith("Nov20_Staged_Hard_Negatives/"):
            nov20_full.append(idx)
        elif p.startswith("Staged_Groupmate_Hard_Negatives/"):
            staged_prev.append(idx)
        elif p.startswith("m11_surveillance_weapons/usrt_handgun/"):
            usrt_hg.append(idx)
        elif p.startswith("m11_surveillance_weapons/"):
            surv_kn.append(idx)
        else:
            base_and_costi.append(idx)

    rng.shuffle(usrt_hg)
    rng.shuffle(surv_kn)

    # Oversample Nov 20 household non-weapons 2x so Faster R-CNN firmly suppresses TV remotes / ladles / rags
    selected = (
        base_and_costi
        + staged_prev
        + nov20_full * 2
        + crops
        + surv_kn[:450]
        + usrt_hg[:450]
    )
    rng.shuffle(selected)
    return Subset(dataset, selected)


def train_one_epoch_with_components(model, optimizer, data_loader, device, epoch, scaler=None):
    model.train()
    total_loss = 0.0
    comp_sums = {
        "loss_classifier": 0.0,
        "loss_box_reg": 0.0,
        "loss_objectness": 0.0,
        "loss_rpn_box_reg": 0.0,
    }
    num_batches = 0
    optimizer.zero_grad()

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += float(losses.item())
        for k in comp_sums:
            if k in loss_dict:
                comp_sums[k] += float(loss_dict[k].item())
        num_batches += 1

        if (i + 1) % 300 == 0 or (i + 1) == len(data_loader):
            loss_str = " | ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
            print(f"  [Epoch {epoch+1}/{EPOCHS}] Step {i+1}/{len(data_loader)} - Total: {losses.item():.4f} ({loss_str})", flush=True)

    denom = max(1, num_batches)
    return round(total_loss / denom, 6), {k: round(v / denom, 6) for k, v in comp_sums.items()}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"  MODEL 13 (FINAL SOTA) TRAINING PIPELINE | Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80, flush=True)

    train_full_ds = ThirteenthModelDataset(split_name="train", augment=True, include_hard_negatives=True)
    m9_test_ds = ThirteenthModelDataset(split_name="test", augment=False, include_hard_negatives=False)
    ext_test_ds = ThirteenthModelDataset(
        split_name="test", augment=False, include_hard_negatives=False,
        data_root=ROOT / "training" / "unified_external_benchmark",
    )
    cctv_test_ds = ThirteenthModelDataset(
        split_name="test", augment=False, include_hard_negatives=False,
        data_root=ROOT / "training" / "unified_external_benchmark",
        ids_file_override=ROOT / "training" / "unified_external_benchmark" / "strictly_unseen_internal_ids.txt",
    )

    m9_loader = DataLoader(m9_test_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
    ext_loader = DataLoader(ext_test_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
    cctv_loader = DataLoader(cctv_test_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = get_model(num_classes=3, anchor_scales=ANCHOR_SCALES, pretrained=False)
    model.rpn.box_coder.bbox_xform_clip = RPN_CLIP
    model.roi_heads.box_coder.bbox_xform_clip = ROI_CLIP

    ckpt = torch.load(BASE_M12_PATH, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd)
    model.to(device)
    print(f"Initialized from Model 12 SOTA: {BASE_M12_PATH.name}", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=BASE_LR, weight_decay=0.0005)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.000010)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history = {"model": "Model 13 (Final Production SOTA)", "epochs": []}
    t_all = time.time()

    for epoch in range(EPOCHS):
        t0 = time.time()
        subset = build_curated_epoch_subset(train_full_ds, epoch)
        train_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} | Items: {len(subset)} | LR: {optimizer.param_groups[0]['lr']:.6f} ---", flush=True)

        avg_loss, comp_losses = train_one_epoch_with_components(model, optimizer, train_loader, device, epoch, scaler)
        scheduler.step()

        m9 = evaluate_on_loader(model, m9_loader, device)
        ext = evaluate_on_loader(model, ext_loader, device)
        cctv = evaluate_on_loader(model, cctv_loader, device)
        grid = evaluate_no_weapon_grid_cells(model, device)
        st44 = evaluate_staged_val_44_frames(model, device)
        ep_min = round((time.time() - t0) / 60.0, 2)

        print(
            f"[Epoch {epoch+1} Summary | {ep_min} min] Loss={avg_loss:.6f}\n"
            f"  M9 279  : mAP={m9['mAP']:.2%} F1={m9['F1']:.2%} P={m9['Precision']:.2%} R={m9['Recall']:.2%} Neg={m9['neg_clean']}/51\n"
            f"  Ext 350 : mAP={ext['mAP']:.2%} F1={ext['F1']:.2%} P={ext['Precision']:.2%} R={ext['Recall']:.2%} Neg={ext['neg_clean']}/100\n"
            f"  CCTV 185: mAP={cctv['mAP']:.2%} F1={cctv['F1']:.2%} P={cctv['Precision']:.2%} R={cctv['Recall']:.2%} Neg={cctv['neg_clean']}/40\n"
            f"  Grid 18 : {grid['clean_cells']}/18 | Staged 44: Raw={st44['raw_clean_frames']}/44 PostCCTV={st44['post_cctv_clean_frames']}/44",
            flush=True,
        )

        payload = {
            "epoch": epoch + 1,
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "train_loss": avg_loss,
            "loss_components": comp_losses,
            "m9_test_metrics": m9,
            "external_test_metrics": ext,
            "unseen_cctv_metrics": cctv,
            "grid_18_metrics": grid,
            "staged_val_44_metrics": st44,
            "anchor_scales": ANCHOR_SCALES,
            "aspect_ratios": (0.5, 1.0, 2.0),
            "rpn_bbox_xform_clip": RPN_CLIP,
            "roi_bbox_xform_clip": ROI_CLIP,
            "architecture": "FasterRCNN_ResNet50_FPN_V2_Model13_Final_SOTA",
        }
        torch.save(payload, LATEST_M13_PATH)

        history["epochs"].append({
            "epoch": epoch + 1,
            "duration_minutes": ep_min,
            "train_loss": avg_loss,
            "loss_components": comp_losses,
            "m9_test_279": m9,
            "external_350": ext,
            "unseen_cctv_185": cctv,
            "grid_18_cells": grid,
            "staged_val_44": st44,
        })
        HISTORY_JSON_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"\n[DONE] Model 13 2-epoch training completed in {(time.time() - t_all)/60.0:.2f} min -> {LATEST_M13_PATH.name}")


if __name__ == "__main__":
    main()

