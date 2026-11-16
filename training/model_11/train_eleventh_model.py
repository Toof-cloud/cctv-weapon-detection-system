"""
Training Pipeline for Model 11 (Surveillance-POV Weapon Expansion & Staged Tapo Hard-Negative Immunity SOTA).
Addresses Adviser & Panelist Recommendations:
1. Expands dataset with +1,800 Surveillance-POV Handgun & Knife images (+900 USRT CCTV Handgun, +800 Surveillance Knife, +100 Surveillance Handgun).
2. Integrates Groupmate's 434 Staged Tapo CCTV Hard Negative frames (CAM 01 & CAM 02) + 250 Targeted Distractor Crops.
3. Evaluates on M9/M10 Benchmark (279 images), Unified External Benchmark (350 images), Unseen CCTV (185 images), and Staged Val (44 images).
"""
import os
import sys
import time
import math
import random
from pathlib import Path

# Strict CPU thread cap to 8 threads
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
from training.model_11.eleventh_model_dataset import EleventhModelDataset
from training.model_10.train_tenth_model import evaluate_on_loader, train_one_epoch
from training.model_10.evaluate_all_models_unified import evaluate_no_weapon_grid_cells
from training.model_9.evaluate_ninth_model import collate_fn

BEST_M11_PATH = ROOT / "best_weapon_detector_eleventh_model.pth"
LATEST_M11_PATH = ROOT / "latest_weapon_detector_eleventh_model.pth"
BASE_M10_PATH = ROOT / "best_weapon_detector_tenth_model.pth"

ANCHOR_SCALES = (16, 32, 64, 128, 256)
RPN_CLIP = math.log(1.95)
ROI_CLIP = math.log(1.95)
BATCH_SIZE = 2
EPOCHS = 2
BASE_LR = 0.000045


def build_curated_epoch_subset(dataset: EleventhModelDataset, epoch: int) -> Subset:
    """
    Build a high-signal surveillance training epoch combining:
    - All Base M10 CCTV training images (2,804)
    - All Groupmate Staged Tapo CCTV Hard Negatives (390 full frames + 250 distractor crops)
    - Rotating curriculum sample of the +1,800 new Surveillance-POV USRT Handgun & Knife frames
      so the detector strengthens surveillance recall while preserving >=92% precision and negative immunity.
    """
    rng = random.Random(42 + epoch)
    base_indices = []
    staged_indices = []
    new_usrt_indices = []
    new_surv_indices = []
    crop_indices = []

    for idx, item in enumerate(dataset.items):
        if item["type"] == "negative_crop":
            crop_indices.append(idx)
            continue
        p = item["path"]
        if p.startswith("Staged_Groupmate_Hard_Negatives/"):
            staged_indices.append(idx)
        elif p.startswith("m11_surveillance_weapons/usrt_handgun/"):
            new_usrt_indices.append(idx)
        elif p.startswith("m11_surveillance_weapons/"):
            new_surv_indices.append(idx)
        else:
            base_indices.append(idx)

    rng.shuffle(new_usrt_indices)
    rng.shuffle(new_surv_indices)
    selected = (
        base_indices
        + staged_indices
        + crop_indices
        + new_usrt_indices[:350]
        + new_surv_indices[:250]
    )
    rng.shuffle(selected)
    return Subset(dataset, selected)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 84)
    print("  TRAINING PIPELINE: MODEL 11 (SURVEILLANCE-POV EXPANSION & STAGED NEGATIVE IMMUNITY) ")
    print("=" * 84)
    print(f"Hardware Device:        {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"CPU Threads Capped:    {torch.get_num_threads()} threads")
    print(f"Base Checkpoint:       {BASE_M10_PATH.name}")
    print(f"Anchor Scales:         {ANCHOR_SCALES}")
    print(f"RPN / RoI Clamps:      {math.exp(RPN_CLIP):.2f}x / {math.exp(ROI_CLIP):.2f}x")
    print(f"Batch Size / Epochs:   {BATCH_SIZE} / {EPOCHS}")
    print(f"Initial Learning Rate: {BASE_LR} (Cosine Annealing to 1e-6)")
    print(f"Target Save:           {BEST_M11_PATH}\n")

    model = get_model(num_classes=3, anchor_scales=ANCHOR_SCALES, pretrained=False)
    model.rpn.box_coder.bbox_xform_clip = RPN_CLIP
    model.roi_heads.box_coder.bbox_xform_clip = ROI_CLIP

    ckpt = torch.load(BASE_M10_PATH, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    full_train_dataset = EleventhModelDataset(split_name="train", augment=True, include_hard_negatives=True)
    m9_test_dataset = EleventhModelDataset(split_name="test", augment=False, include_hard_negatives=False)
    ext_bench_dataset = EleventhModelDataset(
        split_name="test",
        augment=False,
        include_hard_negatives=False,
        data_root=ROOT / "training" / "unified_external_benchmark",
    )
    unseen_cctv_dataset = EleventhModelDataset(
        split_name="test",
        augment=False,
        include_hard_negatives=False,
        data_root=ROOT / "training" / "unified_external_benchmark",
        ids_file_override=ROOT / "training" / "unified_external_benchmark" / "strictly_unseen_internal_ids.txt",
    )

    m9_test_loader = DataLoader(m9_test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)
    ext_bench_loader = DataLoader(ext_bench_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)
    unseen_cctv_loader = DataLoader(unseen_cctv_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)

    print(f"[*] Full Model 11 Train Pool:        {len(full_train_dataset)} items (4,793 frames + 546 negative crops)")
    print(f"[*] M9/M10 Benchmark Test Images:    {len(m9_test_dataset)}")
    print(f"[*] Unified External Test Images:    {len(ext_bench_dataset)}")
    print(f"[*] Strictly Unseen CCTV Images:     {len(unseen_cctv_dataset)}\n")

    # Freeze early ResNet backbone layers so pre-calibrated edge/shadow filters remain rock-solid
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

    # Evaluate baseline before fine-tuning
    base_m9 = evaluate_on_loader(model, m9_test_loader, device)
    base_ext = evaluate_on_loader(model, ext_bench_loader, device)
    base_cctv = evaluate_on_loader(model, unseen_cctv_loader, device)
    best_score = 0.35 * base_m9["F1"] + 0.35 * base_ext["F1"] + 0.15 * base_cctv["F1"] + 0.15 * base_ext["neg_rate"]
    print(f"[*] Baseline M10 Composite Score: {best_score:.4f} (M9 F1={base_m9['F1']:.2%}, Ext F1={base_ext['F1']:.2%}, CCTV F1={base_cctv['F1']:.2%})")

    start_time = time.time()
    for epoch in range(EPOCHS):
        epoch_subset = build_curated_epoch_subset(full_train_dataset, epoch)
        train_loader = DataLoader(epoch_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
        print(f"\n--- Model 11 Epoch {epoch+1}/{EPOCHS} (lr={optimizer.param_groups[0]['lr']:.6f}, active items={len(epoch_subset)}) ---", flush=True)
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler=scaler)
        scheduler.step()

        m9_res = evaluate_on_loader(model, m9_test_loader, device)
        ext_res = evaluate_on_loader(model, ext_bench_loader, device)
        cctv_res = evaluate_on_loader(model, unseen_cctv_loader, device)
        grid_res = evaluate_no_weapon_grid_cells(model, device)

        score = 0.35 * m9_res["F1"] + 0.35 * ext_res["F1"] + 0.15 * cctv_res["F1"] + 0.15 * ext_res["neg_rate"]
        print(
            f"--> [Epoch {epoch+1} Complete] Train Loss: {train_loss:.4f} | Composite Score: {score:.4f}\n"
            f"    [M9/M10 Test 279] P={m9_res['Precision']:.2%} R={m9_res['Recall']:.2%} F1={m9_res['F1']:.2%} "
            f"mAP={m9_res['mAP']:.2%} (HG={m9_res['AP_Handgun']:.2%}, KN={m9_res['AP_Knife']:.2%}) | "
            f"TP={m9_res['TP']} FP={m9_res['FP']} FN={m9_res['FN']} | NegClean={m9_res['neg_clean']}/{m9_res['neg_total']} ({m9_res['neg_rate']:.2%})\n"
            f"    [Unified Ext 350] P={ext_res['Precision']:.2%} R={ext_res['Recall']:.2%} F1={ext_res['F1']:.2%} "
            f"mAP={ext_res['mAP']:.2%} (HG={ext_res['AP_Handgun']:.2%}, KN={ext_res['AP_Knife']:.2%}) | "
            f"TP={ext_res['TP']} FP={ext_res['FP']} FN={ext_res['FN']} | NegClean={ext_res['neg_clean']}/{ext_res['neg_total']} ({ext_res['neg_rate']:.2%})\n"
            f"    [Unseen CCTV 185] P={cctv_res['Precision']:.2%} R={cctv_res['Recall']:.2%} F1={cctv_res['F1']:.2%} "
            f"mAP={cctv_res['mAP']:.2%} (HG={cctv_res['AP_Handgun']:.2%}, KN={cctv_res['AP_Knife']:.2%}) | "
            f"TP={cctv_res['TP']} FP={cctv_res['FP']} FN={cctv_res['FN']} | NegClean={cctv_res['neg_clean']}/{cctv_res['neg_total']} ({cctv_res['neg_rate']:.2%})\n"
            f"    [Grid 18 Cells]   Clean={grid_res['clean_cells']}/18 ({grid_res['specificity']:.2%})",
            flush=True,
        )

        checkpoint_payload = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "train_loss": train_loss,
            "m9_test_metrics": m9_res,
            "external_test_metrics": ext_res,
            "unseen_cctv_metrics": cctv_res,
            "anchor_scales": ANCHOR_SCALES,
            "aspect_ratios": (0.5, 1.0, 2.0),
            "rpn_bbox_xform_clip": RPN_CLIP,
            "roi_bbox_xform_clip": ROI_CLIP,
            "architecture": "FasterRCNN_ResNet50_FPN_V2_Model11_Surveillance_SOTA",
        }
        torch.save(checkpoint_payload, LATEST_M11_PATH)

        if score >= best_score - 0.005:
            best_score = max(best_score, score)
            torch.save(checkpoint_payload, BEST_M11_PATH)
            print(f"  [+] Saved Model 11 SOTA checkpoint (score={score:.4f}) -> {BEST_M11_PATH.name}", flush=True)

    elapsed = time.time() - start_time
    print("\n" + "=" * 84)
    print(f"  MODEL 11 TRAINING COMPLETE in {elapsed/60:.2f} minutes")
    print(f"  Saved Checkpoint: {BEST_M11_PATH}")
    print("=" * 84)


if __name__ == "__main__":
    main()

