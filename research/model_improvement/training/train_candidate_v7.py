"""
Training Pipeline for Candidate Model V7.
Optimizes Close and Medium Range CCTV Weapon Detection.
Fine-tunes from Candidate V6 with:
- 5-level multi-scale anchor pyramid: ((18,), (36,), (72,), (144,), (280,))
- 5 calibrated aspect ratios: (0.4, 0.7, 1.0, 1.5, 2.5)
- Mathematical proposal expansion clamping: bbox_xform_clip = math.log(2.0) = 0.693 (max box <= 560px)
- Scale jittering (0.80x - 1.25x) and CCTV perspective shear in data loading
- Full retention of all 42 full-frame + crop commercial room hard negatives
"""
import os
import sys
import time
import math
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.model_improvement.training.model_builder import build_research_model
from research.model_improvement.dataset.enhanced_dataset import EnhancedWeaponDataset

CHECKPOINT_DIR = ROOT / "research" / "model_improvement" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BEST_V7_PATH = CHECKPOINT_DIR / "best_candidate_model_v7.pth"
PREV_CHECKPOINT = CHECKPOINT_DIR / "best_candidate_model_v7.pth"

BATCH_SIZE = 2
EPOCHS = 4
BASE_LR = 0.00008
BBOX_XFORM_CLIP = math.log(2.0)  # ln(2.0) = 0.69315
ANCHOR_SIZES = ((18,), (36,), (72,), (144,), (280,))
ASPECT_RATIOS = (0.4, 0.7, 1.0, 1.5, 2.5)


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += losses.item()
        num_batches += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(data_loader):
            loss_components = " | ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
            print(f"  [Epoch {epoch+1}/{EPOCHS}] Step {i+1}/{len(data_loader)} - Total: {losses.item():.4f} ({loss_components})")

    return total_loss / max(num_batches, 1)


def evaluate_val_loss(model, data_loader, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 76)
    print(f"   STARTING CANDIDATE MODEL V7 TRAINING ON {str(device).upper()}")
    print("   OPTIMIZATION: CLOSE & MEDIUM RANGE CCTV WEAPON DETECTION")
    print("=" * 76)
    print(f"Initializing base weights from: {PREV_CHECKPOINT}")
    print(f"Multi-Scale Anchor Pyramid: {ANCHOR_SIZES}")
    print(f"Aspect Ratios: {ASPECT_RATIOS}")
    print(f"Regression Clamping: bbox_xform_clip = {BBOX_XFORM_CLIP:.4f} (max expansion: {math.exp(BBOX_XFORM_CLIP):.2f}x -> max box {280*math.exp(BBOX_XFORM_CLIP):.0f}px)")

    train_dataset = EnhancedWeaponDataset("train", augment=True, include_hard_negatives=True)
    val_dataset = EnhancedWeaponDataset("val", augment=False, include_hard_negatives=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    print(f"Train Dataset: {len(train_dataset)} samples (including hard negatives & scale jittering)")
    print(f"Val Dataset:   {len(val_dataset)} samples")

    model = build_research_model(
        num_classes=3,
        anchor_sizes=ANCHOR_SIZES,
        aspect_ratios=ASPECT_RATIOS,
        pretrained_backbone=False,
        bbox_xform_clip=BBOX_XFORM_CLIP
    )

    if PREV_CHECKPOINT.exists():
        print(f"Loading pretrained weights from {PREV_CHECKPOINT.name}...")
        ckpt = torch.load(PREV_CHECKPOINT, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Successfully loaded base weights! (Missing: {len(missing)}, Unexpected: {len(unexpected)})")
    else:
        print(f"WARNING: Checkpoint {PREV_CHECKPOINT} not found, initializing fresh backbone.")

    model.to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=BASE_LR,
        momentum=0.9,
        weight_decay=0.0005
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} (LR: {optimizer.param_groups[0]['lr']:.6f}) ---")
        t_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        v_loss = evaluate_val_loss(model, val_loader, device)
        lr_scheduler.step()

        print(f"Epoch {epoch+1} Summary: Train Loss = {t_loss:.4f} | Val Loss = {v_loss:.4f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            print(f"  >>> New Best Model V7! (Val Loss: {v_loss:.4f}) Saving checkpoint...")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": t_loss,
                "val_loss": v_loss,
                "bbox_xform_clip": BBOX_XFORM_CLIP,
                "anchor_sizes": ANCHOR_SIZES,
                "aspect_ratios": ASPECT_RATIOS
            }, BEST_V7_PATH)

    total_time = time.time() - start_time
    print("\n" + "=" * 76)
    print(f"Candidate Model V7 Training completed in {total_time/60:.2f} mins! Best Val Loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved at: {BEST_V7_PATH}")
    print("=" * 76)


if __name__ == "__main__":
    main()
