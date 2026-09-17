"""
Training Pipeline for Candidate Model V8.
Fenced Regression Anchor Geometry & Stage-Decoupled Clamping:
- 5-level multi-scale anchor pyramid: ((16,), (32,), (64,), (128,), (192,))
- 5 moderated aspect ratios: (0.5, 0.75, 1.0, 1.5, 2.0)
- Stage-decoupled regression clamping:
    RPN clamp = ln(1.4) (1.4x coarse expansion)
    RoI clamp = ln(1.3) (1.3x fine refinement)
    Compound theoretical limit: 1.4 * 1.3 = 1.82x
    Maximum reachable dimension: 271.5px * 1.82 = 494.1px (105px safety gap below 600px danger zone)
- Scale jittering (0.80x - 1.25x) and CCTV perspective shear (±10%)
- Full retention of all 42 full-frame + crop commercial room hard negatives
- Initializes from Candidate V7 checkpoint
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
BEST_V8_PATH = CHECKPOINT_DIR / "best_candidate_model_v8.pth"
PREV_CHECKPOINT = CHECKPOINT_DIR / "best_candidate_model_v7.pth"

BATCH_SIZE = 2
EPOCHS = 4
BASE_LR = 0.00008

ANCHOR_SIZES = ((16,), (32,), (64,), (128,), (192,))
ASPECT_RATIOS = (0.5, 0.75, 1.0, 1.5, 2.0)
RPN_CLIP = math.log(1.4)  # ~0.3365 (1.4x)
ROI_CLIP = math.log(1.3)  # ~0.2624 (1.3x)


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
    print(f"   STARTING CANDIDATE MODEL V8 TRAINING ON {str(device).upper()}")
    print("   FENCED REGRESSION & STAGE-DECOUPLED ANCHOR GEOMETRY")
    print("=" * 76)
    print(f"Initializing base weights from: {PREV_CHECKPOINT}")
    print(f"Multi-Scale Anchor Pyramid: {ANCHOR_SIZES}")
    print(f"Aspect Ratios: {ASPECT_RATIOS}")
    print(f"RPN Clamp: ln(1.4) = {RPN_CLIP:.4f} (1.40x)")
    print(f"RoI Clamp: ln(1.3) = {ROI_CLIP:.4f} (1.30x)")
    compound = math.exp(RPN_CLIP) * math.exp(ROI_CLIP)
    max_base = 192 / math.sqrt(0.5)
    max_reach = max_base * compound
    print(f"Compound Expansion Limit: {compound:.2f}x | Max Reachable Box: {max_reach:.1f}px (Safety gap: {600-max_reach:.1f}px)")

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
        rpn_bbox_xform_clip=RPN_CLIP,
        roi_bbox_xform_clip=ROI_CLIP
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
            print(f"  >>> New Best Model V8! (Val Loss: {v_loss:.4f}) Saving checkpoint...")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": t_loss,
                "val_loss": v_loss,
                "rpn_bbox_xform_clip": RPN_CLIP,
                "roi_bbox_xform_clip": ROI_CLIP,
                "anchor_sizes": ANCHOR_SIZES,
                "aspect_ratios": ASPECT_RATIOS
            }, BEST_V8_PATH)

    total_time = time.time() - start_time
    print("\n" + "=" * 76)
    print(f"Candidate Model V8 Training completed in {total_time/60:.2f} mins! Best Val Loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved at: {BEST_V8_PATH}")
    print("=" * 76)


if __name__ == "__main__":
    main()
