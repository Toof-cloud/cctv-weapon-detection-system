"""
Training Pipeline for Candidate Model V6.
Fine-tunes from Candidate V4 with:
- 42 targeted CCTV hard-negatives (acrylic sign dividers, POS monitors, store counters, wall switches).
- 41 full-frame commercial room backgrounds.
- Calibrated box regression delta clamping: bbox_xform_clip = math.log(3.2) = 1.163
  (Eliminates furniture/counters while allowing full geometric span for elongated knives).
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
BEST_V6_PATH = CHECKPOINT_DIR / "best_candidate_model_v6.pth"
PREV_CHECKPOINT = CHECKPOINT_DIR / "best_candidate_model_v4.pth"

BATCH_SIZE = 2
EPOCHS = 5
BASE_LR = 0.0001
BBOX_XFORM_CLIP = 1.163  # ln(3.2) = 1.163


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
    # Model in train mode to compute loss on val
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
    print(f"   STARTING CANDIDATE MODEL V6 TRAINING ON {str(device).upper()}")
    print("=" * 76)
    print(f"Initializing from base: {PREV_CHECKPOINT}")
    print(f"Regression Clamping: bbox_xform_clip = {BBOX_XFORM_CLIP} (max expansion: {math.exp(BBOX_XFORM_CLIP):.2f}x)")

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

    print(f"Train Dataset: {len(train_dataset)} samples (including hard negatives)")
    print(f"Val Dataset:   {len(val_dataset)} samples")

    model = build_research_model(
        num_classes=3,
        pretrained_backbone=False,
        bbox_xform_clip=BBOX_XFORM_CLIP
    )

    if PREV_CHECKPOINT.exists():
        print(f"Loading pretrained weights from {PREV_CHECKPOINT.name}...")
        ckpt = torch.load(PREV_CHECKPOINT, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print("  Successfully loaded weights!")
    else:
        print(f"WARNING: Checkpoint {PREV_CHECKPOINT} not found, training with baseline weights.")

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
            print(f"  >>> New Best Model V6! (Val Loss: {v_loss:.4f}) Saving checkpoint...")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": t_loss,
                "val_loss": v_loss,
                "bbox_xform_clip": BBOX_XFORM_CLIP,
                "anchor_sizes": ((12,), (20,), (36,), (64,), (96,)),
                "aspect_ratios": (0.5, 0.7, 1.0, 1.4, 2.0)
            }, BEST_V6_PATH)

    total_time = time.time() - start_time
    print("\n" + "=" * 76)
    print(f"Training completed in {total_time/60:.2f} mins! Best Val Loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved at: {BEST_V6_PATH}")
    print("=" * 76)


if __name__ == "__main__":
    main()
