"""
Training Pipeline for Candidate Model V9.
Targeted Hard-Negative Mining & Neural Confidence Depressing:
- 5-level multi-scale anchor pyramid: ((16,), (32,), (64,), (128,), (180,))
- 5 moderated aspect ratios: (0.5, 0.75, 1.0, 1.33, 1.75)
- Stage-decoupled regression clamping:
    RPN clamp = ln(1.4) (1.4x coarse expansion)
    RoI clamp = ln(1.3) (1.3x fine refinement)
    Compound theoretical limit: 1.4 * 1.3 = 1.82x
    Maximum reachable dimension: 180px * sqrt(1.75) * 1.82 ≈ 433px (scaled Full HD: ~570px)
    Strictly eliminates mid-sized desk segment traps (600-714px)
- Injected 56+ targeted hard-negative samples:
    * Masked faces, beanies, and sunglasses (Robber head false alarms)
    * Skeleton bone printed gloves
    * Floor mat neon diagonal rubber borders
    * Wall light switch plates and electrical outlets
    * Retail counter packaging
- Initializes from Candidate Model V8 checkpoint
"""
import os
import sys
import time
import math
from pathlib import Path

# Limit CPU threads to 8
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import torch
torch.set_num_threads(8)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.model_improvement.training.model_builder import build_research_model
from research.model_improvement.dataset.enhanced_dataset import EnhancedWeaponDataset

CHECKPOINT_DIR = ROOT / "research" / "model_improvement" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BEST_V9_PATH = CHECKPOINT_DIR / "best_candidate_model_v9.pth"
PREV_CHECKPOINT = CHECKPOINT_DIR / "best_candidate_model_v8.pth"

BATCH_SIZE = 2
EPOCHS = 4
BASE_LR = 0.00006

ANCHOR_SIZES = ((16,), (32,), (64,), (128,), (180,))
ASPECT_RATIOS = (0.5, 0.75, 1.0, 1.33, 1.75)
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
    print("        TRAINING PIPELINE: CANDIDATE MODEL V9 (NEURAL CONFIDENCE DEPRESSING)        ")
    print("=" * 76)
    print(f"Device:               {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Base Checkpoint:      {PREV_CHECKPOINT.name}")
    print(f"Anchor Scales:        {ANCHOR_SIZES}")
    print(f"Aspect Ratios:        {ASPECT_RATIOS}")
    print(f"RPN / RoI Clamps:     {math.exp(RPN_CLIP):.2f}x / {math.exp(ROI_CLIP):.2f}x (Compound: {math.exp(RPN_CLIP)*math.exp(ROI_CLIP):.2f}x)")
    print(f"Batch Size / Epochs:  {BATCH_SIZE} / {EPOCHS}")
    print(f"Base Learning Rate:   {BASE_LR}")
    print(f"Target Save Checkpoint:{BEST_V9_PATH}\n")

    # 1. Build Candidate V9 Model
    model = build_research_model(
        num_classes=3,
        anchor_sizes=ANCHOR_SIZES,
        aspect_ratios=ASPECT_RATIOS,
        pretrained_backbone=False,
        rpn_bbox_xform_clip=RPN_CLIP,
        roi_bbox_xform_clip=ROI_CLIP,
    )

    # 2. Transfer Weights from Candidate V8 Checkpoint
    if PREV_CHECKPOINT.exists():
        print(f"[*] Initializing weights from Candidate V8: {PREV_CHECKPOINT}")
        ckpt = torch.load(PREV_CHECKPOINT, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        print(f"[*] Weights transferred with complete compatibility! Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        print(f"[!] Warning: Candidate V8 checkpoint not found at {PREV_CHECKPOINT}. Initializing fresh.")

    model.to(device)

    # 3. Enhanced Dataset with V9 Residual Hard Negatives
    print("\n[*] Initializing Enhanced Dataset with V9 hard-negative background mining...")
    train_dataset = EnhancedWeaponDataset(split_name="train", augment=True, include_hard_negatives=True)
    val_dataset = EnhancedWeaponDataset(split_name="val", augment=False, include_hard_negatives=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    print(f"[*] Training Samples: {len(train_dataset)} ({len(train_loader)} batches)")
    print(f"[*] Validation Samples: {len(val_dataset)} ({len(val_loader)} batches)\n")

    # 4. Optimization Setup with Cosine Annealing
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=BASE_LR, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        print(f"--- Epoch {epoch+1}/{EPOCHS} (Current LR: {scheduler.get_last_lr()[0]:.6f}) ---")

        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss = evaluate_val_loss(model, val_loader, device)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        print(f"--> [Epoch {epoch+1} Complete] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} (Elapsed: {epoch_time:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"    [+] New Best Model! Saving checkpoint to: {BEST_V9_PATH.name} (Val Loss: {best_val_loss:.4f})")
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "anchor_sizes": ANCHOR_SIZES,
                "aspect_ratios": ASPECT_RATIOS,
                "rpn_bbox_xform_clip": RPN_CLIP,
                "roi_bbox_xform_clip": ROI_CLIP,
                "description": "Candidate Model V9 (180px anchor ceiling, 1.82x decoupled clamp, V9 hard-negatives)"
            }
            torch.save(checkpoint, BEST_V9_PATH)

    total_time = time.time() - start_time
    print("\n" + "=" * 76)
    print(f"CANDIDATE MODEL V9 TRAINING COMPLETE in {total_time:.1f}s!")
    print(f"Best Checkpoint: {BEST_V9_PATH}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print("=" * 76)


if __name__ == "__main__":
    main()
