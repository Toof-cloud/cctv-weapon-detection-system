"""
Training Pipeline for Candidate Model V11 (Multi-Angle & Hand-Shadow Invariance SOTA).
Key Enhancements:
1. 8px RPN Anchor Scale: ((8,), (16,), (32,), (64,), (128,)) for distant/small CCTV threats (<25px).
2. Stage-decoupled regression clamping: RPN ln(1.4) + RoI ln(1.4) to maintain bounded geometry (<=501px).
3. 296 Targeted Hard Negatives (including 111 new V11 crops):
   - CAM2_SCENE002 empty hand gesture with door shadow
   - CAM2_SCENE002 actor ponytail over shirt
   - CAM2_SCENE001 floor & counter shadows
   - Unseen 225703 customer shoulder/jacket shadow
   - Unseen 230249 cashier black cap & POS terminal display
4. Synthetic Hand-Grip Occlusion (CutMix on grip) to force feature extraction on blade/slide edges.
5. Hardware Config: CPU strictly capped at 8 threads; GPU runs with full CUDA acceleration.
6. Base Weights: Initialized from best_candidate_model_v10.pth.
"""
import os
import sys
import time
import math
from pathlib import Path

# Strict CPU thread cap to 8 threads
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import torch
torch.set_num_threads(8)
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_analysis.build_model import get_model
from research.model_improvement.dataset.enhanced_dataset import EnhancedWeaponDataset

CHECKPOINT_DIR = ROOT / "research" / "model_improvement" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BEST_V11_PATH = CHECKPOINT_DIR / "best_candidate_model_v11.pth"
LATEST_V11_PATH = CHECKPOINT_DIR / "latest_candidate_model_v11.pth"
BASE_V10_PATH = CHECKPOINT_DIR / "best_candidate_model_v10.pth"

BATCH_SIZE = 2
ACCUM_STEPS = 1
EPOCHS = 4
BASE_LR = 0.0001
RPN_CLIP = math.log(1.4)  # 1.4x
ROI_CLIP = math.log(1.4)  # 1.4x
ANCHOR_SCALES = (8, 16, 32, 64, 128)


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None):
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values()) / ACCUM_STEPS
            scaler.scale(losses).backward()
            if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(data_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values()) / ACCUM_STEPS
            losses.backward()
            if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(data_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += (losses.item() * ACCUM_STEPS)
        num_batches += 1

        if (i + 1) % 100 == 0 or (i + 1) == len(data_loader):
            loss_components = " | ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
            print(f"  [Epoch {epoch+1}/{EPOCHS}] Step {i+1}/{len(data_loader)} - Total: {(losses.item() * ACCUM_STEPS):.4f} ({loss_components})", flush=True)

    return total_loss / max(num_batches, 1)


def evaluate_val_loss(model, data_loader, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
            else:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            total_loss += losses.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("      TRAINING PIPELINE: CANDIDATE MODEL V11 (SOTA MULTI-ANGLE & SHADOW IMMUNITY)      ")
    print("=" * 80)
    print(f"Hardware Device:        {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"CPU Threads Capped:    {torch.get_num_threads()} threads")
    print(f"Base Checkpoint:       {BASE_V10_PATH.name}")
    print(f"Anchor Scales:         {ANCHOR_SCALES} (Includes 8px for distant threats)")
    print(f"RPN / RoI Clamps:      {math.exp(RPN_CLIP):.2f}x / {math.exp(ROI_CLIP):.2f}x (Compound: {math.exp(RPN_CLIP)*math.exp(ROI_CLIP):.2f}x)")
    print(f"Batch Size / Epochs:   {BATCH_SIZE} / {EPOCHS}")
    print(f"Initial Learning Rate: {BASE_LR} (Cosine Annealing to 1e-6)")
    print(f"Target Save:           {BEST_V11_PATH}\n")

    # 1. Build Model Architecture with 8px anchor scales
    model = get_model(num_classes=3, anchor_scales=ANCHOR_SCALES, pretrained=False)
    model.rpn.box_coder.bbox_xform_clip = RPN_CLIP
    model.roi_heads.box_coder.bbox_xform_clip = ROI_CLIP

    # 2. Transfer Weights from Model V10
    if BASE_V10_PATH.exists():
        print(f"[*] Initializing weights from Model V10 Checkpoint: {BASE_V10_PATH.name}")
        ckpt = torch.load(BASE_V10_PATH, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        print(f"[*] Complete weight transfer verified! Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        raise FileNotFoundError(f"Base Model V10 checkpoint not found at {BASE_V10_PATH}")

    model.to(device)

    # 3. Load Enhanced Dataset with V11 + V10 + V9 Targeted Hard Negatives
    print("\n[*] Initializing Enhanced Dataset with V11 door shadow / hand gesture / shoulder negatives...")
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

    print(f"[*] Training Samples:   {len(train_dataset)} ({len(train_loader)} batches)")
    print(f"[*] Validation Samples: {len(val_dataset)} ({len(val_loader)} batches)\n")

    # 4. Optimizer, Scaler & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=BASE_LR, momentum=0.9, weight_decay=0.0005)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} (lr={optimizer.param_groups[0]['lr']:.6f}) ---")
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler=scaler)
        val_loss = evaluate_val_loss(model, val_loader, device)
        scheduler.step()

        print(f"--> [Epoch {epoch+1} Complete] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save Checkpoint
        checkpoint_payload = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "anchor_scales": ANCHOR_SCALES,
            "aspect_ratios": (0.5, 1.0, 2.0),
            "rpn_bbox_xform_clip": RPN_CLIP,
            "roi_bbox_xform_clip": ROI_CLIP,
            "architecture": "FasterRCNN_ResNet50_FPN_V2_CandidateV11",
        }
        torch.save(checkpoint_payload, LATEST_V11_PATH)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint_payload, BEST_V11_PATH)
            print(f"  [+] Best validation loss improved to {best_val_loss:.4f}. Saved checkpoint: {BEST_V11_PATH.name}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"  CANDIDATE MODEL V11 TRAINING COMPLETE in {elapsed/60:.2f} minutes")
    print(f"  Best Validation Loss: {best_val_loss:.4f}")
    print(f"  Saved Checkpoint:     {BEST_V11_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
