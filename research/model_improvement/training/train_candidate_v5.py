"""
Candidate Model V5 Training Pipeline: Scale-Aware Architectural Clamping & Full-Frame Background Suppression.
Focus:
1. Full-Resolution (1080p) Empty Background Frames (P2-P5 FPN feature alignment).
2. Mathematical Box Regression Delta Clamping (max proposal <= 240px).
3. 100% weight transfer from Candidate V4 (mAP 94.78%).
4. Low learning-rate fine-tuning with early backbone freezing.
Outputs saved to research/model_improvement/checkpoints/best_candidate_model_v5.pth.
"""
import os
import sys
import time
import math
import argparse
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.model_improvement.training.model_builder import build_research_model
from research.model_improvement.dataset.enhanced_dataset import EnhancedWeaponDataset

OUTPUT_DIR = ROOT / "research" / "model_improvement" / "checkpoints"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BEST_CHECKPOINT = OUTPUT_DIR / "best_candidate_model_v5.pth"
LATEST_CHECKPOINT = OUTPUT_DIR / "latest_candidate_model_v5.pth"
V4_CHECKPOINT = OUTPUT_DIR / "best_candidate_model_v4.pth"

ANCHOR_SIZES = ((12,), (20,), (36,), (64,), (96,))
ASPECT_RATIOS = (0.5, 0.7, 1.0, 1.4, 2.0)
BBOX_CLIP = math.log(2.5)  # Max box expansion 2.5x anchor size (<= 240px)


def collate_fn(batch):
    return tuple(zip(*batch))


def load_v4_weights(model, checkpoint_path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Candidate V4 checkpoint not found at {checkpoint_path}")

    print(f"Loading weights from Candidate V4: {checkpoint_path.name}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print(f"  -> Successfully loaded 100% of weights from Candidate V4 (strict=True)!")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train Candidate Model V5 with Scale-Aware Clamping & Full-Frame Backgrounds")
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--lr", type=float, default=0.00006, help="Fine-tuning learning rate (default: 0.00006)")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads limit (default: 8)")
    parser.add_argument("--log-freq", type=int, default=50, help="Batch logging frequency (default: 50)")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 76)
    print("   RESEARCH PIPELINE: CANDIDATE MODEL V5 (SCALE-AWARE ARCHITECTURE)")
    print("=" * 76)
    print(f"Device:                 {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} ({device})")
    print(f"Pre-trained Source:     Candidate V4 ({V4_CHECKPOINT.name})")
    print(f"Aspect Ratios:          {ASPECT_RATIOS}")
    print(f"Scales:                 {ANCHOR_SIZES}")
    print(f"BBox Regressor Clamping: bbox_xform_clip={BBOX_CLIP:.3f} (Max Box <= 240px)")
    print(f"Hard Negatives:         50 samples (Crops + Full-Frame 1080p Backgrounds)")
    print(f"Total Epochs:           {args.epochs}")
    print(f"Batch Size:             {args.batch_size}")
    print(f"Base Learning Rate:     {args.lr}")
    print(f"Target Checkpoint:      {BEST_CHECKPOINT}")
    print("=" * 76)

    # 1. Build Model with Delta Clamping & Load V4 Weights
    model = build_research_model(
        num_classes=3,
        anchor_sizes=ANCHOR_SIZES,
        aspect_ratios=ASPECT_RATIOS,
        pretrained_backbone=False,
        bbox_xform_clip=BBOX_CLIP,
    )
    model = load_v4_weights(model, V4_CHECKPOINT)

    # 2. Freeze early backbone stages to lock core handgun representation
    for name, param in model.backbone.body.named_parameters():
        if any(prefix in name for prefix in ["conv1", "bn1", "layer1"]):
            param.requires_grad = False
    print("Frozen backbone stages: conv1, bn1, layer1.")

    model.to(device)

    # 3. Setup Dataset with full-frame + crop hard negatives
    train_dataset = EnhancedWeaponDataset(split_name="train", augment=True, include_hard_negatives=True)
    val_dataset = EnhancedWeaponDataset(split_name="val", augment=False, include_hard_negatives=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    print(f"Dataset summary: {len(train_dataset)} train samples, {len(val_dataset)} val samples.")

    # 4. Optimizer & Cosine Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(trainable_params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    best_val_loss = float("inf")
    start_time = time.time()

    # 5. Training Loop
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0
        num_batches = len(train_loader)

        print(f"\n>>> Epoch {epoch}/{args.epochs} (LR: {scheduler.get_last_lr()[0]:.6f})")

        for b_idx, (images, targets) in enumerate(train_loader, 1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                print(f"[!] Loss is {losses.item()}, skipping batch {b_idx}")
                continue

            losses.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
            optimizer.step()

            total_train_loss += losses.item()

            if b_idx % args.log_freq == 0 or b_idx == num_batches:
                loss_str = " | ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
                print(f"  [Batch {b_idx:04d}/{num_batches:04d}] Total: {losses.item():.4f} ({loss_str})")

        scheduler.step()
        avg_train_loss = total_train_loss / max(num_batches, 1)

        # Validation Loss Evaluation
        val_loss = 0.0
        val_batches = len(val_loader)
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                val_loss += sum(loss for loss in loss_dict.values()).item()

        avg_val_loss = val_loss / max(val_batches, 1)
        epoch_dur = time.time() - epoch_start
        print(f"--- Epoch {epoch} Finished in {epoch_dur:.1f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} ---")

        # Save Checkpoints
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "anchor_sizes": ANCHOR_SIZES,
            "aspect_ratios": ASPECT_RATIOS,
            "bbox_xform_clip": BBOX_CLIP,
        }
        torch.save(state, LATEST_CHECKPOINT)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(state, BEST_CHECKPOINT)
            print(f"  [*] BEST CHECKPOINT UPDATED: {BEST_CHECKPOINT.name} (Val Loss: {best_val_loss:.4f})")

    total_dur = time.time() - start_time
    print("\n" + "=" * 76)
    print(f"TRAINING COMPLETE in {total_dur / 60:.1f} minutes.")
    print(f"Best Checkpoint: {BEST_CHECKPOINT} (Val Loss: {best_val_loss:.4f})")
    print("=" * 76)


if __name__ == "__main__":
    main()
