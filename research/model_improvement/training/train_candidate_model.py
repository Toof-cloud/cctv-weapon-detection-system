"""
Isolated Training Pipeline for Model Improvement Research.
Trains Faster R-CNN with Multi-Scale & Aspect-Ratio Calibrated Anchors (0.25, 0.5, 1.0, 2.0, 4.0),
transferring pre-trained surveillance weights from Model 9.
Outputs strictly saved to research/model_improvement/checkpoints/.
"""
import os
import sys
import time
import argparse
from pathlib import Path

# Enforce strict CPU thread capping to preserve workstation responsiveness
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
BEST_CHECKPOINT = OUTPUT_DIR / "best_candidate_model_v2.pth"
LATEST_CHECKPOINT = OUTPUT_DIR / "latest_candidate_model_v2.pth"
BASE_MODEL_PATH = ROOT / "best_weapon_detector_ninth_model.pth"


def collate_fn(batch):
    return tuple(zip(*batch))


def load_model9_weights(model, checkpoint_path):
    """Transfers backbone, FPN, and RoI head weights from Model 9, skipping RPN shape mismatches."""
    if not checkpoint_path.exists():
        print(f"[!] Warning: Base model {checkpoint_path} not found. Training from ImageNet default.")
        return model

    print(f"Loading pre-trained surveillance weights from {checkpoint_path.name}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    # Filter out RPN head layers due to 5 aspect ratios vs 3
    filtered = {
        k: v for k, v in state_dict.items()
        if not (k.startswith("rpn.head.cls_logits") or k.startswith("rpn.head.bbox_pred"))
    }
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"  -> Transferred {len(filtered)} layers successfully.")
    print(f"  -> Fresh RPN Head layers initialized for 5 aspect ratios: {[k for k in missing if 'rpn' in k]}")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train Candidate Model with Calibrated Anchors")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--lr-backbone", type=float, default=0.0005, help="Fine-tuning LR for backbone (default: 0.0005)")
    parser.add_argument("--lr-rpn", type=float, default=0.0025, help="Learning rate for new RPN head (default: 0.0025)")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads limit (default: 8)")
    parser.add_argument("--log-freq", type=int, default=40, help="Batch logging frequency (default: 40)")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 76)
    print("   RESEARCH PIPELINE: CANDIDATE MODEL WITH CALIBRATED ANCHORS (V1)")
    print("=" * 76)
    print(f"Device:                {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} ({device})")
    print(f"Pre-trained Source:    {BASE_MODEL_PATH.name}")
    print(f"Aspect Ratios:         (0.33, 0.5, 1.0, 2.0, 3.0) [Realistic Weapon Geometry]")
    print(f"Scales:                ((12,), (24,), (48,), (96,), (160,)) [Human Reach Capped]")
    print(f"Total Epochs:          {args.epochs}")
    print(f"Batch Size:            {args.batch_size}")
    print(f"Target Best Checkpoint: {BEST_CHECKPOINT}")
    print("=" * 76)

    # 1. Instantiate Model & Load Pretrained Weights
    model = build_research_model(
        num_classes=3,
        anchor_sizes=((12,), (24,), (48,), (96,), (160,)),
        aspect_ratios=(0.33, 0.5, 1.0, 2.0, 3.0),
        pretrained_backbone=True,
    )
    model = load_model9_weights(model, BASE_MODEL_PATH)
    model.to(device)

    # 2. Datasets & DataLoaders
    print("\nPreparing enhanced research datasets...")
    train_dataset = EnhancedWeaponDataset("train", augment=True)
    val_dataset = EnhancedWeaponDataset("val", augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    print(f"  -> Train Images: {len(train_dataset)} ({len(train_loader)} batches)")
    print(f"  -> Val Images:   {len(val_dataset)} ({len(val_loader)} batches)")

    # 3. Differential Learning Rates: higher for fresh RPN head, smaller for backbone
    rpn_params = [p for n, p in model.named_parameters() if "rpn.head" in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if "rpn.head" not in n and p.requires_grad]

    optimizer = optim.SGD(
        [
            {"params": other_params, "lr": args.lr_backbone, "weight_decay": 0.0005, "momentum": 0.9},
            {"params": rpn_params, "lr": args.lr_rpn, "weight_decay": 0.0005, "momentum": 0.9},
        ]
    )
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        rpn_cls_loss = 0.0
        rpn_box_loss = 0.0
        roi_cls_loss = 0.0
        roi_box_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader, start=1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += losses.item()
            rpn_cls_loss += loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0)).item()
            rpn_box_loss += loss_dict.get("loss_objectness", torch.tensor(0.0)).item()
            roi_cls_loss += loss_dict.get("loss_classifier", torch.tensor(0.0)).item()
            roi_box_loss += loss_dict.get("loss_box_reg", torch.tensor(0.0)).item()

            if batch_idx % args.log_freq == 0 or batch_idx == len(train_loader):
                cur_loss = running_loss / batch_idx
                print(
                    f"Epoch [{epoch}/{args.epochs}] Step [{batch_idx}/{len(train_loader)}] - "
                    f"Total Loss: {cur_loss:.4f} (RPN_obj: {rpn_box_loss/batch_idx:.4f}, "
                    f"RPN_box: {rpn_cls_loss/batch_idx:.4f}, RoI_cls: {roi_cls_loss/batch_idx:.4f}, "
                    f"RoI_box: {roi_box_loss/batch_idx:.4f})"
                )

        train_loss = running_loss / len(train_loader)

        # Validation Loss Evaluation
        val_running_loss = 0.0
        with torch.no_grad():
            for val_images, val_targets in val_loader:
                val_images = [img.to(device) for img in val_images]
                val_targets = [{k: v.to(device) for k, v in t.items()} for t in val_targets]
                val_loss_dict = model(val_images, val_targets)
                val_losses = sum(loss for loss in val_loss_dict.values())
                val_running_loss += val_losses.item()

        val_loss = val_running_loss / max(1, len(val_loader))
        epoch_sec = time.time() - epoch_start
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"\n>>> Epoch {epoch:02d} Finished in {epoch_sec:.1f}s | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | LR: {lr_now:.6f}"
        )

        # Save Checkpoints
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "train_loss": train_loss,
            "anchor_aspect_ratios": (0.33, 0.5, 1.0, 2.0, 3.0),
            "anchor_sizes": ((12,), (24,), (48,), (96,), (160,)),
        }
        torch.save(ckpt_data, LATEST_CHECKPOINT)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt_data, BEST_CHECKPOINT)
            print(f"    [*] NEW BEST MODEL SAVED: {BEST_CHECKPOINT.name} (Val Loss: {val_loss:.4f})\n")

        lr_scheduler.step()

    total_time = time.time() - start_time
    print("\n" + "=" * 76)
    print(f"TRAINING COMPLETE IN {total_time/60:.2f} MINUTES!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Saved Checkpoint:     {BEST_CHECKPOINT}")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
