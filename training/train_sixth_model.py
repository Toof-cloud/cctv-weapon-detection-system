import os
import sys
import time
import argparse
from pathlib import Path

# Thread guards for CPU stability
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_analysis.build_model import get_model
from training.sixth_model_dataset import SixthModelDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_MODEL = "best_weapon_detector_sixth_model.pth"


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN Model 6 (VIRAT CCTV Negatives + Surveillance-Augmented Knives)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate (default: 0.005)")
    args = parser.parse_args()

    print("=" * 70)
    print("      MODEL 6 TRAINING: VIRAT CCTV NEGATIVES + SURVEILLANCE KNIVES    ")
    print("=" * 70)

    train_dataset = SixthModelDataset("train", augment=True)
    validation_dataset = SixthModelDataset("validation", augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    print(f"Hardware Device:    {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} ({DEVICE})")
    print(f"Train Images:       {len(train_dataset)}")
    print(f"Validation Images:  {len(validation_dataset)}")
    print(f"Batch Size:         {args.batch_size} ({len(train_loader)} batches per epoch)")
    print(f"Total Epochs:       {args.epochs}")
    print(f"Learning Rate:      {args.lr}")
    print(f"Output Checkpoint:  {OUTPUT_MODEL}")
    print("=" * 70)

    model = get_model(num_classes=3).to(DEVICE)
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005,
    )

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        print(f"\n>>> Epoch [{epoch}/{args.epochs}] Starting...")

        for batch_idx, (images, targets) in enumerate(train_loader, 1):
            images = [img.to(DEVICE) for img in images]
            targets = [
                {k: v.to(DEVICE) for k, v in t.items()}
                for t in targets
            ]

            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

            if batch_idx % 100 == 0 or batch_idx == len(train_loader):
                cur_avg = running_loss / batch_idx
                cls_loss = loss_dict.get("loss_classifier", torch.tensor(0.0)).item()
                box_loss = loss_dict.get("loss_box_reg", torch.tensor(0.0)).item()
                rpn_cls = loss_dict.get("loss_objectness", torch.tensor(0.0)).item()
                rpn_box = loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0)).item()
                print(
                    f"  Epoch [{epoch:02d}/{args.epochs:02d}] "
                    f"Batch [{batch_idx:03d}/{len(train_loader):03d}] "
                    f"Total Loss: {cur_avg:.4f} | "
                    f"Cls: {cls_loss:.4f} Box: {box_loss:.4f} RPN: {rpn_cls:.4f}"
                )

        train_loss = running_loss / len(train_loader)

        # Validation Phase
        print(f"  Validating Epoch [{epoch}/{args.epochs}]...")
        val_running_loss = 0.0
        with torch.no_grad():
            for images, targets in validation_loader:
                images = [img.to(DEVICE) for img in images]
                targets = [
                    {k: v.to(DEVICE) for k, v in t.items()}
                    for t in targets
                ]
                loss_dict = model(images, targets)
                losses = sum(loss_dict.values())
                val_running_loss += losses.item()

        val_loss = val_running_loss / len(validation_loader)
        epoch_elapsed = time.time() - epoch_start

        print(
            f"--> Epoch [{epoch:02d}/{args.epochs:02d}] Finished in {epoch_elapsed:.1f}s | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # Save Best Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = ROOT / OUTPUT_MODEL
            torch.save(model.state_dict(), str(ckpt_path))
            print(f"    [*] NEW BEST CHECKPOINT SAVED: {OUTPUT_MODEL} (Val Loss: {best_val_loss:.4f})")

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"MODEL 6 TRAINING COMPLETED in {total_time / 60:.1f} minutes!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Saved Checkpoint:     {ROOT / OUTPUT_MODEL}")
    print("=" * 70)


if __name__ == "__main__":
    main()
