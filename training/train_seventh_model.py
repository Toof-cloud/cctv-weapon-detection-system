import os
import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_BEST_MODEL = "best_weapon_detector_seventh_model.pth"
OUTPUT_LATEST_MODEL = "latest_weapon_detector_seventh_model.pth"


def collate_fn(batch):
    return tuple(zip(*batch))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train Faster R-CNN Model 7 (Hard Negative Mining + Small-Scale RPN Anchors + Cosine Annealing)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for training and validation (default: 2)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
        help="Initial learning rate (default: 0.005)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of CPU threads to allocate for PyTorch/BLAS (default: 8)",
    )
    parser.add_argument(
        "--log-freq",
        type=int,
        default=50,
        help="How many batches between console progress updates (default: 50)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Apply strict CPU thread limiting to prevent 100% CPU lockup on Windows
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)

    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(min(args.threads, 4))
    except RuntimeError:
        pass  # Can only be set before parallel work starts

    from dataset_analysis.build_model import get_model
    from training.seventh_model_dataset import SeventhModelDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 76)
    print("      MODEL 7 TRAINING PIPELINE: HARD NEGATIVE MINING & ANCHOR TUNING")
    print("=" * 76)
    print(f"Hardware Device:       {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} ({device})")
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Total GPU VRAM:        {vram_gb:.2f} GB")
    print(f"Allocated CPU Threads: {args.threads} (Capped to preserve system responsiveness)")
    print(f"Batch Size:            {args.batch_size}")
    print(f"Total Epochs:          {args.epochs}")
    print(f"Initial Learning Rate: {args.lr} (Cosine Annealing to 1e-5)")
    print(f"Anchor Configuration:  Small Scales ((8,), (16,), (32,), (64,), (128,))")
    print(f"Best Model Checkpoint: {ROOT / OUTPUT_BEST_MODEL}")
    print("=" * 76)

    print("\nLoading datasets and preparing data loaders...")
    train_dataset = SeventhModelDataset("train", augment=True)
    validation_dataset = SeventhModelDataset("validation", augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # 0 workers avoids thread multiplication and IPC overhead on Windows
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    total_batches = len(train_loader)
    print(f"Train Dataset Size:    {len(train_dataset)} images ({total_batches} batches/epoch)")
    print(f"Val Dataset Size:      {len(validation_dataset)} images ({len(validation_loader)} batches/epoch)")

    print("\nInitializing Faster R-CNN ResNet50-FPN-v2 with small RPN anchors...")
    model = get_model(num_classes=3, small_anchors=True).to(device)

    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5,
    )

    best_val_loss = float("inf")
    training_start = time.time()

    print("\nStarting training loop...\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        current_lr = optimizer.param_groups[0]["lr"]

        print(f">>> [EPOCH {epoch:02d}/{args.epochs:02d}] Initiated | Current Learning Rate: {current_lr:.6f}")

        batch_start_time = time.time()
        for batch_idx, (images, targets) in enumerate(train_loader, 1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

            if batch_idx % args.log_freq == 0 or batch_idx == total_batches:
                elapsed = time.time() - batch_start_time
                batches_left = total_batches - batch_idx
                speed = batch_idx / max(elapsed, 0.001)
                eta_sec = batches_left / max(speed, 0.001)
                eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60):02d}s"

                cur_avg = running_loss / batch_idx
                pct = (batch_idx / total_batches) * 100.0

                cls_loss = loss_dict.get("loss_classifier", torch.tensor(0.0)).item()
                box_loss = loss_dict.get("loss_box_reg", torch.tensor(0.0)).item()
                rpn_cls = loss_dict.get("loss_objectness", torch.tensor(0.0)).item()
                rpn_box = loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0)).item()

                print(
                    f"  [Ep {epoch:02d}/{args.epochs:02d}] "
                    f"Batch [{batch_idx:03d}/{total_batches:03d}] ({pct:5.1f}%) | "
                    f"Loss: {cur_avg:.4f} (Cls: {cls_loss:.3f}, Box: {box_loss:.3f}, RPN-Obj: {rpn_cls:.3f}, RPN-Box: {rpn_box:.3f}) | "
                    f"ETA: {eta_str}"
                )

        train_epoch_loss = running_loss / total_batches

        # Validation Phase
        print(f"  --> Running Validation on {len(validation_dataset)} images...")
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            # In torchvision Faster R-CNN, calling model.train() calculates losses.
            # Setting model.train() with torch.no_grad() gives evaluation loss without updating gradients or BN stats.
            model.train()
            for images, targets in validation_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss_dict.values())
                val_running_loss += losses.item()

        val_epoch_loss = val_running_loss / max(len(validation_loader), 1)
        epoch_duration = time.time() - epoch_start
        dur_str = f"{int(epoch_duration // 60)}m {int(epoch_duration % 60):02d}s"

        print(
            f"=== [EPOCH {epoch:02d}/{args.epochs:02d} COMPLETE] "
            f"Time: {dur_str} | "
            f"Train Loss: {train_epoch_loss:.4f} | "
            f"Val Loss: {val_epoch_loss:.4f}"
        )

        # Save latest checkpoint for recovery
        latest_path = ROOT / OUTPUT_LATEST_MODEL
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_epoch_loss,
            },
            str(latest_path),
        )

        # Save best checkpoint
        if val_epoch_loss < best_val_loss:
            improved_by = best_val_loss - val_epoch_loss
            best_val_loss = val_epoch_loss
            best_path = ROOT / OUTPUT_BEST_MODEL
            torch.save(model.state_dict(), str(best_path))
            print(f"  [*] BEST CHECKPOINT SAVED: {OUTPUT_BEST_MODEL} (Val Loss: {best_val_loss:.4f}, Improvement: -{improved_by:.4f})")

        print("-" * 76)

        # Advance Cosine Annealing learning rate schedule
        scheduler.step()

    total_training_time = time.time() - training_start
    total_dur_str = f"{int(total_training_time // 60)}m {int(total_training_time % 60):02d}s"

    print("\n" + "=" * 76)
    print(f"MODEL 7 TRAINING FINISHED SUCCESSFULLY in {total_dur_str}!")
    print(f"Best Validation Loss:     {best_val_loss:.4f}")
    print(f"Final Best Checkpoint:    {ROOT / OUTPUT_BEST_MODEL}")
    print(f"Latest Recovery State:    {ROOT / OUTPUT_LATEST_MODEL}")
    print("=" * 76)
    print("\nNext step: evaluate Model 7 on the test set using:")
    print("  .venv\\Scripts\\python.exe training/evaluate_seventh_model.py\n")


if __name__ == "__main__":
    main()
