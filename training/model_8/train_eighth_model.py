import os
import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_BEST_MODEL = "best_weapon_detector_eighth_model.pth"
OUTPUT_LATEST_MODEL = "latest_weapon_detector_eighth_model.pth"


def collate_fn(batch):
    return tuple(zip(*batch))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train Faster R-CNN Model 8 (Balanced Anchors + Sanitized Negatives + Specular Jitter)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Number of training epochs (default: 12)",
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

    # Apply strict CPU thread limiting to preserve system responsiveness
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
        pass

    from dataset_analysis.build_model import get_model
    from training.model_8.eighth_model_dataset import EighthModelDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 76)
    print("      MODEL 8 TRAINING PIPELINE: BALANCED ANCHORS & SANITIZED NEGATIVES")
    print("=" * 76)
    print(f"Hardware Device:       {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} ({device})")
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Total GPU VRAM:        {vram_gb:.2f} GB")
    print(f"Allocated CPU Threads: {args.threads} (Capped to preserve system responsiveness)")
    print(f"Batch Size:            {args.batch_size}")
    print(f"Total Epochs:          {args.epochs}")
    print(f"Initial Learning Rate: {args.lr} (Cosine Annealing to 1e-5)")
    print(f"Anchor Configuration:  Balanced Scales ((16,), (32,), (64,), (128,), (256,))")
    print(f"Best Model Checkpoint: {ROOT / OUTPUT_BEST_MODEL}")
    print("=" * 76)

    print("\nLoading Model 8 datasets and preparing data loaders...")
    train_dataset = EighthModelDataset("train", augment=True)
    val_dataset = EighthModelDataset("val", augment=False)

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

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    print("\nInitializing Faster R-CNN with Balanced Anchor Scales...")
    model = get_model(num_classes=3, anchor_scales=(16, 32, 64, 128, 256))
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005,
    )

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5,
    )

    best_val_loss = float("inf")
    total_training_start = time.time()

    print("\n" + "-" * 76)
    print("Beginning Model 8 Training Execution...")
    print("-" * 76)

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        batch_count = len(train_loader)

        print(f"\n>>> Epoch {epoch:02d}/{args.epochs:02d} (Current LR: {lr_scheduler.get_last_lr()[0]:.6f})")

        for step, (images, targets) in enumerate(train_loader, start=1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_loss += losses.item()

            if step % args.log_freq == 0 or step == batch_count:
                avg_so_far = train_loss / step
                elapsed = time.time() - epoch_start_time
                print(
                    f"  [Epoch {epoch:02d} | Batch {step:04d}/{batch_count:04d}] "
                    f"Train Loss: {avg_so_far:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

        avg_train_loss = train_loss / batch_count

        # Validation phase
        print("  Evaluating on validation split...")
        val_loss = 0.0
        val_batches = len(val_loader)

        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Model in train mode to compute validation loss
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        avg_val_loss = val_loss / max(1, val_batches)
        epoch_elapsed = time.time() - epoch_start_time

        print(f"  --- Epoch {epoch:02d} Summary ---")
        print(f"  Avg Train Loss: {avg_train_loss:.4f}")
        print(f"  Avg Val Loss:   {avg_val_loss:.4f}")
        print(f"  Epoch Duration: {epoch_elapsed:.1f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), ROOT / OUTPUT_BEST_MODEL)
            print(f"  [*] BEST MODEL CHECKPOINT SAVED: {OUTPUT_BEST_MODEL} (Val Loss: {best_val_loss:.4f})")

        torch.save(model.state_dict(), ROOT / OUTPUT_LATEST_MODEL)
        lr_scheduler.step()

    total_training_time = time.time() - total_training_start
    mins = int(total_training_time // 60)
    secs = int(total_training_time % 60)

    print("\n" + "=" * 76)
    print("MODEL 8 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Total Training Time:    {mins}m {secs}s")
    print(f"Best Validation Loss:   {best_val_loss:.4f}")
    print(f"Best Checkpoint:        {ROOT / OUTPUT_BEST_MODEL}")
    print("=" * 76)


if __name__ == "__main__":
    main()
