import os
import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_BEST_MODEL = "best_weapon_detector_ninth_model.pth"
OUTPUT_LATEST_MODEL = "latest_weapon_detector_ninth_model.pth"


def collate_fn(batch):
    return tuple(zip(*batch))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train Faster R-CNN Model 9 (CCTV Hard Negatives + Directional Motion Blur)"
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
    from training.model_9.ninth_model_dataset import NinthModelDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 76)
    print("      MODEL 9 TRAINING PIPELINE: SURVEILLANCE NEGATIVES & MOTION BLUR")
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

    print("\nLoading Model 9 datasets and preparing data loaders...")
    train_dataset = NinthModelDataset("train", augment=True)
    val_dataset = NinthModelDataset("val", augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Training Samples:   {len(train_dataset)} ({len(train_loader)} batches)")
    print(f"Validation Samples: {len(val_dataset)} ({len(val_loader)} batches)")

    print("\nInitializing Faster R-CNN (ResNet-50-FPN) with Balanced Anchor Generator...")
    # Class 0: Background, Class 1: Handgun, Class 2: Knife
    # Balanced anchor scales ((16,), (32,), (64,), (128,), (256,))
    model = get_model(num_classes=3, anchor_scales=(16, 32, 64, 128, 256))

    # Warm-start weights from Model 8 for fast and stable convergence
    model8_path = ROOT / "best_weapon_detector_eighth_model.pth"
    if model8_path.exists():
        print(f"Warm-starting backbone and head from Model 8: {model8_path.name}")
        checkpoint = torch.load(model8_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
    else:
        print("Training from pre-trained ImageNet backbone.")

    model.to(device)

    # Trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    total_start_time = time.time()

    print("\n" + "-" * 76)
    print("Beginning Training Iterations...")
    print("-" * 76)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        running_loss_classifier = 0.0
        running_loss_box_reg = 0.0
        running_loss_objectness = 0.0
        running_loss_rpn_box_reg = 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n[Epoch {epoch}/{args.epochs}] Current Learning Rate: {current_lr:.6f}")

        for batch_idx, (images, targets) in enumerate(train_loader, start=1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            optimizer.step()

            loss_val = losses.item()
            running_loss += loss_val
            running_loss_classifier += loss_dict.get("loss_classifier", torch.tensor(0.0)).item()
            running_loss_box_reg += loss_dict.get("loss_box_reg", torch.tensor(0.0)).item()
            running_loss_objectness += loss_dict.get("loss_objectness", torch.tensor(0.0)).item()
            running_loss_rpn_box_reg += loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0)).item()

            if batch_idx % args.log_freq == 0 or batch_idx == len(train_loader):
                avg_b_loss = running_loss / batch_idx
                print(
                    f"  Batch [{batch_idx:04d}/{len(train_loader):04d}] "
                    f"Loss: {loss_val:.4f} (Avg: {avg_b_loss:.4f}) | "
                    f"Cls: {loss_dict.get('loss_classifier', 0):.3f} | "
                    f"Box: {loss_dict.get('loss_box_reg', 0):.3f} | "
                    f"RPN-Obj: {loss_dict.get('loss_objectness', 0):.3f}"
                )

        train_epoch_loss = running_loss / len(train_loader)

        # Validation Phase
        print("  Evaluating on validation set...")
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                val_loss += sum(loss for loss in loss_dict.values()).item()

        val_epoch_loss = val_loss / len(val_loader)
        lr_scheduler.step()
        epoch_time = time.time() - epoch_start

        print(
            f"--> Epoch {epoch:02d} Complete in {epoch_time:.1f}s | "
            f"Train Loss: {train_epoch_loss:.4f} | "
            f"Val Loss: {val_epoch_loss:.4f}"
        )

        # Save latest model checkpoint
        latest_path = ROOT / OUTPUT_LATEST_MODEL
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_epoch_loss,
            },
            latest_path,
        )

        # Save best model checkpoint
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_path = ROOT / OUTPUT_BEST_MODEL
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                },
                best_path,
            )
            print(f"  [*] New Best Validation Loss ({best_val_loss:.4f})! Saved checkpoint to: {OUTPUT_BEST_MODEL}")

    total_time = time.time() - total_start_time
    print("\n" + "=" * 76)
    print("                MODEL 9 TRAINING COMPLETE!")
    print(f"Total Time:             {total_time / 60:.2f} minutes")
    print(f"Best Validation Loss:   {best_val_loss:.4f}")
    print(f"Final Model Weights:    {ROOT / OUTPUT_BEST_MODEL}")
    print("=" * 76)


if __name__ == "__main__":
    main()
