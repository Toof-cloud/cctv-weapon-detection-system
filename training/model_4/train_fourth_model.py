import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_analysis.build_model import get_model
from training.model_4.fourth_model_dataset import FourthModelDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_MODEL = ROOT / "best_weapon_detector_fourth_model.pth"
NUM_EPOCHS = 10
BATCH_SIZE = 2


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    print("=" * 60)
    print("FASTER R-CNN MODEL 4 RETRAINING")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_dataset = FourthModelDataset("train")
    val_dataset = FourthModelDataset("validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    validation_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    print(f"Train images:      {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Batch size:        {BATCH_SIZE}")
    print(f"Epochs:            {NUM_EPOCHS}")
    print("-" * 60)

    model = get_model(num_classes=3).to(DEVICE)
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.33
    )

    best_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [image.to(DEVICE) for image in images]
            targets = [
                {key: value.to(DEVICE) for key, value in target.items()}
                for target in targets
            ]

            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

            if (batch_idx + 1) % 150 == 0 or (batch_idx + 1) == len(train_loader):
                print(
                    f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
                    f"Step [{batch_idx + 1}/{len(train_loader)}] "
                    f"Batch Loss: {losses.item():.4f}"
                )

        lr_scheduler.step()
        average_loss = running_loss / len(train_loader)
        print(f"\n---> Epoch {epoch + 1}/{NUM_EPOCHS} Complete - Average Loss: {average_loss:.4f}")

        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(model.state_dict(), str(OUTPUT_MODEL))
            print(f"*** New Best Model Saved: {OUTPUT_MODEL.name} (Loss: {best_loss:.4f}) ***\n")
        else:
            print(f"Current best: {best_loss:.4f}\n")

    print("=" * 60)
    print(f"Training completed successfully! Final model: {OUTPUT_MODEL}")
    print("=" * 60)


if __name__ == "__main__":
    main()
