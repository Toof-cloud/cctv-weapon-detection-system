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
from training.model_5.fifth_model_dataset import FifthModelDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_MODEL = "best_weapon_detector_fifth_model.pth"
NUM_EPOCHS = 10
BATCH_SIZE = 2


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    print("=" * 65)
    print("STARTING MODEL 5 TRAINING (CCTV-GUN / USRT + CLEAN KNIFE)")
    print("=" * 65)

    train_dataset = FifthModelDataset("train")
    validation_dataset = FifthModelDataset("validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = get_model(num_classes=3).to(DEVICE)
    optimizer = optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )
    best_loss = float("inf")

    print(f"Device: {DEVICE}")
    print(f"Train images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(validation_loader.dataset)}")
    print(f"Target model output: {OUTPUT_MODEL}")
    print("=" * 65)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader, 1):
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

            if batch_idx % 200 == 0 or batch_idx == len(train_loader):
                avg_b_loss = running_loss / batch_idx
                print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] Loss: {avg_b_loss:.4f}")

        training_loss = running_loss / len(train_loader)

        # Validation phase
        val_running_loss = 0.0
        with torch.no_grad():
            for images, targets in validation_loader:
                images = [image.to(DEVICE) for image in images]
                targets = [
                    {key: value.to(DEVICE) for key, value in target.items()}
                    for target in targets
                ]
                loss_dict = model(images, targets)
                losses = sum(loss_dict.values())
                val_running_loss += losses.item()

        validation_loss = val_running_loss / len(validation_loader)
        print(f"--> Epoch {epoch + 1}/{NUM_EPOCHS} Complete | Train Loss: {training_loss:.4f} | Val Loss: {validation_loss:.4f}")

        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(model.state_dict(), str(ROOT / OUTPUT_MODEL))
            print(f"    [*] Checkpoint saved: {OUTPUT_MODEL} (Val Loss: {best_loss:.4f})")

    print("\n" + "=" * 65)
    print("MODEL 5 TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Best model checkpoint saved to: {OUTPUT_MODEL}")
    print("=" * 65)


if __name__ == "__main__":
    main()
