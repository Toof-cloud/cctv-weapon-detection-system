import sys
from pathlib import Path
import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_analysis.build_model import get_model
from training.third_model_dataset import ThirdModelDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_MODEL = "best_weapon_detector_third_model.pth"
NUM_EPOCHS = 10
BATCH_SIZE = 2


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    train_loader = DataLoader(
        ThirdModelDataset("train"),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    validation_loader = DataLoader(
        ThirdModelDataset("validation"),
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

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = [image.to(DEVICE) for image in images]
            targets = [
                {key: value.to(DEVICE) for key, value in target.items()}
                for target in targets
            ]
            losses = sum(model(images, targets).values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            running_loss += losses.item()

        average_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - loss: {average_loss:.4f}")

        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(model.state_dict(), OUTPUT_MODEL)
            print(f"Saved: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()