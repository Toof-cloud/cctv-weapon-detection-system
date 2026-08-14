import torch
import torch.optim as optim

from build_model import get_model
from create_dataloaders import (
    train_loader,
    val_loader,
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)

model = get_model()
model.to(DEVICE)

params = [
    p
    for p in model.parameters()
    if p.requires_grad
]

optimizer = optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005,
)

num_epochs = 10

best_loss = float("inf")

for epoch in range(num_epochs):

    model.train()

    running_loss = 0

    for images, targets in train_loader:

        images = [
            img.to(DEVICE)
            for img in images
        ]

        targets = [
            {
                k: v.to(DEVICE)
                for k, v in t.items()
            }
            for t in targets
        ]

        loss_dict = model(
            images,
            targets,
        )

        losses = sum(
            loss
            for loss in loss_dict.values()
        )

        optimizer.zero_grad()

        losses.backward()

        optimizer.step()

        running_loss += losses.item()

    avg_loss = (
        running_loss
        / len(train_loader)
    )

    print(
        f"Epoch {epoch + 1}/{num_epochs}"
    )

    print(
        f"Training Loss: "
        f"{avg_loss:.4f}"
    )

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            "best_weapon_detector.pth",
        )

        print(
            "Saved best model."
        )

print()
print(
    "Training completed."
)