from dataset_analysis.build_model import get_model
import torch
from pathlib import Path

MODEL_PATH = Path("best_weapon_detector.pth")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

model = get_model(num_classes=3)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
    )
)

model.to(device)
model.eval()

print("Checkpoint loaded successfully.")
