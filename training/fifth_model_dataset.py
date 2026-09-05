from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class FifthModelDataset(Dataset):
    def __init__(self, split_name):
        root = Path(__file__).resolve().parent
        self.dataset_root = root.parent / "dataset_analysis" / "3rd_Model_Dataset"
        data_root = root / "fifth_model_data"
        self.annotations = pd.read_csv(data_root / "annotations.csv")
        with open(data_root / f"{split_name}_ids.txt", "r", encoding="utf-8") as file:
            self.image_paths = [line.strip() for line in file if line.strip()]
        self.annotations = self.annotations[
            self.annotations["image_path"].isin(self.image_paths)
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        relative_path = self.image_paths[index]
        image_file = self.dataset_root / relative_path
        image = Image.open(image_file).convert("RGB")
        image_width, image_height = image.size
        image_rows = self.annotations[
            self.annotations["image_path"] == relative_path
        ]

        boxes = []
        labels = []
        for _, row in image_rows.iterrows():
            scale_x = image_width / row["width"]
            scale_y = image_height / row["height"]
            x1 = row["xmin"] * scale_x
            y1 = row["ymin"] * scale_y
            x2 = row["xmax"] * scale_x
            y2 = row["ymax"] * scale_y

            # Ensure strict positive width and height
            if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
                continue

            x1 = max(0.0, min(x1, image_width - 1.0))
            y1 = max(0.0, min(y1, image_height - 1.0))
            x2 = max(x1 + 1.0, min(x2, float(image_width)))
            y2 = max(y1 + 1.0, min(y2, float(image_height)))

            boxes.append([x1, y1, x2, y2])
            labels.append(int(row["class_id"]))

        image_tensor = torch.from_numpy(np.array(image))
        image_tensor = image_tensor.permute(2, 0, 1).float() / 255.0

        if boxes:
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([index]),
            }
        else:
            # Negative / Background frame with 0 target boxes
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([index]),
            }
        return image_tensor, target
