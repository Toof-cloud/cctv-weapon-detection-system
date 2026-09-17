from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ThirdModelDataset(Dataset):
    def __init__(self, split_name):
        root = Path(__file__).resolve().parent
        self.dataset_root = root.parents[1] / "dataset_analysis" / "3rd_Model_Dataset"
        data_root = root / "third_model_data"
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
        image = Image.open(self.dataset_root / relative_path).convert("RGB")
        image_width, image_height = image.size
        image_rows = self.annotations[
            self.annotations["image_path"] == relative_path
        ]

        boxes = []
        labels = []
        for _, row in image_rows.iterrows():
            scale_x = image_width / row["width"]
            scale_y = image_height / row["height"]
            boxes.append(
                [
                    row["xmin"] * scale_x,
                    row["ymin"] * scale_y,
                    row["xmax"] * scale_x,
                    row["ymax"] * scale_y,
                ]
            )
            labels.append(int(row["class_id"]))

        image_tensor = torch.from_numpy(__import__("numpy").array(image))
        image_tensor = image_tensor.permute(2, 0, 1).float() / 255.0
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index]),
        }
        return image_tensor, target