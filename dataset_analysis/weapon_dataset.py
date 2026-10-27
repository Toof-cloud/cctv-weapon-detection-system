import os

import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset


class WeaponDataset(Dataset):
    HANDGUN = "/m/0gxl3"
    KNIFE = "/m/04ctx"
    KITCHEN_KNIFE = "/m/058qzx"

    def __init__(
        self,
        image_ids_file,
        annotations_csv,
        images_dir,
        transforms=None,
    ):
        self.transforms = transforms
        self.images_dir = images_dir

        self.annotations = pd.read_csv(
            annotations_csv
        )

        with open(
            image_ids_file,
            "r",
            encoding="utf-8",
        ) as f:
            self.image_ids = [
                line.strip()
                for line in f
                if line.strip()
            ]

        self.annotations = self.annotations[
            self.annotations["ImageID"].isin(
                self.image_ids
            )
        ]

    def __len__(self):
        return len(self.image_ids)

    def _label_to_id(self, label_name):
        if label_name == self.HANDGUN:
            return 1

        if (
            label_name == self.KNIFE
            or label_name == self.KITCHEN_KNIFE
        ):
            return 2

        return 0

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        image_path = os.path.join(
            self.images_dir,
            f"{image_id}.jpg",
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        width, height = image.size

        image_annotations = self.annotations[
            self.annotations["ImageID"]
            == image_id
        ]

        boxes = []
        labels = []

        for _, row in image_annotations.iterrows():

            xmin = float(row["XMin"]) * width
            xmax = float(row["XMax"]) * width
            ymin = float(row["YMin"]) * height
            ymax = float(row["YMax"]) * height

            boxes.append(
                [xmin, ymin, xmax, ymax]
            )

            labels.append(
                self._label_to_id(
                    row["LabelName"]
                )
            )

        boxes = torch.as_tensor(
            boxes,
            dtype=torch.float32,
        )

        labels = torch.as_tensor(
            labels,
            dtype=torch.int64,
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        image = (
            torch.from_numpy(
                __import__("numpy")
                .array(image)
            )
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        return image, target