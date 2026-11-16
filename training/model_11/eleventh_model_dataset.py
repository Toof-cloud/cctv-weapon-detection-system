"""
Model 11 Dataset Loader (Surveillance-POV Weapon Expansion + Groupmate Staged Hard Negatives).
Supports:
- Base Model 10 Corpus + 1,800 New Surveillance-POV Handgun & Knife Images (USRT CCTV + Filtered Surveillance Blades/Pistols)
- 434 Full-Frame Staged Tapo CCTV Hard Negatives (CAM_01 & CAM_02) + 250 Distractor ROI Crops from Groupmate's Annotations
- CCTV domain augmentations (1D PSF motion blur, blade glint, specular jitter, surveillance compression)
"""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from research.model_improvement.dataset.enhanced_dataset import (
    apply_directional_motion_blur,
    apply_blade_edge_glint,
)


class EleventhModelDataset(Dataset):
    def __init__(
        self,
        split_name: str = "train",
        augment: bool = True,
        include_hard_negatives: bool = True,
        data_root: Path = None,
        ids_file_override: Path = None,
    ):
        self.split_name = split_name
        self.is_train = (split_name == "train") and augment
        root = Path(__file__).resolve().parents[2]
        self.dataset_root = root / "dataset_analysis" / "3rd_Model_Dataset"

        if data_root is None:
            data_root = root / "training" / "model_11" / "eleventh_model_data"
        self.data_root = Path(data_root)
        self.annotations = pd.read_csv(self.data_root / "annotations.csv")

        split_file = ids_file_override if ids_file_override is not None else (self.data_root / f"{split_name}_ids.txt")
        if split_file.exists():
            with open(split_file, "r", encoding="utf-8") as file:
                image_paths = [line.strip() for line in file if line.strip()]
        else:
            image_paths = sorted(
                self.annotations[self.annotations["split"] == split_name]["image_path"].unique()
            )

        self.annotations = self.annotations[
            self.annotations["image_path"].isin(image_paths)
        ]

        self.items = [{"type": "dataset_item", "path": p} for p in image_paths]

        # Inject V9-V12 targeted CCTV hard negative crops (including groupmate's staged distractor crops)
        if self.is_train and include_hard_negatives:
            neg_paths = []
            for folder_name in [
                "hard_negatives",
                "hard_negatives_fullframe",
                "hard_negatives_v9",
                "hard_negatives_v10",
                "hard_negatives_v11",
                "hard_negatives_v12_staged",
            ]:
                nd = root / "research" / "model_improvement" / "dataset" / folder_name
                if nd.exists():
                    neg_paths.extend(sorted(list(nd.glob("*.jpg"))))
            for np_path in neg_paths:
                self.items.append({"type": "negative_crop", "path": np_path})

    def __len__(self):
        return len(self.items)

    def _apply_surveillance_augmentations(self, image, boxes, is_knife, is_handgun):
        w, h = image.size

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            boxes = [[w - b[2], b[1], w - b[0], b[3]] for b in boxes]

        motion_prob = 0.25 if is_knife else 0.15
        if random.random() < motion_prob:
            ksize = random.choice([5, 7])
            angle = random.uniform(0.0, 180.0)
            image = apply_directional_motion_blur(image, kernel_size=ksize, angle=angle)

        if is_knife and random.random() < 0.25:
            image = apply_blade_edge_glint(image, boxes)

        if is_handgun and random.random() < 0.30:
            specular_factor = random.choice([0.85, 0.92, 1.12, 1.20])
            image = ImageEnhance.Brightness(image).enhance(specular_factor)

        if random.random() < 0.25:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.88, 1.12))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.88, 1.12))

        return image, boxes

    def __getitem__(self, idx):
        item = self.items[idx]
        if item["type"] == "negative_crop":
            image = Image.open(item["path"]).convert("RGB")
            if self.is_train and random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            img_np = np.array(image, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([idx]),
            }
            return image_tensor, target

        rel_path = item["path"]
        img_path = self.dataset_root / rel_path
        image = Image.open(img_path).convert("RGB")
        records = self.annotations[self.annotations["image_path"] == rel_path]

        boxes = []
        labels = []
        is_knife = False
        is_handgun = False

        for _, row in records.iterrows():
            cid = int(row["class_id"])
            if cid > 0:
                x1, y1, x2, y2 = float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])
                if (x2 - x1) > 1.0 and (y2 - y1) > 1.0:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cid)
                    if cid == 1:
                        is_handgun = True
                    elif cid == 2:
                        is_knife = True

        if self.is_train:
            image, boxes = self._apply_surveillance_augmentations(image, boxes, is_knife, is_handgun)

        w, h = image.size
        valid_boxes = []
        valid_labels = []
        for b, l in zip(boxes, labels):
            x1 = max(0.0, min(float(w - 1), b[0]))
            y1 = max(0.0, min(float(h - 1), b[1]))
            x2 = max(x1 + 1.0, min(float(w), b[2]))
            y2 = max(y1 + 1.0, min(float(h), b[3]))
            if (x2 - x1) >= 2.0 and (y2 - y1) >= 2.0:
                valid_boxes.append([x1, y1, x2, y2])
                valid_labels.append(l)

        img_np = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()

        if len(valid_boxes) > 0:
            boxes_tensor = torch.as_tensor(valid_boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(valid_labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
        }
        return image_tensor, target

