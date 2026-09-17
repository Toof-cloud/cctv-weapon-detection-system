from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import Dataset


class SixthModelDataset(Dataset):
    def __init__(self, split_name, augment=True):
        self.split_name = split_name
        self.is_train = (split_name == "train") and augment
        root = Path(__file__).resolve().parent
        self.dataset_root = root.parents[1] / "dataset_analysis" / "3rd_Model_Dataset"
        data_root = root / "sixth_model_data"
        self.annotations = pd.read_csv(data_root / "annotations.csv")

        with open(data_root / f"{split_name}_ids.txt", "r", encoding="utf-8") as file:
            self.image_paths = [line.strip() for line in file if line.strip()]

        self.annotations = self.annotations[
            self.annotations["image_path"].isin(self.image_paths)
        ]

    def __len__(self):
        return len(self.image_paths)

    def _apply_surveillance_augmentations(self, image, boxes, is_knife):
        """
        Applies surveillance domain augmentations to bridge domain gap:
        - Mild blur (simulates CCTV lens / motion blur)
        - Contrast & brightness jitter (simulates CCTV exposure/lighting changes)
        - Downscaling & upscaling (simulates CCTV compression/low sensor resolution)
        - Horizontal flip (with bounding box flip)
        """
        w, h = image.size

        # 1. Horizontal Flip (50% probability)
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                flipped_x1 = w - x2
                flipped_x2 = w - x1
                flipped_boxes.append([flipped_x1, y1, flipped_x2, y2])
            boxes = flipped_boxes

        # 2. CCTV Lens/Motion Blur (higher chance for knife to harmonize studio DSLR with CCTV)
        blur_prob = 0.45 if is_knife else 0.20
        if random.random() < blur_prob:
            radius = random.choice([1.0, 1.5, 2.0])
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        # 3. Contrast / Brightness Jitter (simulates low-light CCTV & sensor noise)
        jitter_prob = 0.40 if is_knife else 0.25
        if random.random() < jitter_prob:
            factor_b = random.uniform(0.8, 1.2)
            factor_c = random.uniform(0.8, 1.2)
            image = ImageEnhance.Brightness(image).enhance(factor_b)
            image = ImageEnhance.Contrast(image).enhance(factor_c)

        # 4. CCTV Downsampling Simulation (downscale and upscale back to simulate compression)
        downsample_prob = 0.35 if is_knife else 0.15
        if random.random() < downsample_prob:
            scale = random.uniform(0.5, 0.75)
            small_w, small_h = max(16, int(w * scale)), max(16, int(h * scale))
            image = image.resize((small_w, small_h), resample=Image.BILINEAR).resize((w, h), resample=Image.BILINEAR)

        return image, boxes

    def __getitem__(self, index):
        relative_path = self.image_paths[index]
        image_file = self.dataset_root / relative_path

        if not image_file.exists():
            raise FileNotFoundError(f"Image not found: {image_file}")

        image = Image.open(image_file).convert("RGB")
        image_width, image_height = image.size

        image_rows = self.annotations[
            self.annotations["image_path"] == relative_path
        ]

        boxes = []
        labels = []
        has_knife = False

        for _, row in image_rows.iterrows():
            scale_x = image_width / row["width"]
            scale_y = image_height / row["height"]
            x1 = row["xmin"] * scale_x
            y1 = row["ymin"] * scale_y
            x2 = row["xmax"] * scale_x
            y2 = row["ymax"] * scale_y

            if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
                continue

            x1 = max(0.0, min(x1, image_width - 1.0))
            y1 = max(0.0, min(y1, image_height - 1.0))
            x2 = max(x1 + 1.0, min(x2, float(image_width)))
            y2 = max(y1 + 1.0, min(y2, float(image_height)))

            boxes.append([x1, y1, x2, y2])
            cid = int(row["class_id"])
            labels.append(cid)
            if cid == 2:
                has_knife = True

        # Apply augmentations during training
        if self.is_train:
            image, boxes = self._apply_surveillance_augmentations(image, boxes, is_knife=has_knife)

        image_tensor = torch.from_numpy(np.array(image))
        image_tensor = image_tensor.permute(2, 0, 1).float() / 255.0

        if boxes:
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([index]),
            }
        else:
            # Negative / Authentic Background frame (VIRAT or USRT No_Gun)
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([index]),
            }

        return image_tensor, target
