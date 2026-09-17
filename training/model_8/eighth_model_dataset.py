from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import Dataset


class EighthModelDataset(Dataset):
    """Dataset loader for Model 8 with Specular Highlight and CCTV Augmentations."""

    def __init__(self, split_name, augment=True):
        self.split_name = split_name
        self.is_train = (split_name == "train") and augment
        root = Path(__file__).resolve().parent
        self.dataset_root = root.parents[1] / "dataset_analysis" / "3rd_Model_Dataset"
        data_root = root / "eighth_model_data"
        self.annotations = pd.read_csv(data_root / "annotations.csv")

        # Create split text files if they don't exist yet
        split_file = data_root / f"{split_name}_ids.txt"
        if not split_file.exists():
            split_paths = sorted(
                self.annotations[self.annotations["split"] == split_name]["image_path"].unique()
            )
            with open(split_file, "w", encoding="utf-8") as f:
                for p in split_paths:
                    f.write(f"{p}\n")

        with open(split_file, "r", encoding="utf-8") as file:
            self.image_paths = [line.strip() for line in file if line.strip()]

        self.annotations = self.annotations[
            self.annotations["image_path"].isin(self.image_paths)
        ]

    def __len__(self):
        return len(self.image_paths)

    def _apply_surveillance_augmentations(self, image, boxes, is_knife, is_handgun):
        """
        Applies surveillance domain augmentations:
        - Horizontal flip (with bounding box flip)
        - Specular highlight / metal finish simulation for handguns (+-25% brightness/contrast)
        - CCTV motion/lens blur
        - Photometric contrast & brightness jitter
        - Resolution downscaling & upscaling (compression simulation)
        - Synthetic Infrared (IR) / Night-Vision mode simulation
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

        # 2. Specular Highlight & Metal Tone Augmentation (for Handguns)
        # Simulates reflective chrome, stainless steel, and matte black finishes
        if is_handgun and random.random() < 0.40:
            specular_factor = random.choice([0.75, 0.85, 1.15, 1.25, 1.35])
            image = ImageEnhance.Brightness(image).enhance(specular_factor)
            contrast_factor = random.choice([0.80, 0.90, 1.20, 1.30])
            image = ImageEnhance.Contrast(image).enhance(contrast_factor)

        # 3. CCTV Lens / Motion Blur
        blur_prob = 0.45 if is_knife else 0.25
        if random.random() < blur_prob:
            radius = random.choice([1.0, 1.5, 2.0])
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        # 4. Contrast / Brightness Jitter (general)
        jitter_prob = 0.40 if is_knife else 0.25
        if random.random() < jitter_prob:
            factor_b = random.uniform(0.8, 1.2)
            factor_c = random.uniform(0.8, 1.2)
            image = ImageEnhance.Brightness(image).enhance(factor_b)
            image = ImageEnhance.Contrast(image).enhance(factor_c)

        # 5. CCTV Downsampling Simulation
        downsample_prob = 0.35 if is_knife else 0.20
        if random.random() < downsample_prob:
            scale = random.uniform(0.5, 0.75)
            small_w, small_h = max(16, int(w * scale)), max(16, int(h * scale))
            image = image.resize((small_w, small_h), resample=Image.BILINEAR).resize((w, h), resample=Image.BILINEAR)

        # 6. Synthetic Infrared (IR) / Night-Vision Simulation (20% probability)
        if random.random() < 0.20:
            gray = np.array(image.convert("L"))
            r_chan = (gray * 0.15).astype(np.uint8)
            g_chan = (gray * 0.85).astype(np.uint8)
            b_chan = (gray * 0.15).astype(np.uint8)
            ir_np = np.stack([r_chan, g_chan, b_chan], axis=-1)
            noise = np.random.normal(0, 4, ir_np.shape).astype(np.int16)
            ir_np = np.clip(ir_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(ir_np)

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
        is_knife = False
        is_handgun = False

        for _, row in image_rows.iterrows():
            class_id = int(row["class_id"])
            if class_id == 0:
                continue

            if class_id == 2:
                is_knife = True
            elif class_id == 1:
                is_handgun = True

            xmin = float(row["xmin"])
            ymin = float(row["ymin"])
            xmax = float(row["xmax"])
            ymax = float(row["ymax"])

            xmin = max(0.0, min(xmin, float(image_width - 1)))
            ymin = max(0.0, min(ymin, float(image_height - 1)))
            xmax = max(xmin + 1.0, min(xmax, float(image_width)))
            ymax = max(ymin + 1.0, min(ymax, float(image_height)))

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_id)

        if self.is_train:
            image, boxes = self._apply_surveillance_augmentations(image, boxes, is_knife, is_handgun)

        image_np = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

        if len(boxes) > 0:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index]),
        }

        return image_tensor, target
