from pathlib import Path
import math
import random
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import Dataset


def apply_directional_motion_blur(image_pil: Image.Image, kernel_size: int = 9, angle: float = 0.0) -> Image.Image:
    """
    Applies authentic 1D Point Spread Function (PSF) motion blur to simulate rapid weapon movement.
    """
    img_np = np.array(image_pil)
    # Generate linear motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    rad = math.radians(angle)
    dx = math.cos(rad)
    dy = math.sin(rad)

    for i in range(-center, center + 1):
        x = int(round(center + i * dx))
        y = int(round(center + i * dy))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0

    k_sum = kernel.sum()
    if k_sum > 0:
        kernel /= k_sum
    else:
        kernel[center, center] = 1.0

    blurred = cv2.filter2D(img_np, -1, kernel)
    return Image.fromarray(blurred)


class NinthModelDataset(Dataset):
    """Dataset loader for Model 9 with Directional Motion Blur and Foreshortening Augmentations."""

    def __init__(self, split_name, augment=True):
        self.split_name = split_name
        self.is_train = (split_name == "train") and augment
        root = Path(__file__).resolve().parent
        self.dataset_root = root.parents[1] / "dataset_analysis" / "3rd_Model_Dataset"
        data_root = root / "ninth_model_data"
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
        - Horizontal flip (with bounding box coordinate update)
        - 1D Directional Motion Blur (simulating rapid weapon draw / knife thrust)
        - Specular highlight / metal finish simulation for handguns
        - Photometric contrast & brightness jitter
        - CCTV downsampling & compression simulation
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

        # 2. 1D Directional Motion Blur (Higher probability for knives: 35%, 20% for guns)
        motion_prob = 0.35 if is_knife else 0.20
        if random.random() < motion_prob:
            ksize = random.choice([5, 7, 9, 11])
            angle = random.uniform(0.0, 180.0)
            image = apply_directional_motion_blur(image, kernel_size=ksize, angle=angle)

        # 3. Specular Highlight & Metal Tone Augmentation (for Handguns)
        if is_handgun and random.random() < 0.40:
            specular_factor = random.choice([0.75, 0.85, 1.15, 1.25, 1.35])
            image = ImageEnhance.Brightness(image).enhance(specular_factor)
            contrast_factor = random.choice([0.80, 0.90, 1.20, 1.30])
            image = ImageEnhance.Contrast(image).enhance(contrast_factor)

        # 4. Contrast / Brightness Jitter (general surveillance lighting)
        jitter_prob = 0.40 if is_knife else 0.25
        if random.random() < jitter_prob:
            factor_b = random.uniform(0.8, 1.2)
            factor_c = random.uniform(0.8, 1.2)
            image = ImageEnhance.Brightness(image).enhance(factor_b)
            image = ImageEnhance.Contrast(image).enhance(factor_c)

        # 5. CCTV Downsampling Simulation (Compression & low resolution)
        downsample_prob = 0.35 if is_knife else 0.20
        if random.random() < downsample_prob:
            scale = random.uniform(0.5, 0.75)
            small_w, small_h = max(16, int(w * scale)), max(16, int(h * scale))
            image = image.resize((small_w, small_h), resample=Image.BILINEAR).resize((w, h), resample=Image.BILINEAR)

        # 6. Synthetic Infrared (IR) / Night-Vision Simulation (15% probability)
        if random.random() < 0.15:
            gray = np.array(image.convert("L"))
            r_chan = (gray * 0.15).astype(np.uint8)
            g_chan = (gray * 0.85).astype(np.uint8)
            b_chan = (gray * 0.15).astype(np.uint8)
            ir_np = np.stack([r_chan, g_chan, b_chan], axis=-1)
            noise = np.random.normal(0, 4, ir_np.shape).astype(np.int16)
            ir_noisy = np.clip(ir_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(ir_noisy)

        return image, boxes

    def __getitem__(self, idx):
        rel_path = self.image_paths[idx]
        img_path = self.dataset_root / rel_path

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at {img_path}")

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        img_annotations = self.annotations[self.annotations["image_path"] == rel_path]

        boxes = []
        labels = []

        is_knife = False
        is_handgun = False

        for _, row in img_annotations.iterrows():
            cid = int(row["class_id"])
            if cid > 0:
                xmin = max(0.0, min(float(w), float(row["xmin"])))
                ymin = max(0.0, min(float(h), float(row["ymin"])))
                xmax = max(0.0, min(float(w), float(row["xmax"])))
                ymax = max(0.0, min(float(h), float(row["ymax"])))

                # Ensure non-zero area
                if xmax - xmin >= 2.0 and ymax - ymin >= 2.0:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(cid)
                    if cid == 2:
                        is_knife = True
                    elif cid == 1:
                        is_handgun = True

        if self.is_train:
            image, boxes = self._apply_surveillance_augmentations(image, boxes, is_knife, is_handgun)

        image_tensor = torch.from_numpy(np.array(image).transpose((2, 0, 1))).float() / 255.0

        target = {}
        if len(boxes) > 0:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target["boxes"] = boxes_tensor
        target["labels"] = labels_tensor
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image_tensor, target
