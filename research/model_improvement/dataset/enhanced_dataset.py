"""
Enhanced Dataset Loader for Model Improvement Research.
Preserves Model 9 surveillance domain augmentations while adding targeted
metallic blade edge-glint and slender aspect ratio jittering.
"""
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
    """Authentic 1D Point Spread Function (PSF) motion blur for rapid weapon movement."""
    img_np = np.array(image_pil)
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


def apply_blade_edge_glint(image_pil: Image.Image, boxes: list) -> Image.Image:
    """
    Simulates specular reflection along blade edges under ambient surveillance lighting.
    Enhances high-frequency edge visibility without creating artificial block artifacts.
    """
    img_np = np.array(image_pil)
    h, w = img_np.shape[:2]
    overlay = img_np.copy().astype(np.float32)

    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        # Select slender region (knife)
        if bh / max(bw, 1) > 1.8 or bw / max(bh, 1) > 1.8:
            crop = overlay[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size > 0:
                # Add directional glint line across weapon bounding box
                glint_boost = random.uniform(1.2, 1.45)
                crop *= glint_boost
                overlay[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = np.clip(crop, 0, 255)

    return Image.fromarray(overlay.astype(np.uint8))


class EnhancedWeaponDataset(Dataset):
    def __init__(self, split_name: str = "train", augment: bool = True, include_hard_negatives: bool = True):
        self.split_name = split_name
        self.is_train = (split_name == "train") and augment
        root = Path(__file__).resolve().parents[3]
        self.dataset_root = root / "dataset_analysis" / "3rd_Model_Dataset"
        data_root = root / "training" / "model_9" / "ninth_model_data"
        self.annotations = pd.read_csv(data_root / "annotations.csv")

        split_file = data_root / f"{split_name}_ids.txt"
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

        self.items = [{"type": "weapon", "path": p} for p in image_paths]

        # Inject targeted hard-negatives (counters, railings, glass dividers) in training
        if self.is_train and include_hard_negatives:
            neg_paths = []
            hard_neg_dir = root / "research" / "model_improvement" / "dataset" / "hard_negatives"
            if hard_neg_dir.exists():
                neg_paths.extend(sorted(list(hard_neg_dir.glob("*.jpg"))))
            full_neg_dir = root / "research" / "model_improvement" / "dataset" / "hard_negatives_fullframe"
            if full_neg_dir.exists():
                neg_paths.extend(sorted(list(full_neg_dir.glob("*.jpg"))))
            for np_path in neg_paths:
                self.items.append({"type": "negative", "path": np_path})
            print(f"  [EnhancedWeaponDataset] Injected {len(neg_paths)} targeted hard-negative background samples (crops + full-frame).")

    def __len__(self):
        return len(self.items)

    def _apply_surveillance_augmentations(self, image, boxes, is_knife, is_handgun):
        w, h = image.size

        # 1. Horizontal Flip (50% probability)
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                flipped_boxes.append([w - x2, y1, w - x1, y2])
            boxes = flipped_boxes

        # 2. 1D Directional Motion Blur (High probability for knives: 35%, 20% for guns)
        motion_prob = 0.35 if is_knife else 0.20
        if random.random() < motion_prob:
            ksize = random.choice([5, 7, 9, 11])
            angle = random.uniform(0.0, 180.0)
            image = apply_directional_motion_blur(image, kernel_size=ksize, angle=angle)

        # 3. Blade Edge Glint (30% probability for knives)
        if is_knife and random.random() < 0.30:
            image = apply_blade_edge_glint(image, boxes)

        # 4. Specular Highlight & Metal Tone Augmentation (for Handguns)
        if is_handgun and random.random() < 0.40:
            specular_factor = random.choice([0.75, 0.85, 1.15, 1.25, 1.35])
            image = ImageEnhance.Brightness(image).enhance(specular_factor)
            contrast_factor = random.choice([0.80, 0.90, 1.20, 1.30])
            image = ImageEnhance.Contrast(image).enhance(contrast_factor)

        # 5. Contrast / Brightness Jitter (general surveillance lighting)
        jitter_prob = 0.40 if is_knife else 0.25
        if random.random() < jitter_prob:
            factor_b = random.uniform(0.8, 1.2)
            factor_c = random.uniform(0.8, 1.2)
            image = ImageEnhance.Brightness(image).enhance(factor_b)
            image = ImageEnhance.Contrast(image).enhance(factor_c)

        # 6. CCTV Downsampling Simulation (Compression & low resolution)
        downsample_prob = 0.35 if is_knife else 0.20
        if random.random() < downsample_prob:
            scale = random.uniform(0.5, 0.75)
            small_w, small_h = max(16, int(w * scale)), max(16, int(h * scale))
            image = image.resize((small_w, small_h), resample=Image.BILINEAR).resize((w, h), resample=Image.BILINEAR)

        # 7. Synthetic Infrared (IR) / Night-Vision Simulation (15% probability)
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
        item = self.items[idx]
        
        if item["type"] == "negative":
            img_path = Path(item["path"])
            image = Image.open(img_path).convert("RGB")
            boxes = []
            labels = []
            is_knife = False
            is_handgun = False
            if self.is_train:
                # Apply general lighting jitter and downsampling to negative crops
                w, h = image.size
                if random.random() < 0.5:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                if random.random() < 0.3:
                    factor_b = random.uniform(0.8, 1.2)
                    image = ImageEnhance.Brightness(image).enhance(factor_b)
                if random.random() < 0.2:
                    scale = random.uniform(0.6, 0.9)
                    small_w, small_h = max(16, int(w * scale)), max(16, int(h * scale))
                    image = image.resize((small_w, small_h), resample=Image.BILINEAR).resize((w, h), resample=Image.BILINEAR)
        else:
            rel_path = item["path"]
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

