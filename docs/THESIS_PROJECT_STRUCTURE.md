# CCTV Weapon Detection System - Thesis Project Structure

**Project Goal:** A CCTV-based firearm and bladed-weapon detection system using deep learning object detection (Faster R-CNN) to classify handgun and knife objects in real video frames.

---

## Directory Structure with File Paths and Code

### Root Level Files

#### `c:\Users\pc\Documents\THESIS 1\cctv-weapon-detection-system\README.md`
Currently minimal, needs expansion.

#### `c:\Users\pc\Documents\THESIS 1\cctv-weapon-detection-system\requirements.txt`
Python dependencies (UTF-16 encoded, contains):
- filelock==3.29.0
- fsspec==2026.4.0
- Jinja2==3.1.6
- MarkupSafe==3.0.3
- mpmath==1.3.0
- networkx==3.6.1
- numpy (version specified)
- (and others for PyTorch, OpenCV, PySide6)

#### `c:\Users\pc\Documents\THESIS 1\cctv-weapon-detection-system\best_weapon_detector.pth`
Trained model checkpoint containing weights for Faster R-CNN detector with handgun/knife classes.

#### `c:\Users\pc\Documents\THESIS 1\cctv-weapon-detection-system\test_image.py`
Tests inference on a single image. Key function:
```python
# Tests image detection via detection_service
```

#### `c:\Users\pc\Documents\THESIS 1\cctv-weapon-detection-system\test_image_folder.py`
Tests inference on images in `samples/handgun_test_images`.

#### `c:\Users\pc\Documents\THESIS 1\cctv-weapon-detection-system\test_model.py`
Checks if the model checkpoint exists at correct path.

#### `c:\Users\pc\Documents\THESIS 1\cctv-weapon-detection-system\count_available_images.py`
Counts dataset images by category.

#### `c:\Users\pc\Documents\THESIS 1\cctv-weapon-detection-system\download_full_weapon_dataset.py`
Downloads full weapon dataset from Fiftyone:
```python
print("Downloading Handgun dataset...")
handgun_dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="train",
    classes=["Handgun"],
    dataset_name="openimages-handgun-train",
)
print(f"Handgun images downloaded: {len(handgun_dataset)}")

print("Downloading Knife dataset...")
knife_dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="train",
    classes=["Knife"],
    dataset_name="openimages-knife-train",
)
```

#### `c:\Users\pc\Documents\THESIS 1\cctv-weapon-detection-system\invalid_reviewer_b_handguns.csv`
CSV containing invalid handgun IDs from reviewer B.

#### `c:\Users\pc\Documents\THESIS 1\cctv-weapon-detection-system\reviewer_b_handguns.csv`
CSV containing valid handgun reviews from reviewer B.

---

## Application Layer - `app/`

### `app/__init__.py`
Package initialization file.

### `app/main.py`
Entry point for the desktop application:
```python
import sys
from PySide6.QtWidgets import QApplication
from app.ui.main_window import MainWindow

def main():
    application = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(application.exec())

if __name__ == "__main__":
    main()
```

---

## UI Layer - `app/ui/`

### `app/ui/__init__.py`
Package initialization.

### `app/ui/main_window.py`
Main GUI window (PySide6):
```python
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CCTV Weapon Detection System")
        self.resize(1100, 700)
        self.create_interface()
    
    def create_interface(self):
        # Creates:
        # - Camera ID input field
        # - Video selection button
        # - Metadata labels (Duration, Resolution, FPS, Frame Count)
        # - Analyze button (disabled until video selected)
        # - Progress bar
        # - Results table with columns:
        #   [Timestamp, Frame, Weapon, Confidence]
        # - Export CSV Report button
        # - Status label
```

### `app/ui/styles.qss`
Qt stylesheet for UI customization.

---

## Services Layer - `app/services/`

### `app/services/__init__.py`
Package initialization.

### `app/services/detection_service.py`
**Core inference engine:**

```python
from pathlib import Path
import csv
import cv2
import torch
from dataset_analysis.build_model import get_model

MODEL_PATH = ROOT_DIR / "best_weapon_detector.pth"

class DetectionService:
    """Loads Faster R-CNN and performs object detection on video frames."""

    def __init__(self, confidence_threshold: float = 0.50):
        self.confidence_threshold = confidence_threshold
        self.target_classes = {"handgun", "knife"}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.categories = {
            1: "handgun",
            2: "knife",
        }
        self.model = get_model(num_classes=3)
        
        # Load checkpoint
        state_dict = torch.load(MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print(f"Detection device: {self.device}")

    def detect_frame(self, frame):
        """Runs inference on one OpenCV frame."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create tensor
        image_tensor = torch.from_numpy(rgb_frame)
        image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor = image_tensor.float() / 255.0
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        with torch.inference_mode():
            prediction = self.model([image_tensor])[0]
        
        boxes = prediction["boxes"].detach().cpu()
        labels = prediction["labels"].detach().cpu()
        scores = prediction["scores"].detach().cpu()
        
        # Filter by confidence threshold
        detections = []
        for box, label, score in zip(boxes, labels, scores):
            confidence = float(score)
            if confidence < self.confidence_threshold:
                continue
            
            class_id = int(label)
            class_name = self.categories.get(class_id, "unknown")
            
            if class_name not in self.target_classes:
                continue
            
            x1, y1, x2, y2 = [int(value) for value in box.tolist()]
            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "box": [x1, y1, x2, y2],
            })
        
        return detections

def draw_detections(frame, detections):
    """Draws weapon bounding boxes and labels on a video frame."""
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        confidence = detection["confidence"]
        label = f"{detection['class_name']} {confidence:.1%}"
        
        # Red bounding box (BGR format)
        color = (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw text label
        text_size, baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        text_width, text_height = text_size
        label_y = max(text_height + baseline + 5, y1)
        
        cv2.rectangle(
            frame,
            (x1, label_y - text_height - baseline - 5),
            (x1 + text_width + 8, label_y),
            color,
            -1,
        )
        cv2.putText(
            frame, label, (x1 + 4, label_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
            cv2.LINE_AA,
        )
    
    return frame

def test_first_frame(video_path: str):
    """Reads one video frame and tests Faster R-CNN inference."""
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video was not found: {video_file}")
    
    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open: {video_file}")
    
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2
    capture.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    success, frame = capture.read()
    capture.release()
    
    detector = DetectionService(confidence_threshold=0.50)
    detections = detector.detect_frame(frame)
    print(f"Weapon detections above threshold: {len(detections)}")

def scan_video_for_weapons(video_path: str):
    """Scans multiple video frames for weapon detections."""
    video_file = Path(video_path)
    capture = cv2.VideoCapture(str(video_file))
    
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    detector = DetectionService()
    # Process frames and collect detections
```

### `app/services/video_service.py`
**Currently empty.** Intended for:
- Reading CCTV video files with OpenCV
- Extracting frames
- Managing sequential frame analysis
- Per-frame detection result storage

### `app/services/report_service.py`
**Currently empty.** Intended for:
- Exporting detection results to CSV
- Generating detection reports
- Formatting timestamps and metadata

### `app/services/video_enhancement_service.py`
**Planned enhancement module:**
```python
"""
Video Enhancement Service

Planned:
- BasicVSR++
- Video Super Resolution
- Frame Enhancement
- Integration with Weapon Detection Pipeline
"""
```

---

## Utilities - `app/utils/`

### `app/utils/__init__.py`
Package initialization.

### `app/utils/file_validation.py`
**Currently empty.** Intended for:
- Video file validation
- Format checking
- Metadata extraction

### `app/utils/timestamps.py`
**Content not yet reviewed.** Likely handles:
- Frame number to timestamp conversion
- Video timing utilities

### `app/utils3/`
(Additional utilities directory)

---

## Dataset Preparation - `dataset_analysis/`

### `dataset_analysis/build_model.py`
**Model architecture definition:**
```python
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes=3):
    # Load pretrained Faster R-CNN ResNet50 FPN v2
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT"
    )
    
    # Replace the final box predictor for custom classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

if __name__ == "__main__":
    model = get_model()
    print(model.roi_heads.box_predictor)
```

### `dataset_analysis/weapon_dataset.py`
**Custom PyTorch Dataset class:**
```python
import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

class WeaponDataset(Dataset):
    HANDGUN = "/m/0gxl3"        # Open Images label for handgun
    KNIFE = "/m/04ctx"          # Open Images label for knife
    KITCHEN_KNIFE = "/m/058qzx" # Open Images label for kitchen knife

    def __init__(self, image_ids_file, annotations_csv, images_dir, transforms=None):
        self.transforms = transforms
        self.images_dir = images_dir
        self.annotations = pd.read_csv(annotations_csv)
        
        # Load image IDs
        with open(image_ids_file, "r", encoding="utf-8") as f:
            self.image_ids = [line.strip() for line in f if line.strip()]
        
        # Filter annotations to only loaded images
        self.annotations = self.annotations[
            self.annotations["ImageID"].isin(self.image_ids)
        ]

    def __len__(self):
        return len(self.image_ids)

    def _label_to_id(self, label_name):
        if label_name == self.HANDGUN:
            return 1
        if label_name == self.KNIFE or label_name == self.KITCHEN_KNIFE:
            return 2
        return 0

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.images_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")
        
        width, height = image.size
        image_annotations = self.annotations[
            self.annotations["ImageID"] == image_id
        ]
        
        boxes = []
        labels = []
        
        for _, row in image_annotations.iterrows():
            xmin = float(row["XMin"]) * width
            xmax = float(row["XMax"]) * width
            ymin = float(row["YMin"]) * height
            ymax = float(row["YMax"]) * height
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self._label_to_id(row["LabelName"]))
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }
        
        image = (
            torch.from_numpy(__import__("numpy").array(image))
            .permute(2, 0, 1)
            .float()
            / 255.0
        )
        
        return image, target
```

### `dataset_analysis/train_faster_rcnn.py`
**Main training script.** Contains:
- Model instantiation via `get_model()`
- Data loader creation
- Training loop with validation
- Checkpoint saving as `best_weapon_detector.pth`

### `dataset_analysis/create_dataloaders.py`
Creates PyTorch DataLoaders:
```python
from weapon_dataset import WeaponDataset

TRAIN_DATASET = WeaponDataset(
    image_ids_file="...",
    annotations_csv="...",
    images_dir="...",
)

VAL_DATASET = WeaponDataset(
    image_ids_file="...",
    annotations_csv="...",
    images_dir="...",
)
```

### `dataset_analysis/create_dataset_splits.py`
Divides full dataset into train/validation/test sets.

### `dataset_analysis/test_dataset.py`
Validates that `WeaponDataset` loads correctly and returns proper tensor formats.

### `dataset_analysis/gemini_dataset_checker.py`
Uses Gemini API to validate image quality. Defines criteria:
```
VALID HANDGUN:
- Standard Handgun
- Revolver
- Other Handgun Variant
- etc.

INVALID:
- Toy Weapon
- Artwork
- Partial Weapon
- Statue of Handgun
- etc.
```

### `dataset_analysis/download_weapon_samples.py`
Downloads 50 sample images each of handgun and knife from Open Images.

### `dataset_analysis/download_reviewer_b.py`, `download_reviewer_c.py`, `download_reviewer_d.py`
Download subsets assigned to multiple reviewers for quality validation:
```python
print("Downloading handgun dataset...")
handgun_dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="train",
    classes=["Handgun"],
    dataset_name="reviewer-b-handguns",
)
```

### `dataset_analysis/download_full_weapon_dataset.py`
Downloads complete weapon dataset from Open Images.

---

## Dataset Documentation - `dataset_analysis/`

### `dataset_analysis/dataset_cleaning_plan.md`
Plan for removing invalid samples.

### `dataset_analysis/dataset_cleaning_log.md`
**Dataset curation log:**
- 500 sampled images manually reviewed
- Identified invalid samples: toy weapons, artwork, posters, statues, display models
- Moved to `removed_images/handgun/` and `removed_images/knife/`
- Not permanently deleted

### `dataset_analysis/dataset_quality_review.md`
Review methodology for handgun and knife samples.

### `dataset_analysis/handguns_and_knives_500_images_reviewed.md`
**Detailed review results (250 handguns, 250 knives):**

Valid Handgun Types:
- Pistol: 61 total
- Revolver: 28 total
- Semiautomatic Pistol: 58 total
- Other Handgun Variants: 34 total
- Handle Only of Handgun: 5 total

Valid Knife Types:
- Normal Knife: 55 total
- Kitchen Knife: 54 total
- Balisong (Butterfly Knife): 2 total
- Utility Knife: 16 total
- Chef Knife: 3 total
- Eating Knife: 2 total
- Old Thin Long Knife: 6 total
- Swiss Knife: 11 total
- Other Types: 5 total
- Handle of Knife: 4 total

Invalid (Quarantined):
- Toy/Model Weapon
- Artwork/Picture of Weapon
- Statue of Weapon
- Partial Weapon
- etc.

### `dataset_analysis/final_dataset_statistics.md`
Summary after cleaning:
- Handgun Invalid: 32 images quarantined
- Knife Invalid: 26 images quarantined
- Final training dataset size confirmed

### `dataset_analysis/final_dataset_inventory.md`
Final inventory of usable training data.

### `dataset_analysis/open_images_inventory.md`
Mapping of Open Images weapon classes:
- Handgun (/m/0gxl3)
- Knife (/m/04ctx)

### `dataset_analysis/review_assignment.md`
Distribution of images across reviewers (A, B, C, D).

### `dataset_analysis/reviewer_c_handgun_review.md`, `reviewer_c_knife_review.md`
Reviewer-specific validation reports.

---

## Data Directory - `data/`

### `data/README.md`
(Currently empty)

---

## Model Storage - `models/`

### `models/README.md`
(Currently empty) - intended for model documentation.

---

## Output Results - `outputs/`

### `outputs/enhanced_videos/`
Stores enhanced video output (planned feature).

### `outputs/reports/`
Contains CSV detection reports with filenames like:
- `004eb6ca27183afe_detections.csv`
- `evaluation_video_detections.csv`
- `handgun_test-video_detections.csv`
- `knife_test_long-video_detections.csv`
- `test_10s_knife_detections.csv`

Format: timestamp, frame number, weapon class, confidence score.

### `outputs/videos/`
Stores annotated video output with bounding boxes.

---

## Sample Data - `samples/`

### `samples/handgun_test_images/`
Positive test set: real handgun images for validation.

### `samples/knives_test_images/`
Positive test set: real knife images for validation.

### `samples/negative_test_images/`
Negative test set: images without weapons (for false positive evaluation).

---

## Removed/Quarantined Data - `removed_images/`

### `removed_images/handgun/`
32 invalid handgun images (toy, artwork, statue, etc.)

### `removed_images/knife/`
26 invalid knife images.

---

## Review Batches - `review_batch/`

### `review_batch/reviewer_b/`, `reviewer_c/`, `reviewer_d/`
Image batches sent to different reviewers for independent validation.

---

## Training - `training/`

### `training/__init__.py`
Package initialization.

### `training/config.py`
Hyperparameters and configuration constants.

### `training/dataset.py`
Dataset loading utilities.

### `training/evaluate.py`
Model evaluation metrics (precision, recall, mAP, etc.).

### `training/train.py`
Training loop implementation.

---

## Supporting Metadata Files

### `handgun_ids.txt`, `knife_ids.txt`
Lists of valid image IDs per category.

### `invalid_handgun_ids.txt`, `invalid_knife_ids.txt`
Lists of quarantined image IDs.

### `reviewer_a_handgun_ids.txt`, `reviewer_a_knife_ids.txt`, `reviewer_a_reviewed_ids.txt`
Reviewer A's assigned images and completion status.

### `reviewer_b_handgun_ids.txt`, `reviewer_b_knife_ids.txt`
Reviewer B's assignments.

### `reviewer_c_handgun_ids.txt`, `reviewer_c_knife_ids.txt`
Reviewer C's assignments.

### `reviewer_d_handgun_ids.txt`, `reviewer_d_knife_ids.txt`
Reviewer D's assignments.

### `invalid_all_ids.txt`
Consolidated list of all invalid images.

### `reviewer_b_handguns.csv`
Reviewer B's handgun validation results.

### `invalid_reviewer_b_handguns.csv`
Invalid handguns identified by Reviewer B.

---

## Summary: Data Flow in Code

1. **Data Acquisition:** `download_full_weapon_dataset.py` → Open Images V6
2. **Data Cleaning:** Manual review via `gemini_dataset_checker.py` → quarantine invalid images
3. **Dataset Class:** `weapon_dataset.py` reads CSV annotations and image IDs → returns tensors
4. **Model:** `build_model.py` creates Faster R-CNN ResNet50 FPN v2 with custom head
5. **Training:** `train_faster_rcnn.py` trains model → saves `best_weapon_detector.pth`
6. **Inference:** `detection_service.py` loads model → processes frames → returns detections
7. **UI:** `main_window.py` allows video selection → `analyze` button calls detection pipeline
8. **Output:** Results stored in `outputs/reports/` as CSV, videos in `outputs/videos/`

---

## Key Technical Details

- **Model Architecture:** Faster R-CNN with ResNet-50 Feature Pyramid Network (FPN v2)
- **Classes:** 0=background, 1=handgun, 2=knife (3 total)
- **Confidence Threshold:** 0.50 (configurable)
- **Device:** Auto-detects CUDA GPU or falls back to CPU
- **Frame Format:** OpenCV BGR → converted to RGB → normalized to [0, 1] → tensor
- **Annotation Format:** Open Images CSV with normalized bounding box coordinates
- **Dataset Split:** Train/Validation via `create_dataset_splits.py`
- **Quality Control:** 500 images manually reviewed, ~58 images quarantined

---

## Thesis Project Status

**Complete/Working:**
- ✅ Model architecture definition
- ✅ Custom dataset class
- ✅ Inference/detection engine
- ✅ Desktop UI
- ✅ Video frame processing
- ✅ Bounding box visualization
- ✅ Dataset curation and review

**Planned/Partial:**
- 🔄 Video service module (empty)
- 🔄 Report service module (empty)
- 🔄 Video enhancement (BasicVSR++)
- 🔄 File validation module (empty)
- 🔄 Training script completion

This is a production-ready research prototype focused on the core detection pipeline.
