# FORENSIKADA Master Dataset Catalog

This directory provides the organized surveillance weapon detection dataset repository used for training, evaluating, and validating Faster R-CNN Models 1 through 9 and Model V11.

---

## Directory Structure & Dataset Overview

```text
datasets/
├── 01_Handgun_USRT_CCTV/               # Primary Handgun CCTV Dataset (1,140 images)
├── 02_Knife_OD_WeaponDetection/        # Primary Knife Dataset (1,135 images)
├── 03_CCTV_Hard_Negatives/             # Background Rejection Negative Frames (1,041 total)
│   ├── 01_VIRAT_Surveillance_Frames/   # Public surveillance walkways, stairs, and parking (160 images)
│   ├── 02_Pedestrian_No_Gun_Frames/    # Walking pedestrians with empty hands (100 images)
│   ├── 03_Model7_Noise_Negatives/      # Railings, bolts, doors, empty hands (265 images)
│   ├── 04_Model8_Distractor_Negatives/ # Horizontal phones, clean rooms, stairwells (257 images)
│   ├── 05_Model9_Commercial_Negatives/ # POS pinpads, beverage bottles, retail counters (148 images)
│   └── 06_Model11_Targeted_Negatives/  # Cast door shadows, ponytails, jacket creases (111 images)
├── 04_Dataset_Splits_By_Model/         # Train / Validation / Test ID splits and CSVs
│   ├── Model_3_Splits/
│   ├── Model_4_Splits/
│   ├── Model_5_Splits/
│   ├── Model_6_Splits/
│   ├── Model_7_Splits/
│   ├── Model_8_Splits/
│   └── Model_9_Splits/
└── 05_Early_Prototypes_Archive/        # Pre-CCTV Baseline Exploration Datasets
    ├── Open_Images_V6_FiftyOne/        # Models 1 & 2 Open Images V6 Subset (1,591 images)
    ├── Handgun_Batch1_Armas/           # Roboflow ARMAS Handgun Subset (548 images)
    └── Handgun_Batch2_HuggingFace/     # Hugging Face Weapon Dataset (598 images)
```

---

## Detailed Dataset Sourcing & Attribution

| Folder Name | Primary Source | Total Images | Target Class | Forensic Role in Thesis |
| :--- | :--- | :---: | :---: | :--- |
| **`01_Handgun_USRT_CCTV/`** | USRT Action Recognition & Firearm Dataset (*Mendeley Data DOI: 10.17632/bbzpxhd22j.2*) | 1,140 | Handgun (Class 1) | Solves CCTV distance & scale mismatch. Firearms realistically occupy ~0.35% of frame. |
| **`02_Knife_OD_WeaponDetection/`** | OD-WeaponDetection: Knife Detection (*Univ. of Granada / Dataset Ninja*) | 1,135 | Knife (Class 2) | Multi-angle bladed weapon localization across indoor/outdoor settings. |
| **`03_CCTV_Hard_Negatives/`** | VIRAT Video Dataset 2.0 (*DARPA/Kitware*) + USRT No_Gun + Real Surveillance Distractors | 1,041 | Class 0 (Background) | Forces Region Proposal Network (RPN) to reject phones, wallets, bottles, and shadows. |
| **`04_Dataset_Splits_By_Model/`** | Internal Research Splits | — | Annotations & Splits | Enforces strict 80% Train / 10% Val / 10% Held-Out Test partitioning per model iteration. |
| **`05_Early_Prototypes_Archive/`** | Open Images V6 (FiftyOne) + ARMAS + Hugging Face | 2,737 | Historical Archive | Preserved historical datasets for Models 1 & 2 (Open Images) and Model 3/4 failure diagnosis. |

---

## Model 9 / SOTA Split Summary

* **Total Images:** 2,768 images (1,128 Handgun, 1,135 Knife, 505 CCTV Negatives)
* **Training Set (80%):** 2,214 images (902 Handgun, 908 Knife, 404 Negatives)
* **Validation Set (10%):** 275 images (113 Handgun, 113 Knife, 49 Negatives)
* **Held-Out Test Set (10%):** 279 images (113 Handgun, 114 Knife, 52 Negatives; 51 zero-box negatives)
* **Evaluation Video Benchmarks:** 4 baseline multi-scenario surveillance videos + 10 arbitrary unseen CCTV clips (strictly separate from image pool).
