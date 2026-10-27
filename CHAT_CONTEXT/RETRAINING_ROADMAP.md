# Retraining Roadmap

## Thesis Title

Multi-Camera CCTV-Based Surveillance System for Post-Incident Crime-Related Object Detection with Forensic-Oriented Validation

---

# Current System State

## Detection Model

- Faster R-CNN
- ResNet50 FPN v2

Classes:

- Background
- Handgun
- Knife

Current checkpoint:

best_weapon_detector.pth

---

# Current Findings

## Knife Detection

Status:

✅ Functional

Experiment Results:

Original:
- 23 detections

Enhanced:
- 28 detections

Improvement:
- +5 detections
- +21.74%

---

## Handgun Detection

Status:

❌ False-positive prone

Observation:

The model frequently generates handgun detections on non-handgun regions.

Several handgun detections cover most of the frame:

Example:

Frame 38

Confidence:
73.51%

Bounding Box:

2,7,1920,1044

This suggests poor localization and poor CCTV generalization.

---

# Current Limitation

Current training dataset:

Open Images

Issues:

- Professional photography
- Product photographs
- Internet images
- Large domain gap with CCTV footage

Target environment:

- CCTV
- Surveillance cameras
- Low resolution
- Motion blur
- Compression artifacts
- Long-distance objects

Conclusion:

Dataset quality is likely the primary cause of current handgun false positives.

---

# Dataset Replacement Strategy

Adviser Recommendation:

Replace the current dataset rather than continuously adding datasets.

Goal:

Create a new primary training dataset.

---

# Dataset Requirements

## Handgun

Accept:

- Pistol
- Semi-automatic pistol
- Compact pistol
- Revolver
- Concealed handgun
- CCTV handgun imagery
- Real handgun imagery

Reject:

- Toy guns
- Nerf guns
- Airsoft guns
- Water guns
- Drawings
- Artwork
- Logos
- Video game screenshots

---

## Knife

Accept:

- Kitchen knife
- Utility knife
- Pocket knife
- Combat knife
- Hunting knife
- Chef knife
- Tactical knife
- Machete

Reject:

- Toy knives
- Plastic knives
- Drawings
- Cartoons
- Logos
- Scissors
- Screwdrivers

---

# Dataset Targets

Minimum:

Handgun:
- 2000 images

Knife:
- 2000 images

Preferred:

Handgun:
- 3000–4000 images

Knife:
- 3000–4000 images

---

# Dataset Cleaning Strategy

Validation categories:

KEEP

- Real weapon
- Correct class
- CCTV-relevant
- Identifiable object

REMOVE

- Toy weapon
- Artwork
- Video game content
- Wrong label
- Not visible
- Duplicate

---

# Dataset Split

Training:
80%

Validation:
20%

Testing:

Custom staged multi-camera CCTV dataset

The staged dataset should remain separate from training.

Purpose:

Final evaluation dataset.

---

# Planned Retraining Experiments

## Experiment A

Current Model

Dataset:

Open Images

Checkpoint:

best_weapon_detector.pth

Purpose:

Baseline

---

## Experiment B

Retrained Model

Dataset:

Replacement Dataset

Checkpoint:

best_weapon_detector_v2.pth

Purpose:

Performance comparison

---

# Planned Evaluation Metrics

Detection Metrics:

- Precision
- Recall
- F1-Score
- AP
- mAP@50
- mAP@50:95

Detection Quality:

- Detection Count
- Confidence Score
- False Positives
- False Negatives

Performance Metrics:

- Detection Runtime
- Enhancement Runtime
- Total Pipeline Runtime

---

# Planned Data Augmentation Study

Current Model:

No augmentation used.

Future Retraining:

Evaluate:

- Horizontal Flip
- Brightness Adjustment
- Contrast Adjustment
- Motion Blur
- Gaussian Blur
- Low-Light Simulation
- JPEG Compression Simulation

Purpose:

Improve CCTV generalization.

---

# Future Research Questions

1. Does replacing Open Images reduce handgun false positives?

2. Does CCTV-oriented data improve handgun detection performance?

3. Does BasicVSR++ still improve detection after retraining?

4. Does data augmentation improve CCTV weapon detection?

5. Does the retrained model outperform the current model?

---

# Success Criteria

The retrained model should:

- Reduce handgun false positives
- Improve precision
- Improve recall
- Improve mAP
- Improve CCTV generalization
- Maintain or improve knife detection performance

---

# Current Phase

Completed:

✅ Faster R-CNN Integration

✅ BasicVSR++ Integration

✅ WSL Integration

✅ Enhancement Pipeline

✅ End-to-End Testing

✅ Knife Experiment

✅ Handgun Experiment

✅ Benchmarking

Current Focus:

🔄 Dataset Selection

🔄 Dataset Cleaning

🔄 Dataset Replacement

Next Phase:

🔄 Retraining

🔄 Evaluation

🔄 Chapter 4 Final Results
