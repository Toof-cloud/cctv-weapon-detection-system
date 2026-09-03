# PROJECT STATUS

Last Updated: September 2026

---

# Thesis Title

Multi-Camera CCTV-Based Surveillance System for Post-Incident Crime-Related Object Detection with Forensic-Oriented Validation

---

# Current Project Phase

SYSTEM IMPLEMENTATION: ✅ COMPLETED

RESEARCH AND EVALUATION: 🔄 ONGOING

CURRENT FOCUS:
Dataset Replacement → Retraining → Final Evaluation

---

# Current System Status

## Video Processing

✅ Video Upload

✅ Video Validation

✅ Metadata Extraction

✅ Frame Extraction

✅ Video Reconstruction

## BasicVSR++

✅ WSL Integration

✅ Ubuntu Environment Setup

✅ MMagic Installation

✅ BasicVSR++ Integration

✅ REDS4 Checkpoint Integration

✅ Enhanced Frame Generation

✅ Successful Enhancement Pipeline

### Verified Results

Successfully processed:

- 150 original frames
- 150 enhanced frames

No frame loss observed.

---

## Faster R-CNN Detector

✅ Faster R-CNN ResNet50-FPN-v2 Integrated

✅ Custom Weapon Detection Model

✅ Handgun Class Implemented

✅ Knife Class Implemented

✅ Bounding Box Generation

✅ Confidence Score Generation

✅ CSV Logging

✅ Annotated Frame Generation

✅ Annotated Video Generation

---

# Current Workflow

Video
↓
Video Validation
↓
Metadata Extraction
↓
Frame Extraction
↓
BasicVSR++
↓
Enhanced Frames
↓
Faster R-CNN
↓
Detection Results
↓
CSV Report
↓
Annotated Frames
↓
Annotated Video
↓
Forensic Reporting

---

# Experimental Results

## Knife Experiment

Original:

23 detections

Enhanced:

28 detections

Improvement:

+5 detections

+21.74%

### Manual Review

Validated Detections:

- Frame 62
- Frame 63

False Positive:

- Frame 66

Questionable:

- Frame 11
- Frame 12

---

## Handgun Experiment

Original:

147 detections

Enhanced:

157 detections

Observed Improvement:

+6.80%

However:

Manual review indicated that many additional handgun detections were false positives.

---

# Current Performance Assessment

## Knife Detection

Status:

ACCEPTABLE

Observation:

BasicVSR++ contributed additional valid knife detections.

---

## Handgun Detection

Status:

NEEDS IMPROVEMENT

Observation:

Frequent false positives observed in CCTV-style footage.

Likely Cause:

Training dataset does not sufficiently represent surveillance environments.

---

# Benchmark Results

Video:

145 Frames

BasicVSR++ Enhancement:

185.33 seconds

Faster R-CNN Detection:

29.72 seconds

Total Pipeline Runtime:

215.04 seconds

Processing Time Per Frame:

1.4831 seconds/frame

---

# Current Datasets

## Original Training Dataset

Open Images

Classes:

- Handgun
- Knife

Status:

To be replaced.

Reason:

Poor CCTV generalization.

---

## Staged Dataset

Custom Multi-Camera CCTV Dataset

Status:

Collected

Contains:

- Prop Handguns
- Prop Knives

Approximately:

25 Videos

Primary Planned Use:

Evaluation Dataset

---

# Current Metrics Available

✅ Detection Count

✅ Confidence Scores

✅ Bounding Box Coordinates

✅ Timestamps

✅ Runtime Benchmarking

✅ CSV Reports

✅ Annotated Videos

---

# Metrics Planned for Final Evaluation

🔄 Precision

🔄 Recall

🔄 F1-Score

🔄 IoU

🔄 Average Precision (AP)

🔄 Mean Average Precision (mAP@50)

🔄 False Positive Analysis

---

# Data Augmentation Status

Current Model:

❌ No Data Augmentation Used

Future Experiment:

✅ Horizontal Flip

✅ Brightness Adjustment

✅ Contrast Adjustment

✅ Random Scaling

✅ Gaussian Blur

✅ Motion Blur

✅ Noise Injection

✅ Low-Light Simulation

---

# Current Limitations

1. Handgun false positives remain high.

2. Dataset replacement has not yet been completed.

3. Retraining has not started.

4. Precision, Recall, F1, AP, and mAP have not yet been computed.

5. Temporal Consistency is not yet implemented.

6. Multi-Camera Corroboration is not yet implemented.

---

# Next Steps

1. Finalize replacement handgun and knife datasets.

2. Validate annotations and dataset quality.

3. Create final train/validation split.

4. Retrain Faster R-CNN.

5. Evaluate Precision, Recall, F1, AP, and mAP.

6. Compare new model against current model.

7. Re-evaluate BasicVSR++ using retrained detector.

8. Complete Chapter 4 experimental evaluation.

---

# Current Project Completion Estimate

System Development:
≈ 90-95%

Research and Evaluation:
≈ 70-75%

Overall Thesis Progress:
≈ 75%