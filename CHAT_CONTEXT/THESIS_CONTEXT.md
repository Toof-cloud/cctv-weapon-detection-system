# THESIS CONTEXT

---

# Thesis Title

Multi-Camera CCTV-Based Surveillance System for Post-Incident Crime-Related Object Detection with Forensic-Oriented Validation

---

# Problem Statement

Crime-related objects such as handguns and knives are often difficult to identify in CCTV footage due to:

- Low resolution
- Motion blur
- Compression artifacts
- Poor lighting
- Long-distance viewing angles

These limitations negatively affect object detection performance and forensic investigations.

The study proposes the use of video enhancement and object detection to improve post-incident CCTV analysis.

---

# Research Objectives

1. Develop a CCTV-based object detection system capable of detecting handguns and knives.

2. Integrate BasicVSR++ video enhancement with Faster R-CNN.

3. Improve the visibility of crime-related objects in CCTV footage.

4. Generate forensic-oriented reports for post-incident investigations.

5. Evaluate the impact of enhancement on weapon detection performance.

---

# Proposed System Architecture

Pre-recorded CCTV Footage
↓
Video Preparation
↓
Frame Extraction
↓
BasicVSR++ Enhancement
↓
Enhanced Frames
↓
Faster R-CNN Detection
↓
Forensic-Oriented Validation
↓
Structured Output and Reporting
↓
Human Analyst Review

---

# Current Implementation Architecture

User Uploads CCTV Video
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
CSV Report
↓
Annotated Frames
↓
Annotated Video

---

# BasicVSR++ Implementation

Environment:

- WSL2 Ubuntu
- MMagic
- MMCV

Checkpoint:

basicvsr_plusplus_reds4.pth

Purpose:

- Restore detail
- Improve sharpness
- Improve edge visibility
- Improve weapon visibility

---

# Faster R-CNN Implementation

Detector:

Faster R-CNN

Backbone:

ResNet50-FPN-v2

Classes:

0 = Background

1 = Handgun

2 = Knife

---

# Current Dataset Situation

Original Dataset:

Open Images

Current Issue:

Handgun detections produce frequent false positives.

Decision:

Dataset replacement recommended by thesis adviser.

Target:

Handgun Dataset:
3000+ Images

Knife Dataset:
3000+ Images

Annotation Requirement:

Bounding Box Annotations

---

# Planned Dataset Split

Training:

80%

Validation:

20%

Testing:

Custom Staged CCTV Dataset

---

# Data Preprocessing

Applied During Training:

- Image Loading
- RGB Conversion
- Tensor Conversion
- Normalization
- Bounding Box Parsing

---

# Data Augmentation

Current Model:

No augmentation used.

Future Retraining:

- Horizontal Flip
- Brightness Adjustment
- Contrast Adjustment
- Random Scaling
- Motion Blur
- Gaussian Blur
- Low-Light Simulation

---

# Forensic-Oriented Validation

The proposed system includes:

1. Confidence Threshold Validation

2. IoU Validation

3. Temporal Consistency

4. Multi-Camera Corroboration

5. Human Analyst Review

---

# Structured Forensic Report

## Case Information

- Case ID
- Report ID
- Report Date
- Analyst Information

## Source Information

- Video Filename
- Camera ID
- Resolution
- FPS
- Duration

## Detection Evidence

- Frame Number
- Timestamp
- Object Class
- Confidence Score
- Bounding Box

## Validation Results

- Confidence Status
- IoU Status
- Temporal Consistency Status
- Multi-Camera Status

## Detection Summary

- Total Detections
- Average Confidence
- Confirmed Detections
- Rejected Detections

## Supporting Evidence

- Annotated Frames
- Annotated Video
- CSV Detection Reports

---

# Evaluation Metrics

Primary Metrics:

- Precision
- Recall
- F1-Score
- IoU
- AP (Average Precision)
- mAP@50

Additional Metrics:

- Detection Count
- Confidence Score
- False Positives
- False Negatives

Performance Metrics:

- Enhancement Runtime
- Detection Runtime
- Total Processing Time
- Processing Time Per Frame

---

# Experimental Findings

Knife Detection:

Original:
23 detections

Enhanced:
28 detections

Improvement:
+21.74%

---

Handgun Detection:

Original:
147 detections

Enhanced:
157 detections

Result:

Additional detections were largely false positives.

Conclusion:

Dataset quality and CCTV generalization remain the primary challenge.

---

# Current Research Direction

Immediate Priority:

1. Dataset Replacement
2. Dataset Cleaning
3. Retraining
4. Evaluation
5. BasicVSR++ Reassessment

Goal:

Reduce handgun false positives while maintaining or improving knife detection performance.

---

# Future Work

- Temporal Consistency Implementation
- Multi-Camera Corroboration
- Advanced Forensic Reports
- Weapon Tracking
- Cross-Camera Validation
- Real-Time Optimization