# 03 — STANDARDIZED EVALUATION METHODOLOGY ACROSS MODELS 1 TO 13 (SAME METRICS METHOD)

**System Name:** FORENSIKADA (Multi-Camera CCTV Weapon Detection System)  
**Primary Evaluation Script:** [evaluate_all_models_unified.py](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training/model_10/evaluate_all_models_unified.py) & [train_tenth_model.py](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training/model_10/train_tenth_model.py)  

---

## 1. Why Our Evaluation Methodology Is 100% Fair Across Models 1 to 13

In early Thesis 1 development (Models 1–7), each model iteration was originally evaluated on its own contemporaneous validation/test split. To eliminate any metric discrepancy across iterations, **all 13 model checkpoints (`best_weapon_detector.pth` through `best_weapon_detector_thirteenth_model.pth`) were re-evaluated through one single unified evaluation harness** under identical rules:

1. **Identical Operating Point:**
   - **Confidence Threshold ($\tau_{\text{conf}}$):** Fixed at **`0.50` (`50%`)** for all static and video ground-truth benchmarks (following Olmos et al., *Neurocomputing* 2018 & Ruiz-Santaquiteria et al., *IEEE Access* 2021).
   - **Intersection-over-Union Threshold ($\tau_{\text{IoU}}$):** Fixed at **`0.50`** (standard PASCAL VOC / COCO `AP@0.50` criterion).
2. **Identical Greedy Confidence-Sorted Box Matching:**
   - Within each image and for each weapon class $c \in \{\text{handgun}, \text{knife}\}$, predicted boxes with score $s \ge 0.50$ are sorted in descending order of confidence.
   - Each prediction is matched to the highest-IoU unmatched ground-truth box of the same class $c$.
   - If $\text{IoU} \ge 0.50$, the prediction is counted as **1 True Positive (`TP`)** and that ground-truth box is marked matched (preventing double-counting).
   - If $\text{IoU} < 0.50$ or no ground-truth box of class $c$ remains unmatched, the prediction is counted as **1 False Positive (`FP`)**.
   - Any ground-truth weapon box remaining unmatched after all predictions are processed is counted as **1 False Negative (`FN`)**.
3. **Identical Precision, Recall, F1-Score, and mAP@0.50 Formulas:**
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \qquad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \qquad \text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
   - **Per-Class Average Precision ($\text{AP}_{50,c}$):** Computed by integrating the monotonically interpolated precision-recall curve across all predictions of class $c$ (ranked by confidence score down to $0.05$) against total ground-truth positives $N_{\text{GT},c}$.
   - **Mean Average Precision ($\text{mAP}@0.50$):**
     $$\text{mAP}@0.50 = \frac{\text{AP}_{50,\text{handgun}} + \text{AP}_{50,\text{knife}}}{2}$$
4. **Identical Hard-Negative Rejection Specificity Rate:**
   - Evaluated on pure Class-0 non-weapon images ($N_{\text{neg}}$ images containing zero weapons, including people holding everyday look-alikes):
     $$\text{Hard-Negative Specificity} = \frac{N_{\text{clean}}}{N_{\text{neg}}} = \frac{\text{Number of negative images with } 0 \text{ predictions at } s \ge 0.50}{\text{Total negative images}}$$

---

## 2. Mathematical Zero-Leakage Verification Matrix

Every dataset preparation script ([prepare_twelfth_model_dataset.py](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training/model_12/prepare_twelfth_model_dataset.py), [prepare_thirteenth_model_dataset.py](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training/model_13/prepare_thirteenth_model_dataset.py)) executes hard Python `assert` checks verifying empty set intersections before training begins:

$$\text{Train}_{\text{M1..M13}} \cap \text{Ext350} = \emptyset \quad (0\text{ overlapping files})$$
$$\text{Train}_{\text{M1..M13}} \cap \text{CCTV185} = \emptyset \quad (0\text{ overlapping files})$$
$$\text{Train}_{\text{M8..M13}} \cap \text{Test279} = \emptyset \quad (0\text{ overlapping files})$$
$$\text{Train}_{\text{M11..M13}} \cap \text{Val650} = \emptyset \quad (0\text{ overlapping files})$$

| Evaluation Split | Scope of Fairness | Zero-Leakage Status | Notes for Thesis Manuscript |
| :--- | :--- | :---: | :--- |
| **Unified External Benchmark (`Ext 350`)** | **Fair across ALL Models 1–13** | **0.0% leakage (`0/350`)** | Primary benchmark for comparing all 13 models on equal footing. |
| **Strictly Unseen Internal CCTV (`CCTV 185`)** | **Fair across ALL Models 1–13** | **0.0% leakage (`0/185`)** | Primary surveillance-domain benchmark across all 13 models. |
| **Frozen Internal Test Set (`Test 279`)** | **Fair across Models 8–13** | **0.0% leakage (`0/279`)** | Frozen internal test split shared identically by Models 8, 9, 10, 11, 12, and 13. |
| **Held Non-Weapon Grid (`18 Cells`)** | **Fair across Models 8–13** | **0.0% leakage (`0/18`)** | Fixed 18-cell held everyday object stress test. |
| **Staged Tapo Validation (`44 Frames`)** | **Fair across Models 11–13** | **0.0% leakage (`0/44`)** | Held-out validation split from Batch 1 Tapo `CAM 01`/`02` recordings. |
| **16-Stream Real CCTV Video GT (`2,538 Frames`)** | **Fair across Models 9–13** | **0.0% leakage (`0/16` videos)** | Frame-by-frame bounding-box ground truth across 6 Original/Enhanced weapon pairs + 4 No-Weapon videos. |
| **Nov 20 Household Sample (`40 Frames`)** | **In-Domain Check (M12 vs M13)** | *In-domain for M13* | M13 mined frames from Scenes `#01–#06`; report as an in-domain household look-alike suppression check alongside zero-leakage `Staged 44` & `Ext 350`. |
