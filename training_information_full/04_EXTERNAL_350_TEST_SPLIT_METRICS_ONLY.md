# 04 — UNIFIED EXTERNAL 350 TEST SPLIT METRICS ONLY (MODELS 1 TO 13, ZERO DATA LEAKAGE)

**System Name:** FORENSIKADA (Multi-Camera CCTV Weapon Detection System)  
**Benchmark Directory:** `training/unified_external_benchmark/` (`test_ids.txt`)  
**Composition:** **350 Strictly Unseen Images** (`134 Handgun`, `116 Knife`, `100 Hard Negatives` including `60 Held Everyday Non-Weapon Confounders`)  
**Overlap with Any Training or Validation Set (Models 1–13):** **`0.0%` (`0` overlapping images)**  
**Evaluation Criteria:** $\text{IoU} \ge 0.50$, $\text{Confidence} \ge 0.50$  

---

## 1. Master Comparative Table — All 13 Models on Unified External 350 Benchmark

| Model Iteration | Checkpoint Filename | Precision | Recall | F1-Score | Handgun AP@0.50 | Knife AP@0.50 | Overall mAP@0.50 | TP | FP | FN | Clean Negatives (Out of 100) | Hard-Negative Specificity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1** | `best_weapon_detector.pth` | 18.91% | 19.05% | 18.98% | 06.72% | 17.76% | 12.24% | 52 | 223 | 221 | 43 / 100 | 43.00% |
| **Model 2** | `best_weapon_detector_retrained.pth` | 27.36% | 20.15% | 23.21% | 03.91% | 22.89% | 13.40% | 55 | 146 | 218 | 57 / 100 | 57.00% |
| **Model 3** | `best_weapon_detector_third_model.pth` | 34.34% | 24.91% | 28.87% | 03.91% | 32.90% | 18.41% | 68 | 130 | 205 | 62 / 100 | 62.00% |
| **Model 4** | `best_weapon_detector_fourth_model.pth` | 19.76% | 24.18% | 21.75% | 01.87% | 35.64% | 18.76% | 66 | 268 | 207 | 55 / 100 | 55.00% |
| **Model 5** | `best_weapon_detector_fifth_model.pth` | 61.98% | 59.71% | 60.82% | 73.16% | 36.95% | 55.05% | 163 | 100 | 110 | 73 / 100 | 73.00% |
| **Model 6** | `best_weapon_detector_sixth_model.pth` | 48.83% | 61.17% | 54.31% | 76.45% | 32.72% | 54.59% | 167 | 175 | 106 | 60 / 100 | 60.00% |
| **Model 7** | `best_weapon_detector_seventh_model.pth` | 45.11% | 60.81% | 51.79% | 73.29% | 30.21% | 51.75% | 166 | 202 | 107 | 60 / 100 | 60.00% |
| **Model 8** | `best_weapon_detector_eighth_model.pth` | 55.38% | 65.93% | 60.20% | 80.33% | 38.91% | 59.62% | 180 | 145 | 93 | 59 / 100 | 59.00% |
| **Model 9** | `best_weapon_detector_ninth_model.pth` | 67.14% | 51.65% | 58.39% | 81.27% | 12.35% | 46.81% | 141 | **69** | 132 | 82 / 100 | 82.00% |
| **Model 10** | `best_weapon_detector_tenth_model.pth` | **72.05%** | 67.03% | 69.45% | 81.60% | 43.17% | 62.38% | 183 | 71 | 90 | **84 / 100** | **84.00%** |
| **Model 11** | `best_weapon_detector_eleventh_model.pth` | 71.21% | 68.86% | 70.02% | 81.52% | 45.61% | 63.56% | 188 | 76 | 85 | 83 / 100 | 83.00% |
| **Model 12 (Calibrated, $\alpha=0.30$)** | `best_weapon_detector_twelfth_model.pth` | 69.61% | 72.16% | 70.86% | 82.19% | 48.85% | 65.52% | 197 | 86 | 76 | 82 / 100 | 82.00% |
| **Model 12 (Epoch 2, $\alpha=1.00$)** | `latest_weapon_detector_twelfth_model.pth` | 63.75% | 77.29% | 69.87% | 82.90% | 55.93% | 69.42% | 211 | 120 | 62 | 77 / 100 | 77.00% |
| **Model 13 (Calibrated, $\alpha=0.50$)** | **`best_weapon_detector_thirteenth_model.pth`** | 69.72% | **80.95%** | **74.92%** | **83.48%** | **66.16%** | **74.82%** | **221** | 96 | **52** | 81 / 100 | 81.00% |
| **Model 13 (Epoch 2, $\alpha=1.00$)** | `latest_weapon_detector_thirteenth_model.pth` | 70.39% | 78.39% | 74.18% | 76.94% | **70.03%** | 73.48% | 214 | 90 | 59 | 82 / 100 | 82.00% |

---

## 2. Per-Class External 350 Confusion & Error Breakdown (Models 9 to 13)

| Model Iteration | Handgun TP | Handgun FP | Handgun FN | Handgun AP@0.50 | Knife TP | Knife FP | Knife FN | Knife AP@0.50 | Total Weapons Detected (Out of 273 Boxes) | Total Weapons Missed (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 9 (Thesis 1 Final)** | 112 | 29 | 26 | 81.27% | 29 | 40 | 106 | 12.35% | 141 / 273 (51.65%) | 132 (48.35%) |
| **Model 10** | 112 | 22 | 26 | 81.60% | 71 | 49 | 64 | 43.17% | 183 / 273 (67.03%) | 90 (32.97%) |
| **Model 11** | 113 | 22 | 25 | 81.52% | 75 | 54 | 60 | 45.61% | 188 / 273 (68.86%) | 85 (31.14%) |
| **Model 12 ($\alpha=0.30$)** | 115 | 22 | 23 | 82.19% | 82 | 64 | 53 | 48.85% | 197 / 273 (72.16%) | 76 (27.84%) |
| **Model 13 ($\alpha=0.50$)** | **116** | **16** | **22** | **83.48%** | **105** | 80 | **30** | **66.16%** | **221 / 273 (80.95%)** | **52 (19.05%)** |

---

## 3. Key Takeaways for Chapter 4 Discussion (External 350)

1. **The Model 5 CCTV Pivot (`+42.81 pp` mAP jump):** Switching from web/macro photos (Models 1–4, `12.24%–18.76%` mAP) to real surveillance CCTV (USRT + OD-WeaponDetection in Model 5) immediately surged External 350 mAP to **`55.05%`**.
2. **Solving Model 9's Knife Truncation (`12.35% → 43.17% → 48.85% → 66.16%` Knife AP):** Restoring the 256px anchor stage and $\ln(1.95)$ box clamping in Model 10 recovered `+42` external knife detections, while full surveillance-POV expansion in Models 12–13 pushed external Knife AP to **`66.16%`** (`70.03%` raw Epoch 2) and reduced missed external knives from **`106 FN` (Model 9) down to `30 FN` (Model 13)**.
3. **Breaking the 80% Unseen External Recall Barrier:** Model 13 (`alpha = 0.50`) is the first model in project history to exceed 80% recall (**`80.95%`**, `221/273` weapons detected) and 74% mAP (**`74.82%`**) on the strictly unseen 350-image external benchmark.
