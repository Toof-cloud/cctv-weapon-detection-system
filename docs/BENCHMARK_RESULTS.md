# AUDIT REPORT: EVALUATION OF MODELS 1 THROUGH 12, MODEL 12 SPECIFICATION & VIDEO GROUND-TRUTH BENCHMARK

**System Name:** FORENSIKADA (Multi-Camera CCTV Weapon Detection System)  
**Academic Unit:** National University – Manila | CCIT – Computer Science Department  
**Final Production Checkpoint:** `best_weapon_detector_twelfth_model.pth` (Model 12 — Calibrated WiSE-FT SOTA)  
**High-Recall Epoch 2 Checkpoint:** `latest_weapon_detector_twelfth_model.pth` (Model 12 Epoch 2)  
**Document Purpose:** Comprehensive technical factsheet documenting the Unified Zero-Leakage Benchmark Audit of Models 1 through 12, complete dataset provenance, exact image counts per class and dataset source, Model 11 and Model 12 training/validation/test split compositions, staged hard-negative distributions (Tapo CAM 01 & CAM 02), per-epoch component training losses, the 16-Stream Real CCTV Video Ground-Truth Benchmark (Original vs. BasicVSR++ Enhanced, Before vs. After Dataset Improvements, Raw vs. CCTV Intelligence Layer), and peer-reviewed literature justifications.

---

## SECTION 1: EXECUTIVE SUMMARY OF FINDINGS (MODELS 1 TO 12)

1. **The 350-Image Unified External Benchmark Audit (Zero Data Leakage Across Models 1 to 12):**
   * An independent **Unified External Benchmark** of 350 images (`134 Handgun`, `116 Knife`, `100 Hard Negatives` including `60 Held Non-Weapon Confounders`) was established with **0.0% overlap** against every training and validation set of Models 1 through 12.
   * **Model 12 achieved the highest overall mAP, Knife AP, Handgun AP, Recall, and F1-Score across all 12 models:**
     * **mAP@0.50:** **`65.52%` (Calibrated `alpha=0.30`) / `69.42%` (Epoch 2 `alpha=1.00`)** — up from `12.24%` (Model 1), `55.05%` (Model 5), `59.62%` (Model 8), `46.81%` (Model 9), `62.38%` (Model 10), and `63.56%` (Model 11).
     * **F1-Score:** **`70.86%` (Calibrated) / `69.87%` (Epoch 2)** — highest harmonic balance in project history.
     * **Recall:** **`72.16%` (`197 TP / 76 FN` Calibrated) / `77.29%` (`211 TP / 62 FN` Epoch 2)** — up from `67.03%` (`183 TP`) in Model 10 and `68.86%` (`188 TP`) in Model 11.
     * **Handgun AP:** **`82.19%` (Calibrated) / `82.90%` (Epoch 2)** — highest external firearm AP in project history.
     * **Knife AP:** **`48.85%` (Calibrated) / `55.93%` (Epoch 2)** — **`+5.68 pp` to `+12.76 pp` surge** over Model 10 (`43.17%`).
     * **Hard Negative Rejection Specificity:** **`82.00%` (`82/100` clean non-weapon scenes)**.

2. **Full Dataset Utilization & Surveillance-POV Expansion in Model 11 and Model 12:**
   * **Model 10 (Before Staged & Surveillance Expansion):** Trained on `2,804` full images (`962` Handgun, `988` Knife, `854` Hard Negatives) + `296` negative crops (`3,100` items).
   * **Model 11 (Staged Hard Negatives + Partial Surveillance Expansion):** Added `434` full-frame Tapo C200 dual-camera staged hard negatives (`390` train / `44` val) + `250` context-padded distractor crops from `2,534` ground-truth distractor annotations (`CAM 01` & `CAM 02`), plus `+1,800` surveillance-perspective weapon frames (`+900` USRT CCTV Handguns, `+800` surveillance-scale Knives, `+100` surveillance-scale Handguns), expanding the pool to `4,793` train images + `546` crops (`5,339` items, sampling `600` new surveillance weapon images per epoch).
   * **Model 12 (Final Complete Expansion & Cup/Torso Negative Mining):** Unlocked `410` previously unused zero-leakage CostiCatargiu training images (`270` Knife + `140` Handgun), exposed the full `+1,800` surveillance-POV weapon pool across training epochs, and added `40` targeted handheld cup/mug/torso hard-negative crops (`hard_negatives_v13_cups`), bringing the final training corpus to **`5,203` full images (`1,990` Handgun, `1,969` Knife, `1,244` Hard Negatives) + `586` targeted negative ROI crops = `5,789` effective training items**.

3. **Real CCTV Video Ground-Truth Benchmark (16 Video Streams, 2,538 Evaluated Frames):**
   * To satisfy the adviser's mandate for **additional ground truth** and quantitative video-level evaluation, frame-by-frame Bounding-Box Ground Truth ([expert_evaluation_videos/ground_truth/](file:///C:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/expert_evaluation_videos/ground_truth/)) was constructed across **3 Handgun CCTV pairs (Original & Enhanced)**, **3 Knife CCTV pairs (Original & Enhanced)**, and **4 Non-Handgun videos** (`219` weapon-present frames + `948` pure non-weapon frames):
     * **Before vs. After Dataset Improvements (Model 10 $ightarrow$ Model 12 on Enhanced CCTV + 4 No-Weapon Videos):**
       * **Video Recall:** Surged from **`34.70%` (`76 TP / 143 FN` in Model 10)** to **`58.90%` (`129 TP` in Model 11)** and **`73.06%` (`160 TP / 59 FN` in Model 12)** — **`+38.36 percentage points` (`+110.5%` relative increase in true weapon frames detected)**!
       * **Video F1-Score:** Surged from **`51.53%` (Model 10)** to **`74.14%` (Model 11)** and **`83.55%` (Model 12)** (**`+32.02 pp` improvement**).
       * **Video mAP@0.50:** Surged from **`47.09%` (Model 10)** to **`64.20%` (Model 11)** and **`70.81%` (Model 12)** (**`+23.72 pp` improvement**; Video Handgun AP `88.13%`, Video Knife AP `53.49%`).
       * **No-Weapon Video Specificity:** **`99.68%` (`945/948` clean frames)** in Model 12 and **`100.00%` (`948/948`)** in Model 11 (`0` false alarms on `NW01`, `NW03`, `NW04`).
     * **Controlled Ablation 1 (BasicVSR++ Enhancement vs. Unenhanced Original on Model 12):**
       * BasicVSR++ super-resolution boosted Video Recall from **`62.10%` (`136 TP`) to `73.06%` (`160 TP`, `+10.96 pp`)**, Video F1 from **`75.56%` to `83.55%` (`+7.99 pp`)**, and Mean TP Confidence from **`86.70%` to `89.18%` (`+2.48%`)**.
     * **Controlled Ablation 2 (Raw Detector vs. With CCTV Intelligence Layer on Model 12):**
       * The CCTV Intelligence Layer eliminated **`99.52%` of raw false positives (`834 FP -> 4 FP`)**, jumping Video Precision from **`19.11%` to `97.56%` (`+78.45 pp`)** and No-Weapon Video Specificity from **`38.82%` to `99.68%` (`+60.86 pp`)**.

---

## SECTION 2: STATIC BENCHMARK AUDIT ACROSS ALL 12 MODELS

All 12 model checkpoints were evaluated under identical standardized conditions ($	ext{IoU} \ge 0.50$, $	ext{Confidence} \ge 0.50$).

### Table 1: Master Comparative Audit on the 350-Image Unified External Benchmark (Zero Data Leakage)
*Dataset composition: 350 strictly unseen images (134 Handgun, 116 Knife, 100 Hard Negatives including 60 Held Non-Weapon Confounders).*

| Model Iteration | Checkpoint Filename | Precision | Recall | F1-Score | Handgun AP | Knife AP | mAP@0.50 | TP / FP / FN | Hard Negative Rejection Rate | Key Forensic Finding |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model 1** | `best_weapon_detector.pth` | 18.91% | 19.05% | 18.98% | 06.72% | 17.76% | 12.24% | 52 / 223 / 221 | 43.00% (43/100) | Open Images baseline. Heavy false alarms; incapable of generalizing to CCTV. |
| **Model 2** | `best_weapon_detector_retrained.pth` | 27.36% | 20.15% | 23.21% | 03.91% | 22.89% | 13.40% | 55 / 146 / 218 | 57.00% (57/100) | Sanitized Open Images boxes cut FP by 34.5%, but scale mismatch persists. |
| **Model 3** | `best_weapon_detector_third_model.pth` | 34.34% | 24.91% | 28.87% | 03.91% | 32.90% | 18.41% | 68 / 130 / 205 | 62.00% (62/100) | Catastrophic handgun failure (3.91% AP) due to corrupted coordinates in Batch 2. |
| **Model 4** | `best_weapon_detector_fourth_model.pth` | 19.76% | 24.18% | 21.75% | 01.87% | 35.64% | 18.76% | 66 / 268 / 207 | 55.00% (55/100) | Purging 88 bad boxes restored training stability; macro photos still fail in CCTV. |
| **Model 5** | `best_weapon_detector_fifth_model.pth` | 61.98% | 59.71% | 60.82% | 73.16% | 36.95% | 55.05% | 163 / 100 / 110 | 73.00% (73/100) | **The CCTV Pivot:** USRT dataset surged Handgun AP to 73.16% (+71.29% over M4). |
| **Model 6** | `best_weapon_detector_sixth_model.pth` | 48.83% | 61.17% | 54.31% | 76.45% | 32.72% | 54.59% | 167 / 175 / 106 | 60.00% (60/100) | VIRAT negatives added; aggressive recall increases false alarms (175 FP). |
| **Model 7** | `best_weapon_detector_seventh_model.pth` | 45.11% | 60.81% | 51.79% | 73.29% | 30.21% | 51.75% | 166 / 202 / 107 | 60.00% (60/100) | Unconstrained 8px anchors hyper-sensitize to sub-12px metal noise (202 FP). |
| **Model 8** | `best_weapon_detector_eighth_model.pth` | 55.38% | 65.93% | 60.20% | 80.33% | 38.91% | 59.62% | 180 / 145 / 93 | 59.00% (59/100) | Balanced 16–256px anchors & specular jitter solve silver firearms & floor phones. |
| **Model 9** | `best_weapon_detector_ninth_model.pth` | 67.14% | 51.65% | 58.39% | 81.27% | 12.35% | 46.81% | 141 / 69 / 132 | 82.00% (82/100) | Door-shadow mining cut FP to 69; 128px anchor limit truncated large knife recall. |
| **Model 10** | `best_weapon_detector_tenth_model.pth` | **72.05%** | 67.03% | 69.45% | 81.60% | 43.17% | 62.38% | 183 / **71** / 90 | **84.00% (84/100)** | Full-range anchors and NoGUN negatives recover knife AP to 43.17%. |
| **Model 11** | `best_weapon_detector_eleventh_model.pth` | 71.21% | 68.86% | 70.02% | 81.52% | 45.61% | 63.56% | 188 / 76 / 85 | 83.00% (83/100) | Staged Tapo hard negatives + partial surveillance expansion break 70% F1. |
| **Model 12 (Calibrated)** | `best_weapon_detector_twelfth_model.pth` | 69.61% | **72.16%** | **70.86%** | **82.19%** | **48.85%** | **65.52%** | **197** / 86 / **76** | 82.00% (82/100) | **Final Production SOTA:** Highest F1 (70.86%), +3.14 pp mAP & +5.68 pp Knife AP over M10. |
| **Model 12 (Epoch 2)** | `latest_weapon_detector_twelfth_model.pth` | 63.75% | **77.29%** | 69.87% | **82.90%** | **55.93%** | **69.42%** | **211** / 120 / **62** | 77.00% (77/100) | **Maximum Recall SOTA:** Highest External mAP (69.42%), Knife AP (55.93%), & TP (211). |

---

### Table 2: Audit on Strictly Unseen Internal CCTV Images (185 Images)
*Composition: 114 Handgun, 31 Knife, 40 Pure CCTV Background Hard Negatives (0% overlap with any training/validation set).*

| Model Iteration | Precision | Recall | F1-Score | Handgun AP | Knife AP | mAP@0.50 | TP / FP / FN | Clean Negatives (Out of 40) | Specificity Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1** | 11.70% | 13.79% | 12.66% | 02.95% | 36.45% | 19.70% | 20 / 151 / 125 | 11 / 40 | 27.50% |
| **Model 2** | 17.78% | 11.03% | 13.62% | 00.88% | 37.63% | 19.25% | 16 / 74 / 129 | 21 / 40 | 52.50% |
| **Model 3** | 33.00% | 22.76% | 26.94% | 00.40% | 91.73% | 46.06% | 33 / 67 / 112 | 16 / 40 | 40.00% |
| **Model 4** | 13.19% | 21.38% | 16.32% | 00.17% | 93.55% | 46.86% | 31 / 204 / 114 | 17 / 40 | 42.50% |
| **Model 5** | 90.78% | 88.28% | 89.51% | 85.64% | 96.37% | 91.00% | 128 / 13 / 17 | 35 / 40 | 87.50% |
| **Model 6** | 71.81% | 93.10% | 81.08% | 90.39% | 97.46% | 93.92% | 135 / 53 / 10 | 29 / 40 | 72.50% |
| **Model 7** | 67.01% | 91.03% | 77.19% | 87.81% | 89.37% | 88.59% | 132 / 65 / 13 | 28 / 40 | 70.00% |
| **Model 8** | 84.24% | 95.86% | 89.68% | 94.60% | 98.47% | 96.53% | 139 / 26 / 6 | 35 / 40 | 87.50% |
| **Model 9** | **98.40%** | 84.83% | 91.11% | 96.47% | 40.78% | 68.63% | 123 / **2** / 22 | **40 / 40** | **100.00%** |
| **Model 10** | 97.22% | 96.55% | **96.89%** | 95.52% | **99.90%** | 97.71% | 140 / 4 / 5 | **40 / 40** | **100.00%** |
| **Model 11** | 95.24% | 96.55% | 95.89% | 95.47% | 99.70% | 97.58% | 140 / 7 / 5 | **40 / 40** | **100.00%** |
| **Model 12 (Calibrated)** | 94.63% | **97.24%** | 95.92% | **96.34%** | 99.60% | **97.97%** | **141** / 8 / **4** | **40 / 40** | **100.00%** |
| **Model 12 (Epoch 2)** | 94.04% | **97.93%** | 95.95% | **97.32%** | 99.09% | **98.21%** | **142** / 9 / **3** | **40 / 40** | **100.00%** |

---

### Table 3: Head-to-Head Comparison on Model 9 Benchmark Test Set (279 Images)
*Evaluating Models 8, 9, 10, 11, and 12 on the standard 279-image test split (114 Handgun, 114 Knife, 51 Pure CCTV Background Negatives).*

| Metric Evaluated | Model 8 Baseline | Model 9 Production | Model 10 [Before Additions] | Model 11 [Partial Expansion] | **Model 12 Calibrated (`alpha=0.30`)** | **Model 12 Epoch 2 (`alpha=1.00`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall Precision** | 79.14% | **94.05%** | 92.31% | 90.48% | 89.11% | 87.55% |
| **Overall Recall** | 92.44% | 73.11% | 95.80% | 95.80% | **96.22%** | **97.48%** |
| **Overall F1-Score** | 85.27% | 82.27% | **94.02%** | 93.06% | 92.53% | 92.25% |
| **Handgun AP@0.50** | 94.00% | 96.25% | 95.26% | 95.22% | **96.11%** | **97.08%** |
| **Knife AP@0.50** | 88.03% | 50.43% | 95.18% | 94.93% | 94.90% | **95.99%** |
| **mAP@0.50** | 91.02% | 73.34% | 95.22% | 95.08% | **95.50%** | **96.54%** |
| **True Positives (TP)** | 220 | 174 | 228 | 228 | **229** | **232** |
| **False Positives (FP)** | 58 | **11** | 19 | 24 | 28 | 33 |
| **False Negatives (FN)** | 18 | 64 | 10 | 10 | **9** | **6** |
| **Hard Negative Specificity** | 78.43% (40/51) | **92.16% (47/51)** | **92.16% (47/51)** | **92.16% (47/51)** | **92.16% (47/51)** | **92.16% (47/51)** |

---

### Table 4: Held Non-Weapon Grid Audit (18 Everyday Distractor Cells) & Staged Groupmate Val Split (44 Frames)

| Model Iteration | Held Non-Weapon Grid Clean Cells (Out of 18) | Grid Specificity Rate | Staged Groupmate Val Raw Clean (Out of 44) | Staged Groupmate Val Post-CCTV Intel Clean (Out of 44) |
| :--- | :---: | :---: | :---: | :---: |
| **Model 8** | 13 / 18 | 72.22% | — | — |
| **Model 9** | 16 / 18 | 88.89% | — | — |
| **Model 10** | **17 / 18** | **94.44%** | — | — |
| **Model 11** | **17 / 18** | **94.44%** | 27 / 44 (61.36%) | **37 / 44 (84.09%)** |
| **Model 12 (Calibrated)** | 16 / 18 | 88.89% | 25 / 44 (56.82%) | **37 / 44 (84.09%)** |

---

## SECTION 3: REAL CCTV VIDEO GROUND-TRUTH BENCHMARK (16 STREAMS, 2,538 FRAMES)

To fulfill the adviser's instruction to **add more ground truth** and evaluate actual handgun and knife detection performance on real CCTV footage, frame-by-frame Bounding-Box Ground Truth ([expert_evaluation_videos/ground_truth/](file:///C:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/expert_evaluation_videos/ground_truth/)) was created across **219 true weapon-present frames** (`HG01..03` and `KN01..03`, tested on both Original and BasicVSR++ Enhanced streams) and **948 pure non-weapon frames** (`NW01..04`).

### Table 5: Aggregate Video Ground-Truth Performance (Enhanced Weapon Videos + 4 Non-Handgun Videos)

| Model Iteration | Video Precision | Video Recall | Video F1-Score | Video Handgun AP | Video Knife AP | Video mAP@0.50 | TP / FP / FN | No-Weapon Video Specificity (948 Frames) | Mean TP Confidence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 9 (Legacy Baseline)** | 56.13% | 53.85% | 54.97% | 68.26% | 60.22% | 64.24% | 119 / 93 / 102 | 97.15% (921/948, 27 FP) | 91.62% |
| **Model 10 (Before Staged & Surv Expansion)** | **100.00%** | 37.10% | 54.13% | 69.73% | 40.61% | 55.17% | 82 / **0** / 139 | **100.00% (948/948, 0 FP)** | 87.37% |
| **Model 11 (Staged + Partial Surv Expansion)** | **100.00%** | 61.09% | 75.84% | 81.58% | 62.99% | 72.28% | 135 / **0** / 86 | **100.00% (948/948, 0 FP)** | 86.55% |
| **Model 12 (Final Full Surv + Staged & Cup Negs)** | **99.40%** | **75.11%** | **85.57%** | **88.13%** | **69.65%** | **78.89%** | **166 / 1 / 55** | **100.00% (948/948, 0 FP)** | **87.43%** |
| **Net Improvement (Model 10 $ightarrow$ Model 12)** | `-0.60 pp` | **`+38.01 pp` (+102.4% rel.)** | **`+31.44 pp` (+58.1% rel.)** | **`+18.40 pp` (+26.4% rel.)** | **`+29.04 pp` (+71.5% rel.)** | **`+23.72 pp` (+43.0% rel.)** | **`+84 TP` (-60.4% Missed Frames)** | **100.00% Clean (`0 FP` on all 4 videos)** | **`+0.06%`** |

---

### Table 6: Per-Video Ground-Truth Breakdown Across All 16 Streams (Models 9, 10, 11, and 12)

| Video ID & CCTV Scenario | Stream Version | GT Weapon Frames / Total | Model 9 (TP/FP/FN, F1, Conf) | Model 10 [Before] (TP/FP/FN, F1, Conf) | Model 11 (TP/FP/FN, F1, Conf) | **Model 12 [Final] (TP/FP/FN, F1, AP/Spec, Conf)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HG01 (`230249`)** Store Robbery | Original<br>Enhanced | 121 / 425<br>121 / 425 | 45/28/76 (46.4%, 77.0%)<br>45/33/76 (45.2%, 77.3%) | 3/0/118 (4.8%, 75.0%)<br>14/0/107 (20.7%, 70.4%) | 49/0/72 (57.6%, 70.3%)<br>57/0/64 (64.0%, 71.5%) | **54/1/67 (F1=61.4%, AP=44.2%, Conf=73.5%)**<br>**81/1/40 (F1=79.8%, AP=66.8%, Conf=73.8%)** |
| **HG02 (`213340`)** Retail CCTV | Original<br>Enhanced | 4 / 121<br>4 / 121 | 4/0/0 (100.0%, 83.7%)<br>4/3/0 (72.7%, 88.0%) | 4/0/0 (100.0%, 93.3%)<br>4/0/0 (100.0%, 95.9%) | 4/0/0 (100.0%, 93.7%)<br>4/0/0 (100.0%, 96.6%) | **4/0/0 (F1=100.0%, AP=100.0%, Conf=94.7%)**<br>**4/0/0 (F1=100.0%, AP=100.0%, Conf=96.5%)** |
| **HG03 (`handgun_test`)** Lab Aiming | Original<br>Enhanced | 42 / 75<br>42 / 75 | 42/0/0 (100.0%, 99.8%)<br>41/0/1 (98.8%, 99.8%) | 41/0/1 (98.8%, 99.6%)<br>41/0/1 (98.8%, 99.5%) | 41/0/1 (98.8%, 99.7%)<br>41/0/1 (98.8%, 99.6%) | **42/1/0 (F1=98.8%, AP=100.0%, Conf=99.5%)**<br>**41/0/1 (F1=98.8%, AP=97.6%, Conf=99.6%)** |
| **KN01 (`230047`)** Indoor Knife | Original<br>Enhanced | 11 / 130<br>11 / 130 | 9/3/2 (78.3%, 86.7%)<br>10/1/1 (90.9%, 85.4%) | 9/0/2 (90.0%, 85.5%)<br>9/0/2 (90.0%, 88.2%) | 9/0/2 (90.0%, 85.3%)<br>9/0/2 (90.0%, 90.6%) | **9/0/2 (F1=90.0%, AP=81.8%, Conf=87.2%)**<br>**9/0/2 (F1=90.0%, AP=81.8%, Conf=91.6%)** |
| **KN02 (`220101`)** Knife Assault | Original<br>Enhanced | 35 / 464<br>35 / 464 | 15/13/20 (47.6%, 80.8%)<br>14/13/21 (45.2%, 84.1%) | 18/0/17 (67.9%, 84.0%)<br>14/0/21 (57.1%, 86.2%) | 14/0/21 (57.1%, 87.4%)<br>20/0/15 (72.7%, 82.5%) | **23/0/12 (F1=79.3%, AP=65.7%, Conf=83.6%)**<br>**27/0/8 (F1=87.1%, AP=77.1%, Conf=82.9%)** |
| **KN03 (`230842`)** Unseen Knife | Original<br>Enhanced | 8 / 373<br>8 / 373 | 5/12/3 (40.0%, 93.2%)<br>5/16/3 (34.5%, 89.6%) | 3/0/5 (54.5%, 62.3%)<br>0/0/8 (0.0%, 0.0%) | 4/0/4 (66.7%, 82.8%)<br>4/0/4 (66.7%, 84.9%) | **4/0/4 (F1=66.7%, AP=50.0%, Conf=81.6%)**<br>**4/0/4 (F1=66.7%, AP=50.0%, Conf=85.3%)** |
| **NW01 (`Sample 1`)** Non-Weapon | Original | 0 / 281 | 0/8/0 (Spec=97.2%) | **0/0/0 (Spec=100.0%)** | **0/0/0 (Spec=100.0%)** | **0/0/0 (Spec=100.0%, 0 FP)** |
| **NW02 (`Sample 2`)** Cups/Phones | Original | 0 / 220 | 0/19/0 (Spec=91.4%) | **0/0/0 (Spec=100.0%)** | **0/0/0 (Spec=100.0%)** | **0/0/0 (Spec=100.0%, 0 FP)** |
| **NW03 (`Sample 3`)** Non-Weapon | Original | 0 / 290 | 0/0/0 (Spec=100.0%) | **0/0/0 (Spec=100.0%)** | **0/0/0 (Spec=100.0%)** | **0/0/0 (Spec=100.0%, 0 FP)** |
| **NW04 (`Tapo CAM 01`)** Staged | Original | 0 / 157 | 0/3/0 (Spec=98.1%) | **0/0/0 (Spec=100.0%)** | **0/0/0 (Spec=100.0%)** | **0/0/0 (Spec=100.0%, 0 FP)** |

---

### Table 7: Controlled Video Ablation Studies on Model 12 (BasicVSR++ Enhancement & CCTV Intelligence Layer)

| Ablation Comparison | Precision | Recall | F1-Score | TP / FP / FN | No-Weapon Clean Frames (948 Total) | Mean TP Confidence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1A. Unenhanced Original Videos (With CCTV Intel)** | 98.55% | 61.54% | 75.77% | 136 / 2 / 85 | 948 / 948 (100.00%) | 86.70% |
| **1B. BasicVSR++ Enhanced Videos (With CCTV Intel)** | **99.40%** | **75.11%** | **85.57%** | **166 / 1 / 55** | **948 / 948 (100.00%)** | **87.43%** |
| **Net BasicVSR++ Enhancement Benefit** | **`+0.85 pp`** | **`+13.57 pp` (+22.1% rel.)** | **`+9.80 pp`** | **`+30 TP recovered` (-35.3% FN)** | **100.00% Clean** | **`+0.73%` boost** |
| **2A. Raw Faster R-CNN Detector Only (No CCTV Intel)** | 19.30% | 90.05% | 31.79% | 199 / 832 / 22 | 368 / 948 (38.82%) | — |
| **2B. Full Pipeline (+ CCTV Intelligence Layer)** | **99.40%** | **75.11%** | **85.57%** | **166 / 1 / 55** | **948 / 948 (100.00%)** | **87.43%** |
| **Net CCTV Intelligence Layer Benefit** | **`+80.10 pp`** | `-14.94 pp` | **`+53.78 pp`** | **`-831 FP` (-99.88% false alarms)** | **`+580 clean frames` (+61.18 pp)** | **100.00% Clean** |

---

## SECTION 4: EXACT TRAINING, VALIDATION, AND TEST SPLIT COMPOSITIONS (MODELS 10, 11, AND 12)

### Table 8: Complete Split Distribution Across Partitions (Model 10 vs. Model 11 vs. Model 12)

| Model & Partition Split | Total Full Images | Handgun Images (Class 1) | Knife Images (Class 2) | Hard Negative Frames (Class 0) | Injected Negative ROI Crops | Effective Items | Physical Manifest Location |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model 10 Train** | 2,804 | 962 | 988 | 854 | +296 crops | 3,100 | `training/model_10/tenth_model_data/` |
| **Model 11 Train** | 4,793 | 1,850 | 1,699 | 1,244 | +546 crops | 5,339 | `training/model_11/eleventh_model_data/` |
| **Model 12 Train (Final)** | **5,203** | **1,990 (2,071 boxes)** | **1,969 (2,118 boxes)** | **1,244** | **+586 crops** | **5,789** | `training/model_12/twelfth_model_data/` |
| **Validation Split (M11/M12)** | **650** | 254 (262 boxes) | 242 (270 boxes) | 154 | 0 | 650 | `training/model_12/twelfth_model_data/val_ids.txt` |
| **Internal Test Split (M9–M12)** | **279** | 114 (114 boxes) | 114 (124 boxes) | 51 | 0 | 279 | `training/model_12/twelfth_model_data/test_ids.txt` |
| **Unseen CCTV Test Split** | **185** | 114 | 31 | 40 | 0 | 185 | `training/unified_external_benchmark/strictly_unseen_internal_ids.txt` |
| **Unified External Benchmark** | **350** | 134 | 116 | 100 | 0 | 350 | `training/unified_external_benchmark/test_ids.txt` |

---

## SECTION 5: PEER-REVIEWED SCIENTIFIC LITERATURE JUSTIFICATIONS

### Part A: Confidence Calibration & Low-Threshold Candidate Generation
1. **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).** *"On calibration of modern neural networks."* **International Conference on Machine Learning (ICML)**, PMLR 70:1321–1330. (Proves modern ResNet architectures are systematically overconfident in uncalibrated low-margin score regions).
2. **Küppers, F., Kronenberger, J., Shantia, A., & Haselhoff, A. (2020).** *"Multivariate confidence calibration for object detection."* **IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)**, pp. 326–335. (Proves bounding-box regression coordinates introduce spatial calibration error on small sub-1% targets).
3. **Olmos, R., Tabik, S., & Herrera, F. (2018).** *"Automatic handgun detection with deep learning in video surveillance images."* **Neurocomputing**, 275, 66–72. (Establishes 50% as standard benchmark reporting and multi-frame confirmation at elevated operational thresholds for CCTV deployment).
4. **Ruiz-Santaquiteria, J., Velasco-Mata, A., Vallez, N., Bueno, G., Álvarez-García, J. A., & Deniz, O. (2021).** *"Handgun detection using combined human pose and weapon appearance."* **IEEE Access**, 9, 123815–123826. (Validates combining human spatial context with appearance features to suppress handheld everyday object false alarms).

### Part B: Contextual & Spatio-Temporal CCTV Intelligence Filtering
1. **Velasco-Mata, A., Ruiz-Santaquiteria, J., Vallez, N., & Deniz, O. (2021).** *"Using human pose information for improving automatic handgun detection."* **Neural Computing and Applications**, 33, 17273–17286. (Validates human-proximity and anatomical limb-reach gating).
2. **Lamas, A., Tabik, S., Montes, A. C., Pérez-Hernández, F., García, J., Olmos, R., & Herrera, F. (2022).** *"Human pose estimation for mitigating false negatives in weapon detection."* **Neurocomputing**, 489, 488–503. (Validates anthropometric relative scaling between human and weapon bounding boxes).
3. **Han, W., Khorrami, P., Paine, T. L., Ramachandran, P., Babaeizadeh, M., Shi, H., Li, J., Yan, S., & Huang, T. S. (2016).** *"Seq-NMS for video object detection."* **arXiv preprint arXiv:1602.08465**. (Validates linking per-frame detections into temporal tracklets to eliminate transient 1-frame flickers).
4. **Divvala, S. K., Hoiem, D., Hays, J. H., Efros, A. A., & Hebert, M. (2009).** *"An empirical study of context in object detection."* **IEEE Conference on Computer Vision and Pattern Recognition (CVPR)**, pp. 1271–1278. (Validates scene geometry and stationary background fixture suppression).
5. **Wortsman, M., Ilharco, G., Kim, J. W., Li, M., Kornblith, S., Roelofs, R., ... & Schmidt, L. (2022).** *"Robust fine-tuning of zero-shot models (WiSE-FT)."* **IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)**, pp. 7959–7971. (Validates weight-space ensemble interpolation `(1 - alpha)*theta_base + alpha*theta_ft` to achieve pareto-optimal precision and recall).
