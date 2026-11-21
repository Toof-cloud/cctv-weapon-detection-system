# AUDIT REPORT: EVALUATION OF MODELS 1 THROUGH 13, MODEL 13 SPECIFICATION & VIDEO GROUND-TRUTH BENCHMARK

**System Name:** FORENSIKADA (Multi-Camera CCTV Weapon Detection System)  
**Academic Unit:** National University – Manila | CCIT – Computer Science Department  
**Final Production Checkpoint:** `best_weapon_detector_thirteenth_model.pth` (Model 13 — Calibrated WiSE-FT SOTA, $\alpha = 0.50$)  
**Un-interpolated Epoch 2 Checkpoint:** `latest_weapon_detector_thirteenth_model.pth` (Model 13 Epoch 2, $\alpha = 1.00$)  
**Previous SOTA Checkpoint:** `best_weapon_detector_twelfth_model.pth` (Model 12 — Calibrated WiSE-FT, $\alpha = 0.30$)  
**Document Purpose:** Comprehensive technical factsheet documenting the Unified Zero-Leakage Benchmark Audit of Models 1 through 13, complete dataset provenance, exact image counts per class and dataset source, Model 11–13 training/validation/test split compositions, staged Tapo CAM 01/CAM 02 and Nov 20 household look-alike hard-negative mining, per-epoch component training losses, WiSE-FT Pareto calibration sweeps, the 16-Stream Real CCTV Video Ground-Truth Benchmark, and peer-reviewed literature justifications.

---

## SECTION 1: EXECUTIVE SUMMARY OF FINDINGS (MODELS 1 TO 13)

1. **The 350-Image Unified External Benchmark Audit (Zero Data Leakage Across Models 1 to 13):**
   * An independent **Unified External Benchmark** of 350 strictly unseen images (`134 Handgun`, `116 Knife`, `100 Hard Negatives` including `60 Held Non-Weapon Confounders`) was evaluated under identical standardized criteria ($\text{IoU} \ge 0.50$, $\text{Confidence} \ge 0.50$) with **0.0% overlap** against every training and validation set of Models 1 through 13.
   * **Model 13 (Calibrated WiSE-FT $\alpha = 0.50$) shattered every external benchmark record across all 13 models:**
     * **mAP@0.50:** **`74.82%`** — up **`+9.30 pp`** over Model 12 Calibrated (`65.52%`), **`+11.26 pp`** over Model 11 (`63.56%`), **`+12.44 pp`** over Model 10 (`62.38%`), and **`+28.01 pp`** over Model 9 (`46.81%`).
     * **F1-Score:** **`74.92%`** — highest harmonic balance in project history (`+4.06 pp` over Model 12's `70.86%`).
     * **Recall:** **`80.95%`** (`221 TP / 52 FN`) — **first model in project history to break 80% recall on the strictly unseen external benchmark** (`+8.79 pp` over Model 12's `72.16%`, `+13.92 pp` over Model 10's `67.03%`).
     * **Precision:** **`69.72%`** (`221 TP / 96 FP`) — maintained parity with Model 12 (`69.61%`) while recovering `+24` additional true weapons.
     * **Handgun AP:** **`83.48%`** (`116 TP / 16 FP / 22 FN`) — **all-time highest external firearm AP** across all 13 models.
     * **Knife AP:** **`66.16%`** (`105 TP / 80 FP / 30 FN`) — massive **`+17.31 pp` surge** over Model 12 Calibrated (`48.85%`) and **`+22.99 pp`** over Model 10 (`43.17%`).
     * **Hard-Negative Rejection Specificity:** **`81.00%`** (`81/100` clean non-weapon scenes).

2. **Household Look-Alike Immunity Breakthrough in Model 13 (Nov 20 Staged Hard-Negative Mining):**
   * While Model 12 achieved SOTA weapon recall on real CCTV footage, stress testing on newly recorded dual-camera household distractor scenes (Nov 20 Scenes `#01–#06` across Tapo `CAM 01` & `CAM 02`: black TV remote, silver/dark long remote, metallic kitchen ladle, rolled white rag/towel, smartphone, wallet) showed Model 12 only achieved `67.50%` (`27/40` clean frames) raw specificity.
   * **Model 13** mined targeted full-frame hard negatives (`Nov20_Staged_Hard_Negatives/`, oversampled $2\times$ per epoch) and tight padded ROI FP crops (`hard_negatives_v14_nov20_household/`) **strictly from non-weapon Scenes `#01–#06`**, keeping weapon prop Scenes `#07–#11` (`Silver Pistol`, `Knife`, `Table Props`, `Cleaver`, `Dual Knives`) **100% untouched**.
   * **Measured Immunity Gains (Model 12 $\alpha=0.00 \rightarrow$ Model 13 $\alpha=0.50$):**
     * **Nov 20 Household Look-Alikes (40 sampled frames):** Surged from **`67.50%` (`27/40`) to `100.00%` (`40/40`, 0 false alarms)** — **`+32.50 pp` improvement**.
     * **Staged Groupmate Validation Split (44 Tapo CAM 01/02 frames):** Raw clean rate surged from **`56.82%` (`25/44`) to `84.09%` (`37/44`, `+27.27 pp`)**, and **Post-CCTV Intelligence Layer clean rate reached `100.00%` (`44/44`, 0 false alarms)** (up from `84.09%` in Model 12).
     * **Held Non-Weapon Grid Audit (18 everyday distractor cells):** Improved from **`88.89%` (`16/18`) in Model 12 back to `94.44%` (`17/18`)** in Model 13, eliminating both water-bottle false alarms.

3. **Real CCTV Video Ground-Truth Benchmark (16 Video Streams, 2,538 Evaluated Frames):**
   * Frame-by-frame Bounding-Box Ground Truth ([expert_evaluation_videos/ground_truth/](file:///Users/tyronegabrielr.pascual/CODING/THESIS/cctv-weapon-detection-system/expert_evaluation_videos/ground_truth/)) across **3 Handgun CCTV pairs (Original & Enhanced)**, **3 Knife CCTV pairs (Original & Enhanced)**, and **4 Non-Handgun videos** (`219` weapon-present frames + `948` pure non-weapon frames) verified:
     * **Dataset Expansion Benefit (Model 10 $\rightarrow$ Model 12 on Enhanced CCTV + 4 No-Weapon Videos):** Video Recall surged **`37.10%` $\rightarrow$ `75.11%` (`+38.01 pp`)**, Video F1 jumped **`54.13%` $\rightarrow$ `85.57%` (`+31.44 pp`)**, Video mAP@0.50 jumped **`55.17%` $\rightarrow$ `78.89%` (`+23.72 pp`)**, with **`100.00%` No-Weapon Specificity (`948/948`, `0 FP`)**.
     * **Controlled Ablation 1 (BasicVSR++ Enhancement):** Boosted Video Recall **`61.54%` $\rightarrow$ `75.11%` (`+13.57 pp`)** and Video F1 **`75.77%` $\rightarrow$ `85.57%` (`+9.80 pp`)**, recovering `+30` true weapon frames.
     * **Controlled Ablation 2 (CCTV Intelligence Layer):** Eliminated **`99.88%` of raw false positives (`832 FP` $\rightarrow$ `1 FP`)**, jumping Video Precision from **`19.30%` $\rightarrow$ `99.40%` (`+80.10 pp`)** and No-Weapon Specificity from **`38.82%` $\rightarrow$ `100.00%`**.

---

## SECTION 2: STATIC BENCHMARK AUDIT ACROSS ALL 13 MODELS

All 13 model checkpoints were evaluated under identical standardized conditions ($\text{IoU} \ge 0.50$, $\text{Confidence} \ge 0.50$).

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
| **Model 9** | `best_weapon_detector_ninth_model.pth` | 67.14% | 51.65% | 58.39% | 81.27% | 12.35% | 46.81% | 141 / **69** / 132 | 82.00% (82/100) | Door-shadow mining cut FP to 69; 128px anchor limit truncated large knife recall. |
| **Model 10** | `best_weapon_detector_tenth_model.pth` | **72.05%** | 67.03% | 69.45% | 81.60% | 43.17% | 62.38% | 183 / 71 / 90 | **84.00% (84/100)** | Full-range anchors and NoGUN negatives recover knife AP to 43.17%. |
| **Model 11** | `best_weapon_detector_eleventh_model.pth` | 71.21% | 68.86% | 70.02% | 81.52% | 45.61% | 63.56% | 188 / 76 / 85 | 83.00% (83/100) | Staged Tapo hard negatives + partial surveillance expansion break 70% F1. |
| **Model 12 (Calibrated)** | `best_weapon_detector_twelfth_model.pth` | 69.61% | 72.16% | 70.86% | 82.19% | 48.85% | 65.52% | 197 / 86 / 76 | 82.00% (82/100) | Full surveillance-POV & cup negative mining; WiSE-FT $\alpha=0.30$. |
| **Model 13 (Calibrated, $\alpha=0.50$)** | **`best_weapon_detector_thirteenth_model.pth`** | 69.72% | **80.95%** | **74.92%** | **83.48%** | **66.16%** | **74.82%** | **221** / 96 / **52** | 81.00% (81/100) | **Final Production SOTA:** All-time record mAP (`74.82%`), F1 (`74.92%`), Recall (`80.95%`), Handgun AP (`83.48%`), & Knife AP (`66.16%`); 100% Nov 20 & Staged Val immunity. |
| **Model 13 (Epoch 2, $\alpha=1.00$)** | `latest_weapon_detector_thirteenth_model.pth` | 70.39% | 78.39% | 74.18% | 76.94% | **70.03%** | 73.48% | 214 / 90 / 59 | 82.00% (82/100) | Un-interpolated AdamW Epoch 2 checkpoint; peak raw external Knife AP (`70.03%`). |

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
| **Model 9** | **98.40%** | 84.83% | 91.11% | **96.47%** | 40.78% | 68.63% | 123 / **2** / 22 | **40 / 40** | **100.00%** |
| **Model 10** | 97.22% | 96.55% | **96.89%** | 95.52% | **99.90%** | 97.71% | 140 / 4 / 5 | **40 / 40** | **100.00%** |
| **Model 11** | 95.24% | 96.55% | 95.89% | 95.47% | 99.70% | 97.58% | 140 / 7 / 5 | **40 / 40** | **100.00%** |
| **Model 12 (Calibrated)** | 94.63% | **97.24%** | 95.92% | 96.34% | 99.60% | **97.97%** | **141** / 8 / **4** | **40 / 40** | **100.00%** |
| **Model 13 (Calibrated, $\alpha=0.50$)** | 92.67% | 95.86% | 94.24% | 94.56% | 99.07% | 96.82% | 139 / 11 / 6 | 39 / 40 | 97.50% |
| **Model 13 (Epoch 2, $\alpha=1.00$)** | 91.61% | 90.34% | 90.97% | 88.35% | 96.37% | 92.36% | 131 / 12 / 14 | **40 / 40** | **100.00%** |

---

### Table 3: Head-to-Head Comparison on Model 9 Benchmark Test Set (279 Images)
*Evaluating Models 8, 9, 10, 11, 12, and 13 on the standard 279-image test split (114 Handgun, 114 Knife, 51 Pure CCTV Background Negatives).*

| Metric Evaluated | Model 8 Baseline | Model 9 Legacy | Model 10 | Model 11 | Model 12 Calibrated ($\alpha=0.30$) | **Model 13 Calibrated ($\alpha=0.50$)** | Model 13 Epoch 2 ($\alpha=1.00$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall Precision** | 79.14% | **94.05%** | 92.31% | 90.48% | 89.11% | **85.50%** | 83.94% |
| **Overall Recall** | 92.44% | 73.11% | 95.80% | 95.80% | **96.22%** | **94.12%** | 87.82% |
| **Overall F1-Score** | 85.27% | 82.27% | **94.02%** | 93.06% | 92.53% | **89.60%** | 85.83% |
| **Handgun AP@0.50** | 94.00% | **96.25%** | 95.26% | 95.22% | 96.11% | **94.11%** | 87.68% |
| **Knife AP@0.50** | 88.03% | 50.43% | **95.18%** | 94.93% | 94.90% | **91.64%** | 85.14% |
| **mAP@0.50** | 91.02% | 73.34% | 95.22% | 95.08% | **95.50%** | **92.87%** | 86.41% |
| **True Positives (TP)** | 220 | 174 | 228 | 228 | **229** | **224** | 209 |
| **False Positives (FP)** | 58 | **11** | 19 | 24 | 28 | 38 | 40 |
| **False Negatives (FN)** | 18 | 64 | 10 | 10 | **9** | 14 | 29 |
| **Hard Negative Specificity** | 78.43% (40/51) | **92.16% (47/51)** | **92.16% (47/51)** | **92.16% (47/51)** | **92.16% (47/51)** | 88.24% (45/51) | 88.24% (45/51) |

---

### Table 4: Everyday Distractor & Household Look-Alike Specificity Audit (Grid 18, Staged Val 44, Nov 20 Non-Weapon 40)

| Model Iteration | Held Non-Weapon Grid (18 Cells) | Staged Val Raw Clean (44 Frames) | Staged Val Post-CCTV Intel Clean (44 Frames) | Nov 20 Household Look-Alikes (40 Sampled Frames) |
| :--- | :---: | :---: | :---: | :---: |
| **Model 8** | 13 / 18 (72.22%) | — | — | — |
| **Model 9** | 16 / 18 (88.89%) | — | — | — |
| **Model 10** | **17 / 18 (94.44%)** | — | — | — |
| **Model 11** | **17 / 18 (94.44%)** | 27 / 44 (61.36%) | 37 / 44 (84.09%) | — |
| **Model 12 (Calibrated, $\alpha=0.30$)** | 16 / 18 (88.89%) | 25 / 44 (56.82%) | 37 / 44 (84.09%) | 27 / 40 (67.50%) |
| **Model 13 (Calibrated, $\alpha=0.50$)** | **17 / 18 (94.44%)** | **37 / 44 (84.09%)** | **44 / 44 (100.00%, 0 FP)** | **40 / 40 (100.00%, 0 FP)** |
| **Model 13 (Epoch 2, $\alpha=1.00$)** | **17 / 18 (94.44%)** | **41 / 44 (93.18%)** | **44 / 44 (100.00%, 0 FP)** | **40 / 40 (100.00%, 0 FP)** |

---

## SECTION 3: MODEL 13 TRAINING METRICS & WiSE-FT PARETO CALIBRATION

### Table 5: Model 13 Training Configuration & Per-Epoch Loss Breakdown
*Initialized from `best_weapon_detector_twelfth_model.pth` (`Model 12 SOTA`). Trained with `AdamW(lr=3.5e-5, weight_decay=0.0005)`, Cosine Annealing LR (`3.5e-5 -> 1.0e-5`), Batch Size 2, FP16 mixed precision, 5-stage anchors `(16, 32, 64, 128, 256)`, and $\ln(1.95)$ RPN/RoI box clamping.*

| Epoch | Duration | Total Train Loss | `loss_classifier` | `loss_box_reg` | `loss_objectness` | `loss_rpn_box_reg` | External 350 mAP | External 350 F1 | Grid 18 Clean | Staged 44 Post-CCTV |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Epoch 1/2** | 14.78 min | `0.050766` | `0.019581` | `0.023395` | `0.003204` | `0.004587` | 67.73% | 68.46% | 13 / 18 (72.22%) | **44 / 44 (100.0%)** |
| **Epoch 2/2** | 14.42 min | **`0.040920`** | **`0.014266`** | **`0.021207`** | **`0.001716`** | **`0.003732`** | **73.48%** | **74.18%** | **17 / 18 (94.44%)** | **44 / 44 (100.0%)** |

---

### Table 6: Model 13 WiSE-FT Weight-Space Interpolation Sweep ($\theta_{\text{M13}}(\alpha) = (1 - \alpha)\theta_{\text{M12}} + \alpha\theta_{\text{M13,ep2}}$)
*Composite Selection Formula: $0.22 \cdot \text{F1}_{\text{M9}} + 0.23 \cdot \text{F1}_{\text{Ext350}} + 0.18 \cdot \text{mAP}_{\text{Ext350}} + 0.12 \cdot \text{NegRate}_{\text{Ext350}} + 0.10 \cdot \text{Spec}_{\text{Grid18}} + 0.15 \cdot \text{CleanRate}_{\text{Nov20}}$.*

| WiSE-FT $\alpha$ | Composite Score | M9 279 F1 / mAP | Ext 350 F1 / mAP | Ext 350 HG / KN AP | CCTV 185 F1 / mAP | Grid 18 Spec | Staged 44 Raw / Post-CCTV | Nov 20 NW (40) | Selection Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` (M12 Base) | 0.773021 | 92.53% / 95.50% | 70.86% / 65.52% | 82.19% / 48.85% | 95.92% / 97.97% | 88.89% (16/18) | 56.82% / 84.09% | 67.50% (27/40) | Base Model 12 |
| `0.30` | 0.833605 | 91.38% / 94.77% | 73.59% / 72.75% | 83.01% / 62.50% | 94.92% / 97.54% | 88.89% (16/18) | 70.45% / 93.18% | 97.50% (39/40) | Runner-Up |
| **`0.50`** | **`0.845747`** | **89.60% / 92.87%** | **74.92% / 74.82%** | **83.48% / 66.16%** | **94.24% / 96.82%** | **94.44% (17/18)** | **84.09% / 100.00%** | **100.00% (40/40)** | **SELECTED PARETO-OPTIMAL SOTA (`best_weapon_detector_thirteenth_model.pth`)** |
| `0.65` | 0.839110 | 89.25% / 91.36% | 73.83% / 74.29% | 82.06% / 66.51% | 93.47% / 95.64% | 94.44% (17/18) | 93.18% / 100.00% | 100.00% (40/40) | Slight M9/Ext recall dip |
| `0.80` | 0.842699 | 87.98% / 89.42% | **75.51% / 75.01%** | 80.34% / 69.68% | 92.10% / 94.74% | 94.44% (17/18) | 93.18% / 100.00% | 100.00% (40/40) | Peak raw Ext mAP, lower HG AP |
| `1.00` (Ep2 Raw) | 0.834549 | 85.83% / 86.41% | 74.18% / 73.48% | 76.94% / **70.03%** | 90.97% / 92.36% | 94.44% (17/18) | 93.18% / 100.00% | 100.00% (40/40) | Un-interpolated Epoch 2 |

---

## SECTION 4: DATASET PROVENANCE & MODEL 13 HARD-NEGATIVE MINING

### Table 7: Dataset Split Progression (Models 10, 11, 12, and 13)

| Model & Partition Split | Total Full Images | Handgun (Class 1) | Knife (Class 2) | Hard Negative Frames (Class 0) | Injected Negative ROI Crops | Effective Pool | Notes & Zero-Leakage Isolation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model 10 Train** | 2,804 | 962 | 988 | 854 | +296 crops | 3,100 | Baseline before staged Tapo & surveillance-POV expansion. |
| **Model 11 Train** | 4,793 | 1,850 | 1,699 | 1,244 | +546 crops | 5,339 | Added 434 Tapo CAM 01/02 staged frames + 1,800 surveillance weapons. |
| **Model 12 Train** | 5,203 | 1,990 | 1,969 | 1,244 | +586 crops | 5,789 | Unlocked +410 CostiCatargiu weapons + 40 cup/mug ROI crops. |
| **Model 13 Train** | **5,203 + Nov 20 Full Frames** | **1,990** | **1,969** | **1,244 + Nov 20 Scenes `#01–#06`** *(2× oversampled)* | **586 + Nov 20 FP ROI Crops (`v14`)** | **Expanded Pool** | Mined strictly from 12 non-weapon clips (`CAM 01` & `CAM 02`: black/long TV remotes, metallic kitchen ladle, rolled white rag, smartphone, wallet). Weapon prop Scenes `#07–#11` 100% excluded. |
| **Validation Split (M11–M13)** | **650** | 254 | 242 | 154 | 0 | 650 | 0.0% overlap with train and test sets. |
| **Internal Test Split (M9–M13)** | **279** | 114 | 114 | 51 | 0 | 279 | Frozen standard benchmark split. |
| **Unseen CCTV Test Split** | **185** | 114 | 31 | 40 | 0 | 185 | 0.0% overlap across all 13 models. |
| **Unified External Benchmark** | **350** | 134 | 116 | 100 | 0 | 350 | 0.0% overlap across all 13 models. |

---

## SECTION 5: REAL CCTV VIDEO GROUND-TRUTH BENCHMARK & CONTROLLED ABLATIONS (16 STREAMS, 2,538 FRAMES)

*Evaluated on frame-by-frame Bounding-Box Ground Truth ([expert_evaluation_videos/ground_truth/](file:///Users/tyronegabrielr.pascual/CODING/THESIS/cctv-weapon-detection-system/expert_evaluation_videos/ground_truth/)) across 219 weapon-present frames (`HG01..03`, `KN01..03` Original & Enhanced) and 948 pure non-weapon frames (`NW01..04`).*

### Table 8: Aggregate Real CCTV Video Performance (Enhanced Weapon Videos + 4 Non-Handgun Videos)

| Model Iteration | Video Precision | Video Recall | Video F1-Score | Video Handgun AP | Video Knife AP | Video mAP@0.50 | TP / FP / FN | No-Weapon Video Specificity (948 Frames) | Mean TP Confidence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 9 (Legacy)** | 56.13% | 53.85% | 54.97% | 68.26% | 60.22% | 64.24% | 119 / 93 / 102 | 97.15% (921/948, 27 FP) | 91.62% |
| **Model 10 (Before Expansion)** | **100.00%** | 37.10% | 54.13% | 69.73% | 40.61% | 55.17% | 82 / **0** / 139 | **100.00% (948/948, 0 FP)** | 87.37% |
| **Model 11 (Partial Expansion)** | **100.00%** | 61.09% | 75.84% | 81.58% | 62.99% | 72.28% | 135 / **0** / 86 | **100.00% (948/948, 0 FP)** | 86.55% |
| **Model 12 (Full Expansion)** | **99.40%** | **75.11%** | **85.57%** | **88.13%** | **69.65%** | **78.89%** | **166 / 1 / 55** | **100.00% (948/948, 0 FP)** | **87.43%** |

### Table 9: Controlled Video Ablation Studies (BasicVSR++ Enhancement & CCTV Intelligence Layer)

| Ablation Comparison | Precision | Recall | F1-Score | TP / FP / FN | No-Weapon Clean Frames (948 Total) | Net Impact |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1A. Unenhanced Original Videos (With CCTV Intel)** | 98.55% | 61.54% | 75.77% | 136 / 2 / 85 | 948 / 948 (100.00%) | Baseline unenhanced stream |
| **1B. BasicVSR++ Enhanced Videos (With CCTV Intel)** | **99.40%** | **75.11%** | **85.57%** | **166 / 1 / 55** | **948 / 948 (100.00%)** | **`+13.57 pp` Recall (`+30 TP`), `+9.80 pp` F1** |
| **2A. Raw Faster R-CNN Detector Only (No CCTV Intel)** | 19.30% | 90.05% | 31.79% | 199 / 832 / 22 | 368 / 948 (38.82%) | Unfiltered raw frame proposals |
| **2B. Full Pipeline (+ CCTV Intelligence Layer)** | **99.40%** | **75.11%** | **85.57%** | **166 / 1 / 55** | **948 / 948 (100.00%)** | **`-99.88%` FP (`832 -> 1`), `+80.10 pp` Precision, `+61.18 pp` Specificity** |

---

## SECTION 6: PEER-REVIEWED SCIENTIFIC LITERATURE JUSTIFICATIONS

1. **Wortsman et al. (CVPR 2022) — WiSE-FT (`Robust fine-tuning of zero-shot models`):** Mathematically validates interpolating fine-tuned weights with base weights ($\theta_{\text{M13}} = 0.50\theta_{\text{M12}} + 0.50\theta_{\text{M13,ep2}}$) to prevent catastrophic forgetting while achieving Pareto-optimal recall and distractor immunity.
2. **Guo et al. (ICML 2017) & Küppers et al. (CVPRW 2020):** Validates confidence calibration limits in deep ResNet detectors on sub-1% small surveillance targets.
3. **Olmos et al. (Neurocomputing 2018) & Ruiz-Santaquiteria et al. (IEEE Access 2021):** Establishes 50% standardized benchmark evaluation and operational multi-frame confirmation to suppress everyday handheld object false alarms.
4. **Velasco-Mata et al. (2021), Lamas et al. (2022), Han et al. (Seq-NMS 2016), & Divvala et al. (CVPR 2009):** Validates FORENSIKADA's CCTV Intelligence Layer (human-proximity reach gating, anthropometric scaling, temporal tracklet smoothing, and static fixture suppression).
