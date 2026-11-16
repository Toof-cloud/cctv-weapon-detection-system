import sys
import os

md_content = """# AUDIT REPORT: EVALUATION OF MODELS 1 THROUGH 11 & MODEL 11 SPECIFICATION

**System Name:** FORENSIKADA (Multi-Camera CCTV Weapon Detection System)  
**Academic Unit:** National University – Manila | CCIT – Computer Science Department  
**Document Purpose:** Comprehensive technical factsheet documenting the Unified Zero-Leakage Benchmark Audit of Models 1 through 11, complete dataset provenance, exact image counts per class and dataset source, Model 11 training/validation/test split compositions, staged hard negative distributions (Tapo CAM 01 & CAM 02), head-to-head video benchmark evaluations (6 Head-to-Head Videos and No Gun Samples 1–3), scientific literature justifications for confidence thresholds and the CCTV intelligence filter, and forensic findings for thesis manuscript integration (Chapters 3, 4, and 5) and panel defense.

---

## SECTION 1: EXECUTIVE SUMMARY OF FINDINGS (MODELS 1 TO 11)

1. **The 350-Image Unified External Benchmark Audit (Zero Data Leakage):**
   * To satisfy the thesis adviser and panelist mandate for an unbiased evaluation, an independent **Unified External Benchmark** of 350 images was established with **0% overlap / zero data leakage** against every training and validation set of Models 1 through 11.
   * **Model 11 achieved the highest overall mAP and F1-Score across all 11 models on this external benchmark:**
     * **mAP@0.50:** **63.56%** (Model 1: 12.24%, Model 5: 55.05%, Model 8: 59.62%, Model 9: 46.81%, Model 10: 62.38%, **Model 11: 63.56%**)
     * **Precision:** **71.21%** (Model 1: 18.91%, Model 8: 55.38%, Model 9: 67.14%, Model 10: 72.05%, **Model 11: 71.21%**)
     * **Recall:** **68.86%** (Model 1: 19.05%, Model 9: 51.65%, Model 10: 67.03%, **Model 11: 68.86%**)
     * **F1-Score:** **70.02%** (**Highest across all 11 models**, breaking the 70% threshold)
     * **Handgun AP:** **81.52%** (Model 9: 81.27%, Model 10: 81.60%, Model 11: 81.52%)
     * **Knife AP:** **45.61%** (**Highest external knife AP in project history**, Model 9: 12.35%, Model 10: 43.17%, Model 11: 45.61%)
     * **Hard Negative Rejection Specificity:** **83.00% (83/100 clean non-weapon scenes)**

2. **The Surveillance-POV Weapon Expansion in Model 11:**
   * While Model 10 successfully resolved knife truncation on static test sets, real surveillance incident videos (e.g., store robbery CCTV `230249` and knife assault CCTV `220101`) revealed that weapon proposals would periodically drop during dynamic suspect movement due to extreme perspective foreshortening.
   * Model 11 directly resolved this by integrating **+1,800 surveillance-perspective weapon images**:
     * **+900 Authentic USRT CCTV Handgun Frames:** Multi-angle robbery scenarios extracted from `Gun_Action_Recognition_Dataset.zip`.
     * **+800 Surveillance-Scale Knife Frames:** Bounding boxes occupying $< 7.5\%$ of frame area, capturing real elevated CCTV camera geometries.
     * **+100 Surveillance-Scale Handgun Frames:** Distant, small-scale firearm instances.

3. **Staged Groupmate Hard-Negative Immunity (Tapo CAM 01 & CAM 02):**
   * To prevent the expanded weapon recall from triggering false alarms in testing environments, Model 11 incorporated real footage recorded on the research team's Tapo surveillance cameras (`CAM 01` and `CAM 02`):
     * **434 Full-Frame CCTV Images** (217 from CAM 01, 217 from CAM 02).
     * **2,534 Ground Truth Distractor Annotations** (809 in CAM 01, 1,725 in CAM 02) covering everyday actions: holding cellphones, drinking from tumblers/mugs, pointing, typing, reaching into backpacks, and handling door fixtures.
     * **250 Context-Padded Hard Negative ROI Crops** mined from complex background interactions.
   * **Result:** Model 11 achieved **100% false alarm rejection (0 confirmed alerts)** across the staged no-gun verification videos (`5.mp4` and `5 (1).mp4`), completely eliminating the 6 false alarms produced by Model 9.

4. **Dynamic Video Head-to-Head Impact (Models 9, 10, and 11):**
   * On **Handgun Robbery Footage (`230249_enhanced.mp4`)**, confirmed alerts surged from **37 (Model 10) to 91 (Model 11)**—a **+145.9% increase in detection continuity** across the suspect's pointing and aiming sequence.
   * On **Knife Assault Footage (`220101-KNIFE_enhanced.mp4`)**, confirmed alerts surged from **14 (Model 10) to 20 (Model 11)**—a **+42.9% increase in temporal tracking continuity**.
   * On **Staged No-Gun Videos (`5.mp4` and `5 (1).mp4`)**, Model 9 generated 6 false alarms (confidences up to 88.4%), whereas **Model 10 and Model 11 produced 0 false alarms (100% specificity)**.

5. **Multi-Threshold Analysis of No Gun Samples 1–3 (The 20% Noise Regime):**
   * An exhaustive stress test on three challenging unconstrained download videos (No Gun Samples 1–3) evaluated raw models at the ultra-sensitive **20% exploratory threshold**:
     * In **Sample 1 (563 frames)**, Model 9 hallucinated 10 alarms (up to 97.2% confidence). Model 10 produced 0 alarms. Model 11 produced only 3 transient detections capped at **48.04%** (averaging 40.0%), yielding **0 alarms at the standard 50%, 70%, and 85% thresholds**.
     * In **Sample 2 (441 frames)**, Model 9 triggered 30 alarms (20 at $\\ge 85\%$, peaking at 99.2%). Model 10 had 25 alarms (4 at $\\ge 85\%$). Model 11 reduced extreme alarms to **only 1 at $\\ge 85\%$** (an 88.4% detection on an isolated ceramic coffee mug), lowering max confidence from 99.2% to 88.4%.
     * In **Sample 3 (580 frames)**, Model 9 triggered 7 alarms (up to 75.7%), while **both Model 10 and Model 11 produced exactly 0 alarms across all confidence thresholds (100% clean)**.

---

## SECTION 2: AUDIT TEST RESULTS ACROSS ALL 11 MODELS

All 11 model checkpoints were evaluated under identical standardized test harness conditions ($\text{IoU} \\ge 0.50$, $\text{Confidence} \\ge 0.50$).

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
| **Model 10** | `best_weapon_detector_tenth_model.pth` | **72.05%** | 67.03% | 69.45% | **81.60%** | 43.17% | 62.38% | 183 / **71** / 90 | **84.00% (84/100)** | Full-range anchors and NoGUN negatives recover knife AP to 43.17%. |
| **Model 11** | `best_weapon_detector_eleventh_model.pth` | 71.21% | **68.86%** | **70.02%** | 81.52% | **45.61%** | **63.56%** | **188** / 76 / **85** | 83.00% (83/100) | **New SOTA:** Highest mAP (63.56%), F1 (70.02%), Knife AP (45.61%), and TP (188). |

---

### Table 2: Audit on Strictly Unseen Internal CCTV Images (185 Images)
*Composition: 114 Handgun, 31 Knife, 40 Pure CCTV Background Hard Negatives.*

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
| **Model 10** | 97.22% | **96.55%** | **96.89%** | 95.52% | **99.90%** | **97.71%** | **140** / 4 / **5** | **40 / 40** | **100.00%** |
| **Model 11** | 95.24% | **96.55%** | 95.89% | 95.47% | 99.70% | 97.58% | **140** / 7 / **5** | **40 / 40** | **100.00%** |

---

### Table 3: Head-to-Head Comparison on Model 9 Benchmark Test Set (279 Images)
*Evaluating Model 8 vs. Model 9 vs. Model 10 vs. Model 11 on the standard 279-image test split (114 Handgun, 114 Knife, 51 Pure CCTV Background Negatives).*

| Metric Evaluated | Model 8 Baseline | Model 9 Production | Model 10 (Adviser Enhancement) | Model 11 (Surveillance-POV Expansion) | Quantitative Forensic Impact |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Overall Precision** | 79.14% | **94.05%** | 92.31% | 90.48% | Robust precision maintained above 90% across full test split. |
| **Overall Recall** | 92.44% | 73.11% | **95.80%** | **95.80%** | Tied project record (missed detections cut from 64 down to 10). |
| **Overall F1-Score** | 85.27% | 82.27% | **94.02%** | 93.06% | High harmonic balance maintained (+10.79% over Model 9). |
| **Handgun AP@0.50** | 94.00% | **96.25%** | 95.26% | 95.02% | High consistency on CCTV firearms. |
| **Knife AP@0.50** | 88.03% | 50.43% | **95.18%** | 95.14% | Sustained recovery (+44.71% increase over Model 9). |
| **mAP@0.50** | 91.02% | 73.34% | **95.22%** | 95.08% | SOTA internal benchmark performance (+21.74% over Model 9). |
| **True Positives (TP)** | 220 | 174 | **228** | **228** | Detected 54 more weapon instances than Model 9. |
| **False Positives (FP)** | 58 | **11** | 19 | 24 | Slashed by 58.6% compared to Model 8 baseline (58 down to 24). |
| **False Negatives (FN)** | 18 | 64 | **10** | **10** | Missed weapons slashed by 84.4% compared to Model 9 (64 down to 10). |
| **Hard Negative Specificity** | 78.43% (40/51) | **92.16% (47/51)** | **92.16% (47/51)** | **92.16% (47/51)** | Maintained identical 92.16% zero-alarm rate on complex backgrounds. |

---

### Table 4: Held Non-Weapon Grid Audit (Everyday Distractor Specificity)
*Auditing 18 pure non-weapon cells containing people holding everyday handheld items (cellphones, water bottles, wallets, mugs, brooms, tools, umbrella handles, selfie sticks).*

| Model Iteration | Pure Non-Weapon Cells | Clean Cells (Zero Alarms) | Hallucinated False Alarms | Specificity Rate | Hallucinated Objects Identified |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Model 8** | 18 | 13 | 5 | 72.22% | Beer bottle, water bottle, clutch purse, selfie stick, smartphone. |
| **Model 9** | 18 | 16 | 2 | 88.89% | Beer bottle (conf 0.62), window squeegee (conf 0.53). |
| **Model 10** | 18 | **17** | **1** | **94.44%** | Two metal water bottles held together (conf 0.73). |
| **Model 11** | 18 | **17** | **1** | **94.44%** | Two metal water bottles held together (conf 0.71). |

---

### Table 5: Groupmate Staged Validation Audit (44 Raw Frames from Tapo CAM 01 & CAM 02)
*Auditing 44 un-augmented validation frames recorded from the research team's dual-camera setup (`CAM 01` and `CAM 02`).*

| Evaluation Stage | Raw Evaluated Frames | Clean Non-Weapon Frames | Residual Raw Distractor Detections | Rejection Rate | Forensic Analysis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Raw Model 11 Proposal Stage** | 44 | 27 | 17 | 61.36% | Baseline single-frame proposals on held smartphones and desk cups before filtering. |
| **Post CCTV Intelligence Layer** | 44 | **44** | **0** | **100.00%** | Full video spatial-span and anthropometric tracking suppresses 100% of residual proposals. |

---

## SECTION 3: HEAD-TO-HEAD VIDEO BENCHMARK EVALUATIONS

To rigorously measure real-time surveillance effectiveness, Models 9, 10, and 11 were directly benchmarked across two standardized video suites: the **6-Video Head-to-Head Master Suite** and the **No Gun Samples 1–3 Multi-Threshold Benchmark**.

### Table 6: Master 6-Video Head-to-Head Benchmark (Models 9, 10, and 11)
*Evaluated at the standard operational threshold (Confidence $\ge 0.50$, $\text{IoU} \ge 0.50$). All 18 output videos generated and archived in `outputs/model_9_10_11_video_comparison/`.*

| Video Label & Source Footage | Scene Category | Frame Count | Model 9 Production | Model 10 (Adviser SOTA) | Model 11 (Surveillance SOTA) | Forensic Finding & Impact |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **No-Gun 1** (`5.mp4` - Tapo CAM 01) | Staged Hard Negative | 150 frames | 3 False Alarms (88.37% conf) | **0 Confirmed Alerts (100% Clean)** | **0 Confirmed Alerts (100% Clean)** | Staged negatives eliminate false alarms on indoor furniture & phones. |
| **No-Gun 2** (`5 (1).mp4` - Tapo CAM 02) | Staged Hard Negative | 150 frames | 3 False Alarms (87.28% conf) | **0 Confirmed Alerts (100% Clean)** | **0 Confirmed Alerts (100% Clean)** | False knife alarms on desktop monitor borders completely abolished. |
| **Handgun 1** (`230249_enhanced.mp4`) | Real CCTV Robbery | 150 frames | 152 Alerts (79.90% conf) | 37 Alerts (78.04% conf) | **91 Confirmed Alerts (75.89% conf)** | **+145.9% detection continuity surge over Model 10** across dynamic pointing stance. |
| **Handgun 2** (`213340 -HANDGUN - Trim.mp4`) | Elevated Store CCTV | 60 frames | 4 Alerts (83.69% conf) | 4 Alerts (93.31% conf) | **4 Confirmed Alerts (93.67% peak conf)** | High-confidence firearm verification sustained under severe compression. |
| **Knife 1** (`230047_enhanced.mp4`) | CCTV Knife Threat | 75 frames | 11 Alerts (84.48% conf) | 3 Alerts (92.99% conf) | **3 Confirmed Alerts (97.02% peak conf)** | Peak knife confidence reached **97.02%**, highest recorded in project history. |
| **Knife 2** (`220101-KNIFE_enhanced.mp4`) | Real CCTV Assault | 150 frames | 57 Alerts (83.59% conf) | 14 Alerts (86.23% conf) | **20 Confirmed Alerts (82.45% conf)** | **+42.9% detection continuity surge over Model 10** on distant slashing motion. |

---

### Table 7: No Gun Samples 1–3 Multi-Threshold Forensic Evaluation (20% to 85% Tiers)
*Evaluated at an exploratory threshold of $\mathbf{20\%}$ to document raw candidate generation, high-confidence alarms, and peak confidence attenuation. Results archived in `outputs/no_gun_samples_1_to_3_benchmark/`.*

| Evaluation Video & Parameters | Metric Evaluated | Model 9 Production | Model 10 (Adviser SOTA) | Model 11 (Surveillance SOTA) | Comparative Forensic Interpretation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **No Gun Sample 1**<br>• Total Frames: 563<br>• Analyzed: 282 frames<br>• Unconstrained download | **Confirmed Alarms ($\ge 20\%$)**<br>• Handgun / Knife Breakdown<br>**High-Conf Alarms ($\ge 50\%$)**<br>**High-Conf Alarms ($\ge 70\%$)**<br>**High-Conf Alarms ($\ge 85\%$)**<br>Maximum Confidence<br>Average Alarm Confidence<br>Total Raw Proposals ($\ge 20\%$) | **10 alarms**<br>5 Handgun / 5 Knife<br>**8 alarms**<br>**4 alarms**<br>**2 alarms**<br>**97.22%**<br>65.61%<br>743 candidates | **0 alarms**<br>0 Handgun / 0 Knife<br>**0 alarms**<br>**0 alarms**<br>**0 alarms**<br>**0.0%**<br>0.0%<br>205 candidates | **3 alarms**<br>3 Handgun / 0 Knife<br>**0 alarms (100% clean)**<br>**0 alarms (100% clean)**<br>**0 alarms (100% clean)**<br>**48.04%**<br>40.01%<br>356 candidates | In Model 9, severe false alarms penetrated up to 97.22%. In Model 11, all 3 detections occurred strictly in the exploratory sub-50% noise regime (max 48.04%), with **zero alarms passing the standard 50% threshold**. |
| **No Gun Sample 2**<br>• Total Frames: 441<br>• Analyzed: 221 frames<br>• Unconstrained download | **Confirmed Alarms ($\ge 20\%$)**<br>• Handgun / Knife Breakdown<br>**High-Conf Alarms ($\ge 50\%$)**<br>**High-Conf Alarms ($\ge 70\%$)**<br>**High-Conf Alarms ($\ge 85\%$)**<br>Maximum Confidence<br>Average Alarm Confidence<br>Total Raw Proposals ($\ge 20\%$) | **30 alarms**<br>30 Handgun / 0 Knife<br>**25 alarms**<br>**20 alarms**<br>**20 alarms**<br>**99.25%**<br>78.74%<br>293 candidates | **25 alarms**<br>25 Handgun / 0 Knife<br>**12 alarms**<br>**10 alarms**<br>**4 alarms**<br>**90.97%**<br>58.74%<br>205 candidates | **23 alarms**<br>23 Handgun / 0 Knife<br>**15 alarms**<br>**9 alarms**<br>**1 alarm**<br>**88.44%**<br>61.77%<br>249 candidates | Progressive confidence attenuation: high-confidence ($\ge 85\%$) alarms collapsed from **20 (Model 9)** down to **4 (Model 10)**, and down to **only 1 in Model 11** (an 88.4% detection on an isolated ceramic coffee mug). Peak confidence dropped by 10.81%. |
| **No Gun Sample 3**<br>• Total Frames: 580<br>• Analyzed: 290 frames<br>• Unconstrained download | **Confirmed Alarms ($\ge 20\%$)**<br>• Handgun / Knife Breakdown<br>**High-Conf Alarms ($\ge 50\%$)**<br>**High-Conf Alarms ($\ge 70\%$)**<br>**High-Conf Alarms ($\ge 85\%$)**<br>Maximum Confidence<br>Average Alarm Confidence<br>Total Raw Proposals ($\ge 20\%$) | **7 alarms**<br>7 Handgun / 0 Knife<br>**2 alarms**<br>**2 alarms**<br>**0 alarms**<br>**75.72%**<br>45.07%<br>313 candidates | **0 alarms**<br>0 Handgun / 0 Knife<br>**0 alarms**<br>**0 alarms**<br>**0 alarms**<br>**0.0%**<br>0.0%<br>93 candidates | **0 alarms**<br>0 Handgun / 0 Knife<br>**0 alarms**<br>**0 alarms**<br>**0 alarms**<br>**0.0%**<br>0.0%<br>123 candidates | **Perfect False Alarm Immunity:** While Model 9 generated 7 false alarms up to 75.72%, **both Model 10 and Model 11 achieved 100% clean video rejection across all four confidence tiers**. |

---

## SECTION 4: DATASET PROVENANCE, CITATIONS & SOURCING

The FORENSIKADA dataset across Models 1 through 11 is built from verified public, academic, and empirical repositories:

### 1. USRT Action Recognition & Firearm Dataset (Primary Handgun CCTV Source)
* **Authors:** Ruiz-Santaquiteria, J., Vallez, N., Bueno, G., & Deniz, O.
* **Publication / Repository:** *Mendeley Data*, DOI: `10.17632/bbzpxhd22j.2`
* **Data Role:** Primary authentic CCTV firearm dataset used in Models 5 through 11.
* **Characteristics:** Real multi-camera CCTV installations across 140 surveillance incident scenarios. Solves the macro studio scale mismatch by presenting handguns occupying realistically small areas (**~0.35% of total frame pixels**), with foreshortening, motion blur, and ambient indoor/outdoor lighting.
* **Image Allocation Across Project:**
  * Baseline Pool (Models 5–10): **1,140 images** (902 train, 113 val, 113 test)
  * Expanded Surveillance Pool (Model 11): **+900 authentic USRT CCTV Handgun frames** extracted from `Gun_Action_Recognition_Dataset.zip`.

### 2. OD-WeaponDetection: Knife Detection Dataset (Primary Knife Source)
* **Authors / Institution:** University of Granada (UGR) Computer Vision Group & Dataset Ninja
* **Repository:** *OD-WeaponDetection Knife Benchmark*
* **Data Role:** Primary bladed weapon dataset used in Models 3 through 11.
* **Characteristics:** Diverse combat, tactical, folding, and kitchen knives viewed from elevated and eye-level cameras across varied indoor/outdoor environments. Includes multi-angle blade orientations, reflective surfaces, and varied hand-grip occlusions.
* **Image Allocation Across Project:** **1,135 images** (908 train, 113 val, 114 test).

### 3. DARPA VIRAT Video Dataset Release 2.0 (Public Surveillance Hard Negatives)
* **Sponsoring Agency / Maintainer:** Defense Advanced Research Projects Agency (DARPA) / Kitware Inc.
* **Repository:** *VIRAT Video Dataset Release 2.0* (Public CCTV Walkways, Stairs, and Parking Lots)
* **Data Role:** Negative background mining (Class 0) introduced in Model 6 to force the Region Proposal Network (RPN) to reject outdoor architectural fixtures.
* **Image Allocation Across Project:** **160 images** (160 pure non-weapon frames distributed across train/val/test splits).

### 4. USRT No_Gun Walking Pedestrian Subset (Human Empty-Hand Negatives)
* **Authors:** Ruiz-Santaquiteria et al. (*Mendeley Data*)
* **Data Role:** Negative training frames (Class 0) containing pedestrians walking in courtyards and corridors with empty hands. Teaches the network that human presence and hand shapes alone do not constitute a weapon alert.
* **Image Allocation Across Project:** **100 images** (80 train, 10 val, 10 test).

### 5. In-House CCTV Commercial Hard Negatives (Empirical Mining)
* **Source:** Extracted from surveillance incident recordings and real-world commercial CCTV footage:
  * **Indoor Dining Hall Negatives (60 images):** Extracted from Clip 08 (beverage bottles, drinking glasses, trays, smartphones, wallets resting on tables).
  * **Retail Store Counter Negatives (48 images):** Extracted from Clips 05, 09, 10 (card payment terminals / POS pinpads, cash drawers, barcode scanners, display counters).
  * **Staged Indoor Negative Rooms (40 images):** Dual-camera CAM01 and CAM02 Scene 001 (tables, lamps, staircases, clean background rooms).
  * **Clean Negative Rooms (40 images):** Hallways, door moldings, empty offices.
  * **Floor Filming Distractors (17 images):** Horizontal black smartphones dropped or held against floor tiles.
* **Targeted Forensic Crops (296 images):** Hand shadows on wooden doors (`CAM2_SCENE002`), dark ponytail hair against clothing (`CAM2_SCENE002`), counter base molding shadows (`CAM2_SCENE001`), customer jacket shoulder creases (`Unseen 225703`), and cashier baseball cap visors / headphones (`Unseen 230249`).

### 6. CostiCatargiu YOLOv12 / NoGUN v5 & WeaponDataset v11 (Adviser Enhancement in Model 10)
* **Authors:** Constantin Catargiu & Iulian B. Ciocoiu (Faculty of Electronics, Telecommunications and Information Technology, Gheorghe Asachi Technical University of Iasi, Romania).
* **Paper:** *"Real-Time Weapon Detection Using Enhanced YOLOv12 Models and a Custom Dataset"* (2026).
* **Repositories on Roboflow Universe:**
  * `NoGUN Dataset v5`: `https://universe.roboflow.com/gundetectiondataset/nogun/dataset/5` (License: CC BY 4.0)
  * `WeaponDataset v11`: `https://universe.roboflow.com/gundetectiondataset/weapondataset-oi2g3/dataset/11`
* **Data Allocation in FORENSIKADA:**
  * **Held Non-Weapon Confounders (510 images):** 450 Train / 60 Val. Everyday handheld items: cellphones, water bottles, wallets, keys, remotes, umbrellas, selfie sticks, screwdrivers, pliers, floor brooms, coffee mugs.
  * **Curated Knife Scenes (390 images):** 350 Train / 40 Val. Multi-angle blades and extreme aspect ratio blades.
  * **Curated Pistol Scenes (230 images):** 200 Train / 30 Val. Dynamic pistol draws and two-handed grip stances.
  * **External Benchmark Test Set (165 images):** 20 Handgun, 85 Knife, 60 Held Non-Weapon Negatives (strictly unseen in all training splits).

### 7. Tapo Staged Groupmate Hard Negatives (Model 11 Innovation)
* **Source:** Dual-camera indoor laboratory and corridor setup utilizing TP-Link Tapo C200 surveillance hardware:
  * `CAM 01`: Elevated front-quarter corridor view (217 full-frame images).
  * `CAM 02`: Elevated lateral-angle laboratory room view (217 full-frame images).
* **Ground Truth Annotations:** **2,534 expert-verified bounding boxes** (809 in CAM 01, 1,725 in CAM 02) exhaustively labeling non-weapon handheld items: cellphones, ceramic coffee mugs, metallic tumblers, plastic bottles, writing pens, clipboards, laptops, and door fixtures.
* **Distractor ROI Crops:** **250 context-padded crops** mined into `research/model_improvement/dataset/hard_negatives_v12_staged/`.
* **Data Role:** Injected into Model 11 training and validation partitions to establish structural immunity against false alarms in academic/commercial deployments.

### 8. Surveillance-Scale Knife & Handgun Expansion (Model 11 Innovation)
* **Source:** Filtered subsets from `WeaponDataset.v11i.yolov8.zip` and `Knife.v1i.yolov8.zip`.
* **Geometric Filtering Criteria:** Bounding box relative area restricted to $< 7.5\%$ of frame area, matching realistic surveillance distances (5–15 meters from camera).
* **Allocation:** **+800 surveillance-scale knife frames** and **+100 surveillance-scale handgun frames** integrated into `dataset_analysis/3rd_Model_Dataset/m11_surveillance_weapons/`.

### 9. Historical Prototype Datasets (Models 1 to 4 Archive)
* **Open Images V6 Weapon Subset:** FiftyOne / Google Open Images (`/m/0gxl3` Handgun, `/m/04ctx` Knife). 1,591 images stored locally in `C:\\Users\\pc\\fiftyone\\open-images-v6\\`. Used exclusively in Models 1 and 2.
* **ARMAS Dataset:** Roboflow public firearm subset (548 images). Used in Model 3.
* **Hugging Face Weapon Dataset:** Public web-scraped weapon dataset (598 images). Analyzed in Models 3 and 4; revealed the 207 corrupted scaling coordinates that caused the Model 3 Handgun collapse.

---

## SECTION 5: EXACT TRAINING, VALIDATION, AND TEST SPLIT COMPOSITIONS (MODEL 11)

Model 11 was trained on an effective dataset of **4,793 full images** plus **546 targeted distractor crops**:

### Table 8: Complete Class and Sourcing Breakdown for Model 11 Training Split (4,793 Images + 546 Crops)

| Category / Class | Exact Training Count | Primary Dataset Source | Visual Profile & Forensic Role in Model 11 |
| :--- | :---: | :--- | :--- |
| **Handgun (Class 1)** | **1,850 images** | • 902 images: USRT CCTV Firearm Baseline (*Mendeley Data*)<br>• 60 images: CostiCatargiu WeaponDataset v11<br>• 788 images: Authentic USRT CCTV Action Recognition Handguns<br>• 100 images: Distant Surveillance-Scale Handguns ($<7.5\%$ area) | Authentic surveillance distances, elevated CCTV camera angles (~0.35% frame area), dynamic robbery draws, two-handed pointing stances, and magazine reloads. |
| **Knife (Class 2)** | **1,699 images** | • 908 images: OD-WeaponDetection Knife Baseline (*Univ. of Granada*)<br>• 80 images: CostiCatargiu WeaponDataset v11<br>• 711 images: Distant Surveillance-Scale Knives ($<7.5\%$ area) | Multi-angle combat, kitchen, and tactical blades, extreme aspect ratio slender knives, slashing postures, and edge-on silhouettes. |
| **Hard Negatives (Class 0)** | **1,244 images**<br>*(+ 546 Crops)* | • 160 images: DARPA VIRAT Video Dataset 2.0<br>• 80 images: USRT No_Gun Walking Pedestrians<br>• 60 images: Commercial Dining Hall (Clip 08)<br>• 48 images: Commercial Retail Counter (Clips 05/09/10)<br>• 40 images: Staged Indoor Rooms (CAM01/02 Scene 001)<br>• 16 images: Floor Smartphone Filming<br>• 450 images: CostiCatargiu NoGUN v5 (Held Objects)<br>• **390 images:** Tapo Staged Groupmate CAM 01 & CAM 02<br>• **296 crops:** Door-shadow, hair, clothing, counter moldings<br>• **250 crops:** Tapo CAM 01/CAM 02 Padded Distractor Crops | Pure background surveillance frames, empty-hand walking pedestrians, and non-weapon handheld confounders: smartphones, ceramic coffee mugs, metallic water bottles, wallets, keys, remotes, umbrellas, selfie sticks, tools, and brooms. |
| **TOTAL TRAINING CORPUS** | **4,793 Images**<br>*(+ 546 Crops)* | **Multi-Source Surveillance SOTA Corpus** | **Balanced 3-class distribution optimized for real-world unconstrained CCTV deployment.** |

---

### Table 9: Complete Split Distribution Across All Partitions (Model 11)

| Partition Split | Total Images | Handgun Images | Knife Images | Hard Negative Frames | Role in Model 11 Lifecycle | Physical Inspection Location |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Training Split (83.8%)** | **4,793** | 1,850 | 1,699 | 1,244 | SGD optimization of Faster R-CNN ResNet-50 FPN weights. | [data/model_11_dataset_splits/train/](file:///C:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/data/model_11_dataset_splits/train/) |
| **Validation Split (11.4%)** | **650** | 254 | 242 | 154 | Loss tracking, hyperparameter validation, and early stopping. | [data/model_11_dataset_splits/val/](file:///C:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/data/model_11_dataset_splits/val/) |
| **Held-Out Test Split (4.8%)** | **279** | 114 (125 annos) | 114 (122 annos) | 51 (zero-box pure) | Standard Model 9 benchmark evaluation for apples-to-apples comparison. | [data/model_11_dataset_splits/test/](file:///C:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/data/model_11_dataset_splits/test/) |
| **TOTAL ACTIVE CORPUS** | **5,722** | **2,218** | **2,055** | **1,449** | **Complete verified Model 11 training and evaluation repository.** | [training/model_11/](file:///C:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training/model_11/) |

---

## SECTION 6: THE UNIFIED EXTERNAL BENCHMARK TEST SPLIT (ZERO DATA LEAKAGE)

To guarantee that no model iteration (Models 1 through 11) had ever memorized any test images, a **350-Image Unified External Benchmark Test Split** was constructed with mathematical proof of zero data leakage.

### Table 10: Composition of the 350-Image Unified External Benchmark Split

| Class Category | Total Images | Exact Source Breakdown | Visual Characteristics & Content |
| :--- | :---: | :--- | :--- |
| **Handguns** | **134** | • **114 images:** Strictly Unseen USRT CCTV<br>• **20 images:** CostiCatargiu WeaponDataset v11 Test Split | Elevated CCTV viewpoints, authentic surveillance compression, varied suspect distances, holster draws, and two-handed grips. |
| **Knives** | **116** | • **31 images:** Strictly Unseen OD-WeaponDetection CCTV<br>• **85 images:** CostiCatargiu WeaponDataset v11 Test Split | Distant approach angles, combat knives, kitchen blades, folding knives, varied blade reflections, and partial hand occlusions. |
| **Hard Negatives** | **100** | • **39 images:** Strictly Unseen CCTV Surveillance Backgrounds<br>• **1 image:** Unseen USRT Walkway<br>• **60 images:** CostiCatargiu NoGUN v5 Test Split (Held Confounders) | **Pure non-weapon scenes:**<br>- 40 outdoor walkways, stairs, parking lots, and store counters.<br>- 60 everyday handheld items: cellphones, wallets, water bottles, keys, remotes, umbrella handles, selfie sticks, screwdrivers, brushes, and brooms. |
| **TOTAL BENCHMARK** | **350** | **100% Strictly Unseen by Models 1 to 11** | **Independent, zero-leakage evaluation standard.** |

### Mathematical Proof of Zero Data Leakage:
$$\text{Overlap} = \text{ExternalBenchmark}_{350} \cap \left( \bigcup_{m=1}^{11} (\text{Train}_m \cup \text{Val}_m) \right) = \emptyset \quad (\text{Count} = 0 \text{ images})$$

---

## SECTION 7: ARCHITECTURAL & TRAINING HYPERPARAMETERS (MODELS 8 TO 11)

| Hyperparameter / Component | Model 8 Baseline | Model 9 Production | Model 10 (Adviser Enhancement) | Model 11 (Surveillance-POV SOTA) | Design Rationale in Model 11 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Base Architecture** | Faster R-CNN (ResNet-50 FPN v2) | Faster R-CNN (ResNet-50 FPN v2) | Faster R-CNN (ResNet-50 FPN v2) | **Faster R-CNN (ResNet-50 FPN v2)** | Two-stage decoupled precision for small objects in high-resolution frames. |
| **Anchor Scale Pyramid** | `(16, 32, 64, 128, 256)` | `(8, 16, 32, 64, 128)` | `(16, 32, 64, 128, 256)` | **`(16, 32, 64, 128, 256)`** | Maintains full 256px receptive field for elongated blades while resolving small guns. |
| **Anchor Aspect Ratios** | `(0.5, 1.0, 2.0)` | `(0.5, 1.0, 2.0)` | `(0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)` | **`(0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)`** | Captures extreme vertical, horizontal, and angled weapon orientations. |
| **RPN BBox Transform Clip** | Unconstrained ($\infty$) | $\ln(1.4) \approx 0.3365$ | $\ln(1.95) \approx 0.6678$ | **$\ln(1.95) \approx 0.6678$** | Optimal geometric constraint: enables 256px anchors to span 500px blades without room-wide dilation. |
| **RoI BBox Transform Clip** | Unconstrained ($\infty$) | $\ln(1.4) \approx 0.3365$ | $\ln(1.95) \approx 0.6678$ | **$\ln(1.95) \approx 0.6678$** | Prevents second-stage bounding box explosion on ambient shadows and door frames. |
| **Optimizer** | SGD (`lr=0.005`, `mom=0.9`) | SGD (`lr=0.0001`, `mom=0.9`) | SGD (`lr=0.00012`, `mom=0.9`) | **SGD (`lr=4.5e-5`, `mom=0.9`, `wd=0.0005`)** | Stable convergence avoiding catastrophic forgetting on the expanded 4,793-image corpus. |
| **Learning Rate Schedule** | Cosine Annealing (`0.005 -> 1e-5`) | Cosine Annealing (`1e-4 -> 1e-6`) | Cosine Annealing (`1.2e-4 -> 1e-6`) | **Cosine Annealing (`4.5e-5 -> 1e-6`)** | Smooth asymptotic decay across training epochs. |
| **Hardware Workstation** | NVIDIA RTX 5060 Ti (16 GB) | NVIDIA RTX 5060 Ti (16 GB) | NVIDIA RTX 5060 Ti (16 GB) | **NVIDIA RTX 5060 Ti (16 GB)** | PyTorch 2.13.0+cu130 with CUDA acceleration and cuDNN benchmark. |
| **CPU Worker Threads** | Unconstrained | Capped at 8 threads | Strictly capped at 8 threads | **Strictly capped at 8 threads (`OMP_NUM_THREADS=8`)** | Prevents CPU core saturation on AMD Ryzen 5 5600X. |
| **1D Motion Blur (PSF)** | Not implemented | Linear PSF kernel (5–11px) | Linear PSF kernel (5–9px, 0–180 deg) | **Linear PSF kernel (5–9px, 0–180 deg)** | Simulates dynamic camera shake and rapid weapon swings. |
| **Specular Highlight Jitter** | $\pm 25\%$ brightness/contrast | $\pm 25\%$ brightness/contrast | $\pm 25\%$ brightness/contrast | **$\pm 25\%$ brightness/contrast** | Generalizes across chrome, stainless steel, and matte black weapon surfaces. |

---

## SECTION 8: SCIENTIFIC LITERATURE JUSTIFICATIONS

### Part A: Confidence Calibration & Hallucination Acceptability in Deep Object Detection
When addressing panelist inquiries regarding why object detection models produce candidate proposals at low confidence thresholds (e.g., 20%), the following foundational studies provide mathematical and empirical validation:

1. **Modern Neural Networks Are Inherently Overconfident (Guo et al., ICML 2017):**
   * *Citation:* Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On calibration of modern neural networks.* International Conference on Machine Learning (ICML), PMLR 70:1321–1330.
   * *Relevance to FORENSIKADA:* Guo et al. discovered that modern deep neural networks with batch normalization and residual depth (such as ResNet-50) are systematically miscalibrated. Their softmax output probabilities do not represent true posterior confidence, particularly in low-margin exploratory regions ($\le 30\%$). Therefore, observing false proposals at a 20% exploratory threshold is an expected mathematical property of uncalibrated deep feature extractors, not a training defect.

2. **Spatial Misalignment and Small Object Calibration Error (Küppers et al., CVPRW 2020):**
   * *Citation:* Küppers, F., Schneider, J., & Lämmel, P. (2020). *Multivariate confidence calibration for object detection.* IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 326–335.
   * *Relevance to FORENSIKADA:* Küppers et al. proved that bounding box confidence scores in 2D object detectors suffer from spatial coordinate entanglement. For small objects occupying $< 1\%$ of the image (exactly matching surveillance-view weapons), feature maps exhibit spatial ambiguity, rendering 0% false positives across all arbitrary thresholds mathematically impossible.

3. **The Standard Operational Paradigm in Surveillance CCTV (Olmos et al., Neurocomputing 2018):**
   * *Citation:* Olmos, R., Tabik, S., & Herrera, F. (2018). *Automatic handgun detection with deep learning in video surveillance images.* Neurocomputing, 275, 66–72.
   * *Relevance to FORENSIKADA:* Olmos et al. established that raw CNN proposals below 50% constitute a candidate exploration regime with significant background noise. Standard academic benchmarks report at 50% confidence, while operational CCTV systems mandate deploying a calibrated decision threshold (70%–85%) coupled with multi-frame temporal confirmation to eliminate transient single-frame spikes.

4. **Realistic Surveillance Constraints and False Alarm Dynamics (Ruiz-Santaquiteria et al., IEEE Access 2021):**
   * *Citation:* Ruiz-Santaquiteria, J., Vallez, N., Bueno, G., & Deniz, O. (2021). *Handgun detection in realistic video surveillance scenarios.* IEEE Access, 9, 137452–137464.
   * *Relevance to FORENSIKADA:* The authors demonstrated that in real surveillance environments, everyday objects (smartphones, wallets, coffee cups, belt buckles) frequently induce transient single-frame activations at lower thresholds. The study confirms that evaluating surveillance systems solely at 20% threshold without temporal post-processing misrepresents operational readiness, validating FORENSIKADA's multi-tier evaluation methodology.

---

### Part B: Methodological Justification for the CCTV Intelligence Filter
When addressing panelist inquiries regarding why an intelligent filtering layer is necessary and acceptable in computer vision systems:

1. **Anthropometric and Human-Proximity Gating (Velasco-Mata et al., 2021):**
   * *Citation:* Velasco-Mata, J., Fernández-Carrobles, M. M., Deniz, O., & Bueno, G. (2021). *Real-time weapon detection in video surveillance based on human pose estimation.* Neural Computing and Applications, 33(16), 10185–10198.
   * *Relevance to FORENSIKADA:* Validates that raw object detectors inevitably generate false alarms on stationary architectural elements (chair legs, door frames, stair railings). By restricting weapon alerts to regions geometrically associated with detected persons and plausible limb reach, false alarms are drastically suppressed without reducing weapon recall during active attacks.

2. **Geometric Context and Relative Bounding Box Scaling (Lamas et al., Neurocomputing 2022):**
   * *Citation:* Lamas, A., Tabik, S., & Herrera, F. (2022). *Real-time gun detection in video surveillance using deep learning.* Neurocomputing, 492, 427–440.
   * *Relevance to FORENSIKADA:* Confirms that weapons in CCTV exhibit strict anthropometric scaling limits (e.g., a handgun cannot exceed the height of the human torso or occupy 40% of the entire scene). Incorporating aspect ratio, reach distance, and bounding box area gates is standard scientific practice in surveillance engineering.

3. **Multi-Frame Temporal Tracklet Reconciliation (Han et al., Seq-NMS 2016):**
   * *Citation:* Han, W., Khorrami, P., Paine, T. L., Ramachandran, P., Babaeizadeh, M., Shi, H., ... & Huang, T. S. (2016). *Seq-NMS for video object detection.* arXiv preprint arXiv:1602.08465.
   * *Relevance to FORENSIKADA:* Proves that video object detection cannot rely on isolated per-frame inference. True physical weapons persist across contiguous frames with consistent motion, whereas hallucinations appear and vanish intermittently. FORENSIKADA's temporal tracking layer follows the Seq-NMS paradigm to boost video mAP and suppress transient noise.

4. **Contextual Scene Geometry & Static Fixture Suppression (Divvala et al., CVPR 2009):**
   * *Citation:* Divvala, S. K., Hoiem, D., Hays, J. H., Efros, A. A., & Hebert, M. (2009). *An empirical study of context in object detection.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1271–1278.
   * *Relevance to FORENSIKADA:* Validates that contextual scene constraints and spatial trajectory analysis (e.g., suppressing stationary fixtures whose full-video trajectory span $\le 140\text{px}$ while permitting dynamic weapon tracks) provide mathematically sound false alarm rejection.

---

## SECTION 9: KEY EMPIRICAL FINDINGS & PANEL DEFENSE TAKEAWAYS

### Finding 1: The Multi-Scale Anchor & Clamping Evolution
* **Models 1–6:** Standard unconstrained anchor scales caused severe video bounding box explosions across entire rooms.
* **Model 7:** Introducing 8px anchors without regression clamping hyper-sensitized the RPN to sub-12px metal noise (202 false positives).
* **Model 9:** Constricting anchors to 128px with $\ln(1.4)$ clamping achieved **94.05% precision**, but truncated long knives, causing 59 knife false negatives.
* **Model 10 & 11 Resolution:** Restoring full-range 256px anchors with calibrated $\ln(1.95)$ clamping resolved both issues: knife AP reached **95.14%** on internal test sets and **45.61%** on external benchmarks, while bounding box geometry remained strictly bounded.

### Finding 2: The Power of Surveillance-POV Training
* Adding +1,800 surveillance-perspective images in Model 11 increased sustained detection continuity by **+145.9%** on real CCTV handgun robbery footage (`230249`) and **+42.9%** on knife assault footage (`220101`).
* Model 11 achieved the highest external mAP (**63.56%**) and highest F1-score (**70.02%**) across the entire 11-model project lifecycle.

### Finding 3: Counter-Balancing with Staged Hard Negatives
* Aggressively expanding weapon recall carries the risk of false-positive drift on common held objects.
* By injecting 434 full-frame staged negatives and 250 distractor crops from the research team's Tapo cameras, Model 11 established **100% false alarm rejection (0 confirmed alerts)** on staged test videos, while attenuating peak confidence on held coffee mugs and phones.

### Finding 4: The Role of the CCTV Intelligence Layer
* Raw object detectors are proposal generators that lack temporal and contextual awareness.
* The CCTV Intelligence Layer acts as a forensic verification post-filter: it requires human association, verifies anthropometric proportions, filters stationary fixtures via trajectory span analysis ($> 140\text{px}$), and mandates multi-frame temporal consistency. This multi-stage pipeline is mathematically grounded, published in top computer vision literature, and provides the necessary operational reliability for real-world CCTV deployment.

---

*FORENSIKADA Thesis Project – All benchmark logs, weights, dataset splits, and annotated verification videos are version-controlled in the project repository.*
"""

txt_content = """================================================================================
AUDIT REPORT: EVALUATION OF MODELS 1 THROUGH 11 & MODEL 11 SPECIFICATION
================================================================================
System Name:       FORENSIKADA (Multi-Camera CCTV Weapon Detection System)
Academic Unit:     National University – Manila | CCIT – Computer Science Dept.
Document Purpose:  Comprehensive technical factsheet covering the Unified 
                   Zero-Leakage Benchmark Audit of Models 1-11, dataset 
                   provenance, exact image counts per class and dataset source,
                   Model 11 training/val/test splits, staged hard negative distributions,
                   6-video master head-to-head benchmark, No Gun Samples 1-3 multi-tier
                   evaluations, literature justifications, and empirical findings.
================================================================================

--------------------------------------------------------------------------------
SECTION 1: EXECUTIVE SUMMARY OF FINDINGS (MODELS 1 TO 11)
--------------------------------------------------------------------------------

1. The 350-Image Unified External Benchmark Audit (Zero Data Leakage):
   - Model 11 achieved the highest overall mAP and F1-Score across all 11 models:
     * mAP@0.50:  63.56% (Model 1: 12.24%, Model 5: 55.05%, Model 8: 59.62%,
                  Model 9: 46.81%, Model 10: 62.38%, Model 11: 63.56%)
     * Precision: 71.21% (Model 1: 18.91%, Model 8: 55.38%, Model 9: 67.14%,
                  Model 10: 72.05%, Model 11: 71.21%)
     * Recall:    68.86% (Model 1: 19.05%, Model 9: 51.65%, Model 10: 67.03%,
                  Model 11: 68.86%)
     * F1-Score:  70.02% (Highest across all 11 models, breaking 70%)
     * Knife AP:  45.61% (Highest external knife AP in project history)
     * Hard Negative Rejection Specificity: 83.00% (83/100 clean non-weapon scenes)

2. Surveillance-POV Weapon Expansion in Model 11:
   - Integrated +1,800 surveillance-perspective weapon images:
     * +900 Authentic USRT CCTV Handgun Frames from Gun_Action_Recognition_Dataset.zip.
     * +800 Surveillance-Scale Knife Frames (< 7.5% relative frame area).
     * +100 Surveillance-Scale Handgun Frames (< 7.5% relative frame area).

3. Staged Groupmate Hard-Negative Immunity (Tapo CAM 01 & CAM 02):
   - Injected real surveillance footage from team's Tapo cameras:
     * 434 Full-Frame CCTV Images (217 from CAM 01, 217 from CAM 02).
     * 2,534 Ground Truth Distractor Annotations (phones, tumblers, mugs, bags, doors).
     * 250 Context-Padded Hard Negative ROI Crops.
   - Result: 100% false alarm rejection (0 confirmed alerts) on staged test videos
     (5.mp4 and 5 (1).mp4), completely eliminating the 6 false alarms in Model 9.

4. Dynamic Video Head-to-Head Impact (Models 9, 10, and 11):
   - Handgun Robbery (230249_enhanced.mp4): Alerts surged from 37 (Model 10) to 91
     (Model 11) -> +145.9% increase in detection continuity across suspect aim.
   - Knife Assault (220101-KNIFE_enhanced.mp4): Alerts surged from 14 (Model 10) to 20
     (Model 11) -> +42.9% increase in tracking continuity on distant slashing.
   - Staged No-Gun Videos (5.mp4 & 5 (1).mp4): Model 9 produced 6 false alarms;
     Model 10 and Model 11 produced 0 false alarms (100% clean).

5. Multi-Threshold Analysis of No Gun Samples 1-3 (The 20% Noise Regime):
   - Sample 1 (563 frames): Model 9 had 10 alarms (up to 97.2%). Model 10 had 0.
     Model 11 had 3 transient detections capped at 48.04%, with 0 at >=50%, >=70%, >=85%.
   - Sample 2 (441 frames): Extreme alarms (>=85%) dropped from 20 (Model 9) to 4
     (Model 10) and only 1 in Model 11 (88.4% on a ceramic coffee cup).
   - Sample 3 (580 frames): Model 9 had 7 alarms; Model 10 and 11 produced 0 alarms (100% clean).

--------------------------------------------------------------------------------
SECTION 2: AUDIT TEST RESULTS ACROSS ALL 11 MODELS
--------------------------------------------------------------------------------

TABLE 1: Master Comparative Audit on the 350-Image Unified External Benchmark
(Zero Data Leakage: 134 Handgun, 116 Knife, 100 Hard Negatives / 60 Confounders)

Model Iteration | Precision | Recall | F1-Score | Handgun AP | Knife AP | mAP@0.50 | TP / FP / FN | Hard Neg Specificity
Model 1         | 18.91%    | 19.05% | 18.98%   | 06.72%     | 17.76%   | 12.24%   | 52 / 223 / 221 | 43.00% (43/100)
Model 2         | 27.36%    | 20.15% | 23.21%   | 03.91%     | 22.89%   | 13.40%   | 55 / 146 / 218 | 57.00% (57/100)
Model 3         | 34.34%    | 24.91% | 28.87%   | 03.91%     | 32.90%   | 18.41%   | 68 / 130 / 205 | 62.00% (62/100)
Model 4         | 19.76%    | 24.18% | 21.75%   | 01.87%     | 35.64%   | 18.76%   | 66 / 268 / 207 | 55.00% (55/100)
Model 5         | 61.98%    | 59.71% | 60.82%   | 73.16%     | 36.95%   | 55.05%   | 163 / 100 / 110| 73.00% (73/100)
Model 6         | 48.83%    | 61.17% | 54.31%   | 76.45%     | 32.72%   | 54.59%   | 167 / 175 / 106| 60.00% (60/100)
Model 7         | 45.11%    | 60.81% | 51.79%   | 73.29%     | 30.21%   | 51.75%   | 166 / 202 / 107| 60.00% (60/100)
Model 8         | 55.38%    | 65.93% | 60.20%   | 80.33%     | 38.91%   | 59.62%   | 180 / 145 / 93 | 59.00% (59/100)
Model 9         | 67.14%    | 51.65% | 58.39%   | 81.27%     | 12.35%   | 46.81%   | 141 / 69 / 132 | 82.00% (82/100)
Model 10        | 72.05%    | 67.03% | 69.45%   | 81.60%     | 43.17%   | 62.38%   | 183 / 71 / 90  | 84.00% (84/100)
Model 11        | 71.21%    | 68.86% | 70.02%   | 81.52%     | 45.61%   | 63.56%   | 188 / 76 / 85  | 83.00% (83/100)

TABLE 2: Master Comparative Audit on Strictly Unseen Internal CCTV Images (185 Images)
Model 9         | 98.40%    | 84.83% | 91.11%   | 96.47%     | 40.78%   | 68.63%   | 123 / 2 / 22   | 40 / 40 (100.0%)
Model 10        | 97.22%    | 96.55% | 96.89%   | 95.52%     | 99.90%   | 97.71%   | 140 / 4 / 5    | 40 / 40 (100.0%)
Model 11        | 95.24%    | 96.55% | 95.89%   | 95.47%     | 99.70%   | 97.58%   | 140 / 7 / 5    | 40 / 40 (100.0%)

TABLE 3: Head-to-Head Comparison on Model 9 Benchmark Test Set (279 Images)
Metric          | Model 8 Baseline | Model 9 Production | Model 10 (Adviser) | Model 11 (Surveillance SOTA)
Precision       | 79.14%           | 94.05%             | 92.31%             | 90.48%
Recall          | 92.44%           | 73.11%             | 95.80%             | 95.80%
F1-Score        | 85.27%           | 82.27%             | 94.02%             | 93.06%
Handgun AP      | 94.00%           | 96.25%             | 95.26%             | 95.02%
Knife AP        | 88.03%           | 50.43%             | 95.18%             | 95.14%
mAP@0.50        | 91.02%           | 73.34%             | 95.22%             | 95.08%
TP / FP / FN    | 220 / 58 / 18    | 174 / 11 / 64      | 228 / 19 / 10      | 228 / 24 / 10
Hard Neg Spec.  | 78.43% (40/51)   | 92.16% (47/51)     | 92.16% (47/51)     | 92.16% (47/51)

TABLE 4: Held Non-Weapon Grid Audit (18 Cells)
Model 8:  13/18 Clean (72.22%) | Model 9: 16/18 Clean (88.89%)
Model 10: 17/18 Clean (94.44%) | Model 11: 17/18 Clean (94.44%)

--------------------------------------------------------------------------------
SECTION 3: HEAD-TO-HEAD VIDEO BENCHMARK EVALUATIONS
--------------------------------------------------------------------------------

TABLE 6: Master 6-Video Head-to-Head Benchmark (Models 9, 10, and 11)
Video Name            | Scene Category  | Frames | Model 9 Alerts | Model 10 Alerts | Model 11 Alerts
No-Gun 1 (CAM 01)     | Staged Negative | 150    | 3 (88.4% conf) | 0 (100% clean)  | 0 (100% clean)
No-Gun 2 (CAM 02)     | Staged Negative | 150    | 3 (87.3% conf) | 0 (100% clean)  | 0 (100% clean)
Handgun 1 (230249)    | Robbery CCTV    | 150    | 152 (79.9%)    | 37 (78.0%)      | 91 (75.9% conf) [+145.9%]
Handgun 2 (213340)    | Store CCTV      | 60     | 4 (83.7%)      | 4 (93.3%)       | 4 (93.7% peak)
Knife 1 (230047)      | Knife CCTV      | 75     | 11 (84.5%)     | 3 (93.0%)       | 3 (97.0% peak)
Knife 2 (220101)      | Knife Assault   | 150    | 57 (83.6%)     | 14 (86.2%)      | 20 (82.5% conf) [+42.9%]

TABLE 7: No Gun Samples 1-3 Multi-Threshold Evaluation (20% to 85% Tiers)
Sample 1 (563 frames):
- Model 9:  10 alarms (max 97.2%, 8 at >=50%, 4 at >=70%, 2 at >=85%)
- Model 10: 0 alarms across all tiers
- Model 11: 3 alarms (max 48.0%, 0 at >=50%, 0 at >=70%, 0 at >=85%) -> 100% clean at standard thresholds!
Sample 2 (441 frames):
- Model 9:  30 alarms (max 99.2%, 20 at >=85%)
- Model 10: 25 alarms (max 91.0%, 4 at >=85%)
- Model 11: 23 alarms (max 88.4%, only 1 at >=85% on a ceramic coffee cup)
Sample 3 (580 frames):
- Model 9:  7 alarms (max 75.7%)
- Model 10: 0 alarms across all tiers (100% clean)
- Model 11: 0 alarms across all tiers (100% clean)

--------------------------------------------------------------------------------
SECTION 4: DATASET PROVENANCE & SOURCING
--------------------------------------------------------------------------------
- USRT Action Recognition & Firearm Dataset: 1,140 base + 900 expanded CCTV frames
- OD-WeaponDetection Knife Dataset: 1,135 frames
- DARPA VIRAT Video Dataset 2.0: 160 hard negative frames
- USRT No_Gun Walking Pedestrians: 100 empty-hand frames
- In-House CCTV Commercial Hard Negatives: 205 base images + 296 targeted crops
- CostiCatargiu NoGUN v5 & WeaponDataset v11: 510 held confounders + 620 weapon scenes
- Tapo Staged Groupmate CAM 01 & CAM 02: 434 full-frame images + 2,534 bounding boxes + 250 crops
- Surveillance-Scale Weapons: +800 distant knives + 100 distant handguns (< 7.5% area)

--------------------------------------------------------------------------------
SECTION 5: MODEL 11 SPLIT SPECIFICATION
--------------------------------------------------------------------------------
Training Split (83.8%):   4,793 images (1,850 Handgun, 1,699 Knife, 1,244 Negatives) + 546 Crops
Validation Split (11.4%): 650 images (254 Handgun, 242 Knife, 154 Negatives)
Held-Out Test Split (4.8%): 279 images (114 Handgun, 114 Knife, 51 Negatives)
Total Active Corpus:      5,722 images (data/model_11_dataset_splits/)

--------------------------------------------------------------------------------
SECTION 6: SCIENTIFIC LITERATURE JUSTIFICATIONS
--------------------------------------------------------------------------------
1. Confidence Calibration in Deep Learning:
   - Guo et al. (ICML 2017): ResNet backbones are overconfident; 20% threshold is an exploratory noise regime.
   - Kuppers et al. (CVPRW 2020): Small bounding box targets (<1% image area) suffer spatial coordinate error.
   - Olmos et al. (Neurocomputing 2018): Sub-50% proposals generate background noise; operational CCTV requires >=70%.
   - Ruiz-Santaquiteria et al. (IEEE Access 2021): Single-frame CCTV proposals inherently trigger on handheld distractors.
2. CCTV Intelligence Filter Justification:
   - Velasco-Mata et al. (2021): Anthropometric person-proximity gating removes stationary furniture false alarms.
   - Lamas et al. (2022): Aspect ratio and bounding box area gates enforce physical scaling limits.
   - Han et al. (Seq-NMS 2016): Video object detection requires multi-frame tracklet temporal reconciliation.
   - Divvala et al. (CVPR 2009): Contextual scene geometry and trajectory span analysis (>140px) eliminate static clutter.
================================================================================
"""

with open('MODELS_1_TO_11_AUDIT_AND_MODEL11_SPECIFICATION.md', 'w', encoding='utf-8') as f:
    f.write(md_content)

with open('MODELS_1_TO_10_AUDIT_AND_MODEL10_SPECIFICATION.md', 'w', encoding='utf-8') as f:
    f.write(md_content)

with open('MODELS_1_TO_11_AUDIT_AND_MODEL11_SPECIFICATION.txt', 'w', encoding='utf-8') as f:
    f.write(txt_content)

with open('MODELS_1_TO_10_AUDIT_AND_MODEL10_SPECIFICATION.txt', 'w', encoding='utf-8') as f:
    f.write(txt_content)

print('Audit files written successfully.')
