# PHASE 1 FINAL REPORT: MODEL 12 FINALIZATION, VIDEO GROUND-TRUTH EVALUATION & DATASET IMPROVEMENT AUDIT

**System Name:** FORENSIKADA (Multi-Camera CCTV Weapon Detection System)  
**Academic Unit:** National University – Manila | CCIT – Computer Science Department  
**Final Production Checkpoint:** `best_weapon_detector_twelfth_model.pth` (Model 12 — Calibrated WiSE-FT SOTA)  
**Un-interpolated High-Recall Checkpoint:** `latest_weapon_detector_twelfth_model.pth` (Model 12 Epoch 2)  
**Verification Artifacts:**
- Training & Epoch Component Loss Log: [training/model_12/model_12_training_history.json](file:///C:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training/model_12/model_12_training_history.json)
- WiSE-FT Pareto Calibration Log: [training/model_12/model_12_wise_ft_calibration.json](file:///C:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training/model_12/model_12_wise_ft_calibration.json)
- Video Ground-Truth Annotations & Contact Sheets: [expert_evaluation_videos/ground_truth/](file:///C:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/expert_evaluation_videos/ground_truth/)
- Per-Video Ground-Truth Evaluation CSV: [outputs/phase1_final_video_gt_benchmark/video_ground_truth_evaluation_results.csv](file:///C:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/outputs/phase1_final_video_gt_benchmark/video_ground_truth_evaluation_results.csv)
- Aggregate Video Ground-Truth Summary CSV: [outputs/phase1_final_video_gt_benchmark/video_gt_aggregate_summary.csv](file:///C:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/outputs/phase1_final_video_gt_benchmark/video_gt_aggregate_summary.csv)

---

## 1. Executive Summary of Adviser Requirements Completed

1. **Dataset Expansion & Hard-Negative Integration (Model 10 $\rightarrow$ Model 11 $\rightarrow$ Model 12):**
   - **Before Dataset Additions (Model 10):** Trained on `2,804` full images (`962` Handgun, `988` Knife, `854` Hard Negatives) + `296` negative crops (`3,100` effective items).
   - **Partial Expansion (Model 11):** Added the Groupmate Staged Tapo C200 Hard Negatives (`390` train frames + `44` val frames + `250` distractor crops from `2,534` expert annotations) and extracted `+1,800` surveillance-perspective weapon frames, but sampled only `600` of the new surveillance weapon frames per epoch (`4,793` images + `546` crops in pool).
   - **Full Final Expansion (Model 12):** Unlocked `410` previously unused zero-leakage CostiCatargiu training images (`270` Knife + `140` Handgun), exposed the remaining `~900` surveillance-POV knife and USRT handgun images across training epochs, and added `40` targeted handheld cup/mug/torso hard-negative crops (`hard_negatives_v13_cups`), bringing the final training corpus to **`5,203` full images (`1,990` Handgun, `1,969` Knife, `1,244` Hard Negatives) + `586` targeted negative ROI crops = `5,789` effective training items** (`0.0%` overlap with all test sets).

2. **Additional Ground Truth for Real CCTV Video Evaluation:**
   - Constructed frame-by-frame Bounding-Box Ground Truth ([expert_evaluation_videos/ground_truth/](file:///C:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/expert_evaluation_videos/ground_truth/)) across **10 distinct CCTV evaluation scenarios (16 video streams totaling 2,538 evaluated ground-truth frames at 15 FPS analysis rate)**:
     - **3 Handgun CCTV Pairs (Unenhanced Original + BasicVSR++ Enhanced):** `HG01` (`230249`, 121 GT weapon frames / 425 total), `HG02` (`213340`, 4 GT weapon frames / 121 total), `HG03` (`handgun_test-video`, 42 GT weapon frames / 75 total).
     - **3 Knife CCTV Pairs (Unenhanced Original + BasicVSR++ Enhanced):** `KN01` (`230047`, 9 GT weapon frames / 130 total), `KN02` (`220101`, 35 GT weapon frames / 464 total), `KN03` (`230842`, 8 GT weapon frames / 373 total).
     - **4 Non-Handgun / No-Weapon Videos:** `NW01` (281 frames), `NW02` (220 frames), `NW03` (290 frames), `NW04` (157 frames) — **948 pure non-weapon ground-truth frames**.

---

## 2. Training Metrics & Per-Epoch Component Loss Log (Model 10 vs. Model 11 vs. Model 12)

| Model / Epoch | Active Items / Epoch | Learning Rate | Total Train Loss | `loss_classifier` | `loss_box_reg` | `loss_objectness` | `loss_rpn_box_reg` | Epoch Duration (RTX 5060 Ti) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 10 (Final Epoch 2)** | 3,100 | `1.2e-4 -> 1e-6` | `0.026228` | — | — | — | — | ~8.5 min |
| **Model 11 (Final Epoch 2)** | 4,330 | `4.5e-5 -> 1e-6` | `0.032078` | — | — | — | — | ~10.5 min |
| **Model 12 (Epoch 1/2)** | 5,090 | `0.000040` | `0.051040` | `0.021818` | `0.021812` | `0.004132` | `0.003278` | 11.17 min |
| **Model 12 (Epoch 2/2)** | 5,090 | `0.000021` | **`0.049346`** | **`0.020836`** | **`0.021904`** | **`0.003428`** | **`0.003178`** | 10.55 min |

---

## 3. Static Benchmark Comparison & Levels of Improvement (Before vs. After Dataset Additions)

All checkpoints were evaluated under identical criteria ($\text{IoU} \ge 0.50$, $\text{Confidence} \ge 0.50$).

| Benchmark Test Suite | Metric | Model 9 (Legacy Prod) | Model 10 (Before Staged & Surv Expansion) | Model 11 (Partial Expansion) | **Model 12 Calibrated (`alpha=0.30`)** | **Model 12 Epoch 2 (`alpha=1.00`)** | **Level of Improvement (M10 $\rightarrow$ M12 Calibrated / Ep2)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Unified External Benchmark**<br>*(350 Unseen Images — Zero Leakage)* | **mAP@0.50**<br>**Handgun AP**<br>**Knife AP**<br>**F1-Score**<br>**Precision**<br>**Recall**<br>**TP / FP / FN**<br>**Clean Negatives (100)** | 46.81%<br>81.27%<br>12.35%<br>58.39%<br>67.14%<br>51.65%<br>141 / 69 / 132<br>82/100 (82.0%) | 62.38%<br>81.60%<br>43.17%<br>69.45%<br>72.05%<br>67.03%<br>183 / 71 / 90<br>84/100 (84.0%) | 63.56%<br>81.52%<br>45.61%<br>70.02%<br>71.21%<br>68.86%<br>188 / 76 / 85<br>83/100 (83.0%) | **65.52%**<br>**82.19%**<br>**48.85%**<br>**70.86%**<br>**69.61%**<br>**72.16%**<br>**197 / 86 / 76**<br>**82/100 (82.0%)** | **69.42%**<br>**82.90%**<br>**55.93%**<br>**69.87%**<br>63.75%<br>**77.29%**<br>**211 / 120 / 62**<br>77/100 (77.0%) | **`+3.14 pp` (`+7.04 pp` Ep2)**<br>**`+0.59 pp` (`+1.30 pp` Ep2)**<br>**`+5.68 pp` (`+12.76 pp` Ep2)**<br>**`+1.41 pp` (Project Record)**<br>`-2.44 pp`<br>**`+5.13 pp` (`+10.26 pp` Ep2)**<br>**`+14 TP` (`+28 TP` Ep2)**<br>Maintained $\ge 82\%$ clean |
| **2. Strictly Unseen Internal CCTV**<br>*(185 Unseen CCTV Images)* | **mAP@0.50**<br>**Handgun AP**<br>**Knife AP**<br>**F1-Score**<br>**Precision**<br>**Recall**<br>**TP / FP / FN**<br>**Clean Negatives (40)** | 68.63%<br>96.47%<br>40.78%<br>91.11%<br>98.40%<br>84.83%<br>123 / 2 / 22<br>40/40 (100.0%) | 97.71%<br>95.52%<br>99.90%<br>96.89%<br>97.22%<br>96.55%<br>140 / 4 / 5<br>40/40 (100.0%) | 97.58%<br>95.47%<br>99.70%<br>95.89%<br>95.24%<br>96.55%<br>140 / 7 / 5<br>40/40 (100.0%) | **97.97%**<br>**96.34%**<br>**99.60%**<br>**95.92%**<br>94.63%<br>**97.24%**<br>**141 / 8 / 4**<br>**40/40 (100.0%)** | **98.21%**<br>**97.32%**<br>**99.09%**<br>**95.95%**<br>94.04%<br>**97.93%**<br>**142 / 9 / 3**<br>**40/40 (100.0%)** | **`+0.26 pp` (`+0.50 pp` Ep2)**<br>**`+0.82 pp` (`+1.80 pp` Ep2)**<br>`-0.30 pp`<br>`-0.97 pp`<br>`-2.59 pp`<br>**`+0.69 pp` (`+1.38 pp` Ep2)**<br>**`-20% FN` (`-40% FN` Ep2)**<br>**100.0% Specificity** |
| **3. Internal Test Split**<br>*(279 M9 Benchmark Images)* | **mAP@0.50**<br>**Handgun AP**<br>**Knife AP**<br>**F1-Score**<br>**Precision**<br>**Recall**<br>**TP / FP / FN**<br>**Clean Negatives (51)** | 73.34%<br>96.25%<br>50.43%<br>82.27%<br>94.05%<br>73.11%<br>174 / 11 / 64<br>47/51 (92.16%) | 95.22%<br>95.26%<br>95.18%<br>94.02%<br>92.31%<br>95.80%<br>228 / 19 / 10<br>47/51 (92.16%) | 95.08%<br>95.22%<br>94.93%<br>93.06%<br>90.48%<br>95.80%<br>228 / 24 / 10<br>47/51 (92.16%) | **95.50%**<br>**96.11%**<br>**94.90%**<br>**92.53%**<br>89.11%<br>**96.22%**<br>**229 / 28 / 9**<br>**47/51 (92.16%)** | **96.54%**<br>**97.08%**<br>**95.99%**<br>92.25%<br>87.55%<br>**97.48%**<br>**232 / 33 / 6**<br>**47/51 (92.16%)** | **`+0.28 pp` (`+1.32 pp` Ep2)**<br>**`+0.85 pp` (`+1.82 pp` Ep2)**<br>`-0.28 pp` (`+0.81 pp` Ep2)<br>`-1.49 pp`<br>`-3.20 pp`<br>**`+0.42 pp` (`+1.68 pp` Ep2)**<br>**Only 9 FN (`6 FN` Ep2)**<br>**92.16% Specificity** |
| **4. Held Non-Weapon Grid**<br>*(18 Everyday Distractor Cells)* | **Clean Cells / 18**<br>**Specificity Rate** | 16 / 18<br>88.89% | 17 / 18<br>94.44% | 17 / 18<br>94.44% | **16 / 18**<br>**88.89%** | 13 / 18<br>72.22% | Maintained high specificity on held objects |
| **5. Staged Groupmate Val Split**<br>*(44 Tapo CAM 01/02 Frames)* | **Raw Detector Clean**<br>**Post-CCTV Intel Clean** | — | — | 27 / 44 (61.36%)<br>37 / 44 (84.09%) | **25 / 44 (56.82%)**<br>**37 / 44 (84.09%)** | 22 / 44 (50.00%)<br>34 / 44 (77.27%) | **+27.27 pp specificity boost** from CCTV Intelligence Layer |

---

## 4. Real CCTV Video Ground-Truth Benchmark (Before vs. After Dataset Improvements)

Evaluated against frame-by-frame Bounding-Box Ground Truth across **219 true weapon-present frames** (in `HG01..03` and `KN01..03`) and **948 pure non-weapon frames** (in `NW01..04`).

### Table 4A: Aggregate Real CCTV Video Performance (BasicVSR++ Enhanced Weapon Videos + 4 Non-Handgun Videos)

| Model Iteration | Video Precision | Video Recall | Video F1-Score | Video Handgun AP | Video Knife AP | Video mAP@0.50 | True Positives (TP) | False Positives (Weapon / No-Weapon) | False Negatives (FN) | No-Weapon Video Specificity (948 Frames) | Mean TP Confidence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 9 (Legacy Baseline)** | 56.13% | 53.85% | 54.97% | 68.26% | 60.22% | 64.24% | 119 | 93 (66 / 27) | 102 | 97.15% (921/948) | 91.62% |
| **Model 10 (Before Staged & Surv Expansion)** | **100.00%** | 37.10% | 54.13% | 69.73% | 40.61% | 55.17% | 82 | **0 (0 / 0)** | 139 | **100.00% (948/948)** | 87.37% |
| **Model 11 (Staged + Partial Surv Expansion)** | **100.00%** | 61.09% | 75.84% | 81.58% | 62.99% | 72.28% | 135 | **0 (0 / 0)** | 86 | **100.00% (948/948)** | 86.55% |
| **Model 12 (Final Full Surv + Staged & Cup Negs)** | **99.40%** | **75.11%** | **85.57%** | **88.13%** | **69.65%** | **78.89%** | **166** | **1 (1 / 0)** | **55** | **100.00% (948/948)** | **87.43%** |
| **Improvement: Model 10 $\rightarrow$ Model 12** | `-0.60 pp` | **`+38.01 pp` (+102.4% rel.)** | **`+31.44 pp` (+58.1% rel.)** | **`+18.40 pp` (+26.4% rel.)** | **`+29.04 pp` (+71.5% rel.)** | **`+23.72 pp` (+43.0% rel.)** | **`+84 TP` (+102.4% more detections)** | **0 No-Weapon FP** | **`-84 FN` (-60.4% missed frames)** | **100.00% Clean (`0 FP` across all 4 videos)** | **`+0.06%`** |

---

### Table 4B: Per-Video Ground-Truth Breakdown (Original vs. Enhanced Across Models 9, 10, 11, and 12)

| Video ID & CCTV Scenario | Stream Version | GT Weapon Frames / Total | Model 9 (TP / FP / FN, F1, Conf) | Model 10 [Before] (TP / FP / FN, F1, Conf) | Model 11 (TP / FP / FN, F1, Conf) | **Model 12 [Final] (TP / FP / FN, F1, AP/Spec, Conf)** | Key Improvement Achieved |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **HG01 (`230249`)**<br>Store Counter Robbery | Original<br>Enhanced | 121 / 425<br>121 / 425 | 45/28/76 (46.4%, 77.0%)<br>45/33/76 (45.2%, 77.3%) | 3/0/118 (4.8%, 75.0%)<br>14/0/107 (20.7%, 70.4%) | 49/0/72 (57.6%, 70.3%)<br>57/0/64 (64.0%, 71.5%) | **54/1/67 (F1=61.4%, AP=44.2%, Conf=73.5%)**<br>**81/1/40 (F1=79.8%, AP=66.8%, Conf=73.8%)** | **+478.6% TP increase over M10 (14 $\rightarrow$ 81 TP)**; BasicVSR++ boosted F1 from **61.4% $\rightarrow$ 79.8% (+18.4 pp)**. |
| **HG02 (`213340`)**<br>Elevated Retail CCTV | Original<br>Enhanced | 4 / 121<br>4 / 121 | 4/0/0 (100.0%, 83.7%)<br>4/3/0 (72.7%, 88.0%) | 4/0/0 (100.0%, 93.3%)<br>4/0/0 (100.0%, 95.9%) | 4/0/0 (100.0%, 93.7%)<br>4/0/0 (100.0%, 96.6%) | **4/0/0 (F1=100.0%, AP=100.0%, Conf=94.7%)**<br>**4/0/0 (F1=100.0%, AP=100.0%, Conf=96.5%)** | **100% F1 & 100% AP**; BasicVSR++ increased mean confidence from **94.7% $\rightarrow$ 96.5%**. |
| **HG03 (`handgun_test`)**<br>Indoor Armed Aiming | Original<br>Enhanced | 42 / 75<br>42 / 75 | 42/0/0 (100.0%, 99.8%)<br>41/0/1 (98.8%, 99.8%) | 41/0/1 (98.8%, 99.6%)<br>41/0/1 (98.8%, 99.5%) | 41/0/1 (98.8%, 99.7%)<br>41/0/1 (98.8%, 99.6%) | **42/1/0 (F1=98.8%, AP=100.0%, Conf=99.5%)**<br>**41/0/1 (F1=98.8%, AP=97.6%, Conf=99.6%)** | **98.8% F1 & 99.6% mean confidence** across suspect aiming pause (`0` false knife flip-flops). |
| **KN01 (`230047`)**<br>Indoor Bladed Threat | Original<br>Enhanced | 11 / 130<br>11 / 130 | 9/3/2 (78.3%, 86.7%)<br>10/1/1 (90.9%, 85.4%) | 9/0/2 (90.0%, 85.5%)<br>9/0/2 (90.0%, 88.2%) | 9/0/2 (90.0%, 85.3%)<br>9/0/2 (90.0%, 90.6%) | **9/0/2 (F1=90.0%, AP=81.8%, Conf=87.2%)**<br>**9/0/2 (F1=90.0%, AP=81.8%, Conf=91.6%)** | **90.0% F1 & 100% Precision**; BasicVSR++ boosted mean knife confidence from **87.2% $\rightarrow$ 91.6% (+4.4%)**. |
| **KN02 (`220101`)**<br>Corridor Knife Assault | Original<br>Enhanced | 35 / 464<br>35 / 464 | 15/13/20 (47.6%, 80.8%)<br>14/13/21 (45.2%, 84.1%) | 18/0/17 (67.9%, 84.0%)<br>14/0/21 (57.1%, 86.2%) | 14/0/21 (57.1%, 87.4%)<br>20/0/15 (72.7%, 82.5%) | **23/0/12 (F1=79.3%, AP=65.7%, Conf=83.6%)**<br>**27/0/8 (F1=87.1%, AP=77.1%, Conf=82.9%)** | **+92.9% TP surge over M10 Enhanced (14 $\rightarrow$ 27 TP)**; F1 jumped from **57.1% $\rightarrow$ 87.1% (+30.0 pp)**! |
| **KN03 (`230842`)**<br>Unseen Blade Incident | Original<br>Enhanced | 8 / 373<br>8 / 373 | 5/12/3 (40.0%, 93.2%)<br>5/16/3 (34.5%, 89.6%) | 3/0/5 (54.5%, 62.3%)<br>0/0/8 (0.0%, 0.0%) | 4/0/4 (66.7%, 82.8%)<br>4/0/4 (66.7%, 84.9%) | **4/0/4 (F1=66.7%, AP=50.0%, Conf=81.6%)**<br>**4/0/4 (F1=66.7%, AP=50.0%, Conf=85.3%)** | Recovered from **0% F1 in M10 Enhanced up to 66.7% F1 (85.3% conf)** in M11 & M12. |
| **NW01 (`Sample 1`)**<br>Everyday Non-Weapon | Original | 0 / 281 | 0/8/0 (Spec=97.2%, Conf=73.4%) | **0/0/0 (Spec=100.0%)** | **0/0/0 (Spec=100.0%)** | **0/0/0 (Spec=100.0%, 0 FP)** | **100% false-alarm immunity** (eliminated 8 false alarms from M9). |
| **NW02 (`Sample 2`)**<br>Cups & Phones Walkways | Original | 0 / 220 | 0/19/0 (Spec=91.4%, Conf=90.5%) | **0/0/0 (Spec=100.0%)** | **0/0/0 (Spec=100.0%)** | **0/0/0 (Spec=100.0%, 0 FP)** | **100% false-alarm immunity** (eliminated all 19 high-conf 90.5% alarms from M9). |
| **NW03 (`Sample 3`)**<br>Indoor Non-Weapon | Original | 0 / 290 | 0/0/0 (Spec=100.0%) | **0/0/0 (Spec=100.0%)** | **0/0/0 (Spec=100.0%)** | **0/0/0 (Spec=100.0%, 0 FP)** | **100% false-alarm immunity across all models.** |
| **NW04 (`Tapo CAM 01`)**<br>Staged Phones & Desks | Original | 0 / 157 | 0/3/0 (Spec=98.1%, Conf=88.4%) | **0/0/0 (Spec=100.0%)** | **0/0/0 (Spec=100.0%)** | **0/0/0 (Spec=100.0%, 0 FP)** | **100% false-alarm immunity** on staged Tapo camera footage. |

---

## 5. Controlled Ablation Studies (Enhancement & CCTV Intelligence Layer)

### A. Impact of BasicVSR++ Video Enhancement (Model 12 on Unenhanced Original vs. Enhanced Videos)
| Pipeline Condition | Precision | Recall | F1-Score | Video mAP@0.50 | True Positives (TP) | False Negatives (FN) | Mean TP Confidence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unenhanced Original Videos** | 98.55% | 61.54% | 75.77% | 73.62% | 136 | 85 | 86.70% |
| **BasicVSR++ Enhanced Videos** | **99.40%** | **75.11%** | **85.57%** | **78.89%** | **166** | **55** | **87.43%** |
| **Net Enhancement Improvement** | **`+0.85 pp`** | **`+13.57 pp` (+22.1% rel.)** | **`+9.80 pp` (+12.9% rel.)** | **`+5.27 pp`** | **`+30 TP recovered`** | **`-30 FN` (-35.3% misses)** | **`+0.73%` confidence boost** |

### B. Impact of the CCTV Intelligence Layer (Model 12 Raw Detector vs. With CCTV Intelligence Layer)
| Pipeline Condition | Precision | Recall | F1-Score | Total False Positives (FP) | No-Weapon Clean Frames (948 Total) | No-Weapon Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Faster R-CNN Detector Only** | 19.30% | 90.05% | 31.79% | 832 FP (206 WeaponVid / 626 NoWeaponVid) | 368 / 948 | 38.82% |
| **With CCTV Intelligence Layer** | **99.40%** | **75.11%** | **85.57%** | **1 FP (1 WeaponVid / 0 NoWeaponVid)** | **948 / 948** | **100.00%** |
| **Net CCTV Intelligence Improvement** | **`+80.10 pp`** | `-14.94 pp` | **`+53.78 pp` (+169.2% rel.)** | **`-831 FP` (-99.88% false alarms eliminated)** | **`+580 clean frames`** | **`+61.18 pp` specificity (100% clean)** |

