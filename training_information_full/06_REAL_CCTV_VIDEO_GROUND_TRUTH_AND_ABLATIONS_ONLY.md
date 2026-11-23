# 06 — 16-STREAM REAL CCTV VIDEO GROUND-TRUTH & CONTROLLED ABLATION METRICS ONLY

**System Name:** FORENSIKADA (Multi-Camera CCTV Weapon Detection System)  
**Ground-Truth Directory:** [expert_evaluation_videos/ground_truth/](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/expert_evaluation_videos/ground_truth/)  
**Evaluation Script:** [evaluate_expert_videos_with_ground_truth.py](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training/model_12/evaluate_expert_videos_with_ground_truth.py) & [evaluate_model13_video_gt.py](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training/model_13/evaluate_model13_video_gt.py)  
**Benchmark Composition:** **16 Real CCTV Video Streams (`2,538` evaluated frames at `15 FPS`)**:
- `3 Handgun CCTV Pairs` (`HG01`, `HG02`, `HG03` — Unenhanced Original & BasicVSR++ Enhanced = 6 streams)
- `3 Knife CCTV Pairs` (`KN01`, `KN02`, `KN03` — Unenhanced Original & BasicVSR++ Enhanced = 6 streams)
- `4 Non-Handgun / No-Weapon CCTV Videos` (`NW01`, `NW02`, `NW03`, `NW04` = `948` pure non-weapon frames)

---

## 1. Aggregate Real CCTV Video Performance Across Models 9, 10, 11, 12, and 13
*(Evaluated on BasicVSR++ Enhanced Weapon Videos + 4 No-Weapon Videos at $\text{Confidence} \ge 0.50$, $\text{IoU} \ge 0.35$, with CCTV Intelligence Layer ON).*

| Model Iteration | Video Precision | Video Recall | Video F1-Score | Video Handgun AP | Video Knife AP | Video mAP@0.50 | TP | FP (Weapon / No-Weapon) | FN | No-Weapon Video Specificity (948 Frames) | Mean TP Confidence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 9 (Thesis 1 Legacy)** | 56.13% | 53.85% | 54.97% | 68.26% | 60.22% | 64.24% | 119 | 93 (`66 W / 27 NW`) | 102 | 97.15% (`921 / 948`) | 91.62% |
| **Model 10 (Before Expansion)** | **100.00%** | 37.10% | 54.13% | 69.73% | 40.61% | 55.17% | 82 | **0 (`0 W / 0 NW`)** | 139 | **100.00% (`948 / 948`, 0 FP)** | 87.37% |
| **Model 11 (Partial Expansion)** | **100.00%** | 61.09% | 75.84% | 81.58% | 62.99% | 72.28% | 135 | **0 (`0 W / 0 NW`)** | 86 | **100.00% (`948 / 948`, 0 FP)** | 86.55% |
| **Model 12 ($\alpha=0.30$ — Video SOTA)** | **99.40%** | **75.11%** | **85.57%** | **88.13%** | **69.65%** | **78.89%** | **166** | **1 (`1 W / 0 NW`)** | **55** | **100.00% (`948 / 948`, 0 FP)** | **87.43%** |
| **Model 13 ($\alpha=0.50$ — Household HN Ablation)** | 68.48% | 51.13% | 58.55% | 62.86% | 11.95% | 37.41% | 113 | 52 (`34 W / 18 NW`) | 108 | 98.10% (`930 / 948`, 18 FP) | 88.24% |

---

## 2. Per-Video Ground-Truth Breakdown (Model 10 vs. Model 12 Video SOTA vs. Model 13)
*(Enhanced Weapon Videos + 4 No-Weapon Videos, CCTV Intelligence Layer ON, $\text{Confidence} \ge 0.50$).*

| Video ID | Category & Description | GT Weapon Frames | **Model 10** (`TP / FP / FN` · `F1` · `AP/Spec`) | **Model 12 Video SOTA** (`TP / FP / FN` · `F1` · `AP/Spec`) | **Model 13 HN Ablation** (`TP / FP / FN` · `F1` · `AP/Spec`) |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **`HG01_Enhanced`** | Handgun CCTV 1 — Store Counter Robbery (`230249`) | 121 / 425 | `40 / 0 / 81` · `49.7%` F1 · `40.5%` AP | **`84 / 0 / 37` · `82.0%` F1 · `73.5%` AP** | `63 / 32 / 58` · `58.3%` F1 · `46.0%` AP |
| **`HG02_Enhanced`** | Handgun CCTV 2 — Night Street Sidewalk (`213340`) | 4 / 121 | `0 / 0 / 4` · `0.0%` F1 · `71.0%` AP | **`3 / 0 / 1` · `85.7%` F1 · `93.3%` AP** | `3 / 2 / 1` · `66.7%` F1 · `45.0%` AP |
| **`HG03_Enhanced`** | Handgun CCTV 3 — Hallway Draw (`handgun_test-video`) | 42 / 77 | `41 / 0 / 1` · `98.8%` F1 · `97.6%` AP | **`41 / 1 / 1` · `97.6%` F1 · `97.6%` AP** | **`41 / 0 / 1` · `98.8%` F1 · `97.6%` AP** |
| **`KN01_Enhanced`** | Knife CCTV 1 — Fast Indoor Swing (`230047`) | 11 / 100 | `0 / 0 / 11` · `0.0%` F1 · `27.3%` AP | **`9 / 0 / 2` · `90.0%` F1 · `90.9%` AP** | `3 / 0 / 8` · `42.9%` F1 · `27.3%` AP |
| **`KN02_Enhanced`** | Knife CCTV 2 — Street Confrontation (`220101`) | 35 / 325 | `1 / 0 / 34` · `5.6%` F1 · `32.0%` AP | **`24 / 0 / 11` · `81.4%` F1 · `68.0%` AP** | `3 / 0 / 32` · `15.8%` F1 · `8.6%` AP |
| **`KN03_Enhanced`** | Knife CCTV 3 — Narrow Corridor (`230842`) | 8 / 373 | `0 / 0 / 8` · `0.0%` F1 · `62.5%` AP | **`5 / 0 / 3` · `76.9%` F1 · `50.0%` AP** | `0 / 0 / 8` · `0.0%` F1 · `0.0%` AP |
| **`NW01`** | No-Weapon 1 — Indoor Customer Interaction | 0 / 282 | **`0 FP` · `100.0%` Spec** | **`0 FP` · `100.0%` Spec** | **`0 FP` · `100.0%` Spec** |
| **`NW02`** | No-Weapon 2 — Convenience Store Counter & Cup | 0 / 221 | **`0 FP` · `100.0%` Spec** | **`0 FP` · `100.0%` Spec** | `15 FP` · `93.2%` Spec *(drink can & tray)* |
| **`NW03`** | No-Weapon 3 — Pedestrian Walkway | 0 / 290 | **`0 FP` · `100.0%` Spec** | **`0 FP` · `100.0%` Spec** | **`0 FP` · `100.0%` Spec** |
| **`NW04`** | No-Weapon 4 — Close-Up Empty Hand Movement | 0 / 155 | **`0 FP` · `100.0%` Spec** | **`0 FP` · `100.0%` Spec** | `3 FP` · `98.1%` Spec |

---

## 3. Controlled Video Ablation Studies (Proving BasicVSR++ & CCTV Intelligence Layer)

### Ablation Study 1: With vs. Without BasicVSR++ Super-Resolution Enhancement (on Model 12 Video SOTA)

| Pipeline Configuration | Video Precision | Video Recall | Video F1-Score | Video mAP@0.50 | TP / FP / FN | No-Weapon Specificity | Measured Net Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Without BasicVSR++ (Original Unenhanced CCTV)** | 98.55% | 61.54% | 75.77% | 73.62% | 136 / 2 / 85 | 948 / 948 (100.00%) | Baseline unenhanced stream |
| **With BasicVSR++ Enhancement** | **99.40%** | **75.11%** | **85.57%** | **78.89%** | **166 / 1 / 55** | **948 / 948 (100.00%)** | **`+13.57 pp` Recall (`+30 TP`), `+9.80 pp` F1, `+5.27 pp` mAP** |

### Ablation Study 2: Raw Faster R-CNN Detector vs. Full Pipeline (+ CCTV Intelligence & Temporal Validation)

| Pipeline Configuration | Video Precision | Video Recall | Video F1-Score | TP / FP / FN | No-Weapon Clean Frames (948 Total) | Measured Net Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Raw Faster R-CNN Detector Only (No CCTV Intelligence)** | 19.30% | 90.05% | 31.79% | 199 / **832** / 22 | 368 / 948 (38.82%) | Unfiltered frame-by-frame proposals |
| **Full Pipeline (+ CCTV Intelligence & Temporal Filter)** | **99.40%** | **75.11%** | **85.57%** | **166 / 1 / 55** | **948 / 948 (100.00%)** | **`-99.88%` False Positives (`832 → 1`), `+80.10 pp` Precision, `+61.18 pp` Specificity** |
