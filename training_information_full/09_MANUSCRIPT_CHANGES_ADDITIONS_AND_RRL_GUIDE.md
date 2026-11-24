# 09 — COMPLETE MANUSCRIPT CHANGES, ADDITIONS & RRL / RELATED STUDIES GUIDE (CHAPTERS 1 TO 5)

**System Name:** FORENSIKADA (Multi-Camera CCTV Weapon Detection System)  
**Purpose of This Document:** A complete, chapter-by-chapter checklist and copy-paste reference for your documentator detailing **every addition, update, and Review of Related Literature (RRL) / Related Studies citation** required in the manuscript to reflect our final post-Model 9 upgrades (**Models 10–13**, **WiSE-FT Weight-Space Calibration**, **BasicVSR++ Super-Resolution**, **CCTV Intelligence Layer**, **Multi-Camera Corroboration**, **16-Stream Video Ground-Truth Evaluation**, and **Forensic Traceability + 24/7 Web Prototype**).

---

## PART 1: OVERVIEW OF WHAT CHANGED SINCE THESIS 1 (MODEL 9)

| Component / Requirement | Original Thesis 1 State (Up to Model 9) | Final Thesis 2 State (Models 10–13 + Full Pipeline) | Which Chapters Need Updates |
| :--- | :--- | :--- | :--- |
| **1. Detector Iterative Progression** | Ended at **Model 9** (`2,377` train images; `62.25%` External 350 mAP). | Expanded through **Model 10** (`3,100` items), **Model 11** (`3,489` items), **Model 12 — Production Video SOTA** (`4,889` items; **`85.57%` Video F1**, **`99.40%` Video Precision**, **`100%` No-Weapon Video Specificity**), and **Model 13 — Static/Household Hard-Negative SOTA** (`6,269` pool; **`74.82%` External 350 mAP**). | **Chapter 3** (Dataset & Training Methodology) & **Chapter 4** (Sections 1, 2, 4) |
| **2. Weight-Space Calibration (WiSE-FT)** | Direct use of raw final epoch weights without weight-space interpolation. | Added **WiSE-FT (Wortsman et al., CVPR 2022)** linear interpolation $\theta(\alpha) = (1-\alpha)\theta_{\text{base}} + \alpha\theta_{\text{ft}}$ ($\alpha=0.30$ for Model 12, $\alpha=0.50$ for Model 13) to prevent catastrophic forgetting of hard-negative specificity. | **Chapter 2** (RRL), **Chapter 3** (Calibration Formula), & **Chapter 4** (Section 1 Sweep Tables) |
| **3. Post-Detection CCTV Intelligence Layer** | Basic frame-by-frame Faster R-CNN bounding-box thresholding. | Added **`CCTVSceneIntelligence`** ([app/services/cctv_intelligence.py](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/app/services/cctv_intelligence.py)): 4-stage post-processing filter combining **(a) Geometric & Aspect-Ratio Gate**, **(b) Static Background Memory (`IoU >= 0.65`)**, **(c) Person-Proximity Contextual Gate**, and **(d) Multi-Frame Temporal Persistence (`min_hits = 3` in `5`-frame window)**. | **Chapter 1** (Scope note), **Chapter 2** (RRL), **Chapter 3** (System Pipeline), & **Chapter 4** (Section 3.2 Controlled Ablation) |
| **4. Video Super-Resolution (BasicVSR++)** | Tested informally without a controlled frame-level ground-truth comparison. | Conducted a **controlled paired Original vs. BasicVSR++ Enhanced ablation** across `1,590` weapon video frames, proving BasicVSR++ boosts Video Recall by **`+13.57 pp`** (`66.82% -> 80.39%`) and Video F1 by **`+8.61 pp`** (`79.77% -> 88.38%`). | **Chapter 2** (RRL), **Chapter 3** (Enhancement Pipeline), & **Chapter 4** (Section 3 Before/After Figures & Table) |
| **5. Real CCTV Video Ground Truth (GT)** | Video testing reported raw alert counts rather than true frame-level accuracy. | Annotated a **16-Stream Real CCTV Video Ground-Truth Benchmark (`2,538` frames)** to measure true **Video Precision, Recall, F1, Video mAP@0.50, and False Alarms/Min**. | **Chapter 3** (Evaluation Metrics) & **Chapter 4** (Section 4 Video Detection Testing) |
| **6. Multi-Camera Corroboration & Forensic Reports** | Conceptual multi-camera workflow. | Evaluated synchronized **`CAM-01` & `CAM-02` pairs** in **table form** (proving cross-camera blind-spot recovery), added **Human-in-the-Loop Observation Review (`Accept`/`Reject`/`Uncertain`)**, **SHA-256 Chain-of-Custody Forensic PDF Reports**, and deployed a **1:1 Desktop + 24/7 Cloud Web Application**. | **Chapter 2** (RRL), **Chapter 3** (Forensic & Multi-Cam Architecture), **Chapter 4** (Sections 5, 6, 7), & **Chapter 5** |

---

## PART 2: CHAPTER-BY-CHAPTER ADDITIONS TO THE MANUSCRIPT

### A. CHAPTER 1 — INTRODUCTION (Minor Alignment Additions Only)
> **Rule for Chapter 1:** Do **not** alter your approved Research Objectives. Only add clarifying sentences in **Scope and Delimitations** and **Conceptual Framework (IPO)** so the panel sees how the new modules fulfill your existing objectives.

1. **Conceptual Framework (Input–Process–Output):**
   - Under **Process**, list the complete 5-stage pipeline:
     1. *Video Ingestion & Cryptographic Hashing (SHA-256)*
     2. *Optional Offline Video Super-Resolution Enhancement (BasicVSR++)*
     3. *Two-Stage Weapon Detection (Faster R-CNN ResNet-50 FPN v2 with WiSE-FT Calibration)*
     4. *Post-Detection CCTV Intelligence, Person-Proximity & Temporal Consistency Validation (`min_hits = 3`)*
     5. *Multi-Camera Cross-Stream Corroboration, Human-in-the-Loop Observation Review (`Accept`/`Reject`/`Uncertain`), and Forensic PDF/CSV Report Generation*
2. **Scope and Delimitations (Add 1 Paragraph):**
   - *"To address false-positive hallucinations common in single-frame object detectors operating on compressed surveillance video, the system integrates a post-detection CCTV Intelligence and Temporal Validation Layer (`CCTVSceneIntelligence`) alongside weight-space ensemble calibration (WiSE-FT). Furthermore, to facilitate external evaluation by video forensic analysts and computing professionals, the desktop forensic workstation is mirrored by a 1:1 web-based evaluation platform."*

---

### B. CHAPTER 2 — REVIEW OF RELATED LITERATURE AND STUDIES (RRL ADDITIONS)
Add the following **6 thematic RRL subsections / paragraphs** to Chapter 2 so that every technical decision in Chapters 3 and 4 is backed by peer-reviewed literature:

#### 1. Two-Stage Firearm & Knife Detection in Surveillance Video & The `0.50` Operational Threshold
- **Citations to Include:**
  - **Olmos, R., Tabik, S., & Herrera, F. (2018).** *Automatic handgun detection alarm in videos using deep learning.* Neurocomputing, 275, 66–72. `https://doi.org/10.1016/j.neucom.2017.05.012`
  - **Fernández-Carrobles, M. M., Deniz, O., & Maroto, F. (2019).** *Gun and knife detection based on Faster R-CNN for video surveillance.* Pattern Recognition and Image Analysis (IbPRIA 2019), LNCS 11868, 441–452. `https://doi.org/10.1007/978-3-030-31321-0_38`
  - **Li, Y., Mao, H., Girshick, R., & He, K. (2021).** *Benchmarking detection transfer learning with vision transformers (Faster R-CNN ResNet-50 FPN v2).* arXiv:2111.11429.
- **Copy-Paste RRL Paragraph for Chapter 2:**
  > *"Olmos et al. (2018) and Fernández-Carrobles et al. (2019) demonstrated that two-stage region-proposal architectures (Faster R-CNN) outperform single-stage detectors in reducing background false alarms when detecting small, distant handguns and bladed weapons in CCTV footage. Modernized backbone designs such as Faster R-CNN ResNet-50 FPN v2 (Li et al., 2021) incorporate multi-scale Feature Pyramid Networks with small anchor scales (`16 px` to `256 px`), enabling localization of weapons occupying less than `1%–3%` of a 1080p surveillance frame at a standard operational confidence threshold of $\tau_{\text{conf}} = 0.50$."*

#### 2. Hard-Negative Mining & Human-Object Contextual Gating (CCTV Intelligence Layer)
- **Citations to Include:**
  - **Ruiz-Santaquiteria, J., Velasco-Mata, A., Vallez, N., Bueno, G., Álvarez-García, J. A., & Deniz, O. (2021).** *Handgun detection using combined human pose and weapon appearance.* IEEE Access, 9, 123815–123826. `https://doi.org/10.1109/ACCESS.2021.3110335`
  - **Velasco-Mata, A., Ruiz-Santaquiteria, J., Vallez, N., & Deniz, O. (2021).** *Using human pose information for handgun detection.* Neural Computing and Applications, 33(24), 17273–17286. `https://doi.org/10.1007/s00521-021-06317-8`
  - **Lamas, A., Tabik, S., Montes, A. C., Pérez-Hernández, F., García, J., Olmos, R., & Herrera, F. (2022).** *Human pose estimation for mitigating false negatives in weapon detection in video-surveillance.* Neurocomputing, 489, 488–503. `https://doi.org/10.1016/j.neucom.2021.12.059`
- **Copy-Paste RRL Paragraph for Chapter 2:**
  > *"A persistent challenge in surveillance weapon detection is the high rate of false positives triggered by everyday handheld distractors (smartphones, wallets, drinking cups, and metallic utensils) and static retail fixtures. Ruiz-Santaquiteria et al. (2021), Velasco-Mata et al. (2021), and Lamas et al. (2022) established that coupling appearance-based weapon detectors with human-proximity and anatomical context constraints significantly suppresses non-weapon false alarms found on tables, shelves, or background structures unconnected to a human actor. This literature directly motivates FORENSIKADA's Person-Proximity Gate, Static Background Memory mask, and targeted household hard-negative mining."*

#### 3. Neural Network Overconfidence Miscalibration & WiSE-FT Weight-Space Ensembling
- **Citations to Include:**
  - **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).** *On calibration of modern neural networks.* Proceedings of the 34th International Conference on Machine Learning (ICML 2017), PMLR 70, 1321–1330.
  - **Küppers, F., Kronenberger, J., Shantia, A., & Haselhoff, A. (2020).** *Multivariate confidence calibration for object detection.* IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW 2020), 1422–1430.
  - **Wortsman, M., Ilharco, G., Kim, J. W., Li, M., Kornblith, S., Roelofs, R., ... & Schmidt, L. (2022).** *Robust fine-tuning of zero-shot models (WiSE-FT: Weight-Space Ensembles for Fine-Tuning).* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022), 7959–7971.
- **Copy-Paste RRL Paragraph for Chapter 2:**
  > *"Guo et al. (2017) and Küppers et al. (2020) showed that deep convolutional detectors often suffer from overconfidence miscalibration, assigning high softmax probabilities (`80%–98%`) to out-of-distribution look-alike objects that share specular or geometric edges with target classes. Furthermore, Wortsman et al. (2022) demonstrated that aggressive fine-tuning on new domain data can induce catastrophic forgetting of previously learned decision boundaries, and proposed Weight-Space Ensembles for Fine-Tuning (WiSE-FT)—linearly interpolating between base and fine-tuned weights, $\theta(\alpha) = (1-\alpha)\theta_{\text{base}} + \alpha\theta_{\text{ft}}$—to achieve superior accuracy across both in-distribution and out-of-distribution benchmarks at zero additional inference cost."*

#### 4. Recurrent Video Super-Resolution for Degraded Surveillance Footage (BasicVSR++)
- **Citations to Include:**
  - **Chan, K. C., Zhou, S., Xu, X., & Loy, C. C. (2022).** *BasicVSR++: Improving video super-resolution with enhanced propagation and alignment.* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022), 5972–5981. `https://doi.org/10.1109/CVPR52688.2022.00588`
- **Copy-Paste RRL Paragraph for Chapter 2:**
  > *"Real-world CCTV cameras frequently suffer from heavy H.264 compression artifacts, motion blur, and low pixel density on distant subjects. Chan et al. (2022) introduced BasicVSR++, a recurrent video super-resolution architecture utilizing second-order grid propagation and optical-flow-guided deformable alignment to reconstruct sharp object boundaries across adjacent video frames. Applying BasicVSR++ prior to object detection restores high-frequency metallic edges on small firearms and knife blades that are otherwise lost in raw CCTV streams."*

#### 5. Spatiotemporal Persistence & Multi-Camera Cross-View Corroboration
- **Citations to Include:**
  - **Wojke, N., Bewley, A., & Paulus, D. (2017).** *Simple online and realtime tracking with a deep association metric (DeepSORT).* IEEE International Conference on Image Processing (ICIP 2017), 3645–3649.
  - **Ristani, E., & Tomasi, C. (2018).** *Features for multi-target multi-camera tracking and re-identification.* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2018), 6036–6046.
- **Copy-Paste RRL Paragraph for Chapter 2:**
  > *"Single-frame detections in video surveillance are vulnerable to transient motion blur and single-Camera line-of-sight occlusions. Temporal track confirmation across consecutive frames (`min_hits >= 3` within a sliding window; Wojke et al., 2017) filters out 1-to-2 frame noise spikes, while synchronized multi-camera corroboration (Ristani & Tomasi, 2018) enables a secondary camera perspective (`CAM-02`) to recover weapon threats when the suspect's body or arm is occluded from the primary camera (`CAM-01`)."*

#### 6. Digital Video Forensics, Chain-of-Custody Integrity & Human-in-the-Loop Verification
- **Citations to Include:**
  - **SWGDE (Scientific Working Group on Digital Evidence). (2018).** *SWGDE Best Practices for Digital Forensic Video Analysis.* Version 1.0.
  - **ISO/IEC 27037:2012.** *Information technology — Security techniques — Guidelines for identification, collection, acquisition, and preservation of digital evidence.*
  - **ISO/IEC 25010:2011.** *Systems and software engineering — Systems and software Quality Requirements and Evaluation (SQuaRE) — System and software quality models.*
- **Copy-Paste RRL Paragraph for Chapter 2:**
  > *"Under ISO/IEC 27037:2012 and SWGDE (2018) guidelines for forensic video analysis, automated AI detections cannot be treated as standalone legal proof without cryptographic chain-of-custody verification (SHA-256 hashing of the original media), transparent logging of filtering criteria, and mandatory human analyst verification (`Accept`, `Reject`, or `Uncertain`) prior to generating an auditable forensic report."*

---

### C. CHAPTER 3 — METHODOLOGY (EXACT SECTIONS & FORMULAS TO ADD)

#### 1. Add Section: "Iterative Dataset Expansion & Hard-Negative Mining (Models 10 to 13)"
Copy the exact dataset table from [01_TRAINING_INFORMATION_AND_DATASETS_ONLY.md](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training_information_full/01_TRAINING_INFORMATION_AND_DATASETS_ONLY.md) showing:
- **Zero-Leakage Guarantee:** `0` images from Open Images, ARMAS, or HuggingFace in Models 5–13, and `0.0%` filename/hash overlap between training sets and the 4 test benchmarks (`Internal 279`, `Unseen CCTV 185`, `External 350`, and `16-Stream Video GT`).
- **Model 12 Dataset (`5,203` full images + `586` ROI crops = `5,789` pool; `4,889` active items/epoch):** Unlocked `+1,400` real surveillance weapon frames (`+788` USRT handguns + `+612` real CCTV knives).
- **Model 13 Dataset (`5,443` full images + `826` ROI crops = `6,269` pool; `5,810` active items/epoch):** Added `+240` full frames + `+240` tight FP ROI crops strictly from Nov 20 non-weapon household scenes (`#01–#06`: black remote, long remote, metallic ladle, rolled white rag, smartphone, wallet), keeping weapon prop scenes (`#07–#11`) 100% excluded.

#### 2. Add Section: "WiSE-FT Weight-Space Ensemble Calibration"
Include the mathematical equation in Chapter 3:
$$\theta_{\text{calibrated}}(\alpha) = (1 - \alpha)\,\theta_{\text{base}} + \alpha\,\theta_{\text{fine-tuned}}, \quad \alpha \in [0, 1]$$
Explain that we swept $\alpha \in \{0.00, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00\}$ on validation/benchmark splits and selected **$\alpha = 0.30$ for Model 12** (`best_weapon_detector_twelfth_model.pth`) and **$\alpha = 0.50$ for Model 13** (`best_weapon_detector_thirteenth_model.pth`).

#### 3. Add Section: "Post-Detection CCTV Intelligence and Temporal Validation Layer (`CCTVSceneIntelligence`)"
Describe the 4 sub-gates implemented in [app/services/cctv_intelligence.py](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/app/services/cctv_intelligence.py):
1. **Geometric & Aspect-Ratio Gate:** Rejects boxes covering `> 22%` of the frame area or extreme slivers (`aspect_ratio > 6.5` or `< 0.15`).
2. **Static Background Memory (`IoU >= 0.65`):** Locks bounding boxes that remain motionless at the same coordinates for `>= 12` consecutive analyzed frames without a nearby person into a static clutter mask.
3. **Person-Proximity Contextual Gate:** Runs a COCO person detector (`conf >= 0.35`) and verifies that candidate weapon boxes fall within an expanded human interaction region (`15%` horizontal / `10%` vertical padding), suppressing unattended objects on tables or shelves.
4. **Multi-Frame Temporal Persistence (`min_hits = 3`, `window = 5`):** Buffers candidate detections over a 5-frame sliding window (`IoU >= 0.25` track matching) and promotes a track from `PENDING_TEMPORAL` to **`CONFIRMED_ALERT`** only when it persists across at least **`3` of `5` frames** (or immediately at `conf >= 0.88`).

#### 4. Add Section: "Standardized Static & Video Ground-Truth Evaluation Protocol"
Copy the mathematical definitions from [03_STANDARDIZED_EVALUATION_METHODOLOGY_MODELS_1_TO_13.md](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training_information_full/03_STANDARDIZED_EVALUATION_METHODOLOGY_MODELS_1_TO_13.md) ($\text{IoU} \ge 0.50$, $\text{Conf} \ge 0.50$, Precision, Recall, F1, AP@0.50, mAP@0.50, No-Weapon Specificity) and clearly state the distinction:
- **Static Image Benchmarks (`Internal 279`, `Unseen CCTV 185`, `External 350`):** Evaluated on **Raw Faster R-CNN (`CCTV Intelligence OFF`)** for a 100% fair single-frame comparison across all 13 models.
- **Real CCTV Video GT Benchmark (`16` streams, `2,538` frames):** Evaluated on the **Full Video Pipeline (`CCTV Intelligence ON`)** plus a controlled ablation comparing `Raw Detector` vs. `+ CCTV Intelligence` and `Original` vs. `BasicVSR++ Enhanced`.

---

### D. CHAPTER 4 — RESULTS AND DISCUSSION (MAPPING EVERY TABLE TO YOUR 7 SECTIONS)

Your Chapter 4 outline has **7 numbered sections**. Every single table is ready in [training_information_full/](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training_information_full/):

| Chapter 4 Section in Your Outline | Exact Tables & Discussions to Insert | Source File to Copy From |
| :--- | :--- | :--- |
| **1. Model Training (Models 1–13)** | • Per-epoch training loss table ($\mathcal{L}_{\text{total}}$, $\mathcal{L}_{\text{classifier}}$, $\mathcal{L}_{\text{box\_reg}}$, $\mathcal{L}_{\text{objectness}}$, $\mathcal{L}_{\text{rpn\_box\_reg}}$) for Models 10–13<br>• WiSE-FT $\alpha$ calibration sweep tables for Model 12 ($\alpha=0.30$) and Model 13 ($\alpha=0.50$)<br>• Master 13-Model progression tables on `External 350`, `Unseen CCTV 185`, and `Internal 279`<br>• Justification of **Model 12 as Production Video SOTA** (`85.57%` Video F1, `99.40%` Precision, `0 FP` on real CCTV) and **Model 13 as Static/Household Hard-Negative SOTA** (`74.82%` External mAP, `44/44` Staged Tapo Val) | [02_TRAINING_METRICS_LOSSES_AND_CALIBRATION_ONLY.md](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training_information_full/02_TRAINING_METRICS_LOSSES_AND_CALIBRATION_ONLY.md)<br>[04_EXTERNAL_350_TEST_SPLIT_METRICS_ONLY.md](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training_information_full/04_EXTERNAL_350_TEST_SPLIT_METRICS_ONLY.md)<br>[05_INTERNAL_279_AND_UNSEEN_CCTV_185_METRICS_ONLY.md](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training_information_full/05_INTERNAL_279_AND_UNSEEN_CCTV_185_METRICS_ONLY.md) |
| **2. Datasets** | • Complete Train / Val / Test image & ROI crop counts per model (`Models 1–13`)<br>• Per-class Handgun / Knife / Hard-Negative breakdown<br>• Zero-leakage audit matrix across all test splits | [01_TRAINING_INFORMATION_AND_DATASETS_ONLY.md](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training_information_full/01_TRAINING_INFORMATION_AND_DATASETS_ONLY.md)<br>[03_STANDARDIZED_EVALUATION_METHODOLOGY_MODELS_1_TO_13.md](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training_information_full/03_STANDARDIZED_EVALUATION_METHODOLOGY_MODELS_1_TO_13.md) |
| **3. BasicVSR++ Enhancement** | • 3 side-by-side Before- vs.-After PNG figures (`HG01`, `KN02`, `MC03`)<br>• Controlled Paired Ablation Table (`6` Original vs. `6` BasicVSR++ Enhanced streams, `1,590` frames) showing **`+13.57 pp` Recall gain** (`66.82% -> 80.39%`) and **`+40` additional true weapon frames recovered** (`179 -> 219` TPs) | [06_REAL_CCTV_VIDEO_GROUND_TRUTH_AND_ABLATIONS_ONLY.md](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training_information_full/06_REAL_CCTV_VIDEO_GROUND_TRUTH_AND_ABLATIONS_ONLY.md)<br>[08_MULTI_CAMERA_TEMPORAL_AND_FORENSIC_VALIDATION_ONLY.md](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training_information_full/08_MULTI_CAMERA_TEMPORAL_AND_FORENSIC_VALIDATION_ONLY.md) |
| **4. Detection Testing (Static & Video GT)** | • Per-class TP / FP / FN & Confusion Matrix breakdown on `External 350`, `Internal 279`, and `Unseen CCTV 185`<br>• **16-Stream Real CCTV Video Ground-Truth Benchmark (`2,538` frames)** comparing Models 9, 10, 11, 12, and 13<br>• **Controlled Ablation of the CCTV Intelligence Layer** showing Precision rising from `81.89% -> 99.40%` (`+17.51 pp`) and No-Weapon False Positives dropping from `55 -> 0 FP` (`100.00%` Specificity) | [04_EXTERNAL_350_TEST_SPLIT_METRICS_ONLY.md](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training_information_full/04_EXTERNAL_350_TEST_SPLIT_METRICS_ONLY.md)<br>[06_REAL_CCTV_VIDEO_GROUND_TRUTH_AND_ABLATIONS_ONLY.md](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training_information_full/06_REAL_CCTV_VIDEO_GROUND_TRUTH_AND_ABLATIONS_ONLY.md)<br>[07_CONFIDENCE_THRESHOLD_HALLUCINATION_AND_LITERATURE_GUIDE.md](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training_information_full/07_CONFIDENCE_THRESHOLD_HALLUCINATION_AND_LITERATURE_GUIDE.md) |
| **5. Temporal & Multi-Camera Validation (Table Form)** | • **Table-Form Multi-Camera Corroboration & Cross-Camera Match Evaluation** across synchronized `CAM-01` & `CAM-02` pairs (`MC01–MC03`, `SCENE007`, and `Nov 20` Tapo pairs)<br>• Examples of **Successful Dual-Camera Corroboration** (`MC03` & `Pair #08`), **Cross-Camera Occlusion Recovery** (`MC02` where `CAM-02` rescues a `CAM-01` blind spot), and **Unsuccessful / Partial Validation** (`Pair #10` cleaver angle & `Pair #01` simultaneous remote trigger) | [08_MULTI_CAMERA_TEMPORAL_AND_FORENSIC_VALIDATION_ONLY.md](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training_information_full/08_MULTI_CAMERA_TEMPORAL_AND_FORENSIC_VALIDATION_ONLY.md) |
| **6. Forensic Validation** | • Traceability records table (`validation_status`, `rejection_reason`, `SHA-256` hash, `bounding_box`, `timestamp_formatted`)<br>• Human-in-the-loop Analyst Review workflow (`Accept` / `Reject` / `Uncertain`) and sample generated Forensic PDF reports | [08_MULTI_CAMERA_TEMPORAL_AND_FORENSIC_VALIDATION_ONLY.md](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training_information_full/08_MULTI_CAMERA_TEMPORAL_AND_FORENSIC_VALIDATION_ONLY.md) |
| **7. Prototype Testing** | • Completed feature matrix of the Desktop App (`mockup_ui/app.py`) and 1:1 24/7 Cloud Web App (`web_app/app.py`)<br>• Functional test results (`10/10` modules `PASS`)<br>• Documented errors, edge-case limitations (overhead specular knife attenuation, zoom-induced static tracker displacement), and mitigations<br>• Summary table placeholder for your **1 Video Forensic Analyst + 30 CCIT Computing Professionals** questionnaire scores | [08_MULTI_CAMERA_TEMPORAL_AND_FORENSIC_VALIDATION_ONLY.md](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training_information_full/08_MULTI_CAMERA_TEMPORAL_AND_FORENSIC_VALIDATION_ONLY.md) |

---

### E. CHAPTER 5 — SUMMARY, CONCLUSIONS, AND RECOMMENDATIONS (KEY POINTS TO ADD)

1. **Conclusions (Backed by Measured Ablations):**
   - **Dataset & WiSE-FT Conclusion:** Expanding the training corpus with real surveillance crops and applying WiSE-FT weight-space calibration ($\alpha = 0.30$ in Model 12; $\alpha = 0.50$ in Model 13) increased real-CCTV Video Recall by **`+38.01 pp`** (`37.10% -> 75.11%`) and Video F1 by **`+31.44 pp`** (`54.13% -> 85.57%`) over Model 10 while reaching **`74.82%` mAP** on the strictly unseen 350-image external benchmark.
   - **BasicVSR++ Super-Resolution Conclusion:** Offline recurrent video super-resolution (BasicVSR++) measurably improves small/distant weapon detection on compressed CCTV footage, increasing frame-level Video Recall by **`+13.57 pp`** (`66.82% -> 80.39%`) and Video mAP by **`+10.00 pp`** (`73.11% -> 83.11%`) on identical video pairs.
   - **CCTV Intelligence & Multi-Camera Conclusion:** Coupling Faster R-CNN with the 4-stage CCTV Intelligence Layer (`CCTVSceneIntelligence`) boosted real-video precision from **`81.89%` to `99.40%` (`+17.51 pp`)** and eliminated **`100%` of false alarms (`55 -> 0 FP`)** across `948` real non-weapon convenience store and walkway frames, while multi-camera corroboration recovered weapon incidents completely occluded from a single camera perspective.
2. **Recommendations for Future Work:**
   - **Keypoint-Guided Hand-Grip Attention:** Integrate lightweight 17-keypoint wrist/finger pose estimation (extending Lamas et al., 2022) directly into the ROI classification head to distinguish cylindrical drink cans held against dark clothing from handgun grips without suppressing thin overhead knife blades.
   - **Homography-Aware Static Background Registration:** Replace pixel-coordinate centroid tracking with feature-based homography compensation so stationary counter trays remain masked even during digital PTZ camera zooms.
   - **TensorRT / ONNX FP16 Quantization for Live BasicVSR++:** Accelerate BasicVSR++ via TensorRT FP16 so super-resolution can run inline on live RTSP streams rather than as an offline pre-processing step.
