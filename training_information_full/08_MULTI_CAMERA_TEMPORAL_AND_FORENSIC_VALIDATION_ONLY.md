# 08 — MULTI-CAMERA CORROBORATION, TEMPORAL VALIDATION & FORENSIC REPORT RELIABILITY ONLY (TABLE FORM)

**System Name:** FORENSIKADA (Multi-Camera CCTV Weapon Detection System)  
**Production Video Detector:** **Model 12 (`best_weapon_detector_twelfth_model.pth`, WiSE-FT $\alpha=0.30$)**  
**Source JSON & Annotated Outputs:**
- [multicam_corroboration_evaluation.json](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/outputs/03_chapter4_benchmarks_and_figures/multicam_corroboration_evaluation.json)
- [outputs/01_final_model12_annotated_library/MULTI_CAMERA/](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/outputs/01_final_model12_annotated_library/MULTI_CAMERA/)
- [outputs/03_chapter4_benchmarks_and_figures/basicvsr_before_after_figures/](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/outputs/03_chapter4_benchmarks_and_figures/basicvsr_before_after_figures/)

---

## 1. Table-Form Multi-Camera Corroboration & Cross-Camera Match Evaluation (Chapter 4 Section 5)

Evaluated across **12 Synchronized Dual-Camera Test Scenarios** (`6` canonical `MC01–MC03` Original & BasicVSR++ Enhanced pairs + `6` Nov 20 Tapo 1080p synchronized `CAM-01`/`CAM-02` pairs) at $\text{Confidence} \ge 0.50$ with the CCTV Intelligence Layer ON and cross-camera temporal matching window $\Delta t \le 2.5\text{ s}$:

| Scenario ID | Multi-Camera Test Scenario | Target Class | Stream Version | `CAM-01` Alerts (Peak Conf) | `CAM-02` Alerts (Peak Conf) | Combined Alerts | Synchronized Cross-Camera Matches (`CAM1, CAM2`) | Missed by `CAM-01`, Recovered by `CAM-02` | Missed by `CAM-02`, Recovered by `CAM-01` | Cross-Camera Class Mismatches | Dual-Camera Corroborated? | Forensic Validation Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`MC01_Original`** | Staged Handgun Pair (`SCENE004`) | Handgun | Original | `4` (`99.2%`) | `13` (`99.5%`) | `17` | `4, 8` | **`+5` frames** | `0` | `0` | **YES** | **Successful Dual-Camera Corroboration** |
| **`MC01_Enhanced`** | Staged Handgun Pair (`SCENE004`) | Handgun | **BasicVSR++** | `4` (`99.2%`) | `10` (`99.4%`) | `14` | `4, 8` | **`+2` frames** | `0` | `0` | **YES** | **Successful Dual-Camera Corroboration** |
| **`MC02_Original`** | Staged Knife Pair 1 (`SCENE005`) | Knife | Original | `0` (`0.0%`) | `15` (`99.5%`) | `15` | `0, 0` | **`+15` frames (100% rescue)** | `0` | `0` | No *(Single-Cam Rescue)* | **Blind-Spot Occlusion on CAM-01 Rescued by CAM-02** |
| **`MC02_Enhanced`** | Staged Knife Pair 1 (`SCENE005`) | Knife | **BasicVSR++** | `0` (`0.0%`) | `15` (`99.1%`) | `15` | `0, 0` | **`+15` frames (100% rescue)** | `0` | `0` | No *(Single-Cam Rescue)* | **Blind-Spot Occlusion on CAM-01 Rescued by CAM-02** |
| **`MC03_Original`** | Staged Knife Pair 2 (`SCENE012`) | Knife | Original | `3` (`99.4%`) | `10` (`97.0%`) | `13` | `3, 10` | `0` | `0` | `0` | **YES** | **Successful Dual-Camera Corroboration** |
| **`MC03_Enhanced`** | Staged Knife Pair 2 (`SCENE012`) | Knife | **BasicVSR++** | `3` (`99.7%`) | **`41` (`99.1%`)** | **`44`** | **`3, 17`** | **`+24` frames (`+310%` boost)** | `0` | `0` | **YES** | **BasicVSR++ Multiplies CAM-02 Detections 4.1×** |
| **`NOV20_S05_Phone`** | Staged Smartphone Distractor | No-Weapon | Tapo 1080p | `1` (`59.3%`) | `0` (`0.0%`) | `1` | `0, 0` | `0` | `0` *(1 uncorroborated FP)* | `0` | **NO (Rejected)** | **Cross-Camera Corroboration Filters Out Single-Cam FP** |
| **`NOV20_S06_Wallet`** | Staged Wallet Distractor | No-Weapon | Tapo 1080p | `0` (`0.0%`) | `1` (`55.6%`) | `1` | `0, 0` | `0` *(1 uncorroborated FP)* | `0` | `0` | **NO (Rejected)** | **Cross-Camera Corroboration Filters Out Single-Cam FP** |
| **`NOV20_S07_Pistol`** | Staged Silver Pistol Prop | Weapon Prop | Tapo 1080p | `30` (`97.0%`) | `45` (`98.8%`) | `75` | **`30, 45`** | `0` | `0` | `0` | **YES** | **High-Density Dual-Camera Corroboration** |
| **`NOV20_S08_Knife`** | Staged Kitchen Knife Pickup | Knife | Tapo 1080p | `3` (`98.9%`) | `34` (`99.7%`) | `37` | **`3, 28`** | **`+6` frames** | `0` | `0` | **YES** | **CAM-02 Compensates for CAM-01 Static-Trap Delay** |
| **`NOV20_S10_Cleaver`** | Staged Butcher Cleaver | Knife | Tapo 1080p | `48` (`99.7%`) | `4` (`93.2%`) | `52` | **`48, 4`** | `0` | **`+44` frames** | `0` | **YES** | **CAM-01 Compensates for CAM-02 Viewing Angle** |
| **`NOV20_S11_DualKnives`** | Staged Dual Knives Held Up | Knife | Tapo 1080p | `8` (`99.3%`) | `3` (`99.6%`) | `11` | **`8, 3`** | `0` | **`+5` frames** | `0` | **YES** | **Successful Dual-Camera Corroboration** |

---

## 2. Controlled Comparative Testing Summary (Enhancement vs. Temporal vs. Multi-Camera)

| Controlled Comparative Factor | Baseline (Without Module) | Treatment (With Module Active) | Measured Performance Gain | Forensic Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| **1. BasicVSR++ Video Enhancement**<br>*(16-Stream Video GT + `MC03`)* | `61.54%` Video Recall · `75.77%` F1 · `10` alerts on `MC03_CAM02_Original` | **`75.11%` Video Recall · `85.57%` F1 · `41` alerts on `MC03_CAM02_Enhanced`** | **`+13.57 pp` Recall (`+30 TP`), `+9.80 pp` F1, `4.1×` alerts on `MC03_CAM02`** | Super-resolution reconstructs blade/barrel edges lost to CCTV compression. |
| **2. Temporal & Spatiotemporal Validation**<br>*(CCTV Intelligence Layer, `min_hits=3`)* | `19.30%` Video Precision · `832` Raw FPs · `38.82%` No-Weapon Specificity | **`99.40%` Video Precision · `1` FP · `100.00%` (`948/948`) No-Weapon Specificity** | **`-99.88%` False Positives (`832 → 1`), `+80.10 pp` Precision, `+61.18 pp` Specificity** | Eliminates single-frame flickers, static background traps, and non-person clutter. |
| **3. Multi-Camera Corroboration**<br>*(10 Weapon Pairs + 2 Distractor Pairs)* | Single camera alone (`CAM-01` misses `MC02` entirely: `8/10` weapon scenes; `1` borderline FP on `Phone`/`Wallet`) | **Dual-Camera Union (`10/10 = 100%` weapon scene recall) + Cross-Camera Intersection (`0/2 = 0%` distractor false corroboration)** | **`+20.0 pp` Scene-Level Recall (`80% → 100%`) & `100%` elimination of single-angle borderline FPs (`Phone` & `Wallet`)** | Second viewpoint rescues blind-spot occlusions (`MC02`, `S08`, `S10`) while filtering uncorroborated single-camera noise. |

---

## 3. High-Resolution BasicVSR++ Before-vs-After Sample Figures for Chapter 4

Ready-to-embed side-by-side PNG figures generated in [outputs/03_chapter4_benchmarks_and_figures/basicvsr_before_after_figures/](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/outputs/03_chapter4_benchmarks_and_figures/basicvsr_before_after_figures/):
1. **Handgun Hallway Draw (`HG03`, Frame 84):** [Figure_BasicVSR_HG03_Handgun_Before_vs_After.png](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/outputs/03_chapter4_benchmarks_and_figures/basicvsr_before_after_figures/Figure_BasicVSR_HG03_Handgun_Before_vs_After.png)
2. **Fast Indoor Knife Swing (`KN01`, Frame 122):** [Figure_BasicVSR_KN01_Knife_Before_vs_After.png](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/outputs/03_chapter4_benchmarks_and_figures/basicvsr_before_after_figures/Figure_BasicVSR_KN01_Knife_Before_vs_After.png)
3. **Multi-Camera Handgun (`MC01_CAM02`, Frame 44):** [Figure_BasicVSR_MC01_CAM02_Before_vs_After.png](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/outputs/03_chapter4_benchmarks_and_figures/basicvsr_before_after_figures/Figure_BasicVSR_MC01_CAM02_Before_vs_After.png)

---

## 4. Forensic Report Reliability, Traceability & Human-in-the-Loop Verification

| Forensic Requirement | Implementation in FORENSIKADA | Verification Result |
| :--- | :--- | :--- |
| **Cryptographic Chain-of-Custody** | Computes 64-character **SHA-256 digital hash** of every input video file (`CAM-01`, `CAM-02`, or Custom Upload) prior to inference and embeds it in the PDF header and JSON audit log. | **100% deterministic (`28/28` library videos verified)** |
| **Frame-to-Report Traceability** | Every detection row logs exact `camera_id`, `frame_number`, `timestamp_formatted` (`MM:SS.mmm`), `object_label`, `confidence_score`, `bounding_box [x1, y1, x2, y2]`, `validation_status`, and `analyst_decision`. | **100% match between CSV, JSON, UI table, and generated ReportLab PDF** |
| **Human-in-the-Loop Observation Review** | Every flagged frame can be inspected in the UI and tagged as **`Accept`**, **`Reject`**, or **`Uncertain`** with optional analyst notes and UTC review timestamp before exporting the final Forensic PDF. | **Verified across Desktop App (`25/25` unit tests) & Web App (`0` exceptions)** |
