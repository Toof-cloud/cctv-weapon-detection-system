# FORENSIKADA: Multi-Camera CCTV Weapon Detection System

**Academic Unit:** National University – Manila | College of Computing and Information Technologies (CCIT) – Computer Science Department  
**Default Production Model:** **Model 12 (`best_weapon_detector_twelfth_model.pth`)** — Faster R-CNN ResNet-50-FPN v2 with Full Surveillance-POV Expansion, Staged Tapo Hard-Negative Immunity, and Spatio-Temporal CCTV Intelligence Layer.

---

## 1. Key Documentation & Adviser Reports

| Document File | Description |
| :--- | :--- |
| **[PHASE1_MODEL12_FINAL_ADVISER_REPORT.md](file:///C:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/PHASE1_MODEL12_FINAL_ADVISER_REPORT.md)** | Complete Phase 1 report: Model 12 training losses, Before-vs-After dataset improvement tables, 16-Stream Video Ground-Truth Benchmark, and BasicVSR++ & CCTV Intelligence ablation studies. |
| **[MODELS_1_TO_12_AUDIT_AND_FINAL_SPECIFICATION.md](file:///C:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/MODELS_1_TO_12_AUDIT_AND_FINAL_SPECIFICATION.md)** | Full technical audit across all 12 model iterations (350 External, 185 Unseen CCTV, 279 Internal, 18-Cell Grid, 44-Frame Staged Val, and 16-Stream Video GT Benchmark). |
| **[Metrics_For_Training.txt](file:///C:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/Metrics_For_Training.txt)** | Plain-text chronological log of training parameters and evaluation metrics from Model 1 through Model 12. |
| **[HOW_TO_RUN_SYSTEM.md](file:///C:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/HOW_TO_RUN_SYSTEM.md)** | Step-by-step instructions for launching the Forensikada Desktop UI, CLI pipeline, and benchmark suites. |

---

## 2. Final Production Performance Summary (Model 12)

* **Unified External Benchmark (`350` Unseen Images — `0.0%` Data Leakage):**
  * **mAP@0.50:** `65.52%` (Calibrated) / `69.42%` (Epoch 2) — up from `62.38%` in Model 10 (`+3.14 pp` to `+7.04 pp`)
  * **F1-Score:** `70.86%` (Project Record) | **Recall:** `72.16%` (`197 TP`) / `77.29%` (`211 TP` Epoch 2)
  * **Handgun AP:** `82.19%` / `82.90%` | **Knife AP:** `48.85%` / `55.93%` (`+5.68 pp` to `+12.76 pp` over Model 10)
* **Strictly Unseen Internal CCTV Benchmark (`185` Images — `0.0%` Data Leakage):**
  * **mAP@0.50:** `97.97%` (Calibrated) / `98.21%` (Epoch 2) | **Recall:** `97.24%` / `97.93%` | **Clean Negatives:** `40/40 (100.00%)`
* **Internal Benchmark Test Split (`279` Images):**
  * **mAP@0.50:** `95.50%` (Calibrated) / `96.54%` (Epoch 2) | **Recall:** `96.22%` / `97.48%` | **Clean Negatives:** `47/51 (92.16%)`
* **Real CCTV Video Ground-Truth Benchmark (`16` Streams, `2,538` Evaluated Frames):**
  * **Video Precision:** `99.40%` | **Video Recall:** `75.11%` (`+38.01 pp` over Model 10) | **Video F1-Score:** `85.57%` (`+31.44 pp` over Model 10) | **Video mAP@0.50:** `78.89%` (`+23.72 pp` over Model 10) | **No-Weapon Video Specificity:** `100.00% (948/948 clean frames across all 4 Non-Handgun videos)`

---

## 3. Repository Structure

* `mockup_ui/` — Primary Forensikada Desktop UI (Single-Video & Multi-Camera Forensic Analysis, Observation Review, SHA-256 PDF/CSV Reports, and 25 automated unit tests).
* `app/` — Core backend services (`detection_service.py`, `cctv_intelligence.py`, `video_enhancement_service.py`, `multi_camera_service.py`, `report_service.py`).
* `expert_evaluation_videos/` — Packaged 28-video Expert Evaluation Library (`HANDGUN/`, `KNIFE/`, `NO_WEAPON/`, `MULTI_CAMERA/`) and frame-by-frame Bounding-Box Ground Truth (`ground_truth/`).
* `training/` — Reproducible dataset preparation, training, calibration, and evaluation pipelines for `model_1_2` through `model_12` and `unified_external_benchmark`.
* `dataset_analysis/` — Model builder (`build_model.py`) and dataset curation utilities.
* `benchmarks/` — Multi-camera and temporal consistency benchmark scripts.

---

## 4. Quick Start

```powershell
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Launch the Forensikada Desktop Application
python mockup_ui/app.py

# 3. Run the automated unit test suite (25 tests)
python -m unittest mockup_ui/test_motion.py mockup_ui/test_settings.py mockup_ui/test_forensic_report.py mockup_ui/test_report_metrics.py mockup_ui.test_mockup.VideoAdapterTests
```
