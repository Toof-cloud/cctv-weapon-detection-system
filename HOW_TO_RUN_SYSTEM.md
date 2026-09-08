# FORENSIKADA: Formal Terminal Execution & Operational Guide

**System Name:** FORENSIKADA (Multi-Camera CCTV Weapon Detection System)  
**Target Environment:** Windows 10/11 Pro 64-bit | Python 3.11.6 | PyTorch with CUDA Acceleration  
**Hardware Reference:** AMD Ryzen 5 5600X / NVIDIA GeForce RTX 5060 Ti (16 GB VRAM)  

---

## 1. Environment Activation & Pre-Flight Checks

Open **PowerShell** in the project root directory (`c:\Users\pc\Documents\THESIS 1\cctv-weapon-detection-system`):

```powershell
# Step 1: Allow local script execution for the terminal session
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Step 2: Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

*(You should see `(.venv)` displayed on your PowerShell prompt).*

### Hardware Acceleration Sanity Check
Verify that PyTorch recognizes your NVIDIA GPU and CUDA acceleration:

```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

**Expected Output:**
```
CUDA Available: True
Device: NVIDIA GeForce RTX 5060 Ti
```

---

## 2. Launching the Graphical User Interface (Desktop App)

To run the interactive desktop application built with PySide6:

```powershell
python app/main.py
```

* **Function:** Opens the dual-camera surveillance dashboard where operators can upload video feeds, monitor live bounding boxes, view chronological detections, and export incident reports.

---

## 3. Running Single-Camera Forensic Pipeline via Terminal

The primary command-line tool is `run_full_pipeline.py`. It accepts input videos, applies detection with Faster R-CNN, filters environmental distractors via the CCTV Intelligence Layer, and exports structured forensic logs.

### Command Syntax
```powershell
python run_full_pipeline.py `
  --video "<path_to_input_video>" `
  --output "<output_folder>" `
  --model "<model_choice_or_path>" `
  --confidence <float> `
  --fps <int> `
  [--no-enhancement] `
  [--no-cctv-intelligence] `
  [--no-temporal-consistency]
```

### Argument Reference
| Parameter | Default | Description |
| :--- | :---: | :--- |
| `--video` *(Required)* | None | Path to the source CCTV recording (e.g. `samples/handgun_test-video.mp4`). |
| `--output` | `outputs/full_pipeline_run` | Directory where videos, CSV logs, keyframe crops, and PDF reports are saved. |
| `--model` | `auto` | Specifies the model: `auto` (uses Model 9), `ninth`, `eighth`, `seventh`, `sixth`, `fifth`, or direct path to a `.pth` file. |
| `--confidence` | `0.50` | Minimum detection confidence threshold (between `0.0` and `1.0`). |
| `--fps` | `5` | Frames-per-second sampling rate (e.g., `10` for high temporal density, `5` for standard throughput). |
| `--no-enhancement` | *False* | **Recommended for speed:** Bypasses BasicVSR++ super-resolution and runs directly on the source footage. |
| `--no-cctv-intelligence` | *False* | Disables motion tracking, person proximity gating, and geometric filtering (raw detector proposals only). |
| `--no-temporal-consistency` | *False* | Disables multi-frame tracklet association and majority voting. |

---

### Practical Terminal Examples (Single Camera)

#### Example A: Rapid Unenhanced Forensic Run (Recommended for Everyday Testing)
Runs Model 9 on authentic handgun robbery CCTV footage at 10 FPS directly without video super-resolution:

```powershell
python run_full_pipeline.py `
  --video "samples/handgun_test-video.mp4" `
  --output "outputs/single_cam_handgun_test" `
  --model "ninth" `
  --confidence 0.50 `
  --fps 10 `
  --no-enhancement
```

#### Example B: Staged Incident Verification with Custom Output Directory
Runs detection on the staircase brandishing scene:

```powershell
python run_full_pipeline.py `
  --video "samples/CAM02_Scene_004.mp4" `
  --output "outputs/single_cam_staged_test" `
  --model "ninth" `
  --confidence 0.50 `
  --fps 10 `
  --no-enhancement
```

---

## 4. Running Multi-Camera Dual-Stream Surveillance via Terminal

To ingest two simultaneous surveillance camera streams (e.g. Entrance `CAM-01` and Hallway `CAM-02`) and reconcile them into a single, unified chronological timeline:

### Command Syntax
```powershell
python run_full_pipeline.py `
  --video "samples/CAM01_Scene 004.mp4" `
  --cam1-id "CAM-01" `
  --camera2 "samples/CAM02_Scene_004.mp4" `
  --cam2-id "CAM-02" `
  --output "outputs/multicam_scene004_run" `
  --model "ninth" `
  --confidence 0.50 `
  --fps 10
```

### What This Produces in `--output`:
1. `annotated_CAM-01_...mp4` – Annotated surveillance video recording for Camera 1 with burned timestamps.
2. `annotated_CAM-02_...mp4` – Annotated surveillance video recording for Camera 2 with burned timestamps.
3. `reports/CASE-..._forensic_detections.csv` – Unified chronological incident log interleaved by `timestamp_seconds`.
4. `reports/CASE-..._forensic_incident_report.pdf` – Evidentiary PDF audit report conforming to forensic computing requirements.
5. `evidence_crops/` – High-resolution cropped images of detected weapon instances organized by camera channel.

---

## 5. Running Automated Benchmark Suites

All reproducible benchmarks are located in the [`benchmarks/`](benchmarks/) folder:

### A. Four Baseline Surveillance Sample Videos
Evaluates the 4 primary thesis test videos (`evaluation_video.mp4`, `NEW_KNIFE_VIDEO_11s.mp4`, `handgun_test-video.mp4`, `CAM02_Scene_004.mp4`):

```powershell
# Run baseline benchmark with Model 9 (State-of-the-Art)
python benchmarks/run_ninth_model_benchmark.py

# Run baseline benchmark with Model 8
python benchmarks/run_eighth_model_benchmark.py

# Run baseline benchmark with Model 5
python benchmarks/run_fifth_model_benchmark.py
```

### B. Ten Real-World Unseen CCTV Incidents
Evaluates all 10 unconstrained web surveillance incident recordings in `samples/unseen_samples/`:

```powershell
# Evaluate all 10 unseen clips with Model 9
python benchmarks/run_unseen_jabez_model9_benchmark.py

# Evaluate all 10 unseen clips with Model 8
python benchmarks/run_unseen_jabez_model8_benchmark.py

# Evaluate all 10 unseen clips with Model 5 (No Enhancement)
python benchmarks/run_unseen_jabez_model5_benchmark.py
```

*Results are saved in:*
* Model 9: `outputs/unseen_jabez_model9_evaluation/`
* Model 8: `outputs/unseen_jabez_model8_evaluation/`
* Model 5: `outputs/unseen_jabez_model5_evaluation/`

---

## 6. Training & Evaluating Models from Scratch

All training routines are located inside [`training/`](training/) organized per model generation.

### Step 1: Prepare & Split Dataset
Generates hard negatives, motion-blurred augmentations, and split annotation CSVs:

```powershell
python training/model_9/prepare_ninth_model_dataset.py
```

### Step 2: Train Model
Runs the training loop with Cosine Annealing learning rate schedule, automatic validation loss tracking, and checkpoint saving:

```powershell
python training/model_9/train_ninth_model.py --epochs 12 --batch-size 2
```

* **Generated Checkpoint:** `best_weapon_detector_ninth_model.pth` *(Saved to root upon validation improvement).*

### Step 3: Evaluate on Independent Test Set
Computes official test set metrics: mAP@0.50, Precision, Recall, F1-Score, Class APs, Confusion Matrix, and Hard Negative Rejection Rate:

```powershell
python training/model_9/evaluate_ninth_model.py
```

---

## 7. Understanding the Forensic Detection Output Logs

Every detection log exported as `forensic_detections.csv` contains 8 standardized evidentiary columns:

| Column | Data Type | Description & Example |
| :--- | :---: | :--- |
| `source_video` | `str` | Name of the video file analyzed (`CAM02_Scene_004.mp4`). |
| `frame_number` | `int` | Exact video frame index (`45`). |
| `timestamp_seconds` | `float` | Exact timestamp in seconds from video start (`1.800`). |
| `camera_id` | `str` | Surveillance camera identifier (`CAM-01`, `CAM-02`). |
| `object_label` | `str` | Classified weapon class (`handgun` or `knife`). |
| `bounding_box` | `list` | Coordinates in `[x1, y1, x2, y2]` pixel format (`[551, 351, 582, 396]`). |
| `confidence_score` | `float` | Model confidence between 0.0 and 1.0 (`0.9393`). |
| `validation_status` | `str` | Forensic filter outcome: `CONFIRMED_ALERT`, `VALIDATED_TEMPORAL`, `SUPPRESSED_TEMPORAL_FLICKER`, `STATIC_BACKGROUND_TRAP`, `GEOMETRIC_OVERSIZED`, `NO_PERSON_PROXIMITY`, or `BELOW_CLASS_THRESHOLD`. |

---

## 8. Common Troubleshooting Tips

1. **PowerShell Script Execution Blocked:**
   * Run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activating `.venv`.
2. **CPU Saturation (100% Core Load):**
   * All benchmark and training scripts automatically cap background worker threads to 8 threads via `os.environ["OMP_NUM_THREADS"] = "8"`. Do not set concurrency higher than your Ryzen 5 5600X's physical core count (6 cores / 12 threads).
3. **Out of Memory (GPU VRAM):**
   * If running higher resolution streams (> 1080p), use `--fps 5` or `--fps 10` to maintain lightweight frame buffer queues. Faster R-CNN on the RTX 5060 Ti consumes ~1.8 GB VRAM during inference.
4. **Where to Check Overall Model Metrics:**
   * Open [Model_Training_and_Testing_Procedures_For_Manuscript.md](Model_Training_and_Testing_Procedures_For_Manuscript.md) or [Metrics_For_Training.txt](Metrics_For_Training.txt).
