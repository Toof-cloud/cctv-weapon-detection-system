# FORENSIKADA Sample Media Directory (`samples/`)

This directory is organized into 5 standardized categories for testing, demonstration, and expert evaluation. *(Note: Large `.mp4`/`.mov` video files inside `samples/` are excluded from Git tracking via `.gitignore`, while this index is version-controlled.)*

---

## Directory Structure

### `01_expert_evaluation_library/` *(Primary Library for Expert Evaluation & Web App)*
Mirrors the packaged **28-video Expert Evaluation Library** (`expert_evaluation_videos/`):
* **`HANDGUN/`:** 3 Handgun CCTV pairs (`HG01_Original.mp4` / `HG01_Enhanced.mp4`, `HG02_Original.mp4` / `HG02_Enhanced.mp4`, `HG03_Original.mp4` / `HG03_Enhanced.mp4`).
* **`KNIFE/`:** 3 Knife CCTV pairs (`KN01_Original.mp4` / `KN01_Enhanced.mp4`, `KN02_Original.mp4` / `KN02_Enhanced.mp4`, `KN03_Original.mp4` / `KN03_Enhanced.mp4`).
* **`NO_WEAPON/`:** 4 Non-Handgun / No-Weapon verification videos (`NW01.mp4` through `NW04.mp4`).
* **`MULTI_CAMERA/`:** 3 Synchronized dual-camera (`CAM 01` & `CAM 02`) scenes (`MC01` No-Weapon, `MC02` Handgun, `MC03` Knife) in both Original and BasicVSR++ Enhanced formats.

### `02_core_test_videos/`
Core single-video smoke-test and verification clips used during development:
* `handgun_test-video.mp4`, `evaluation_video.mp4`, `knife_test_long-video.mp4`
* `Screen Recording 2026-09-10 213340 -HANDGUN - Trim.mp4`
* `CAM01_Scene 004.mp4`, `CAM02_Scene_004.mp4`
* `5.mp4`, `5 (1).mp4`, `1869451F17C5_20260920155825616_cam0.mp4`
* `test_10s_general.mp4`, `test_10s_knife.mp4`, `test_10s_negative.mp4`

### `03_staged_and_multicamera/`
Dual-camera and staged indoor surveillance recordings:
* `staged_dataset/`, `NEW_STAGED_CAM-01/`, `NEW_STAGED_CAM-02/`, `JABEZ-HOUSE-CCTV/`, `MULTI-CAMERA_SESSION/`

### `04_test_images/`
Static image verification batches:
* `handgun_test_images/`, `knives_test_images/`, `negative_test_images/`, `004eb6ca27183afe.jpg`

### `05_historical_consultations/`
Archived consultation, defense, and pre-enhancement source collections:
* `DEFENSE/`, `Consultation_for_Sir_Labanan/`, `consultation_final/`, `enhanced_videos_output/`, `full_run_videos_with_screen_recording/`, `To_enhance/`, `NEW-VIDEOS/`, `unseen_samples/`, `costi_demo_clips/`

