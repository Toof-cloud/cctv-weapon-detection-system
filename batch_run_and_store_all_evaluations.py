"""
Batch Full-Run Evaluation & Storage Script.

Runs the complete forensic detection pipeline with Model V11 and CCTV Intelligence
across all test videos in 'samples/full_run videos_with_screen_recording/'.
Stores:
1. Annotated surveillance videos (.mp4) with color-coded bounding boxes and OSD banners.
2. Detected frame evidence captures (.png).
3. Structured detection audit logs (.csv).
4. Forensic reports (.html and .txt summary).
5. Comprehensive evaluation summary matrix (.csv and console markdown).
"""

from __future__ import annotations

import csv
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Setup workspace paths
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import torch
from run_full_pipeline import detect_video_with_model, generate_forensic_summary
from app.services.report_service import ReportService

V11_MODEL_PATH = ROOT_DIR / "research" / "model_improvement" / "checkpoints" / "best_candidate_model_v11.pth"
EVAL_BASE_DIR = ROOT_DIR / "outputs" / "full_run_evaluation"
VIDEOS_OUT_DIR = EVAL_BASE_DIR / "annotated_videos"
FRAMES_OUT_DIR = EVAL_BASE_DIR / "annotated_frames"
CSV_OUT_DIR = EVAL_BASE_DIR / "reports_csv"
FORENSIC_OUT_DIR = EVAL_BASE_DIR / "reports_forensic"

# Ensure target directories exist
for p in (VIDEOS_OUT_DIR, FRAMES_OUT_DIR, CSV_OUT_DIR, FORENSIC_OUT_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Define test videos and their ground truth labels
SAMPLES_DIR = ROOT_DIR / "samples" / "full_run videos_with_screen_recording"

GROUND_TRUTH_MAPPING = {
    # 1080p Knife series
    "1080 knife/1869451F17C5_20260920155928685_cam0.mp4": "knife",
    "1080 knife/1869451F17C5_20260920160055605_cam0.mp4": "knife",
    "1080 knife/1869451F17C5_20260920160128392_cam0.mp4(WOODEN).mp4": "knife",
    # 720p Knife series
    "720 w knife/1869451F17C5_20260920155724609_cam0.mp4": "knife",
    "720 w knife/1869451F17C5_20260920155759078_cam0.mp4": "knife",
    "720 w knife/1869451F17C5_20260920155825616_cam0.mp4(WOODEN).mp4": "knife",
    # 720p Negative Control series (No weapons)
    "720 w-o knife/1869451F17C5_20260920155118064_cam0(1).mp4": "none",
    "720 w-o knife/1869451F17C5_20260920155118064_cam0.mp4": "none",
    "720 w-o knife/1869451F17C5_20260920155218941_cam0(1).mp4": "none",
    "720 w-o knife/1869451F17C5_20260920155218941_cam0.mp4": "none",
    "720 w-o knife/1869451F17C5_20260920155258445_cam0(1).mp4": "none",
    "720 w-o knife/1869451F17C5_20260920155258445_cam0.mp4": "none",
    "720 w-o knife/1869451F17C5_20260920155440307_cam0(1).mp4": "none",
    "720 w-o knife/1869451F17C5_20260920155440307_cam0.mp4": "none",
    "720 w-o knife/1869451F17C5_20260920155503889_cam0.mp4": "none",
    # Real Surveillance CCTV series
    "cctv handgun/Screen Recording 2026-09-07 230249.mp4": "handgun",
    "cctv knife/Screen Recording 2026-09-07 230047.mp4": "knife",
    # Staged Multi-Camera Datasets
    "jabez knife multicamera (staged dataset)/CAM2_SCENE004.mp4": "knife",
    "jabez knife multicamera (staged dataset)/MultiCamera_FULL_RUN.mp4": "knife",
    "tyrone handgun (staged dataset)/CAM2_SCENE012.mp4": "handgun",
}


def main():
    print("=" * 80)
    print("BATCH FORENSIC EVALUATION & EVIDENCE ARCHIVAL PIPELINE")
    print(f"Model Checkpoint: {V11_MODEL_PATH.name}")
    print(f"Target Output Directory: {EVAL_BASE_DIR}")
    print(f"Confidence Threshold: 0.40 | CCTV Intelligence: ENABLED | Temporal Consistency: ENABLED")
    print("=" * 80)

    if not V11_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Candidate Model V11 checkpoint not found: {V11_MODEL_PATH}")

    evaluation_matrix = []
    start_total_time = time.time()

    video_items = list(GROUND_TRUTH_MAPPING.items())
    total_videos = len(video_items)

    for idx, (rel_path, ground_truth) in enumerate(video_items, 1):
        video_file = SAMPLES_DIR / rel_path
        if not video_file.is_file():
            print(f"[{idx}/{total_videos}] SKIPPING (file not found): {rel_path}")
            continue

        clean_name = video_file.stem.replace("(", "_").replace(")", "_").replace(" ", "_")
        folder_prefix = video_file.parent.name.replace(" ", "_").replace("(", "").replace(")", "")
        item_slug = f"{folder_prefix}_{clean_name}"

        annotated_video_path = VIDEOS_OUT_DIR / f"{item_slug}_annotated.mp4"
        csv_report_path = CSV_OUT_DIR / f"{item_slug}_detections.csv"
        detected_frames_folder = FRAMES_OUT_DIR / item_slug
        summary_txt_path = FORENSIC_OUT_DIR / f"{item_slug}_summary.txt"

        print(f"\n[{idx}/{total_videos}] Processing: {video_file.parent.name} / {video_file.name}")
        print(f"  Ground Truth: {ground_truth.upper()}")
        print(f"  Output Video: {annotated_video_path.name}")
        print(f"  Output Frames: {detected_frames_folder.name}/")

        cap = cv2.VideoCapture(str(video_file))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / max(1.0, fps)
        cap.release()

        v_start = time.time()

        try:
            pipeline_result = detect_video_with_model(
                input_path=video_file,
                output_path=annotated_video_path,
                model_path=V11_MODEL_PATH,
                confidence_threshold=0.40,
                analysis_fps=min(5, math.ceil(fps)),
                save_detected_frames=True,
                csv_output_path=csv_report_path,
                enable_cctv_intelligence=True,
                enable_temporal_consistency=True,
                min_temporal_hits=2,
                camera_id="CAM-01",
            )
        except Exception as e:
            print(f"  ERROR processing {video_file.name}: {e}")
            continue

        elapsed = time.time() - v_start

        # Move/organize detected frames folder if saved in default location
        default_frames_dir = annotated_video_path.parent / f"{annotated_video_path.stem}_detected_frames"
        if default_frames_dir.is_dir() and default_frames_dir != detected_frames_folder:
            if detected_frames_folder.exists():
                for f in detected_frames_folder.glob("*.png"):
                    f.unlink()
            else:
                detected_frames_folder.mkdir(parents=True, exist_ok=True)
            saved_pngs = list(default_frames_dir.glob("*.png"))
            for png in saved_pngs:
                target_png = detected_frames_folder / png.name
                png.replace(target_png)
            try:
                default_frames_dir.rmdir()
            except Exception:
                pass
            frame_png_count = len(saved_pngs)
        else:
            frame_png_count = len(list(detected_frames_folder.glob("*.png"))) if detected_frames_folder.is_dir() else 0

        # Generate Forensic Summary
        if csv_report_path.exists():
            forensic_summary = generate_forensic_summary(csv_report_path, summary_txt_path)
            confirmed_counts = forensic_summary.get("class_counts", {})
            total_confirmed = forensic_summary.get("confirmed_alerts", sum(confirmed_counts.values()))
            total_suppressed = forensic_summary.get("suppressed_traps", 0)
        else:
            confirmed_counts = {}
            total_confirmed = 0
            total_suppressed = 0

        # Determine Primary Detected Class
        if confirmed_counts:
            detected_class = max(confirmed_counts.keys(), key=lambda k: confirmed_counts[k])
        else:
            detected_class = "none"

        # Determine Evaluation Pass/Fail Status
        if ground_truth == "none":
            status = "PASS (CLEAN)" if total_confirmed == 0 else f"FAIL ({total_confirmed} false alerts)"
        else:
            if detected_class == ground_truth:
                status = f"PASS ({detected_class.upper()} {total_confirmed} alerts)"
            elif total_confirmed == 0:
                status = "FAIL (MISSED DETECTION)"
            else:
                status = f"FAIL (MISCLASSIFIED as {detected_class.upper()})"

        print(f"  Result: {status} | Frames with alerts: {frame_png_count} PNGs | Time: {elapsed:.1f}s")

        evaluation_matrix.append({
            "Video": video_file.name,
            "Folder": video_file.parent.name,
            "Resolution": f"{width}x{height}",
            "Frames": total_frames,
            "DurationSec": round(duration, 1),
            "GroundTruth": ground_truth,
            "DetectedClass": detected_class,
            "ConfirmedAlerts": total_confirmed,
            "KnifeAlerts": confirmed_counts.get("knife", 0),
            "HandgunAlerts": confirmed_counts.get("handgun", 0),
            "SuppressedTraps": total_suppressed,
            "Status": status,
            "TimeSec": round(elapsed, 1),
            "AnnotatedVideo": annotated_video_path.name,
            "FramesCaptured": frame_png_count,
        })

    total_time = time.time() - start_total_time

    # Save Evaluation Summary CSV
    summary_csv_path = EVAL_BASE_DIR / "FULL_RUN_EVALUATION_SUMMARY.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Video", "Folder", "Resolution", "Frames", "DurationSec",
            "GroundTruth", "DetectedClass", "ConfirmedAlerts", "KnifeAlerts",
            "HandgunAlerts", "SuppressedTraps", "Status", "TimeSec", "AnnotatedVideo", "FramesCaptured"
        ])
        writer.writeheader()
        writer.writerows(evaluation_matrix)

    print("\n" + "=" * 100)
    print("FULL RUN BATCH EVALUATION RESULTS SUMMARY MATRIX")
    print("=" * 100)
    print(f"{'Folder / Video':<38} | {'Resolution':<9} | {'GT':<7} | {'Pred':<7} | {'Alerts':<6} | {'Status':<24}")
    print("-" * 100)
    passed_count = 0
    for row in evaluation_matrix:
        folder_vid = f"{row['Folder'][:14]}/{row['Video'][:20]}"
        print(f"{folder_vid:<38} | {row['Resolution']:<9} | {row['GroundTruth']:<7} | {row['DetectedClass']:<7} | {row['ConfirmedAlerts']:<6} | {row['Status']:<24}")
        if "PASS" in row["Status"]:
            passed_count += 1
    print("-" * 100)
    pass_rate = (passed_count / len(evaluation_matrix)) * 100 if evaluation_matrix else 0.0
    print(f"Total Evaluated: {len(evaluation_matrix)} | Passed: {passed_count}/{len(evaluation_matrix)} ({pass_rate:.1f}%) | Total Time: {total_time:.1f}s")
    print(f"Summary CSV: {summary_csv_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
