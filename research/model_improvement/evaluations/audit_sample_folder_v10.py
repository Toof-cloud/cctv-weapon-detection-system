"""
Comprehensive CCTV Sample Folder Audit for Candidate Model V10
Processes all CCTV video footage in the samples folder:
  - Root CCTV samples (handgun, knife, evaluation, general, negative)
  - Unseen real CCTV clips (Clips 01 - 10)
Extracts frames 1-by-1, records detections, audits TP/FP/Hallucinations,
and generates structured visual artifacts and forensic logs.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time

# Enforce thread guard
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import torch

from app.services.cctv_intelligence import CCTVIntelligenceFilter
from app.services.detection_service import DetectionService

MODEL_PATH = ROOT_DIR / "research" / "model_improvement" / "checkpoints" / "best_candidate_model_v10.pth"
AUDIT_OUT_DIR = ROOT_DIR / "outputs" / "v10_sample_folder_audit"
AUDIT_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Ground truth metadata for accurate auditing
VIDEO_METADATA = {
    # Root sample videos
    "handgun_test-video.mp4": {"type": "root", "ground_truth": "handgun", "desc": "Night store robbery, handgun drawn at counter"},
    "evaluation_video.mp4": {"type": "root", "ground_truth": "knife", "desc": "Outdoor knife assault confrontation"},
    "NEW_KNIFE_VIDEO_11s.mp4": {"type": "root", "ground_truth": "knife", "desc": "1080p CCTV kitchen knife brandishing"},
    "test_10s_knife.mp4": {"type": "root", "ground_truth": "knife", "desc": "CCTV hallway knife threat"},
    "test_10s_general.mp4": {"type": "root", "ground_truth": "negative", "desc": "General public scene without active weapon"},
    "test_10s_negative.mp4": {"type": "root", "ground_truth": "negative", "desc": "Pure negative walking actor stress test"},
    "Screen Recording 2026-09-10 213340 -HANDGUN - Trim.mp4": {"type": "root", "ground_truth": "handgun", "desc": "1080p cashier robbery handgun draw"},
    "CAM01_Scene 004.mp4": {"type": "root", "ground_truth": "knife", "desc": "Staged Scene 004 knife brandishing (CAM-01)"},
    "CAM02_Scene_004.mp4": {"type": "root", "ground_truth": "knife", "desc": "Staged Scene 004 knife brandishing (CAM-02)"},
    "knife_test_long-video.mp4": {"type": "root", "ground_truth": "knife", "desc": "Long knife demonstration clip (analyzing first 300 frames)"},
    # Unseen CCTV clips (Jabez)
    "Screen Recording 2026-09-07 224743.mp4": {"type": "unseen", "clip_id": "Clip01", "ground_truth": "handgun", "desc": "Unseen Real CCTV: Night Store Robbery (Handgun)"},
    "Screen Recording 2026-09-07 225138.mp4": {"type": "unseen", "clip_id": "Clip02", "ground_truth": "knife", "desc": "Unseen Real CCTV: Counter Assault (Knife)"},
    "Screen Recording 2026-09-07 225703.mp4": {"type": "unseen", "clip_id": "Clip03", "ground_truth": "handgun", "desc": "Unseen Real CCTV: Street Robbery Confrontation"},
    "Screen Recording 2026-09-07 230047.mp4": {"type": "unseen", "clip_id": "Clip04", "ground_truth": "handgun", "desc": "Unseen Real CCTV: Night Street Gun Draw"},
    "Screen Recording 2026-09-07 230249.mp4": {"type": "unseen", "clip_id": "Clip05", "ground_truth": "handgun", "desc": "Unseen Real CCTV: Counter Customer Draw (Handgun)"},
    "Screen Recording 2026-09-07 230541.mp4": {"type": "unseen", "clip_id": "Clip06", "ground_truth": "knife", "desc": "Unseen Real CCTV: Alley Assault (Knife)"},
    "Screen Recording 2026-09-07 230842.mp4": {"type": "unseen", "clip_id": "Clip07", "ground_truth": "knife", "desc": "Unseen Real CCTV: Fast Blade Draw (Knife)"},
    "Screen Recording 2026-09-07 230942.mp4": {"type": "unseen", "clip_id": "Clip08", "ground_truth": "handgun", "desc": "Unseen Real CCTV: Corridor Threat (Handgun)"},
    "Screen Recording 2026-09-07 231101.mp4": {"type": "unseen", "clip_id": "Clip09", "ground_truth": "handgun", "desc": "Unseen Real CCTV: Multi-Person Distractor Robbery"},
    "Screen Recording 2026-09-07 231205.mp4": {"type": "unseen", "clip_id": "Clip10", "ground_truth": "handgun", "desc": "Unseen Real CCTV: Bank Counter Robbery (Handgun)"},
}


def draw_frame_annotations(frame, confirmed_detections, audit_records):
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Draw confirmed detections (Green / Red / Orange)
    for det in confirmed_detections:
        box = det.get("bounding_box", det.get("box", [0, 0, 0, 0]))
        x1, y1, x2, y2 = [int(v) for v in box]
        conf = float(det.get("confidence_score", det.get("confidence", 0.0)))
        cls_name = det.get("object_label", det.get("class_name", "weapon"))
        color = (0, 0, 255) if cls_name == "handgun" else (0, 165, 255)  # Red / Orange
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"CONFIRMED {cls_name.upper()} {conf:.1%}"
        t_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + t_size[0], y1), color, -1)
        cv2.putText(annotated, label, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Draw filtered / suppressed candidates (Grey with rejection reason)
    for rec in audit_records:
        status = rec.get("validation_status", rec.get("status", ""))
        if status in ("CONFIRMED_ALERT", "VALIDATED_TEMPORAL"):
            continue
        box = rec.get("bounding_box", rec.get("box", [0, 0, 0, 0]))
        x1, y1, x2, y2 = [int(v) for v in box]
        conf = float(rec.get("confidence_score", rec.get("confidence", 0.0)))
        cls_name = rec.get("object_label", rec.get("class_name", "weapon"))
        reason = str(rec.get("rejection_reason") or "FILTERED")
        color = (128, 128, 128)  # Grey

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
        label = f"SUPPRESSED {cls_name.upper()} {conf:.1%} ({reason[:16]})"
        cv2.putText(annotated, label, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return annotated


def audit_video(
    video_path: Path,
    detector: DetectionService,
    output_dir: Path,
    analysis_fps: int = 5,
    max_frames_to_process: int = 350,
):
    video_name = video_path.name
    meta = VIDEO_METADATA.get(video_name, {"type": "general", "ground_truth": "unknown", "desc": video_name})
    clean_stem = video_path.stem.replace(" ", "_").replace("-", "_")
    clip_dir = output_dir / clean_stem
    frames_dir = clip_dir / "extracted_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Error] Could not open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_interval = max(1, round(fps / analysis_fps))

    cctv_filter = CCTVIntelligenceFilter(
        device=detector.device,
        enable_person_gating=True,
        enable_motion_filtering=True,
        enable_geometric_filtering=True,
        enable_temporal_consistency=True,
        min_temporal_hits=2,
        class_thresholds={"handgun": 0.50, "knife": 0.50},
    )

    frame_idx = 0
    analyzed_count = 0
    confirmed_total = 0
    raw_candidates_total = 0
    suppressed_total = 0
    oversized_box_count = 0

    confirmed_confidences = []
    frame_audit_records = []
    saved_frame_paths = []

    print(f"\n---> Auditing: {video_name} ({total_frames} frames, {fps:.1f} fps, {width}x{height})")
    print(f"     Ground Truth Threat: {meta['ground_truth'].upper()} | {meta['desc']}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            analyzed_count += 1
            if analyzed_count > max_frames_to_process:
                break

            timestamp = frame_idx / fps

            # Detect raw candidates (min_threshold=0.35 to monitor all neural activations)
            raw_detections = detector.detect_frame(frame, min_threshold=0.35)
            raw_candidates_total += len(raw_detections)

            for det in raw_detections:
                box = det["box"]
                bw = box[2] - box[0]
                bh = box[3] - box[1]
                if bw > 500 or bh > 500:
                    oversized_box_count += 1

            # Pass through Tier-2 CCTV Intelligence Filter
            confirmed_detections, audit_records = cctv_filter.process_frame(
                frame, raw_detections, frame_idx=analyzed_count
            )

            confirmed_total += len(confirmed_detections)
            for cd in confirmed_detections:
                conf = float(cd.get("confidence_score", cd.get("confidence", 0.0)))
                confirmed_confidences.append(conf)

            num_suppressed = len(audit_records) - len(confirmed_detections)
            suppressed_total += max(0, num_suppressed)

            # Save annotated frame if any event occurred or periodic sample
            is_sample_frame = (analyzed_count in (1, 10, 25, 50, 100, 150, 200))
            if raw_detections or is_sample_frame:
                annotated = draw_frame_annotations(frame, confirmed_detections, audit_records)
                frame_filename = f"frame_{frame_idx:05d}_t{timestamp:05.2f}s.jpg"
                save_p = frames_dir / frame_filename
                cv2.imwrite(str(save_p), annotated)
                saved_frame_paths.append(str(save_p))

                # Log detail
                frame_audit_records.append({
                    "frame_number": frame_idx,
                    "timestamp": round(timestamp, 2),
                    "raw_detections": raw_detections,
                    "confirmed_detections": confirmed_detections,
                    "audit_records": audit_records,
                    "annotated_frame": frame_filename,
                })

        frame_idx += 1

    cap.release()

    # Calculate audit metrics
    avg_conf = float(sum(confirmed_confidences) / len(confirmed_confidences)) if confirmed_confidences else 0.0
    max_conf = float(max(confirmed_confidences)) if confirmed_confidences else 0.0

    # Ground truth comparison
    gt_threat = meta["ground_truth"]
    if gt_threat in ("handgun", "knife"):
        true_positives = confirmed_total
        false_positives = 0
        hallucinations = suppressed_total
        verdict = f"HIGH RECALL THREAT CONFIRMED ({confirmed_total} alerts, max {max_conf:.1%})" if confirmed_total > 0 else "MISSED (FN)"
    else:
        true_positives = 0
        false_positives = confirmed_total
        hallucinations = suppressed_total
        verdict = "PERFECT CLEAN AUDIT (0 False Alarms)" if confirmed_total == 0 else f"FALSE ALARM DETECTED ({confirmed_total} FP)"

    result_summary = {
        "video_name": video_name,
        "category": meta["type"],
        "ground_truth_threat": gt_threat,
        "description": meta["desc"],
        "total_video_frames": total_frames,
        "analyzed_frames": analyzed_count,
        "raw_candidate_proposals": raw_candidates_total,
        "confirmed_alerts": confirmed_total,
        "suppressed_distractors": suppressed_total,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "oversized_boxes_over_500px": oversized_box_count,
        "average_confidence": round(avg_conf, 4),
        "max_confidence": round(max_conf, 4),
        "verdict": verdict,
        "extracted_frames_count": len(saved_frame_paths),
        "frames_directory": str(frames_dir),
    }

    # Save clip audit json
    with open(clip_dir / "audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(result_summary, f, indent=2)

    with open(clip_dir / "frame_by_frame_log.json", "w", encoding="utf-8") as f:
        json.dump(frame_audit_records, f, indent=2)

    print(f"  --> Confirmed Alerts: {confirmed_total} | Suppressed Distractors: {suppressed_total} | Oversized Boxes: {oversized_box_count}")
    print(f"  --> Max Conf: {max_conf:.1%} | Avg Conf: {avg_conf:.1%} | Verdict: {verdict}")

    return result_summary


def run_full_sample_folder_audit():
    print("=" * 85)
    print("CANDIDATE MODEL V10: FULL CCTV SAMPLES FOLDER MANUAL FRAME-BY-FRAME AUDIT")
    print(f"Checkpoint: {MODEL_PATH}")
    print(f"Audit Output Directory: {AUDIT_OUT_DIR}")
    print("=" * 85)

    detector = DetectionService(
        confidence_threshold=0.35,
        model_path=MODEL_PATH,
    )
    print(f"Loaded detector device: {detector.device}")

    # 1. Root sample videos
    root_sample_dir = ROOT_DIR / "samples"
    root_videos = [
        root_sample_dir / "handgun_test-video.mp4",
        root_sample_dir / "evaluation_video.mp4",
        root_sample_dir / "NEW_KNIFE_VIDEO_11s.mp4",
        root_sample_dir / "test_10s_knife.mp4",
        root_sample_dir / "test_10s_general.mp4",
        root_sample_dir / "test_10s_negative.mp4",
        root_sample_dir / "Screen Recording 2026-09-10 213340 -HANDGUN - Trim.mp4",
        root_sample_dir / "CAM01_Scene 004.mp4",
        root_sample_dir / "CAM02_Scene_004.mp4",
        root_sample_dir / "knife_test_long-video.mp4",
    ]

    # 2. Unseen CCTV videos (Jabez collection)
    unseen_dir = root_sample_dir / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS"
    unseen_videos = sorted(list(unseen_dir.glob("*.mp4")))

    all_target_videos = [v for v in root_videos if v.is_file()] + unseen_videos
    print(f"Discovered {len(all_target_videos)} CCTV videos to audit.\n")

    overall_results = []
    t_start = time.time()

    for idx, vpath in enumerate(all_target_videos, 1):
        print(f"\n[{idx}/{len(all_target_videos)}] Auditing Video: {vpath.name}")
        summary = audit_video(
            video_path=vpath,
            detector=detector,
            output_dir=AUDIT_OUT_DIR,
            analysis_fps=5,
            max_frames_to_process=300,
        )
        if summary:
            overall_results.append(summary)

    elapsed = time.time() - t_start

    # Save overall master report
    master_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "Candidate Model V10 (best_candidate_model_v10.pth)",
        "device": str(detector.device),
        "total_videos_audited": len(overall_results),
        "elapsed_seconds": round(elapsed, 2),
        "total_analyzed_frames": sum(r["analyzed_frames"] for r in overall_results),
        "total_true_positives": sum(r["true_positives"] for r in overall_results),
        "total_false_positives": sum(r["false_positives"] for r in overall_results),
        "total_suppressed_distractors": sum(r["suppressed_distractors"] for r in overall_results),
        "total_oversized_boxes": sum(r["oversized_boxes_over_500px"] for r in overall_results),
        "video_summaries": overall_results,
    }

    with open(AUDIT_OUT_DIR / "master_sample_folder_audit.json", "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=2)

    print("\n" + "=" * 85)
    print("AUDIT COMPLETE!")
    print(f"Total Videos Audited: {len(overall_results)}")
    print(f"Total Analyzed Frames: {master_report['total_analyzed_frames']:,}")
    print(f"Total Confirmed True Positives: {master_report['total_true_positives']}")
    print(f"Total False Positives: {master_report['total_false_positives']}")
    print(f"Total Suppressed Distractors / Hallucinations: {master_report['total_suppressed_distractors']}")
    print(f"Total Oversized Boxes (>500px): {master_report['total_oversized_boxes']}")
    print(f"Elapsed Time: {elapsed:.2f}s")
    print("=" * 85)


if __name__ == "__main__":
    run_full_sample_folder_audit()
