from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.cctv_intelligence import CCTVIntelligenceFilter
from app.services.detection_service import DetectionService, draw_detections
from app.services.multi_camera_service import CameraInputConfig, MultiCameraService
from app.services.report_service import ReportService
from app.services.video_enhancement_service import VideoEnhancementService
from app.services.video_service import VideoService
from app.utils.timestamps import format_timestamp



def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_report_csv(source_csv: Path, destination_csv: Path) -> Path:
    if not source_csv.exists():
        raise FileNotFoundError(f"Detection report not found: {source_csv}")

    destination_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_csv, destination_csv)
    return destination_csv


def generate_forensic_summary(csv_path: Path, summary_path: Path) -> dict:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV report not found: {csv_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    total_detections = len(rows)
    confirmed_rows = [
        r for r in rows
        if r.get("validation_status", r.get("status", "")) in ("CONFIRMED_ALERT", "VALIDATED_TEMPORAL")
    ]
    suppressed_rows = [
        r for r in rows
        if r not in confirmed_rows
    ]

    confirmed_counts = {}
    total_confirmed_conf = 0.0
    for row in confirmed_rows:
        label = row.get("object_label", row.get("class_name", "weapon"))
        confirmed_counts[label] = confirmed_counts.get(label, 0) + 1
        conf_val = float(row.get("confidence_score", row.get("confidence", 0.0)))
        total_confirmed_conf += conf_val

    avg_confirmed_conf = (total_confirmed_conf / len(confirmed_rows)) if confirmed_rows else 0.0
    first_event_time = None
    if confirmed_rows:
        first_event_time = min(float(r["timestamp_seconds"]) for r in confirmed_rows)
    elif rows:
        first_event_time = min(float(r["timestamp_seconds"]) for r in rows)

    suppression_counts = {}
    for row in suppressed_rows:
        reason = row.get("rejection_reason", "UNKNOWN")
        cat = reason.split("(")[0].strip() if "(" in reason else reason
        suppression_counts[cat] = suppression_counts.get(cat, 0) + 1

    summary = {
        "total_evaluated_records": total_detections,
        "confirmed_alerts": len(confirmed_rows),
        "suppressed_traps": len(suppressed_rows),
        "class_counts": confirmed_counts,
        "average_confidence": round(avg_confirmed_conf, 4),
        "first_event_time_seconds": first_event_time,
        "suppression_reasons": suppression_counts,
        "report_path": str(csv_path),
    }

    lines = []
    lines.append("CCTV Weapon Detection Forensic Summary")
    lines.append("=" * 52)
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total evaluated proposals: {total_detections}")
    lines.append(f"Confirmed Security Alerts: {len(confirmed_rows)}")
    lines.append(f"Suppressed Environmental Traps: {len(suppressed_rows)}")
    lines.append(f"Average confidence (confirmed alerts): {avg_confirmed_conf:.4f}")
    lines.append(f"First confirmed threat time: {first_event_time if first_event_time is not None else 'N/A'} s")
    lines.append("")
    lines.append("Confirmed threat classes:")
    if confirmed_counts:
        for label, count in sorted(confirmed_counts.items()):
            lines.append(f"- {label}: {count}")
    else:
        lines.append("- No confirmed threats")

    if suppression_counts:
        lines.append("")
        lines.append("Suppressed environmental background traps (audit):")
        for cat, count in sorted(suppression_counts.items()):
            lines.append(f"- {cat}: {count}")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def resolve_model_path(model_arg: str | None) -> str | Path | None:
    if model_arg is None or str(model_arg).strip() == "":
        return None

    value = str(model_arg).strip().lower()
    if value in {"auto", "default"}:
        return None
    if value in {"seventh", "model7", "seven"}:
        return ROOT_DIR / "best_weapon_detector_seventh_model.pth"
    if value in {"sixth", "model6", "six"}:
        return ROOT_DIR / "best_weapon_detector_sixth_model.pth"
    if value in {"fifth", "model5", "five"}:
        return ROOT_DIR / "best_weapon_detector_fifth_model.pth"
    if value in {"retrained", "best_retrained", "new"}:
        return ROOT_DIR / "best_weapon_detector_retrained.pth"
    if value in {"original", "baseline", "best"}:
        return ROOT_DIR / "best_weapon_detector.pth"

    path = Path(model_arg).expanduser()
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    return path


def detect_video_with_model(
    input_path: str | Path,
    output_path: str | Path,
    model_path: str | Path | None = None,
    confidence_threshold: float = 0.50,
    analysis_fps: int = 5,
    save_detected_frames: bool = True,
    csv_output_path: str | Path | None = None,
    enable_cctv_intelligence: bool = True,
    enable_temporal_consistency: bool = True,
    camera_id: str = "CAM-01",
):
    input_file = Path(input_path)
    output_file = Path(output_path)
    detected_frames_dir = output_file.parent / f"{output_file.stem}_detected_frames"
    if save_detected_frames:
        detected_frames_dir.mkdir(parents=True, exist_ok=True)
        for existing in detected_frames_dir.glob("*.png"):
            existing.unlink()

    if not input_file.exists():
        raise FileNotFoundError(f"Input video was not found: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_file))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open: {input_file}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        capture.release()
        raise RuntimeError("The input video has an invalid frame rate.")

    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("The input video has invalid dimensions.")

    duration = total_frames / fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_file}")

    detector = DetectionService(
        confidence_threshold=confidence_threshold,
        model_path=model_path,
    )

    cctv_filter = None
    if enable_cctv_intelligence:
        print("[CCTV Intelligence] Activated: Centroid Motion Tracker + MobileNetV3 Person Proximity Gate + Geometric Area Filter + Temporal Consistency")
        cctv_filter = CCTVIntelligenceFilter(
            device=detector.device,
            enable_person_gating=True,
            enable_motion_filtering=True,
            enable_geometric_filtering=True,
            enable_temporal_consistency=enable_temporal_consistency,
            min_temporal_hits=2,
            class_thresholds={
                "handgun": max(0.50, confidence_threshold),
                "knife": max(0.65, confidence_threshold + 0.15),
            },
        )

    frame_interval = max(1, round(fps / analysis_fps))
    frame_number = 0
    analyzed_frames = 0

    total_detections = 0
    detection_records = []

    while True:
        success, frame = capture.read()
        if not success:
            break

        timestamp = frame_number / fps

        if frame_number % frame_interval == 0:
            if cctv_filter is not None:
                raw_detections = detector.detect_frame(frame, min_threshold=0.35)
                confirmed_detections, audit_records = cctv_filter.process_frame(
                    frame, raw_detections, frame_idx=analyzed_frames
                )
                active_detections = confirmed_detections
                evaluated_to_log = audit_records
            else:
                raw_detections = detector.detect_frame(frame)
                active_detections = raw_detections
                evaluated_to_log = [
                    {**d, "status": "CONFIRMED_ALERT", "rejection_reason": ""}
                    for d in raw_detections
                ]

            analyzed_frames += 1
            total_detections += len(active_detections)
            frame = draw_detections(frame, active_detections)

            if active_detections:
                if save_detected_frames:
                    saved_frame_path = detected_frames_dir / f"frame_{frame_number:06d}_detected.png"
                    if not cv2.imwrite(str(saved_frame_path), frame):
                        raise RuntimeError(f"Could not save detected frame: {saved_frame_path}")

                print(f"Frame {frame_number:04d} | {timestamp:05.2f}s | {len(active_detections)} confirmed alert(s)")
                for detection in active_detections:
                    print(
                        f"  {detection['class_name']}: {detection['confidence']:.2%} | Box: {detection['box']}"
                    )

            for rec in evaluated_to_log:
                x1, y1, x2, y2 = rec["box"]
                detection_records.append(
                    {
                        "source_video": input_file.name,
                        "frame_number": frame_number,
                        "timestamp_seconds": round(timestamp, 3),
                        "timestamp_formatted": format_timestamp(timestamp),
                        "camera_id": camera_id,
                        "object_label": rec["class_name"],
                        "class_name": rec["class_name"],
                        "confidence_score": round(rec["confidence"], 4),
                        "confidence": round(rec["confidence"], 4),
                        "bounding_box": f"[{x1}, {y1}, {x2}, {y2}]",
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "status": rec.get("status", "CONFIRMED_ALERT"),
                        "validation_status": rec.get("validation_status") or rec.get("status", "CONFIRMED_ALERT"),
                        "rejection_reason": rec.get("rejection_reason", "") or "",
                        "temporal_track_id": rec.get("temporal_track_id"),
                    }
                )

        timestamp_label = f"{camera_id} | Frame {frame_number} | Time {timestamp:.2f}s"
        cv2.putText(
            frame,
            timestamp_label,
            (12, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        frame_number += 1

        if total_frames > 0:
            progress = (frame_number / total_frames) * 100
            progress_interval = max(1, total_frames // 10)
            if frame_number % progress_interval == 0 or frame_number == total_frames:
                print(f"Progress: {min(progress, 100):.1f}%")

    capture.release()
    writer.release()

    csv_path = (
        Path(csv_output_path)
        if csv_output_path is not None
        else Path("outputs/reports") / f"{input_file.stem}_detections.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Reconcile temporal consistency across whole video
    if cctv_filter and enable_temporal_consistency:
        detection_records = cctv_filter.reconcile_video_records(detection_records, min_hits=2)

    # Export via ReportService (CSV, JSON, HTML, PDF)
    report_service = ReportService(output_dir=csv_path.parent)
    report_service.export_csv(
        records=detection_records,
        output_path=csv_path,
        default_source_video=input_file.name,
        default_camera_id=camera_id,
    )

    case_metadata = {
        "case_id": f"CCTV-{input_file.stem}",
        "analyst_name": "FORENSIKADA Automated Forensic System",
        "model_version": detector.model_path.name if detector.model_path else "Model 7 Faster R-CNN",
        "cameras": [
            {
                "camera_id": camera_id,
                "filename": input_file.name,
                "path": str(input_file.resolve()),
                "resolution": f"{width}x{height}",
                "fps": round(fps, 2),
                "duration_seconds": round(duration, 2),
                "total_frames": total_frames,
            }
        ],
    }

    report_service.export_json(
        records=detection_records,
        case_metadata=case_metadata,
        output_path=csv_path.with_suffix(".json"),
        default_source_video=input_file.name,
        default_camera_id=camera_id,
    )
    report_service.export_html_report(
        records=detection_records,
        case_metadata=case_metadata,
        output_path=csv_path.parent / f"{input_file.stem}_forensic_report.html",
        default_source_video=input_file.name,
        default_camera_id=camera_id,
    )
    report_service.export_pdf_report(
        records=detection_records,
        case_metadata=case_metadata,
        output_path=csv_path.parent / f"{input_file.stem}_forensic_report.pdf",
        default_source_video=input_file.name,
        default_camera_id=camera_id,
    )

    if not output_file.exists():
        raise RuntimeError("Processing finished, but the output video was not created.")

    print()
    print("Video processing completed.")
    print(f"Frames written: {frame_number}")
    print(f"Frames analyzed: {analyzed_frames}")
    print(f"Confirmed weapon alerts: {sum(1 for r in detection_records if r.get('status') == 'CONFIRMED_ALERT')}")
    print(f"Total evaluated records: {len(detection_records)}")
    print(f"Output saved to: {output_file}")
    print(f"CSV report saved to: {csv_path}")
    print(f"Forensic PDF report saved to: {csv_path.parent / f'{input_file.stem}_forensic_report.pdf'}")

    if save_detected_frames:
        print(f"Detected frame images saved to: {detected_frames_dir}")

    return {
        "input_path": str(input_file),
        "output_path": str(output_file),
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": frame_number,
        "analyzed_frames": analyzed_frames,
        "total_detections": total_detections,
        "duration_seconds": duration,
        "csv_report": str(csv_path),
        "detected_frames_dir": str(detected_frames_dir),
    }


def run_full_pipeline(
    video_path: str | Path,
    output_dir: str | Path,
    model_path: str | Path | None = None,
    confidence_threshold: float = 0.50,
    analysis_fps: int = 5,
    enable_enhancement: bool = True,
    enable_cctv_intelligence: bool = True,
) -> dict:
    # *Video Upload
    input_video = Path(video_path).expanduser().resolve()
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    output_root = Path(output_dir).expanduser().resolve()
    ensure_directory(output_root)

    # *Video Validation
    video_service = VideoService()
    video_service.validate_video(str(input_video))

    # *Metadata Extraction
    metadata = video_service.get_metadata(str(input_video))

    # *Frame Extraction
    enhanced_video = None
    if enable_enhancement:
        enhancer = VideoEnhancementService()
        candidate_enhanced_video = output_root / "enhanced_video.mp4"
        try:
            print("\n[1/4] Enhancing video...")
            # BasicVSR++ enhancement stage
            enhancer.enhance_video(
                str(input_video),
                str(candidate_enhanced_video),
            )
            enhanced_video = candidate_enhanced_video
            print(f"Enhanced video saved to: {enhanced_video}")
        except Exception as exc:
            print(f"Enhancement skipped due to error: {exc}")
            if candidate_enhanced_video.exists():
                candidate_enhanced_video.unlink()

    def run_detection(label, source_video):
        detection_result = detect_video_with_model(
            input_path=source_video,
            output_path=output_root / f"annotated_output_{label}.mp4",
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            analysis_fps=analysis_fps,
            save_detected_frames=True,
            csv_output_path=output_root / f"forensic_detections_{label}.csv",
            enable_cctv_intelligence=enable_cctv_intelligence,
        )
        summary_path = output_root / f"forensic_summary_{label}.txt"
        summary = generate_forensic_summary(
            Path(detection_result["csv_report"]),
            summary_path,
        )
        return detection_result, summary_path, summary

    # *Faster R-CNN detection stage
    print("\n[2/4] Running detection without enhancement...")
    original_result, original_summary_path, original_summary = run_detection(
        "no_enhancement",
        input_video,
    )

    enhanced_result = None
    enhanced_summary_path = None
    enhanced_summary = None
    if enhanced_video is not None:
        print("\n[3/4] Running detection on enhanced video...")
        enhanced_result, enhanced_summary_path, enhanced_summary = run_detection(
            "enhanced",
            enhanced_video,
        )
    else:
        print("\n[3/4] Enhanced detection skipped because enhancement failed.")

    print("\n[4/4] Video detection completed.")
    print(f"No-enhancement annotated video: {original_result['output_path']}")
    print(f"No-enhancement CSV: {original_result['csv_report']}")
    print(f"No-enhancement summary: {original_summary_path}")
    if enhanced_result is not None:
        print(f"Enhanced annotated video: {enhanced_result['output_path']}")
        print(f"Enhanced CSV: {enhanced_result['csv_report']}")
        print(f"Enhanced summary: {enhanced_summary_path}")

    return {
        "input_video": str(input_video),
        "working_video": str(enhanced_video or input_video),
        "enhanced_video": str(enhanced_video) if enhanced_video else None,
        "annotated_video": original_result["output_path"],
        "annotated_video_no_enhancement": original_result["output_path"],
        "annotated_video_enhanced": enhanced_result["output_path"] if enhanced_result else None,
        "csv_report": original_result["csv_report"],
        "csv_report_no_enhancement": original_result["csv_report"],
        "csv_report_enhanced": enhanced_result["csv_report"] if enhanced_result else None,
        "summary_report": str(original_summary_path),
        "summary_report_no_enhancement": str(original_summary_path),
        "summary_report_enhanced": str(enhanced_summary_path) if enhanced_summary_path else None,
        "selected_model": str(resolve_model_path(model_path) if model_path is not None else "auto"),
        "detected_frames_dir": original_result["detected_frames_dir"],
        "detected_frames_no_enhancement": original_result["detected_frames_dir"],
        "detected_frames_enhanced": enhanced_result["detected_frames_dir"] if enhanced_result else None,
        "summary": original_summary,
        "summary_no_enhancement": original_summary,
        "summary_enhanced": enhanced_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full CCTV weapon detection pipeline: upload -> enhancement -> detection -> forensic report."
    )
    parser.add_argument("--video", "--camera1", type=str, default=None, help="Path to the primary CCTV input video (Camera 1).")
    parser.add_argument("--cam1-id", type=str, default="CAM-01", help="Identifier for Camera 1 (default: CAM-01).")
    parser.add_argument("--camera2", type=str, default=None, help="Path to an optional secondary CCTV input video (Camera 2).")
    parser.add_argument("--cam2-id", type=str, default="CAM-02", help="Identifier for Camera 2 (default: CAM-02).")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/full_pipeline_run",
        help="Output folder for the enhanced video, annotated video, CSV, and summary report.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="auto",
        help="Detection checkpoint to use: auto, seventh, sixth, fifth, or path to .pth.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.50,
        help="Detection confidence threshold for handgun and knife detection.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames-per-second sampling rate used during detection.",
    )
    parser.add_argument(
        "--no-enhancement",
        action="store_true",
        help="Skip the WSL BasicVSR++ enhancement step and run detection directly on the source video.",
    )
    parser.add_argument(
        "--no-cctv-intelligence",
        action="store_true",
        help="Disable CCTV intelligence filters (motion tracking, person proximity gating, and geometric filters).",
    )
    parser.add_argument(
        "--no-temporal-consistency",
        action="store_true",
        help="Disable temporal consistency tracklet persistence and smoothing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    primary_video = args.video
    if not primary_video:
        raise ValueError("Please provide at least one input video via --video or --camera1.")

    if args.camera2:
        print("\n" + "=" * 76)
        print("          MULTI-CAMERA SURVEILLANCE PIPELINE ACTIVATED (2 CAMERAS)          ")
        print("=" * 76)
        cam_configs = [
            CameraInputConfig(camera_id=args.cam1_id, video_path=primary_video),
            CameraInputConfig(camera_id=args.cam2_id, video_path=args.camera2),
        ]
        mc_service = MultiCameraService(
            output_dir=args.output,
            confidence_threshold=args.confidence,
            analysis_fps=args.fps,
            enable_cctv_intelligence=not args.no_cctv_intelligence,
            enable_temporal_consistency=not args.no_temporal_consistency,
            model_path=resolve_model_path(args.model),
        )
        result = mc_service.process_cameras(cam_configs)
        print("\n[Finished] Multi-camera surveillance pipeline completed successfully.")
        print(f"Unified Forensic CSV: {result['unified_csv']}")
        print(f"Unified Forensic PDF: {result['unified_pdf']}")
    else:
        result = run_full_pipeline(
            video_path=primary_video,
            output_dir=args.output,
            model_path=resolve_model_path(args.model),
            confidence_threshold=args.confidence,
            analysis_fps=args.fps,
            enable_enhancement=not args.no_enhancement,
            enable_cctv_intelligence=not args.no_cctv_intelligence,
        )
        print("\n[4/4] Full pipeline finished successfully.")
        print(f"Model used: {result['selected_model']}")
        print(f"Result summary: {result['summary']}")

