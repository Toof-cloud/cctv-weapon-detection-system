import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import torch

from app.services.cctv_intelligence import CCTVIntelligenceFilter
from app.services.detection_service import DetectionService, draw_detections
from app.services.report_service import ReportService
from app.utils.file_validation import validate_video_file
from app.utils.timestamps import format_timestamp


@dataclass
class CameraInputConfig:
    camera_id: str
    video_path: str | Path
    location_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiCameraService:
    """
    Coordinates multi-camera video analysis (accepts 2 or more camera inputs).
    
    Responsibilities:
    1. Validates and manages multiple video inputs (e.g. CAM-01 and CAM-02).
    2. Runs Faster R-CNN detection with CCTV Intelligence and Temporal Consistency.
    3. Produces annotated video recordings for each camera.
    4. Merges multi-camera detections into a unified, chronologically traceable timeline.
    5. Exports unified forensic CSV, JSON, HTML, and PDF incident reports via ReportService.
    """

    def __init__(
        self,
        output_dir: Path | str = ROOT_DIR / "outputs" / "multi_camera_run",
        confidence_threshold: float = 0.50,
        analysis_fps: int = 5,
        enable_cctv_intelligence: bool = True,
        enable_temporal_consistency: bool = True,
        model_path: Optional[Path | str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.confidence_threshold = confidence_threshold
        self.analysis_fps = analysis_fps
        self.enable_cctv_intelligence = enable_cctv_intelligence
        self.enable_temporal_consistency = enable_temporal_consistency
        self.model_path = model_path

        self.report_service = ReportService(output_dir=self.output_dir / "reports")

    def process_cameras(
        self,
        camera_configs: List[CameraInputConfig],
        case_id: Optional[str] = None,
        analyst_name: str = "Forensic Automated Pipeline",
        progress_callback: Optional[Callable[[str, int, int, float], None]] = None,
    ) -> Dict[str, Any]:
        """
        Analyzes all camera streams and produces a unified forensic report.
        
        progress_callback(camera_id, current_frame, total_frames, percent_done)
        """
        if not camera_configs:
            raise ValueError("No camera configurations provided to MultiCameraService.")

        start_time = time.time()
        case_ref = case_id or f"CASE-CCTV-{int(time.time())}"

        # 1. Validate all camera feeds
        validated_cameras = []
        for cam in camera_configs:
            ok, err, meta = validate_video_file(cam.video_path)
            if not ok:
                raise ValueError(f"Camera [{cam.camera_id}] validation failed: {err}")
            cam.metadata = meta
            validated_cameras.append({
                "camera_id": cam.camera_id,
                "filename": meta["filename"],
                "path": meta["path"],
                "resolution": meta["resolution"],
                "fps": meta["fps"],
                "duration_seconds": meta["duration_seconds"],
                "total_frames": meta["total_frames"],
                "location_name": cam.location_name or f"Channel {cam.camera_id}",
            })

        print(f"\n[MultiCameraService] Initializing detection engine for {len(camera_configs)} cameras...")
        detector = DetectionService(
            confidence_threshold=self.confidence_threshold,
            model_path=self.model_path,
        )

        all_camera_records: List[Dict[str, Any]] = []
        camera_results: Dict[str, Any] = {}
        all_keyframe_crops: List[Dict[str, Any]] = []

        crops_dir = self.output_dir / "evidence_crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        # 2. Process each camera feed
        for cam_idx, cam in enumerate(camera_configs, start=1):
            cam_id = cam.camera_id
            vid_path = Path(cam.video_path)
            print(f"\n--- [{cam_idx}/{len(camera_configs)}] Processing Camera: {cam_id} ({vid_path.name}) ---")

            cap = cv2.VideoCapture(str(vid_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            cam_out_video = self.output_dir / f"annotated_{cam_id}_{vid_path.stem}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(cam_out_video), fourcc, fps, (width, height))

            # Per-camera intelligence filter maintains isolated camera spatial/motion states
            cctv_filter = None
            if self.enable_cctv_intelligence:
                cctv_filter = CCTVIntelligenceFilter(
                    device=detector.device,
                    enable_person_gating=True,
                    enable_motion_filtering=True,
                    enable_geometric_filtering=True,
                    enable_temporal_consistency=self.enable_temporal_consistency,
                    min_temporal_hits=2,
                    class_thresholds={
                        "handgun": max(0.50, self.confidence_threshold),
                        "knife": max(0.50, self.confidence_threshold),
                    },
                )

            frame_interval = max(1, round(fps / self.analysis_fps))
            frame_num = 0
            cam_records = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                ts_seconds = frame_num / fps if fps > 0 else 0.0
                ts_formatted = format_timestamp(ts_seconds)

                if frame_num % frame_interval == 0:
                    raw_dets = detector.detect_frame(
                        frame,
                        min_threshold=min(self.confidence_threshold, 0.40),
                    )

                    if cctv_filter:
                        confirmed_list, evaluated_list = cctv_filter.process_frame(
                            frame_bgr=frame,
                            raw_detections=raw_dets,
                            frame_idx=frame_num,
                        )
                        records_to_log = evaluated_list
                        visual_confirmed = confirmed_list
                    else:
                        records_to_log = [
                            {
                                **d,
                                "status": "CONFIRMED_ALERT",
                                "validation_status": "CONFIRMED_ALERT",
                                "rejection_reason": "",
                            }
                            for d in raw_dets
                            if d["confidence"] >= self.confidence_threshold
                        ]
                        visual_confirmed = records_to_log

                    # Draw confirmed threat detections on video frame
                    if visual_confirmed:
                        frame = draw_detections(frame, visual_confirmed)


                    for rec in records_to_log:
                        box = rec["box"]
                        norm_rec = {
                            "source_video": vid_path.name,
                            "frame_number": frame_num,
                            "timestamp_seconds": round(ts_seconds, 3),
                            "timestamp_formatted": ts_formatted,
                            "camera_id": cam_id,
                            "object_label": rec["class_name"],
                            "bounding_box": f"[{box[0]}, {box[1]}, {box[2]}, {box[3]}]",
                            "x1": box[0],
                            "y1": box[1],
                            "x2": box[2],
                            "y2": box[3],
                            "confidence_score": round(float(rec["confidence"]), 4),
                            "validation_status": rec.get("validation_status", "CONFIRMED_ALERT"),
                            "rejection_reason": rec.get("rejection_reason", "") or "",
                            "temporal_track_id": rec.get("temporal_track_id"),
                        }
                        cam_records.append(norm_rec)

                        # Save keyframe crop for confirmed alerts
                        if norm_rec["validation_status"] in ("CONFIRMED_ALERT", "VALIDATED_TEMPORAL") and len(all_keyframe_crops) < 12:
                            crop_fname = f"crop_{cam_id}_f{frame_num:04d}_{rec['class_name']}.png"
                            crop_path = crops_dir / crop_fname
                            if not crop_path.exists():
                                cv2.imwrite(str(crop_path), frame)
                                all_keyframe_crops.append({
                                    "camera_id": cam_id,
                                    "frame_number": frame_num,
                                    "timestamp": ts_formatted,
                                    "label": rec["class_name"],
                                    "confidence": float(rec["confidence"]),
                                    "image_path": str(crop_path.resolve()),
                                })

                # Burn watermark
                watermark = f"{cam_id} | Frame {frame_num:04d} | {ts_formatted}"
                cv2.putText(
                    frame,
                    watermark,
                    (15, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                writer.write(frame)
                frame_num += 1

                if progress_callback and frame_num % 10 == 0:
                    pct = (frame_num / total_frames) * 100.0
                    progress_callback(cam_id, frame_num, total_frames, pct)

            cap.release()
            writer.release()

            # Reconcile temporal consistency across the whole camera feed
            if cctv_filter and self.enable_temporal_consistency:
                cam_records = cctv_filter.reconcile_video_records(cam_records, min_hits=2)

            camera_results[cam_id] = {
                "annotated_video": str(cam_out_video.resolve()),
                "total_records": len(cam_records),
                "confirmed_alerts": sum(1 for r in cam_records if r["validation_status"] in ("CONFIRMED_ALERT", "VALIDATED_TEMPORAL")),
            }
            all_camera_records.extend(cam_records)

        # 3. Compile Unified Chronological Multi-Camera Timeline
        all_camera_records.sort(key=lambda r: (r["timestamp_seconds"], r["camera_id"], r["frame_number"]))

        # 4. Generate Reports via ReportService
        case_metadata = {
            "case_id": case_ref,
            "analyst_name": analyst_name,
            "model_version": detector.model_path.name if detector.model_path else "Model 7 Faster R-CNN",
            "cameras": validated_cameras,
        }

        reports_dir = self.output_dir / "reports"
        csv_path = reports_dir / f"{case_ref}_forensic_detections.csv"
        json_path = reports_dir / f"{case_ref}_forensic_audit.json"
        html_path = reports_dir / f"{case_ref}_forensic_incident_report.html"
        pdf_path = reports_dir / f"{case_ref}_forensic_incident_report.pdf"

        self.report_service.export_csv(all_camera_records, csv_path)
        self.report_service.export_json(all_camera_records, case_metadata, json_path)
        self.report_service.export_html_report(
            records=all_camera_records,
            case_metadata=case_metadata,
            output_path=html_path,
            keyframe_crops=all_keyframe_crops,
        )
        self.report_service.export_pdf_report(
            records=all_camera_records,
            case_metadata=case_metadata,
            output_path=pdf_path,
            keyframe_crops=all_keyframe_crops,
        )

        total_elapsed = time.time() - start_time
        confirmed_total = sum(1 for r in all_camera_records if r["validation_status"] in ("CONFIRMED_ALERT", "VALIDATED_TEMPORAL"))

        print("\n" + "=" * 76)
        print(f"MULTI-CAMERA ANALYSIS COMPLETED in {total_elapsed:.1f}s!")
        print(f"Total Cameras Processed:      {len(camera_configs)}")
        print(f"Total Unified Records:        {len(all_camera_records)}")
        print(f"Total Confirmed Threats:      {confirmed_total}")
        print(f"Unified CSV Log:              {csv_path}")
        print(f"Unified Incident Report (PDF):{pdf_path}")
        print("=" * 76)

        return {
            "case_id": case_ref,
            "elapsed_seconds": round(total_elapsed, 2),
            "camera_results": camera_results,
            "total_records": len(all_camera_records),
            "confirmed_alerts": confirmed_total,
            "unified_csv": str(csv_path.resolve()),
            "unified_json": str(json_path.resolve()),
            "unified_html": str(html_path.resolve()),
            "unified_pdf": str(pdf_path.resolve()),
            "records": all_camera_records,
        }
