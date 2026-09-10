"""Run the existing video detection service and return its completed artifacts."""
from __future__ import annotations

from collections import Counter
from contextlib import redirect_stdout
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

MOCKUP_DIR = Path(__file__).resolve().parent
ROOT_DIR = MOCKUP_DIR.parent
TEMP_DIR = MOCKUP_DIR / "temp"
OUTPUT_DIR = MOCKUP_DIR / "outputs"
MODEL_FILENAME = "best_weapon_detector_ninth_model.pth"
MODEL_PATH = MOCKUP_DIR / "models" / MODEL_FILENAME
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
sys.dont_write_bytecode = True
if str(ROOT_DIR) in sys.path:
    sys.path.remove(str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR))


class VideoInputError(ValueError):
    """The video cannot be decoded or has unusable metadata."""


class ModelSetupError(RuntimeError):
    """The original trained checkpoint/runtime is unavailable."""


def discover_available_models() -> list[tuple[str, Path]]:
    """Only the installed ninth checkpoint is available to the UI."""
    return [(f"Ninth model ({MODEL_FILENAME})", MODEL_PATH)] if MODEL_PATH.is_file() else []


def default_model_path() -> Path | None:
    """Never fall back to another checkpoint when the ninth model is missing."""
    return MODEL_PATH if MODEL_PATH.is_file() else None


def timecode(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    minutes, remaining = divmod(milliseconds, 60_000)
    secs, millis = divmod(remaining, 1000)
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    first_frame: Any = field(repr=False)
    size_bytes: int = 0
    modified_ns: int = 0


def read_video(path: str | Path) -> VideoInfo:
    """Reuse VideoService validation/metadata; decode one frame for the preview."""
    import cv2
    from app.services.video_service import VideoService

    path = Path(path).expanduser().resolve()
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise VideoInputError("Choose an MP4, AVI, MOV, MKV, WebM or M4V video.")
    try:
        service = VideoService()
        service.validate_video(str(path))
        metadata = service.get_metadata(str(path))
        fps = float(metadata["fps"])
        frame_count = int(metadata["total_frames"])
        if not math.isfinite(fps) or fps <= 0 or fps > 1000 or frame_count <= 0:
            raise VideoInputError("This video has invalid timing information. Try an MP4 with a fixed frame rate.")
        capture = cv2.VideoCapture(str(path))
        try:
            ok, first_frame = capture.read()
        finally:
            capture.release()
        if not ok or first_frame is None:
            raise VideoInputError("The first video frame could not be decoded. Try another video.")
        height, width = first_frame.shape[:2]
        stat = path.stat()
        return VideoInfo(path, width, height, fps, frame_count, frame_count / fps,
                         first_frame, stat.st_size, stat.st_mtime_ns)
    except VideoInputError:
        raise
    except (OSError, RuntimeError, ValueError, cv2.error) as exc:
        raise VideoInputError("This video could not be opened. It may be damaged, missing, or use an unsupported codec.") from exc


@dataclass
class VideoAnalysisResult:
    video: VideoInfo
    output_path: Path
    csv_path: Path
    detections: list[dict]
    model_path: str
    device: str
    elapsed_seconds: float
    threshold: float
    analyzed_frames: int
    completed_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    run_id: str = field(default_factory=lambda: uuid4().hex)
    review_path: Path | None = None
    metric_input_path: Path | None = None
    temporal_consistency_enabled: bool | None = None

    @property
    def counts(self) -> dict[str, int]:
        return dict(Counter(row["class_name"] for row in self.detections))

    @property
    def positive_frames(self) -> int:
        return len({row["frame_number"] for row in self.detections})

    @property
    def by_frame(self) -> dict[int, list[dict]]:
        result: dict[int, list[dict]] = {}
        for row in self.detections:
            result.setdefault(row["frame_number"], []).append(row)
        return result


class _PipelineOutput(io.TextIOBase):
    """Expose the existing helper's progress while leaving its code unchanged."""
    def __init__(self, progress, destination):
        self.progress = progress
        self.destination = destination
        self.buffer = ""

    def write(self, value):
        if self.destination is not None:
            try:
                self.destination.write(value)
            except UnicodeEncodeError:
                encoding = getattr(self.destination, "encoding", None) or "utf-8"
                safe_val = value.encode(encoding, errors="replace").decode(encoding)
                self.destination.write(safe_val)
        self.buffer += value
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            match = re.match(r"^Progress:\s*([0-9.]+)%", line.strip())
            if match:
                self.progress("Scanning the video for handguns and knives…", min(98, int(float(match.group(1)))))
            elif line.startswith("Loaded model:"):
                self.progress("Trained model loaded. Scanning video frames…", 0)
        return len(value)

    def flush(self):
        if self.destination is not None:
            self.destination.flush()


class ModelBridge:
    def __init__(self, model_path: Path | str | None = None):
        self.model_path: Path | None = None
        self.set_model_path(model_path)
        self.device = ""
        self.enable_cctv_intelligence: bool = False
        self.enable_temporal_consistency: bool = False

    def set_model_path(self, path: Path | str | None):
        if path is not None and Path(path).resolve() != MODEL_PATH.resolve():
            raise ModelSetupError(f"This application uses only {MODEL_FILENAME} in mockup_ui/models.")
        self.model_path = default_model_path()

    def find_model(self) -> bool:
        self.model_path = default_model_path()
        return self.model_path is not None

    def _prepare_runtime(self, progress):
        if not self.find_model():
            raise ModelSetupError(f"Trained model not found. Place {MODEL_FILENAME} in mockup_ui/models; the imported video will be analyzed automatically when it is found.")
        progress("Loading the original Faster R-CNN detection service…", -1)
        try:
            import torch
            from run_full_pipeline import detect_video_with_model
        except (ImportError, OSError, RuntimeError) as exc:
            raise ModelSetupError("The detector runtime could not load. Use the Python environment that already runs your trained model.") from exc
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        return detect_video_with_model

    def analyze_video(self, video: VideoInfo, threshold: float = 0.50,
                      progress: Callable[[str, int], None] = lambda text, percent: None,
                      enable_cctv_intelligence: bool | None = None,
                      enable_temporal_consistency: bool | None = None,
                      camera_id: str = "CAM-01"):
        """Return a result only after the existing pipeline finishes the video."""
        if enable_cctv_intelligence is None:
            enable_cctv_intelligence = getattr(self, "enable_cctv_intelligence", False)
        if enable_temporal_consistency is None:
            enable_temporal_consistency = getattr(self, "enable_temporal_consistency", False)
        if not 0.01 <= threshold <= 0.99:
            raise ValueError("Confidence threshold must be between 1% and 99%.")
        stat = video.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (video.size_bytes, video.modified_ns):
            raise VideoInputError("The selected file changed. Import the video again before analysis.")
        started = perf_counter()
        detect_video = self._prepare_runtime(progress)
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="video-run-", dir=TEMP_DIR))
        output_path = run_dir / "annotated.mp4"
        csv_path = run_dir / "detections.csv"
        # Reuse the original full-video helper: it loads DetectionService, runs
        # the trained model, draws boxes, and writes the video and CSV itself.
        with redirect_stdout(_PipelineOutput(progress, sys.stdout)):
            details = detect_video(
                input_path=video.path, output_path=output_path, model_path=self.model_path,
                confidence_threshold=threshold, analysis_fps=math.ceil(video.fps),
                save_detected_frames=False, csv_output_path=csv_path,
                enable_cctv_intelligence=enable_cctv_intelligence,
                enable_temporal_consistency=enable_temporal_consistency,
                camera_id=camera_id,
            )
        progress("Checking the annotated video and detection results…", 99)
        if details["total_frames"] != video.frame_count or details["analyzed_frames"] != video.frame_count:
            raise VideoInputError("The video ended before all frames were analyzed. Try another clip.")
        rendered = read_video(output_path)
        if rendered.frame_count != video.frame_count:
            raise VideoInputError("The annotated video could not be written completely. Check available disk space and try again.")
        
        records = []
        # Keep every original status, including suppressed flickers, for benchmark metrics.
        # The display CSV below intentionally contains only active observations.
        metric_input_path = None
        if csv_path.exists():
            metric_input_path = run_dir / "metric_input.csv"
            shutil.copy2(csv_path, metric_input_path)
            with csv_path.open(newline="", encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    cls_name = row.get("class_name") or row.get("object_label", "weapon")
                    conf = float(row.get("confidence") or row.get("confidence_score", 0.0))
                    if all(k in row for k in ("x1", "y1", "x2", "y2")):
                        box = [int(float(row[k])) for k in ("x1", "y1", "x2", "y2")]
                    elif "bounding_box" in row:
                        import ast
                        raw_b = row["bounding_box"].strip()
                        box = [int(float(v)) for v in ast.literal_eval(raw_b)] if raw_b.startswith("[") else [0, 0, 0, 0]
                    else:
                        box = [0, 0, 0, 0]

                    val_status = row.get("validation_status") or row.get("status") or "CONFIRMED_ALERT"
                    rej_reason = row.get("rejection_reason", "")
                    # Suppressed environmental background traps are excluded from confirmed active detections
                    if val_status in ("SUPPRESSED_TEMPORAL_FLICKER", "STATIC_BACKGROUND_TRAP",
                                      "ANTHROPOMETRIC_SCALE_VIOLATION", "NO_PERSON_IN_SCENE",
                                      "NO_PERSON_PROXIMITY", "BELOW_CLASS_THRESHOLD",
                                      "HANDHELD_PHONE_ASPECT_RATIO"):
                        continue

                    records.append({
                        "frame_number": int(row["frame_number"]),
                        "timestamp_seconds": float(row["timestamp_seconds"]),
                        "class_name": cls_name,
                        "confidence": conf,
                        "box": box,
                        "automated_validation_status": val_status,
                        "rejection_reason": rej_reason,
                    })

        # Standardize detections.csv so downstream forensic_report and tests find both legacy and forensic headers
        fieldnames = [
            "frame_number", "timestamp_seconds", "class_name", "confidence",
            "x1", "y1", "x2", "y2", "object_label", "confidence_score",
            "bounding_box", "validation_status", "rejection_reason"
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                b = r["box"]
                writer.writerow({
                    "frame_number": r["frame_number"],
                    "timestamp_seconds": r["timestamp_seconds"],
                    "class_name": r["class_name"],
                    "confidence": r["confidence"],
                    "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3],
                    "object_label": r["class_name"],
                    "confidence_score": r["confidence"],
                    "bounding_box": f"[{b[0]}, {b[1]}, {b[2]}, {b[3]}]",
                    "validation_status": r.get("automated_validation_status", "CONFIRMED_ALERT"),
                    "rejection_reason": r.get("rejection_reason", ""),
                })

        progress("Analysis complete", 100)
        return VideoAnalysisResult(video, output_path, csv_path, records, str(self.model_path),
                                   self.device, perf_counter() - started, threshold,
                                   int(details["analyzed_frames"]), metric_input_path=metric_input_path,
                                   temporal_consistency_enabled=enable_temporal_consistency)


def result_summary(result: VideoAnalysisResult) -> dict:
    return {
        "run_id": result.run_id,
        "source_video": str(result.video.path), "model_path": result.model_path,
        "source_size_bytes": result.video.size_bytes, "source_modified_ns": result.video.modified_ns,
        "device": result.device, "confidence_threshold": result.threshold,
        "fps": result.video.fps, "duration_seconds": result.video.duration,
        "width": result.video.width, "height": result.video.height,
        "frame_count": result.video.frame_count, "completed_at_utc": result.completed_at_utc,
        "analyzed_frames": result.analyzed_frames, "frames_with_detections": result.positive_frames,
        "total_frame_detections": len(result.detections), "counts_per_class": result.counts,
        "count_definition": "Per-frame observations; the same object may be counted in multiple frames. No tracking IDs are assigned.",
        "frame_numbering": "Zero-based", "box_format": "[x1, y1, x2, y2] in source-frame pixels",
        "elapsed_seconds": round(result.elapsed_seconds, 3), "detections": result.detections,
        "temporal_consistency_enabled": result.temporal_consistency_enabled,
    }


def save_result(result: VideoAnalysisResult, output_dir: Path = OUTPUT_DIR) -> Path:
    """Export the completed analysis and a snapshot of saved analyst reviews."""
    from mockup_ui.observation_review import ReviewStore, atomic_json
    store = ReviewStore.for_result(result)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^a-zA-Z0-9_-]", "_", result.video.path.stem)[:60] or "video"
    name = f"{stem}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    destination = output_dir / name
    destination.mkdir(parents=True, exist_ok=False)
    summary = result_summary(result)
    summary["observations"] = store.observations
    shutil.copy2(result.output_path, destination / "annotated.mp4")
    shutil.copy2(result.csv_path, destination / "detections.csv")
    if result.metric_input_path is not None:
        shutil.copy2(result.metric_input_path, destination / "metric_input.csv")
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    snapshot = dict(store.data)
    snapshot["artifacts"] = {"annotatedVideo": str(destination / "annotated.mp4"), "detectionsCsv": str(destination / "detections.csv")}
    atomic_json(destination / "observation_reviews.json", snapshot)
    from mockup_ui.forensic_report import write_forensic_report
    write_forensic_report(destination / "summary.json", destination)
    return destination
