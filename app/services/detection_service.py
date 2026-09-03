from pathlib import Path
import sys

import csv
import cv2
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset_analysis.build_model import get_model

DEFAULT_MODEL_CANDIDATES = [
    ROOT_DIR / "best_weapon_detector_retrained.pth",
    ROOT_DIR / "best_weapon_detector.pth",
]

class DetectionService:
    """Loads Faster R-CNN and performs object detection on video frames."""

    def __init__(
        self,
        confidence_threshold: float = 0.50,
        model_path: str | Path | None = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.target_classes = {"handgun", "knife"}
        self.model_path = self._resolve_model_path(model_path)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.categories = {
            1: "handgun",
            2: "knife",
        }

        self.model = get_model(num_classes=3)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint was not found: {self.model_path}"
            )

        state_dict = torch.load(
            self.model_path,
            map_location=self.device,
        )

        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded model: {self.model_path.name}")
        print(f"Detection device: {self.device}")

        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    @staticmethod
    def _resolve_model_path(model_path: str | Path | None = None) -> Path:
        if model_path is not None:
            path = Path(model_path)
            if not path.is_absolute():
                path = (ROOT_DIR / path).resolve()
            return path

        for candidate in DEFAULT_MODEL_CANDIDATES:
            if candidate.exists():
                return candidate

        searched = ", ".join(str(p) for p in DEFAULT_MODEL_CANDIDATES)
        raise FileNotFoundError(
            f"No model checkpoint found. Tried: {searched}"
        )

    def detect_frame(self, frame):
        """Runs inference on one OpenCV frame."""

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        image_tensor = torch.from_numpy(rgb_frame)
        image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor = image_tensor.float() / 255.0
        image_tensor = image_tensor.to(self.device)

        with torch.inference_mode():
            prediction = self.model([image_tensor])[0]

        boxes = prediction["boxes"].detach().cpu()
        labels = prediction["labels"].detach().cpu()
        scores = prediction["scores"].detach().cpu()

        detections = []

        for box, label, score in zip(boxes, labels, scores):
            confidence = float(score)

            if confidence < self.confidence_threshold:
                continue

            class_id = int(label)
            class_name = self.categories.get(
                class_id,
                "unknown",
            )

            if class_name not in self.target_classes:
                continue

            x1, y1, x2, y2 = [
                int(value) for value in box.tolist()
            ]

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "box": [x1, y1, x2, y2],
                }
            )

        return detections

    def detect_frames(
        self,
        frames_directory: str,
        csv_output_path: str = None,
    ):
        """Run Faster R-CNN on all PNG frames in a directory."""

        frames_dir = Path(frames_directory)

        if not frames_dir.exists():
            raise FileNotFoundError(
                f"Frames directory not found: {frames_dir}"
            )

        frame_paths = sorted(
            frames_dir.glob("*.png")
        )

        if not frame_paths:
            raise RuntimeError(
                "No PNG frames found."
            )

        detections_report = []

        print(
            f"Processing {len(frame_paths)} frames..."
        )

        for frame_path in frame_paths:
            frame = cv2.imread(
                str(frame_path)
            )

            if frame is None:
                continue

            detections = self.detect_frame(
                frame
            )

            if detections:
                print(
                    f"{frame_path.name}: "
                    f"{len(detections)} detection(s)"
                )

            for detection in detections:
                x1, y1, x2, y2 = detection["box"]
                detections_report.append(
                    {
                        "frame_name": frame_path.name,
                        "class_name": detection["class_name"],
                        "confidence": round(
                            detection["confidence"],
                            4,
                        ),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )

        print(
            f"Total detections: "
            f"{len(detections_report)}"
        )

        if csv_output_path:
            csv_file = Path(
                csv_output_path
            )
            csv_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with open(
                csv_file,
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "frame_name",
                        "class_name",
                        "confidence",
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    detections_report
                )

            print(
                f"CSV saved to: {csv_file}"
            )

        return detections_report

    def detect_and_annotate_frames(
        self,
        frames_directory: str,
        annotated_directory: str,
        csv_output_path: str = None,
    ):
        """Detect weapons, draw boxes, and save annotated PNG frames."""

        frames_dir = Path(frames_directory)
        annotated_dir = Path(annotated_directory)

        if not frames_dir.exists():
            raise FileNotFoundError(
                f"Frames directory not found: {frames_dir}"
            )

        frame_paths = sorted(
            frames_dir.glob("*.png")
        )

        if not frame_paths:
            raise RuntimeError(
                "No PNG frames found."
            )

        annotated_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for old_frame in annotated_dir.glob("*.png"):
            old_frame.unlink()

        detections_report = []

        print(
            f"Annotating {len(frame_paths)} frames..."
        )

        for frame_path in frame_paths:
            frame = cv2.imread(
                str(frame_path)
            )

            if frame is None:
                continue

            detections = self.detect_frame(
                frame
            )
            annotated_frame = draw_detections(
                frame,
                detections,
            )

            output_path = annotated_dir / frame_path.name
            if not cv2.imwrite(
                str(output_path),
                annotated_frame,
            ):
                raise RuntimeError(
                    f"Could not write annotated frame: {output_path}"
                )

            for detection in detections:
                x1, y1, x2, y2 = detection["box"]
                detections_report.append(
                    {
                        "frame_name": frame_path.name,
                        "class_name": detection["class_name"],
                        "confidence": round(
                            detection["confidence"],
                            4,
                        ),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )

        print(
            f"Total detections: "
            f"{len(detections_report)}"
        )

        if csv_output_path:
            csv_file = Path(csv_output_path)
            csv_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with open(
                csv_file,
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "frame_name",
                        "class_name",
                        "confidence",
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    detections_report
                )

            print(
                f"CSV saved to: {csv_file}"
            )

        return detections_report

def draw_detections(frame, detections):
    """Draws weapon bounding boxes and labels on a video frame."""

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        confidence = detection["confidence"]

        label = (
            f"{detection['class_name']} "
            f"{confidence:.1%}"
        )

        # Red bounding box in OpenCV BGR format.
        color = (0, 0, 255)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        # Calculate the text background size.
        text_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )

        text_width, text_height = text_size

        # Keep the label inside the frame.
        label_y = max(
            text_height + baseline + 5,
            y1,
        )

        cv2.rectangle(
            frame,
            (
                x1,
                label_y - text_height - baseline - 5,
            ),
            (
                x1 + text_width + 8,
                label_y,
            ),
            color,
            -1,
        )

        cv2.putText(
            frame,
            label,
            (x1 + 4, label_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame

def test_first_frame(video_path: str):
    """Reads one video frame and tests Faster R-CNN inference."""

    video_file = Path(video_path)

    if not video_file.exists():
        raise FileNotFoundError(
            f"Video was not found: {video_file}"
        )

    capture = cv2.VideoCapture(str(video_file))

    if not capture.isOpened():
        raise RuntimeError(
            f"OpenCV could not open: {video_file}"
        )

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    middle_frame = total_frames // 2

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        middle_frame,
    )

    success, frame = capture.read()
    capture.release()

    print(f"Testing frame: {middle_frame}")

    if not success or frame is None:
        raise RuntimeError(
            "OpenCV could not read the selected video frame."
        )

    detector = DetectionService(
        confidence_threshold=0.50
    )

    detections = detector.detect_frame(frame)

    print(f"Frame dimensions: {frame.shape}")
    print(
        f"Weapon detections above threshold: {len(detections)}"
    )

    for detection in detections:
        print(
            f"{detection['class_name']}: "
            f"{detection['confidence']:.2%}, "
            f"box={detection['box']}"
        )

def scan_video_for_weapons(video_path: str):
    """Scans multiple video frames for weapon detections."""

    video_file = Path(video_path)

    if not video_file.exists():
        raise FileNotFoundError(
            f"Video was not found: {video_file}"
        )

    capture = cv2.VideoCapture(str(video_file))

    if not capture.isOpened():
        raise RuntimeError(
            f"OpenCV could not open: {video_file}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0:
        capture.release()
        raise RuntimeError(
            "The video has an invalid frame rate."
        )

    duration = total_frames / fps

    print(f"Video: {video_file.name}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")

    detector = DetectionService(
        confidence_threshold=0.25
    )

    # Analyze approximately five frames per second.
    analysis_fps = 5
    frame_interval = max(
        1,
        round(fps / analysis_fps),
    )

    frame_number = 0
    analyzed_frames = 0
    total_detections = 0

    while frame_number < total_frames:
        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number,
        )

        success, frame = capture.read()

        if not success or frame is None:
            print(
                f"Could not read frame {frame_number}."
            )
            break

        detections = detector.detect_frame(frame)
        timestamp = frame_number / fps
        analyzed_frames += 1

        print(
            f"Frame {frame_number:04d} | "
            f"{timestamp:05.2f}s | "
            f"{len(detections)} weapon detection(s)"
        )

        for detection in detections:
            total_detections += 1

            print(
                f"  {detection['class_name']}: "
                f"{detection['confidence']:.2%} | "
                f"Box: {detection['box']}"
            )

        frame_number += frame_interval

    capture.release()

    print()
    print("Scan completed.")
    print(f"Analyzed frames: {analyzed_frames}")
    print(f"Total weapon detections: {total_detections}")


def scan_video_for_knives(video_path: str):
    """Backward-compatible alias for scan_video_for_weapons."""

    return scan_video_for_weapons(video_path)

def process_video(
    input_path: str,
    output_path: str,
    confidence_threshold: float = 0.50,
    analysis_fps: int = 5,
):
    """Processes a complete video and saves an annotated MP4."""

    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input video was not found: {input_file}"
        )

    # Create the output directory if it does not exist.
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    capture = cv2.VideoCapture(str(input_file))

    if not capture.isOpened():
        raise RuntimeError(
            f"OpenCV could not open: {input_file}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    if fps <= 0:
        capture.release()
        raise RuntimeError(
            "The input video has an invalid frame rate."
        )

    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(
            "The input video has invalid dimensions."
        )

    duration = total_frames / fps

    print("Starting video processing")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    print(
        f"Visible confidence threshold: "
        f"{confidence_threshold:.2f}"
    )

    # mp4v provides broad compatibility for an initial MP4 output.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_file),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not create output video: {output_file}"
        )

    detector = DetectionService(
        confidence_threshold=confidence_threshold
    )

    # Analyze approximately the requested number of frames per second.
    frame_interval = max(
        1,
        round(fps / analysis_fps),
    )

    frame_number = 0
    analyzed_frames = 0
    total_detections = 0
    detection_records = []

    while True:
        success, frame = capture.read()

        if not success:
            break

        timestamp = frame_number / fps

        # Only selected frames undergo Faster R-CNN inference.
        if frame_number % frame_interval == 0:
            detections = detector.detect_frame(frame)

            analyzed_frames += 1
            total_detections += len(detections)

            frame = draw_detections(
                frame,
                detections,
            )

            if detections:
                print(
                    f"Frame {frame_number:04d} | "
                    f"{timestamp:05.2f}s | "
                    f"{len(detections)} detection(s)"
                )

                for detection in detections:
                    x1, y1, x2, y2 = detection["box"]
                    detection_records.append(
                        {
                            "frame_number": frame_number,
                            "timestamp_seconds": round(
                                timestamp,
                                2,
                            ),
                            "class_name": detection[
                                "class_name"
                            ],
                            "confidence": round(
                                detection["confidence"],
                                4,
                            ),
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        }
                    )
                    print(
                        f"  {detection['class_name']}: "
                        f"{detection['confidence']:.2%} | "
                        f"Box: {detection['box']}"
                    )

        # Add the frame number and timestamp to every output frame.
        timestamp_label = (
            f"Frame {frame_number} | "
            f"Time {timestamp:.2f}s"
        )

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
            progress = (
                frame_number / total_frames
            ) * 100

            # Print progress at approximately every 10%.
            progress_interval = max(
                1,
                total_frames // 10,
            )

            if (
                frame_number % progress_interval == 0
                or frame_number == total_frames
            ):
                print(
                    f"Progress: "
                    f"{min(progress, 100):.1f}%"
                )

    capture.release()
    writer.release()

    reports_dir = Path(
        "outputs/reports"
    )
    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    csv_path = (
        reports_dir
        / f"{input_file.stem}_detections.csv"
    )

    with open(
        csv_path,
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer_csv = csv.DictWriter(
            csv_file,
            fieldnames=[
                "frame_number",
                "timestamp_seconds",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
            ],
        )
        writer_csv.writeheader()
        writer_csv.writerows(
            detection_records
        )

    if not output_file.exists():
        raise RuntimeError(
            "Processing finished, but the output "
            "video was not created."
        )

    output_size = output_file.stat().st_size

    print()
    print("Video processing completed.")
    print(f"Frames written: {frame_number}")
    print(f"Frames analyzed: {analyzed_frames}")
    print(f"Weapon detections: {total_detections}")
    print(f"Output size: {output_size:,} bytes")
    print(f"Output saved to: {output_file}")
    print(f"CSV report saved to: {csv_path}")

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
    }

if __name__ == "__main__":
    process_video(
        input_path="samples/evaluation_video.mp4",
        output_path=(
            "outputs/videos/"
            "Knife_test_3(newmodel).mp4"
        ),
        confidence_threshold=0.50,
        analysis_fps=25,
    )
