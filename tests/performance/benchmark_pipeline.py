import argparse
import csv
import time
from pathlib import Path

from app.services.detection_service import DetectionService
from app.services.video_enhancement_service import (
    VideoEnhancementService
)


DEFAULT_INPUT_FRAMES = "outputs/knife_original_frames"
DEFAULT_ENHANCED_FRAMES = "outputs/benchmark_enhanced_frames"
DEFAULT_REPORT = "outputs/reports/benchmark_results.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark BasicVSR++ enhancement and Faster R-CNN "
            "detection."
        )
    )
    parser.add_argument(
        "--input-frames",
        default=DEFAULT_INPUT_FRAMES,
        help="Directory containing original PNG frames.",
    )
    parser.add_argument(
        "--enhanced-frames",
        default=DEFAULT_ENHANCED_FRAMES,
        help="Directory for generated enhanced PNG frames.",
    )
    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT,
        help="CSV path for benchmark results.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.50,
        help="Faster R-CNN confidence threshold.",
    )
    return parser.parse_args()


def write_results(report_path, results):
    report_file = Path(report_path)
    report_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "metric",
        "value",
        "unit",
    ]

    with report_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(
            {
                "metric": metric,
                "value": f"{value:.6f}",
                "unit": unit,
            }
            for metric, value, unit in results
        )


def benchmark(input_frames_dir, enhanced_frames_dir, report_path,
              confidence_threshold):
    input_directory = Path(input_frames_dir)
    frame_count = len(
        list(input_directory.glob("*.png"))
    )

    if frame_count == 0:
        raise RuntimeError(
            f"No PNG frames found in {input_directory}"
        )

    pipeline_start = time.perf_counter()

    enhancement_start = time.perf_counter()
    enhancer = VideoEnhancementService()
    enhancer.run_basicvsrpp(
        input_directory.as_posix(),
        Path(enhanced_frames_dir).as_posix(),
    )
    enhancement_time = (
        time.perf_counter() - enhancement_start
    )

    detection_start = time.perf_counter()
    detector = DetectionService(
        confidence_threshold=confidence_threshold
    )
    detector.detect_frames(
        frames_directory=enhanced_frames_dir,
    )
    detection_time = (
        time.perf_counter() - detection_start
    )

    pipeline_time = (
        time.perf_counter() - pipeline_start
    )

    enhancement_per_frame = enhancement_time / frame_count
    detection_per_frame = detection_time / frame_count
    total_per_frame = pipeline_time / frame_count

    projections = []
    for duration_minutes in (1, 5, 10):
        projected_frames = duration_minutes * 60 * 25
        projected_seconds = (
            projected_frames * total_per_frame
        )
        projections.append(
            (
                f"projected_{duration_minutes}_minute_seconds",
                projected_seconds,
                "seconds",
            )
        )

    results = [
        ("frames_processed", frame_count, "frames"),
        (
            "basicvsrpp_enhancement_time",
            enhancement_time,
            "seconds",
        ),
        ("faster_rcnn_detection_time", detection_time, "seconds"),
        ("total_pipeline_time", pipeline_time, "seconds"),
        (
            "enhancement_time_per_frame",
            enhancement_per_frame,
            "seconds/frame",
        ),
        (
            "detection_time_per_frame",
            detection_per_frame,
            "seconds/frame",
        ),
        (
            "total_time_per_frame",
            total_per_frame,
            "seconds/frame",
        ),
        *projections,
    ]

    write_results(report_path, results)

    print()
    print(f"Frames Processed: {frame_count}")
    print()
    print("BasicVSR++ Enhancement Time:")
    print(f"{enhancement_time:.2f} seconds")
    print()
    print("Detection Time:")
    print(f"{detection_time:.2f} seconds")
    print()
    print("Total Pipeline Time:")
    print(f"{pipeline_time:.2f} seconds")
    print()
    print("Enhancement Time Per Frame:")
    print(f"{enhancement_per_frame:.4f} seconds/frame")
    print()
    print("Detection Time Per Frame:")
    print(f"{detection_per_frame:.4f} seconds/frame")
    print()
    print("Total Time Per Frame:")
    print(f"{total_per_frame:.4f} seconds/frame")
    print()
    print("Projected Total Processing Time at 25 FPS:")
    for duration_minutes in (1, 5, 10):
        projected_seconds = next(
            value
            for metric, value, unit in results
            if metric == (
                f"projected_{duration_minutes}_minute_seconds"
            )
        )
        print(
            f"{duration_minutes} minute video: "
            f"{projected_seconds:.2f} seconds"
        )
    print()
    print(f"CSV report saved to: {report_path}")

    return results


if __name__ == "__main__":
    arguments = parse_args()
    benchmark(
        input_frames_dir=arguments.input_frames,
        enhanced_frames_dir=arguments.enhanced_frames,
        report_path=arguments.report,
        confidence_threshold=arguments.confidence_threshold,
    )
