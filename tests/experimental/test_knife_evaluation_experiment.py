import csv
from pathlib import Path

from app.services.detection_service import (
    DetectionService
)
from app.services.video_enhancement_service import (
    VideoEnhancementService
)
from app.services.video_service import VideoService


video_path = "samples/evaluation_video.mp4"
original_frames = "outputs/knife_original_frames"
enhanced_frames = "outputs/knife_enhanced_frames"
annotated_frames = "outputs/knife_annotated_frames"
original_report = (
    "outputs/reports/evaluation_original.csv"
)
enhanced_report = (
    "outputs/reports/evaluation_enhanced.csv"
)
annotated_report = (
    "outputs/reports/evaluation_enhanced_annotated.csv"
)
annotated_video = (
    "outputs/videos/evaluation_enhanced_detected.mp4"
)

video_service = VideoService()
video_service.extract_frames(
    video_path,
    original_frames,
)

enhancer = VideoEnhancementService()
enhancer.run_basicvsrpp(
    original_frames,
    enhanced_frames,
)

detector = DetectionService(
    confidence_threshold=0.50
)

original_results = detector.detect_frames(
    frames_directory=original_frames,
    csv_output_path=original_report,
)
enhanced_results = detector.detect_frames(
    frames_directory=enhanced_frames,
    csv_output_path=enhanced_report,
)

annotated_results = detector.detect_and_annotate_frames(
    frames_directory=enhanced_frames,
    annotated_directory=annotated_frames,
    csv_output_path=annotated_report,
)

metadata = video_service.get_metadata(video_path)
video_service.rebuild_video(
    annotated_frames,
    annotated_video,
    metadata["fps"],
)


def summarize(results):
    confidences = [
        result["confidence"]
        for result in results
    ]
    return {
        "detections": len(results),
        "average_confidence": (
            round(
                sum(confidences) / len(confidences),
                4,
            )
            if confidences
            else 0
        ),
        "handgun": sum(
            result["class_name"] == "handgun"
            for result in results
        ),
        "knife": sum(
            result["class_name"] == "knife"
            for result in results
        ),
    }


print("Knife evaluation comparison:")
print(f"Original: {summarize(original_results)}")
print(f"Enhanced: {summarize(enhanced_results)}")
print(
    f"Annotated frames: {len(annotated_results)} detections"
)
print(f"Annotated video: {annotated_video}")
