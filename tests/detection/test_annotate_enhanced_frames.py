from app.services.detection_service import (
    DetectionService
)
from app.services.video_service import VideoService


input_video_path = "samples/handgun_test-video.mp4"
annotated_directory = "outputs/frames_annotated"
output_video_path = (
    "outputs/videos/enhanced_detected_video.mp4"
)

detector = DetectionService(
    confidence_threshold=0.50
)

results = detector.detect_and_annotate_frames(
    frames_directory="outputs/frames_enhanced",
    annotated_directory=annotated_directory,
    csv_output_path=(
        "outputs/reports/"
        "enhanced_annotated_detections.csv"
    ),
)

video_service = VideoService()
metadata = video_service.get_metadata(
    input_video_path
)
video_service.rebuild_video(
    annotated_directory,
    output_video_path,
    metadata["fps"],
)

print(
    f"Annotated frames: {len(results)} detections"
)
print(
    f"Annotated video: {output_video_path}"
)