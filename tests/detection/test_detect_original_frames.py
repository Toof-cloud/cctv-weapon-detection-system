from app.services.detection_service import (
    DetectionService
)


detector = DetectionService(
    confidence_threshold=0.50
)

results = detector.detect_frames(
    frames_directory="outputs/frames",
    csv_output_path=(
        "outputs/reports/"
        "original_frame_detections.csv"
    ),
)

print(
    f"Results: {len(results)}"
)
