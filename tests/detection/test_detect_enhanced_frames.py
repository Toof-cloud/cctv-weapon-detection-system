from app.services.detection_service import (
    DetectionService
)


detector = DetectionService(
    confidence_threshold=0.50
)

results = detector.detect_frames(
    frames_directory="outputs/frames_enhanced",
    csv_output_path=(
        "outputs/reports/"
        "enhanced_frame_detections.csv"
    ),
)

print(
    f"Results: {len(results)}"
)