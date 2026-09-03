# tests/detection/test_new_knife_baseline.py

from app.services.detection_service import process_video

process_video(
    input_path="samples/NEW_KNIFE_VIDEO_11s.mp4",
    output_path="outputs/videos/new_knife_baseline.mp4",
    confidence_threshold=0.50,
    analysis_fps=15,
)