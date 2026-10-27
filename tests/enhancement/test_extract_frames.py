# test_extract_frames.py

from app.services.video_service import VideoService

video = VideoService()

count = video.extract_frames(
    "samples/handgun_test-video.mp4",
    "outputs/frames"
)

print(
    f"Frames extracted: {count}"
)