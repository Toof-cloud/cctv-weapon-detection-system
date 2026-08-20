from app.services.video_service import (
    VideoService
)

video = VideoService()

metadata = video.get_metadata(
    "samples/handgun_test-video.mp4"
)

print(metadata)