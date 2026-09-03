# tests/enhancement/test_new_video_metadata.py

from app.services.video_service import VideoService

video = VideoService()

metadata = video.get_metadata(
    "samples/NEW_KNIFE_VIDEO_11s.mp4"
)

print(metadata)