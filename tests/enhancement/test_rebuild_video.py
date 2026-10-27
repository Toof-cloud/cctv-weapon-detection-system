from app.services.video_service import VideoService

video = VideoService()

video.rebuild_video(
    frames_directory="outputs/frames",
    output_video_path="outputs/videos/rebuilt_test.mp4",
    fps=25,
)

print("Video rebuilt successfully.")