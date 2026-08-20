from app.services.video_enhancement_service import (
    VideoEnhancementService
)

enhancer = VideoEnhancementService()

enhancer.enhance_video(
    input_video_path="samples/handgun_test-video.mp4",
    output_video_path="outputs/videos/enhanced_test.mp4",
)

print("Pipeline completed.")