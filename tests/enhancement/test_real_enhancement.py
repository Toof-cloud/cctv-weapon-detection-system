from app.services.video_enhancement_service import (
    VideoEnhancementService
)


enhancer = VideoEnhancementService()

print("Starting BasicVSR++ enhancement...")
print("Input folder: outputs/frames")
print("Output folder: outputs/frames_enhanced")

enhancer.run_basicvsrpp(
    "outputs/frames",
    "outputs/frames_enhanced"
)

print("BasicVSR++ complete.")