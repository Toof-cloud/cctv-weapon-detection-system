from app.services.video_enhancement_service import (
    VideoEnhancementService
)


service = VideoEnhancementService()

print(
    service.verify_wsl_access()
)
