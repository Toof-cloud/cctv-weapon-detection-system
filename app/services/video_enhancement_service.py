from pathlib import Path
import shutil

from app.services.video_service import VideoService


class VideoEnhancementService:

    def __init__(self):
        self.video_service = VideoService()

    def enhance_video(
        self,
        input_video_path: str,
        output_video_path: str,
    ):

        frames_dir = "outputs/frames"
        enhanced_dir = "outputs/frames_enhanced"

        metadata = self.video_service.get_metadata(
            input_video_path
        )

        self.video_service.extract_frames(
            input_video_path,
            frames_dir,
        )

        self.run_basicvsrpp(
            frames_dir,
            enhanced_dir,
        )

        self.video_service.rebuild_video(
            enhanced_dir,
            output_video_path,
            metadata["fps"],
        )

    def run_basicvsrpp(
        self,
        input_frames_dir,
        output_frames_dir,
    ):
        """
        Temporary placeholder.

        Copies original frames into the
        enhanced frames directory.

        Later this will be replaced by
        actual BasicVSR++ inference.
        """

        input_dir = Path(input_frames_dir)
        output_dir = Path(output_frames_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        frame_paths = sorted(
            input_dir.glob("*.png")
        )

        copied_frames = 0

        for frame_path in frame_paths:

            destination = (
                output_dir / frame_path.name
            )

            shutil.copy2(
                frame_path,
                destination,
            )

            copied_frames += 1

        print(
            f"Copied {copied_frames} frames "
            f"to {output_dir}"
        )