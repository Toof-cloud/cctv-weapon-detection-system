import subprocess
from pathlib import Path

import cv2

from app.services.video_service import VideoService


class VideoEnhancementService:

    def __init__(self):
        self.video_service = VideoService()

        self.wsl_project_dir = (
            "/home/pc/thesis/mmagic"
        )

    def enhance_video(
        self,
        input_video_path: str,
        output_video_path: str,
    ):

        frames_dir = "outputs/frames_for_enhancement"
        enhanced_dir = "outputs/frames_enhanced"

        metadata = self.video_service.get_metadata(
            input_video_path
        )

        for directory in [frames_dir, enhanced_dir]:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            for old_frame in path.glob("*.png"):
                old_frame.unlink()

        self.video_service.extract_frames(
            input_video_path,
            frames_dir,
        )

        self.resize_frames_for_enhancement(
            frames_dir,
            max_dimension=960,
        )

        self.run_basicvsrpp(
            frames_dir,
            enhanced_dir,
        )

        self.restore_frame_resolution(
            enhanced_dir,
            metadata["width"],
            metadata["height"],
        )

        self.video_service.rebuild_video(
            enhanced_dir,
            output_video_path,
            metadata["fps"],
        )

    @staticmethod
    def resize_frames_for_enhancement(
        frames_directory: str,
        max_dimension: int,
    ):
        frames_dir = Path(frames_directory)

        for frame_path in frames_dir.glob("*.png"):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(
                    f"Could not read enhancement frame: {frame_path}"
                )

            height, width = frame.shape[:2]
            largest_dimension = max(width, height)
            if largest_dimension <= max_dimension:
                continue

            scale = max_dimension / largest_dimension
            resized = cv2.resize(
                frame,
                (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
            if not cv2.imwrite(str(frame_path), resized):
                raise RuntimeError(
                    f"Could not resize enhancement frame: {frame_path}"
                )

    @staticmethod
    def restore_frame_resolution(
        frames_directory: str,
        width: int,
        height: int,
    ):
        frames_dir = Path(frames_directory)

        for frame_path in frames_dir.glob("*.png"):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(
                    f"Could not read enhanced frame: {frame_path}"
                )

            restored = cv2.resize(
                frame,
                (width, height),
                interpolation=cv2.INTER_CUBIC,
            )
            if not cv2.imwrite(str(frame_path), restored):
                raise RuntimeError(
                    f"Could not restore enhanced frame: {frame_path}"
                )

    def run_basicvsrpp(
        self,
        input_frames_dir,
        output_frames_dir,
    ):
        """Run chunked BasicVSR++ inference through WSL."""

        output_dir = Path(output_frames_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for output_frame in output_dir.glob("*.png"):
            output_frame.unlink()

        input_wsl = (
            "/mnt/c/Users/pc/Documents/THESIS 1/"
            "cctv-weapon-detection-system/"
            f"{input_frames_dir}"
        )
        output_wsl = (
            "/mnt/c/Users/pc/Documents/THESIS 1/"
            "cctv-weapon-detection-system/"
            f"{output_frames_dir}"
        )

        result = subprocess.run(
            [
                "wsl",
                "/home/pc/thesis/mmagic_env/bin/python",
                "/home/pc/thesis/mmagic/"
                "basicvsr_enhance_folder.py",
                input_wsl,
                output_wsl,
            ],
            capture_output=True,
            text=True,
        )

        print(result.stdout)

        if result.returncode != 0:
            raise RuntimeError(
                result.stderr
            )

    def verify_wsl_access(self):
        result = subprocess.run(
            [
                "wsl",
                "bash",
                "-c",
                "echo BasicVSR_OK",
            ],
            capture_output=True,
            text=True,
        )

        return (
            result.returncode == 0
            and "BasicVSR_OK"
            in result.stdout
        )