from pathlib import Path

import cv2


class VideoService:
    """
    Handles video validation, metadata extraction,
    frame extraction, and video reconstruction.
    """

    def validate_video(self, video_path: str) -> bool:
        video_file = Path(video_path)

        if not video_file.exists():
            raise FileNotFoundError(
                f"Video not found: {video_file}"
            )

        capture = cv2.VideoCapture(str(video_file))

        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open video: {video_file}"
            )

        capture.release()
        return True

    def get_metadata(self, video_path: str):
        capture = cv2.VideoCapture(video_path)

        fps = capture.get(cv2.CAP_PROP_FPS)

        total_frames = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        width = int(
            capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        duration = (
            total_frames / fps
            if fps > 0
            else 0
        )

        capture.release()

        return {
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "duration_seconds": duration,
        }

    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
    ):
        output_directory = Path(output_dir)

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        capture = cv2.VideoCapture(video_path)

        frame_number = 0

        while True:
            success, frame = capture.read()

            if not success:
                break

            frame_path = (
                output_directory
                / f"frame_{frame_number:06d}.png"
            )

            cv2.imwrite(
                str(frame_path),
                frame,
            )

            frame_number += 1

        capture.release()

        print(
            f"Extracted {frame_number} frames "
            f"to {output_directory}"
        )

        return frame_number

    def rebuild_video(
        self,
        frames_directory: str,
        output_video_path: str,
        fps: float,
    ):
        frames_dir = Path(frames_directory)

        frame_paths = sorted(
            frames_dir.glob("*.png")
        )

        if not frame_paths:
            raise RuntimeError(
                "No frames found."
            )

        first_frame = cv2.imread(
            str(frame_paths[0])
        )

        height, width = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            output_video_path,
            fourcc,
            fps,
            (width, height),
        )

        for frame_path in frame_paths:
            frame = cv2.imread(
                str(frame_path)
            )

            writer.write(frame)

        writer.release()

        print(
            f"Video rebuilt: "
            f"{output_video_path}"
        )