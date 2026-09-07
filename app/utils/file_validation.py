from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".ts"}


def validate_video_file(path: str | Path) -> Tuple[bool, str, Dict[str, Any]]:
    """Validates video file format, decodability, and extracts metadata.
    
    Returns:
        (is_valid, error_message, metadata_dict)
    """
    file_path = Path(path)
    if not file_path.exists():
        return False, f"File does not exist: {file_path}", {}

    if not file_path.is_file():
        return False, f"Path is not a file: {file_path}", {}

    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        return (
            False,
            f"Unsupported video extension '{ext}'. Supported: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}",
            {},
        )

    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        return False, f"OpenCV could not decode video container: {file_path.name}", {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))

    # Read the first frame to confirm decodability
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return False, f"Failed to decode initial video frame from: {file_path.name}", {}

    if fps <= 0 or width <= 0 or height <= 0:
        return False, "Video stream has invalid dimension or frame rate properties.", {}

    duration = total_frames / fps if fps > 0 else 0.0

    metadata = {
        "filename": file_path.name,
        "path": str(file_path.resolve()),
        "file_size_bytes": file_path.stat().st_size,
        "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "total_frames": total_frames,
        "duration_seconds": round(duration, 2),
    }

    return True, "", metadata
