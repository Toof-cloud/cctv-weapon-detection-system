import math
from typing import Optional


def frame_to_seconds(frame_number: int, fps: float) -> float:
    """Converts a frame number to elapsed time in seconds."""
    if fps <= 0:
        return 0.0
    return float(frame_number) / float(fps)


def seconds_to_frame(seconds: float, fps: float) -> int:
    """Converts elapsed seconds to the corresponding frame index."""
    if fps <= 0:
        return 0
    return int(round(seconds * fps))


def format_timestamp(seconds: float, include_milliseconds: bool = True) -> str:
    """Formats seconds into human-readable HH:MM:SS.mmm or MM:SS.mmm string.
    
    Examples:
        format_timestamp(3.98) -> '00:03.980'
        format_timestamp(3665.25) -> '01:01:05.250'
    """
    if seconds is None or math.isnan(seconds) or seconds < 0:
        seconds = 0.0

    total_millis = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        base = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        base = f"{minutes:02d}:{secs:02d}"

    if include_milliseconds:
        return f"{base}.{total_millis:03d}"
    return base


def seconds_to_timecode(seconds: float, fps: float = 30.0) -> str:
    """Formats seconds into SMPTE-style timecode string (HH:MM:SS:FF)."""
    if seconds < 0:
        seconds = 0.0
    fps = max(1.0, fps)
    total_frames = int(round(seconds * fps))
    frame_rem = total_frames % int(round(fps))
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frame_rem:02d}"
