"""
Extracts 30 full-resolution (1080p/720p) uncropped background frames
containing the exact empty room fixtures causing false alarms:
1. Clip 04 store counter
2. CAM02 staircase handrail and stringer
3. Clip 02 acrylic basket divider and counter
4. Clip 03 cashier counter and ceiling signs
"""
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[3]
SAMPLES_DIR = ROOT / "samples"
UNSEEN_DIR = SAMPLES_DIR / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS"

OUT_DIR = ROOT / "research" / "model_improvement" / "dataset" / "hard_negatives_fullframe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

jobs = [
    # 1. Clip 04: Empty store counter
    {
        "video": UNSEEN_DIR / "Screen Recording 2026-09-07 230047.mp4",
        "prefix": "full_counter_clip04",
        "frames": [0, 5, 10, 15, 20, 25, 30, 40]
    },
    # 2. CAM02: Empty staircase and banister
    {
        "video": SAMPLES_DIR / "CAM02_Scene_004.mp4",
        "prefix": "full_staircase_cam02",
        "frames": [0, 5, 10, 15, 20, 25, 30, 35]
    },
    # 3. Clip 02: Empty pawnshop counter and acrylic divider
    {
        "video": UNSEEN_DIR / "Screen Recording 2026-09-07 225138.mp4",
        "prefix": "full_divider_clip02",
        "frames": [0, 5, 10, 15, 20, 25, 30]
    },
    # 4. Clip 03: Cashier counter and ceiling fixtures
    {
        "video": UNSEEN_DIR / "Screen Recording 2026-09-07 225703.mp4",
        "prefix": "full_cashier_clip03",
        "frames": [0, 2, 4, 6, 8, 10, 12]
    },
]

count = 0
for job in jobs:
    vpath = job["video"]
    if not vpath.exists():
        print(f"[!] Warning: Missing video {vpath}")
        continue
    cap = cv2.VideoCapture(str(vpath))
    for fidx in job["frames"]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if ret and frame is not None:
            out_file = OUT_DIR / f"{job['prefix']}_frame_{fidx:04d}.jpg"
            cv2.imwrite(str(out_file), frame)
            count += 1
    cap.release()

print(f"Extracted {count} full-resolution background frames into {OUT_DIR}!")
