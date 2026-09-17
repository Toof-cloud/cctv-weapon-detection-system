import cv2
import os
from pathlib import Path

ROOT = Path(".")
NEG_DIR = ROOT / "research" / "model_improvement" / "dataset" / "hard_negatives"
FULL_DIR = ROOT / "research" / "model_improvement" / "dataset" / "hard_negatives_fullframe"
NEG_DIR.mkdir(parents=True, exist_ok=True)
FULL_DIR.mkdir(parents=True, exist_ok=True)

# 1. Extract Clip 02 acrylic divider crops and full frames
clip02_path = "samples/unseen_samples/VIDEO_CLIPS_JABEZ_UNSEEN/VIDEO CLIPS/Screen Recording 2026-09-07 225138.mp4"
cap = cv2.VideoCapture(clip02_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

sample_frames = [0, 20, 50, 80, 120, 150]
fidx = 0
extracted_crops = 0
extracted_full = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if fidx in sample_frames:
        # Full frame negative
        full_name = f"neg_clip02_fullframe_f{fidx:04d}.jpg"
        cv2.imwrite(str(FULL_DIR / full_name), frame)
        extracted_full += 1

        # Acrylic divider crop [1724, 676, 1919, 1011]
        crop = frame[670:1015, 1720:1920]
        crop_name = f"neg_clip02_acrylic_divider_f{fidx:04d}.jpg"
        cv2.imwrite(str(NEG_DIR / crop_name), crop)
        extracted_crops += 1

    fidx += 1
cap.release()

# 2. Extract Clip 03 POS monitor crop
clip03_path = "samples/unseen_samples/VIDEO_CLIPS_JABEZ_UNSEEN/VIDEO CLIPS/Screen Recording 2026-09-07 225703.mp4"
cap = cv2.VideoCapture(clip03_path)
fidx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if fidx in [5, 10, 15, 125, 130]:
        # Full frame
        full_name = f"neg_clip03_fullframe_f{fidx:04d}.jpg"
        cv2.imwrite(str(FULL_DIR / full_name), frame)
        extracted_full += 1

        # Monitor crop [1409, 618, 1463, 677] with padding
        crop = frame[600:700, 1390:1480]
        crop_name = f"neg_clip03_pos_monitor_f{fidx:04d}.jpg"
        cv2.imwrite(str(NEG_DIR / crop_name), crop)
        extracted_crops += 1
    fidx += 1
cap.release()

print(f"Extracted {extracted_crops} new hard negative crops to {NEG_DIR}")
print(f"Extracted {extracted_full} new full-frame negatives to {FULL_DIR}")
