"""
Extracts 15 diverse hard negative background crops from false-alarm regions:
- Clip 04 store counters and newspaper racks
- Clip 02 acrylic basket divider
- Clip 06 store glass counter
- CAM02 staircase handrail and stringer
"""
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[3]
SAMPLES_DIR = ROOT / "samples"
UNSEEN_DIR = SAMPLES_DIR / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS"

NEG_OUT_DIR = ROOT / "research" / "model_improvement" / "dataset" / "hard_negatives"
NEG_OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Clip 04: Multiple counter regions across frames
cap4 = cv2.VideoCapture(str(UNSEEN_DIR / "Screen Recording 2026-09-07 230047.mp4"))
for fidx in [0, 10, 25, 50, 100]:
    cap4.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ret, f = cap4.read()
    if ret:
        cv2.imwrite(str(NEG_OUT_DIR / f"neg_counter_clip04_f{fidx}_1.jpg"), f[200:700, 100:1000])
        cv2.imwrite(str(NEG_OUT_DIR / f"neg_counter_clip04_f{fidx}_2.jpg"), f[300:900, 800:1650])
cap4.release()

# 2. Clip 02: Basket divider
cap2 = cv2.VideoCapture(str(UNSEEN_DIR / "Screen Recording 2026-09-07 225138.mp4"))
for fidx in [5, 35, 70]:
    cap2.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ret, f = cap2.read()
    if ret:
        cv2.imwrite(str(NEG_OUT_DIR / f"neg_divider_clip02_f{fidx}.jpg"), f[600:1050, 1680:1920])
cap2.release()

# 3. CAM02: Staircase railing
cap_cam = cv2.VideoCapture(str(SAMPLES_DIR / "CAM02_Scene_004.mp4"))
for fidx in [0, 20, 50]:
    cap_cam.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ret, f = cap_cam.read()
    if ret:
        cv2.imwrite(str(NEG_OUT_DIR / f"neg_railing_cam02_f{fidx}.jpg"), f[0:550, 850:1280])
cap_cam.release()

print(f"Generated {len(list(NEG_OUT_DIR.glob('*.jpg')))} diverse hard negative background crops!")
