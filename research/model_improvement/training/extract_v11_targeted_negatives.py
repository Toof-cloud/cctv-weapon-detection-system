"""
Targeted Hard Negative Extractor for Candidate Model V11.
Extracts empirical failure patches identified during user testing:
1. CAM2_SCENE002: Empty hand gesture resting on door with cast shadow (frames 0-60).
2. CAM2_SCENE002: Actor dark ponytail against pink shirt (frames 0-60).
3. CAM2_SCENE001: Counter base and floor shadows (frames 55-90).
4. Unseen 225703: Customer jacket / shoulder shadow (frames 160-185).
5. Unseen 230249: Cashier black baseball cap / headphones (frames 400-620).
6. Unseen 230249: Cash register display / POS terminal (frames 400-620).
"""
import os
import sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[3]
NEG_V11_DIR = ROOT / "research" / "model_improvement" / "dataset" / "hard_negatives_v11"
NEG_V11_DIR.mkdir(parents=True, exist_ok=True)


def extract_crops(video_path: Path, frame_specs: list, prefix: str):
    if not video_path.exists():
        print(f"[Warning] Video not found: {video_path}")
        return 0

    cap = cv2.VideoCapture(str(video_path))
    count = 0
    spec_map = {item["frame"]: item for item in frame_specs}
    max_frame = max(spec_map.keys()) if spec_map else 0

    cur_fn = 0
    while True:
        ret, frame = cap.read()
        if not ret or cur_fn > max_frame + 2:
            break

        if cur_fn in spec_map:
            spec = spec_map[cur_fn]
            h, w = frame.shape[:2]
            for crop_box, name in spec.get("crops", []):
                x1, y1, x2, y2 = [int(v) for v in crop_box]
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(x1 + 10, min(w, x2))
                y2 = max(y1 + 10, min(h, y2))
                crop_img = frame[y1:y2, x1:x2]
                if crop_img.size > 0:
                    out_path = NEG_V11_DIR / f"{prefix}_f{cur_fn:04d}_{name}.jpg"
                    cv2.imwrite(str(out_path), crop_img)
                    count += 1
        cur_fn += 1

    cap.release()
    return count


def main():
    print("=" * 80)
    print("EXTRACTING CANDIDATE V11 TARGETED HARD NEGATIVES")
    print(f"Target Output Directory: {NEG_V11_DIR}")
    print("=" * 80)

    total_extracted = 0

    # 1. CAM2_SCENE002: Door hand shadow and ponytail
    cam2_s002 = ROOT / "samples" / "NEW_STAGED_CAM-02" / "NEW_CAMERA 02" / "CAM2_SCENE002.mp4"
    specs_cam2_s002 = []
    for fn in range(0, 65, 3):
        specs_cam2_s002.append({
            "frame": fn,
            "crops": [
                ([445, 330, 510, 420], "door_hand_shadow"),
                ([340, 120, 410, 230], "ponytail_pink_shirt"),
            ]
        })
    c1 = extract_crops(cam2_s002, specs_cam2_s002, "cam2_s002")
    print(f"[*] Extracted {c1} crops from CAM2_SCENE002.mp4 (door hand shadow + ponytail)")
    total_extracted += c1

    # 2. CAM2_SCENE001: Floor & counter base shadows
    cam2_s001 = ROOT / "samples" / "NEW_STAGED_CAM-02" / "NEW_CAMERA 02" / "CAM2_SCENE001.mp4"
    specs_cam2_s001 = []
    for fn in range(50, 95, 4):
        specs_cam2_s001.append({
            "frame": fn,
            "crops": [
                ([60, 410, 850, 680], "counter_floor_shadow"),
                ([330, 150, 410, 260], "cam2_s1_gesture_shadow"),
            ]
        })
    c2 = extract_crops(cam2_s001, specs_cam2_s001, "cam2_s001")
    print(f"[*] Extracted {c2} crops from CAM2_SCENE001.mp4 (floor shadows + gesture)")
    total_extracted += c2

    # 3. Unseen 225703: Customer jacket / shoulder shadow
    unseen_225703 = ROOT / "samples" / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS" / "Screen Recording 2026-09-07 225703.mp4"
    specs_225703 = []
    for fn in range(160, 185, 2):
        specs_225703.append({
            "frame": fn,
            "crops": [
                ([240, 340, 350, 430], "customer_shoulder_shadow"),
            ]
        })
    c3 = extract_crops(unseen_225703, specs_225703, "unseen_225703")
    print(f"[*] Extracted {c3} crops from 225703 (customer shoulder shadow)")
    total_extracted += c3

    # 4. Unseen 230249: Cashier black cap / headphones & POS terminal
    unseen_230249 = ROOT / "samples" / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS" / "Screen Recording 2026-09-07 230249.mp4"
    specs_230249 = []
    for fn in range(400, 620, 15):
        specs_230249.append({
            "frame": fn,
            "crops": [
                ([1500, 200, 1850, 500], "cashier_cap_headphones"),
                ([950, 250, 1150, 430], "pos_terminal_display"),
            ]
        })
    c4 = extract_crops(unseen_230249, specs_230249, "unseen_230249")
    print(f"[*] Extracted {c4} crops from 230249 (cashier cap/headphones + POS terminal)")
    total_extracted += c4

    print("=" * 80)
    print(f"TOTAL V11 TARGETED HARD NEGATIVES EXTRACTED: {total_extracted}")
    print("=" * 80)


if __name__ == "__main__":
    main()
