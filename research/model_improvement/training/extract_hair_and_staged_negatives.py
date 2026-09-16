import os
import sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[3]
NEG_DIR = ROOT / "research" / "model_improvement" / "dataset" / "hard_negatives_v10"
NEG_DIR.mkdir(parents=True, exist_ok=True)

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
                    out_path = NEG_DIR / f"{prefix}_f{cur_fn:04d}_{name}.jpg"
                    cv2.imwrite(str(out_path), crop_img)
                    count += 1
        cur_fn += 1

    cap.release()
    return count

def main():
    print("=" * 76)
    print("EXTRACTING CANDIDATE V10 TARGETED HARD NEGATIVES (HAIR, PONYTAILS, CLOTHING)")
    print(f"Target Directory: {NEG_DIR}")
    print("=" * 76)

    total = 0

    # 1. CAM2_SCENE003.mp4: Actor ponytail and dark hair clump (frames 38 to 58)
    cam2_s003 = ROOT / "samples" / "NEW_STAGED_CAM-02" / "NEW_CAMERA 02" / "CAM2_SCENE003.mp4"
    specs_cam2_s003 = []
    for fn in range(38, 58):
        specs_cam2_s003.append({
            "frame": fn,
            "crops": [
                ([335, 120, 395, 225], "ponytail_hair"),
                ([320, 80, 480, 240], "head_and_hair"),
            ]
        })
    c1 = extract_crops(cam2_s003, specs_cam2_s003, "cam2_s003")
    print(f"[*] Extracted {c1} ponytail and head crops from CAM2_SCENE003.mp4")
    total += c1

    # 2. CAM1_SCENE003.mov: Other angle head/hair (frames 38 to 55)
    cam1_s003 = ROOT / "samples" / "NEW_STAGED_CAM-01" / "NEW_CAMERA 01" / "CAM1_SCENE003.mov"
    specs_cam1_s003 = []
    for fn in range(38, 55, 2):
        specs_cam1_s003.append({
            "frame": fn,
            "crops": [
                ([380, 100, 520, 260], "head_profile_hair"),
            ]
        })
    c2 = extract_crops(cam1_s003, specs_cam1_s003, "cam1_s003")
    print(f"[*] Extracted {c2} head crops from CAM1_SCENE003.mov")
    total += c2

    # 3. CAM1_SCENE001.mov: Door shadows and edge contrasts (frames 15 to 30)
    cam1_s001 = ROOT / "samples" / "NEW_STAGED_CAM-01" / "NEW_CAMERA 01" / "CAM1_SCENE001.mov"
    specs_cam1_s001 = []
    for fn in range(15, 30, 2):
        specs_cam1_s001.append({
            "frame": fn,
            "crops": [
                ([500, 400, 700, 600], "door_shadow_crease"),
            ]
        })
    c3 = extract_crops(cam1_s001, specs_cam1_s001, "cam1_s001")
    print(f"[*] Extracted {c3} shadow crops from CAM1_SCENE001.mov")
    total += c3

    print(f"\n[DONE] Successfully extracted {total} Candidate V10 targeted hard negatives!")

if __name__ == "__main__":
    main()
