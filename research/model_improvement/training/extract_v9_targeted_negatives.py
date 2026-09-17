import os
import sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[3]
NEG_DIR = ROOT / "research" / "model_improvement" / "dataset" / "hard_negatives_v9"
NEG_DIR.mkdir(parents=True, exist_ok=True)

def extract_crops_from_video(video_path: Path, frame_targets: list, category_prefix: str):
    """Extracts bounding box crops and full frames for hard negative training."""
    if not video_path.exists():
        print(f"[Warning] Video not found: {video_path}")
        return 0

    cap = cv2.VideoCapture(str(video_path))
    extracted = 0
    cur_fn = 0

    # Build lookup map by frame_idx
    target_map = {}
    for item in frame_targets:
        target_map[item["frame"]] = item

    max_target_frame = max(target_map.keys()) if target_map else 0

    while True:
        ret, frame = cap.read()
        if not ret or cur_fn > max_target_frame + 5:
            break

        if cur_fn in target_map:
            spec = target_map[cur_fn]
            h, w = frame.shape[:2]

            # 1. Save specific crop if crop coordinates provided
            if "crop" in spec:
                x1, y1, x2, y2 = spec["crop"]
                x1 = max(0, min(w - 1, int(x1)))
                y1 = max(0, min(h - 1, int(y1)))
                x2 = max(x1 + 10, min(w, int(x2)))
                y2 = max(y1 + 10, min(h, int(y2)))
                crop_img = frame[y1:y2, x1:x2]
                
                crop_fname = f"{category_prefix}_f{cur_fn:04d}_{spec.get('name', 'crop')}.jpg"
                cv2.imwrite(str(NEG_DIR / crop_fname), crop_img)
                extracted += 1

            # 2. Save full frame as negative if weapon is absent or occluded
            if spec.get("save_full_frame", False):
                full_fname = f"full_{category_prefix}_f{cur_fn:04d}.jpg"
                cv2.imwrite(str(NEG_DIR / full_fname), frame)
                extracted += 1

        cur_fn += 1

    cap.release()
    return extracted

def main():
    print("=" * 80)
    print("EXTRACTING CANDIDATE V9 TARGETED HARD NEGATIVES")
    print(f"Target Directory: {NEG_DIR}")
    print("=" * 80)

    total_extracted = 0

    # Video 1: 213340 (Masked face, skeleton gloves, floor mat neon border)
    vid_213340 = ROOT / "samples" / "Screen Recording 2026-09-10 213340 -HANDGUN - Trim.mp4"
    if not vid_213340.exists():
        vid_213340 = ROOT / "samples" / "NEW-VIDEOS" / "NEW HANDGUNS VIDEO CLIPS" / "Screen Recording 2026-09-10 213340 -HANDGUN.mp4"

    # Targets for 213340:
    # Frame 195-200: Head mask with beanie and sunglasses: [330, 250, 480, 420]
    # Frame 188-194: Skeleton gloves: [780, 520, 930, 680]
    # Frame 208-216: Floor mat neon strip: [950, 420, 1200, 820]
    targets_213340 = []
    # Head & mask
    for fn in range(193, 202):
        targets_213340.append({
            "frame": fn,
            "crop": [310, 240, 500, 440],
            "name": "masked_face_beanie",
            "save_full_frame": (fn in (195, 198)),
        })
    # Skeleton gloves
    for fn in range(188, 195):
        targets_213340.append({
            "frame": fn,
            "crop": [770, 510, 940, 690],
            "name": "skeleton_glove",
            "save_full_frame": (fn == 191),
        })
    # Floor mat neon strip
    for fn in range(208, 218):
        targets_213340.append({
            "frame": fn,
            "crop": [940, 410, 1220, 830],
            "name": "floor_mat_strip",
            "save_full_frame": (fn in (211, 215)),
        })

    cnt1 = extract_crops_from_video(vid_213340, targets_213340, "robbery_213340")
    print(f"[*] Video 213340: Extracted {cnt1} hard negatives (masked face, gloves, floor mat).")
    total_extracted += cnt1

    # Video 2: 231205 (Wall switch plate on light wall)
    vid_231205 = ROOT / "samples" / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS" / "Screen Recording 2026-09-07 231205.mp4"
    targets_231205 = []
    for fn in range(15, 28):
        targets_231205.append({
            "frame": fn,
            "crop": [130, 480, 230, 580], # Wall light switch coordinates
            "name": "wall_light_switch",
            "save_full_frame": (fn in (18, 22)),
        })

    cnt2 = extract_crops_from_video(vid_231205, targets_231205, "switch_231205")
    print(f"[*] Video 231205: Extracted {cnt2} hard negatives (wall light switch plate).")
    total_extracted += cnt2

    # Video 3: 230942 (Packaging cardboard on retail counter)
    vid_230942 = ROOT / "samples" / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS" / "Screen Recording 2026-09-07 230942.mp4"
    targets_230942 = []
    for fn in range(92, 102):
        targets_230942.append({
            "frame": fn,
            "crop": [400, 300, 680, 520], # Counter packaging
            "name": "counter_packaging",
            "save_full_frame": (fn in (94, 98)),
        })

    cnt3 = extract_crops_from_video(vid_230942, targets_230942, "packaging_230942")
    print(f"[*] Video 230942: Extracted {cnt3} hard negatives (counter packaging).")
    total_extracted += cnt3

    print(f"\n[Finished] Total Candidate V9 Hard Negatives Extracted: {total_extracted}")
    print(f"Files saved in: {NEG_DIR}")

if __name__ == "__main__":
    main()
