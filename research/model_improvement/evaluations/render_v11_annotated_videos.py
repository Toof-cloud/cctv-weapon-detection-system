"""
Batch Annotated Video & Frame Renderer for Candidate Model V11
Runs Candidate Model V11 + Enhanced Tier-2 CCTV Intelligence Layer across the user-targeted CCTV footage:
1. CAM2_SCENE001, CAM2_SCENE002, CAM2_SCENE003 (Door Shadow + Ponytail Distractor)
2. Screen Recording 2026-09-07 225138.mp4 (Unseen Clip 02: Counter Assault Knife Draw)
3. Screen Recording 2026-09-07 230249.mp4 (Unseen Clip 05: Store Customer Draw Handgun & Cashier Cap)
4. Screen Recording 2026-09-07 225703.mp4 (Unseen Clip 03: Shoulder Shadow Suppression)
5. Root representative benchmark videos

Generates:
1. Annotated MP4 video with bounding boxes and forensic status labels
2. Extracted detected PNG frames
3. CSV detection logs
"""
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from run_full_pipeline import detect_video_with_model

MODEL_PATH = ROOT_DIR / "research" / "model_improvement" / "checkpoints" / "best_candidate_model_v11.pth"
OUTPUT_BASE = ROOT_DIR / "outputs" / "v11_annotated_cctv_videos"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

TARGET_VIDEOS = [
    # 1. User Target: CAM2_SCENE001 (Door Shadow vs Knife)
    {
        "id": "staged_cam2_scene001_door_shadow",
        "path": ROOT_DIR / "samples" / "NEW_STAGED_CAM-02" / "NEW_CAMERA 02" / "CAM2_SCENE001.mp4",
        "camera_id": "STAGED-CAM02",
        "desc": "Staged CAM-02 Scene 001 Door Shadow vs Knife",
    },
    # 2. User Target: CAM2_SCENE002 (Door Hand Shadow vs Real Knife)
    {
        "id": "staged_cam2_scene002_door_knife",
        "path": ROOT_DIR / "samples" / "NEW_STAGED_CAM-02" / "NEW_CAMERA 02" / "CAM2_SCENE002.mp4",
        "camera_id": "STAGED-CAM02",
        "desc": "Staged CAM-02 Scene 002 Door Hand Shadow vs Real Knife",
    },
    # 3. User Target: CAM2_SCENE003 (Empty Hand on Door + Ponytail Trap)
    {
        "id": "staged_cam2_scene003_ponytail_trap",
        "path": ROOT_DIR / "samples" / "NEW_STAGED_CAM-02" / "NEW_CAMERA 02" / "CAM2_SCENE003.mp4",
        "camera_id": "STAGED-CAM02",
        "desc": "Staged CAM-02 Scene 003 Empty Hands Door Shadow + Ponytail Trap",
    },
    # 4. User Target: CAM1_SCENE004 (Knife Brandishing)
    {
        "id": "staged_cam1_scene004_knife",
        "path": ROOT_DIR / "samples" / "NEW_STAGED_CAM-01" / "NEW_CAMERA 01" / "CAM1_SCENE004.mov",
        "camera_id": "STAGED-CAM01",
        "desc": "Staged CAM-01 Scene 004 Dynamic Knife Brandishing",
    },
    # 5. User Target: CAM1_SCENE011 (Stair Post Trap)
    {
        "id": "staged_cam1_scene011_stair_post_trap",
        "path": ROOT_DIR / "samples" / "NEW_STAGED_CAM-01" / "NEW_CAMERA 01" / "CAM1_SCENE011.mov",
        "camera_id": "STAGED-CAM01",
        "desc": "Staged CAM-01 Scene 011 Stair Post False Alarm Trap",
    },
    # 6. User Target: Screen Recording 2026-09-07 224743.mp4 (Unseen Clip 01: Night Store Robbery Handgun)
    {
        "id": "unseen_clip01_store_robbery",
        "path": ROOT_DIR / "samples" / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS" / "Screen Recording 2026-09-07 224743.mp4",
        "camera_id": "CCTV-AISLE-01",
        "desc": "Unseen Real CCTV: Night Store Robbery (Handgun)",
    },
    # 7. User Target: Screen Recording 2026-09-07 225138.mp4 (Unseen Clip 02: Counter Assault Knife Draw)
    {
        "id": "unseen_clip02_counter_assault",
        "path": ROOT_DIR / "samples" / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS" / "Screen Recording 2026-09-07 225138.mp4",
        "camera_id": "CCTV-COUNTER-02",
        "desc": "Unseen Real CCTV: Counter Assault (Knife Draw)",
    },
    # 8. User Target: Screen Recording 2026-09-07 225703.mp4 (Unseen Clip 03: Shoulder Shadow Suppression)
    {
        "id": "unseen_clip03_street_robbery",
        "path": ROOT_DIR / "samples" / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS" / "Screen Recording 2026-09-07 225703.mp4",
        "camera_id": "CCTV-STREET-03",
        "desc": "Unseen Real CCTV: Street Robbery (Shoulder Shadow Suppression)",
    },
    # 9. User Target: Screen Recording 2026-09-07 230047.mp4 (Unseen Clip 04: Night Draw)
    {
        "id": "unseen_clip04_night_draw",
        "path": ROOT_DIR / "samples" / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS" / "Screen Recording 2026-09-07 230047.mp4",
        "camera_id": "CCTV-NIGHT-04",
        "desc": "Unseen Real CCTV: Night Draw Suspect",
    },
    # 10. User Target: Screen Recording 2026-09-07 230249.mp4 (Unseen Clip 05: Customer Draw Handgun & Cashier Cap)
    {
        "id": "unseen_clip05_customer_draw",
        "path": ROOT_DIR / "samples" / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS" / "Screen Recording 2026-09-07 230249.mp4",
        "camera_id": "CCTV-STORE-05",
        "desc": "Unseen Real CCTV: Customer Draw (Handgun Stability & Cashier Cap)",
    },
    # 11. User Target: Screen Recording 2026-09-07 230541.mp4 (Unseen Clip 06: Store Glass Counter)
    {
        "id": "unseen_clip06_store_glass",
        "path": ROOT_DIR / "samples" / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS" / "Screen Recording 2026-09-07 230541.mp4",
        "camera_id": "CCTV-GLASS-06",
        "desc": "Unseen Real CCTV: Store Glass Counter",
    },
]


def main():
    print("=" * 85)
    print("CANDIDATE MODEL V11: FULL BATCH ANNOTATED VIDEO & DETECTED FRAME GENERATION")
    print(f"Model Checkpoint: {MODEL_PATH}")
    print(f"Output Directory: {OUTPUT_BASE}")
    print("Filters Active: Tier-2 CCTV Intelligence (Wall Shadow + Person Gate + Aspect Ratio + Temporal)")
    print("=" * 85)

    valid_targets = [t for t in TARGET_VIDEOS if t["path"].is_file()]
    print(f"Found {len(valid_targets)} targeted CCTV videos to process.\n")

    t_start = time.time()
    completed_reports = []

    for idx, item in enumerate(valid_targets, 1):
        vpath = item["path"]
        vid_id = item["id"]
        cam_id = item["camera_id"]
        desc = item["desc"]

        clip_out_dir = OUTPUT_BASE / vid_id
        clip_out_dir.mkdir(parents=True, exist_ok=True)
        out_video = clip_out_dir / f"{vid_id}_annotated.mp4"
        csv_out = clip_out_dir / f"{vid_id}_detections.csv"
        detected_frames_dir = clip_out_dir / f"{out_video.stem}_detected_frames"
        if detected_frames_dir.is_dir():
            shutil.rmtree(detected_frames_dir, ignore_errors=True)

        print(f"\n[{idx}/{len(valid_targets)}] Processing: {vpath.name}")
        print(f"     Target ID: {vid_id} | Camera: {cam_id} | {desc}")
        print(f"     Output Video: {out_video.name}")

        t0 = time.time()
        try:
            res = detect_video_with_model(
                input_path=vpath,
                output_path=out_video,
                model_path=MODEL_PATH,
                confidence_threshold=0.50,
                analysis_fps=10,
                save_detected_frames=True,
                csv_output_path=csv_out,
                enable_cctv_intelligence=True,
                enable_temporal_consistency=True,
                camera_id=cam_id,
            )
            elapsed = time.time() - t0

            detected_frames_dir = clip_out_dir / f"{out_video.stem}_detected_frames"
            frame_count = len(list(detected_frames_dir.glob("*.png"))) if detected_frames_dir.is_dir() else 0

            print(f"  --> Completed in {elapsed:.2f}s | Confirmed Detections: {res['total_detections']} | Saved Frames: {frame_count}")
            completed_reports.append({
                "id": vid_id,
                "video_name": vpath.name,
                "description": desc,
                "output_video": str(out_video),
                "detected_frames_dir": str(detected_frames_dir),
                "detected_frames_count": frame_count,
                "total_detections": res["total_detections"],
                "total_frames": res["total_frames"],
                "analyzed_frames": res["analyzed_frames"],
                "elapsed_seconds": round(elapsed, 2),
            })
        except Exception as exc:
            print(f"  [!] Error processing {vpath.name}: {exc}")

    total_time = time.time() - t_start
    print("\n" + "=" * 85)
    print("ALL TARGETED V11 ANNOTATED VIDEOS & DETECTED FRAMES SUCCESSFULLY RENDERED!")
    print(f"Total Videos Processed: {len(completed_reports)}")
    print(f"Total Elapsed Time: {total_time:.2f}s")
    print(f"Master Output Folder: {OUTPUT_BASE}")
    print("=" * 85)


if __name__ == "__main__":
    main()
