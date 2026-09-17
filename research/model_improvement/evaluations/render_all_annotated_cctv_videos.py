"""
Batch Annotated Video & Frame Renderer for Candidate Model V10
Runs Candidate Model V10 + Tier-2 CCTV Intelligence Layer across all CCTV
footage in the samples directory and generates:
1. Annotated MP4 video with bounding boxes and forensic timestamps
2. Extracted detected PNG frames
3. CSV detection logs
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Thread guard
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from run_full_pipeline import detect_video_with_model

MODEL_PATH = ROOT_DIR / "research" / "model_improvement" / "checkpoints" / "best_candidate_model_v10.pth"
OUTPUT_BASE = ROOT_DIR / "outputs" / "v10_annotated_cctv_videos"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

TARGET_VIDEOS = [
    # Category 1: Root Sample CCTV Videos
    {
        "id": "root_01_handgun_test",
        "path": ROOT_DIR / "samples" / "handgun_test-video.mp4",
        "camera_id": "STORE-CAM01",
        "desc": "Handgun CCTV Test Video (Night Store Robbery)",
    },
    {
        "id": "root_02_new_knife_11s",
        "path": ROOT_DIR / "samples" / "NEW_KNIFE_VIDEO_11s.mp4",
        "camera_id": "KITCHEN-CAM01",
        "desc": "1080p CCTV Kitchen Knife Brandishing",
    },
    {
        "id": "root_03_evaluation_knife",
        "path": ROOT_DIR / "samples" / "evaluation_video.mp4",
        "camera_id": "OUTDOOR-CAM01",
        "desc": "Outdoor Knife Assault Confrontation",
    },
    {
        "id": "root_04_test_10s_knife",
        "path": ROOT_DIR / "samples" / "test_10s_knife.mp4",
        "camera_id": "CORRIDOR-CAM01",
        "desc": "CCTV Hallway Knife Threat",
    },
    {
        "id": "root_05_test_10s_general",
        "path": ROOT_DIR / "samples" / "test_10s_general.mp4",
        "camera_id": "PUBLIC-CAM01",
        "desc": "General Public Walkway (Negative Clean Test)",
    },
    {
        "id": "root_06_test_10s_negative",
        "path": ROOT_DIR / "samples" / "test_10s_negative.mp4",
        "camera_id": "ACTOR-CAM01",
        "desc": "Pure Negative Walking Actor Stress Test",
    },
    {
        "id": "root_07_handgun_trim",
        "path": ROOT_DIR / "samples" / "Screen Recording 2026-09-10 213340 -HANDGUN - Trim.mp4",
        "camera_id": "COUNTER-CAM01",
        "desc": "1080p Cashier Robbery Handgun Draw",
    },
    {
        "id": "root_08_knife_long",
        "path": ROOT_DIR / "samples" / "knife_test_long-video.mp4",
        "camera_id": "DEMO-CAM01",
        "desc": "Knife Demonstration Long Footage",
    },
    # Category 2: 10 Real Unseen CCTV Surveillance Clips (Jabez collection)
    {
        "id": "unseen_clip01_store_robbery",
        "path": ROOT_DIR / "samples" / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS" / "Screen Recording 2026-09-07 224743.mp4",
        "camera_id": "CCTV-AISLE-01",
        "desc": "Unseen Real CCTV: Night Store Robbery (Handgun)",
    },
    {
        "id": "unseen_clip02_counter_assault",
        "path": ROOT_DIR / "samples" / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS" / "Screen Recording 2026-09-07 225138.mp4",
        "camera_id": "CCTV-COUNTER-02",
        "desc": "Unseen Real CCTV: Counter Assault (Knife)",
    },
    {
        "id": "unseen_clip03_street_robbery",
        "path": ROOT_DIR / "samples" / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS" / "Screen Recording 2026-09-07 225703.mp4",
        "camera_id": "CCTV-STREET-03",
        "desc": "Unseen Real CCTV: Street Robbery Confrontation (Handgun)",
    },
    {
        "id": "unseen_clip04_night_draw",
        "path": ROOT_DIR / "samples" / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS" / "Screen Recording 2026-09-07 230047.mp4",
        "camera_id": "CCTV-NIGHT-04",
        "desc": "Unseen Real CCTV: Night Street Gun Draw (Handgun)",
    },
    {
        "id": "unseen_clip05_customer_draw",
        "path": ROOT_DIR / "samples" / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS" / "Screen Recording 2026-09-07 230249.mp4",
        "camera_id": "CCTV-STORE-05",
        "desc": "Unseen Real CCTV: Store Customer Draw (Handgun)",
    },
    {
        "id": "unseen_clip06_store_glass",
        "path": ROOT_DIR / "samples" / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS" / "Screen Recording 2026-09-07 230541.mp4",
        "camera_id": "CCTV-GLASS-06",
        "desc": "Unseen Real CCTV: Store Robbery through Glass (Baseline)",
    },
    {
        "id": "unseen_clip07_fast_blade",
        "path": ROOT_DIR / "samples" / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS" / "Screen Recording 2026-09-07 230842.mp4",
        "camera_id": "CCTV-ALLEY-07",
        "desc": "Unseen Real CCTV: Fast Blade Draw (Knife)",
    },
    {
        "id": "unseen_clip08_corridor_threat",
        "path": ROOT_DIR / "samples" / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS" / "Screen Recording 2026-09-07 230942.mp4",
        "camera_id": "CCTV-HALL-08",
        "desc": "Unseen Real CCTV: Corridor Threat (Handgun)",
    },
    {
        "id": "unseen_clip09_multi_person",
        "path": ROOT_DIR / "samples" / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS" / "Screen Recording 2026-09-07 231101.mp4",
        "camera_id": "CCTV-POS-09",
        "desc": "Unseen Real CCTV: Multi-Person Distractor Robbery (Handgun)",
    },
    {
        "id": "unseen_clip10_bank_counter",
        "path": ROOT_DIR / "samples" / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS" / "Screen Recording 2026-09-07 231205.mp4",
        "camera_id": "CCTV-BANK-10",
        "desc": "Unseen Real CCTV: Bank Counter Robbery (Handgun)",
    },
    # Category 3: Staged Multi-Camera Benchmark Scenes
    {
        "id": "staged_cam1_scene004_knife",
        "path": ROOT_DIR / "samples" / "NEW_STAGED_CAM-01" / "NEW_CAMERA 01" / "CAM1_SCENE004.mov",
        "camera_id": "STAGED-CAM01",
        "desc": "Staged CAM-01 Scene 004 Knife Brandishing",
    },
    {
        "id": "staged_cam2_scene003_ponytail_trap",
        "path": ROOT_DIR / "samples" / "NEW_STAGED_CAM-02" / "NEW_CAMERA 02" / "CAM2_SCENE003.mp4",
        "camera_id": "STAGED-CAM02",
        "desc": "Staged CAM-02 Scene 003 Handgun + Ponytail Distractor Trap",
    },
    {
        "id": "staged_cam1_scene011_stair_post_trap",
        "path": ROOT_DIR / "samples" / "NEW_STAGED_CAM-01" / "NEW_CAMERA 01" / "CAM1_SCENE011.mov",
        "camera_id": "STAGED-CAM01",
        "desc": "Staged CAM-01 Scene 011 Firearm + Stair Railing Post Trap",
    },
    {
        "id": "staged_cam2_scene009_negative_actor",
        "path": ROOT_DIR / "samples" / "NEW_STAGED_CAM-02" / "NEW_CAMERA 02" / "CAM2_SCENE009.mp4",
        "camera_id": "STAGED-CAM02",
        "desc": "Staged CAM-02 Scene 009 Negative Walking Actor Test",
    },
]


def render_all_videos():
    print("=" * 85)
    print("CANDIDATE MODEL V10: FULL BATCH ANNOTATED VIDEO & DETECTED FRAME GENERATION")
    print(f"Model Checkpoint: {MODEL_PATH}")
    print(f"Output Directory: {OUTPUT_BASE}")
    print("Filters Active: Tier-2 CCTV Intelligence (Person Gate + Aspect Ratio + Temporal)")
    print("=" * 85)

    valid_targets = [t for t in TARGET_VIDEOS if t["path"].is_file()]
    print(f"Found {len(valid_targets)} valid CCTV videos to process.\n")

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
                analysis_fps=5,
                save_detected_frames=True,
                csv_output_path=csv_out,
                enable_cctv_intelligence=True,
                enable_temporal_consistency=True,
                camera_id=cam_id,
            )
            elapsed = time.time() - t0

            # Count saved detected frames
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
    print("ALL ANNOTATED VIDEOS & DETECTED FRAMES SUCCESSFULLY RENDERED!")
    print(f"Total Videos Processed: {len(completed_reports)}")
    print(f"Total Elapsed Time: {total_time:.2f}s")
    print(f"Master Output Folder: {OUTPUT_BASE}")
    print("=" * 85)


if __name__ == "__main__":
    render_all_videos()
