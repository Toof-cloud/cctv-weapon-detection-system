import os
import sys
import glob
import time
from pathlib import Path

# Set up project path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.video_enhancement_service import VideoEnhancementService

INPUT_DIR = ROOT_DIR / "samples" / "To_enhance"
OUTPUT_DIR = ROOT_DIR / "samples" / "enhanced_videos"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Supported video extensions
    video_extensions = ("*.mp4", "*.mov", "*.avi", "*.mkv")
    video_files = []
    for ext in video_extensions:
        video_files.extend(INPUT_DIR.glob(ext))
    
    video_files = sorted(list(set(video_files)))
    total_videos = len(video_files)
    
    if total_videos == 0:
        print(f"No video files found in {INPUT_DIR}")
        return

    print(f"==================================================")
    print(f"Starting Batch BasicVSR++ Enhancement")
    print(f"Input Directory : {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Total Videos    : {total_videos}")
    print(f"VRAM Safety Cap : ~4.5 GB peak (chunk_size=5, 960px)")
    print(f"==================================================")

    service = VideoEnhancementService()
    if not service.verify_wsl_access():
        print("ERROR: Could not verify WSL access. Please ensure WSL is running.")
        return

    start_all = time.time()
    for idx, vid_path in enumerate(video_files, 1):
        # Determine output filename: keep format mp4 for compatibility
        out_name = vid_path.stem + "_enhanced.mp4"
        out_path = OUTPUT_DIR / out_name

        if out_path.exists() and out_path.stat().st_size > 1000:
            print(f"[{idx}/{total_videos}] Skipping already enhanced: {out_name}")
            continue

        print(f"\n--------------------------------------------------")
        print(f"[{idx}/{total_videos}] Enhancing: {vid_path.name}")
        print(f"Destination: {out_name}")
        print(f"--------------------------------------------------")
        
        t0 = time.time()
        try:
            service.enhance_video(str(vid_path), str(out_path))
            elapsed = time.time() - t0
            print(f"[{idx}/{total_videos}] Finished {vid_path.name} in {elapsed:.1f}s ({elapsed/60:.2f} min)")
        except Exception as e:
            print(f"[{idx}/{total_videos}] FAILED on {vid_path.name}: {e}")

    total_elapsed = time.time() - start_all
    print(f"\n==================================================")
    print(f"Batch Video Enhancement Completed!")
    print(f"Total Time: {total_elapsed:.1f}s ({total_elapsed/60:.2f} minutes)")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"==================================================")

if __name__ == "__main__":
    main()
