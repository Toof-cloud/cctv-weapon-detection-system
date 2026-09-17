import os
import sys
import time
from pathlib import Path

# Thread guard
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_full_pipeline import detect_video_with_model, generate_forensic_summary

MODEL_PATH = ROOT / "best_weapon_detector_ninth_model.pth"
INPUT_DIR = ROOT / "samples" / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS"
OUTPUT_BASE = ROOT / "outputs" / "unseen_jabez_model9_evaluation"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

def run_unseen_benchmark():
    videos = sorted(list(INPUT_DIR.glob("*.mp4")))
    print("=" * 80)
    print("STARTING UNSEEN CCTV EVALUATION (MODEL 9 + SURVEILLANCE INTELLIGENCE LAYER)")
    print(f"Model Checkpoint: {MODEL_PATH}")
    print(f"Total Unseen Clips: {len(videos)}")
    print(f"Output Directory: {OUTPUT_BASE}")
    print("=" * 80)

    results = []

    for idx, vpath in enumerate(videos, start=1):
        clean_name = f"clip_{idx:02d}_{vpath.stem.replace(' ', '_').replace('-', '_')}"
        clip_out_dir = OUTPUT_BASE / clean_name
        clip_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{idx}/{len(videos)}] Processing Unseen Video: {vpath.name}")
        t0 = time.time()

        det_res = detect_video_with_model(
            input_path=vpath,
            output_path=clip_out_dir / "annotated_output.mp4",
            model_path=MODEL_PATH,
            confidence_threshold=0.50,
            analysis_fps=10,
            save_detected_frames=True,
            csv_output_path=clip_out_dir / "forensic_detections.csv",
            enable_cctv_intelligence=True,
            enable_temporal_consistency=True,
            camera_id=f"CAM-{idx:02d}",
        )

        summary_txt = clip_out_dir / "forensic_summary.txt"
        generate_forensic_summary(Path(det_res["csv_report"]), summary_txt)
        elapsed = time.time() - t0

        summary_content = summary_txt.read_text(encoding="utf-8") if summary_txt.exists() else ""
        confirmed_count = det_res.get("confirmed_threats", 0)

        results.append({
            "index": idx,
            "filename": vpath.name,
            "clean_name": clean_name,
            "elapsed_sec": round(elapsed, 1),
            "summary": summary_content,
            "out_dir": clip_out_dir,
        })
        print(f"  -> Finished in {elapsed:.1f}s. Confirmed Alerts: {confirmed_count}")

    print("\n" + "=" * 80)
    print("ALL 10 UNSEEN VIDEOS EVALUATION COMPLETED WITH MODEL 9!")
    print("=" * 80)

if __name__ == "__main__":
    run_unseen_benchmark()
