"""
Verify Model 10 (best_weapon_detector_tenth_model.pth) on:
  1. All 5 unique Non-Weapon Surveillance Videos (samples/full_run videos_with_screen_recording/720 w-o knife/*.mp4)
     to confirm 0 false weapon hallucinations on non-weapon CCTV footage.
  2. The 4 Official Weapon Surveillance Benchmark Videos:
     - samples/handgun_test-video.mp4
     - samples/evaluation_video.mp4
     - samples/NEW_KNIFE_VIDEO_11s.mp4
     - samples/CAM02_Scene_004.mp4
"""
import csv
import json
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from run_full_pipeline import detect_video_with_model

MODEL_PATH = str(ROOT_DIR / "best_weapon_detector_tenth_model.pth")
OUT_DIR = ROOT_DIR / "outputs" / "model_10_video_verification"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_confirmed_alerts_from_csv(video_stem: str):
    csv_path = ROOT_DIR / "outputs" / "reports" / f"{video_stem}_detections.csv"
    confirmed = []
    class_counts = {"handgun": 0, "knife": 0}
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                status = row.get("validation_status", "")
                if status in ("CONFIRMED_ALERT", "VALIDATED_TEMPORAL"):
                    lbl = row.get("object_label", "")
                    conf = float(row.get("confidence_score", 0.0))
                    confirmed.append((lbl, conf))
                    if lbl in class_counts:
                        class_counts[lbl] += 1
    avg_conf = round(sum(c for _, c in confirmed) / len(confirmed) * 100.0, 2) if confirmed else 0.0
    return {
        "confirmed_alerts": len(confirmed),
        "class_counts": class_counts,
        "average_confidence_pct": avg_conf,
    }


def main():
    no_weapon_dir = ROOT_DIR / "samples" / "full_run videos_with_screen_recording" / "720 w-o knife"
    unique_no_weapon = [
        no_weapon_dir / "1869451F17C5_20260920155118064_cam0.mp4",
        no_weapon_dir / "1869451F17C5_20260920155218941_cam0.mp4",
        no_weapon_dir / "1869451F17C5_20260920155258445_cam0.mp4",
        no_weapon_dir / "1869451F17C5_20260920155440307_cam0.mp4",
        no_weapon_dir / "1869451F17C5_20260920155503889_cam0.mp4",
    ]
    weapon_benchmarks = [
        ROOT_DIR / "samples" / "handgun_test-video.mp4",
        ROOT_DIR / "samples" / "evaluation_video.mp4",
        ROOT_DIR / "samples" / "NEW_KNIFE_VIDEO_11s.mp4",
        ROOT_DIR / "samples" / "CAM02_Scene_004.mp4",
    ]

    summary = {"non_weapon_surveillance_videos": {}, "official_4_weapon_videos": {}}

    print("=" * 95)
    print("  MODEL 10 SURVEILLANCE VIDEO HALLUCINATION & WEAPON BENCHMARK VERIFICATION")
    print("=" * 95)

    for v in unique_no_weapon:
        out_mp4 = OUT_DIR / f"m10_{v.name}"
        stats = detect_video_with_model(
            input_path=str(v),
            output_path=str(out_mp4),
            model_path=MODEL_PATH,
            confidence_threshold=0.50,
            enable_cctv_intelligence=True,
            enable_temporal_consistency=True,
        )
        parsed = parse_confirmed_alerts_from_csv(v.stem)
        summary["non_weapon_surveillance_videos"][v.name] = {
            "total_frames": stats.get("total_frames", 0),
            "analyzed_frames": stats.get("analyzed_frames", 0),
            **parsed,
        }
        print(
            f"[Non-Weapon Video] {v.name:<45} | Analyzed: {stats.get('analyzed_frames', 0):<4} | "
            f"Confirmed Alerts: {parsed['confirmed_alerts']:<3} | Classes: {parsed['class_counts']}"
        )

    for v in weapon_benchmarks:
        if not v.exists():
            continue
        out_mp4 = OUT_DIR / f"m10_{v.name}"
        stats = detect_video_with_model(
            input_path=str(v),
            output_path=str(out_mp4),
            model_path=MODEL_PATH,
            confidence_threshold=0.50,
            enable_cctv_intelligence=True,
            enable_temporal_consistency=True,
        )
        parsed = parse_confirmed_alerts_from_csv(v.stem)
        summary["official_4_weapon_videos"][v.name] = {
            "total_frames": stats.get("total_frames", 0),
            "analyzed_frames": stats.get("analyzed_frames", 0),
            **parsed,
        }
        print(
            f"[Weapon Benchmark] {v.name:<45} | Analyzed: {stats.get('analyzed_frames', 0):<4} | "
            f"Confirmed Alerts: {parsed['confirmed_alerts']:<3} | Classes: {parsed['class_counts']} | "
            f"Avg Conf: {parsed['average_confidence_pct']}%"
        )

    json_out = OUT_DIR / "model_10_video_verification_summary.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[*] Saved verification JSON to {json_out}")


if __name__ == "__main__":
    main()
