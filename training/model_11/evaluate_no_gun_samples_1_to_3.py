"""
Evaluate 'no gun sample 1.mp4', 'no gun sample 2.mp4', and 'no gun sample 3.mp4'
across Model 9, Model 10, and Model 11 at a 20% confidence threshold (0.20).
Saves all annotated videos, CSVs, and forensic reports into a dedicated folder:
  outputs/no_gun_samples_1_to_3_benchmark/
and audits both total confirmed alarms (>=20%) and high-confidence alarms (>=50%, >=70%, >=85%).
"""
import json
import os
import shutil
import sys
from pathlib import Path
import pandas as pd

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import torch
torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_full_pipeline import detect_video_with_model

OUT_DIR = ROOT / "outputs" / "no_gun_samples_1_to_3_benchmark"
FLAT_VID_DIR = OUT_DIR / "all_9_annotated_videos"
DOWNLOADS = Path("C:/Users/pc/Downloads")

MODELS = [
    ("Model 9", "model_9", ROOT / "best_weapon_detector_ninth_model.pth"),
    ("Model 10", "model_10", ROOT / "best_weapon_detector_tenth_model.pth"),
    ("Model 11", "model_11", ROOT / "best_weapon_detector_eleventh_model.pth"),
]

VIDEOS = [
    ("No Gun Sample 1", "no_gun_sample_1", DOWNLOADS / "no gun sample 1.mp4"),
    ("No Gun Sample 2", "no_gun_sample_2", DOWNLOADS / "no gun sample 2.mp4"),
    ("No Gun Sample 3", "no_gun_sample_3", DOWNLOADS / "no gun sample 3.mp4"),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FLAT_VID_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for m_name, m_tag, m_path in MODELS:
        for v_label, v_tag, v_path in VIDEOS:
            vid_out_dir = OUT_DIR / m_tag / v_tag
            vid_out_dir.mkdir(parents=True, exist_ok=True)
            out_mp4 = vid_out_dir / f"{m_tag}_{v_tag}_20pct_annotated.mp4"
            out_csv = vid_out_dir / f"{m_tag}_{v_tag}_20pct_detections.csv"

            print(f"\n[*] Running {m_name} on {v_label} at 20% threshold...", flush=True)
            details = detect_video_with_model(
                input_path=v_path,
                output_path=out_mp4,
                model_path=m_path,
                confidence_threshold=0.20,
                analysis_fps=15,
                save_detected_frames=False,
                csv_output_path=out_csv,
                enable_cctv_intelligence=True,
                enable_temporal_consistency=True,
                camera_id="CAM-NOGUN",
            )

            # Link/copy into flat folder for one-click comparison
            flat_dst = FLAT_VID_DIR / f"{v_tag}__{m_tag}_20pct_annotated.mp4"
            if flat_dst.exists():
                flat_dst.unlink()
            try:
                os.link(out_mp4, flat_dst)
            except Exception:
                shutil.copy2(out_mp4, flat_dst)

            df = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
            if len(df) > 0:
                conf_df = df[df["validation_status"].isin(["CONFIRMED_ALERT", "VALIDATED_TEMPORAL"])]
                conf_df = conf_df[conf_df["rejection_reason"].fillna("").astype(str).str.strip() == ""]
                hg_alerts = int((conf_df["object_label"] == "handgun").sum())
                kn_alerts = int((conf_df["object_label"] == "knife").sum())
                total_alerts = len(conf_df)
                high_50 = int((conf_df["confidence_score"] >= 0.50).sum())
                high_70 = int((conf_df["confidence_score"] >= 0.70).sum())
                high_85 = int((conf_df["confidence_score"] >= 0.85).sum())
                max_conf = round(float(conf_df["confidence_score"].max() * 100.0), 2) if total_alerts > 0 else 0.0
                avg_conf = round(float(conf_df["confidence_score"].mean() * 100.0), 2) if total_alerts > 0 else 0.0
                # Also check raw unconfirmed high-confidence proposals (>=70%)
                raw_high_70 = int((df["confidence_score"] >= 0.70).sum())
                raw_total = len(df)
                top_alarms = (
                    conf_df.sort_values("confidence_score", ascending=False)
                    .head(5)[["frame_number", "object_label", "confidence_score", "bounding_box"]]
                    .to_dict(orient="records")
                    if total_alerts > 0
                    else []
                )
            else:
                hg_alerts = kn_alerts = total_alerts = high_50 = high_70 = high_85 = raw_high_70 = raw_total = 0
                max_conf = avg_conf = 0.0
                top_alarms = []

            summary_rows.append({
                "model": m_name,
                "video": v_label,
                "total_frames": details.get("total_frames", 0),
                "analyzed_frames": details.get("analyzed_frames", 0),
                "confirmed_alarms_20pct": total_alerts,
                "handgun_alarms": hg_alerts,
                "knife_alarms": kn_alerts,
                "high_conf_50pct": high_50,
                "high_conf_70pct": high_70,
                "high_conf_85pct": high_85,
                "max_conf_pct": max_conf,
                "avg_conf_pct": avg_conf,
                "raw_candidates_total": raw_total,
                "raw_candidates_70pct": raw_high_70,
                "top_confirmed_alarms": top_alarms,
                "annotated_video": str(out_mp4),
            })

    df_out = pd.DataFrame(summary_rows)
    df_out.to_csv(OUT_DIR / "no_gun_samples_20pct_summary.csv", index=False)
    (OUT_DIR / "no_gun_samples_20pct_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    print("\n=========================================================================================================")
    print("  NO-GUN SAMPLES 1-3 AUDIT AT 20% THRESHOLD (MODEL 9 vs MODEL 10 vs MODEL 11)")
    print("=========================================================================================================")
    for r in summary_rows:
        print(
            f"  {r['model']:9s} | {r['video']:16s} | Confirmed (>=20%): {r['confirmed_alarms_20pct']:3d} "
            f"(HG={r['handgun_alarms']:2d}, KN={r['knife_alarms']:2d}) | "
            f">=50%: {r['high_conf_50pct']:2d} | >=70%: {r['high_conf_70pct']:2d} | >=85%: {r['high_conf_85pct']:2d} | "
            f"MaxConf: {r['max_conf_pct']:5.1f}% | RawCand: {r['raw_candidates_total']:3d} (Raw>=70%: {r['raw_candidates_70pct']:3d})"
        )
        if r["top_confirmed_alarms"]:
            print(f"      Top Confirmed Alarms: {r['top_confirmed_alarms']}")


if __name__ == "__main__":
    main()

