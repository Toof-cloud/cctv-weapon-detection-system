"""
Run Handgun 2 (CCTV 225703) across Models 9, 10, 11 and aggregate all 6-video results into JSON/CSV/Markdown.
"""
import json
import os
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

OUT_DIR = ROOT / "outputs" / "model_9_10_11_video_comparison"

MODELS = [
    ("Model 9", "model_9", ROOT / "best_weapon_detector_ninth_model.pth"),
    ("Model 10", "model_10", ROOT / "best_weapon_detector_tenth_model.pth"),
    ("Model 11", "model_11", ROOT / "best_weapon_detector_eleventh_model.pth"),
]

VIDEOS = [
    ("No-Gun 1 (Tapo CAM 01)", "No-Gun", "no-gun_1_tapo_cam_01", ROOT / "samples" / "5.mp4"),
    ("No-Gun 2 (Tapo CAM 02)", "No-Gun", "no-gun_2_tapo_cam_02", ROOT / "samples" / "5 (1).mp4"),
    ("Handgun 1 (CCTV 230249)", "Handgun", "handgun_1_cctv_230249", ROOT / "samples" / "Consultation for sir labanan" / "Handgun Enhanced" / "Screen Recording 2026-09-07 230249_enhanced.mp4"),
    ("Handgun 2 (CCTV 213340)", "Handgun", "handgun_2_cctv_213340", ROOT / "samples" / "Screen Recording 2026-09-10 213340 -HANDGUN - Trim.mp4"),
    ("Knife 1 (CCTV 230047)", "Knife", "knife_1_cctv_230047", ROOT / "samples" / "Consultation for sir labanan" / "Knife Enhanced" / "Screen Recording 2026-09-07 230047_enhanced.mp4"),
    ("Knife 2 (CCTV 220101)", "Knife", "knife_2_cctv_220101", ROOT / "samples" / "Consultation for sir labanan" / "Knife Enhanced" / "Screen Recording 2026-09-10 220101-KNIFE_enhanced.mp4"),
]


def main():
    summary_rows = []
    for m_name, m_tag, m_path in MODELS:
        for v_label, v_cat, v_tag, v_path in VIDEOS:
            vid_out_dir = OUT_DIR / m_tag / v_tag
            vid_out_dir.mkdir(parents=True, exist_ok=True)
            out_mp4 = vid_out_dir / f"{m_tag}_{v_tag}_annotated.mp4"
            out_csv = vid_out_dir / f"{m_tag}_{v_tag}_detections.csv"

            if not out_csv.exists():
                print(f"[*] Running {m_name} on {v_label} ({v_path.name})...", flush=True)
                detect_video_with_model(
                    input_path=v_path,
                    output_path=out_mp4,
                    model_path=m_path,
                    confidence_threshold=0.50,
                    analysis_fps=15,
                    save_detected_frames=False,
                    csv_output_path=out_csv,
                    enable_cctv_intelligence=True,
                    enable_temporal_consistency=True,
                    camera_id="CAM-EVAL",
                )

            df = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
            if len(df) > 0:
                conf_df = df[df["validation_status"].isin(["CONFIRMED_ALERT", "VALIDATED_TEMPORAL"])]
                conf_df = conf_df[conf_df["rejection_reason"].fillna("").astype(str).str.strip() == ""]
                hg_alerts = int((conf_df["object_label"] == "handgun").sum())
                kn_alerts = int((conf_df["object_label"] == "knife").sum())
                total_alerts = len(conf_df)
                avg_conf = round(float(conf_df["confidence_score"].mean() * 100.0), 2) if total_alerts > 0 else 0.0
                total_eval = len(df)
            else:
                hg_alerts, kn_alerts, total_alerts, avg_conf, total_eval = 0, 0, 0, 0.0, 0

            summary_rows.append({
                "model": m_name,
                "video_label": v_label,
                "category": v_cat,
                "confirmed_alerts": total_alerts,
                "handgun_alerts": hg_alerts,
                "knife_alerts": kn_alerts,
                "avg_confidence_pct": avg_conf,
                "total_candidate_proposals": total_eval,
                "annotated_video": str(out_mp4),
            })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "video_comparison_results.csv", index=False)
    (OUT_DIR / "video_comparison_results.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    print("\n==========================================================================================")
    print("  6-VIDEO HEAD-TO-HEAD BENCHMARK SUMMARY (MODEL 9 vs MODEL 10 vs MODEL 11)")
    print("==========================================================================================")
    for r in summary_rows:
        print(
            f"  {r['model']:9s} | {r['video_label']:25s} | Confirmed Alerts: {r['confirmed_alerts']:3d} "
            f"(HG={r['handgun_alerts']:3d}, KN={r['knife_alerts']:3d}) | Avg Conf: {r['avg_confidence_pct']:5.1f}% | "
            f"Raw Candidates: {r['total_candidate_proposals']:3d}"
        )


if __name__ == "__main__":
    main()
