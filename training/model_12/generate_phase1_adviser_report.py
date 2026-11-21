"""
Aggregate and print all Phase 1 metrics:
1. Static Image Benchmarks (Models 9, 10, 11, 12 across External 350, Unseen CCTV 185, Internal 279, Grid 18, Staged Val 44).
2. Real CCTV Video Ground-Truth Evaluation across 16 Video Streams (3 Handgun Orig/Enh pairs, 3 Knife Orig/Enh pairs, 4 Non-Handgun videos):
   - Model 9 vs. Model 10 (Before Staged & Surv Expansion) vs. Model 11 vs. Model 12 (After Full Staged & Surv Expansion)
   - Unenhanced Original vs. BasicVSR++ Enhanced
   - Raw Detector vs. Full Pipeline (+ CCTV Intelligence Layer)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_EVAL_DIR = ROOT / "outputs" / "phase1_final_video_gt_benchmark"


def main():
    df = pd.read_csv(OUT_EVAL_DIR / "video_ground_truth_evaluation_results.csv")

    print("=" * 110)
    print("  1. PER-VIDEO GROUND-TRUTH BENCHMARK RESULTS (WITH CCTV INTELLIGENCE LAYER)")
    print("=" * 110)
    sub = df[df["pipeline_mode"] == "With CCTV Intelligence"]
    for pid in ["HG01", "HG02", "HG03", "KN01", "KN02", "KN03", "NW01", "NW02", "NW03", "NW04"]:
        psub = sub[sub["pair_id"] == pid]
        for ver in psub["video_version"].unique():
            vsub = psub[psub["video_version"] == ver]
            parts = [f"{pid}_{ver:8s}"]
            for m in ["model_9", "model_10", "model_11", "model_12"]:
                r = vsub[vsub["model_tag"] == m].iloc[0]
                tp, fp, fn = int(r["TP"]), int(r["FP"]), int(r["FN"])
                p_pct = r["Precision"] * 100.0
                r_pct = r["Recall"] * 100.0
                f1_pct = r["F1"] * 100.0
                conf = r["mean_confidence_pct"]
                if pd.notna(r["AP_Handgun"]):
                    ap_val = r["AP_Handgun"] * 100.0
                elif pd.notna(r["AP_Knife"]):
                    ap_val = r["AP_Knife"] * 100.0
                else:
                    ap_val = r["neg_specificity"] * 100.0
                parts.append(
                    f"{m}: TP={tp:2d} FP={fp:2d} FN={fn:2d} P={p_pct:5.1f}% R={r_pct:5.1f}% F1={f1_pct:5.1f}% AP/Spec={ap_val:5.1f}% Conf={conf:4.1f}%"
                )
            print(" | ".join(parts))

    print("\n" + "=" * 110)
    print("  2. AGGREGATE VIDEO GROUND-TRUTH PERFORMANCE: BEFORE (MODEL 10) vs. AFTER (MODEL 11 & MODEL 12)")
    print("=" * 110)
    summary_by_model = []
    for m_tag in ["model_9", "model_10", "model_11", "model_12"]:
        for ver_filter, ver_label in [("enhanced", "Enhanced Weapon Videos + No-Weapon"), ("original", "Unenhanced Original Weapon Videos + No-Weapon")]:
            w_sub = sub[(sub["model_tag"] == m_tag) & (sub["category"].isin(["Handgun", "Knife"])) & (sub["video_version"] == ver_filter)]
            nw_sub = sub[(sub["model_tag"] == m_tag) & (sub["category"] == "No-Weapon")]

            tp_sum = int(w_sub["TP"].sum())
            fp_w = int(w_sub["FP"].sum())
            fp_nw = int(nw_sub["FP"].sum())
            fp_sum = fp_w + fp_nw
            fn_sum = int(w_sub["FN"].sum())

            prec = tp_sum / max(1, tp_sum + fp_sum)
            rec = tp_sum / max(1, tp_sum + fn_sum)
            f1 = 2 * prec * rec / max(1e-8, prec + rec)

            hg_aps = w_sub[w_sub["category"] == "Handgun"]["AP_Handgun"].dropna().tolist()
            kn_aps = w_sub[w_sub["category"] == "Knife"]["AP_Knife"].dropna().tolist()
            mean_hg_ap = float(np.mean(hg_aps)) if hg_aps else 0.0
            mean_kn_ap = float(np.mean(kn_aps)) if kn_aps else 0.0
            video_map = (mean_hg_ap + mean_kn_ap) / 2.0

            nw_total_frames = int(nw_sub["gt_negative_frames"].sum())
            nw_clean_frames = int(nw_sub["neg_clean_frames"].sum())
            nw_spec = nw_clean_frames / max(1, nw_total_frames)

            tp_confs = w_sub[w_sub["TP"] > 0]["mean_tp_confidence_pct"].tolist()
            avg_tp_conf = float(np.mean(tp_confs)) if tp_confs else 0.0

            row = {
                "model_tag": m_tag,
                "stream_set": ver_label,
                "TP": tp_sum,
                "FP_weapon_videos": fp_w,
                "FP_noweapon_videos": fp_nw,
                "FP_total": fp_sum,
                "FN": fn_sum,
                "Precision_pct": round(prec * 100.0, 2),
                "Recall_pct": round(rec * 100.0, 2),
                "F1_pct": round(f1 * 100.0, 2),
                "Video_AP_Handgun_pct": round(mean_hg_ap * 100.0, 2),
                "Video_AP_Knife_pct": round(mean_kn_ap * 100.0, 2),
                "Video_mAP_pct": round(video_map * 100.0, 2),
                "NoWeapon_Clean_Frames": f"{nw_clean_frames}/{nw_total_frames}",
                "NoWeapon_Specificity_pct": round(nw_spec * 100.0, 2),
                "Mean_TP_Confidence_pct": round(avg_tp_conf, 2),
            }
            summary_by_model.append(row)
            print(
                f"  {m_tag:8s} | {ver_label:45s} | "
                f"P={row['Precision_pct']:6.2f}% R={row['Recall_pct']:6.2f}% F1={row['F1_pct']:6.2f}% | "
                f"Video mAP={row['Video_mAP_pct']:6.2f}% (HG={row['Video_AP_Handgun_pct']:5.2f}%, KN={row['Video_AP_Knife_pct']:5.2f}%) | "
                f"TP={tp_sum:3d} FP={fp_sum:2d} (NW_FP={fp_nw:2d}) FN={fn_sum:3d} | "
                f"NW_Spec={row['NoWeapon_Specificity_pct']:6.2f}% ({row['NoWeapon_Clean_Frames']}) | Conf={row['Mean_TP_Confidence_pct']:5.2f}%"
            )

    print("\n" + "=" * 110)
    print("  3. ABLATION: RAW DETECTOR ONLY vs. WITH CCTV INTELLIGENCE LAYER (MODEL 12 ON ALL VIDEOS)")
    print("=" * 110)
    for mode in ["Raw Detector Only", "With CCTV Intelligence"]:
        m12_w = df[(df["model_tag"] == "model_12") & (df["pipeline_mode"] == mode) & (df["category"].isin(["Handgun", "Knife"])) & (df["video_version"] == "enhanced")]
        m12_nw = df[(df["model_tag"] == "model_12") & (df["pipeline_mode"] == mode) & (df["category"] == "No-Weapon")]
        tp = int(m12_w["TP"].sum())
        fp_w = int(m12_w["FP"].sum())
        fp_nw = int(m12_nw["FP"].sum())
        fp = fp_w + fp_nw
        fn = int(m12_w["FN"].sum())
        p = tp / max(1, tp + fp) * 100.0
        r = tp / max(1, tp + fn) * 100.0
        f1 = 2 * p * r / max(1e-8, p + r)
        nw_clean = int(m12_nw["neg_clean_frames"].sum())
        nw_tot = int(m12_nw["gt_negative_frames"].sum())
        print(
            f"  {mode:24s} | P={p:6.2f}% R={r:6.2f}% F1={f1:6.2f}% | "
            f"TP={tp:3d} FP_Total={fp:3d} (WeaponVid_FP={fp_w:2d}, NoWeaponVid_FP={fp_nw:2d}) FN={fn:3d} | "
            f"No-Weapon Clean Frames: {nw_clean}/{nw_tot} ({nw_clean/max(1,nw_tot)*100:.2f}%)"
        )

    pd.DataFrame(summary_by_model).to_csv(OUT_EVAL_DIR / "video_gt_aggregate_summary.csv", index=False)
    (OUT_EVAL_DIR / "video_gt_aggregate_summary.json").write_text(json.dumps(summary_by_model, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

