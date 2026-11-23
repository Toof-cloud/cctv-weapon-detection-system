"""
Evaluate Model 13 (best_weapon_detector_thirteenth_model.pth) on the exact frozen
16-Stream Real CCTV Video Ground-Truth Benchmark (HG01..03, KN01..03 Original & Enhanced + NW01..04)
for apples-to-apples comparison against Models 9, 10, 11, and 12.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_full_pipeline import detect_video_with_model
from training.model_12.evaluate_expert_videos_with_ground_truth import (
    GT_DIR,
    NO_WEAPON_VIDEOS_CATALOG,
    OUT_EVAL_DIR,
    SINGLE_VIDEOS_CATALOG,
    evaluate_predictions_against_video_gt,
)

MODEL_13_PATH = ROOT / "best_weapon_detector_thirteenth_model.pth"
RUN_DIR = OUT_EVAL_DIR / "runs" / "model_13"
OUT_JSON = ROOT / "training" / "model_13" / "model_13_video_gt_benchmark.json"


def aggregate_rows(rows: list[dict]) -> dict:
    w_rows = [r for r in rows if r["category"] in ("Handgun", "Knife")]
    nw_rows = [r for r in rows if r["category"] == "No-Weapon"]

    tp = sum(int(r["TP"]) for r in w_rows)
    fp_w = sum(int(r["FP"]) for r in w_rows)
    fp_nw = sum(int(r["FP"]) for r in nw_rows)
    fp = fp_w + fp_nw
    fn = sum(int(r["FN"]) for r in w_rows)

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-8, prec + rec)

    hg_aps = [float(r["AP_Handgun"]) for r in w_rows if r.get("AP_Handgun") is not None]
    kn_aps = [float(r["AP_Knife"]) for r in w_rows if r.get("AP_Knife") is not None]
    hg_ap = sum(hg_aps) / len(hg_aps) if hg_aps else 0.0
    kn_ap = sum(kn_aps) / len(kn_aps) if kn_aps else 0.0
    v_map = (hg_ap + kn_ap) / 2.0

    nw_tot = sum(int(r["total_analyzed_frames"]) for r in nw_rows)
    nw_clean = sum(int(r["neg_clean_frames"]) for r in nw_rows)
    nw_spec = nw_clean / max(1, nw_tot)

    tp_confs = [float(r["mean_tp_confidence_pct"]) * int(r["TP"]) for r in w_rows if int(r["TP"]) > 0]
    mean_tp_conf = sum(tp_confs) / max(1, tp)

    return {
        "precision": round(prec * 100.0, 2),
        "recall": round(rec * 100.0, 2),
        "f1": round(f1 * 100.0, 2),
        "handgun_ap": round(hg_ap * 100.0, 2),
        "knife_ap": round(kn_ap * 100.0, 2),
        "video_map": round(v_map * 100.0, 2),
        "tp": tp,
        "fp": fp,
        "fp_weapon_videos": fp_w,
        "fp_no_weapon_videos": fp_nw,
        "fn": fn,
        "nw_clean_frames": nw_clean,
        "nw_total_frames": nw_tot,
        "nw_fp": fp_nw,
        "nw_specificity": round(nw_spec * 100.0, 2),
        "mean_tp_conf": round(mean_tp_conf, 2),
    }


def main() -> None:
    t0 = time.time()
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Run Model 13 on all 16 streams (6 weapon pairs Original & Enhanced + 4 No-Weapon)
    exp_root = ROOT / "expert_evaluation_videos"
    for item in SINGLE_VIDEOS_CATALOG:
        pid = item["pair_id"]
        cat_dir = "HANDGUN" if item["category"] == "Handgun" else "KNIFE"
        for ver_tag, v_path in [("original", item["original_path"]), ("enhanced", item["enhanced_path"])]:
            if not v_path.exists():
                v_path = exp_root / cat_dir / f"{pid}_{ver_tag.title()}.mp4"
            out_csv = RUN_DIR / f"{pid}_{ver_tag}_cctv_on.csv"
            out_mp4 = RUN_DIR / f"{pid}_{ver_tag}_cctv_on.mp4"
            if not out_csv.exists():
                print(f"[*] Running Model 13 on {pid} ({ver_tag})...", flush=True)
                detect_video_with_model(
                    input_path=v_path,
                    output_path=out_mp4,
                    model_path=MODEL_13_PATH,
                    confidence_threshold=0.50,
                    analysis_fps=15,
                    save_detected_frames=False,
                    csv_output_path=out_csv,
                    enable_cctv_intelligence=True,
                    enable_temporal_consistency=True,
                    camera_id=pid,
                )

    for item in NO_WEAPON_VIDEOS_CATALOG:
        pid = item["pair_id"]
        v_path = item["video_path"]
        if not v_path.exists():
            v_path = exp_root / "NO_WEAPON" / f"{pid}.mp4"
        out_csv = RUN_DIR / f"{pid}_original_cctv_on.csv"
        out_mp4 = RUN_DIR / f"{pid}_original_cctv_on.mp4"
        if not out_csv.exists():
            print(f"[*] Running Model 13 on {pid} (No-Weapon)...", flush=True)
            detect_video_with_model(
                input_path=v_path,
                output_path=out_mp4,
                model_path=MODEL_13_PATH,
                confidence_threshold=0.50,
                analysis_fps=15,
                save_detected_frames=False,
                csv_output_path=out_csv,
                enable_cctv_intelligence=True,
                enable_temporal_consistency=True,
                camera_id=pid,
            )

    # 2. Evaluate against exact frozen ground-truth CSVs
    eval_rows = []
    for item in SINGLE_VIDEOS_CATALOG:
        pid = item["pair_id"]
        df_gt = pd.read_csv(GT_DIR / f"{pid}_gt.csv")
        for ver_tag in ["original", "enhanced"]:
            csv_p = RUN_DIR / f"{pid}_{ver_tag}_cctv_on.csv"
            for cctv_mode, use_cctv in [("With CCTV Intelligence", True), ("Raw Detector Only", False)]:
                res = evaluate_predictions_against_video_gt(df_gt, csv_p, use_cctv_intelligence=use_cctv, conf_thresh=0.50)
                eval_rows.append({
                    "pair_id": pid,
                    "title": item["title"],
                    "category": item["category"],
                    "video_version": ver_tag,
                    "pipeline_mode": cctv_mode,
                    **res,
                })

    for item in NO_WEAPON_VIDEOS_CATALOG:
        pid = item["pair_id"]
        df_gt = pd.read_csv(GT_DIR / f"{pid}_gt.csv")
        csv_p = RUN_DIR / f"{pid}_original_cctv_on.csv"
        for cctv_mode, use_cctv in [("With CCTV Intelligence", True), ("Raw Detector Only", False)]:
            res = evaluate_predictions_against_video_gt(df_gt, csv_p, use_cctv_intelligence=use_cctv, conf_thresh=0.50)
            eval_rows.append({
                "pair_id": pid,
                "title": item["title"],
                "category": "No-Weapon",
                "video_version": "original",
                "pipeline_mode": cctv_mode,
                **res,
            })

    # 3. Compute exact aggregates matching Table 8 & Table 9
    enh_cctv = [r for r in eval_rows if r["pipeline_mode"] == "With CCTV Intelligence" and (r["video_version"] == "enhanced" or r["category"] == "No-Weapon")]
    orig_cctv = [r for r in eval_rows if r["pipeline_mode"] == "With CCTV Intelligence" and (r["video_version"] == "original" or r["category"] == "No-Weapon")]
    enh_raw = [r for r in eval_rows if r["pipeline_mode"] == "Raw Detector Only" and (r["video_version"] == "enhanced" or r["category"] == "No-Weapon")]

    summary = {
        "model": "Model 13 (Calibrated WiSE-FT alpha=0.50)",
        "checkpoint": MODEL_13_PATH.name,
        "elapsed_seconds": round(time.time() - t0, 1),
        "enhanced_with_cctv_intel": aggregate_rows(enh_cctv),
        "original_with_cctv_intel": aggregate_rows(orig_cctv),
        "enhanced_raw_detector_only": aggregate_rows(enh_raw),
        "per_video_rows": eval_rows,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("MODEL 13 — 16-STREAM REAL CCTV VIDEO GROUND-TRUTH BENCHMARK RESULTS")
    print("=" * 90)
    for k in ["enhanced_with_cctv_intel", "original_with_cctv_intel", "enhanced_raw_detector_only"]:
        s = summary[k]
        print(
            f"  {k:<28} | P={s['precision']:5.2f}% R={s['recall']:5.2f}% F1={s['f1']:5.2f}% "
            f"mAP={s['video_map']:5.2f}% (HG={s['handgun_ap']:5.2f}%, KN={s['knife_ap']:5.2f}%) | "
            f"TP={s['tp']} FP={s['fp']} FN={s['fn']} | NW={s['nw_clean_frames']}/{s['nw_total_frames']} ({s['nw_specificity']:.2f}%)"
        )
    print("\nPer-Video Breakdown (Enhanced + No-Weapon, With CCTV Intelligence):")
    for r in enh_cctv:
        ap_val = (r["AP_Handgun"] * 100.0) if r.get("AP_Handgun") is not None else ((r["AP_Knife"] * 100.0) if r.get("AP_Knife") is not None else (r["neg_specificity"] * 100.0))
        print(
            f"  {r['pair_id']:<5} ({r['category']:<9}) | TP={r['TP']:3d} FP={r['FP']:3d} FN={r['FN']:3d} | "
            f"P={r['Precision']*100:5.1f}% R={r['Recall']*100:5.1f}% F1={r['F1']*100:5.1f}% AP/Spec={ap_val:5.1f}%"
        )
    print(f"\n[+] Saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
