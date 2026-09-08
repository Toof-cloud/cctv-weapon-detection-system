import os
import sys
import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Thread guard
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def compute_tcr_for_csv(csv_path: Path) -> Dict[str, Any]:
    """
    Computes Temporal Consistency Rate (TCR) according to the exact thesis specification:
    TCR = (N_TS / N_TE) * 100
    where:
    - N_TS: Number of Temporally Supported detection observations (belonging to validated persistent tracklet, H >= 2)
    - N_TE: Total number of detection observations eligible for temporal evaluation.
    
    Categories:
    - Temporally Supported: validation_status in ('VALIDATED_TEMPORAL', 'CONFIRMED_ALERT')
    - Isolated: validation_status == 'SUPPRESSED_TEMPORAL_FLICKER'
    - Interrupted: gaps in tracking exceeding permitted threshold
    - Not Evaluable: boundary observations (frame 0 or last frame where preceding/succeeding frames are unavailable).
    """
    if not csv_path.exists():
        return {}

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {}

    # Identify min/max frames to categorize boundary 'Not Evaluable'
    frame_nums = [int(r["frame_number"]) for r in rows if r.get("frame_number", "").isdigit()]
    min_frame = min(frame_nums) if frame_nums else 0
    max_frame = max(frame_nums) if frame_nums else 0

    # Categorization buckets
    supported_rows = []
    isolated_rows = []
    interrupted_count = 0
    not_evaluable_rows = []
    
    # Tracklet analysis for continuity & interruption
    track_frames: Dict[str, List[int]] = {}

    for r in rows:
        fn = int(r.get("frame_number", 0))
        status = r.get("validation_status", "")
        label = r.get("object_label", "")

        # 1. Boundary check -> Not Evaluable
        if fn == min_frame or fn == max_frame:
            not_evaluable_rows.append(r)
            continue

        # 2. Status categorization
        if status in ("VALIDATED_TEMPORAL", "CONFIRMED_ALERT"):
            supported_rows.append(r)
            track_frames.setdefault(label, []).append(fn)
        elif status == "SUPPRESSED_TEMPORAL_FLICKER":
            isolated_rows.append(r)

    # Calculate track interruptions (gaps > 15 frames between supported frames of the same class)
    for lbl, f_list in track_frames.items():
        f_sorted = sorted(f_list)
        for i in range(len(f_sorted) - 1):
            gap = f_sorted[i+1] - f_sorted[i]
            if gap > 15: # permitted missed-frame gap exceeded
                interrupted_count += 1

    # Counts
    n_ts = len(supported_rows)
    n_isolated = len(isolated_rows)
    n_interrupted = interrupted_count
    n_not_evaluable = len(not_evaluable_rows)

    # N_TE = N_TS + Isolated + Interrupted (eligible observations per manuscript)
    n_te = n_ts + n_isolated + n_interrupted
    tcr = (n_ts / n_te * 100.0) if n_te > 0 else 0.0

    # Class-specific TCR
    hg_ts = sum(1 for r in supported_rows if r.get("object_label") == "handgun")
    hg_iso = sum(1 for r in isolated_rows if r.get("object_label") == "handgun")
    hg_te = hg_ts + hg_iso
    tcr_hg = (hg_ts / hg_te * 100.0) if hg_te > 0 else 0.0

    knife_ts = sum(1 for r in supported_rows if r.get("object_label") == "knife")
    knife_iso = sum(1 for r in isolated_rows if r.get("object_label") == "knife")
    knife_te = knife_ts + knife_iso
    tcr_knife = (knife_ts / knife_te * 100.0) if knife_te > 0 else 0.0

    return {
        "csv_name": csv_path.name,
        "n_ts": n_ts,
        "n_isolated": n_isolated,
        "n_interrupted": n_interrupted,
        "n_not_evaluable": n_not_evaluable,
        "n_te": n_te,
        "tcr": round(tcr, 2),
        "tcr_handgun": round(tcr_hg, 2),
        "tcr_knife": round(tcr_knife, 2),
        "hg_ts": hg_ts,
        "knife_ts": knife_ts,
    }


def compute_mccr_for_multicam(unified_csv: Path, delta_t: float = 1.5) -> Dict[str, Any]:
    """
    Computes Multi-Camera Corroboration Rate (MCCR) according to the exact thesis specification:
    MCCR = (N_CC / N_MC) * 100
    where:
    - N_CC: Number of eligible observations receiving compatible cross-camera support (Class match within delta_t)
    - N_MC: Total number of observations eligible for multi-camera comparison.
    
    Categories:
    - Corroborated: Validated alert in Cam A matching confirmed alert in Cam B within delta_t with identical class
    - Not Corroborated: Validated alert in Cam A with no matching support in Cam B during concurrent recording
    - Uncertain: Concurrent alerts in Cam A and Cam B within delta_t but with conflicting weapon classes
    - Not Applicable: Observations outside the concurrent recording time window of both cameras (excluded from N_MC).
    """
    if not unified_csv.exists():
        return {}

    with open(unified_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {}

    # Separate by camera
    cams = sorted(list(set(r.get("camera_id", "") for r in rows if r.get("camera_id"))))
    if len(cams) < 2:
        return {"error": "Requires at least 2 cameras for MCCR comparison"}

    cam_a_id, cam_b_id = cams[0], cams[1]
    
    # Determine concurrent alignable time interval [0, min(duration_A, duration_B)]
    import cv2
    duration_a = 0.0
    duration_b = 0.0
    for r in rows:
        vid_name = r.get("source_video", "")
        cid = r.get("camera_id", "")
        if cid == cam_a_id and duration_a == 0.0:
            for cand in [ROOT / "samples" / vid_name, ROOT / "samples" / "staged_dataset" / "CAM01" / "CAMERA 01" / vid_name]:
                if cand.exists():
                    cap = cv2.VideoCapture(str(cand))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if fps > 0: duration_a = fc / fps
                    cap.release()
                    break
        elif cid == cam_b_id and duration_b == 0.0:
            for cand in [ROOT / "samples" / vid_name, ROOT / "samples" / "staged_dataset" / "CAM02" / "CAMERA 02" / vid_name]:
                if cand.exists():
                    cap = cv2.VideoCapture(str(cand))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if fps > 0: duration_b = fc / fps
                    cap.release()
                    break

    # Fallback to max timestamp in rows if video file not accessible
    if duration_a == 0.0:
        ts_a = [float(r["timestamp_seconds"]) for r in rows if r.get("camera_id") == cam_a_id and r.get("timestamp_seconds")]
        duration_a = max(ts_a) if ts_a else 10.0
    if duration_b == 0.0:
        ts_b = [float(r["timestamp_seconds"]) for r in rows if r.get("camera_id") == cam_b_id and r.get("timestamp_seconds")]
        duration_b = max(ts_b) if ts_b else 10.0

    concurrent_start = 0.0
    concurrent_end = min(duration_a, duration_b)

    # Filter confirmed / validated alerts
    confirmed_rows = [
        r for r in rows 
        if r.get("validation_status") in ("CONFIRMED_ALERT", "VALIDATED_TEMPORAL")
    ]

    n_cc = 0
    not_corroborated = 0
    uncertain = 0
    not_applicable = 0

    corroborated_pairs = []

    for r in confirmed_rows:
        t = float(r["timestamp_seconds"])
        cam_id = r["camera_id"]
        label = r["object_label"]
        conf = float(r["confidence_score"])

        # 1. Not Applicable check (outside concurrent alignable recording window)
        if t < concurrent_start or t > concurrent_end:
            not_applicable += 1
            continue

        # Target opposing camera records
        opposing_cams = [
            o for o in confirmed_rows 
            if o["camera_id"] != cam_id and abs(float(o["timestamp_seconds"]) - t) <= delta_t
        ]

        if not opposing_cams:
            not_corroborated += 1
        else:
            # Check class compatibility
            matching_class_cams = [o for o in opposing_cams if o["object_label"] == label]
            if matching_class_cams:
                n_cc += 1
                best_opp = matching_class_cams[0]
                # Joint Bayesian Corroborated Confidence: 1 - (1 - c1)(1 - c2)
                fused_conf = 1.0 - (1.0 - conf) * (1.0 - float(best_opp["confidence_score"]))
                corroborated_pairs.append({
                    "cam_a": cam_id,
                    "cam_b": best_opp["camera_id"],
                    "time_a": t,
                    "time_b": float(best_opp["timestamp_seconds"]),
                    "class": label,
                    "conf_a": conf,
                    "conf_b": float(best_opp["confidence_score"]),
                    "fused_conf": round(fused_conf, 4)
                })
            else:
                uncertain += 1

    # N_MC denominator excludes Not Applicable per manuscript specification
    n_mc = n_cc + not_corroborated + uncertain
    mccr = (n_cc / n_mc * 100.0) if n_mc > 0 else 0.0

    return {
        "unified_csv": unified_csv.name,
        "camera_a": cam_a_id,
        "camera_b": cam_b_id,
        "concurrent_interval": f"[{concurrent_start:.2f}s, {concurrent_end:.2f}s]",
        "n_cc": n_cc,
        "not_corroborated": not_corroborated,
        "uncertain": uncertain,
        "not_applicable": not_applicable,
        "n_mc": n_mc,
        "mccr": round(mccr, 2),
        "corroborated_pairs_sample": corroborated_pairs[:5]
    }


def main():
    print("=" * 95)
    print("      OFFICIAL TCR & MCCR EVALUATION ENGINE (MANUSCRIPT SPECIFICATION COMPLIANT)      ")
    print("=" * 95)

    # 1. Evaluate TCR across Model 9 Baseline Sample Videos
    print("\n[PART 1] TEMPORAL CONSISTENCY RATE (TCR) EVALUATION - BASELINE 4 VIDEOS (MODEL 9)")
    print("-" * 95)
    print(f"{'Video Scenario':<32} | {'N_TS':<6} | {'Isolated':<8} | {'Interr':<6} | {'N_TE':<6} | {'TCR (Total)':<11} | {'TCR (HG)':<8} | {'TCR (Knife)':<11}")
    print("-" * 95)

    baseline_jobs = [
        ("Handgun Normal (handgun_test)", ROOT / "outputs" / "ninth_model_outputs" / "handgun_detection_cctv_footage" / "forensic_detections_no_enhancement.csv"),
        ("Knife Normal (evaluation_video)", ROOT / "outputs" / "ninth_model_outputs" / "knife_detection_evaluationvideo" / "forensic_detections_no_enhancement.csv"),
        ("Knife CCTV (NEW_KNIFE_VIDEO)", ROOT / "outputs" / "ninth_model_outputs" / "knife_detection_cctv_footage" / "forensic_detections_no_enhancement.csv"),
        ("Handgun Staged (CAM02_Scene004)", ROOT / "outputs" / "ninth_model_outputs" / "handgun_detection_staged_dataset" / "forensic_detections_no_enhancement.csv"),
    ]

    total_ts = 0
    total_iso = 0
    total_int = 0
    total_te = 0

    for name, p in baseline_jobs:
        res = compute_tcr_for_csv(p)
        if not res: continue
        total_ts += res["n_ts"]
        total_iso += res["n_isolated"]
        total_int += res["n_interrupted"]
        total_te += res["n_te"]
        print(f"{name:<32} | {res['n_ts']:<6} | {res['n_isolated']:<8} | {res['n_interrupted']:<6} | {res['n_te']:<6} | {res['tcr']:<6}%     | {res['tcr_handgun']:<6}%  | {res['tcr_knife']:<6}%")

    overall_baseline_tcr = (total_ts / total_te * 100.0) if total_te > 0 else 0.0
    print("-" * 95)
    print(f"{'OVERALL BASELINE BENCHMARK':<32} | {total_ts:<6} | {total_iso:<8} | {total_int:<6} | {total_te:<6} | {overall_baseline_tcr:.2f}%")

    # 2. Evaluate TCR across 10 Unseen CCTV Clips (Model 9)
    print("\n\n[PART 2] TEMPORAL CONSISTENCY RATE (TCR) - 10 UNSEEN REAL-WORLD CCTV CLIPS (MODEL 9)")
    print("-" * 95)
    print(f"{'Clip Identifier':<32} | {'N_TS':<6} | {'Isolated':<8} | {'Interr':<6} | {'N_TE':<6} | {'TCR (Total)':<11} | {'Primary Threat Class':<20}")
    print("-" * 95)

    unseen_base = ROOT / "outputs" / "unseen_jabez_model9_evaluation"
    unseen_ts = 0
    unseen_iso = 0
    unseen_int = 0
    unseen_te = 0

    if unseen_base.exists():
        for d in sorted(unseen_base.iterdir()):
            if not d.is_dir(): continue
            cp = d / "forensic_detections.csv"
            res = compute_tcr_for_csv(cp)
            if not res: continue
            unseen_ts += res["n_ts"]
            unseen_iso += res["n_isolated"]
            unseen_int += res["n_interrupted"]
            unseen_te += res["n_te"]
            p_class = f"Handgun ({res['hg_ts']})" if res['hg_ts'] > res['knife_ts'] else f"Knife ({res['knife_ts']})" if res['knife_ts'] > 0 else "None"
            print(f"{d.name[:32]:<32} | {res['n_ts']:<6} | {res['n_isolated']:<8} | {res['n_interrupted']:<6} | {res['n_te']:<6} | {res['tcr']:<6}%     | {p_class:<20}")

    overall_unseen_tcr = (unseen_ts / unseen_te * 100.0) if unseen_te > 0 else 0.0
    print("-" * 95)
    print(f"{'OVERALL UNSEEN CCTV CLIPS':<32} | {unseen_ts:<6} | {unseen_iso:<8} | {unseen_int:<6} | {unseen_te:<6} | {overall_unseen_tcr:.2f}%")

    # 3. Evaluate MCCR for Multi-Camera Surveillance
    print("\n\n[PART 3] MULTI-CAMERA CORROBORATION RATE (MCCR) - DUAL-VIEW CCTV BENCHMARK")
    print("-" * 95)
    
    multicam_reports = [
        ("Model 9 Multi-Camera Evaluation (Scene 004)", ROOT / "outputs" / "multicam_model9_scene004_evaluation" / "reports" / "CASE-CCTV-1788874158_forensic_detections.csv"),
        ("Dual-Camera Full Proposal Run (Scene 004)", ROOT / "outputs" / "scene004_multicam_run" / "reports" / "CASE-CCTV-1788751996_forensic_detections.csv"),
    ]
    
    mccr_results = []
    for label, report_path in multicam_reports:
        if not report_path.exists():
            continue
        mccr_res = compute_mccr_for_multicam(report_path, delta_t=1.5)
        mccr_results.append((label, mccr_res))
        print(f"\n--- Scenario: {label} ---")
        print(f"Evaluated Report: {mccr_res['unified_csv']}")
        print(f"Cameras Compared: {mccr_res['camera_a']} and {mccr_res['camera_b']}")
        print(f"Concurrent Aligned Window: {mccr_res['concurrent_interval']}")
        print(f"Corroborated Observations (N_CC):         {mccr_res['n_cc']}")
        print(f"Not Corroborated Observations:            {mccr_res['not_corroborated']}")
        print(f"Uncertain Observations (Class Mismatch):  {mccr_res['uncertain']}")
        print(f"Not Applicable (Excluded from Denom):      {mccr_res['not_applicable']}")
        print(f"Total Eligible Observations (N_MC):       {mccr_res['n_mc']}")
        print(f"--> MULTI-CAMERA CORROBORATION RATE (MCCR): {mccr_res['mccr']}%")

        if mccr_res.get("corroborated_pairs_sample"):
            print("\nSample Cross-Camera Corroborated Incidents with Joint Probabilistic Fusion:")
            for pair in mccr_res["corroborated_pairs_sample"]:
                print(f"  * Threat: {pair['class'].upper()} | {pair['cam_a']} ({pair['time_a']:.2f}s, Conf: {pair['conf_a']*100:.1f}%) <-> {pair['cam_b']} ({pair['time_b']:.2f}s, Conf: {pair['conf_b']*100:.1f}%) => Fused Certainty: {pair['fused_conf']*100:.2f}%")

    print("\n" + "=" * 95)
    print("                    EVALUATION EXECUTION COMPLETED SUCCESSFULLY                       ")
    print("=" * 95)

if __name__ == "__main__":
    main()
