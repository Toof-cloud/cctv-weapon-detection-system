import os
import sys
import time
import csv
import json
from pathlib import Path
from typing import Dict, List, Any

# Strict thread management - limit CPU threads to 8
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
import torch
torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from app.services.multi_camera_service import MultiCameraService, CameraInputConfig
from app.services.detection_service import DetectionService
from app.services.cctv_intelligence import CCTVIntelligenceFilter
from app.utils.timestamps import format_timestamp
from benchmarks.calculate_tcr_and_mccr import compute_tcr_for_csv, compute_mccr_for_multicam


def draw_audit_box(frame: np.ndarray, rec: Dict[str, Any]) -> np.ndarray:
    """Draws color-coded bounding boxes and forensic metadata onto frame for visual audit."""
    vis = frame.copy()
    x1 = int(rec["x1"])
    y1 = int(rec["y1"])
    x2 = int(rec["x2"])
    y2 = int(rec["y2"])
    label = rec["object_label"]
    conf = float(rec["confidence_score"])
    status = rec.get("validation_status", "")
    reason = rec.get("rejection_reason", "")

    # Color scheme:
    # Green/Cyan: Confirmed/Validated (>= 0.50)
    # Amber/Yellow: Low confidence (< 0.50, suppressed)
    # Magenta/Red: Spatial/Motion/Person rejection
    if status in ("CONFIRMED_ALERT", "VALIDATED_TEMPORAL"):
        color = (0, 220, 0) if label == "handgun" else (255, 180, 0)
        status_text = f"CONFIRMED ({status})"
    elif "LOW_CONFIDENCE" in status or "BELOW" in reason:
        color = (0, 200, 255) # Amber
        status_text = f"SUPPRESSED (<0.50): {conf:.1%}"
    elif "UNPHYSICAL" in reason or "ANTHROPOMETRIC" in reason:
        color = (180, 0, 255) # Purple/Magenta
        status_text = f"SUPPRESSED (ANATOMIC): {reason[:28]}"
    elif "STATIC" in reason:
        color = (0, 140, 255) # Orange
        status_text = f"SUPPRESSED (STATIC)"
    else:
        color = (0, 0, 255) # Red
        status_text = f"SUPPRESSED: {status}"

    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    banner = f"{label.upper()} {conf:.1%} | {status_text}"
    (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    by1 = max(0, y1 - th - 6)
    cv2.rectangle(vis, (x1, by1), (x1 + tw + 6, by1 + th + 6), color, -1)
    cv2.putText(vis, banner, (x1 + 3, by1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def run_staged_scene(
    scene_idx: int,
    cam1_path: Path,
    cam2_path: Path,
    model_path: Path,
    base_output_dir: Path,
    analysis_fps: int = 20,
) -> Dict[str, Any]:
    """Runs dual-camera pipeline for a single staged scene with full frame audit."""
    scene_name = f"SCENE{scene_idx:03d}"
    scene_out = base_output_dir / f"scene_{scene_idx:03d}"
    scene_out.mkdir(parents=True, exist_ok=True)
    audit_frames_dir = base_output_dir / "audit_frames" / f"scene_{scene_idx:03d}"
    audit_frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"  PROCESSING DUAL-CAMERA {scene_name}")
    print(f"  CAM1: {cam1_path.name}")
    print(f"  CAM2: {cam2_path.name}")
    print(f"{'='*80}")

    case_id = f"CASE-NEW-STAGED-SCENE{scene_idx:03d}"
    
    # 1. Initialize MultiCameraService with Candidate V8
    mc_service = MultiCameraService(
        output_dir=scene_out,
        confidence_threshold=0.50,
        analysis_fps=analysis_fps,
        enable_cctv_intelligence=True,
        enable_temporal_consistency=True,
        model_path=model_path,
    )

    cam_configs = [
        CameraInputConfig(camera_id="CAM-01", video_path=cam1_path, location_name=f"Staged Scene {scene_idx:03d} - Cam 1"),
        CameraInputConfig(camera_id="CAM-02", video_path=cam2_path, location_name=f"Staged Scene {scene_idx:03d} - Cam 2"),
    ]

    result = mc_service.process_cameras(cam_configs, case_id=case_id)
    records = result["records"]
    unified_csv = Path(result["unified_csv"])

    # 2. Split into per-camera CSVs for individual TCR calculation
    reports_dir = scene_out / "reports"
    cam1_csv = reports_dir / f"{case_id}_CAM1_detections.csv"
    cam2_csv = reports_dir / f"{case_id}_CAM2_detections.csv"

    fieldnames = list(dict.fromkeys([k for r in records for k in r.keys()])) if records else [
        "source_video", "frame_number", "timestamp_seconds", "timestamp_formatted",
        "camera_id", "object_label", "bounding_box", "x1", "y1", "x2", "y2",
        "confidence_score", "validation_status", "rejection_reason", "temporal_track_id"
    ]

    cam1_records = [r for r in records if r["camera_id"] == "CAM-01"]
    cam2_records = [r for r in records if r["camera_id"] == "CAM-02"]

    for csv_file, rec_list in [(cam1_csv, cam1_records), (cam2_csv, cam2_records)]:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rec_list)

    # 3. Compute TCR and MCCR
    tcr_cam1 = compute_tcr_for_csv(cam1_csv)
    tcr_cam2 = compute_tcr_for_csv(cam2_csv)
    mccr_res = compute_mccr_for_multicam(unified_csv, delta_t=1.5)

    # 4. Extract annotated frames for all detected frames (frame-by-frame audit)
    # Read both source videos to extract high-quality audit frames
    saved_audit_count = 0
    records_by_cam_frame: Dict[str, Dict[int, List[Dict[str, Any]]]] = {"CAM-01": {}, "CAM-02": {}}
    for r in records:
        cid = r["camera_id"]
        fn = int(r["frame_number"])
        records_by_cam_frame[cid].setdefault(fn, []).append(r)

    for cid, vpath in [("CAM-01", cam1_path), ("CAM-02", cam2_path)]:
        target_frames = records_by_cam_frame.get(cid, {})
        if not target_frames:
            continue

        cap = cv2.VideoCapture(str(vpath))
        cur_fn = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if cur_fn in target_frames:
                frame_recs = target_frames[cur_fn]
                vis = frame.copy()
                for rec in frame_recs:
                    vis = draw_audit_box(vis, rec)
                
                # Watermark
                ts_str = format_timestamp(cur_fn / (cap.get(cv2.CAP_PROP_FPS) or 15.0))
                wm = f"{cid} | Scene {scene_idx:03d} | Frame {cur_fn:04d} ({ts_str})"
                cv2.putText(vis, wm, (15, vis.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

                # Tag primary status
                has_confirmed = any(r.get("validation_status") in ("CONFIRMED_ALERT", "VALIDATED_TEMPORAL") for r in frame_recs)
                prefix = "ALERT" if has_confirmed else "FILTERED"
                primary_label = frame_recs[0]["object_label"]
                out_img_path = audit_frames_dir / f"{prefix}_{cid}_f{cur_fn:04d}_{primary_label}.jpg"
                cv2.imwrite(str(out_img_path), vis)
                saved_audit_count += 1

            cur_fn += 1
        cap.release()

    print(f"  [Audit Frames] Saved {saved_audit_count} annotated inspection frames to {audit_frames_dir.name}")
    print(f"  [TCR] CAM-01: {tcr_cam1.get('tcr', 0.0)}% | CAM-02: {tcr_cam2.get('tcr', 0.0)}%")
    print(f"  [MCCR] Scene {scene_idx:03d}: {mccr_res.get('mccr', 0.0)}% (N_CC={mccr_res.get('n_cc', 0)}, N_MC={mccr_res.get('n_mc', 0)})")

    return {
        "scene_idx": scene_idx,
        "scene_name": scene_name,
        "cam1_records": len(cam1_records),
        "cam2_records": len(cam2_records),
        "total_records": len(records),
        "confirmed_total": sum(1 for r in records if r.get("validation_status") in ("CONFIRMED_ALERT", "VALIDATED_TEMPORAL")),
        "tcr_cam1": tcr_cam1,
        "tcr_cam2": tcr_cam2,
        "mccr": mccr_res,
        "audit_frames_saved": saved_audit_count,
        "unified_csv": str(unified_csv),
        "records": records,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="New Staged Multi-Camera Dataset Benchmark")
    parser.add_argument("--scenes", type=int, nargs="+", default=list(range(1, 13)), help="Scene indices to run (1-12)")
    parser.add_argument("--model", type=str, default="research/model_improvement/checkpoints/best_candidate_model_v8.pth", help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="outputs/new_staged_multicam_run", help="Base output directory")
    parser.add_argument("--model-label", type=str, default="Candidate Model V8", help="Human-readable model label")
    args = parser.parse_args()

    print("=" * 95)
    print(f"      NEW STAGED MULTI-CAMERA DATASET BENCHMARK & TCR/MCCR AUDIT PIPELINE: {args.model_label.upper()}      ")
    print("=" * 95)

    cam1_dir = ROOT / "samples" / "NEW_STAGED_CAM-01" / "NEW_CAMERA 01"
    cam2_dir = ROOT / "samples" / "NEW_STAGED_CAM-02" / "NEW_CAMERA 02"
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    base_output_dir = Path(args.output)
    if not base_output_dir.is_absolute():
        base_output_dir = ROOT / base_output_dir
    base_output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"[Error] Model checkpoint not found at: {model_path}")
        sys.exit(1)

    print(f"Model: {model_path.name} ({args.model_label})")
    print(f"Device: {'CUDA (RTX 5060 Ti)' if torch.cuda.is_available() else 'CPU'}")
    print(f"Input CAM1: {cam1_dir}")
    print(f"Input CAM2: {cam2_dir}")
    print(f"Output Directory: {base_output_dir}")
    print(f"Target Scenes: {args.scenes}\n")

    scene_results: List[Dict[str, Any]] = []
    all_combined_records: List[Dict[str, Any]] = []

    start_benchmark = time.time()

    for i in args.scenes:
        f1 = cam1_dir / f"CAM1_SCENE{i:03d}.mov"
        f2 = cam2_dir / f"CAM2_SCENE{i:03d}.mp4"
        if not f1.exists() or not f2.exists():
            print(f"[Warning] Scene {i:03d} missing: f1={f1.exists()}, f2={f2.exists()}")
            continue

        res = run_staged_scene(
            scene_idx=i,
            cam1_path=f1,
            cam2_path=f2,
            model_path=model_path,
            base_output_dir=base_output_dir,
            analysis_fps=20, # Evaluates all frames
        )
        scene_results.append(res)
        all_combined_records.extend(res["records"])

    total_benchmark_time = time.time() - start_benchmark

    # Save Master Combined CSV of all 12 scenes
    master_csv = base_output_dir / "ALL_12_SCENES_unified_forensic_detections.csv"
    if all_combined_records:
        master_fieldnames = list(dict.fromkeys([k for r in all_combined_records for k in r.keys()]))
        with open(master_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=master_fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_combined_records)

    # Master TCR computation across all 12 scenes
    all_cam1_records = [r for r in all_combined_records if r["camera_id"] == "CAM-01"]
    all_cam2_records = [r for r in all_combined_records if r["camera_id"] == "CAM-02"]
    
    master_cam1_csv = base_output_dir / "ALL_12_SCENES_CAM1_detections.csv"
    master_cam2_csv = base_output_dir / "ALL_12_SCENES_CAM2_detections.csv"

    for csv_file, rec_list in [(master_cam1_csv, all_cam1_records), (master_cam2_csv, all_cam2_records)]:
        if rec_list:
            c_fields = list(dict.fromkeys([k for r in rec_list for k in r.keys()]))
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=c_fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rec_list)

    master_tcr_cam1 = compute_tcr_for_csv(master_cam1_csv)
    master_tcr_cam2 = compute_tcr_for_csv(master_cam2_csv)

    # Aggregate scene-level MCCR
    tot_ncc = sum(s["mccr"].get("n_cc", 0) for s in scene_results)
    tot_nmc = sum(s["mccr"].get("n_mc", 0) for s in scene_results)
    overall_mccr = (tot_ncc / tot_nmc * 100.0) if tot_nmc > 0 else 0.0

    # Aggregate TCR totals
    tot_ts = master_tcr_cam1.get("n_ts", 0) + master_tcr_cam2.get("n_ts", 0)
    tot_iso = master_tcr_cam1.get("n_isolated", 0) + master_tcr_cam2.get("n_isolated", 0)
    tot_int = master_tcr_cam1.get("n_interrupted", 0) + master_tcr_cam2.get("n_interrupted", 0)
    tot_te = master_tcr_cam1.get("n_te", 0) + master_tcr_cam2.get("n_te", 0)
    overall_tcr = (tot_ts / tot_te * 100.0) if tot_te > 0 else 0.0

    # Confidence distribution analysis (Professor's Acceptance Criterion: < 0.50 for false shapes)
    confirmed_scores = [float(r["confidence_score"]) for r in all_combined_records if r.get("validation_status") in ("CONFIRMED_ALERT", "VALIDATED_TEMPORAL")]
    suppressed_scores = [float(r["confidence_score"]) for r in all_combined_records if r.get("validation_status") not in ("CONFIRMED_ALERT", "VALIDATED_TEMPORAL")]

    avg_conf = np.mean(confirmed_scores) if confirmed_scores else 0.0
    min_conf = np.min(confirmed_scores) if confirmed_scores else 0.0
    max_conf = np.max(confirmed_scores) if confirmed_scores else 0.0

    avg_supp_conf = np.mean(suppressed_scores) if suppressed_scores else 0.0
    max_supp_conf = np.max(suppressed_scores) if suppressed_scores else 0.0

    # Print Summary Table
    print("\n" + "=" * 95)
    print("                    12-SCENE MULTI-CAMERA BENCHMARK RESULTS MATRIX                    ")
    print("=" * 95)
    print(f"{'Scene':<10} | {'CAM1 TCR':<10} | {'CAM2 TCR':<10} | {'MCCR':<8} | {'N_CC':<6} | {'N_MC':<6} | {'Alerts':<8} | {'Audit Imgs':<10}")
    print("-" * 95)

    for s in scene_results:
        t1 = s["tcr_cam1"].get("tcr", 0.0)
        t2 = s["tcr_cam2"].get("tcr", 0.0)
        mc = s["mccr"].get("mccr", 0.0)
        ncc = s["mccr"].get("n_cc", 0)
        nmc = s["mccr"].get("n_mc", 0)
        alerts = s["confirmed_total"]
        imgs = s["audit_frames_saved"]
        print(f"{s['scene_name']:<10} | {t1:6.1f}%    | {t2:6.1f}%    | {mc:5.1f}%  | {ncc:<6} | {nmc:<6} | {alerts:<8} | {imgs:<10}")

    print("-" * 95)
    print(f"{'OVERALL':<10} | {master_tcr_cam1.get('tcr', 0.0):6.1f}%    | {master_tcr_cam2.get('tcr', 0.0):6.1f}%    | {overall_mccr:5.1f}%  | {tot_ncc:<6} | {tot_nmc:<6} | {len(confirmed_scores):<8} | {sum(s['audit_frames_saved'] for s in scene_results):<10}")
    print("=" * 95)

    print("\n[CONFIDENCE AUDIT - PROFESSOR'S ACCEPTANCE CRITERION VERIFICATION]")
    print(f"  * Confirmed Weapons (Alerts >= 0.50): Mean = {avg_conf*100:.1f}%, Min = {min_conf*100:.1f}%, Max = {max_conf*100:.1f}%")
    print(f"  * Suppressed Candidates / Shapes:     Mean = {avg_supp_conf*100:.1f}%, Max = {max_supp_conf*100:.1f}%")
    print(f"  * Professor's Criterion Met:          {'YES - All false candidates suppressed/depressed!' if max_supp_conf < 0.50 or len(suppressed_scores) == 0 else 'CHECK RESIDUALS'}")

    print(f"\n[EXECUTION TIME]: Total time elapsed: {total_benchmark_time:.1f}s across 12 scenes (3,625 frames).")
    print(f"[OUTPUT]: Master Unified CSV: {master_csv}")
    print(f"[AUDIT FRAMES]: {base_output_dir / 'audit_frames'}")


if __name__ == "__main__":
    main()
