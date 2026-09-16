import os
import sys
import time
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Optional

# Strict CPU thread guard
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
import torch
torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from app.services.detection_service import DetectionService
from app.services.cctv_intelligence import CCTVIntelligenceFilter

OUT_DIR = ROOT / "outputs" / "real_cctv_m9_vs_v9_benchmark_improved"
AUDIT_IMG_DIR = OUT_DIR / "audit_frames"
OUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_IMG_DIR.mkdir(parents=True, exist_ok=True)

M9_PATH = ROOT / "mockup_ui" / "models" / "best_weapon_detector_ninth_model.pth"
if not M9_PATH.exists():
    M9_PATH = ROOT / "best_weapon_detector_ninth_model.pth"

V9_PATH = ROOT / "research" / "model_improvement" / "checkpoints" / "best_candidate_model_v9.pth"


def draw_side_by_side(
    frame: np.ndarray,
    m9_evals: List[Dict[str, Any]],
    v9_evals: List[Dict[str, Any]],
    clip_name: str,
    fn: int,
) -> np.ndarray:
    """Renders a comparative side-by-side frame with Model 9 on left and Candidate V9 on right."""
    p1 = frame.copy()
    p2 = frame.copy()

    def draw_boxes(img: np.ndarray, evals: List[Dict[str, Any]], model_title: str):
        for rec in evals:
            box = rec.get("box", [0, 0, 0, 0])
            x1, y1, x2, y2 = [int(v) for v in box]
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            lbl = rec.get("class_name", "weapon").upper()
            conf = float(rec.get("confidence", 0.0))
            st = str(rec.get("validation_status") or "")
            rs = str(rec.get("rejection_reason") or "")

            if st in ("CONFIRMED_ALERT", "VALIDATED_TEMPORAL"):
                color = (0, 230, 0) if lbl == "HANDGUN" else (255, 200, 0)
                banner = f"CONFIRMED: {lbl} {conf:.1%}"
            elif "OVERSIZED" in st or "OVERSIZED" in rs or "GEOMETRIC" in st:
                color = (0, 0, 255)
                banner = f"SUPPRESSED (OVERSIZED {max(w, h)}px): {conf:.1%}"
            elif "STATIC" in st or "STATIC" in rs:
                color = (0, 140, 255)
                banner = f"SUPPRESSED (STATIC TRAP): {conf:.1%}"
            elif "ANTHROPOMETRIC" in st or "ANTHROPOMETRIC" in rs:
                color = (180, 0, 255)
                banner = f"SUPPRESSED (ANTHROPOMETRIC): {conf:.1%}"
            elif "NO_PERSON" in st or "NO_PERSON" in rs:
                color = (255, 100, 0)
                banner = f"SUPPRESSED (NO HUMAN): {conf:.1%}"
            else:
                color = (0, 200, 255)
                banner = f"SUPPRESSED (<0.50): {conf:.1%}"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, max(0, y1 - 22)), (x1 + 240, y1), (20, 20, 20), -1)
            cv2.putText(img, banner, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

    draw_boxes(p1, m9_evals, "Model 9")
    draw_boxes(p2, v9_evals, "Candidate V9")

    # Resize panels
    h, w = frame.shape[:2]
    target_w, target_h = 960, 540
    p1_res = cv2.resize(p1, (target_w, target_h))
    p2_res = cv2.resize(p2, (target_w, target_h))

    out_w = target_w * 2
    out_h = target_h + 60
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[:60, :] = (30, 30, 30)
    canvas[60:, :target_w] = p1_res
    canvas[60:, target_w:] = p2_res

    # Header text
    m9_alerts = sum(1 for e in m9_evals if e.get("validation_status") in ("CONFIRMED_ALERT", "VALIDATED_TEMPORAL"))
    v9_alerts = sum(1 for e in v9_evals if e.get("validation_status") in ("CONFIRMED_ALERT", "VALIDATED_TEMPORAL"))

    cv2.putText(canvas, f"MODEL 9 BASELINE + TIER-2 | Alerts: {m9_alerts}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 160, 255), 2)
    cv2.putText(canvas, f"CANDIDATE V9 + TIER-2 | Alerts: {v9_alerts} | {clip_name} [f{fn}]", (target_w + 20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 230, 0), 2)

    return canvas


def run_benchmark(frame_stride: int = 2):
    print("=" * 85)
    print("  REAL CCTV SURVEILLANCE BENCHMARK: MODEL 9 vs. CANDIDATE V9 + TIER-2 INTELLIGENCE  ")
    print("=" * 85)
    print(f"Model 9 Checkpoint:      {M9_PATH}")
    print(f"Candidate V9 Checkpoint: {V9_PATH}")
    print(f"Frame Stride:            Every {frame_stride} frames (Analysis rate ~15 FPS)")
    print(f"Output Directory:        {OUT_DIR}")
    print("=" * 85)

    m9_det = DetectionService(model_path=M9_PATH, confidence_threshold=0.25)
    v9_det = DetectionService(model_path=V9_PATH, confidence_threshold=0.25)

    # Suite of real CCTV clips
    clips = [
        # 10 Core Real CCTV Robbery & Attack Footages
        ("Clip01_StoreRobbery", ROOT / "samples/unseen_samples/VIDEO_CLIPS_JABEZ_UNSEEN/VIDEO CLIPS/Screen Recording 2026-09-07 224743.mp4"),
        ("Clip02_CounterAssault", ROOT / "samples/unseen_samples/VIDEO_CLIPS_JABEZ_UNSEEN/VIDEO CLIPS/Screen Recording 2026-09-07 225138.mp4"),
        ("Clip03_RobberyGunConfront", ROOT / "samples/unseen_samples/VIDEO_CLIPS_JABEZ_UNSEEN/VIDEO CLIPS/Screen Recording 2026-09-07 225703.mp4"),
        ("Clip04_KnifeSlashing", ROOT / "samples/unseen_samples/VIDEO_CLIPS_JABEZ_UNSEEN/VIDEO CLIPS/Screen Recording 2026-09-07 230047.mp4"),
        ("Clip05_StoreRobberyCustomer", ROOT / "samples/unseen_samples/VIDEO_CLIPS_JABEZ_UNSEEN/VIDEO CLIPS/Screen Recording 2026-09-07 230249.mp4"),
        ("Clip06_AlleyRobbery", ROOT / "samples/unseen_samples/VIDEO_CLIPS_JABEZ_UNSEEN/VIDEO CLIPS/Screen Recording 2026-09-07 230541.mp4"),
        ("Clip07_FastBladeDraw", ROOT / "samples/unseen_samples/VIDEO_CLIPS_JABEZ_UNSEEN/VIDEO CLIPS/Screen Recording 2026-09-07 230842.mp4"),
        ("Clip08_CornerStoreGun", ROOT / "samples/unseen_samples/VIDEO_CLIPS_JABEZ_UNSEEN/VIDEO CLIPS/Screen Recording 2026-09-07 230942.mp4"),
        ("Clip09_MultiPersonRobbery", ROOT / "samples/unseen_samples/VIDEO_CLIPS_JABEZ_UNSEEN/VIDEO CLIPS/Screen Recording 2026-09-07 231101.mp4"),
        ("Clip10_StreetRobbery", ROOT / "samples/unseen_samples/VIDEO_CLIPS_JABEZ_UNSEEN/VIDEO CLIPS/Screen Recording 2026-09-07 231205.mp4"),
        # Additional New CCTV Footages
        ("NewClip_BeanieGun", ROOT / "samples/NEW-VIDEOS/NEW HANDGUNS VIDEO CLIPS/Screen Recording 2026-09-10 213340 -HANDGUN.mp4"),
        ("NewClip_KitchenKnife", ROOT / "samples/NEW-VIDEOS/NEW HANDGUNS VIDEO CLIPS/KNIFE/Screen Recording 2026-09-10 215511-KNIFE.mp4"),
        ("NewClip_RifleHold", ROOT / "samples/NEW-VIDEOS/NEW HANDGUNS VIDEO CLIPS/1 RIFFLE VIDEO/Screen Recording 2026-09-10 214028-Riffle.mp4"),
    ]

    all_clip_summaries = []
    csv_rows = ["clip_id,clip_name,frame_idx,model_name,class_name,confidence,box_x1,box_y1,box_x2,box_y2,box_w,box_h,status,validation_status,rejection_reason"]

    total_m9_alerts_all = 0
    total_v9_alerts_all = 0
    total_frames_evaluated_all = 0

    audit_frame_count = 0

    for c_idx, (cid, cpath) in enumerate(clips, start=1):
        if not cpath.exists():
            print(f"[Warning] Skipping missing clip: {cpath}")
            continue

        cap = cv2.VideoCapture(str(cpath))
        total_clip_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Isolated filters per stream
        m9_filter = CCTVIntelligenceFilter(device=m9_det.device, min_temporal_hits=2)
        v9_filter = CCTVIntelligenceFilter(device=v9_det.device, min_temporal_hits=2)

        clip_m9_alerts = 0
        clip_v9_alerts = 0
        clip_m9_oversized = 0
        clip_v9_oversized = 0
        clip_m9_max_dim = 0
        clip_v9_max_dim = 0
        clip_eval_frames = 0
        clip_audit_count = 0

        m9_reasons = {}
        v9_reasons = {}

        t0 = time.time()
        print(f"\n[{c_idx}/{len(clips)}] Processing {cid} ({cpath.name}) - {total_clip_frames} frames...")

        fn = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if fn % frame_stride == 0:
                clip_eval_frames += 1

                # 1. Model 9 Detection & Intelligence
                m9_raw = m9_det.detect_frame(frame, min_threshold=0.25)
                m9_conf, m9_eval = m9_filter.process_frame(frame_bgr=frame, raw_detections=m9_raw, frame_idx=fn)

                # 2. Candidate V9 Detection & Intelligence
                v9_raw = v9_det.detect_frame(frame, min_threshold=0.25)
                v9_conf, v9_eval = v9_filter.process_frame(frame_bgr=frame, raw_detections=v9_raw, frame_idx=fn)

                clip_m9_alerts += len(m9_conf)
                clip_v9_alerts += len(v9_conf)

                # Track suppressions
                for r in m9_eval:
                    st = r.get("validation_status", "UNKNOWN")
                    rs = r.get("rejection_reason", "None")
                    m9_reasons[st] = m9_reasons.get(st, 0) + 1
                    b = r["box"]
                    bw, bh = max(1, b[2] - b[0]), max(1, b[3] - b[1])
                    clip_m9_max_dim = max(clip_m9_max_dim, max(bw, bh))
                    if max(bw, bh) > 495: clip_m9_oversized += 1
                    csv_rows.append(f"{cid},{cpath.name},{fn},Model9,{r.get('class_name')},{r.get('confidence'):.4f},{b[0]},{b[1]},{b[2]},{b[3]},{bw},{bh},{r.get('status')},{st},\"{rs}\"")

                for r in v9_eval:
                    st = r.get("validation_status", "UNKNOWN")
                    rs = r.get("rejection_reason", "None")
                    v9_reasons[st] = v9_reasons.get(st, 0) + 1
                    b = r["box"]
                    bw, bh = max(1, b[2] - b[0]), max(1, b[3] - b[1])
                    clip_v9_max_dim = max(clip_v9_max_dim, max(bw, bh))
                    if max(bw, bh) > 495: clip_v9_oversized += 1
                    csv_rows.append(f"{cid},{cpath.name},{fn},CandidateV9,{r.get('class_name')},{r.get('confidence'):.4f},{b[0]},{b[1]},{b[2]},{b[3]},{bw},{bh},{r.get('status')},{st},\"{rs}\"")

                # Save audit frame if either model had an alert or discrepancy or notable event
                has_m9_alert = len(m9_conf) > 0
                has_v9_alert = len(v9_conf) > 0
                has_discrepancy = (has_m9_alert != has_v9_alert)

                if (has_m9_alert or has_v9_alert or has_discrepancy) and (clip_audit_count < 15):
                    # Save side-by-side keyframe
                    if fn % (frame_stride * 4) == 0 or has_discrepancy:
                        vis_card = draw_side_by_side(frame, m9_eval, v9_eval, cid, fn)
                        img_name = f"{cid}_f{fn:04d}.jpg"
                        cv2.imwrite(str(AUDIT_IMG_DIR / img_name), vis_card)
                        clip_audit_count += 1
                        audit_frame_count += 1

            fn += 1

        cap.release()
        elapsed = time.time() - t0
        total_frames_evaluated_all += clip_eval_frames
        total_m9_alerts_all += clip_m9_alerts
        total_v9_alerts_all += clip_v9_alerts

        summary = {
            "clip_id": cid,
            "filename": cpath.name,
            "total_frames": total_clip_frames,
            "evaluated_frames": clip_eval_frames,
            "elapsed_sec": round(elapsed, 1),
            "m9_alerts": clip_m9_alerts,
            "v9_alerts": clip_v9_alerts,
            "m9_oversized_count": clip_m9_oversized,
            "v9_oversized_count": clip_v9_oversized,
            "m9_max_dim": clip_m9_max_dim,
            "v9_max_dim": clip_v9_max_dim,
            "m9_reasons": m9_reasons,
            "v9_reasons": v9_reasons,
        }
        all_clip_summaries.append(summary)

        print(f"  -> Done in {elapsed:.1f}s. Evaluated: {clip_eval_frames} frames.")
        print(f"     Model 9 Alerts: {clip_m9_alerts} | Candidate V9 Alerts: {clip_v9_alerts}")
        print(f"     Model 9 Max Dim: {clip_m9_max_dim}px (Oversized: {clip_m9_oversized}) | V9 Max Dim: {clip_v9_max_dim}px (Oversized: {clip_v9_oversized})")

    # Save outputs
    csv_path = OUT_DIR / "real_cctv_m9_vs_v9_unified_detections.csv"
    csv_path.write_text("\n".join(csv_rows), encoding="utf-8")

    json_path = OUT_DIR / "real_cctv_m9_vs_v9_benchmark_summary.json"
    json_path.write_text(json.dumps({
        "total_clips": len(all_clip_summaries),
        "total_frames_evaluated": total_frames_evaluated_all,
        "total_m9_alerts": total_m9_alerts_all,
        "total_v9_alerts": total_v9_alerts_all,
        "clip_summaries": all_clip_summaries,
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 85)
    print("REAL CCTV BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"Total Evaluated Frames: {total_frames_evaluated_all}")
    print(f"Total Model 9 Alerts:   {total_m9_alerts_all}")
    print(f"Total Candidate V9 Alerts: {total_v9_alerts_all}")
    print(f"Audit Frames Generated: {audit_frame_count} in {AUDIT_IMG_DIR}")
    print(f"Forensic CSV Log:       {csv_path}")
    print(f"Executive JSON Summary: {json_path}")
    print("=" * 85)


if __name__ == "__main__":
    run_benchmark(frame_stride=2)
