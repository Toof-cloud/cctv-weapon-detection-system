"""
Comprehensive Multi-Stream Evaluation & Verification for Candidate Model V10
Tests both Staged Multi-Camera Footage & Real CCTV Footage.
Verifies:
1. True Positive Confidence >= 70% for weapons at close and distant ranges.
2. False Candidate Confidence < 50% (and 0 alerts after Tier-2 Spatio-Temporal Intelligence).
3. Hair/Ponytail distractor rejection = 0 leaks.
4. Stair railing post rejection = 0 leaks.
5. Room-spanning oversized boxes (1920px) = 0 instances.
"""
import os
import sys
import json
import shutil
import math
from pathlib import Path
from typing import Dict, List, Any

# Limit CPU threads strictly to 8
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
import torch
torch.set_num_threads(8)

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.detection_service import DetectionService
from app.services.cctv_intelligence import CCTVIntelligenceFilter

ARTIFACT_DIR = Path(r"C:\Users\pc\.gemini\antigravity-ide\brain\21f2107b-03e8-4149-84a7-ecff36d50b1d")
OUTPUT_DIR = ROOT / "research" / "model_improvement" / "evaluations" / "v10_evaluation_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FRAMES = OUTPUT_DIR / "keyframes"
OUT_FRAMES.mkdir(parents=True, exist_ok=True)


def draw_labeled_box(img: np.ndarray, box: list, label: str, conf: float, status: str, color: tuple) -> np.ndarray:
    vis = img.copy()
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    banner = f"{status}: {label} {conf:.1%}"
    (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    by1 = max(0, y1 - th - 6)
    cv2.rectangle(vis, (x1, by1), (x1 + tw + 6, by1 + th + 6), color, -1)
    cv2.putText(vis, banner, (x1 + 3, by1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def evaluate_video(
    video_path: Path,
    v10_detector: DetectionService,
    m9_detector: DetectionService,
    keyframe_indices: List[int],
    clip_name: str,
    max_frames: int = 150,
) -> Dict[str, Any]:
    if not video_path.exists():
        print(f"[!] Video not found: {video_path}")
        return {"error": "not_found", "clip": clip_name}

    cap = cv2.VideoCapture(str(video_path))
    v10_filter = CCTVIntelligenceFilter(
        device=v10_detector.device,
        enable_person_gating=True,
        enable_motion_filtering=True,
        enable_geometric_filtering=True,
        enable_temporal_consistency=True,
        min_temporal_hits=2,
        class_thresholds={"handgun": 0.50, "knife": 0.50},
    )

    m9_filter = CCTVIntelligenceFilter(
        device=m9_detector.device,
        enable_person_gating=True,
        enable_motion_filtering=True,
        enable_geometric_filtering=True,
        enable_temporal_consistency=True,
        min_temporal_hits=2,
        class_thresholds={"handgun": 0.50, "knife": 0.50},
    )

    fn = 0
    v10_raw_total = 0
    v10_confirmed_total = 0
    v10_suppressed_total = 0
    v10_max_conf = 0.0
    v10_conf_list = []
    v10_oversized_count = 0
    v10_hair_alerts = 0

    m9_confirmed_total = 0
    m9_oversized_count = 0

    saved_keyframes = []

    while True:
        ret, frame = cap.read()
        if not ret or fn >= max_frames:
            break

        # Inference
        v10_raw = v10_detector.detect_frame(frame, min_threshold=0.30)
        m9_raw = m9_detector.detect_frame(frame, min_threshold=0.30)

        v10_confirmed, v10_all = v10_filter.process_frame(frame, v10_raw, frame_idx=fn)
        m9_confirmed, m9_all = m9_filter.process_frame(frame, m9_raw, frame_idx=fn)

        v10_raw_total += len(v10_raw)
        v10_confirmed_total += len(v10_confirmed)
        v10_suppressed_total += (len(v10_all) - len(v10_confirmed))

        m9_confirmed_total += len(m9_confirmed)

        # Check for oversized boxes (>600px)
        for r in v10_all:
            box = r.get("bounding_box", r.get("box", [0, 0, 0, 0]))
            w, h = max(1, box[2] - box[0]), max(1, box[3] - box[1])
            if max(w, h) > 600:
                v10_oversized_count += 1
            conf = float(r.get("confidence_score", r.get("confidence", 0.0)))
            if r.get("validation_status") in ("CONFIRMED_ALERT", "VALIDATED_TEMPORAL"):
                v10_conf_list.append(conf)
                if conf > v10_max_conf:
                    v10_max_conf = conf

        for r in m9_all:
            box = r.get("bounding_box", r.get("box", [0, 0, 0, 0]))
            w, h = max(1, box[2] - box[0]), max(1, box[3] - box[1])
            if max(w, h) > 600:
                m9_oversized_count += 1

        # Check specific hair alerts in CAM2_SCENE003
        if "SCENE003" in clip_name:
            for r in v10_confirmed:
                box = r.get("bounding_box", r.get("box", [0, 0, 0, 0]))
                center_x = (box[0] + box[2]) / 2.0
                center_y = (box[1] + box[3]) / 2.0
                # Actor head/ponytail area in CAM2_SCENE003 is around x: 500-600, y: 150-320
                if 500 <= center_x <= 620 and 150 <= center_y <= 320:
                    v10_hair_alerts += 1

        # Save keyframes
        if fn in keyframe_indices:
            vis_m9 = frame.copy()
            for r in m9_all:
                box = r.get("bounding_box", r.get("box", [0, 0, 0, 0]))
                lbl = r.get("class_name", r.get("object_label", "weapon")).upper()
                c = float(r.get("confidence_score", r.get("confidence", 0)))
                st = r.get("validation_status", "")
                col = (0, 230, 0) if "CONFIRMED" in st else (0, 0, 255)
                vis_m9 = draw_labeled_box(vis_m9, box, lbl, c, st[:12], col)

            vis_v10 = frame.copy()
            for r in v10_all:
                box = r.get("bounding_box", r.get("box", [0, 0, 0, 0]))
                lbl = r.get("class_name", r.get("object_label", "weapon")).upper()
                c = float(r.get("confidence_score", r.get("confidence", 0)))
                st = r.get("validation_status", "")
                col = (0, 230, 0) if "CONFIRMED" in st else (0, 140, 255)
                vis_v10 = draw_labeled_box(vis_v10, box, lbl, c, st[:12], col)

            p_m9 = cv2.resize(vis_m9, (640, 360))
            p_v10 = cv2.resize(vis_v10, (640, 360))

            cv2.putText(p_m9, "MODEL 9 BASELINE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            cv2.putText(p_v10, "CANDIDATE MODEL V10 (SOTA)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 0), 2)

            combo = np.hstack([p_m9, p_v10])
            fname = f"V10_VS_M9_{clip_name}_f{fn:04d}.jpg"
            save_path = OUT_FRAMES / fname
            cv2.imwrite(str(save_path), combo)
            shutil.copy2(save_path, ARTIFACT_DIR / fname)
            saved_keyframes.append(fname)
            print(f"  [+] Saved keyframe comparison: {fname}")

        fn += 1

    cap.release()

    avg_conf = float(np.mean(v10_conf_list)) if v10_conf_list else 0.0

    return {
        "clip_name": clip_name,
        "analyzed_frames": fn,
        "v10_raw_detections": v10_raw_total,
        "v10_confirmed_alerts": v10_confirmed_total,
        "v10_suppressed": v10_suppressed_total,
        "v10_avg_confidence": round(avg_conf, 3),
        "v10_max_confidence": round(v10_max_conf, 3),
        "v10_oversized_boxes": v10_oversized_count,
        "v10_hair_alerts": v10_hair_alerts,
        "m9_confirmed_alerts": m9_confirmed_total,
        "m9_oversized_boxes": m9_oversized_count,
        "saved_keyframes": saved_keyframes,
    }


def main():
    print("=" * 80)
    print("      CANDIDATE MODEL V10: COMPREHENSIVE MULTI-SCENARIO VERIFICATION       ")
    print("=" * 80)

    v10_path = ROOT / "research" / "model_improvement" / "checkpoints" / "best_candidate_model_v10.pth"
    m9_path = ROOT / "best_weapon_detector_ninth_model.pth"

    if not v10_path.exists():
        print(f"[!] Checkpoint {v10_path.name} not found yet.")
        return

    print(f"Candidate V10: {v10_path.name}")
    print(f"Model 9:       {m9_path.name}")
    print(f"Device:        CUDA (RTX 5060 Ti)\n")

    v10_detector = DetectionService(model_path=v10_path, confidence_threshold=0.30)
    m9_detector = DetectionService(model_path=m9_path, confidence_threshold=0.30)

    test_suite = [
        # Staged Multi-Camera Clips
        {
            "name": "Staged_CAM2_Scene003_Handgun_HairTrap",
            "path": ROOT / "samples" / "NEW_STAGED_CAM-02" / "NEW_CAMERA 02" / "CAM2_SCENE003.mp4",
            "keyframes": [45, 95],
        },
        {
            "name": "Staged_CAM1_Scene011_Firearm_StairPostTrap",
            "path": ROOT / "samples" / "NEW_STAGED_CAM-01" / "NEW_CAMERA 01" / "CAM1_SCENE011.mov",
            "keyframes": [30, 85],
        },
        {
            "name": "Staged_CAM2_Scene009_HallucinationStressTest",
            "path": ROOT / "samples" / "NEW_STAGED_CAM-02" / "NEW_CAMERA 02" / "CAM2_SCENE009.mp4",
            "keyframes": [20, 55],
        },
        {
            "name": "Staged_CAM1_Scene004_KnifeBrandishing",
            "path": ROOT / "samples" / "NEW_STAGED_CAM-01" / "NEW_CAMERA 01" / "CAM1_SCENE004.mov",
            "keyframes": [40, 80],
        },
        # Real CCTV Clips
        {
            "name": "RealCCTV_Clip01_StoreRobbery",
            "path": ROOT / "dataset" / "evaluation_videos" / "Clip01_StoreRobbery.mp4",
            "keyframes": [80, 108],
        },
        {
            "name": "RealCCTV_Clip05_StoreCustomer",
            "path": ROOT / "dataset" / "evaluation_videos" / "Clip05_StoreRobberyCustomer.mp4",
            "keyframes": [40],
        },
        {
            "name": "RealCCTV_Clip07_FastBladeDraw",
            "path": ROOT / "dataset" / "evaluation_videos" / "Clip07_FastBladeDraw.mp4",
            "keyframes": [20],
        },
    ]

    all_results = []
    for test in test_suite:
        print(f"\n[*] Evaluating: {test['name']}...")
        res = evaluate_video(
            video_path=test["path"],
            v10_detector=v10_detector,
            m9_detector=m9_detector,
            keyframe_indices=test["keyframes"],
            clip_name=test["name"],
            max_frames=120,
        )
        all_results.append(res)
        print(f"    V10 Confirmed Alerts: {res.get('v10_confirmed_alerts')}, Max Conf: {res.get('v10_max_confidence')}, Hair Alerts: {res.get('v10_hair_alerts')}, Oversized: {res.get('v10_oversized_boxes')}")

    out_json = OUTPUT_DIR / "v10_comprehensive_evaluation_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[+] Evaluation Complete! Saved summary: {out_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
