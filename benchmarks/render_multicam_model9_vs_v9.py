import os
import sys
import shutil
import math
from pathlib import Path
from typing import Dict, List, Any

# Strict CPU thread limit to 8
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

OUT_VID_DIR = ROOT / "outputs" / "model9_vs_v9_multicam_comparison" / "videos"
OUT_IMG_DIR = ROOT / "outputs" / "model9_vs_v9_multicam_comparison" / "frames"
OUT_VID_DIR.mkdir(parents=True, exist_ok=True)
OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACT_DIR = Path(r"C:\Users\pc\.gemini\antigravity-ide\brain\21f2107b-03e8-4149-84a7-ecff36d50b1d")


def draw_intelligence_box(img: np.ndarray, rec: Dict[str, Any], model_label: str) -> np.ndarray:
    """Draws color-coded bounding box and forensic metadata banner onto frame."""
    vis = img.copy()
    box = rec.get("bounding_box", rec.get("box"))
    if isinstance(box, str):
        import json
        box = json.loads(box)
    x1, y1, x2, y2 = [int(v) for v in box]
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    max_dim = max(w, h)

    lbl = rec.get("class_name", rec.get("object_label", "weapon")).upper()
    conf = float(rec.get("confidence_score", rec.get("confidence", 0)))
    status = str(rec.get("validation_status") or "")
    reason = str(rec.get("rejection_reason") or "")

    # Visual color encoding
    if status in ("CONFIRMED_ALERT", "VALIDATED_TEMPORAL"):
        color = (0, 230, 0) if lbl == "HANDGUN" else (255, 200, 0) # Green for Handgun, Cyan/Gold for Knife
        banner = f"CONFIRMED: {lbl} {conf:.1%}"
    elif "OVERSIZED" in status or "OVERSIZED" in reason or "GEOMETRIC" in status:
        color = (0, 0, 255) # Red/Magenta for oversized ballooning
        banner = f"SUPPRESSED (OVERSIZED {max_dim}px): {conf:.1%}"
    elif "STATIC" in status or "STATIC" in reason:
        color = (0, 140, 255) # Orange for static environmental shadow/trap
        banner = f"SUPPRESSED (STATIC TRAP): {conf:.1%}"
    elif "UNPHYSICAL" in status or "UNPHYSICAL" in reason or "PHONE" in status or "PHONE" in reason:
        color = (180, 0, 255) # Purple for anatomic / phone distractor
        banner = f"SUPPRESSED (ANATOMIC/HAIR): {conf:.1%}"
    elif "ANTHROPOMETRIC" in status or "ANTHROPOMETRIC" in reason or "SCALE" in status:
        color = (180, 0, 255) # Purple/Violet for anthropometric scale violation
        banner = f"SUPPRESSED (ANTHROPOMETRIC): {conf:.1%}"
    elif "NO_PERSON" in status or "NO_PERSON" in reason:
        color = (255, 100, 0) # Blue for no human proximity
        banner = f"SUPPRESSED (NO HUMAN): {conf:.1%}"
    else:
        color = (0, 200, 255) # Amber for low confidence < 0.50
        banner = f"SUPPRESSED (<0.50): {conf:.1%}"

    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    by1 = max(0, y1 - th - 6)
    cv2.rectangle(vis, (x1, by1), (x1 + tw + 6, by1 + th + 6), color, -1)
    cv2.putText(vis, banner, (x1 + 3, by1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def render_scene_comparison(
    scene_idx: int,
    cam_id: str,
    vid_path: Path,
    m9_detector: DetectionService,
    v9_detector: DetectionService,
    keyframe_indices: List[int],
    max_frames: int = 150,
):
    if not vid_path.exists():
        print(f"[Warning] Video not found: {vid_path}")
        return

    scene_tag = f"Scene{scene_idx:03d}_{cam_id}"
    print(f"\nProcessing Multi-Camera Comparison: {scene_tag} ({vid_path.name})...")

    cap = cv2.VideoCapture(str(vid_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize isolated Tier-2 filters for each model stream
    m9_filter = CCTVIntelligenceFilter(
        device=m9_detector.device,
        enable_person_gating=True,
        enable_motion_filtering=True,
        enable_geometric_filtering=True,
        enable_temporal_consistency=True,
        min_temporal_hits=2,
        class_thresholds={"handgun": 0.50, "knife": 0.50},
    )

    v9_filter = CCTVIntelligenceFilter(
        device=v9_detector.device,
        enable_person_gating=True,
        enable_motion_filtering=True,
        enable_geometric_filtering=True,
        enable_temporal_consistency=True,
        min_temporal_hits=2,
        class_thresholds={"handgun": 0.50, "knife": 0.50},
    )

    # Resolution settings
    panel_w, panel_h = 960, 540
    out_w = panel_w * 2
    out_h = panel_h + 65 # Header bar

    out_video_path = OUT_VID_DIR / f"COMPARISON_{scene_tag}_M9_vs_CandidateV9.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (out_w, out_h))

    fn = 0
    m9_alerts_total = 0
    v9_alerts_total = 0
    m9_oversized_total = 0
    v9_oversized_total = 0

    while True:
        ret, frame = cap.read()
        if not ret or fn >= max_frames:
            break

        # 1. Model 9 Detection + Intelligence
        m9_raw = m9_detector.detect_frame(frame, min_threshold=0.30)
        m9_conf, m9_eval = m9_filter.process_frame(frame_bgr=frame, raw_detections=m9_raw, frame_idx=fn)

        # 2. Candidate V9 Detection + Intelligence
        v9_raw = v9_detector.detect_frame(frame, min_threshold=0.30)
        v9_conf, v9_eval = v9_filter.process_frame(frame_bgr=frame, raw_detections=v9_raw, frame_idx=fn)

        m9_alerts_total += len(m9_conf)
        v9_alerts_total += len(v9_conf)
        m9_oversized_total += sum(1 for r in m9_eval if "OVERSIZED" in str(r.get("rejection_reason") or "") or "GEOMETRIC" in str(r.get("validation_status") or ""))
        v9_oversized_total += sum(1 for r in v9_eval if "OVERSIZED" in str(r.get("rejection_reason") or "") or "GEOMETRIC" in str(r.get("validation_status") or ""))

        # Render Left Panel: Model 9 + Tier 2
        p1 = frame.copy()
        for r in m9_eval:
            p1 = draw_intelligence_box(p1, r, "Model 9")

        # Render Right Panel: Candidate V9 + Tier 2
        p2 = frame.copy()
        for r in v9_eval:
            p2 = draw_intelligence_box(p2, r, "Candidate V9")

        # Resize panels
        p1_res = cv2.resize(p1, (panel_w, panel_h))
        p2_res = cv2.resize(p2, (panel_w, panel_h))

        # Build Comparative Header Bar
        header = np.zeros((65, out_w, 3), dtype=np.uint8)
        header[:] = (25, 25, 25)

        # Left Header: Model 9
        m9_status_str = f"MODEL 9 BASELINE + TIER-2 ({cam_id})"
        m9_sub_str = f"Alerts: {len(m9_conf)} active | Oversized Hallucinations: {m9_oversized_total} cumulative"
        cv2.putText(header, m9_status_str, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 160, 255), 2)
        cv2.putText(header, m9_sub_str, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)

        # Right Header: Candidate V9
        v9_status_str = f"CANDIDATE MODEL V9 + TIER-2 ({cam_id})"
        v9_sub_str = f"Alerts: {len(v9_conf)} active | Oversized Hallucinations: {v9_oversized_total} (100% Clamped)"
        cv2.putText(header, v9_status_str, (panel_w + 20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 0), 2)
        cv2.putText(header, v9_sub_str, (panel_w + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 220, 170), 1)

        # Combine
        combined_vis = np.vstack([header, np.hstack([p1_res, p2_res])])
        writer.write(combined_vis)

        # Check if keyframe to extract
        if fn in keyframe_indices:
            frame_fname = f"M9_VS_V9_{scene_tag}_f{fn:04d}.jpg"
            img_out_path = OUT_IMG_DIR / frame_fname
            cv2.imwrite(str(img_out_path), combined_vis)
            print(f"  [+] Saved side-by-side keyframe: {frame_fname}")
            
            # Copy to artifact dir for UI display
            shutil.copy2(img_out_path, ARTIFACT_DIR / frame_fname)

        fn += 1

    cap.release()
    writer.release()
    print(f"  [+] Finished Video: {out_video_path.name} ({fn} frames, M9 alerts={m9_alerts_total}, V9 alerts={v9_alerts_total})")


def main():
    m9_path = ROOT / "mockup_ui" / "models" / "best_weapon_detector_ninth_model.pth"
    if not m9_path.exists():
        m9_path = ROOT / "best_weapon_detector_ninth_model.pth"
    v9_path = ROOT / "research" / "model_improvement" / "checkpoints" / "best_candidate_model_v9.pth"

    print("=" * 85)
    print("  RENDERING MULTI-CAMERA EVALUATION: MODEL 9 vs. CANDIDATE V9 + TIER-2 INTELLIGENCE  ")
    print("=" * 85)
    print(f"Model 9:      {m9_path}")
    print(f"Candidate V9: {v9_path}")
    print(f"Device:       CUDA (RTX 5060 Ti)")
    print("=" * 85)

    m9_detector = DetectionService(model_path=m9_path, confidence_threshold=0.30)
    v9_detector = DetectionService(model_path=v9_path, confidence_threshold=0.30)

    # Curated multi-camera target scenes representing diverse operational conditions:
    # 1. Scene 001 CAM-02: Door shadow & knife recall
    # 2. Scene 003 CAM-02: Fast handgun draw + ponytail head distractor
    # 3. Scene 004 CAM-01: Chef knife brandishing
    # 4. Scene 009 CAM-02: Extreme room-spanning oversized knife hallucination in Model 9 (59 instances)
    # 5. Scene 011 CAM-01: Shared multi-camera firearm brandishing
    target_evaluations = [
        {
            "scene_idx": 1,
            "cam_id": "CAM-02",
            "video": ROOT / "samples" / "NEW_STAGED_CAM-02" / "NEW_CAMERA 02" / "CAM2_SCENE001.mp4",
            "keyframes": [18, 45],
        },
        {
            "scene_idx": 3,
            "cam_id": "CAM-02",
            "video": ROOT / "samples" / "NEW_STAGED_CAM-02" / "NEW_CAMERA 02" / "CAM2_SCENE003.mp4",
            "keyframes": [45, 95],
        },
        {
            "scene_idx": 4,
            "cam_id": "CAM-01",
            "video": ROOT / "samples" / "NEW_STAGED_CAM-01" / "NEW_CAMERA 01" / "CAM1_SCENE004.mov",
            "keyframes": [40, 80],
        },
        {
            "scene_idx": 9,
            "cam_id": "CAM-02",
            "video": ROOT / "samples" / "NEW_STAGED_CAM-02" / "NEW_CAMERA 02" / "CAM2_SCENE009.mp4",
            "keyframes": [20, 55],
        },
        {
            "scene_idx": 11,
            "cam_id": "CAM-01",
            "video": ROOT / "samples" / "NEW_STAGED_CAM-01" / "NEW_CAMERA 01" / "CAM1_SCENE011.mov",
            "keyframes": [30, 60, 85],
        },
    ]

    for item in target_evaluations:
        render_scene_comparison(
            scene_idx=item["scene_idx"],
            cam_id=item["cam_id"],
            vid_path=item["video"],
            m9_detector=m9_detector,
            v9_detector=v9_detector,
            keyframe_indices=item["keyframes"],
            max_frames=150,
        )

    print("\n" + "=" * 85)
    print("ALL MULTI-CAMERA COMPARISON VIDEOS AND FRAMES SUCCESSFULLY GENERATED!")
    print(f"Videos: {OUT_VID_DIR}")
    print(f"Frames: {OUT_IMG_DIR}")
    print("=" * 85)


if __name__ == "__main__":
    main()
