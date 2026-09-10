"""
Automated Comprehensive Evaluation Pipeline for Candidate Model V5.
Evaluates Candidate V5 across:
1. The 4 Standard Benchmark Samples (CAM02, NEW_KNIFE, evaluation_video, handgun_test-video)
2. The 10 Unseen CCTV Clips (Clips 01-10)
3. Generates side-by-side comparisons, detected frame extractions, and 1-by-1 audit ledger.
"""
import os
import sys
import time
import math
import json
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.model_improvement.training.model_builder import build_research_model
from dataset_analysis.build_model import get_model as get_model9

MODEL9_PATH = ROOT / "best_weapon_detector_ninth_model.pth"
CANDIDATE_V5_PATH = ROOT / "research" / "model_improvement" / "checkpoints" / "best_candidate_model_v5.pth"

SAMPLES_DIR = ROOT / "samples"
UNSEEN_DIR = ROOT / "samples" / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS"

OUT_DIR_4SAMPLES = ROOT / "research" / "model_improvement" / "evaluations" / "v5_4samples_outputs"
OUT_DIR_10CLIPS = ROOT / "research" / "model_improvement" / "evaluations" / "v5_10clips_outputs"
VISUAL_DIFFS_DIR = ROOT / "research" / "model_improvement" / "visual_comparisons_v5"

OUT_DIR_4SAMPLES.mkdir(parents=True, exist_ok=True)
OUT_DIR_10CLIPS.mkdir(parents=True, exist_ok=True)
VISUAL_DIFFS_DIR.mkdir(parents=True, exist_ok=True)

FOUR_SAMPLE_JOBS = [
    {"name": "Handgun_Staged_CAM02", "path": SAMPLES_DIR / "CAM02_Scene_004.mp4"},
    {"name": "Knife_CCTV_NEW_KNIFE", "path": SAMPLES_DIR / "NEW_KNIFE_VIDEO_11s.mp4"},
    {"name": "Knife_Evaluation_Video", "path": SAMPLES_DIR / "evaluation_video.mp4"},
    {"name": "Handgun_Normal_Video", "path": SAMPLES_DIR / "handgun_test-video.mp4"},
]

NAMES = {1: "Handgun", 2: "Knife"}
COLORS = {1: (0, 0, 255), 2: (0, 165, 255)}


def load_models(device="cuda"):
    print("Loading Model 9 baseline and Candidate Model V5...")
    # Model 9
    m9 = get_model9(num_classes=3, small_anchors=False, anchor_scales=[16, 32, 64, 128, 256])
    ckpt9 = torch.load(MODEL9_PATH, map_location=device)
    m9_state = ckpt9["model_state_dict"] if isinstance(ckpt9, dict) and "model_state_dict" in ckpt9 else ckpt9
    m9.load_state_dict(m9_state)
    m9.to(device).eval()

    # Candidate V5
    ckpt5 = torch.load(CANDIDATE_V5_PATH, map_location=device)
    sizes = ckpt5.get("anchor_sizes", ((12,), (20,), (36,), (64,), (96,)))
    ratios = ckpt5.get("aspect_ratios") or ckpt5.get("anchor_aspect_ratios", (0.5, 0.7, 1.0, 1.4, 2.0))
    clip = ckpt5.get("bbox_xform_clip", math.log(2.5))
    m5 = build_research_model(num_classes=3, anchor_sizes=sizes, aspect_ratios=ratios, pretrained_backbone=False, bbox_xform_clip=clip)
    m5_state = ckpt5["model_state_dict"] if "model_state_dict" in ckpt5 else ckpt5
    m5.load_state_dict(m5_state)
    m5.to(device).eval()

    return m9, m5


def draw_header_and_boxes(frame, boxes, scores, labels, model_title):
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (15, 15, 20), -1)
    cv2.putText(out, model_title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for b, s, l in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [int(v) for v in b]
        c = COLORS.get(l, (0, 255, 0))
        tag = f"{NAMES.get(l, l)} {s:.1%}"
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        tw = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] + 8
        cv2.rectangle(out, (x1, max(0, y1 - 20)), (x1 + tw, max(20, y1)), c, -1)
        cv2.putText(out, tag, (x1 + 4, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def run_video_inference(vpath, m9, m5, out_dir, diff_dir, conf_thresh=0.50, device="cuda"):
    cap = cv2.VideoCapture(str(vpath))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    vname_clean = vpath.stem.replace(" ", "_")
    annotated_out_path = out_dir / f"{vname_clean}_v5_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(annotated_out_path), fourcc, fps, (w, h))

    frames_dir = out_dir / f"{vname_clean}_v5_detected_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    m9_detections_total = 0
    m5_detections_total = 0
    diff_records = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose((2, 0, 1))).float().div(255.0).unsqueeze(0).to(device)

        with torch.no_grad():
            o9 = m9(tensor)[0]
            o5 = m5(tensor)[0]

        b9 = [b for b, s in zip(o9["boxes"].cpu().numpy(), o9["scores"].cpu().numpy()) if s >= conf_thresh]
        s9 = [s for s in o9["scores"].cpu().numpy() if s >= conf_thresh]
        l9 = [l for l, s in zip(o9["labels"].cpu().numpy(), o9["scores"].cpu().numpy()) if s >= conf_thresh]

        b5 = [b for b, s in zip(o5["boxes"].cpu().numpy(), o5["scores"].cpu().numpy()) if s >= conf_thresh]
        s5 = [s for s in o5["scores"].cpu().numpy() if s >= conf_thresh]
        l5 = [l for l, s in zip(o5["labels"].cpu().numpy(), o5["scores"].cpu().numpy()) if s >= conf_thresh]

        m9_detections_total += len(b9)
        m5_detections_total += len(b5)

        # Draw Candidate V5 detections for annotated output video
        annotated_frame = draw_header_and_boxes(frame, b5, s5, l5, "CANDIDATE MODEL V5")
        writer.write(annotated_frame)

        if len(b5) > 0 and frame_idx % 5 == 0:
            frame_save_path = frames_dir / f"frame_{frame_idx:05d}_v5_detected.jpg"
            cv2.imwrite(str(frame_save_path), annotated_frame)

        # Check for meaningful difference between M9 and V5
        has_diff = False
        if len(b9) != len(b5):
            has_diff = True
        elif len(b9) > 0 and len(b5) > 0:
            if abs(max(s5) - max(s9)) > 0.15:
                has_diff = True

        if has_diff and len(diff_records) < 6:
            f9_ren = draw_header_and_boxes(frame, b9, s9, l9, "MODEL 9 SOTA (BASELINE)")
            f5_ren = draw_header_and_boxes(frame, b5, s5, l5, "CANDIDATE MODEL V5")
            side_by_side = np.hstack([f9_ren, f5_ren])
            diff_img_name = f"{vname_clean}_frame_{frame_idx:04d}_diff.jpg"
            cv2.imwrite(str(diff_dir / diff_img_name), side_by_side)
            diff_records.append({
                "frame": frame_idx,
                "image": diff_img_name,
                "m9": [(NAMES.get(l, l), float(f"{s:.3f}"), [int(v) for v in b]) for b, s, l in zip(b9, s9, l9)],
                "v5": [(NAMES.get(l, l), float(f"{s:.3f}"), [int(v) for v in b]) for b, s, l in zip(b5, s5, l5)],
            })

        frame_idx += 1

    cap.release()
    writer.release()

    return {
        "video_name": vpath.name,
        "total_frames": total_frames,
        "model9_total_detections": m9_detections_total,
        "candidate_v5_total_detections": m5_detections_total,
        "diff_count": len(diff_records),
        "diff_records": diff_records,
        "annotated_video": str(annotated_out_path),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not CANDIDATE_V5_PATH.exists():
        print(f"[!] Checkpoint {CANDIDATE_V5_PATH} not found. Awaiting training completion.")
        return

    m9, m5 = load_models(device)

    # 1. Evaluate on 4 Standard Samples
    print("\n" + "=" * 80)
    print("   RUNNING CANDIDATE V5 ON 4 STANDARD MANUSCRIPT SAMPLES")
    print("=" * 80)
    four_samples_results = []
    for item in FOUR_SAMPLE_JOBS:
        vpath = item["path"]
        if not vpath.exists():
            print(f"[!] Missing: {vpath}")
            continue
        print(f"-> Processing: {item['name']} ({vpath.name})...")
        t0 = time.time()
        res = run_video_inference(vpath, m9, m5, OUT_DIR_4SAMPLES, VISUAL_DIFFS_DIR, device=device)
        print(f"   Done in {time.time()-t0:.1f}s | M9 Detections: {res['model9_total_detections']} | V5 Detections: {res['candidate_v5_total_detections']}")
        four_samples_results.append(res)

    out_4s_json = ROOT / "research" / "model_improvement" / "evaluations" / "v5_4samples_summary.json"
    out_4s_json.write_text(json.dumps(four_samples_results, indent=2))

    # 2. Evaluate on 10 Unseen CCTV Clips
    print("\n" + "=" * 80)
    print("   RUNNING CANDIDATE V5 ON 10 UNSEEN CCTV CLIPS")
    print("=" * 80)
    videos = sorted(list(UNSEEN_DIR.glob("*.mp4")))
    ten_clips_results = []
    for idx, vpath in enumerate(videos, start=1):
        print(f"[{idx}/{len(videos)}] Processing Unseen Clip: {vpath.name}...")
        t0 = time.time()
        res = run_video_inference(vpath, m9, m5, OUT_DIR_10CLIPS, VISUAL_DIFFS_DIR, device=device)
        print(f"   Done in {time.time()-t0:.1f}s | M9 Detections: {res['model9_total_detections']} | V5 Detections: {res['candidate_v5_total_detections']}")
        ten_clips_results.append(res)

    out_10c_json = ROOT / "research" / "model_improvement" / "evaluations" / "v5_10clips_summary.json"
    out_10c_json.write_text(json.dumps(ten_clips_results, indent=2))

    print("\n" + "=" * 80)
    print("ALL CANDIDATE V5 EVALUATIONS COMPLETED!")
    print(f"4 Samples Summary: {out_4s_json}")
    print(f"10 Clips Summary:  {out_10c_json}")
    print(f"Visual Diffs Dir:  {VISUAL_DIFFS_DIR}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
