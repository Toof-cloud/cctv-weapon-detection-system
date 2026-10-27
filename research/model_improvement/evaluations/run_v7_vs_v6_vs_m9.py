"""
Comprehensive 3-Way Benchmark & 1-by-1 Frame Audit:
Model 9 Baseline vs. Candidate Model V6 vs. Candidate Model V7

Evaluates:
1. 13 New Video Clips in samples/NEW-VIDEOS (Handgun, Knife, Rifle)
2. 10 Core Thesis Surveillance Clips (Clips 01 - 10)
Generates:
- Side-by-side visual diff images in visual_comparisons_v7/
- Full JSON summary and per-clip statistics
"""
import os
import sys
import json
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.model_improvement.training.model_builder import build_research_model
from dataset_analysis.build_model import get_model as get_model9

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL9_PATH = ROOT / "best_weapon_detector_ninth_model.pth"
CANDIDATE_V6_PATH = ROOT / "research" / "model_improvement" / "checkpoints" / "best_candidate_model_v6.pth"
CANDIDATE_V7_PATH = ROOT / "research" / "model_improvement" / "checkpoints" / "best_candidate_model_v7.pth"

NEW_VIDEOS_DIR = ROOT / "samples" / "NEW-VIDEOS"
OUT_DIR = ROOT / "research" / "model_improvement" / "evaluations" / "v7_evaluation_outputs"
DIFF_DIR = ROOT / "research" / "model_improvement" / "visual_comparisons_v7"
SUMMARY_PATH = ROOT / "research" / "model_improvement" / "evaluations" / "v7_vs_v6_vs_m9_summary.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)
DIFF_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {1: "Handgun", 2: "Knife"}


def load_model9():
    m9 = get_model9(num_classes=3, small_anchors=False, anchor_scales=[16, 32, 64, 128, 256])
    ckpt9 = torch.load(MODEL9_PATH, map_location=DEVICE)
    m9_state = ckpt9["model_state_dict"] if isinstance(ckpt9, dict) and "model_state_dict" in ckpt9 else ckpt9
    m9.load_state_dict(m9_state)
    m9.to(DEVICE).eval()
    return m9


def load_model_v6():
    ckpt6 = torch.load(CANDIDATE_V6_PATH, map_location=DEVICE)
    sizes = ckpt6.get("anchor_sizes", ((12,), (20,), (36,), (64,), (96,)))
    ratios = ckpt6.get("aspect_ratios") or ckpt6.get("anchor_aspect_ratios", (0.5, 0.7, 1.0, 1.4, 2.0))
    clip_val = ckpt6.get("bbox_xform_clip", 1.163)
    v6 = build_research_model(num_classes=3, anchor_sizes=sizes, aspect_ratios=ratios, pretrained_backbone=False, bbox_xform_clip=clip_val)
    v6_state = ckpt6["model_state_dict"] if "model_state_dict" in ckpt6 else ckpt6
    v6.load_state_dict(v6_state)
    v6.to(DEVICE).eval()
    return v6


def load_model_v7():
    ckpt7 = torch.load(CANDIDATE_V7_PATH, map_location=DEVICE)
    sizes = ckpt7.get("anchor_sizes", ((18,), (36,), (72,), (144,), (280,)))
    ratios = ckpt7.get("aspect_ratios", (0.4, 0.7, 1.0, 1.5, 2.5))
    clip_val = ckpt7.get("bbox_xform_clip", 0.69315)
    v7 = build_research_model(num_classes=3, anchor_sizes=sizes, aspect_ratios=ratios, pretrained_backbone=False, bbox_xform_clip=clip_val)
    v7_state = ckpt7["model_state_dict"] if "model_state_dict" in ckpt7 else ckpt7
    v7.load_state_dict(v7_state)
    v7.to(DEVICE).eval()
    return v7


def run_inference(model, frame_tensor, threshold=0.50):
    with torch.no_grad():
        preds = model(frame_tensor)[0]
    boxes = preds["boxes"].cpu().numpy()
    labels = preds["labels"].cpu().numpy()
    scores = preds["scores"].cpu().numpy()

    detections = []
    for b, l, s in zip(boxes, labels, scores):
        if s >= threshold and l in CLASS_MAP:
            x1, y1, x2, y2 = [int(v) for v in b]
            detections.append((CLASS_MAP[l], float(s), (x1, y1, x2, y2)))
    return detections


def render_comparison_card(frame, m9_dets, v6_dets, v7_dets, frame_num, clip_name):
    # Render three side-by-side or stacked panels
    canvas = frame.copy()
    
    # Model 9 detections (Yellow / Orange)
    for cls_name, score, (x1, y1, x2, y2) in m9_dets:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 215, 255), 2)
        cv2.putText(canvas, f"M9 {cls_name}: {score:.2f}", (x1, max(y1 - 6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 2)

    # Candidate V6 detections (Cyan / Blue)
    for cls_name, score, (x1, y1, x2, y2) in v6_dets:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(canvas, f"V6 {cls_name}: {score:.2f}", (x1, max(y1 - 22, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

    # Candidate V7 detections (Green)
    for cls_name, score, (x1, y1, x2, y2) in v7_dets:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(canvas, f"V7 {cls_name}: {score:.2f}", (x1, min(y2 + 18, canvas.shape[0] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Top header bar
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 36), (20, 20, 20), -1)
    header_text = f"{clip_name} | Frame {frame_num:04d} | M9={len(m9_dets)} (Yellow), V6={len(v6_dets)} (Cyan), V7={len(v7_dets)} (Green)"
    cv2.putText(canvas, header_text, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return canvas


def evaluate_video(video_path: Path, m9, v6, v7, threshold=0.50):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_name = video_path.stem

    m9_count = 0
    v6_count = 0
    v7_count = 0
    m9_classes = {"Handgun": 0, "Knife": 0}
    v6_classes = {"Handgun": 0, "Knife": 0}
    v7_classes = {"Handgun": 0, "Knife": 0}
    diff_images = []

    fidx = 0
    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = [F.to_tensor(rgb).to(DEVICE)]

        m9_d = run_inference(m9, tensor, threshold)
        v6_d = run_inference(v6, tensor, threshold)
        v7_d = run_inference(v7, tensor, threshold)

        m9_count += len(m9_d)
        v6_count += len(v6_d)
        v7_count += len(v7_d)

        for c, _, _ in m9_d:
            m9_classes[c] += 1
        for c, _, _ in v6_d:
            v6_classes[c] += 1
        for c, _, _ in v7_d:
            v7_classes[c] += 1

        # Check if there is a meaningful disagreement
        has_diff = (len(v7_d) != len(v6_d)) or (len(v7_d) != len(m9_d))
        if has_diff and len(diff_images) < 15:
            safe_name = clip_name.replace(" ", "_").replace("-", "_")
            diff_filename = f"{safe_name}_f{fidx:04d}_comp.jpg"
            diff_card = render_comparison_card(frame, m9_d, v6_d, v7_d, fidx, clip_name)
            diff_path = DIFF_DIR / diff_filename
            cv2.imwrite(str(diff_path), diff_card)
            diff_images.append({
                "frame": fidx,
                "image": diff_filename,
                "m9": m9_d,
                "v6": v6_d,
                "v7": v7_d
            })

        fidx += 1

    cap.release()
    elapsed = round(time.time() - t0, 1)

    return {
        "video_name": video_path.name,
        "total_frames": total_frames,
        "elapsed_seconds": elapsed,
        "model9": {"total": m9_count, "classes": m9_classes},
        "candidate_v6": {"total": v6_count, "classes": v6_classes},
        "candidate_v7": {"total": v7_count, "classes": v7_classes},
        "diff_count": len(diff_images),
        "diff_samples": diff_images
    }


def main():
    print("=" * 80)
    print("   RUNNING CANDIDATE MODEL V7 COMPREHENSIVE BENCHMARK")
    print("   MODEL 9 vs. CANDIDATE V6 vs. CANDIDATE V7")
    print("=" * 80)

    print("Loading models onto GPU...")
    m9 = load_model9()
    v6 = load_model_v6()
    v7 = load_model_v7()
    print("Models loaded successfully!")

    # Gather video clips across all test suites
    suites = []
    
    # Suite 1: 13 New Video Clips
    new_clips = sorted(list(NEW_VIDEOS_DIR.rglob("*.mp4")))
    for c in new_clips:
        suites.append(("New Sample Clips", c))

    # Suite 2: 10 Thesis Unseen Surveillance Clips
    unseen_dir = ROOT / "samples" / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS"
    if unseen_dir.exists():
        for c in sorted(list(unseen_dir.glob("*.mp4"))):
            suites.append(("10 Thesis Clips", c))

    # Suite 3: 4 Core Benchmark Samples
    for sample in [
        ROOT / "samples" / "CAM02_Scene_004.mp4",
        ROOT / "samples" / "NEW_KNIFE_VIDEO_11s.mp4",
        ROOT / "samples" / "evaluation_video.mp4",
        ROOT / "samples" / "handgun_test-video.mp4",
    ]:
        if sample.exists():
            suites.append(("4 Core Samples", sample))

    print(f"Total video streams to evaluate: {len(suites)}")

    results = []
    for idx, (group, clip) in enumerate(suites):
        print(f"\n[{idx+1}/{len(suites)}] [{group}] Evaluating: {clip.name}...")
        res = evaluate_video(clip, m9, v6, v7)
        res["group"] = group
        results.append(res)
        print(f"  Summary: M9={res['model9']['total']} | V6={res['candidate_v6']['total']} | V7={res['candidate_v7']['total']} (Elapsed: {res['elapsed_seconds']}s)")

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Benchmark completed! Full summary saved at: {SUMMARY_PATH}")
    print(f"Comparison images saved in: {DIFF_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
