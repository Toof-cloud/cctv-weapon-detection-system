"""
Comprehensive Benchmark: Candidate Model V6 vs. Model 9 Baseline
Tested across the 13 brand new CCTV clips in samples/NEW-VIDEOS:
- 7 Handgun clips
- 5 Knife clips
- 1 Rifle clip
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

NEW_VIDEOS_DIR = ROOT / "samples" / "NEW-VIDEOS"
OUT_DIR = ROOT / "research" / "model_improvement" / "evaluations" / "new_videos_outputs"
DIFF_DIR = ROOT / "research" / "model_improvement" / "visual_comparisons_new_videos"
SUMMARY_PATH = ROOT / "research" / "model_improvement" / "evaluations" / "v6_vs_m9_new_videos_summary.json"

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


def draw_detections(frame, detections, color, label_prefix=""):
    for det in detections:
        cls_name, score, (x1, y1, x2, y2) = det
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label_prefix}{cls_name}: {score:.2f}"
        cv2.putText(frame, text, (x1, max(y1 - 6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def process_clip(video_path: Path, m9, v6):
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    safe_name = video_path.stem.replace(" ", "_").replace("-", "_")
    annotated_path = OUT_DIR / f"{safe_name}_annotated.mp4"
    frames_dir = OUT_DIR / f"{safe_name}_detected_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (w, h))

    fidx = 0
    m9_total = 0
    v6_total = 0
    m9_by_class = {"Handgun": 0, "Knife": 0}
    v6_by_class = {"Handgun": 0, "Knife": 0}
    diff_records = []

    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(rgb).to(DEVICE)

        with torch.no_grad():
            out_m9 = m9([tensor])[0]
            out_v6 = v6([tensor])[0]

        def parse(out):
            dets = []
            for b, s, l in zip(out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(), out["labels"].cpu().numpy()):
                if s >= 0.50 and l in CLASS_MAP:
                    dets.append([CLASS_MAP[l], float(round(s, 3)), [int(b[0]), int(b[1]), int(b[2]), int(b[3])]])
            return dets

        dets_m9 = parse(out_m9)
        dets_v6 = parse(out_v6)

        m9_total += len(dets_m9)
        v6_total += len(dets_v6)

        for d in dets_m9:
            m9_by_class[d[0]] += 1
        for d in dets_v6:
            v6_by_class[d[0]] += 1

        annotated = frame.copy()
        draw_detections(annotated, dets_m9, (255, 120, 0), "M9 ")
        draw_detections(annotated, dets_v6, (0, 0, 255), "V6 ")
        writer.write(annotated)

        if len(dets_v6) > 0 and (fidx % 5 == 0 or len(dets_m9) == 0):
            cv2.imwrite(str(frames_dir / f"frame_{fidx:05d}_detected.jpg"), annotated)

        # Check for meaningful diffs (score gap > 0.15 or different detection counts)
        is_diff = (len(dets_m9) != len(dets_v6))
        if not is_diff and len(dets_m9) > 0 and len(dets_v6) > 0:
            if abs(dets_m9[0][1] - dets_v6[0][1]) > 0.15 or dets_m9[0][0] != dets_v6[0][0]:
                is_diff = True

        if is_diff and len(diff_records) < 6:
            diff_m9 = frame.copy()
            diff_v6 = frame.copy()
            draw_detections(diff_m9, dets_m9, (255, 120, 0), "M9 ")
            draw_detections(diff_v6, dets_v6, (0, 0, 255), "V6 ")
            cv2.putText(diff_m9, f"Model 9 ({len(dets_m9)} dets)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 120, 0), 2)
            cv2.putText(diff_v6, f"Candidate V6 ({len(dets_v6)} dets)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            side_by_side = np.hstack([diff_m9, diff_v6])
            diff_name = f"{safe_name}_frame_{fidx:04d}_diff.jpg"
            cv2.imwrite(str(DIFF_DIR / diff_name), side_by_side)
            diff_records.append({
                "frame": fidx,
                "image": diff_name,
                "m9": dets_m9,
                "v6": dets_v6
            })

        fidx += 1

    cap.release()
    writer.release()
    elapsed = time.time() - t0

    return {
        "video_name": video_path.name,
        "category": video_path.parent.name,
        "total_frames": fidx,
        "elapsed_seconds": round(elapsed, 1),
        "model9": {
            "total_detections": m9_total,
            "by_class": m9_by_class
        },
        "candidate_v6": {
            "total_detections": v6_total,
            "by_class": v6_by_class
        },
        "diff_count": len(diff_records),
        "diff_records": diff_records,
        "annotated_video": str(annotated_path)
    }


def main():
    print("=" * 80)
    print("   EVALUATING CANDIDATE V6 vs. MODEL 9 BASELINE ON NEW VIDEOS")
    print("=" * 80)
    print(f"Searching for video clips under: {NEW_VIDEOS_DIR}")
    video_files = sorted(list(NEW_VIDEOS_DIR.glob("**/*.mp4")))
    print(f"Found {len(video_files)} new video clips to evaluate.")

    m9 = load_model9()
    v6 = load_model_v6()
    print("Both models loaded successfully on CUDA!\n")

    summary_results = []

    for idx, vid in enumerate(video_files, 1):
        print(f"[{idx}/{len(video_files)}] Processing: {vid.name} (Category: {vid.parent.name})...")
        res = process_clip(vid, m9, v6)
        summary_results.append(res)
        m9_cnt = res["model9"]["total_detections"]
        v6_cnt = res["candidate_v6"]["total_detections"]
        diff = v6_cnt - m9_cnt
        print(f"   Done in {res['elapsed_seconds']}s | M9: {m9_cnt} | V6: {v6_cnt} (Diff: {diff:+d})")

    SUMMARY_PATH.write_text(json.dumps(summary_results, indent=2))
    print("\n" + "=" * 80)
    print("BENCHMARK ON NEW VIDEOS COMPLETED!")
    print(f"Summary JSON saved to: {SUMMARY_PATH}")
    print(f"Visual Diffs saved to: {DIFF_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
