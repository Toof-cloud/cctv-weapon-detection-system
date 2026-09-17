"""
Candidate Model V6 Benchmark Pipeline:
1. Runs Candidate V6 vs Model 9 across the 4 standard manuscript benchmark samples.
2. Runs Candidate V6 vs Model 9 across the 10 unseen CCTV robbery clips.
3. Saves full frame-by-frame JSON summaries and side-by-side diff frames.
"""
import os
import sys
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from torchvision.transforms import functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.model_improvement.training.model_builder import build_research_model
from dataset_analysis.build_model import get_model as get_model9

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M9_CHECKPOINT = ROOT / "best_weapon_detector_ninth_model.pth"
V6_CHECKPOINT = ROOT / "research" / "model_improvement" / "checkpoints" / "best_candidate_model_v6.pth"

SAMPLES_4 = [
    {"name": "Handgun_Staged_CAM02", "path": ROOT / "samples" / "CAM02_Scene_004.mp4"},
    {"name": "Knife_CCTV_NEW_KNIFE", "path": ROOT / "samples" / "NEW_KNIFE_VIDEO_11s.mp4"},
    {"name": "Knife_Evaluation_Video", "path": ROOT / "samples" / "evaluation_video.mp4"},
    {"name": "Handgun_Normal_Video", "path": ROOT / "samples" / "handgun_test-video.mp4"},
]

UNSEEN_DIR = ROOT / "samples" / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS"

OUT_DIR_4 = ROOT / "research" / "model_improvement" / "evaluations" / "v6_4samples_outputs"
OUT_DIR_10 = ROOT / "research" / "model_improvement" / "evaluations" / "v6_10clips_outputs"
DIFF_DIR = ROOT / "research" / "model_improvement" / "visual_comparisons_v6"

OUT_DIR_4.mkdir(parents=True, exist_ok=True)
OUT_DIR_10.mkdir(parents=True, exist_ok=True)
DIFF_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {1: "Handgun", 2: "Knife"}


def load_model9(ckpt_path):
    model = get_model9(num_classes=3, small_anchors=False, anchor_scales=[16, 32, 64, 128, 256])
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def load_model_v6(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    anchors = ckpt.get("anchor_sizes", ((12,), (20,), (36,), (64,), (96,)))
    ratios = ckpt.get("aspect_ratios") or ckpt.get("anchor_aspect_ratios", (0.5, 0.7, 1.0, 1.4, 2.0))
    clip_val = ckpt.get("bbox_xform_clip", 1.163)
    model = build_research_model(num_classes=3, anchor_sizes=anchors, aspect_ratios=ratios, bbox_xform_clip=clip_val, pretrained_backbone=False)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def draw_detections(frame, detections, color, label_prefix=""):
    for det in detections:
        cls_name, score, (x1, y1, x2, y2) = det
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label_prefix}{cls_name}: {score:.2f}"
        cv2.putText(frame, text, (x1, max(y1 - 6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def run_video_comparison(video_path, m9_model, v6_model, out_dir, prefix=""):
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    base_name = video_path.stem.replace(" ", "_")
    out_video = out_dir / f"{base_name}_v6_annotated.mp4"
    frames_dir = out_dir / f"{base_name}_v6_detected_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))

    fidx = 0
    m9_total = 0
    v6_total = 0
    diff_records = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(rgb).to(DEVICE)

        with torch.no_grad():
            out_m9 = m9_model([tensor])[0]
            out_v6 = v6_model([tensor])[0]

        def parse_out(out):
            dets = []
            for b, s, l in zip(out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(), out["labels"].cpu().numpy()):
                if s >= 0.50 and l in CLASS_MAP:
                    dets.append([CLASS_MAP[l], float(round(s, 3)), [int(b[0]), int(b[1]), int(b[2]), int(b[3])]])
            return dets

        dets_m9 = parse_out(out_m9)
        dets_v6 = parse_out(out_v6)

        m9_total += len(dets_m9)
        v6_total += len(dets_v6)

        annotated = frame.copy()
        draw_detections(annotated, dets_v6, (0, 0, 255), "V6 ")
        writer.write(annotated)

        if len(dets_v6) > 0 and (fidx % 5 == 0 or len(dets_m9) == 0):
            cv2.imwrite(str(frames_dir / f"frame_{fidx:05d}_v6_detected.jpg"), annotated)

        if (len(dets_m9) != len(dets_v6)) or (len(dets_m9) > 0 and len(dets_v6) > 0 and abs(dets_m9[0][1] - dets_v6[0][1]) > 0.15):
            if len(diff_records) < 6:
                diff_frame = frame.copy()
                diff_m9 = frame.copy()
                diff_v6 = frame.copy()
                draw_detections(diff_m9, dets_m9, (255, 120, 0), "M9 ")
                draw_detections(diff_v6, dets_v6, (0, 0, 255), "V6 ")
                cv2.putText(diff_m9, f"Model 9 ({len(dets_m9)} dets)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 120, 0), 2)
                cv2.putText(diff_v6, f"Candidate V6 ({len(dets_v6)} dets)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                side_by_side = np.hstack([diff_m9, diff_v6])
                diff_img_name = f"{base_name}_frame_{fidx:04d}_diff.jpg"
                cv2.imwrite(str(DIFF_DIR / diff_img_name), side_by_side)
                diff_records.append({
                    "frame": fidx,
                    "image": diff_img_name,
                    "m9": dets_m9,
                    "v6": dets_v6
                })

        fidx += 1

    cap.release()
    writer.release()

    return {
        "video_name": video_path.name,
        "total_frames": fidx,
        "model9_total_detections": m9_total,
        "candidate_v6_total_detections": v6_total,
        "diff_count": len(diff_records),
        "diff_records": diff_records,
        "annotated_video": str(out_video)
    }


def main():
    print("Loading Model 9 baseline and Candidate Model V6...")
    m9 = load_model9(M9_CHECKPOINT)
    v6 = load_model_v6(V6_CHECKPOINT)

    print("\n" + "=" * 80)
    print("   RUNNING CANDIDATE V6 ON 4 STANDARD MANUSCRIPT SAMPLES")
    print("=" * 80)
    results_4 = []
    for sample in SAMPLES_4:
        print(f"-> Processing: {sample['name']} ({sample['path'].name})...")
        res = run_video_comparison(sample["path"], m9, v6, OUT_DIR_4)
        results_4.append(res)
        print(f"   Done | M9 Detections: {res['model9_total_detections']} | V6 Detections: {res['candidate_v6_total_detections']}")

    summary_4_path = ROOT / "research" / "model_improvement" / "evaluations" / "v6_4samples_summary.json"
    summary_4_path.write_text(json.dumps(results_4, indent=2))

    print("\n" + "=" * 80)
    print("   RUNNING CANDIDATE V6 ON 10 UNSEEN CCTV CLIPS")
    print("=" * 80)
    unseen_clips = sorted(list(UNSEEN_DIR.glob("*.mp4")))
    results_10 = []
    for i, clip in enumerate(unseen_clips, 1):
        print(f"[{i}/10] Processing Unseen Clip: {clip.name}...")
        res = run_video_comparison(clip, m9, v6, OUT_DIR_10)
        results_10.append(res)
        print(f"   Done | M9 Detections: {res['model9_total_detections']} | V6 Detections: {res['candidate_v6_total_detections']}")

    summary_10_path = ROOT / "research" / "model_improvement" / "evaluations" / "v6_10clips_summary.json"
    summary_10_path.write_text(json.dumps(results_10, indent=2))

    print("\n" + "=" * 80)
    print("ALL CANDIDATE V6 EVALUATIONS COMPLETED!")
    print(f"4 Samples Summary: {summary_4_path}")
    print(f"10 Clips Summary:  {summary_10_path}")
    print(f"Visual Diffs Dir:  {DIFF_DIR}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
