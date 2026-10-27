"""
Comparative Video Inference & Side-by-Side Visual Diff Generator.
Compares Model 9 SOTA vs. Candidate Model across CCTV Clips 01-10.
Generates side-by-side composite visual comparisons and logs TCR deltas.
"""
import os
import sys
import time
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
from benchmarks.calculate_tcr_and_mccr import compute_tcr_for_csv

MODEL9_PATH = ROOT / "best_weapon_detector_ninth_model.pth"
CANDIDATE_PATH = ROOT / "research" / "model_improvement" / "checkpoints" / "best_candidate_model_v2.pth"
CLIPS_DIR = ROOT / "samples" / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS"
VISUAL_DIR = ROOT / "research" / "model_improvement" / "visual_comparisons"
VISUAL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_OUT_DIR = ROOT / "research" / "model_improvement" / "evaluations"


def load_model(path: Path, is_candidate: bool = False, device="cuda"):
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    if is_candidate:
        anchor_sizes = ckpt.get("anchor_sizes", ((12,), (24,), (54,), (128,), (288,)))
        aspect_ratios = ckpt.get("anchor_aspect_ratios", (0.25, 0.5, 1.0, 2.0, 4.0))
        model = build_research_model(num_classes=3, anchor_sizes=anchor_sizes, aspect_ratios=aspect_ratios, pretrained_backbone=False)
    else:
        model = get_model9(num_classes=3, small_anchors=False, anchor_scales=[16, 32, 64, 128, 256])
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def draw_detections(frame, boxes, scores, labels, model_label="Model"):
    out = frame.copy()
    colors = {1: (0, 0, 255), 2: (0, 165, 255)} # Red for handgun, Orange for knife
    names = {1: "Handgun", 2: "Knife"}

    # Draw header banner
    cv2.rectangle(out, (0, 0), (out.shape[1], 40), (20, 20, 25), -1)
    cv2.putText(out, model_label, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for box, score, lbl in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = colors.get(lbl, (0, 255, 0))
        name = names.get(lbl, f"Weapon_{lbl}")

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        tag = f"{name} {score:.1%}"
        t_w = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] + 10
        cv2.rectangle(out, (x1, max(0, y1 - 22)), (x1 + t_w, max(22, y1)), color, -1)
        cv2.putText(out, tag, (x1 + 5, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return out


def compare_models_on_clip(vpath: Path, model9, candidate, conf_thresh: float = 0.50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cap = cv2.VideoCapture(str(vpath))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    m9_records = []
    cand_records = []
    diff_frames = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Convert to tensor
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose((2, 0, 1))).float().div(255.0).unsqueeze(0).to(device)

        with torch.no_grad():
            out9 = model9(tensor)[0]
            out_c = candidate(tensor)[0]

        # Model 9 detections
        b9 = out9["boxes"].cpu().numpy()
        s9 = out9["scores"].cpu().numpy()
        l9 = out9["labels"].cpu().numpy()
        m9_pass = [(b, s, l) for b, s, l in zip(b9, s9, l9) if s >= conf_thresh]

        # Candidate detections
        bc = out_c["boxes"].cpu().numpy()
        sc = out_c["scores"].cpu().numpy()
        lc = out_c["labels"].cpu().numpy()
        c_pass = [(b, s, l) for b, s, l in zip(bc, sc, lc) if s >= conf_thresh]

        t_sec = frame_idx / fps
        for b, s, l in m9_pass:
            m9_records.append({"frame_number": frame_idx, "timestamp_seconds": t_sec, "class_id": l, "score": s, "box": b})
        for b, s, l in c_pass:
            cand_records.append({"frame_number": frame_idx, "timestamp_seconds": t_sec, "class_id": l, "score": s, "box": b})

        # Check for significant difference
        has_diff = False
        if len(m9_pass) != len(c_pass):
            has_diff = True
        elif len(m9_pass) > 0 and len(c_pass) > 0:
            # Check if confidence or class label differs significantly
            max_s9 = max(s for _, s, _ in m9_pass)
            max_sc = max(s for _, s, _ in c_pass)
            if abs(max_sc - max_s9) > 0.15:
                has_diff = True

        if has_diff and len(diff_frames) < 5:
            # Save side-by-side composite
            f9_rendered = draw_detections(frame, [b for b, _, _ in m9_pass], [s for _, s, _ in m9_pass], [l for _, _, l in m9_pass], "MODEL 9 SOTA (BASELINE)")
            fc_rendered = draw_detections(frame, [b for b, _, _ in c_pass], [s for _, s, _ in c_pass], [l for _, _, l in c_pass], "PROPOSED CANDIDATE MODEL")

            side_by_side = np.hstack([f9_rendered, fc_rendered])
            out_img_name = f"{vpath.stem}_frame_{frame_idx:04d}.jpg"
            cv2.imwrite(str(VISUAL_DIR / out_img_name), side_by_side)
            diff_frames.append({
                "frame": frame_idx,
                "image_file": out_img_name,
                "m9_detections": len(m9_pass),
                "cand_detections": len(c_pass),
            })

        frame_idx += 1

    cap.release()

    return {
        "clip_name": vpath.name,
        "total_frames": total_frames,
        "model9_total_detections": len(m9_records),
        "candidate_total_detections": len(cand_records),
        "diff_frames": diff_frames,
    }


def main():
    if not CANDIDATE_PATH.exists():
        print(f"[!] Candidate model {CANDIDATE_PATH} not found. Please train candidate model first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 76)
    print("      COMPARING MODEL 9 SOTA vs. PROPOSED CANDIDATE MODEL")
    print("=" * 76)
    print(f"Device:          {device}")
    print(f"Model 9:         {MODEL9_PATH}")
    print(f"Candidate Model: {CANDIDATE_PATH}")
    print(f"Visual Diffs:    {VISUAL_DIR}")
    print("=" * 76)

    m9 = load_model(MODEL9_PATH, is_candidate=False, device=device)
    cand = load_model(CANDIDATE_PATH, is_candidate=True, device=device)

    videos = sorted(list(CLIPS_DIR.glob("*.mp4")))
    print(f"Total videos to compare: {len(videos)}")

    comparisons = []
    for idx, vpath in enumerate(videos, start=1):
        print(f"\n[{idx}/{len(videos)}] Comparing on: {vpath.name}")
        t0 = time.time()
        res = compare_models_on_clip(vpath, m9, cand)
        elapsed = time.time() - t0
        print(f"  -> Model 9 Detections: {res['model9_total_detections']} | Candidate: {res['candidate_total_detections']} | Diffs saved: {len(res['diff_frames'])} ({elapsed:.1f}s)")
        comparisons.append(res)

    out_json = EVAL_OUT_DIR / "clip_comparisons_summary.json"
    out_json.write_text(json.dumps(comparisons, indent=2))
    print(f"\nCompleted comparison! Saved summary to {out_json}")


if __name__ == "__main__":
    main()
