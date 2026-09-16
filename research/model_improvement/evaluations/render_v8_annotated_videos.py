"""
Render Full Annotated MP4 Videos for Candidate Model V8
Showcases:
1. Screen Recording 2026-09-10 215511-KNIFE.mp4 (door post & counter elimination)
2. Screen Recording 2026-09-07 225703.mp4 (Thesis Clip 03 - close robbery handgun recall)
3. Screen Recording 2026-09-07 230047.mp4 (Thesis Clip 04 - slashing knife dynamics recall)
4. Screen Recording 2026-09-10 213340 -HANDGUN.mp4 (handgun recall & floor mat rejection)
5. Screen Recording 2026-09-10 214028-Riffle.mp4 (TV lower-third banner rejection)

Enforces:
- CPU thread cap: 8 threads
- Zero git activity
"""
import os
import sys
import math
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import cv2
import numpy as np
import torch
torch.set_num_threads(8)

from torchvision.transforms import functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.model_improvement.training.model_builder import build_research_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CANDIDATE_V8_PATH = ROOT / "research" / "model_improvement" / "checkpoints" / "best_candidate_model_v8.pth"
OUT_DIR = ROOT / "research" / "model_improvement" / "annotated_videos_v8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {1: "Handgun", 2: "Knife"}
COLOR_MAP = {
    "Handgun": (0, 165, 255),   # Bright Orange in BGR
    "Knife": (0, 255, 0),        # Bright Green in BGR
}


def load_v8():
    ckpt8 = torch.load(CANDIDATE_V8_PATH, map_location=DEVICE)
    sizes = ckpt8.get("anchor_sizes", ((16,), (32,), (64,), (128,), (192,)))
    ratios = ckpt8.get("aspect_ratios", (0.5, 0.75, 1.0, 1.5, 2.0))
    rpn_clip = ckpt8.get("rpn_bbox_xform_clip", math.log(1.4))
    roi_clip = ckpt8.get("roi_bbox_xform_clip", math.log(1.3))
    v8 = build_research_model(
        num_classes=3, anchor_sizes=sizes, aspect_ratios=ratios,
        pretrained_backbone=False,
        rpn_bbox_xform_clip=rpn_clip,
        roi_bbox_xform_clip=roi_clip
    )
    v8_state = ckpt8["model_state_dict"] if "model_state_dict" in ckpt8 else ckpt8
    v8.load_state_dict(v8_state)
    v8.to(DEVICE).eval()
    return v8


def render_video(vpath: Path, out_path: Path, model, threshold: float = 0.50):
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        print(f"Error: Unable to open video {vpath}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    print(f"Rendering: {vpath.name}")
    print(f"  Resolution: {width}x{height} | Total Frames: {total_frames} | FPS: {fps:.1f}")

    fidx = 0
    t0 = time.time()
    total_dets = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = [F.to_tensor(rgb).to(DEVICE)]

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                preds = model(tensor)[0]

        boxes = preds["boxes"].cpu().numpy()
        labels = preds["labels"].cpu().numpy()
        scores = preds["scores"].cpu().numpy()

        dets = []
        for b, l, s in zip(boxes, labels, scores):
            if s >= threshold and l in CLASS_MAP:
                x1, y1, x2, y2 = [int(v) for v in b]
                dets.append((CLASS_MAP[l], float(s), (x1, y1, x2, y2), x2 - x1, y2 - y1))

        total_dets += len(dets)

        # Draw annotations
        annotated = frame.copy()
        for cls_name, score, (x1, y1, x2, y2), w, h in dets:
            color = COLOR_MAP.get(cls_name, (0, 255, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label_text = f"{cls_name} {score:.2f} ({w}x{h})"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label_text, (x1 + 2, max(th + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        # Top HUD Status Bar
        cv2.rectangle(annotated, (0, 0), (width, 38), (15, 15, 15), -1)
        hud_text = f"Candidate Model V8 | Frame {fidx:04d}/{total_frames:04d} | Dets: {len(dets)} | Clip: {vpath.stem}"
        cv2.putText(annotated, hud_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Bottom watermark
        cv2.rectangle(annotated, (0, height - 24), (width, height), (15, 15, 15), -1)
        sub_text = "Anchor Pyramid: [16, 32, 64, 128, 192]px | Clamps: RPN ln(1.4), RoI ln(1.3) | Max Physical Box: 494px"
        cv2.putText(annotated, sub_text, (15, height - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        writer.write(annotated)
        fidx += 1

    cap.release()
    writer.release()
    elapsed = round(time.time() - t0, 1)
    print(f"  Saved -> {out_path.name} | Detections: {total_dets} | Render Time: {elapsed}s ({fidx / max(1e-3, elapsed):.1f} fps)\n")


def main():
    print("=" * 80)
    print("   RENDERING CANDIDATE MODEL V8 SHOWCASE ANNOTATED VIDEOS")
    print("=" * 80)

    model = load_v8()
    print("Candidate V8 loaded successfully onto GPU!\n")

    targets = [
        ("Screen Recording 2026-09-10 215511-KNIFE.mp4", "samples/NEW-VIDEOS", "v8_annotated_215511_KNIFE.mp4"),
        ("Screen Recording 2026-09-07 225703.mp4", "samples/unseen_samples/VIDEO_CLIPS_JABEZ_UNSEEN/VIDEO CLIPS", "v8_annotated_225703_Clip03_RobberyGun.mp4"),
        ("Screen Recording 2026-09-07 230047.mp4", "samples/unseen_samples/VIDEO_CLIPS_JABEZ_UNSEEN/VIDEO CLIPS", "v8_annotated_230047_Clip04_SlashingKnife.mp4"),
        ("Screen Recording 2026-09-10 213340 -HANDGUN.mp4", "samples/NEW-VIDEOS", "v8_annotated_213340_HANDGUN_FloorMat.mp4"),
        ("Screen Recording 2026-09-10 214028-Riffle.mp4", "samples/NEW-VIDEOS", "v8_annotated_214028_Riffle_Banner.mp4"),
    ]

    for fname, sdir, out_name in targets:
        matches = list((ROOT / sdir).rglob(fname))
        if not matches:
            print(f"Warning: {fname} not found in {sdir}!")
            continue
        vpath = matches[0]
        out_path = OUT_DIR / out_name
        render_video(vpath, out_path, model, threshold=0.50)

    print("=" * 80)
    print(f"ALL 5 SHOWCASE VIDEOS RENDERED TO: {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
