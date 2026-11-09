"""
Run Model 10 (best_weapon_detector_tenth_model.pth) on:
  1. The 3 sample video clips from the CostiCatargiu/Yolov12_WeaponDetection repo:
     - samples/costi_demo_clips/test1gun.mp4 (3 Army Knives close-up)
     - samples/costi_demo_clips/test2gun.mp4 (Close-up Pistol magazine load)
     - samples/costi_demo_clips/test3gun.mp4 (Long gun / Rifle - out of 2-class scope)
  2. A compiled video reel of held non-weapon frames from CostiCatargiu's NoGUN v5 / DatasetExamples
     (costi_held_non_weapons_model10_output.mp4) to visually demonstrate non-hallucination on held objects.
Saves all output MP4s to outputs/model_10_sample_videos/.
"""
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
torch.set_num_threads(8)

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from app.services.detection_service import DetectionService

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = ROOT_DIR / "best_weapon_detector_tenth_model.pth"
SAMPLE_DIR = ROOT_DIR / "samples" / "costi_demo_clips"
OUT_DIR = ROOT_DIR / "outputs" / "model_10_sample_videos"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_model10_overlay(frame_bgr: np.ndarray, boxes, scores, labels, clip_title: str) -> np.ndarray:
    canvas = frame_bgr.copy()
    h, w = canvas.shape[:2]

    # Top banner showing Model 10 status
    cv2.rectangle(canvas, (0, 0), (w, 38), (20, 20, 20), -1)
    status_txt = f"Model 10 (Faster R-CNN ResNet50-FPN) | {clip_title}"
    cv2.putText(canvas, status_txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [int(v) for v in box]
        cls_name = "HANDGUN" if label == 1 else "KNIFE"
        color = (0, 0, 255) if label == 1 else (0, 165, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        tag = f"MODEL 10: {cls_name} {score * 100:.1f}%"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(y1 - 8, th + 45)
        cv2.rectangle(canvas, (x1, ty - th - 6), (x1 + tw + 8, ty + 4), color, -1)
        cv2.putText(canvas, tag, (x1 + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    if len(boxes) == 0:
        badge = "MODEL 10: NO WEAPON DETECTED (CLEAN)"
        cv2.rectangle(canvas, (10, 46), (395, 78), (30, 130, 30), -1)
        cv2.putText(canvas, badge, (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas


def process_clip(svc: DetectionService, in_mp4: Path, out_mp4: Path, title: str, conf_thresh: float = 0.50):
    model = svc.model
    cap = cv2.VideoCapture(str(in_mp4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    total_frames = 0
    detected_frames = 0
    total_boxes = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(Image.fromarray(rgb)).to(DEVICE)
        with torch.no_grad():
            out = model([tensor])[0]

        keep = out["scores"] >= conf_thresh
        boxes = out["boxes"][keep].cpu().numpy()
        scores = out["scores"][keep].cpu().numpy()
        labels = out["labels"][keep].cpu().numpy()

        if len(boxes) > 0:
            detected_frames += 1
            total_boxes += len(boxes)

        annotated = draw_model10_overlay(frame, boxes, scores, labels, title)
        writer.write(annotated)

    cap.release()
    writer.release()
    print(f"[*] Saved {out_mp4.name:<42} | Frames: {total_frames:<3} | Frames w/ Det: {detected_frames:<3} | Boxes: {total_boxes}")


def build_held_non_weapons_reel(svc: DetectionService, out_mp4: Path, conf_thresh: float = 0.50):
    import pandas as pd

    pure_noweapon_cells = [
        (1, 0, 0, "Man holding coffee mug"),
        (1, 0, 2, "Man holding glass beer bottle"),
        (1, 1, 1, "Couple holding sandwich & coffee cup"),
        (1, 1, 2, "Hand holding grey smartphone/card"),
        (1, 1, 3, "Family holding garden rakes"),
        (1, 2, 0, "Worker holding window squeegee"),
        (1, 2, 2, "Man on treadmill holding water bottle"),
        (1, 2, 3, "Hand holding clear water bottle"),
        (1, 3, 3, "Hand holding selfie stick & tripod"),
        (2, 0, 0, "Hand gripping umbrella handle"),
        (2, 0, 2, "Man sweeping street with broom"),
        (2, 1, 1, "Man holding selfie stick"),
        (2, 1, 3, "Gloved hand scrubbing floor with brush"),
        (2, 2, 1, "Woman holding leather clutch purse"),
        (2, 2, 3, "Man holding smartphone in both hands"),
        (2, 3, 0, "Woman holding floor broom"),
        (2, 3, 1, "Hands holding two metal water bottles"),
        (2, 3, 2, "Couple raking leaves in yard"),
    ]

    model = svc.model
    dataset_root = ROOT_DIR / "dataset_analysis" / "3rd_Model_Dataset"
    df_ext = pd.read_csv(ROOT_DIR / "training" / "unified_external_benchmark" / "annotations.csv")
    costi_neg_rel = sorted(
        df_ext[(df_ext["class_id"] == 0) & (df_ext["image_path"].str.startswith("costi_yolov12_curated/"))]["image_path"].unique()
    )

    frames_to_eval = []
    ex_dir = ROOT_DIR / "data" / "costi_yolov12" / "repo" / "DatasetExamples"
    for g_idx, r, c, desc in pure_noweapon_cells:
        gp = ex_dir / f"no_weapon_grid{g_idx}.png"
        if gp.exists():
            full_bgr = cv2.imread(str(gp))
            if full_bgr is not None:
                H, W = full_bgr.shape[:2]
                top_offset = int(H * 0.055)
                grid_h = H - top_offset
                ch, cw = grid_h // 4, W // 4
                y1 = top_offset + r * ch + 8
                y2 = top_offset + (r + 1) * ch - 8
                x1 = c * cw + 8
                x2 = (c + 1) * cw - 8
                crop_bgr = full_bgr[y1:y2, x1:x2]
                frames_to_eval.append((crop_bgr, f"Held Object Grid {g_idx} ({r+1},{c+1}): {desc}"))

    # Next add 32 unseen CostiCatargiu NoGUN v5 / WeaponDataset v11 held non-weapon test scenes
    for rel_p in costi_neg_rel[:32]:
        full_p = dataset_root / rel_p
        if full_p.exists():
            bgr = cv2.imread(str(full_p))
            if bgr is not None:
                frames_to_eval.append((bgr, f"Unseen Costi Held Non-Weapon: {full_p.name[:34]}"))

    target_w, target_h = 960, 720
    fps = 2.0
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (target_w, target_h))

    clean_count = 0
    for idx, (bgr, label_desc) in enumerate(frames_to_eval):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(Image.fromarray(rgb)).to(DEVICE)
        with torch.no_grad():
            out = model([tensor])[0]
        keep = out["scores"] >= conf_thresh
        boxes = out["boxes"][keep].cpu().numpy()
        scores = out["scores"][keep].cpu().numpy()
        labels = out["labels"][keep].cpu().numpy()
        if len(boxes) == 0:
            clean_count += 1

        ih, iw = bgr.shape[:2]
        scale = min((target_w - 40) / iw, (target_h - 100) / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        resized = cv2.resize(bgr, (nw, nh))
        canvas = np.full((target_h, target_w, 3), 28, dtype=np.uint8)
        ox = (target_w - nw) // 2
        oy = 70 + (target_h - 80 - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = resized

        scaled_boxes = []
        for b in boxes:
            scaled_boxes.append([b[0] * scale + ox, b[1] * scale + oy, b[2] * scale + ox, b[3] * scale + oy])

        annotated = draw_model10_overlay(
            canvas,
            scaled_boxes,
            scores,
            labels,
            f"[{idx+1}/{len(frames_to_eval)}] {label_desc}",
        )
        writer.write(annotated)
        writer.write(annotated)

    writer.release()
    print(
        f"[*] Saved {out_mp4.name:<42} | Held Non-Weapon Samples: {len(frames_to_eval)} | "
        f"Clean (0 Hallucinations): {clean_count}/{len(frames_to_eval)} ({clean_count/max(1,len(frames_to_eval))*100:.1f}%)"
    )


def main():
    svc = DetectionService(model_path=str(MODEL_PATH), confidence_threshold=0.50)
    process_clip(svc, SAMPLE_DIR / "test1gun.mp4", OUT_DIR / "test1gun_model10_output.mp4", "Costi Sample 1: 3 Army Knives")
    process_clip(svc, SAMPLE_DIR / "test2gun.mp4", OUT_DIR / "test2gun_model10_output.mp4", "Costi Sample 2: Pistol Load")
    process_clip(svc, SAMPLE_DIR / "test3gun.mp4", OUT_DIR / "test3gun_model10_output.mp4", "Costi Sample 3: Long Gun (Excluded Class)")
    build_held_non_weapons_reel(svc, OUT_DIR / "costi_held_non_weapons_model10_output.mp4", conf_thresh=0.50)


if __name__ == "__main__":
    main()

