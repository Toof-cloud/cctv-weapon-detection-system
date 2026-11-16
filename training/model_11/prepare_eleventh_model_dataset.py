"""
Prepare Model 11 Dataset:
1. Adviser's +1,800 Surveillance-POV Handgun & Knife Expansion (1.5k - 2k target):
   - +800 Train / +100 Val new authentic USRT CCTV Handgun frames from Gun_Action_Recognition_Dataset.zip
   - +710 Train / +90 Val new surveillance-scale (<7% box area) Knife frames from WeaponDataset.v11 & Knife.v1
   - +90 Train / +10 Val new surveillance-scale (<7% box area) Handgun frames from WeaponDataset.v11
2. Groupmate's Staged Hard Negative Dataset (CAM 01 & CAM 02):
   - 434 full 1280x720 frames (217 CAM_01 + 217 CAM_02) -> 390 Train / 44 Val (class_id = 0)
   - 250 targeted distractor ROI crops extracted using groupmate's 2,534 bounding box annotations
3. Strict Zero Data Leakage Guarantee with Both Test Benchmarks (279 Internal + 350 External).
"""
import json
import os
import random
import shutil
import zipfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "dataset_analysis" / "3rd_Model_Dataset"
MODEL10_DATA = ROOT / "training" / "model_10" / "tenth_model_data"
MODEL11_DATA = ROOT / "training" / "model_11" / "eleventh_model_data"
UNIFIED_BENCH_DIR = ROOT / "training" / "unified_external_benchmark"

DOWNLOADS = Path("C:/Users/pc/Downloads")
USRT_ZIP = DOWNLOADS / "Gun_Action_Recognition_Dataset.zip"
WEAPON_V11_ZIP = DOWNLOADS / "WeaponDataset.v11i.yolov8.zip"
KNIFE_V1_ZIP = DOWNLOADS / "Knife.v1i.yolov8.zip"

STAGED_DIR = DATASET_ROOT / "Staged_Groupmate_Hard_Negatives"
M11_SURV_DIR = DATASET_ROOT / "m11_surveillance_weapons"
M11_CROPS_DIR = ROOT / "research" / "model_improvement" / "dataset" / "hard_negatives_v12_staged"


def collect_existing_stems_and_paths():
    """Collect all existing train/val/test image paths and stems to guarantee zero leakage."""
    df_m10 = pd.read_csv(MODEL10_DATA / "annotations.csv")
    df_ext = pd.read_csv(UNIFIED_BENCH_DIR / "annotations.csv")
    all_paths = set(df_m10["image_path"].unique()) | set(df_ext["image_path"].unique())
    all_names = {Path(p).name for p in all_paths}
    # Also collect Roboflow base stems (before .rf.) to prevent augmented duplicates of test images
    rf_stems = set()
    for n in all_names:
        if "_rf_" in n:
            rf_stems.add(n.split("_rf_")[0])
        elif ".rf." in n:
            rf_stems.add(n.split(".rf.")[0])
        # For costi_test_prefix_hash.jpg, extract prefix
        for tag in ("costi_train_", "costi_val_", "costi_test_"):
            if n.startswith(tag):
                body = n[len(tag):]
                if "_" in body:
                    rf_stems.add(body.rsplit("_", 1)[0])
    return df_m10, all_paths, all_names, rf_stems


def extract_usrt_cctv_handgun_frames(existing_names: set[str], target_count: int = 900) -> list[dict]:
    """Extract authentic USRT CCTV Handgun frames not yet used in any split."""
    print(f"[*] Scanning {USRT_ZIP.name} for unused authentic CCTV handgun frames...")
    out_dir = M11_SURV_DIR / "usrt_handgun"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    with zipfile.ZipFile(USRT_ZIP, "r") as zf:
        handgun_jsons = sorted([
            n for n in zf.namelist()
            if "/Handgun/" in n and n.endswith("label.json")
        ])
        for hj in handgun_jsons:
            seq_dir = hj.rsplit("/", 1)[0]
            seq_name = seq_dir.split("/")[-1]
            vid_member = f"{seq_dir}/video.mp4"
            data = json.loads(zf.read(hj))
            img_map = {im["id"]: im for im in data.get("images", [])}
            ann_by_img = {}
            for ann in data.get("annotations", []):
                ann_by_img.setdefault(ann["image_id"], []).append(ann)

            # Group candidate frames per sequence
            seq_frames = []
            for img_id, anns in sorted(ann_by_img.items()):
                im_info = img_map.get(img_id)
                if not im_info:
                    continue
                fname_orig = im_info.get("file_name", "")
                # Determine 1-based or 0-based frame index from file_name or id
                # USRT existing naming: {seq_name}_f{frame_idx:04d}.jpg
                digits = "".join(ch for ch in Path(fname_orig).stem if ch.isdigit())
                fidx = int(digits) if digits else int(img_id)
                cand_name = f"{seq_name}_f{fidx:04d}.jpg"
                if cand_name in existing_names:
                    continue
                # Check nearby frames (+/- 2) to ensure distinct frames from existing test/train
                if any(f"{seq_name}_f{fidx + d:04d}.jpg" in existing_names for d in range(-3, 4)):
                    continue
                boxes = []
                W = int(im_info.get("width", 1920))
                H = int(im_info.get("height", 1080))
                for a in anns:
                    bx, by, bw, bh = a["bbox"]
                    if bw >= 6 and bh >= 6:
                        boxes.append((float(bx), float(by), float(bx + bw), float(by + bh)))
                if boxes:
                    seq_frames.append((fidx, cand_name, W, H, boxes))
            if seq_frames:
                candidates.append((seq_name, vid_member, seq_frames))

    # Sample evenly across all sequences to maximize camera/lighting diversity
    random.seed(42)
    selected_per_seq = {}
    per_seq_quota = max(1, (target_count // max(1, len(candidates))) + 2)
    total_selected = 0
    for seq_name, vid_member, seq_frames in candidates:
        if len(seq_frames) <= per_seq_quota:
            chosen = seq_frames
        else:
            step = len(seq_frames) / per_seq_quota
            chosen = [seq_frames[int(i * step)] for i in range(per_seq_quota)]
        selected_per_seq[seq_name] = (vid_member, chosen)
        total_selected += len(chosen)

    rows = []
    extracted_imgs = 0
    temp_vid = ROOT / "scratch" / "temp_usrt_vid.mp4"
    temp_vid.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(USRT_ZIP, "r") as zf:
        for seq_name, (vid_member, chosen) in selected_per_seq.items():
            if extracted_imgs >= target_count:
                break
            if vid_member not in zf.namelist():
                continue
            with zf.open(vid_member) as src, open(temp_vid, "wb") as out:
                shutil.copyfileobj(src, out)

            cap = cv2.VideoCapture(str(temp_vid))
            for fidx, cand_name, W_json, H_json, boxes in chosen:
                if extracted_imgs >= target_count:
                    break
                out_img_path = out_dir / cand_name
                if not out_img_path.exists():
                    # USRT frame indices in label.json are 1-based
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fidx - 1))
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue
                    cv2.imwrite(str(out_img_path), frame)
                    H, W = frame.shape[:2]
                else:
                    with Image.open(out_img_path) as im:
                        W, H = im.size

                split_tag = "val" if (extracted_imgs % 9 == 0) else "train"
                rel_path = f"m11_surveillance_weapons/usrt_handgun/{cand_name}"
                for x1, y1, x2, y2 in boxes:
                    # Scale if video frame size differs from JSON metadata
                    sx = W / float(W_json) if W_json > 0 else 1.0
                    sy = H / float(H_json) if H_json > 0 else 1.0
                    rows.append({
                        "image_path": rel_path,
                        "width": W,
                        "height": H,
                        "class_id": 1,
                        "xmin": round(max(0.0, x1 * sx), 2),
                        "ymin": round(max(0.0, y1 * sy), 2),
                        "xmax": round(min(float(W), x2 * sx), 2),
                        "ymax": round(min(float(H), y2 * sy), 2),
                        "split": split_tag,
                        "class_name": "Handgun",
                    })
                extracted_imgs += 1
            cap.release()

    if temp_vid.exists():
        temp_vid.unlink()
    print(f"    -> Extracted {extracted_imgs} new USRT CCTV Handgun images ({len(rows)} bounding boxes).")
    return rows


def extract_surveillance_weapons_from_yolo_zips(
    rf_stems: set[str],
    target_knife: int = 800,
    target_handgun: int = 100,
) -> list[dict]:
    """Extract surveillance-scale (0.12% - 7.5% box area) Knife & Handgun images from YOLO archives."""
    print("[*] Scanning WeaponDataset.v11 & Knife.v1 for surveillance-perspective Knife & Handgun frames...")
    out_knife_dir = M11_SURV_DIR / "surveillance_knife"
    out_gun_dir = M11_SURV_DIR / "surveillance_handgun"
    out_knife_dir.mkdir(parents=True, exist_ok=True)
    out_gun_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    knife_count = 0
    gun_count = 0

    # 1. Scan Knife.v1i.yolov8.zip (single class 0 = knife, strong CCTV attack clips like DefenseKnifeAttack)
    if KNIFE_V1_ZIP.exists():
        with zipfile.ZipFile(KNIFE_V1_ZIP, "r") as zf:
            names = set(zf.namelist())
            lbl_members = sorted([
                n for n in names
                if (n.startswith("train/labels/") or n.startswith("valid/labels/")) and n.endswith(".txt")
            ])
            for lm in lbl_members:
                if knife_count >= target_knife:
                    break
                stem = Path(lm).stem
                prefix = stem.split(".rf.")[0][:40].replace(" ", "_")
                if prefix in rf_stems:
                    continue
                img_member = lm.replace("/labels/", "/images/").rsplit(".txt", 1)[0] + ".jpg"
                if img_member not in names:
                    continue
                lines = [
                    l.strip().split()
                    for l in zf.read(lm).decode("utf-8", errors="ignore").splitlines()
                    if len(l.strip().split()) >= 5
                ]
                if not lines:
                    continue
                areas = [float(p[3]) * float(p[4]) for p in lines]
                # Surveillance scale filter: weapon box is 0.1% to 7.5% of total image area
                if not (0.001 <= max(areas) <= 0.075):
                    continue

                out_name = f"m11_k1_{prefix}_{knife_count:04d}.jpg"
                out_path = out_knife_dir / out_name
                if not out_path.exists():
                    with zf.open(img_member) as src, open(out_path, "wb") as out:
                        shutil.copyfileobj(src, out)
                try:
                    with Image.open(out_path) as im:
                        W, H = im.size
                except Exception:
                    if out_path.exists():
                        out_path.unlink()
                    continue

                split_tag = "val" if (knife_count % 9 == 0) else "train"
                rel_path = f"m11_surveillance_weapons/surveillance_knife/{out_name}"
                added_box = False
                for p in lines:
                    cx, cy, bw, bh = float(p[1]), float(p[2]), float(p[3]), float(p[4])
                    x1 = max(0.0, (cx - bw / 2.0) * W)
                    y1 = max(0.0, (cy - bh / 2.0) * H)
                    x2 = min(float(W), (cx + bw / 2.0) * W)
                    y2 = min(float(H), (cy + bh / 2.0) * H)
                    if (x2 - x1) >= 4.0 and (y2 - y1) >= 4.0:
                        rows.append({
                            "image_path": rel_path,
                            "width": W,
                            "height": H,
                            "class_id": 2,
                            "xmin": round(x1, 2),
                            "ymin": round(y1, 2),
                            "xmax": round(x2, 2),
                            "ymax": round(y2, 2),
                            "split": split_tag,
                            "class_name": "Knife",
                        })
                        added_box = True
                if added_box:
                    knife_count += 1
                    rf_stems.add(prefix)

    # 2. Scan WeaponDataset.v11i.yolov8.zip (0: knife, 1: pistol) for remaining surveillance-scale knives & handguns
    if WEAPON_V11_ZIP.exists():
        with zipfile.ZipFile(WEAPON_V11_ZIP, "r") as zf:
            names = set(zf.namelist())
            lbl_members = sorted([
                n for n in names
                if (n.startswith("train/labels/") or n.startswith("valid/labels/")) and n.endswith(".txt")
            ])
            random.seed(101)
            random.shuffle(lbl_members)
            for lm in lbl_members:
                if knife_count >= target_knife and gun_count >= target_handgun:
                    break
                stem = Path(lm).stem
                prefix = stem.split(".rf.")[0][:40].replace(" ", "_")
                if prefix in rf_stems:
                    continue
                img_member = lm.replace("/labels/", "/images/").rsplit(".txt", 1)[0] + ".jpg"
                if img_member not in names:
                    continue
                lines = [
                    l.strip().split()
                    for l in zf.read(lm).decode("utf-8", errors="ignore").splitlines()
                    if len(l.strip().split()) >= 5
                ]
                if not lines:
                    continue
                # In WeaponDataset v11: class 0 = knife, class 1 = pistol
                valid_lines = [p for p in lines if int(float(p[0])) in (0, 1)]
                if not valid_lines:
                    continue
                areas = [float(p[3]) * float(p[4]) for p in valid_lines]
                if not (0.001 <= max(areas) <= 0.070):
                    continue
                classes = {int(float(p[0])) for p in valid_lines}
                is_pure_knife = (classes == {0})
                is_pure_gun = (classes == {1})

                if is_pure_knife and knife_count < target_knife:
                    out_name = f"m11_v11k_{prefix}_{knife_count:04d}.jpg"
                    out_path = out_knife_dir / out_name
                    rel_path = f"m11_surveillance_weapons/surveillance_knife/{out_name}"
                    split_tag = "val" if (knife_count % 9 == 0) else "train"
                    target_cid, target_cname = 2, "Knife"
                elif is_pure_gun and gun_count < target_handgun:
                    out_name = f"m11_v11g_{prefix}_{gun_count:04d}.jpg"
                    out_path = out_gun_dir / out_name
                    rel_path = f"m11_surveillance_weapons/surveillance_handgun/{out_name}"
                    split_tag = "val" if (gun_count % 9 == 0) else "train"
                    target_cid, target_cname = 1, "Handgun"
                else:
                    continue

                if not out_path.exists():
                    with zf.open(img_member) as src, open(out_path, "wb") as out:
                        shutil.copyfileobj(src, out)
                try:
                    with Image.open(out_path) as im:
                        W, H = im.size
                except Exception:
                    if out_path.exists():
                        out_path.unlink()
                    continue

                added_box = False
                for p in valid_lines:
                    cx, cy, bw, bh = float(p[1]), float(p[2]), float(p[3]), float(p[4])
                    x1 = max(0.0, (cx - bw / 2.0) * W)
                    y1 = max(0.0, (cy - bh / 2.0) * H)
                    x2 = min(float(W), (cx + bw / 2.0) * W)
                    y2 = min(float(H), (cy + bh / 2.0) * H)
                    if (x2 - x1) >= 4.0 and (y2 - y1) >= 4.0:
                        rows.append({
                            "image_path": rel_path,
                            "width": W,
                            "height": H,
                            "class_id": target_cid,
                            "xmin": round(x1, 2),
                            "ymin": round(y1, 2),
                            "xmax": round(x2, 2),
                            "ymax": round(y2, 2),
                            "split": split_tag,
                            "class_name": target_cname,
                        })
                        added_box = True
                if added_box:
                    rf_stems.add(prefix)
                    if target_cid == 2:
                        knife_count += 1
                    else:
                        gun_count += 1

    print(f"    -> Extracted {knife_count} Surveillance Knife images and {gun_count} Surveillance Handgun images.")
    return rows


def process_groupmate_staged_hard_negatives(max_crops: int = 250) -> list[dict]:
    """
    Process the 434 Staged Tapo CCTV Hard Negative images (CAM_01 & CAM_02):
    1. Register all 434 full 1280x720 frames as Class 0 (390 Train / 44 Val).
    2. Use the groupmate's 2,534 bounding box annotations to extract context-padded
       distractor crops into research/model_improvement/dataset/hard_negatives_v12_staged.
    """
    print("[*] Processing Groupmate's 434 Staged Tapo CCTV Hard Negatives & 2,534 Distractor Annotations...")
    M11_CROPS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    crop_candidates = []

    idx_global = 0
    for cam_sub, csv_name in [("CAM_01", "CAM_01_annotations.csv"), ("CAM_02", "CAM_02_annotations.csv")]:
        cam_dir = STAGED_DIR / cam_sub
        csv_path = STAGED_DIR / csv_name
        df_cam = pd.read_csv(csv_path)

        img_files = sorted([p for p in cam_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        for img_p in img_files:
            with Image.open(img_p) as im:
                W, H = im.size
            split_tag = "val" if (idx_global % 10 == 0) else "train"
            rel_path = f"Staged_Groupmate_Hard_Negatives/{cam_sub}/{img_p.name}"
            rows.append({
                "image_path": rel_path,
                "width": W,
                "height": H,
                "class_id": 0,
                "xmin": 0.0,
                "ymin": 0.0,
                "xmax": 0.0,
                "ymax": 0.0,
                "split": split_tag,
                "class_name": "Negative",
            })
            idx_global += 1

            # Only mine crops from training split frames to avoid any val leakage
            if split_tag == "train":
                sub_boxes = df_cam[df_cam["filename"] == img_p.name]
                for _, r in sub_boxes.iterrows():
                    x1, y1, x2, y2 = int(r["xmin"]), int(r["ymin"]), int(r["xmax"]), int(r["ymax"])
                    bw, bh = x2 - x1, y2 - y1
                    area_frac = (bw * bh) / float(max(1, W * H))
                    # Filter for localized distractor boxes (umbrella handle, screwdriver, hand, door handle)
                    if 24 <= bw <= 600 and 24 <= bh <= 600 and 0.001 <= area_frac <= 0.35:
                        crop_candidates.append((img_p, W, H, x1, y1, x2, y2, cam_sub))

    random.seed(42)
    random.shuffle(crop_candidates)
    saved_crops = 0
    for img_p, W, H, x1, y1, x2, y2, cam_sub in crop_candidates:
        if saved_crops >= max_crops:
            break
        # Add 35% context padding around the groupmate's annotated distractor box
        pad_x = max(24, int((x2 - x1) * 0.35))
        pad_y = max(24, int((y2 - y1) * 0.35))
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(W, x2 + pad_x)
        cy2 = min(H, y2 + pad_y)
        if (cx2 - cx1) >= 48 and (cy2 - cy1) >= 48:
            with Image.open(img_p) as im:
                crop_im = im.crop((cx1, cy1, cx2, cy2)).convert("RGB")
                crop_name = f"staged_{cam_sub}_{img_p.stem}_crop{saved_crops:03d}.jpg"
                crop_im.save(M11_CROPS_DIR / crop_name, quality=95)
                saved_crops += 1

    print(f"    -> Registered {len(rows)} Full Staged Hard Negative Frames and extracted {saved_crops} Distractor Crops.")
    return rows


def main():
    MODEL11_DATA.mkdir(parents=True, exist_ok=True)
    df_m10, all_paths, all_names, rf_stems = collect_existing_stems_and_paths()

    m10_train_ids = [l.strip() for l in (MODEL10_DATA / "train_ids.txt").read_text().splitlines() if l.strip()]
    m10_val_ids = [l.strip() for l in (MODEL10_DATA / "val_ids.txt").read_text().splitlines() if l.strip()]
    m10_test_ids = [l.strip() for l in (MODEL10_DATA / "test_ids.txt").read_text().splitlines() if l.strip()]

    # 1. Extract +900 USRT CCTV Handgun images
    rows_usrt = extract_usrt_cctv_handgun_frames(all_names, target_count=900)

    # 2. Extract +800 Surveillance Knife & +100 Surveillance Handgun images
    rows_yolo = extract_surveillance_weapons_from_yolo_zips(rf_stems, target_knife=800, target_handgun=100)

    # 3. Process 434 Groupmate Staged Hard Negative images + 250 Distractor Crops
    rows_staged = process_groupmate_staged_hard_negatives(max_crops=250)

    df_new = pd.DataFrame(rows_usrt + rows_yolo + rows_staged)
    new_train_ids = sorted(df_new[df_new["split"] == "train"]["image_path"].unique())
    new_val_ids = sorted(df_new[df_new["split"] == "val"]["image_path"].unique())

    m11_train_ids = sorted(set(m10_train_ids) | set(new_train_ids))
    m11_val_ids = sorted(set(m10_val_ids) | set(new_val_ids))
    m11_test_ids = sorted(set(m10_test_ids))

    df_m11 = pd.concat([df_m10, df_new], ignore_index=True)
    df_m11.to_csv(MODEL11_DATA / "annotations.csv", index=False)
    (MODEL11_DATA / "train_ids.txt").write_text("\n".join(m11_train_ids) + "\n", encoding="utf-8")
    (MODEL11_DATA / "val_ids.txt").write_text("\n".join(m11_val_ids) + "\n", encoding="utf-8")
    (MODEL11_DATA / "test_ids.txt").write_text("\n".join(m11_test_ids) + "\n", encoding="utf-8")

    # Verify zero overlap with Unified External Benchmark (350 images)
    ext_ids = {l.strip() for l in (UNIFIED_BENCH_DIR / "test_ids.txt").read_text().splitlines() if l.strip()}
    overlap = (set(m11_train_ids) | set(m11_val_ids)) & ext_ids
    assert len(overlap) == 0, f"Data leakage detected: {overlap}"

    print("\n============================================================")
    print("[*] MODEL 11 DATASET SUMMARY")
    print("============================================================")
    for split_name, ids in [("TRAIN", m11_train_ids), ("VAL", m11_val_ids), ("TEST (M9/M10 Benchmark)", m11_test_ids)]:
        sub = df_m11[df_m11["image_path"].isin(ids)]
        by_img = sub.groupby("image_path")["class_id"].max()
        n_gun = int((by_img == 1).sum())
        n_knife = int((by_img == 2).sum())
        n_neg = int((by_img == 0).sum())
        print(f"  {split_name:24s}: {len(ids):4d} images | Handgun: {n_gun:4d} | Knife: {n_knife:4d} | Hard Negatives: {n_neg:4d}")
    print(f"  Unified External Test   : {len(ext_ids):4d} images | Overlap with M11 Train/Val = {len(overlap)} (0.0%)")


if __name__ == "__main__":
    main()

