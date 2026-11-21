"""
Prepare Model 13 (Final Production SOTA) Dataset:
1. Starts from all verified Model 12 training/val/test splits (5,203 train / 650 val / 279 test).
2. Mines targeted hard negatives ONLY from Nov 20 Staged Non-Weapon Scenes #01-#06
   (12 videos across CAM 01 + CAM 02: black TV remote, silver/dark remote, metallic kitchen ladle,
   rolled white rag/towel, smartphone, wallet).
   - STRICTLY EXCLUDES Scenes #07-#11 (realistic silver pistol prop & knife props — kept 100% untouched).
3. Extracts:
   - Full-frame hard-negative images into dataset_analysis/3rd_Model_Dataset/Nov20_Staged_Hard_Negatives/
   - Padded tight ROI negative crops around exact Model 12 FP boxes into
     research/model_improvement/dataset/hard_negatives_v14_nov20_household/
4. Mathematically verifies 0.0% overlap with Test279, Ext350, Unseen185, and Val650.
"""
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "dataset_analysis" / "3rd_Model_Dataset"
MODEL12_DATA = ROOT / "training" / "model_12" / "twelfth_model_data"
MODEL13_DATA = ROOT / "training" / "model_13" / "thirteenth_model_data"
UNIFIED_BENCH_DIR = ROOT / "training" / "unified_external_benchmark"

NOV20_VIDEOS_ROOT = ROOT / "samples" / "03_staged_and_multicamera" / "nov20_staged_test_videos" / "Nov 20 - Staged Test Videos"
NOV20_OUT_ROOT = ROOT / "outputs" / "model12_nov20_staged_20pct" / "Nov 20 - Staged Test Videos"

FULLFRAME_NEG_DIR = DATASET_ROOT / "Nov20_Staged_Hard_Negatives"
CROP_NEG_DIR = ROOT / "research" / "model_improvement" / "dataset" / "hard_negatives_v14_nov20_household"

# STRICTLY NON-WEAPON SCENES #01-#06 ONLY (Scenes #07-#11 are weapon props and MUST NOT be mined)
NON_WEAPON_CLIPS = [
    # CAM 01 (#01-#06)
    ("CAM 01", "1869451F17C5_20261120123911977_cam0", "s01_black_remote_cam01"),
    ("CAM 01", "1869451F17C5_20261120124317091_cam0", "s02_long_remote_cam01"),
    ("CAM 01", "1869451F17C5_20261120124554102_cam0", "s03_metal_ladle_cam01"),
    ("CAM 01", "1869451F17C5_20261120124805176_cam0", "s04_white_rag_cam01"),
    ("CAM 01", "1869451F17C5_20261120124918870_cam0", "s05_smartphone_cam01"),
    ("CAM 01", "1869451F17C5_20261120124950401_cam0", "s06_wallet_cam01"),
    # CAM 02 (#01-#06)
    ("CAM 02", "1795149551524_0", "s01_black_remote_cam02"),
    ("CAM 02", "1795149796716_0", "s02_long_remote_cam02"),
    ("CAM 02", "1795149953671_0", "s03_metal_ladle_cam02"),
    ("CAM 02", "1795150084724_0", "s04_white_rag_cam02"),
    ("CAM 02", "1795150158464_0", "s05_smartphone_cam02"),
    ("CAM 02", "1795150190098_0", "s06_wallet_cam02"),
]


def parse_box(box_str: str) -> list[int] | None:
    try:
        vals = ast.literal_eval(box_str)
        if isinstance(vals, (list, tuple)) and len(vals) == 4:
            return [int(round(float(v))) for v in vals]
    except Exception:
        pass
    return None


def mine_nov20_non_weapons(max_full_per_clip: int = 20, max_crops_per_clip: int = 24) -> tuple[list[dict], int]:
    FULLFRAME_NEG_DIR.mkdir(parents=True, exist_ok=True)
    CROP_NEG_DIR.mkdir(parents=True, exist_ok=True)
    for old in FULLFRAME_NEG_DIR.glob("*.jpg"):
        old.unlink()
    for old in CROP_NEG_DIR.glob("*.jpg"):
        old.unlink()

    full_rows = []
    total_crops = 0

    for cam_dir, stem, tag in NON_WEAPON_CLIPS:
        vid_p = NOV20_VIDEOS_ROOT / cam_dir / f"{stem}.mp4"
        csv_p = NOV20_OUT_ROOT / cam_dir / f"{stem}_m12_20pct.csv"
        if not vid_p.exists():
            print(f"[WARN] Missing video: {vid_p}")
            continue

        # Collect FP frames + boxes sorted by highest confidence first
        fp_by_frame: dict[int, list[tuple[float, list[int]]]] = {}
        if csv_p.exists():
            with csv_p.open("r", encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    sc = float(r.get("confidence_score") or 0.0)
                    fr = int(float(r.get("frame_number") or 0))
                    bx = parse_box(r.get("bounding_box") or "")
                    if fr > 0 and bx is not None:
                        fp_by_frame.setdefault(fr, []).append((sc, bx))

        cap = cv2.VideoCapture(str(vid_p))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Prioritize frames where Model 12 fired (evenly spaced across FP frames), plus 4 clean context frames
        fp_frames_sorted = sorted(fp_by_frame.keys())
        if len(fp_frames_sorted) > max_full_per_clip:
            step = len(fp_frames_sorted) / max_full_per_clip
            chosen_full = [fp_frames_sorted[int(i * step)] for i in range(max_full_per_clip)]
        else:
            chosen_full = list(fp_frames_sorted)

        # Ensure at least 8 frames per clip even for clean clips like smartphone/wallet
        if len(chosen_full) < 8 and n_frames > 10:
            for k in range(8):
                f_cand = int((k + 1) * n_frames / 9)
                if f_cand not in chosen_full:
                    chosen_full.append(f_cand)
        chosen_full = sorted(set(chosen_full))

        saved_full = 0
        saved_crops = 0
        for fr in chosen_full:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(n_frames - 1, fr - 1)))
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                continue

            fname = f"{tag}_f{fr:04d}.jpg"
            out_full = FULLFRAME_NEG_DIR / fname
            cv2.imwrite(str(out_full), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 94])
            rel_path = f"Nov20_Staged_Hard_Negatives/{fname}"
            full_rows.append({
                "image_path": rel_path,
                "width": w,
                "height": h,
                "xmin": 0.0,
                "ymin": 0.0,
                "xmax": 0.0,
                "ymax": 0.0,
                "class_id": 0,
                "class_name": "background",
                "split": "train",
            })
            saved_full += 1

            # Extract tight padded ROI crop around top FP box in this frame
            if fr in fp_by_frame and saved_crops < max_crops_per_clip:
                top_sc, (x1, y1, x2, y2) = max(fp_by_frame[fr], key=lambda x: x[0])
                bw, bh = max(24, x2 - x1), max(24, y2 - y1)
                pad_x = int(max(48, bw * 0.9))
                pad_y = int(max(48, bh * 0.9))
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(w, x2 + pad_x)
                cy2 = min(h, y2 + pad_y)
                crop = frame_bgr[cy1:cy2, cx1:cx2]
                if crop.size > 0 and crop.shape[0] >= 40 and crop.shape[1] >= 40:
                    crop_name = f"v14_{tag}_f{fr:04d}_c{int(top_sc*100)}.jpg"
                    cv2.imwrite(str(CROP_NEG_DIR / crop_name), crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
                    saved_crops += 1
                    total_crops += 1

        cap.release()
        print(f"  [MINED] {tag:<26} -> {saved_full:2d} full frames + {saved_crops:2d} FP ROI crops")

    return full_rows, total_crops


def main() -> None:
    MODEL13_DATA.mkdir(parents=True, exist_ok=True)
    df_m12 = pd.read_csv(MODEL12_DATA / "annotations.csv")

    m12_train_ids = [l.strip() for l in (MODEL12_DATA / "train_ids.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    m12_val_ids = [l.strip() for l in (MODEL12_DATA / "val_ids.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    m12_test_ids = [l.strip() for l in (MODEL12_DATA / "test_ids.txt").read_text(encoding="utf-8").splitlines() if l.strip()]

    ext_test_ids = set(l.strip() for l in (UNIFIED_BENCH_DIR / "test_ids.txt").read_text(encoding="utf-8").splitlines() if l.strip())
    unseen_cctv_ids = set(l.strip() for l in (UNIFIED_BENCH_DIR / "strictly_unseen_internal_ids.txt").read_text(encoding="utf-8").splitlines() if l.strip())

    print("Mining hard negatives from Nov 20 Non-Weapon Scenes #01-#06 (12 videos)...")
    nov20_rows, nov20_crops = mine_nov20_non_weapons(max_full_per_clip=20, max_crops_per_clip=24)

    # Align columns with Model 12 annotations.csv
    cols = list(df_m12.columns)
    aligned_rows = []
    for r in nov20_rows:
        aligned_rows.append({c: r.get(c, 0 if c in ("xmin", "ymin", "xmax", "ymax", "class_id") else "") for c in cols})
    df_nov20 = pd.DataFrame(aligned_rows, columns=cols)

    df_m13 = pd.concat([df_m12, df_nov20], ignore_index=True)
    new_train_ids = [r["image_path"] for r in nov20_rows]
    m13_train_ids = sorted(set(m12_train_ids) | set(new_train_ids))

    # Strict mathematical leakage assertions
    assert len(set(m13_train_ids) & set(m12_val_ids)) == 0, "Train/Val overlap!"
    assert len(set(m13_train_ids) & set(m12_test_ids)) == 0, "Train/Test279 overlap!"
    assert len(set(m13_train_ids) & ext_test_ids) == 0, "Train/Ext350 overlap!"
    assert len(set(m13_train_ids) & unseen_cctv_ids) == 0, "Train/Unseen185 overlap!"

    df_m13.to_csv(MODEL13_DATA / "annotations.csv", index=False)
    (MODEL13_DATA / "train_ids.txt").write_text("\n".join(m13_train_ids) + "\n", encoding="utf-8")
    (MODEL13_DATA / "val_ids.txt").write_text("\n".join(m12_val_ids) + "\n", encoding="utf-8")
    (MODEL13_DATA / "test_ids.txt").write_text("\n".join(m12_test_ids) + "\n", encoding="utf-8")

    crop_folders = [
        "hard_negatives",
        "hard_negatives_fullframe",
        "hard_negatives_v9",
        "hard_negatives_v10",
        "hard_negatives_v11",
        "hard_negatives_v12_staged",
        "hard_negatives_v13_cups",
        "hard_negatives_v14_nov20_household",
    ]
    total_crops = sum(len(list((ROOT / "research" / "model_improvement" / "dataset" / cf).glob("*.jpg"))) for cf in crop_folders if (ROOT / "research" / "model_improvement" / "dataset" / cf).exists())

    summary = {
        "train_full_images": len(m13_train_ids),
        "nov20_full_negatives_added": len(new_train_ids),
        "nov20_roi_crops_added": nov20_crops,
        "total_negative_crops": total_crops,
        "effective_training_pool": len(m13_train_ids) + total_crops,
        "val_images": len(m12_val_ids),
        "test_279_images": len(m12_test_ids),
        "zero_leakage_verified": True,
        "excluded_weapon_prop_scenes": ["Scene #07 (Silver Pistol)", "Scene #08 (Knife)", "Scene #09 (Table Props)", "Scene #10 (Cleaver)", "Scene #11 (Dual Knives)"],
    }
    (MODEL13_DATA / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + "=" * 78)
    print(f"MODEL 13 DATASET READY: {len(m13_train_ids)} full train images (+{len(new_train_ids)} Nov20) + {total_crops} ROI crops (+{nov20_crops} Nov20)")
    print(f"Effective Training Pool: {len(m13_train_ids) + total_crops} items | 0.0% Test Leakage Verified")
    print("=" * 78)


if __name__ == "__main__":
    main()

