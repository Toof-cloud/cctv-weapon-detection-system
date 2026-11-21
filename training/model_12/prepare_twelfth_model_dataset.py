"""
Prepare Model 12 (Final Production Model) Dataset:
1. Unlocks all 410 previously unused zero-leakage CostiCatargiu training images
   (270 Knife + 140 Handgun) already verified in eleventh_model_data/annotations.csv
   with split == 'train' and 0% overlap with any test set.
2. Combines them with all 4,793 Model 11 training images:
   - Total Full Training Images: 5,203 (1,990 Handgun, 1,969 Knife, 1,244 Hard Negatives)
   - Validation Images: 650 (254 Handgun, 242 Knife, 154 Hard Negatives)
   - Test Images (Apples-to-Apples M9/M10/M11 Split): 279 (114 Handgun, 114 Knife, 51 Hard Negatives)
3. Extracts 40 additional targeted handheld mug/cup/torso distractor hard negative crops
   from NoGUN v5 train split + background distractor frames into
   research/model_improvement/dataset/hard_negatives_v13_cups/
   (bringing total negative crops to 586, and total effective training items to 5,789).
4. Mathematically verifies 0.0% overlap against:
   - 279-Image Internal Test Split
   - 350-Image Unified External Benchmark
   - 185-Image Strictly Unseen Internal CCTV Benchmark
"""
import json
import random
import shutil
from pathlib import Path
import cv2
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "dataset_analysis" / "3rd_Model_Dataset"
MODEL11_DATA = ROOT / "training" / "model_11" / "eleventh_model_data"
MODEL12_DATA = ROOT / "training" / "model_12" / "twelfth_model_data"
UNIFIED_BENCH_DIR = ROOT / "training" / "unified_external_benchmark"
V13_CUPS_DIR = ROOT / "research" / "model_improvement" / "dataset" / "hard_negatives_v13_cups"


def extract_targeted_cup_and_torso_negatives(df_all: pd.DataFrame, train_ids: set[str], target_crops: int = 40) -> int:
    """
    Extract targeted crops from pure Class-0 negative training images focusing on
    held cups, mugs, bottles, wallets, phones, and mid-torso hand confounders.
    Strictly uses training-split negative images only (zero test/val overlap).
    """
    V13_CUPS_DIR.mkdir(parents=True, exist_ok=True)
    neg_tr = df_all[(df_all["image_path"].isin(train_ids)) & (df_all["class_id"] == 0)]["image_path"].unique()

    priority_keywords = ("mug", "cup", "bottle", "drink", "coffee", "phone", "wallet", "bag", "staged", "cam_01", "cam_02")
    priority_paths = [p for p in neg_tr if any(k in p.lower() for k in priority_keywords)]
    other_paths = [p for p in neg_tr if p not in priority_paths]

    random.seed(120)
    random.shuffle(priority_paths)
    random.shuffle(other_paths)
    candidates = priority_paths + other_paths

    saved = 0
    for rel_p in candidates:
        if saved >= target_crops:
            break
        full_p = DATASET_ROOT / rel_p
        if not full_p.exists():
            continue
        try:
            with Image.open(full_p) as im:
                im_rgb = im.convert("RGB")
                W, H = im_rgb.size
                if W < 160 or H < 160:
                    continue
                # Extract center-torso / handheld region crop (central 50% width, 30%-80% height)
                cx1 = int(W * 0.25)
                cy1 = int(H * 0.25)
                cx2 = int(W * 0.75)
                cy2 = int(H * 0.80)
                crop = im_rgb.crop((cx1, cy1, cx2, cy2))
                out_name = f"v13_cup_torso_{saved:03d}_{Path(rel_p).stem[:30]}.jpg"
                crop.save(V13_CUPS_DIR / out_name, quality=95)
                saved += 1
        except Exception:
            continue
    return saved


def main():
    MODEL12_DATA.mkdir(parents=True, exist_ok=True)
    df_m11 = pd.read_csv(MODEL11_DATA / "annotations.csv")

    m11_train_ids = set(l.strip() for l in (MODEL11_DATA / "train_ids.txt").read_text().splitlines() if l.strip())
    m11_val_ids = set(l.strip() for l in (MODEL11_DATA / "val_ids.txt").read_text().splitlines() if l.strip())
    m11_test_ids = set(l.strip() for l in (MODEL11_DATA / "test_ids.txt").read_text().splitlines() if l.strip())

    ext_test_ids = set(l.strip() for l in (UNIFIED_BENCH_DIR / "test_ids.txt").read_text().splitlines() if l.strip())
    unseen_cctv_ids = set(l.strip() for l in (UNIFIED_BENCH_DIR / "strictly_unseen_internal_ids.txt").read_text().splitlines() if l.strip())

    all_forbidden_test_val = m11_val_ids | m11_test_ids | ext_test_ids | unseen_cctv_ids

    # Recover the 410 unused train images from annotations.csv that are marked split == 'train'
    all_annotated_train = set(df_m11[df_m11["split"] == "train"]["image_path"].unique()) - all_forbidden_test_val
    m12_train_ids = sorted(m11_train_ids | all_annotated_train)
    m12_val_ids = sorted(m11_val_ids)
    m12_test_ids = sorted(m11_test_ids)

    # Verify files physically exist on disk
    verified_train_ids = [p for p in m12_train_ids if (DATASET_ROOT / p).exists()]
    verified_val_ids = [p for p in m12_val_ids if (DATASET_ROOT / p).exists()]
    verified_test_ids = [p for p in m12_test_ids if (DATASET_ROOT / p).exists()]

    # Strict mathematical leakage assertions
    assert len(set(verified_train_ids) & set(verified_val_ids)) == 0, "Train/Val overlap!"
    assert len(set(verified_train_ids) & set(verified_test_ids)) == 0, "Train/Test279 overlap!"
    assert len(set(verified_train_ids) & ext_test_ids) == 0, "Train/Ext350 overlap!"
    assert len(set(verified_train_ids) & unseen_cctv_ids) == 0, "Train/Unseen185 overlap!"

    df_m11.to_csv(MODEL12_DATA / "annotations.csv", index=False)
    (MODEL12_DATA / "train_ids.txt").write_text("\n".join(verified_train_ids) + "\n", encoding="utf-8")
    (MODEL12_DATA / "val_ids.txt").write_text("\n".join(verified_val_ids) + "\n", encoding="utf-8")
    (MODEL12_DATA / "test_ids.txt").write_text("\n".join(verified_test_ids) + "\n", encoding="utf-8")

    added_crops = extract_targeted_cup_and_torso_negatives(df_m11, set(verified_train_ids), target_crops=40)

    summary = {"splits": {}, "negative_crops_added": added_crops}
    print("=" * 78)
    print("  MODEL 12 (FINAL PRODUCTION MODEL) DATASET SPECIFICATION & LEAKAGE AUDIT")
    print("=" * 78)
    for split_name, ids in [
        ("TRAIN", verified_train_ids),
        ("VAL", verified_val_ids),
        ("TEST_279", verified_test_ids),
    ]:
        sub = df_m11[df_m11["image_path"].isin(ids)]
        by_img = sub.groupby("image_path")["class_id"].max()
        n_gun = int((by_img == 1).sum())
        n_knife = int((by_img == 2).sum())
        n_neg = int((by_img == 0).sum())
        n_boxes_gun = int((sub["class_id"] == 1).sum())
        n_boxes_knife = int((sub["class_id"] == 2).sum())
        summary["splits"][split_name] = {
            "total_images": len(ids),
            "handgun_images": n_gun,
            "knife_images": n_knife,
            "hard_negative_images": n_neg,
            "handgun_boxes": n_boxes_gun,
            "knife_boxes": n_boxes_knife,
        }
        print(
            f"  {split_name:10s}: {len(ids):4d} images | "
            f"Handgun: {n_gun:4d} ({n_boxes_gun:4d} boxes) | "
            f"Knife: {n_knife:4d} ({n_boxes_knife:4d} boxes) | "
            f"Hard Negatives: {n_neg:4d}"
        )

    # Count all negative crop files across folders
    crop_folders = [
        "hard_negatives",
        "hard_negatives_fullframe",
        "hard_negatives_v9",
        "hard_negatives_v10",
        "hard_negatives_v11",
        "hard_negatives_v12_staged",
        "hard_negatives_v13_cups",
    ]
    total_crops = 0
    crop_breakdown = {}
    for cf in crop_folders:
        nd = ROOT / "research" / "model_improvement" / "dataset" / cf
        cnt = len(list(nd.glob("*.jpg"))) if nd.exists() else 0
        crop_breakdown[cf] = cnt
        total_crops += cnt

    summary["total_negative_crops"] = total_crops
    summary["crop_breakdown"] = crop_breakdown
    summary["zero_leakage_verified"] = {
        "overlap_with_test_279": len(set(verified_train_ids) & set(verified_test_ids)),
        "overlap_with_ext_350": len(set(verified_train_ids) & ext_test_ids),
        "overlap_with_unseen_185": len(set(verified_train_ids) & unseen_cctv_ids),
    }

    (MODEL12_DATA / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Total Targeted Negative ROI Crops : {total_crops} crops across {len(crop_folders)} folders")
    print(f"  Effective Training Pool           : {len(verified_train_ids) + total_crops} items")
    print(f"  Zero-Leakage Audit                : 0 overlap with Test279, Ext350, and Unseen185")
    print("=" * 78)


if __name__ == "__main__":
    main()

