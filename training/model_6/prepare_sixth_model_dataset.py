import os
import cv2
import random
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path

# Fix random seed for reproducibility
random.seed(42)
np.random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "dataset_analysis" / "3rd_Model_Dataset"

VIRAT_CACHE_DIR = DATASET_ROOT / "VIRAT_video_cache"
VIRAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

VIRAT_FRAMES_DIR = DATASET_ROOT / "VIRAT_CCTV_Negatives"
VIRAT_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = ROOT / "training" / "sixth_model_data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# Sources
FIFTH_MODEL_CSV = ROOT / "training" / "fifth_model_data" / "annotations.csv"
THIRD_MODEL_CSV = ROOT / "training" / "third_model_data" / "annotations.csv"

# 4 Diverse VIRAT Release 2 Surveillance Clips from Kitware
VIRAT_CLIPS = [
    {
        "name": "VIRAT_S_000200_01_000226_000268.mp4",
        "item_id": "56f585248d777f753209ca5c",
        "scene": "Scene0200_StreetParking"
    },
    {
        "name": "VIRAT_S_000201_02_000590_000623.mp4",
        "item_id": "56f585ec8d777f753209ca74",
        "scene": "Scene0201_EntranceStairs"
    },
    {
        "name": "VIRAT_S_000203_05_001122_001159.mp4",
        "item_id": "56f586c78d777f753209ca92",
        "scene": "Scene0203_PedestrianPathway"
    },
    {
        "name": "VIRAT_S_000205_05_001092_001124.mp4",
        "item_id": "56f5872d8d777f753209cab0",
        "scene": "Scene0205_FacilityWalkway"
    },
]


def download_and_extract_virat_negatives(frames_per_clip=40):
    print("=" * 65)
    print("DOWNLOADING & EXTRACTING VIRAT AUTHENTIC CCTV NEGATIVE SAMPLES")
    print("=" * 65)
    virat_rel_paths = []

    for clip in VIRAT_CLIPS:
        cname = clip["name"]
        item_id = clip["item_id"]
        vpath = VIRAT_CACHE_DIR / cname

        if not vpath.exists() or vpath.stat().st_size == 0:
            url = f"https://data.kitware.com/api/v1/item/{item_id}/download"
            print(f"Downloading {cname} from Kitware Data...")
            urllib.request.urlretrieve(url, str(vpath))
            print(f"  -> Saved {cname} ({round(vpath.stat().st_size / 1e6, 1)} MB)")
        else:
            print(f"Using cached video: {cname} ({round(vpath.stat().st_size / 1e6, 1)} MB)")

        cap = cv2.VideoCapture(str(vpath))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"  [!] Warning: Could not open {cname}")
            cap.release()
            continue

        indices = np.linspace(10, total_frames - 10, frames_per_clip, dtype=int)
        extracted = 0

        for f_idx in indices:
            stem = Path(cname).stem
            out_name = f"{stem}_f{f_idx:05d}.jpg"
            out_file = VIRAT_FRAMES_DIR / out_name
            rel_path = f"VIRAT_CCTV_Negatives\\{out_name}"

            if not out_file.exists():
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    cv2.imwrite(str(out_file), frame)
                    virat_rel_paths.append(rel_path)
                    extracted += 1
            else:
                virat_rel_paths.append(rel_path)
                extracted += 1

        cap.release()
        print(f"  -> Extracted {extracted} background frames from {cname}")

    print(f"Total VIRAT CCTV negative frames available: {len(virat_rel_paths)}")
    return virat_rel_paths


def get_usrt_no_gun_negatives():
    no_gun_dir = DATASET_ROOT / "Handgun" / "No_Gun_USRT_CCTV"
    neg_paths = []
    if no_gun_dir.exists():
        for fname in sorted(os.listdir(no_gun_dir)):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                rel_path = f"Handgun\\No_Gun_USRT_CCTV\\{fname}"
                neg_paths.append(rel_path)
    print(f"Total USRT No_Gun negative frames: {len(neg_paths)}")
    return neg_paths


def main():
    print("=" * 65)
    print("PREPARING MODEL 6 DATASET (VIRAT NEGATIVES + SURVEILLANCE KNIVES + USRT GUNS)")
    print("=" * 65)

    # 1. Download & Extract VIRAT authentic CCTV negatives
    virat_neg_paths = download_and_extract_virat_negatives(frames_per_clip=40)

    # 2. Get USRT No_Gun negatives
    usrt_neg_paths = get_usrt_no_gun_negatives()

    # Combined negatives
    all_negatives = sorted(virat_neg_paths + usrt_neg_paths)
    print(f"Combined Negatives: {len(all_negatives)} ({len(virat_neg_paths)} VIRAT + {len(usrt_neg_paths)} USRT)")

    # 3. Load verified USRT Handgun annotations
    df_fifth = pd.read_csv(FIFTH_MODEL_CSV)
    df_hg = df_fifth[df_fifth["class_id"] == 1].copy()
    print(f"Loaded Handgun images: {df_hg['image_path'].nunique()} with {len(df_hg)} bboxes.")

    # 4. Load Howard clean Knife annotations
    df_third = pd.read_csv(THIRD_MODEL_CSV)
    df_knife = df_third[df_third["class_id"] == 2].copy()
    print(f"Loaded Knife images: {df_knife['image_path'].nunique()} with {len(df_knife)} bboxes.")

    # 5. Merge weapon annotations
    df_combined = pd.concat([df_hg, df_knife], ignore_index=True)
    csv_out = DATA_ROOT / "annotations.csv"
    df_combined.to_csv(csv_out, index=False)
    print(f"Saved merged weapon annotations to: {csv_out}")
    print(f"Total weapon annotations: {len(df_combined)}")

    # 6. Build Stratified Splits (70% Train, 15% Val, 15% Test)
    hg_images = sorted(df_hg["image_path"].unique().tolist())
    knife_images = sorted(df_knife["image_path"].unique().tolist())

    random.shuffle(hg_images)
    random.shuffle(knife_images)
    random.shuffle(all_negatives)

    def split_list(items):
        n = len(items)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        train = items[:n_train]
        val = items[n_train:n_train + n_val]
        test = items[n_train + n_val:]
        return train, val, test

    hg_train, hg_val, hg_test = split_list(hg_images)
    knife_train, knife_val, knife_test = split_list(knife_images)
    neg_train, neg_val, neg_test = split_list(all_negatives)

    train_ids = sorted(hg_train + knife_train + neg_train)
    val_ids = sorted(hg_val + knife_val + neg_val)
    test_ids = sorted(hg_test + knife_test + neg_test)

    for sname, sids in [("train", train_ids), ("validation", val_ids), ("test", test_ids)]:
        out_p = DATA_ROOT / f"{sname}_ids.txt"
        with open(out_p, "w", encoding="utf-8") as f:
            for item in sids:
                f.write(f"{item}\n")

    print("\n" + "=" * 65)
    print("MODEL 6 DATASET SPLIT SUMMARY")
    print("=" * 65)
    print(f"TRAIN: {len(train_ids)} images (Handgun: {len(hg_train)}, Knife: {len(knife_train)}, Negatives: {len(neg_train)})")
    print(f"VAL:   {len(val_ids)} images (Handgun: {len(hg_val)}, Knife: {len(knife_val)}, Negatives: {len(neg_val)})")
    print(f"TEST:  {len(test_ids)} images (Handgun: {len(hg_test)}, Knife: {len(knife_test)}, Negatives: {len(neg_test)})")
    print(f"TOTAL: {len(train_ids) + len(val_ids) + len(test_ids)} images")
    print("=" * 65)


if __name__ == "__main__":
    main()
