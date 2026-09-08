import os
import cv2
import random
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path

# Fix random seed for strict reproducibility
random.seed(42)
np.random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "dataset_analysis" / "3rd_Model_Dataset"

VIRAT_CACHE_DIR = DATASET_ROOT / "VIRAT_video_cache"
VIRAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL8_NEG_DIR = DATASET_ROOT / "Model8_Hard_Negatives"
MODEL8_NEG_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = ROOT / "training" / "eighth_model_data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# Weapon annotation sources
FIFTH_MODEL_CSV = ROOT / "training" / "fifth_model_data" / "annotations.csv"
THIRD_MODEL_CSV = ROOT / "training" / "third_model_data" / "annotations.csv"

# 4 Diverse VIRAT Surveillance Clips from Kitware
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


def extract_virat_negatives(frames_per_clip=50):
    print("=" * 65)
    print("EXTRACTING VIRAT AUTHENTIC SURVEILLANCE NEGATIVES")
    print("=" * 65)
    neg_paths = []

    for clip in VIRAT_CLIPS:
        cname = clip["name"]
        item_id = clip["item_id"]
        vpath = VIRAT_CACHE_DIR / cname

        if not vpath.exists():
            url = f"https://data.kitware.com/api/v1/item/{item_id}/download"
            print(f"Downloading {cname} from Kitware...")
            try:
                urllib.request.urlretrieve(url, vpath)
                print(f"Downloaded: {vpath.stat().st_size / (1024*1024):.1f} MB")
            except Exception as e:
                print(f"Failed to download {cname}: {e}")
                continue

        cap = cv2.VideoCapture(str(vpath))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            continue

        indices = np.linspace(10, total_frames - 10, frames_per_clip, dtype=int)
        extracted = 0

        for f_idx in indices:
            out_name = f"virat_{clip['scene']}_f{f_idx:05d}.jpg"
            out_file = MODEL8_NEG_DIR / out_name
            rel_path = f"Model8_Hard_Negatives\\{out_name}"

            if not out_file.exists():
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    if max(h, w) > 1280:
                        scale = 1280.0 / max(h, w)
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                    cv2.imwrite(str(out_file), frame)
                    neg_paths.append(rel_path)
                    extracted += 1
            else:
                neg_paths.append(rel_path)
                extracted += 1

        cap.release()
        print(f"  -> Extracted {extracted} frames from {cname}")

    return neg_paths


def extract_clip_negatives(video_path: Path, prefix: str, frame_count: int = 40):
    if not video_path.exists():
        print(f"  [!] Video not found: {video_path}")
        return []
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    indices = np.linspace(2, min(total_frames - 2, 80), frame_count, dtype=int)
    neg_paths = []
    extracted = 0
    for f_idx in indices:
        out_name = f"{prefix}_f{f_idx:04d}.jpg"
        out_file = MODEL8_NEG_DIR / out_name
        rel_path = f"Model8_Hard_Negatives\\{out_name}"
        if not out_file.exists():
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                cv2.imwrite(str(out_file), frame)
                neg_paths.append(rel_path)
                extracted += 1
        else:
            neg_paths.append(rel_path)
            extracted += 1
    cap.release()
    print(f"  -> Extracted {extracted} negative frames from {video_path.name}")
    return neg_paths


def extract_phone_and_gadget_negatives():
    """Extracts authentic phone/gadget negative frames where actors hold smartphones or gadgets."""
    print("=" * 65)
    print("CURATING SMARTPHONE & GADGET HARD NEGATIVES")
    print("=" * 65)
    neg_paths = []
    
    # 1. From NEW_KNIFE_VIDEO_11s.mp4 (victim on floor filming attacker with horizontal smartphone)
    # Frames 125 to 142 show victim holding horizontal black smartphone to record
    knife_video = ROOT / "samples" / "NEW_KNIFE_VIDEO_11s.mp4"
    if knife_video.exists():
        cap = cv2.VideoCapture(str(knife_video))
        for f_idx in range(125, 142):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                out_name = f"phone_neg_floor_f{f_idx:04d}.jpg"
                out_file = MODEL8_NEG_DIR / out_name
                rel_path = f"Model8_Hard_Negatives\\{out_name}"
                if not out_file.exists():
                    cv2.imwrite(str(out_file), frame)
                neg_paths.append(rel_path)
        cap.release()
        print(f"  -> Extracted {len(neg_paths)} horizontal phone negative frames from restaurant footage")

    return neg_paths


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
    print("PREPARING MODEL 8 DATASET (BALANCED ANCHORS + SANITIZED NEGATIVES)")
    print("=" * 65)

    # 1. Extract VIRAT surveillance negatives (200 frames: streets, parking, stairs, paths)
    virat_negs = extract_virat_negatives(frames_per_clip=50)

    # 2. Extract USRT No_Gun negatives (~100 frames)
    usrt_negs = get_usrt_no_gun_negatives()

    # 3. Extract pure surveillance negative clip frames (40 frames)
    surv_negs = extract_clip_negatives(ROOT / "samples" / "test_10s_negative.mp4", "surv_neg", 40)

    # 4. Extract smartphone/gadget negative frames (17 frames)
    phone_negs = extract_phone_and_gadget_negatives()

    # NOTE: PURGED the 25 contaminated CAM02 negative frames that penalized silver handguns in Model 7!
    all_negatives = sorted(list(set(virat_negs + usrt_negs + surv_negs + phone_negs)))
    print(f"\nTotal Sanitized Hard Negatives Curated for Model 8: {len(all_negatives)}")

    # 5. Load verified Handguns (USRT)
    df_fifth = pd.read_csv(FIFTH_MODEL_CSV)
    df_hg = df_fifth[df_fifth["class_id"] == 1].copy()
    print(f"Loaded Handgun images: {df_hg['image_path'].nunique()} with {len(df_hg)} bboxes.")

    # 6. Load Howard clean Knives
    df_third = pd.read_csv(THIRD_MODEL_CSV)
    df_knife = df_third[df_third["class_id"] == 2].copy()
    print(f"Loaded Knife images: {df_knife['image_path'].nunique()} with {len(df_knife)} bboxes.")

    # 7. Merge weapon annotations
    df_weapons = pd.concat([df_hg, df_knife], ignore_index=True)

    # 8. Create negative entries (zero boxes)
    neg_rows = []
    for rel_p in all_negatives:
        neg_rows.append({
            "image_path": rel_p,
            "class_id": 0,
            "xmin": 0.0,
            "ymin": 0.0,
            "xmax": 0.0,
            "ymax": 0.0,
        })
    df_negs = pd.DataFrame(neg_rows)

    # Group weapons by image for stratified splitting
    img_groups = {}
    for img_p, group in df_weapons.groupby("image_path"):
        cids = set(group["class_id"].tolist())
        if 1 in cids and 2 in cids:
            cat = "both"
        elif 1 in cids:
            cat = "handgun"
        else:
            cat = "knife"
        img_groups[img_p] = cat

    # Also add negatives as 'negative' category
    for rel_p in all_negatives:
        img_groups[rel_p] = "negative"

    # Stratified 80/10/10 split by image category
    train_imgs, val_imgs, test_imgs = set(), set(), set()
    cat_to_imgs = {}
    for img_p, cat in img_groups.items():
        cat_to_imgs.setdefault(cat, []).append(img_p)

    for cat, imgs in cat_to_imgs.items():
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(0.80 * n)
        n_val = int(0.10 * n)
        train_imgs.update(imgs[:n_train])
        val_imgs.update(imgs[n_train:n_train + n_val])
        test_imgs.update(imgs[n_train + n_val:])

    print(f"\nDataset Image Split Summary:")
    print(f"  Training Set:   {len(train_imgs)} images")
    print(f"  Validation Set: {len(val_imgs)} images")
    print(f"  Test Set:       {len(test_imgs)} images")
    print(f"  Total Images:   {len(train_imgs) + len(val_imgs) + len(test_imgs)} images")

    # Combine weapon annotations and negatives
    df_all = pd.concat([df_weapons, df_negs], ignore_index=True)

    def assign_split(p):
        if p in train_imgs:
            return "train"
        elif p in val_imgs:
            return "val"
        else:
            return "test"

    df_all["split"] = df_all["image_path"].apply(assign_split)

    out_csv = DATA_ROOT / "annotations.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"\nSaved Model 8 dataset to: {out_csv}")
    print(f"Total annotation rows: {len(df_all)}")
    print(f"Split distribution:\n{df_all['split'].value_counts()}")


if __name__ == "__main__":
    main()
