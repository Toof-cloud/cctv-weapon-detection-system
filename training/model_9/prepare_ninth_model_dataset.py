import os
import cv2
import random
import numpy as np
import pandas as pd
from pathlib import Path

# Fix random seed for strict reproducibility
random.seed(42)
np.random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "dataset_analysis" / "3rd_Model_Dataset"

MODEL8_DATA_CSV = ROOT / "training" / "eighth_model_data" / "annotations.csv"
MODEL9_NEG_DIR = DATASET_ROOT / "Model9_Hard_Negatives"
MODEL9_NEG_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = ROOT / "training" / "ninth_model_data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)


def extract_clip08_dining_negatives(frame_count: int = 60):
    """
    Extracts authentic surveillance negatives from Clip 08 (peaceful indoor dining).
    Contains people holding beverage bottles, cups, smartphones, wallets, food trays,
    and seated at tables with chair/furniture edges.
    """
    vpath = ROOT / "samples" / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS" / "Screen Recording 2026-09-07 230942.mp4"
    if not vpath.exists():
        print(f"  [!] Clip 08 not found at {vpath}")
        return []

    cap = cv2.VideoCapture(str(vpath))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    indices = np.linspace(15, total_frames - 15, frame_count, dtype=int)
    neg_paths = []
    extracted = 0

    for f_idx in indices:
        out_name = f"clip08_dining_f{f_idx:05d}.jpg"
        out_file = MODEL9_NEG_DIR / out_name
        rel_path = f"Model9_Hard_Negatives\\{out_name}"

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
    print(f"  -> Curated {extracted} dining/bottle/phone negative frames from Clip 08")
    return neg_paths


def extract_peaceful_counter_negatives(frame_count: int = 50):
    """
    Extracts peace-time retail/counter frames before any robbery occurs.
    Includes cashier cash registers, POS card terminals, counter glass, and customer interactions.
    """
    neg_paths = []
    clips = [
        ("Screen Recording 2026-09-07 230249.mp4", "clip05_counter", 1, 60),  # Jimmy John's before robbery
        ("Screen Recording 2026-09-07 231101.mp4", "clip09_counter", 1, 90),  # Pharmacy before robbery
        ("Screen Recording 2026-09-07 231205.mp4", "clip10_counter", 420, 480), # Supermarket counter post-incident
    ]

    base_dir = ROOT / "samples" / "unseen_samples" / "VIDEO_CLIPS_JABEZ_UNSEEN" / "VIDEO CLIPS"

    for cfile, prefix, start_f, end_f in clips:
        vpath = base_dir / cfile
        if not vpath.exists():
            continue
        cap = cv2.VideoCapture(str(vpath))
        num_to_take = frame_count // len(clips)
        f_indices = np.linspace(start_f, end_f, num_to_take, dtype=int)
        for f_idx in f_indices:
            out_name = f"{prefix}_f{f_idx:05d}.jpg"
            out_file = MODEL9_NEG_DIR / out_name
            rel_path = f"Model9_Hard_Negatives\\{out_name}"
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
            else:
                neg_paths.append(rel_path)
        cap.release()

    print(f"  -> Curated {len(neg_paths)} peaceful retail/counter/pinpad negative frames")
    return neg_paths


def extract_staged_scene_negatives(frame_count: int = 40):
    """Extracts empty and non-weapon frames from staged scenes."""
    neg_paths = []
    cam01_empty = ROOT / "samples" / "staged_dataset" / "CAM01" / "CAMERA 01" / "CAM01_Scene001_.mp4"
    cam02_empty = ROOT / "samples" / "staged_dataset" / "CAM02" / "CAMERA 02" / "CAM02_Scene 001.mp4"

    for vpath, prefix in [(cam01_empty, "cam01_scene1"), (cam02_empty, "cam02_scene1")]:
        if not vpath.exists():
            continue
        cap = cv2.VideoCapture(str(vpath))
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if tot <= 0:
            cap.release()
            continue
        indices = np.linspace(2, tot - 2, frame_count // 2, dtype=int)
        for f_idx in indices:
            out_name = f"{prefix}_f{f_idx:04d}.jpg"
            out_file = MODEL9_NEG_DIR / out_name
            rel_path = f"Model9_Hard_Negatives\\{out_name}"
            if not out_file.exists():
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    cv2.imwrite(str(out_file), frame)
                    neg_paths.append(rel_path)
            else:
                neg_paths.append(rel_path)
        cap.release()

    print(f"  -> Curated {len(neg_paths)} staged indoor environment negative frames")
    return neg_paths


def main():
    print("=" * 76)
    print("PREPARING MODEL 9 DATASET: SURVEILLANCE HARD NEGATIVES & WEAPON AUGMENTATION")
    print("=" * 76)

    # 1. Load Model 8 base dataset
    df_m8 = pd.read_csv(MODEL8_DATA_CSV)
    df_weapons = df_m8[df_m8["class_id"] > 0].copy()
    existing_negs = df_m8[df_m8["class_id"] == 0]["image_path"].tolist()

    print(f"Loaded existing Model 8 weapons: {len(df_weapons)} boxes across {df_weapons['image_path'].nunique()} images.")
    print(f"Existing Model 8 hard negatives: {len(existing_negs)} images.")

    # 2. Extract new real-world CCTV negative categories:
    # A. Dining/beverage bottles/smartphones (Clip 08)
    dining_negs = extract_clip08_dining_negatives(frame_count=60)

    # B. Peaceful counter/card pinpad/register frames (Clip 05, 09, 10)
    counter_negs = extract_peaceful_counter_negatives(frame_count=50)

    # C. Staged indoor environmental fixtures (CAM01/CAM02 Scene 001)
    indoor_negs = extract_staged_scene_negatives(frame_count=40)

    all_new_negs = sorted(list(set(dining_negs + counter_negs + indoor_negs)))
    print(f"New Hard Negatives Curated for Model 9: {len(all_new_negs)} frames.")

    all_negatives = sorted(list(set(existing_negs + all_new_negs)))
    print(f"Total Combined Hard Negatives for Model 9: {len(all_negatives)} images.")

    # 3. Create negative annotation entries
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

    # 4. Stratified 80/10/10 split
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

    for rel_p in all_negatives:
        img_groups[rel_p] = "negative"

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

    print(f"\nModel 9 Dataset Image Split Summary:")
    print(f"  Training Set:   {len(train_imgs)} images")
    print(f"  Validation Set: {len(val_imgs)} images")
    print(f"  Test Set:       {len(test_imgs)} images")
    print(f"  Total Images:   {len(train_imgs) + len(val_imgs) + len(test_imgs)} images")

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
    print(f"\nSaved Model 9 dataset to: {out_csv}")
    print(f"Total annotation rows: {len(df_all)}")
    print(f"Split distribution:\n{df_all['split'].value_counts()}")
    print(f"Class distribution:\n{df_all['class_id'].value_counts()}")


if __name__ == "__main__":
    main()
