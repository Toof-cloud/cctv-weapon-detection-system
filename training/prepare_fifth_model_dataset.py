import os
import json
import cv2
import random
import numpy as np
import pandas as pd
from pathlib import Path

# Fix random seed for reproducibility
random.seed(42)
np.random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "dataset_analysis" / "3rd_Model_Dataset"
OUTPUT_DIR = DATASET_ROOT / "Handgun" / "Handgun_USRT_CCTV"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NEG_OUTPUT_DIR = DATASET_ROOT / "Handgun" / "No_Gun_USRT_CCTV"
NEG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = ROOT / "training" / "fifth_model_data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

USRT_BASE = r"C:\Users\pc\Downloads\Gun_Action_Recognition_Dataset\Gun_Action_Recognition_Dataset"
HG_DIR = os.path.join(USRT_BASE, "Handgun")
NO_GUN_DIR = os.path.join(USRT_BASE, "No_Gun")

# Howard's knife annotations
HOWARD_SOURCE = ROOT / "training" / "third_model_data" / "annotations.csv"

def extract_usrt_handguns(target_count=1135):
    print("=" * 60)
    print("EXTRACTING USRT CCTV HANDGUN FRAMES")
    print("=" * 60)
    videos = sorted(os.listdir(HG_DIR))
    print(f"Total Handgun video scenes: {len(videos)}")

    # First pass: collect available annotated frames per video
    video_annotated_frames = {}
    for vname in videos:
        vdir = os.path.join(HG_DIR, vname)
        lbl_p = os.path.join(vdir, "label.json")
        vid_p = os.path.join(vdir, "video.mp4")
        if not (os.path.exists(lbl_p) and os.path.exists(vid_p)):
            continue
        with open(lbl_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        annos_by_img = {}
        for a in data.get("annotations", []):
            if a.get("category_id") == 1:
                bbox = a.get("bbox", [])
                if len(bbox) == 4 and bbox[2] >= 3 and bbox[3] >= 3:
                    annos_by_img.setdefault(a["image_id"], []).append(bbox)
        if annos_by_img:
            video_annotated_frames[vname] = annos_by_img

    num_vids = len(video_annotated_frames)
    frames_per_vid = max(1, target_count // num_vids)
    remainder = target_count - (frames_per_vid * num_vids)
    print(f"Valid videos: {num_vids}. Base frames/video: {frames_per_vid}, remainder: {remainder}")

    hg_records = []
    total_extracted = 0

    for idx, (vname, annos_by_img) in enumerate(video_annotated_frames.items()):
        needed = frames_per_vid + (1 if idx < remainder else 0)
        img_ids = sorted(annos_by_img.keys())
        sample_n = min(needed, len(img_ids))
        indices = np.linspace(0, len(img_ids) - 1, sample_n, dtype=int)
        selected_ids = [img_ids[i] for i in indices]

        vid_p = os.path.join(HG_DIR, vname, "video.mp4")
        cap = cv2.VideoCapture(vid_p)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for frame_id in selected_ids:
            out_fname = f"{vname}_f{frame_id:04d}.jpg"
            out_fpath = OUTPUT_DIR / out_fname
            rel_path = f"Handgun\\Handgun_USRT_CCTV\\{out_fname}"

            if not out_fpath.exists():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
                ret, frame = cap.read()
                if not ret:
                    continue
                cv2.imwrite(str(out_fpath), frame)

            for bbox in annos_by_img[frame_id]:
                bx, by, bw, bh = bbox
                xmin = max(0.0, float(bx))
                ymin = max(0.0, float(by))
                xmax = min(float(w), float(bx + bw))
                ymax = min(float(h), float(by + bh))
                if xmax > xmin + 2 and ymax > ymin + 2:
                    hg_records.append({
                        "image_path": rel_path,
                        "width": w,
                        "height": h,
                        "class_id": 1,
                        "xmin": round(xmin, 1),
                        "ymin": round(ymin, 1),
                        "xmax": round(xmax, 1),
                        "ymax": round(ymax, 1)
                    })
            total_extracted += 1
        cap.release()

    df_hg = pd.DataFrame(hg_records)
    print(f"Extracted {df_hg['image_path'].nunique()} Handgun images with {len(df_hg)} bounding boxes.")
    return df_hg

def extract_no_gun_negatives(target_count=100):
    print("=" * 60)
    print("EXTRACTING USRT CCTV NEGATIVE (NO_GUN) FRAMES")
    print("=" * 60)
    videos = sorted(os.listdir(NO_GUN_DIR))[:target_count]
    neg_paths = []
    for vname in videos:
        vid_p = os.path.join(NO_GUN_DIR, vname, "video.mp4")
        if not os.path.exists(vid_p):
            continue
        cap = cv2.VideoCapture(vid_p)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_f <= 0:
            cap.release()
            continue
        mid_f = total_f // 2
        out_fname = f"{vname}_f{mid_f:04d}.jpg"
        out_fpath = NEG_OUTPUT_DIR / out_fname
        rel_path = f"Handgun\\No_Gun_USRT_CCTV\\{out_fname}"

        if not out_fpath.exists():
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_f)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(str(out_fpath), frame)
                neg_paths.append(rel_path)
        else:
            neg_paths.append(rel_path)
        cap.release()

    print(f"Extracted {len(neg_paths)} negative (No_Gun) frames.")
    return neg_paths

def main():
    # 1. Extract Handgun frames
    df_hg = extract_usrt_handguns(target_count=1135)

    # 2. Extract Negative frames
    neg_paths = extract_no_gun_negatives(target_count=100)

    # 3. Load Howard's clean Knife dataset
    df_all_third = pd.read_csv(HOWARD_SOURCE)
    df_knife = df_all_third[df_all_third["class_id"] == 2].copy()
    print(f"Loaded {df_knife['image_path'].nunique()} Knife images with {len(df_knife)} bounding boxes.")

    # 4. Combine annotations
    df_combined = pd.concat([df_hg, df_knife], ignore_index=True)
    csv_out = DATA_ROOT / "annotations.csv"
    df_combined.to_csv(csv_out, index=False)
    print(f"Saved unified annotations to: {csv_out}")
    print(f"Total annotations: {len(df_combined)}")

    # 5. Build stratified splits (70% Train, 15% Val, 15% Test)
    hg_images = sorted(df_hg["image_path"].unique().tolist())
    knife_images = sorted(df_knife["image_path"].unique().tolist())
    random.shuffle(hg_images)
    random.shuffle(knife_images)
    random.shuffle(neg_paths)

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
    neg_train, neg_val, neg_test = split_list(neg_paths)

    train_ids = sorted(hg_train + knife_train + neg_train)
    val_ids = sorted(hg_val + knife_val + neg_val)
    test_ids = sorted(hg_test + knife_test + neg_test)

    for sname, sids in [("train", train_ids), ("validation", val_ids), ("test", test_ids)]:
        with open(DATA_ROOT / f"{sname}_ids.txt", "w", encoding="utf-8") as f:
            for item in sids:
                f.write(f"{item}\n")

    print("\n" + "=" * 60)
    print("MODEL 5 DATASET SPLIT SUMMARY")
    print("=" * 60)
    print(f"TRAIN: {len(train_ids)} images (Handgun: {len(hg_train)}, Knife: {len(knife_train)}, Neg: {len(neg_train)})")
    print(f"VAL:   {len(val_ids)} images (Handgun: {len(hg_val)}, Knife: {len(knife_val)}, Neg: {len(neg_val)})")
    print(f"TEST:  {len(test_ids)} images (Handgun: {len(hg_test)}, Knife: {len(knife_test)}, Neg: {len(neg_test)})")
    print(f"TOTAL: {len(train_ids) + len(val_ids) + len(test_ids)} images")

    # Compute bounding box area stats
    df_combined["box_area_pct"] = (
        (df_combined["xmax"] - df_combined["xmin"]) * (df_combined["ymax"] - df_combined["ymin"])
    ) / (df_combined["width"] * df_combined["height"]) * 100.0

    print("\nBounding Box Area (% of frame):")
    print("HANDGUN:")
    print(df_combined[df_combined["class_id"] == 1]["box_area_pct"].describe())
    print("\nKNIFE:")
    print(df_combined[df_combined["class_id"] == 2]["box_area_pct"].describe())

if __name__ == "__main__":
    main()
