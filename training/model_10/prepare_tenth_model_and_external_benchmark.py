"""
Prepare Model 10 Dataset (Held Non-Weapon Hard Negatives + Knife/Pistol Recall Boost)
and the Unified External Benchmark Test Split (Zero Leakage Across Models 1 to 10).
"""
import os
import random
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
COSTI_BASE = ROOT / "data" / "costi_yolov12"
MODEL9_DATA = ROOT / "training" / "model_9" / "ninth_model_data"
MODEL10_DATA = ROOT / "training" / "model_10" / "tenth_model_data"
UNIFIED_BENCH_DIR = ROOT / "training" / "unified_external_benchmark"
COSTI_CURATED_IMAGES_DIR = ROOT / "dataset_analysis" / "3rd_Model_Dataset" / "costi_yolov12_curated"


def long_path(p: Path) -> str:
    s = str(p.resolve()) if not str(p).startswith("\\\\?\\") else str(p)
    return s if s.startswith("\\\\?\\") else "\\\\?\\" + s


def build_costi_image_index() -> dict[str, Path]:
    img_map = {}
    for sub in ["nogun_v5", "weapon_v11"]:
        for s in ["train", "valid", "test"]:
            d = COSTI_BASE / sub / s / "images"
            if d.exists():
                for p in Path(long_path(d)).iterdir():
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        img_map[p.stem] = p
    return img_map


def parse_yolo_label_file(lbl_path: Path):
    """
    CostiCatargiu corrected label classes:
      0: knife
      1: long_gun
      2: no_weapon
      3: pistol
    Returns list of (cls_id, cx, cy, w, h).
    """
    boxes = []
    with open(str(lbl_path), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            vals = [float(x) for x in parts[1:]]
            if len(vals) == 4:
                cx, cy, bw, bh = vals
            else:
                # polygon fallback
                xs = vals[0::2]
                ys = vals[1::2]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                bw, bh = x2 - x1, y2 - y1
            boxes.append((cls_id, cx, cy, bw, bh))
    return boxes


def scan_costi_split(split_name: str, img_map: dict[str, Path]):
    lbl_dir = Path(long_path(COSTI_BASE / "repo" / "DatasetLabels" / split_name / "labels"))
    pure_no_weapon = []
    knife_records = []
    pistol_records = []

    for lp in sorted(lbl_dir.iterdir(), key=lambda x: x.name):
        stem = lp.stem
        if stem not in img_map:
            continue
        boxes = parse_yolo_label_file(lp)
        classes = {b[0] for b in boxes}
        # Exclude any image with long_gun (class 1)
        if 1 in classes:
            continue

        img_path = img_map[stem]
        if classes == {2} or len(classes) == 0:
            pure_no_weapon.append((stem, img_path, []))
        else:
            # Weapon boxes: 0 -> Knife (our class 2), 3 -> Handgun (our class 1)
            # Class 2 (no_weapon) boxes in the same image are left unlabeled as in-scene hard negatives!
            weapon_boxes = []
            for cls_id, cx, cy, bw, bh in boxes:
                if cls_id == 0:
                    weapon_boxes.append((2, "Knife", cx, cy, bw, bh))
                elif cls_id == 3:
                    weapon_boxes.append((1, "Handgun", cx, cy, bw, bh))
            if not weapon_boxes:
                continue
            w_classes = {wb[0] for wb in weapon_boxes}
            has_inscene_noweapon = (2 in classes)
            if w_classes == {2}:
                knife_records.append((stem, img_path, weapon_boxes, has_inscene_noweapon))
            elif w_classes == {1}:
                pistol_records.append((stem, img_path, weapon_boxes, has_inscene_noweapon))

    return pure_no_weapon, knife_records, pistol_records


def copy_and_convert_records(records, split_tag: str, dest_dir: Path) -> list[dict]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, item in enumerate(records):
        stem, src_img_path, weapon_boxes = item[0], item[1], item[2]
        short_hash = stem.split(".rf.")[-1] if ".rf." in stem else f"{idx:05d}"
        prefix = stem[:40].replace(" ", "_")
        out_name = f"costi_{split_tag}_{prefix}_{short_hash}.jpg"
        out_img_path = dest_dir / out_name
        rel_path = f"costi_yolov12_curated/{out_name}"

        try:
            with Image.open(long_path(src_img_path)) as im:
                im_rgb = im.convert("RGB")
                W, H = im_rgb.size
                if not out_img_path.exists():
                    im_rgb.save(long_path(out_img_path), quality=95)
        except Exception as e:
            print(f"  [!] Skipping unreadable image {src_img_path.name}: {e}")
            continue

        if not weapon_boxes:
            # Class 0 Hard Negative (held non-weapon confounder)
            rows.append({
                "image_path": rel_path,
                "xmin": 0.0,
                "ymin": 0.0,
                "xmax": 0.0,
                "ymax": 0.0,
                "class_id": 0,
                "class_name": "Negative",
                "split": split_tag,
                "width": W,
                "height": H,
            })
        else:
            for cid, cname, cx, cy, bw, bh in weapon_boxes:
                x1 = max(0.0, (cx - bw / 2.0) * W)
                y1 = max(0.0, (cy - bh / 2.0) * H)
                x2 = min(float(W), (cx + bw / 2.0) * W)
                y2 = min(float(H), (cy + bh / 2.0) * H)
                if (x2 - x1) >= 4.0 and (y2 - y1) >= 4.0:
                    rows.append({
                        "image_path": rel_path,
                        "xmin": round(x1, 2),
                        "ymin": round(y1, 2),
                        "xmax": round(x2, 2),
                        "ymax": round(y2, 2),
                        "class_id": cid,
                        "class_name": cname,
                        "split": split_tag,
                        "width": W,
                        "height": H,
                    })
    return rows


def main():
    random.seed(42)
    print("=" * 80)
    print("BUILDING MODEL 10 DATASET & UNIFIED EXTERNAL BENCHMARK (MODELS 1-10)")
    print("=" * 80)

    img_map = build_costi_image_index()
    print(f"[*] Indexed {len(img_map)} images from NoGUN v5 + WeaponDataset v11.")

    # 1. Scan CostiCatargiu splits
    tr_noweap, tr_knife, tr_pistol = scan_costi_split("train", img_map)
    va_noweap, va_knife, va_pistol = scan_costi_split("valid", img_map)
    te_noweap, te_knife, te_pistol = scan_costi_split("test", img_map)

    print(f"[*] Costi Train Pool: {len(tr_noweap)} no_weapon, {len(tr_knife)} knife, {len(tr_pistol)} pistol")
    print(f"[*] Costi Valid Pool: {len(va_noweap)} no_weapon, {len(va_knife)} knife, {len(va_pistol)} pistol")
    print(f"[*] Costi Test  Pool: {len(te_noweap)} no_weapon, {len(te_knife)} knife, {len(te_pistol)} pistol")

    # Prioritize held-object categories (wallet, water_bottle, phone, umbrella, selfie, etc.) + images with in-scene no_weapon
    def prioritize_held_noweapon(pool, k):
        priority = [x for x in pool if any(tag in x[0].lower() for tag in ["wallet", "bottle", "phone", "cell", "umbrella", "selfie", "remote", "tool", "mug", "cup", "broom", "brush", "card"])]
        rest = [x for x in pool if x not in priority]
        random.shuffle(priority)
        random.shuffle(rest)
        combined = priority + rest
        return combined[:k]

    def prioritize_weapons(pool, k):
        # Prioritize images that ALSO contain an in-scene no_weapon confounder + diverse CCTV/YouTube frames
        with_noweap = [x for x in pool if x[3]]
        without_noweap = [x for x in pool if not x[3]]
        random.shuffle(with_noweap)
        random.shuffle(without_noweap)
        take_mixed = min(len(with_noweap), k // 2)
        return with_noweap[:take_mixed] + without_noweap[:(k - take_mixed)]

    # Curate Model 10 additions from Costi train & valid
    m10_train_noweap = prioritize_held_noweapon(tr_noweap, 450)
    m10_train_knife = prioritize_weapons(tr_knife, 350)
    m10_train_pistol = prioritize_weapons(tr_pistol, 200)

    m10_val_noweap = prioritize_held_noweapon(va_noweap, 60)
    m10_val_knife = prioritize_weapons(va_knife, 40)
    m10_val_pistol = prioritize_weapons(va_pistol, 30)

    # Curate External Benchmark additions from Costi test (100% unseen by all Models 1-10)
    ext_test_noweap = prioritize_held_noweapon(te_noweap, 60)
    ext_test_knife = prioritize_weapons(te_knife, 85)
    ext_test_pistol = prioritize_weapons(te_pistol, 20)

    print("\n[*] Copying & converting curated CostiCatargiu images into 3rd_Model_Dataset/costi_yolov12_curated ...")
    rows_tr = copy_and_convert_records(m10_train_noweap + m10_train_knife + m10_train_pistol, "train", COSTI_CURATED_IMAGES_DIR)
    rows_va = copy_and_convert_records(m10_val_noweap + m10_val_knife + m10_val_pistol, "val", COSTI_CURATED_IMAGES_DIR)
    rows_te = copy_and_convert_records(ext_test_noweap + ext_test_knife + ext_test_pistol, "test", COSTI_CURATED_IMAGES_DIR)

    df_costi_tr = pd.DataFrame(rows_tr)
    df_costi_va = pd.DataFrame(rows_va)
    df_costi_te = pd.DataFrame(rows_te)

    # 2. Build Model 10 Dataset (Base Model 9 + Costi Curated Train/Val, keeping Model 9 Test untouched + Costi Test)
    MODEL10_DATA.mkdir(parents=True, exist_ok=True)
    df_m9 = pd.read_csv(MODEL9_DATA / "annotations.csv")
    m9_train_ids = [l.strip() for l in (MODEL9_DATA / "train_ids.txt").read_text().splitlines() if l.strip()]
    m9_val_ids = [l.strip() for l in (MODEL9_DATA / "val_ids.txt").read_text().splitlines() if l.strip()]
    m9_test_ids = [l.strip() for l in (MODEL9_DATA / "test_ids.txt").read_text().splitlines() if l.strip()]

    m10_train_ids = sorted(set(m9_train_ids) | set(df_costi_tr["image_path"].unique()))
    m10_val_ids = sorted(set(m9_val_ids) | set(df_costi_va["image_path"].unique()))
    # Keep m9_test_ids as Model 10's primary test_ids.txt so Model 9 vs Model 10 is 100% apples-to-apples on the 279-image test set
    m10_test_ids = sorted(set(m9_test_ids))

    df_m10 = pd.concat([df_m9, df_costi_tr, df_costi_va, df_costi_te], ignore_index=True)
    df_m10.to_csv(MODEL10_DATA / "annotations.csv", index=False)
    (MODEL10_DATA / "train_ids.txt").write_text("\n".join(m10_train_ids) + "\n", encoding="utf-8")
    (MODEL10_DATA / "val_ids.txt").write_text("\n".join(m10_val_ids) + "\n", encoding="utf-8")
    (MODEL10_DATA / "test_ids.txt").write_text("\n".join(m10_test_ids) + "\n", encoding="utf-8")

    print(f"[*] Saved Model 10 Dataset to {MODEL10_DATA}:")
    print(f"    Train images: {len(m10_train_ids)} (Base M9: {len(m9_train_ids)} + Costi: {df_costi_tr['image_path'].nunique()})")
    print(f"    Val   images: {len(m10_val_ids)} (Base M9: {len(m9_val_ids)} + Costi: {df_costi_va['image_path'].nunique()})")
    print(f"    Test  images (M9 Benchmark): {len(m10_test_ids)}")

    # 3. Build Adviser's Unified External Benchmark Test Split (Zero overlap with ANY Model 1-10 train/val)
    UNIFIED_BENCH_DIR.mkdir(parents=True, exist_ok=True)
    train_val_union = set(m10_train_ids) | set(m10_val_ids)
    names = {3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth"}
    for m in range(3, 10):
        d = ROOT / "training" / f"model_{m}" / f"{names[m]}_model_data"
        for s in ["train", "val"]:
            sf = d / f"{s}_ids.txt"
            if sf.exists():
                train_val_union |= {l.strip() for l in sf.read_text().splitlines() if l.strip()}
        if (d / "annotations.csv").exists():
            df_m = pd.read_csv(d / "annotations.csv")
            if "split" in df_m.columns:
                train_val_union |= set(df_m[df_m["split"].isin(["train", "val"])]["image_path"].unique())

    # Internal images from Model 9 test set that were NEVER in any Model 1-10 train or val set
    strictly_unseen_internal_ids = sorted(set(m9_test_ids) - train_val_union)
    costi_external_test_ids = sorted(set(df_costi_te["image_path"].unique()))
    unified_external_ids = sorted(set(strictly_unseen_internal_ids) | set(costi_external_test_ids))

    # Verify zero overlap
    overlap = set(unified_external_ids) & train_val_union
    assert len(overlap) == 0, f"Leakage detected: {len(overlap)} images!"

    df_unified = df_m10[df_m10["image_path"].isin(unified_external_ids)].copy()
    df_unified["split"] = "test"
    df_unified.to_csv(UNIFIED_BENCH_DIR / "annotations.csv", index=False)
    (UNIFIED_BENCH_DIR / "test_ids.txt").write_text("\n".join(unified_external_ids) + "\n", encoding="utf-8")
    (UNIFIED_BENCH_DIR / "strictly_unseen_internal_ids.txt").write_text("\n".join(strictly_unseen_internal_ids) + "\n", encoding="utf-8")
    (UNIFIED_BENCH_DIR / "costi_external_test_ids.txt").write_text("\n".join(costi_external_test_ids) + "\n", encoding="utf-8")

    print(f"\n[*] Saved Unified External Benchmark to {UNIFIED_BENCH_DIR}:")
    print(f"    Strictly Unseen Internal CCTV Images: {len(strictly_unseen_internal_ids)}")
    print(f"    Strictly Unseen Costi Test Images:    {len(costi_external_test_ids)}")
    print(f"    Total Unified External Test Images:   {len(unified_external_ids)} (0% overlap with Models 1-10 train/val!)")
    for cid, cname in [(0, "Negative (incl. Held Non-Weapons)"), (1, "Handgun"), (2, "Knife")]:
        cnt = df_unified[df_unified["class_id"] == cid]["image_path"].nunique()
        boxes = len(df_unified[df_unified["class_id"] == cid])
        print(f"      - Class {cid} ({cname}): {cnt} images ({boxes} records)")


if __name__ == "__main__":
    main()
