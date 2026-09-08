import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "dataset_analysis" / "3rd_Model_Dataset"
OUTPUT_ROOT = ROOT / "training" / "fourth_model_data"

HANDGUN_SOURCES = [
    (
        DATASET_ROOT / "Handgun" / "First_Clean_Jabez_Annotation"
        / "VERY_CLEANED_annotations (1).csv",
        DATASET_ROOT / "Handgun" / "Handgun_First_Clean_Jabez",
    ),
    (
        DATASET_ROOT / "Handgun" / "sanitized_handgun_batch2.csv",
        DATASET_ROOT / "Handgun" / "Handgun_Second_Clean_Jabez",
    ),
]
KNIFE_CSV = DATASET_ROOT / "Knife" / "First_Clean_Howard_Annotation" / "B_annotations.csv"
KNIFE_IMAGES = DATASET_ROOT / "Knife" / "Knife_First_Clean_Howard"


def load_handgun_annotations(csv_path, image_root, seen_names):
    rows = []
    source = pd.read_csv(csv_path)

    for _, item in source.iterrows():
        image_name = str(item["filename"]).strip()
        image_path = image_root / image_name
        relative_path = str(image_path.relative_to(DATASET_ROOT))
        if not image_path.exists():
            continue

        try:
            with Image.open(image_path) as im:
                real_w, real_h = im.size
        except Exception:
            continue

        x1 = float(item["xmin"])
        y1 = float(item["ymin"])
        x2 = float(item["xmax"])
        y2 = float(item["ymax"])

        # Validate bounding box geometry
        if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
            continue

        x1 = max(0.0, min(x1, real_w - 1.0))
        y1 = max(0.0, min(y1, real_h - 1.0))
        x2 = max(x1 + 1.0, min(x2, float(real_w)))
        y2 = max(y1 + 1.0, min(y2, float(real_h)))

        seen_names.add(relative_path)
        rows.append(
            {
                "image_path": relative_path,
                "width": real_w,
                "height": real_h,
                "class_id": 1,
                "xmin": round(x1, 2),
                "ymin": round(y1, 2),
                "xmax": round(x2, 2),
                "ymax": round(y2, 2),
            }
        )

    return rows


def load_knife_annotations(seen_paths):
    rows = []
    source = pd.read_csv(KNIFE_CSV)

    for _, item in source.iterrows():
        image_name = Path(str(item["image_path"])).name.strip()
        image_path = KNIFE_IMAGES / image_name
        relative_path = str(image_path.relative_to(DATASET_ROOT))

        if not image_path.exists():
            continue

        try:
            with Image.open(image_path) as im:
                real_w, real_h = im.size
        except Exception:
            continue

        x1 = float(item["x_min"])
        y1 = float(item["y_min"])
        x2 = float(item["x_max"])
        y2 = float(item["y_max"])

        if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
            continue

        x1 = max(0.0, min(x1, real_w - 1.0))
        y1 = max(0.0, min(y1, real_h - 1.0))
        x2 = max(x1 + 1.0, min(x2, float(real_w)))
        y2 = max(y1 + 1.0, min(y2, float(real_h)))

        seen_paths.add(relative_path)
        rows.append(
            {
                "image_path": relative_path,
                "width": real_w,
                "height": real_h,
                "class_id": 2,
                "xmin": round(x1, 2),
                "ymin": round(y1, 2),
                "xmax": round(x2, 2),
                "ymax": round(y2, 2),
            }
        )

    return rows


def main():
    annotations = []
    seen_handguns = set()

    for csv_path, image_root in HANDGUN_SOURCES:
        annotations.extend(
            load_handgun_annotations(
                csv_path,
                image_root,
                seen_handguns,
            )
        )

    annotations.extend(
        load_knife_annotations(set())
    )

    annotations_df = pd.DataFrame(annotations)
    image_df = annotations_df.drop_duplicates("image_path").copy()
    image_df["class_id"] = image_df["class_id"].astype(int)
    image_df = image_df.sort_values("image_path").reset_index(drop=True)

    # 70% train / 30% holdout
    train, holdout = train_test_split(
        image_df,
        test_size=0.30,
        random_state=42,
        stratify=image_df["class_id"],
    )
    # 15% validation / 15% test
    validation, test = train_test_split(
        holdout,
        test_size=0.50,
        random_state=42,
        stratify=holdout["class_id"],
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    annotations_df.to_csv(
        OUTPUT_ROOT / "annotations.csv",
        index=False,
    )

    for name, split in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        split["image_path"].to_csv(
            OUTPUT_ROOT / f"{name}_ids.txt",
            index=False,
            header=False,
        )

    print("=" * 60)
    print("FOURTH MODEL DATASET PREPARATION REPORT")
    print("=" * 60)
    print(f"Total annotations in dataset: {len(annotations_df)}")
    print(f"Total unique usable images:  {len(image_df)}")
    print(f"Handgun images:              {(image_df['class_id'] == 1).sum()} (Annotations: {(annotations_df['class_id'] == 1).sum()})")
    print(f"Knife images:                {(image_df['class_id'] == 2).sum()} (Annotations: {(annotations_df['class_id'] == 2).sum()})")
    print("-" * 60)
    print(f"Train images:                {len(train)}")
    print(f"Validation images:           {len(validation)}")
    print(f"Test images:                 {len(test)}")
    print("-" * 60)

    for name, split in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ]:
        print(
            f"{name:12} classes: "
            f"handgun={(split['class_id'] == 1).sum():4d}, "
            f"knife={(split['class_id'] == 2).sum():4d}"
        )
    print("=" * 60)
    print(f"Saved dataset split to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
