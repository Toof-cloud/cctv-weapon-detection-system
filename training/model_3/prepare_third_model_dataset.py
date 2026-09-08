from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "dataset_analysis" / "3rd_Model_Dataset"
OUTPUT_ROOT = ROOT / "training" / "third_model_data"

HANDGUN_SOURCES = [
    (
        DATASET_ROOT / "Handgun" / "First_Clean_Jabez_Annotation"
        / "VERY_CLEANED_annotations (1).csv",
        DATASET_ROOT / "Handgun" / "Handgun_First_Clean_Jabez",
    ),
    (
        DATASET_ROOT / "Handgun" / "Second_Clean_Jabez_Annotation"
        / "annotations.csv",
        DATASET_ROOT / "Handgun" / "Handgun_Second_Clean_Jabez",
    ),
]
KNIFE_CSV = DATASET_ROOT / "Knife" / "First_Clean_Howard_Annotation" / "B_annotations.csv"
KNIFE_IMAGES = DATASET_ROOT / "Knife" / "Knife_First_Clean_Howard"


def load_handgun_annotations(csv_path, image_root, seen_names):
    rows = []
    source = pd.read_csv(csv_path)

    for _, item in source.iterrows():
        image_name = str(item["filename"])
        image_path = image_root / image_name
        relative_path = str(image_path.relative_to(DATASET_ROOT))
        if not image_path.exists():
            continue

        seen_names.add(relative_path)
        rows.append(
            {
                "image_path": relative_path,
                "width": int(item["width"]),
                "height": int(item["height"]),
                "class_id": 1,
                "xmin": float(item["xmin"]),
                "ymin": float(item["ymin"]),
                "xmax": float(item["xmax"]),
                "ymax": float(item["ymax"]),
            }
        )

    return rows


def load_knife_annotations(seen_paths):
    rows = []
    source = pd.read_csv(KNIFE_CSV)

    for _, item in source.iterrows():
        image_name = Path(str(item["image_path"])).name
        image_path = KNIFE_IMAGES / image_name
        relative_path = str(image_path.relative_to(DATASET_ROOT))

        if not image_path.exists():
            continue

        seen_paths.add(relative_path)
        rows.append(
            {
                "image_path": relative_path,
                "width": int(item["image_width"]),
                "height": int(item["image_height"]),
                "class_id": 2,
                "xmin": float(item["x_min"]),
                "ymin": float(item["y_min"]),
                "xmax": float(item["x_max"]),
                "ymax": float(item["y_max"]),
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

    train, holdout = train_test_split(
        image_df,
        test_size=0.30,
        random_state=42,
        stratify=image_df["class_id"],
    )
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

    print(f"Usable images: {len(image_df)}")
    print(f"Handgun images: {(image_df['class_id'] == 1).sum()}")
    print(f"Knife images: {(image_df['class_id'] == 2).sum()}")
    print(f"Train images: {len(train)}")
    print(f"Validation images: {len(validation)}")
    print(f"Test images: {len(test)}")

    for name, split in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ]:
        print(
            f"{name} classes: "
            f"handgun={(split['class_id'] == 1).sum()}, "
            f"knife={(split['class_id'] == 2).sum()}"
        )


if __name__ == "__main__":
    main()