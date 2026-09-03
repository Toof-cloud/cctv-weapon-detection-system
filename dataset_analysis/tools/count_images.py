from pathlib import Path

dataset_root = Path(r"C:\path\to\knife_dataset_v2")

for split in ["train", "valid", "test"]:
    image_dir = dataset_root / split / "images"

    count = len(
        list(image_dir.glob("*.jpg"))
    )

    print(
        f"{split}: {count} images"
    )