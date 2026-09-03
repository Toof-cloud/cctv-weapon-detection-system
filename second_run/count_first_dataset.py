import pandas as pd
from pathlib import Path

base = Path("dataset_analysis")

# Open Images label mapping used by the project
HANDGUN_LABELS = {"/m/0gxl3"}
KNIFE_LABELS = {"/m/04ctx", "/m/058qzx"}  # knife + kitchen knife

df = pd.read_csv(base / "clean_annotations_existing.csv")

# Load the generated split IDs
split_ids = {}
for split in ["train", "val", "test"]:
    ids_path = base / f"{split}_ids.txt"
    with open(ids_path, "r", encoding="utf-8") as f:
        split_ids[split] = {
            line.strip() for line in f if line.strip()
        }

# Count UNIQUE image IDs per class and split
print("Handgun and Knife counts by split")
print("-" * 50)

for split, image_ids in split_ids.items():
    split_df = df[df["ImageID"].isin(image_ids)].copy()

    handgun_ids = set(split_df[split_df["LabelName"].isin(HANDGUN_LABELS)]["ImageID"])
    knife_ids = set(split_df[split_df["LabelName"].isin(KNIFE_LABELS)]["ImageID"])

    print(f"{split.upper()}:")
    print(f"  Handgun images: {len(handgun_ids)}")
    print(f"  Knife images:   {len(knife_ids)}")
    print(f"  Total unique images in split: {len(image_ids)}")
    print()