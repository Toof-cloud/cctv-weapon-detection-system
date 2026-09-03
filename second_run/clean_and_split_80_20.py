# save as: second_run/clean_and_split_80_20.py

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset_analysis"
INVALID_DIR = DATASET_DIR / "ids"

# Use the repo's cleaned CSV as the source of box annotations
CSV_PATH = DATASET_DIR / "clean_annotations_existing.csv"

# Load invalid IDs from the repo
invalid_ids = set()
for file_name in [
    "invalid_all_ids.txt",
    "invalid_handgun_ids.txt",
    "invalid_knife_ids.txt",
]:
    path = INVALID_DIR / file_name
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            invalid_ids.update(line.strip() for line in f if line.strip())

df = pd.read_csv(CSV_PATH)

print("Original rows:", len(df))
print("Original unique image IDs:", df["ImageID"].nunique())

# Remove invalid IDs
df = df[~df["ImageID"].isin(invalid_ids)].copy()

print("After invalid-ID filtering:", len(df))
print("After filtering unique image IDs:", df["ImageID"].nunique())

# Keep only the valid images used in training
image_ids = sorted(df["ImageID"].unique())

# 80% train / 20% holdout
train_ids, holdout_ids = train_test_split(
    image_ids,
    test_size=0.20,
    random_state=42,
)

# Optional: split holdout into validation/test if you want 80/10/10
val_ids, test_ids = train_test_split(
    holdout_ids,
    test_size=0.50,
    random_state=42,
)

# Save IDs
(DATASET_DIR / "train_ids.txt").write_text(
    "\n".join(train_ids) + "\n",
    encoding="utf-8",
)
(DATASET_DIR / "val_ids.txt").write_text(
    "\n".join(val_ids) + "\n",
    encoding="utf-8",
)
(DATASET_DIR / "test_ids.txt").write_text(
    "\n".join(test_ids) + "\n",
    encoding="utf-8",
)

print()
print("Train images:", len(train_ids))
print("Validation images:", len(val_ids))
print("Test images:", len(test_ids))
print("Total unique valid images:", len(train_ids) + len(val_ids) + len(test_ids))