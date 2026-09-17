import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_CSV = (
    "dataset_analysis/clean_annotations_existing.csv"
)


print("Loading annotations...")

df = pd.read_csv(INPUT_CSV)

# Unique image IDs
image_ids = sorted(df["ImageID"].unique())

print(f"Unique images: {len(image_ids)}")

# 70% train
train_ids, temp_ids = train_test_split(
    image_ids,
    test_size=0.30,
    random_state=42,
)

# remaining 30% → 15% val + 15% test
val_ids, test_ids = train_test_split(
    temp_ids,
    test_size=0.50,
    random_state=42,
)

pd.Series(train_ids).to_csv(
    "dataset_analysis/train_ids.txt",
    index=False,
    header=False,
)

pd.Series(val_ids).to_csv(
    "dataset_analysis/val_ids.txt",
    index=False,
    header=False,
)

pd.Series(test_ids).to_csv(
    "dataset_analysis/test_ids.txt",
    index=False,
    header=False,
)

print()
print("Dataset split completed.")
print(f"Train images: {len(train_ids)}")
print(f"Validation images: {len(val_ids)}")
print(f"Test images: {len(test_ids)}")