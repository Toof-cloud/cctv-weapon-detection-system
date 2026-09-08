from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT_DIR / "handgun_train_annotations.csv"
OUTPUT_CSV = ROOT_DIR / "handgun_train_annotations_fixed.csv"

df = pd.read_csv(INPUT_CSV)

print("Before:")
print(df["class"].value_counts())

# Convert all handgun annotations to class 1
df["class"] = 1

print("\nAfter:")
print(df["class"].value_counts())

df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved: {OUTPUT_CSV}")