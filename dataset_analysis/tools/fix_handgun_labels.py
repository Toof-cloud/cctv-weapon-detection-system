import pandas as pd

INPUT_CSV = "handgun_train_annotations.csv"
OUTPUT_CSV = "handgun_train_annotations_fixed.csv"

df = pd.read_csv(INPUT_CSV)

print("Before:")
print(df["class"].value_counts())

# Convert all handgun annotations to class 1
df["class"] = 1

print("\nAfter:")
print(df["class"].value_counts())

df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved: {OUTPUT_CSV}")