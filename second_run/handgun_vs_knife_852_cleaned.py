import pandas as pd

df = pd.read_csv("dataset_analysis/clean_annotations_existing.csv")

label_map = {
    "/m/0gxl3": "handgun",
    "/m/04ctx": "knife",
    "/m/058qzx": "kitchen_knife",
}

df = df[df["LabelName"].isin(label_map.keys())].copy()
df["class_name"] = df["LabelName"].map(label_map)

counts = df.groupby("class_name")["ImageID"].nunique()
print(counts)