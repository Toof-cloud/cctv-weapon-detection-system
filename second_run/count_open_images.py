import pandas as pd

HANDGUN = "/m/0gxl3"
KNIFE = "/m/04ctx"
KITCHEN_KNIFE = "/m/058qzx"

annotations = pd.read_csv(
    "dataset_analysis/clean_annotations_existing.csv"
)

train_ids = set(
    open(
        "dataset_analysis/train_ids.txt",
        "r",
        encoding="utf-8"
    ).read().splitlines()
)

train_annotations = annotations[
    annotations["ImageID"].isin(train_ids)
]

# Handgun images
handgun_images = train_annotations[
    train_annotations["LabelName"] == HANDGUN
]["ImageID"].nunique()

# Knife images
knife_images = train_annotations[
    train_annotations["LabelName"].isin(
        [KNIFE, KITCHEN_KNIFE]
    )
]["ImageID"].nunique()

# Annotation counts
handgun_boxes = len(
    train_annotations[
        train_annotations["LabelName"] == HANDGUN
    ]
)

knife_boxes = len(
    train_annotations[
        train_annotations["LabelName"].isin(
            [KNIFE, KITCHEN_KNIFE]
        )
    ]
)

print("===== ORIGINAL OPEN IMAGES TRAINING SET =====")
print("Handgun Images:", handgun_images)
print("Knife Images:", knife_images)
print()
print("Handgun Bounding Boxes:", handgun_boxes)
print("Knife Bounding Boxes:", knife_boxes)