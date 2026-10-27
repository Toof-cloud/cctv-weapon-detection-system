import fiftyone.zoo as foz

print("Downloading handgun dataset...")

foz.load_zoo_dataset(
    "open-images-v6",
    split="train",
    label_types=["detections"],
    classes=["Handgun"],
    max_samples=300,
    shuffle=False,
    dataset_name="reviewer-c-handguns",
)

print("Downloading knife dataset...")

foz.load_zoo_dataset(
    "open-images-v6",
    split="train",
    label_types=["detections"],
    classes=["Knife"],
    max_samples=300,
    shuffle=False,
    dataset_name="reviewer-c-knives",
)

print("Download complete")