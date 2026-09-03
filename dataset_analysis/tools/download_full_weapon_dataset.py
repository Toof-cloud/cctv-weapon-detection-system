import fiftyone.zoo as foz

print("Downloading Handgun dataset...")

handgun_dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="train",
    label_types=["detections"],
    classes=["Handgun"],
)

print(f"Handgun images downloaded: {len(handgun_dataset)}")

print("Downloading Knife dataset...")

knife_dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="train",
    label_types=["detections"],
    classes=["Knife"],
)

print(f"Knife images downloaded: {len(knife_dataset)}")

print("Dataset download complete.")