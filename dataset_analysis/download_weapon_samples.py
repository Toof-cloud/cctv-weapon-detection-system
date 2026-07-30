import fiftyone.zoo as foz

# Download 50 handgun images
handgun_dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="train",
    label_types=["detections"],
    classes=["Handgun"],
    max_samples=50,
)

print("Handgun sample downloaded")

# Download 50 knife images
knife_dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="train",
    label_types=["detections"],
    classes=["Knife"],
    max_samples=50,
)

print("Knife sample downloaded")