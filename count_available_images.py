import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="train",
    label_types=["detections"],
    classes=["Knife"],
    max_samples=5000,
    dataset_name="openimages-knife-train",
)

print("Downloaded:", len(dataset))