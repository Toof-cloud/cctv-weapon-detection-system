import fiftyone as fo
import fiftyone.zoo as foz

fo.config.dataset_zoo_dir = r"C:\Users\pc\fiftyone"

dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="train",
    label_types=["detections"],
    classes=["Handgun", "Knife"],
)

print(dataset)
print("Images in dataset:", len(dataset))