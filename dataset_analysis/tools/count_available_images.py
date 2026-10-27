import fiftyone.zoo as foz

for split in ["train", "validation", "test"]:
    dataset = foz.load_zoo_dataset(
        "open-images-v6",
        split=split,
        label_types=["detections"],
        classes=["Knife"],
        dataset_name=f"knife-{split}",
    )

    print(f"{split}: {len(dataset)}")