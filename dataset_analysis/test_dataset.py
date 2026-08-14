from weapon_dataset import WeaponDataset

dataset = WeaponDataset(
    image_ids_file=
    "dataset_analysis/train_ids.txt",

    annotations_csv=
    "dataset_analysis/clean_annotations_existing.csv",

    images_dir=
    r"C:\Users\pc\fiftyone\open-images-v6\train\data",
)

print("Dataset size:", len(dataset))

image, target = dataset[0]

print("Image shape:", image.shape)

print("Boxes:")
print(target["boxes"])

print("Labels:")
print(target["labels"])