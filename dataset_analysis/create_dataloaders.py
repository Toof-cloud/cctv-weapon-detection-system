from torch.utils.data import DataLoader

from weapon_dataset import WeaponDataset


def collate_fn(batch):
    return tuple(zip(*batch))


TRAIN_DATASET = WeaponDataset(
    image_ids_file="dataset_analysis/train_ids.txt",
    annotations_csv="dataset_analysis/clean_annotations_existing.csv",
    images_dir=r"C:\Users\pc\fiftyone\open-images-v6\train\data",
)

VAL_DATASET = WeaponDataset(
    image_ids_file="dataset_analysis/val_ids.txt",
    annotations_csv="dataset_analysis/clean_annotations_existing.csv",
    images_dir=r"C:\Users\pc\fiftyone\open-images-v6\train\data",
)

TEST_DATASET = WeaponDataset(
    image_ids_file="dataset_analysis/test_ids.txt",
    annotations_csv="dataset_analysis/clean_annotations_existing.csv",
    images_dir=r"C:\Users\pc\fiftyone\open-images-v6\train\data",
)

train_loader = DataLoader(
    TRAIN_DATASET,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    VAL_DATASET,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_fn,
)

test_loader = DataLoader(
    TEST_DATASET,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_fn,
)

print("Train images:", len(TRAIN_DATASET))
print("Validation images:", len(VAL_DATASET))
print("Test images:", len(TEST_DATASET))

images, targets = next(iter(train_loader))

print()
print("Batch size:", len(images))
print("First image shape:", images[0].shape)
print("First target labels:", targets[0]["labels"])