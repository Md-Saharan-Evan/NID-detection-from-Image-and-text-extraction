import os
import random
import shutil

import albumentations as A
import numpy as np
from PIL import Image


def create_split_dirs(base_dir):
    """Create directory structure for YOLOv8 training."""
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(base_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "labels", split), exist_ok=True)

def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.15):
    """Split images and labels into train/val/test."""
    image_dir = os.path.join(data_dir, "images")
    label_dir = os.path.join(data_dir, "labels")

    images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    random.shuffle(images)

    n = len(images)
    train_n = int(n * train_ratio)
    val_n = int(n * val_ratio)

    splits = {
        "train": images[:train_n],
        "val": images[train_n:train_n + val_n],
        "test": images[train_n + val_n:]
    }

    for split, files in splits.items():
        for file in files:
            src_img = os.path.join(image_dir, file)
            dst_img = os.path.join(data_dir, "images", split, file)
            shutil.move(src_img, dst_img)

            label_file = file.replace(".jpg", ".txt")
            src_lbl = os.path.join(label_dir, label_file)
            dst_lbl = os.path.join(data_dir, "labels", split, label_file)

            if os.path.exists(src_lbl):
                shutil.move(src_lbl, dst_lbl)
            else:
                print(f"[WARNING] Missing label for {file}, skipping label move.")

def create_dataset_yaml(data_dir):
    """Create dataset.yaml for YOLOv8."""
    yaml_path = os.path.join(data_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"""path: {os.path.abspath(data_dir)}
train: images/train
val: images/val
test: images/test
nc: 1
names: ['NID']
""")

# def augment_training_images(data_dir, n_augmentations=3):
#     """Apply augmentation only to training images and copy corresponding labels."""
#     transform = A.Compose([
#         A.Rotate(limit=45, p=0.5),
#         A.RandomBrightnessContrast(p=0.5),
#         A.GaussNoise(p=0.3),
#         A.HueSaturationValue(p=0.3),
#     ])

#     train_img_dir = os.path.join(data_dir, "images", "train")
#     train_lbl_dir = os.path.join(data_dir, "labels", "train")

#     for img_file in os.listdir(train_img_dir):
#         if not img_file.endswith(".jpg"):
#             continue

#         img_path = os.path.join(train_img_dir, img_file)
#         label_file = img_file.replace(".jpg", ".txt")
#         label_path = os.path.join(train_lbl_dir, label_file)

#         if not os.path.exists(label_path):
#             print(f"[WARNING] Label file not found for {img_file}, skipping.")
#             continue

#         image = np.array(Image.open(img_path))
#         for i in range(n_augmentations):
#             augmented = transform(image=image)["image"]
#             aug_img_name = f"aug_{i}_{img_file}"
#             aug_lbl_name = f"aug_{i}_{label_file}"

#             Image.fromarray(augmented).save(os.path.join(train_img_dir, aug_img_name))
#             shutil.copy(label_path, os.path.join(train_lbl_dir, aug_lbl_name))

if __name__ == "__main__":
    data_dir = "data"

    print("[INFO] Creating train/val/test directories...")
    create_split_dirs(data_dir)

    print("[INFO] Splitting dataset...")
    split_dataset(data_dir)

    print("[INFO] Creating dataset.yaml...")
    create_dataset_yaml(data_dir)


    print("[DONE] Dataset is ready for YOLO training.")
