import os
import glob
import shutil
import random

# Fix seed for reproducibility
random.seed(42)

# Source folder where Kaggle unzipped
RAW_DIR = "raw_data"
BASE_OUT = "dataset"

# Target splits
SPLITS = {"train": 0.70, "val": 0.20, "test": 0.10}

# Discover all images
image_extensions = ("*.jpg", "*.jpeg", "*.png")
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(RAW_DIR, "**", ext), recursive=True))

# Filter to only images that actually have a matching .txt label
valid_pairs = []
for img_path in image_files:
    base_name = os.path.splitext(img_path)[0]
    txt_path = f"{base_name}.txt"
    if os.path.exists(txt_path):
        valid_pairs.append((img_path, txt_path))

print(f"Found {len(valid_pairs)} valid image-label pairs.")

random.shuffle(valid_pairs)
n_total = len(valid_pairs)
n_train = int(n_total * SPLITS["train"])
n_val = int(n_total * SPLITS["val"])

split_assignments = {
    "train": valid_pairs[:n_train],
    "val": valid_pairs[n_train:n_train + n_val],
    "test": valid_pairs[n_train + n_val:]
}

for split_name, pairs in split_assignments.items():
    img_dir = os.path.join(BASE_OUT, split_name, "images")
    lbl_dir = os.path.join(BASE_OUT, split_name, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for img_p, lbl_p in pairs:
        shutil.copy(img_p, os.path.join(img_dir, os.path.basename(img_p)))
        shutil.copy(lbl_p, os.path.join(lbl_dir, os.path.basename(lbl_p)))

    print(f"  [{split_name.upper()}]: {len(pairs)} samples saved.")

print("\nDataset ready in standard YOLO format.")