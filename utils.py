import argparse
import os
import random
import shutil
from pathlib import Path

import cv2
import torch
from tqdm import tqdm


def generate_2x_lr_dataset_color(hr_dir: str, lr_dir: str, scale: int = 2):
    image_files = [f for f in os.listdir(hr_dir) if f.lower().endswith(".png")]

    for filename in tqdm(image_files, desc="Generating 2x LR color images"):
        hr_path = os.path.join(hr_dir, filename)
        lr_path = os.path.join(lr_dir, filename)

        hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)

        h, w, _ = hr_img.shape
        h = (h // scale) * scale
        w = (w // scale) * scale
        hr_img = hr_img[:h, :w, :]

        lr = cv2.resize(hr_img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(lr_path, lr)

    print(f"{len(image_files)} color images processed and saved ({scale}× scale).")


def move_subset(train_hr, train_lr, eval_hr, eval_lr, n_samples=500, seed=42):
    random.seed(seed)

    hr_files = sorted(os.listdir(train_hr))
    lr_files = sorted(os.listdir(train_lr))

    assert len(hr_files) == len(lr_files), "HR and LR counts do not match"
    assert set(hr_files) == set(lr_files), "HR and LR filenames differ"

    subset = random.sample(hr_files, n_samples)

    for fname in subset:
        src_hr = Path(train_hr) / fname
        src_lr = Path(train_lr) / fname

        dst_hr = Path(eval_hr) / fname
        dst_lr = Path(eval_lr) / fname

        shutil.move(str(src_hr), str(dst_hr))
        shutil.move(str(src_lr), str(dst_lr))

    print(f"Moved {n_samples} pairs from train → eval.")


def to_tensor(img):
    return torch.from_numpy(img).float().permute(2, 0, 1) / 255.0

def export_to_pt(lr_dir, hr_dir, output_path="quake_dataset.pt"):
    lr_imgs = []
    hr_imgs = []

    files = sorted([
        f for f in os.listdir(hr_dir)
        if f.endswith(".png") and os.path.exists(os.path.join(lr_dir, f))
    ])

    print(f"Found {len(files)} image pairs.")

    for fname in tqdm(files, desc="Processing images"):
        hr = cv2.imread(os.path.join(hr_dir, fname), cv2.IMREAD_COLOR)
        lr = cv2.imread(os.path.join(lr_dir, fname), cv2.IMREAD_COLOR)

        hr_imgs.append(to_tensor(hr))
        lr_imgs.append(to_tensor(lr))

    torch.save({"lr": lr_imgs, "hr": hr_imgs, "files": files}, output_path)
    print(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    lr_dir = "data/val/lr"
    hr_dir = "data/val/hr"
    export_to_pt(lr_dir, hr_dir, "quake_dataset_val.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 2x LR color images from HR images.")
    parser.add_argument("--hr_dir", type=str, required=True, help="Directory containing high-resolution color images.")
    parser.add_argument("--lr_dir", type=str, required=True, help="Directory to save low-resolution color images.")
    args = parser.parse_args()

    os.makedirs(args.lr_dir, exist_ok=True)
    generate_2x_lr_dataset_color(args.hr_dir, args.lr_dir, scale=3)
