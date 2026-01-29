import os
import cv2
import torch
from tqdm import tqdm

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