import os
from tqdm import tqdm
import cv2


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

    print(f"{len(image_files)} color images processed and saved (2× scale).")
