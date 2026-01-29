import os

import cv2
import torch
from omegaconf import DictConfig


def to_tensor(img):
    img = torch.from_numpy(img).float().permute(2, 0, 1)
    return img / 255.0


class QuakeDataset(torch.utils.data.Dataset):
    def __init__(self, config: DictConfig, split: str = "train"):
        super().__init__()
        self.lr_dir = config.data[split].x
        self.hr_dir = config.data[split].y
        self.patch_size = config.data.patch_size
        self.config = config
        self.scale = config.training.scale

        self.image_list = sorted(
            [f for f in os.listdir(self.hr_dir) if f.endswith(".png") and os.path.exists(os.path.join(self.lr_dir, f))]
        )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        fname = self.image_list[index]

        hr_image = cv2.imread(os.path.join(self.hr_dir, fname), cv2.IMREAD_COLOR)
        lr_image = cv2.imread(os.path.join(self.lr_dir, fname), cv2.IMREAD_COLOR)

        hr_tensor = to_tensor(hr_image)
        lr_tensor = to_tensor(lr_image)

        # Random crop
        _, H, W = lr_tensor.shape
        ps = self.patch_size
        top = torch.randint(0, H - ps, (1,)).item()
        left = torch.randint(0, W - ps, (1,)).item()
        lr_crop = lr_tensor[:, top : top + ps, left : left + ps]
        hr_crop = hr_tensor[
            :,
            top * self.scale : top * self.scale + ps * self.scale,
            left * self.scale : left * self.scale + ps * self.scale,
        ]

        return lr_crop, hr_crop
