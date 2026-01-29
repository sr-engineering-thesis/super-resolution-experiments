import os
import random

import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import Compose, GaussianBlur, RandomApply
from torchvision.transforms.functional import to_tensor

from constants import (
    TEST_HR_DATA_PATH,
    TEST_LR_DATA_PATH,
    TRAIN_HR_DATA_PATH,
    TRAIN_LR_DATA_PATH,
)


class QuakeDataset(Dataset):
    def __init__(
        self,
        device,
        split="train",
        val_size=0.1,
        random_seed=42,
        load_to_ram=False,
        augment=False,
    ):
        self.device = device
        self.split = split
        self.load_to_ram = load_to_ram
        random.seed(random_seed)

        # --- Transforms ---
        if augment and self.split == "train":
            print(
                f"Using Gaussian Blur augmentation: kernel_size=5, sigma=(0.1, 2.0), p=0.3",
                flush=True,
            )
            self.augment = Compose(
                [
                    RandomApply(
                        [
                            GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                        ],
                        p=0.3,
                    ),
                ]
            )
        else:
            self.augment = None

        if split == "test":
            self.hr_dir = TEST_HR_DATA_PATH
            self.lr_dir = TEST_LR_DATA_PATH
        else:
            self.hr_dir = TRAIN_HR_DATA_PATH
            self.lr_dir = TRAIN_LR_DATA_PATH

        self.image_list = sorted(
            [
                name
                for name in os.listdir(self.hr_dir)
                if os.path.isfile(os.path.join(self.hr_dir, name))
            ]
        )

        if split in ["train", "val"]:
            train_files, val_files = train_test_split(
                self.image_list, test_size=val_size, random_state=random_seed
            )
            self.image_list = train_files if split == "train" else val_files

        if self.load_to_ram:
            self.lr_images = []
            self.hr_images = []

            for i, fname in enumerate(self.image_list):
                if i % 50 == 0:
                    print(
                        f"Loaded {i}/{len(self.image_list)} images to RAM", flush=True
                    )
                hr_image = cv2.imread(
                    os.path.join(self.hr_dir, fname), cv2.IMREAD_COLOR_RGB
                )
                hr_image = cv2.cvtColor(hr_image, cv2.COLOR_RGB2YCrCb)[:, :, 0]
                hr_tensor = to_tensor(hr_image)

                lr_image = cv2.imread(
                    os.path.join(self.lr_dir, fname), cv2.IMREAD_COLOR_RGB
                )
                lr_image = cv2.cvtColor(lr_image, cv2.COLOR_RGB2YCrCb)[:, :, 0]
                lr_tensor = to_tensor(lr_image)

                self.hr_images.append(hr_tensor)
                self.lr_images.append(lr_tensor)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        if self.load_to_ram:
            lr_tensor = self.lr_images[index]
            if self.augment:
                lr_tensor = self.augment(lr_tensor)

            return lr_tensor, self.hr_images[index]
        else:
            fname = self.image_list[index]

            hr_image = cv2.imread(os.path.join(self.hr_dir, fname), cv2.IMREAD_COLOR)
            hr_image = cv2.cvtColor(hr_image, cv2.COLOR_RGB2YCrCb)[:, :, 0]
            hr_tensor = to_tensor(hr_image)

            lr_image = cv2.imread(os.path.join(self.lr_dir, fname), cv2.IMREAD_COLOR)
            lr_image = cv2.cvtColor(lr_image, cv2.COLOR_RGB2YCrCb)[:, :, 0]
            lr_tensor = to_tensor(lr_image)

        return lr_tensor, hr_tensor
