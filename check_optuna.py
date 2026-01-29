import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from models.models import ModelConfig, ModelFactory, SRModelWrapper
from src.metrics import im_psnr, im_ssim

RUNS_DIRECTORY = Path("optuna_runs")
# take only when x == 23
RUN_DIRECTORIES = sorted(
    [x for x in RUNS_DIRECTORY.glob("*") if int(x.name.split("_")[-1]) == 24], key=lambda x: int(x.name.split("_")[-1])
)


def get_last_checkpoint_from_run(path: Path):
    checkpoints = sorted((path / "checkpoints").glob("*.pth"))
    return checkpoints[-1] if checkpoints else None


def to_tensor(img):
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(img, (2, 0, 1))).float()


class SRDataset(torch.utils.data.Dataset):
    def __init__(self, hr_dir, lr_dir):
        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
        self.lr_paths = sorted(glob.glob(os.path.join(lr_dir, "*.png")))
        assert len(self.hr_paths) == len(self.lr_paths), "Mismatch HR and LR dataset size"

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr_img = cv2.imread(self.hr_paths[idx])
        lr_img = cv2.imread(self.lr_paths[idx])
        return to_tensor(lr_img), to_tensor(hr_img)


def benchmark(model: SRModelWrapper, dataloader: DataLoader, device: torch.device):
    """Benchmark a model on a dataset."""
    mse_list, psnr_list, ssim_list = [], [], []
    total_images = 0

    with torch.no_grad():
        for i, (lr_tensor, hr_tensor) in enumerate(dataloader):
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)

            output = model(lr_tensor)

            output_np = output.squeeze().cpu().float().numpy()
            output_np = (np.transpose(output_np, (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)

            hr_np = hr_tensor.squeeze().cpu().numpy()
            hr_np = (np.transpose(hr_np, (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)

            mse_list.append(np.mean((hr_np - output_np) ** 2))
            psnr_list.append(im_psnr(hr_np, output_np))
            ssim_list.append(im_ssim(hr_np, output_np))
            total_images += 1

    mse_std = np.std(mse_list)
    psnr_std = np.std(psnr_list)
    ssim_std = np.std(ssim_list)

    return {
        "mse": np.sum(mse_list) / total_images,
        "psnr": np.sum(psnr_list) / total_images,
        "ssim": np.sum(ssim_list) / total_images,
        "mse_std": mse_std,
        "psnr_std": psnr_std,
        "ssim_std": ssim_std,
    }


for run_directory in RUN_DIRECTORIES:
    latest_checkpoint = get_last_checkpoint_from_run(run_directory)
    print(f"Run: {run_directory.name}, Latest Checkpoint: {latest_checkpoint}")


if __name__ == "__main__":
    print("Starting benchmark...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = OmegaConf.load("configs/config.yaml")
    dataset = SRDataset(config.data.test.y, config.data.test.x)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    model_configs = [
        ModelConfig(name="fsrcnn_big", scale=2, precision=torch.float32, checkpoint=get_last_checkpoint_from_run(x))
        for x in RUN_DIRECTORIES
    ]
    models = [ModelFactory.create(cfg, device) for cfg in model_configs]
    model_names = [cfg.name for cfg in model_configs]

    for model, model_name in zip(models, model_names):
        dataset = SRDataset(config.data.test.y, config.data.test.x)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

        metrics = benchmark(model, dataloader, device)
        print(
            f"{model.config.checkpoint}: {metrics['mse']:.4f} +- {metrics['mse_std']:.4f} | {metrics['psnr']:.4f} +- {metrics['psnr_std']:.4f} | {metrics['ssim']:.4f} +- {metrics['ssim_std']:.4f}"
        )
