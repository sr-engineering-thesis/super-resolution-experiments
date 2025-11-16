import glob
import os

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from models.models import ModelConfig, ModelFactory, SRModelWrapper
from src.metrics import im_psnr, im_ssim
from src.visualizer import SRVisualizer


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


if __name__ == "__main__":
    print("Starting benchmark...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = OmegaConf.load("configs/config.yaml")
    visualizer = SRVisualizer(config)

    dataset = SRDataset(config.data.test.y, config.data.test.x)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    model_configs = [
        ModelConfig(name="edsr_r16f64", scale=2, precision=torch.float32, checkpoint='checkpoints/edsr_finetuned.pth'),
        ModelConfig(name="edsr_r16f64", scale=2, precision=torch.float16, checkpoint='checkpoints/edsr_finetuned.pth'),
        ModelConfig(name="edsr_r16f64", scale=2, precision=torch.bfloat16, checkpoint='checkpoints/edsr_finetuned.pth'),
    ]

    # Create models
    models = [ModelFactory.create(cfg, device) for cfg in model_configs]
    model_names = [cfg.name for cfg in model_configs]

    print("\n=== Benchmark Results ===")
    for cfg, model in zip(model_configs, models):
        results = benchmark(
            model,
            dataloader,
            device,
        )
        print(cfg)
        print(f"{cfg.name}: | MSE: {results['mse']:.4f}, PSNR: {results['psnr']:.4f}, SSIM: {results['ssim']:.4f}")
        print(
            f"{cfg.name}: | MSE std: {results['mse_std']:.4f}, PSNR std: {results['psnr_std']:.4f}, SSIM std: {results['ssim_std']:.4f}\n"
        )

    # print("\n=== Generating Visual Comparison ===")
    # gt_patch, pred_patches, pred_metrics = visualizer.run_models(models)

    # visualizer.plot_results(gt_patch, pred_patches, pred_metrics, model_names, out_prefix="comparison_2x")
