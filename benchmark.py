import glob
import os

import cv2
import lpips
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from models.models import ModelConfig, ModelFactory, SRModelWrapper
from src.benchmark_plotter import BenchmarkPlotter, PatchInfo
from src.logger import logger
from src.metrics import im_psnr, im_ssim
from src.visualizer import SRVisualizer


def to_tensor(img):
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(img, (2, 0, 1))).float()


class SRDataset(torch.utils.data.Dataset):
    def __init__(self, hr_dir, lr_dir):
        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, "*.png")))[:2]
        self.lr_paths = sorted(glob.glob(os.path.join(lr_dir, "*.png")))[:2]
        assert len(self.hr_paths) == len(self.lr_paths), "Mismatch HR and LR dataset size"

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr_img = cv2.imread(self.hr_paths[idx])
        lr_img = cv2.imread(self.lr_paths[idx])
        return to_tensor(lr_img), to_tensor(hr_img)


def benchmark(model: SRModelWrapper, dataloader: DataLoader, device: torch.device):
    """Benchmark a model on a dataset."""
    mse_list, psnr_list, ssim_list, lpips_list = [], [], [], []
    total_images = 0

    lpips_fn = lpips.LPIPS(net="vgg").to(device)
    lpips_fn.eval()

    with torch.no_grad():
        for i, (lr_tensor, hr_tensor) in enumerate(dataloader):
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)

            output = model(lr_tensor)
            print(output.max().item(), output.min().item())

            output_lpips = output * 2 - 1
            hr_lpips = hr_tensor * 2 - 1

            lp = lpips_fn(output_lpips, hr_lpips).mean().item()
            lpips_list.append(lp)

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
    perc_std = np.std(lpips_list)

    return {
        "mse": np.sum(mse_list) / total_images,
        "psnr": np.sum(psnr_list) / total_images,
        "ssim": np.sum(ssim_list) / total_images,
        "perc": np.sum(lpips_list) / total_images,
        "mse_std": mse_std,
        "psnr_std": psnr_std,
        "ssim_std": ssim_std,
        "perc_std": perc_std,
    }


if __name__ == "__main__":
    logger.info("Starting benchmark...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    config = OmegaConf.load("configs/config.yaml")
    visualizer = SRVisualizer(config)

    dataset = SRDataset(config.data.test.y, config.data.test.x)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    model_configs = [
        ModelConfig(name="ninasr_b0", scale=2, precision=torch.float16, checkpoint="checkpoints/ninasr_finetuned.pth"),
        ModelConfig(name="ninasr_b0", scale=2, precision=torch.float16, checkpoint="checkpoints/ninasr_finetuned.pth"),
        ModelConfig(name="ninasr_b0", scale=2, precision=torch.bfloat16, checkpoint="checkpoints/ninasr_finetuned.pth"),
    ]
    models = [ModelFactory.create(cfg, device) for cfg in model_configs]
    logger.info(f"Prepared {len(models)} models for benchmarking.")

    PRECISION_MAP = {
        torch.float32: "FP32",
        torch.float16: "FP16",
        torch.bfloat16: "BF16",
    }

    pred_precisions = [PRECISION_MAP[cfg.precision] for cfg in model_configs if cfg is not None]
    model_names = [cfg.name for cfg in model_configs]

    print("\n=== Generating Visual Comparison ===")
    patch_info = PatchInfo(patch_x=640, patch_y=500, offset_x=200, image_idx=111)
    gt_full, pred_fulls, pred_metrics = visualizer.run_models(models, patch_info.image_idx)

    plotter = BenchmarkPlotter(model_names, pred_metrics, pred_precisions, patch_info)
    diff_maps = plotter.compute_global_diff_maps(gt_full, pred_fulls)

    for i, full_img in enumerate([gt_full] + pred_fulls):
        final_image = plotter.render(i, full_img, show_footer=True)
        if isinstance(final_image, tuple):
            plain_img, scaled_img = final_image
            cv2.imwrite("benchmark_comparison_gt_plain.png", plain_img)
            logger.info("Saved ground truth plain comparison image: benchmark_comparison_gt_plain.png")
            cv2.imwrite("benchmark_comparison_gt_scaled.png", scaled_img)
            logger.info("Saved ground truth scaled comparison image: benchmark_comparison_gt_scaled.png")
        else:
            cv2.imwrite(f"benchmark_comparison_{i}.png", final_image)
        if i == 0:
            logger.info(f"Saved ground truth comparison image: benchmark_comparison_{i}.png")
        else:
            logger.info(f"Saved model {model_names[i - 1]} comparison image: benchmark_comparison_{i}.png")
            final_diff_image = plotter.render_diff(i, full_img, diff_maps[i - 1], show_footer=True)
            cv2.imwrite(f"benchmark_diff_{i}.png", final_diff_image)
            logger.info(f"Saved model {model_names[i - 1]} diff image: benchmark_diff_{i}.png")
