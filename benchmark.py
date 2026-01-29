import glob
import os

import cv2
import lpips
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import models as tv_models

from src.benchmark_plotter import BenchmarkPlotter, PatchInfo
from models.models import ModelConfig, ModelFactory, SRModelWrapper
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

    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    lpips_fn.eval()
    
    with torch.no_grad():
        for i, (lr_tensor, hr_tensor) in enumerate(dataloader):
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)

            output = model(lr_tensor)
            print(output.max().item(), output.min().item())
            # perceptual loss
            output_lpips = output * 2 - 1
            hr_lpips = hr_tensor * 2 - 1

            lp = lpips_fn(output_lpips, hr_lpips).mean().item()
            lpips_list.append(lp)

            # pixel metrics
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
    print("Starting benchmark...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = OmegaConf.load("configs/config.yaml")
    visualizer = SRVisualizer(config)

    dataset = SRDataset(config.data.test.y, config.data.test.x)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    model_configs = [
        # ModelConfig(name="fsrcnn_big", scale=2, precision=torch.float32, checkpoint="optuna_runs/trial_24/checkpoints/best_model_20251122_140134.pth")
        # ModelConfig(name="bilinear_torch", scale=2, precision=torch.float32, checkpoint=None),
        # ModelConfig(name="bicubic_torch", scale=2, precision=torch.float32, checkpoint=None),
        # ModelConfig(name="nearest_torch", scale=2, precision=torch.float32, checkpoint=None),
        # ModelConfig(name="lanczos", scale=2, precision=torch.float32, checkpoint=None),
        # ModelConfig(name='ninasr_b0', scale=2, precision=torch.float32, checkpoint=None),
        # ModelConfig(name='rcan', scale=2, precision=torch.float32, checkpoint=None),
        # ModelConfig(name='carn', scale=2, precision=torch.float32, checkpoint=None),
        # ModelConfig(name='carn_m', scale=2, precision=torch.float32, checkpoint=None),
        # ModelConfig(name='edsr_r16f64', scale=2, precision=torch.float32, checkpoint=None),
        # ModelConfig(name='edsr_r32f256', scale=2, precision=torch.float32, checkpoint=None),
        # ModelConfig(name="fsrcnn_big",      scale=2, precision=torch.float32, checkpoint="checkpoints/fsrcnn_finetuned_0.0036803.pth"),
        # ModelConfig(name="fsrcnn_small",    scale=2, precision=torch.float32, checkpoint="checkpoints/fsrcnn_finetuned_0.0051799.pth",),
        # ModelConfig(name="fsrcnn_without",  scale=2, precision=torch.float32, checkpoint="checkpoints/fsrcnn_finetuned.pth"),
        # ModelConfig(name="ninasr_b0",       scale=2, precision=torch.float32, checkpoint="checkpoints/ninasr_finetuned.pth"),
        # ModelConfig(name="ninasr_julia",    scale=2, precision=torch.float32, checkpoint="checkpoints/best_model_20251026_205928.pth",),
        # ModelConfig(name="carn",            scale=2, precision=torch.float32, checkpoint="checkpoints/carn_finetuned.pth"),
        # ModelConfig(name="edsr_r16f64",     scale=2, precision=torch.float32, checkpoint="checkpoints/edsr_finetuned.pth"),
        
        # ModelConfig(name="fsrcnn_big",      scale=2, precision=torch.float16, checkpoint="checkpoints/fsrcnn_finetuned_0.0036803.pth"),
        # ModelConfig(name="fsrcnn_small",    scale=2, precision=torch.float16, checkpoint="checkpoints/fsrcnn_finetuned_0.0051799.pth",),
        # ModelConfig(name="fsrcnn_without",  scale=2, precision=torch.float16, checkpoint="checkpoints/fsrcnn_finetuned.pth"),
        # ModelConfig(name="ninasr_b0",       scale=2, precision=torch.float16, checkpoint="checkpoints/ninasr_finetuned.pth"),
        # ModelConfig(name="ninasr_julia",    scale=2, precision=torch.float16, checkpoint="checkpoints/best_model_20251026_205928.pth",),
        # ModelConfig(name="carn",            scale=2, precision=torch.float16, checkpoint="checkpoints/carn_finetuned.pth"),
        # ModelConfig(name="edsr_r16f64",     scale=2, precision=torch.float16, checkpoint="checkpoints/edsr_finetuned.pth"),
        
        # ModelConfig(name="fsrcnn_big",      scale=2, precision=torch.bfloat16, checkpoint="checkpoints/fsrcnn_finetuned_0.0036803.pth"),
        # ModelConfig(name="fsrcnn_small",    scale=2, precision=torch.bfloat16, checkpoint="checkpoints/fsrcnn_finetuned_0.0051799.pth",),
        # ModelConfig(name="fsrcnn_without",  scale=2, precision=torch.bfloat16, checkpoint="checkpoints/fsrcnn_finetuned.pth"),
        # ModelConfig(name="ninasr_b0",       scale=2, precision=torch.bfloat16, checkpoint="checkpoints/ninasr_finetuned.pth"),
        # ModelConfig(name="ninasr_julia",    scale=2, precision=torch.bfloat16, checkpoint="checkpoints/best_model_20251026_205928.pth",),
        # ModelConfig(name="carn",            scale=2, precision=torch.bfloat16, checkpoint="checkpoints/carn_finetuned.pth"),
        # ModelConfig(name="edsr_r16f64",     scale=2, precision=torch.bfloat16, checkpoint="checkpoints/edsr_finetuned.pth"),
        # ModelConfig(name='ninasr_b0', scale=2, precision=torch.float16, checkpoint='checkpoints/ninasr_finetuned.pth'),
        # ModelConfig(name='ninasr_b0', scale=2, precision=torch.bfloat16, checkpoint='checkpoints/ninasr_finetuned.pth'),
        ModelConfig(name="fsrcnn_big_vgg",      scale=2, precision=torch.float16, checkpoint="outputs/2026-01-18/16-14-54/checkpoints/best_model_20260118_165246.pth"), # VGG Perceptual
        ModelConfig(name="fsrcnn_big_sobel",      scale=2, precision=torch.float16, checkpoint="outputs/2026-01-18/17-12-56/checkpoints/best_model_20260118_175141.pth"), # Sobel
        ModelConfig(name="fsrcnn_big_fourier",      scale=2, precision=torch.float16, checkpoint="outputs/2026-01-18/17-54-23/checkpoints/best_model_20260118_190055.pth"), # Fourier
        ModelConfig(name="fsrcnn_big_mse",      scale=2, precision=torch.float16, checkpoint="outputs/2025-11-05/11-46-51/checkpoints/best_model_20251105_124504.pth"), # MSE
        ModelConfig(name="fsrcnn_big_mae",      scale=2, precision=torch.float16, checkpoint="outputs/2025-11-05/10-38-44/checkpoints/best_model_20251105_110022.pth"), # MAE
    ]
    models = [ModelFactory.create(cfg, device) for cfg in model_configs]
    print(f"Prepared {len(models)} models for benchmarking.")
    # for cfg, model in zip(model_configs, models):
    #     print(model.config)
    #     results = benchmark(model, dataloader, device)
    #     print(f"    MSE: {results['mse']:.4f} ± {results['mse_std']:.4f}")
    #     print(f"    PSNR: {results['psnr']:.4f} ± {results['psnr_std']:.4f}")
    #     print(f"    SSIM: {results['ssim']:.4f} ± {results['ssim_std']:.4f}")
    #     print(f"    Perceptual Loss: {results['perc']:.4f} ± {results['perc_std']:.4f}")

    # classic_interpolations = [
    #     ModelConfig(name="bilinear_torch", scale=2, precision=torch.float32, checkpoint=None),
    #     ModelConfig(name="bicubic_torch", scale=2, precision=torch.float32, checkpoint=None),
    #     ModelConfig(name="nearest_torch", scale=2, precision=torch.float32, checkpoint=None),
    # ]
    # classic_interpolations_models = [ModelFactory.create(cfg, device) for cfg in classic_interpolations]

    # without_finetuning = [
    #     ModelConfig(name='ninasr_b0', scale=2, precision=torch.float32, checkpoint=None),
    #     ModelConfig(name='rcan', scale=2, precision=torch.float32, checkpoint=None),
    #     ModelConfig(name='carn', scale=2, precision=torch.float32, checkpoint=None),
    #     ModelConfig(name='carn_m', scale=2, precision=torch.float32, checkpoint=None),
    #     ModelConfig(name='edsr_r16f64', scale=2, precision=torch.float32, checkpoint=None),
    #     ModelConfig(name='edsr_r32f256', scale=2, precision=torch.float32, checkpoint=None),
    #     ModelConfig(name="fsrcnn_big", scale=2, precision=torch.float32, checkpoint=None),
    # ]
    # without_finetuning_models = [ModelFactory.create(cfg, device) for cfg in without_finetuning]

    # with_finetuning = [
    #     ModelConfig(name="fsrcnn_small",scale=2,precision=torch.float32,checkpoint="checkpoints/fsrcnn_finetuned_0.0051799.pth",),
    #     ModelConfig(name="fsrcnn_without", scale=2, precision=torch.float32, checkpoint="checkpoints/fsrcnn_finetuned.pth"),
    #     ModelConfig(name="fsrcnn_big", scale=2, precision=torch.float32, checkpoint="checkpoints/fsrcnn_finetuned_0.0036803.pth"),
    #     ModelConfig(name="ninasr_b0", scale=2, precision=torch.float32, checkpoint="checkpoints/ninasr_finetuned.pth"),
    #     ModelConfig(name="ninasr_julia",scale=2,precision=torch.float32,checkpoint="checkpoints/best_model_20251026_205928.pth",),
    #     ModelConfig(name="carn", scale=2, precision=torch.float32, checkpoint="checkpoints/carn_finetuned.pth"),
    #     ModelConfig(name="edsr_r16f64", scale=2, precision=torch.float32, checkpoint="checkpoints/edsr_finetuned.pth"),
    # ]
    # with_finetuning_models = [ModelFactory.create(cfg, device) for cfg in with_finetuning]


    # print("\n=== Benchmark Summary ===")
    # print("\n-- Classic Interpolations --")
    # for cfg, model in zip(classic_interpolations, classic_interpolations_models):
    #     print(model.config)
    #     results = benchmark(model, dataloader, device)
    #     print(f"    MSE: {results['mse']:.4f} ± {results['mse_std']:.4f}")
    #     print(f"    PSNR: {results['psnr']:.4f} ± {results['psnr_std']:.4f}")
    #     print(f"    SSIM: {results['ssim']:.4f} ± {results['ssim_std']:.4f}")
    #     print(f"    Perceptual Loss: {results['perc']:.4f} ± {results['perc_std']:.4f}")

    # print("\n\n\n-- Models without Finetuning --")
    # for cfg, model in zip(without_finetuning, without_finetuning_models):
    #     print(model.config)
    #     results = benchmark(model, dataloader, device)
    #     print(f"    MSE: {results['mse']:.4f} ± {results['mse_std']:.4f}")
    #     print(f"    PSNR: {results['psnr']:.4f} ± {results['psnr_std']:.4f}")
    #     print(f"    SSIM: {results['ssim']:.4f} ± {results['ssim_std']:.4f}")
    #     print(f"    Perceptual Loss: {results['perc']:.4f} ± {results['perc_std']:.4f}")

    # print("\n\n\n-- Models with Finetuning --")
    # for cfg, model in zip(with_finetuning, with_finetuning_models):
    #     print(model.config)
    #     results = benchmark(model, dataloader, device)
    #     print(f"    MSE: {results['mse']:.4f} ± {results['mse_std']:.4f}")
    #     print(f"    PSNR: {results['psnr']:.4f} ± {results['psnr_std']:.4f}")
    #     print(f"    SSIM: {results['ssim']:.4f} ± {results['ssim_std']:.4f}")
    #     print(f"    Perceptual Loss: {results['perc']:.4f} ± {results['perc_std']:.4f}")

    PRECISION_MAP = {
    torch.float32: "FP32",
    torch.float16: "FP16",
    torch.bfloat16: "BF16",
    }

    pred_precisions = [PRECISION_MAP[cfg.precision] for cfg in model_configs]
    model_names = [cfg.name for cfg in model_configs]

    print("\n=== Generating Visual Comparison ===")
    patch_info = PatchInfo(patch_x=640, patch_y=500, offset_x=200, image_idx=111)
    # patch_info = PatchInfo(patch_x=330+1120, patch_y=550, offset_x=1120, image_idx=108)
    gt_full, pred_fulls, pred_metrics = visualizer.run_models(models, patch_info.image_idx)

    plotter = BenchmarkPlotter(model_names, pred_metrics, pred_precisions, patch_info)
    diff_maps = plotter.compute_global_diff_maps(gt_full, pred_fulls)

    for i, full_img in enumerate([gt_full] + pred_fulls):
        final_image = plotter.render(i, full_img, show_footer=True)
        if isinstance(final_image, tuple):
            plain_img, scaled_img = final_image
            cv2.imwrite("benchmark_comparison_gt_plain.png", plain_img)
            print("Saved ground truth plain comparison image: benchmark_comparison_gt_plain.png")
            cv2.imwrite("benchmark_comparison_gt_scaled.png", scaled_img)
            print("Saved ground truth scaled comparison image: benchmark_comparison_gt_scaled.png")
        else:
            cv2.imwrite(f"benchmark_comparison_{i}.png", final_image)
        if i == 0:
            print(f"Saved ground truth comparison image: benchmark_comparison_{i}.png")
        else:
            print(f"Saved model {model_names[i-1]} comparison image: benchmark_comparison_{i}.png")
            final_diff_image = plotter.render_diff(i, full_img, diff_maps[i-1], show_footer=True)
            cv2.imwrite(f"benchmark_diff_{i}.png", final_diff_image)
            print(f"Saved model {model_names[i-1]} diff image: benchmark_diff_{i}.png")
