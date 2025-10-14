import glob
import os
from time import monotonic_ns

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchsr.models import carn

from models.fsrcnn.fsrcnn import FSRCNN_BIG_PRETRAIN, FSRCNN_SMALL_PRETRAIN, FSRCNN_WITHOUT_PRETRAIN
from src.metrics import im_mse, im_psnr, im_ssim
from src.visualizer import SRVisualizer

os.makedirs("results", exist_ok=True)

MODEL_REGISTRY = {
    "fsrcnn_finetuned": {
        "class": FSRCNN_WITHOUT_PRETRAIN,
        "ckpt": "checkpoints/fsrcnn_finetuned.pth",
    },
    "fsrcnn_small_pretrained": {
        "class": FSRCNN_SMALL_PRETRAIN,
        "ckpt": "checkpoints/fsrcnn_finetuned_0.0051799.pth",
    },
    "fsrcnn_big_pretrained": {
        "class": FSRCNN_BIG_PRETRAIN,
        "ckpt": "checkpoints/fsrcnn_finetuned_0.0036803.pth",
    },
    "carn": {
        "class": carn,
        "ckpt": "checkpoints/carn_finetuned.pth",
    },
}

def load_model(model_name, device, scale=2):
    info = MODEL_REGISTRY[model_name]
    model_cls = info["class"]
    ckpt_path = info["ckpt"]

    if model_cls is carn:
        model = model_cls(scale=scale, pretrained=True).to(device)
    else:
        model = model_cls(scale=scale).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

MODELS = {name: (lambda n=name: (lambda device: load_model(n, device)))() for name in MODEL_REGISTRY}


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

def model_infer(lr_tensor, model, precision):
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=precision):
            return model(lr_tensor)

def benchmark(model, dataloader, device, precision=torch.float32):
    times = []
    total_mse, total_psnr, total_ssim = 0.0, 0.0, 0.0
    total_images = 0

    with torch.no_grad():
        for i, (lr_tensor, hr_tensor) in enumerate(dataloader):
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)

            torch.cuda.synchronize()
            start_ns = monotonic_ns()

            with torch.autocast(device_type="cuda", dtype=precision):
                output = model(lr_tensor)

            torch.cuda.synchronize()
            end_ns = monotonic_ns()
            times.append((end_ns - start_ns) / 1e6)

            output_np = output.squeeze().cpu().float().numpy()
            output_np = (np.transpose(output_np, (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)

            hr_np = hr_tensor.squeeze().cpu().numpy()
            hr_np = (np.transpose(hr_np, (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)

            mse = im_mse(hr_np, output_np)
            psnr = im_psnr(hr_np, output_np)
            ssim = im_ssim(hr_np, output_np)

            total_mse += mse
            total_psnr += psnr
            total_ssim += ssim
            total_images += 1

    times_np = np.array(times)[5:]
    avg_time, std_time = np.mean(times_np), np.std(times_np)

    avg_metrics = {
        "mse": total_mse / total_images,
        "psnr": total_psnr / total_images,
        "ssim": total_ssim / total_images,
    }
    return avg_time, std_time, avg_metrics


if __name__ == "__main__":
    print("Starting benchmark...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precisions = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    config = OmegaConf.load("configs/config.yaml")
    visualizer = SRVisualizer(config)

    dataset = SRDataset(config.data.test.y, config.data.test.x) # y -> HR, x -> LR
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    for model_name, loader in MODELS.items():
        print(f"\n=== Processing model: {model_name} ===")
        base_model = loader(device)
        base_model.eval()

        models_for_plot = []
        model_labels = []

        for precision_label, precision in precisions.items():
            model = base_model
            model.eval()

            models_for_plot.append(lambda lr_tensor, m=model, p=precision: model_infer(lr_tensor, m, p))
            model_labels.append(f"{model_name}_{precision_label}")

            avg_time, std_time, avg_metrics = benchmark(model, dataloader, device, precision=precision)
            print(f"{precision_label.upper()} inference: {avg_time:.4f} ± {std_time:.4f} ms")
            print(
                f"Avg Metrics: MSE: {avg_metrics['mse']}, PSNR: {avg_metrics['psnr']}, SSIM: {avg_metrics['ssim']}"
            )

        gt_patch, pred_patches, pred_metrics = visualizer.run_models(models_for_plot)
        visualizer.plot_results(
            gt_patch, pred_patches, pred_metrics, model_labels, out_prefix=f"{model_name}_comparison"
        )
