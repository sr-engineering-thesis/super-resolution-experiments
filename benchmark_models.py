import glob
import os
from time import monotonic_ns

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from torchsr.models import carn, carn_m, edsr_baseline, edsr_r16f64, edsr_r32f256, ninasr_b0, rcan

from src.metrics import im_ssim

os.makedirs("results", exist_ok=True)

MODEL_PATH = "checkpoints/best_model_20250831_125529.pth"

IDS_FOR_COMPARISON = [100, 390]

TEST_HR_DIR_PATH = "data/test/hr"
TEST_LR_DIR_PATH = "data/test/lr"

MODELS = {
    "carn": carn,
    # "carn_finetuned": carn,
    "ninasr_b0": ninasr_b0,
    "edsr_baseline": edsr_baseline,
    "edsr_r16f64": edsr_r16f64,
    "edsr_r32f256": edsr_r32f256,
    "carn_m": carn_m,
    "rcan": rcan,
}


def to_tensor(img):
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(img, (2, 0, 1))).float()


hr_images = glob.glob(os.path.join(TEST_HR_DIR_PATH, "*.png"))
lr_images = glob.glob(os.path.join(TEST_LR_DIR_PATH, "*.png"))
print(f"Found {len(hr_images)} HR images and {len(lr_images)} LR images for benchmarking.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hr_images_loaded = [cv2.imread(x) for x in hr_images[:5]]
lr_images_loaded = [cv2.imread(x) for x in lr_images[:5]]
print("HR and LR loaded as images into RAM")

for model_name, model_instance in MODELS.items():
    print(f"Processing model: {model_name}")
    model = model_instance(scale=2, pretrained=True).to(device)
    if model_name != "carn_finetuned":
        model = model_instance(scale=2, pretrained=True).to(device)
    else:  # finetuned model
        model = carn(scale=2, pretrained=False).to(device)
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)

    model.eval()

    times = []
    psnr_list = []
    ssim_list = []
    mse_list = []

    for hr_image, lr_image in list(zip(hr_images_loaded, lr_images_loaded)):
        img_lr = to_tensor(lr_image).unsqueeze(0).to(device)
        img_hr = to_tensor(hr_image).unsqueeze(0).to(device)

        with torch.no_grad():
            torch.cuda.synchronize()
            start_ns = monotonic_ns()
            output = model(img_lr)
            torch.cuda.synchronize()
            end_ns = monotonic_ns()
            times.append((end_ns - start_ns) / 1e6)

        output = output.squeeze().cpu().numpy()
        output = (output * 255.0).clip(0, 255).astype(np.uint8)
        hr_np = img_hr.squeeze().cpu().numpy()
        hr_np = (hr_np * 255.0).clip(0, 255).astype(np.uint8)

        psnr = peak_signal_noise_ratio(hr_np, output)
        psnr_list.append(psnr)

        # calculate SSIM
        ssim = im_ssim(hr_np, output)
        ssim_list.append(ssim)

        # calculate mse
        mse = np.mean((hr_np - output) ** 2)
        mse_list.append(mse)

    times_np = np.array(times)[1:]
    psnr_np = np.array(psnr_list)
    ssim_np = np.array(ssim_list)
    mse_np = np.array(mse_list)

    avg_time = np.mean(times_np)
    std_time = np.std(times_np)

    avg_fps = 1000 / avg_time if avg_time > 0 else 0
    std_fps = (1000 / (avg_time - std_time) - 1000 / (avg_time + std_time)) / 2 if avg_time > std_time else 0

    avg_psnr = np.mean(psnr_np)
    std_psnr = np.std(psnr_np)

    avg_ssim = np.mean(ssim_np)
    std_ssim = np.std(ssim_np)

    avg_mse = np.mean(mse_np)
    std_mse = np.std(mse_np)

    print(f"\n=== Summary for model: {model_name} ===")
    print(f"Average Inference Time: {avg_time:.4f} ± {std_time:.4f} ms")
    print(f"Mean PSNR: {avg_psnr:.4f} ± {std_psnr:.4f} dB")
    print(f"Mean SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
    print(f"Mean MSE: {avg_mse:.4f} ± {std_mse:.4f}")
    print("========================================\n")
