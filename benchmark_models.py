import glob
import os
from time import monotonic_ns

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from torch.utils.data import DataLoader, Dataset
from torchsr.models import ninasr_b0

from src.metrics import im_ssim

os.makedirs("results", exist_ok=True)

# CARN_PATH = "checkpoints/carn_finetuned.pth"
FSRCNN_BIG_PRETRAIN_PATH = "checkpoints/fsrcnn_finetuned_0.0036803.pth"
FSRCNN_SMALL_PRETRAIN_PATH = "checkpoints/fsrcnn_finetuned_0.0051799.pth"
FSRCNN_PATH = "checkpoints/fsrcnn_finetuned.pth"
# NINASR_PATH = "checkpoints/ninasr_finetuned.pth"

IDS_FOR_COMPARISON = [100, 390]

TEST_HR_DIR_PATH = "/mnt/local/rs/test/hr"
TEST_LR_DIR_PATH = "/mnt/local/rs/test/lr"


# def bicubic_upsample(lr_tensor, scale=2):
#     lr_np = lr_tensor.squeeze().cpu().numpy()
#     lr_np = (lr_np * 255.0).clip(0, 255).astype(np.uint8)
#     lr_np = np.transpose(lr_np, (1, 2, 0))
#     h, w, _ = lr_np.shape
#     upsampled = cv2.resize(lr_np, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
#     upsampled = np.transpose(upsampled, (2, 0, 1))
#     upsampled = upsampled.astype(np.float32) / 255.0
#     return torch.from_numpy(upsampled).unsqueeze(0)


MODELS = {
    # "carn_finetuned": carn,
    # "fsrcnn_big_pretrain": FSRCNN_BIG_PRETRAIN,
    # "fsrcnn_small_pretrain": FSRCNN_SMALL_PRETRAIN,
    # "fsrcnn_finetuned": FSRCNN_WITHOUT_PRETRAIN
    # "ninasr_finetuned": ninasr_b0,
    "ninasr_b0": ninasr_b0,
    # "bicubic": None,  # placeholder
}


def to_tensor(img):
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(img, (2, 0, 1))).float()


class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir):
        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, "*.png")))[:50]
        self.lr_paths = sorted(glob.glob(os.path.join(lr_dir, "*.png")))[:50]
        assert len(self.hr_paths) == len(self.lr_paths), "Mismatch HR and LR dataset size"

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr_img = cv2.imread(self.hr_paths[idx])
        lr_img = cv2.imread(self.lr_paths[idx])
        return to_tensor(lr_img), to_tensor(hr_img)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = SRDataset(TEST_HR_DIR_PATH, TEST_LR_DIR_PATH)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

print(f"Found {len(dataset)} HR/LR pairs for benchmarking.")


for model_name, model_instance in MODELS.items():
    print(f"Processing model: {model_name}")
    inference_fn = model_instance(scale=4, pretrained=True).to(device).eval()
    # if model_name == "bicubic":

    #     def inference_fn(lr_tensor):
    #         return bicubic_upsample(lr_tensor, scale=2)
    # else:
    #     if model_name == "fsrcnn_big_pretrain":
    #         model = model_instance(scale=2).to(device)
    #         state_dict = torch.load(FSRCNN_BIG_PRETRAIN_PATH, map_location=device)
    #         model.load_state_dict(state_dict)
    #     elif model_name == "fsrcnn_small_pretrain":
    #         model = model_instance(scale=2).to(device)
    #         state_dict = torch.load(FSRCNN_SMALL_PRETRAIN_PATH, map_location=device)
    #         model.load_state_dict(state_dict)
    #     elif model_name == "fsrcnn_finetuned":
    #         model = model_instance(scale=2).to(device)
    #         state_dict = torch.load(FSRCNN_PATH, map_location=device)
    #         model.load_state_dict(state_dict)

    #     model.eval()
    #     inference_fn = lambda lr: model(lr)

    times, psnr_list, ssim_list, mse_list = [], [], [], []

    for i, (lr_tensor, hr_tensor) in enumerate(dataloader):
        print(f"Processing {i}")
        lr_tensor, hr_tensor = lr_tensor.to(device), hr_tensor.to(device)

        with torch.no_grad():
            torch.cuda.synchronize()
            start_ns = monotonic_ns()
            output = inference_fn(lr_tensor)
            torch.cuda.synchronize()
            end_ns = monotonic_ns()
            times.append((end_ns - start_ns) / 1e6)

        output = output.squeeze().cpu().numpy()
        output = (output * 255.0).clip(0, 255).astype(np.uint8)
        hr_np = hr_tensor.squeeze().cpu().numpy()
        hr_np = (hr_np * 255.0).clip(0, 255).astype(np.uint8)

        psnr = peak_signal_noise_ratio(hr_np, output)
        psnr_list.append(psnr)

        ssim = im_ssim(hr_np, output)
        ssim_list.append(ssim)

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
