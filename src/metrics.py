from dataclasses import dataclass

import numpy as np
import torch
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


@dataclass
class Metrics:
    mse: float
    psnr: float
    ssim: float


def to_numpy_uint8(t: torch.Tensor) -> np.ndarray:
    arr = t.squeeze().detach().cpu().numpy()
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def im_mse(im1: np.ndarray, im2: np.ndarray) -> float:
    assert im1.shape == im2.shape, "Images must be the same shape"
    return mean_squared_error(im1, im2)


def im_psnr(im1: np.ndarray, im2: np.ndarray) -> float:
    assert im1.shape == im2.shape, "Images must be the same shape"
    return peak_signal_noise_ratio(im1, im2, data_range=255)


def im_ssim(im1: np.ndarray, im2: np.ndarray) -> float:
    """
    Compute SSIM for a single image (C,H,W) or grayscale (1,H,W).
    Assumes values are in [0,1] or already scaled to [0,255].
    """

    # If color image (3,H,W), transpose to HWC
    if im1.ndim == 3 and im1.shape[0] == 3:
        im1 = np.transpose(im1, (1, 2, 0))
        im2 = np.transpose(im2, (1, 2, 0))
        return structural_similarity(im1, im2, channel_axis=-1, data_range=255)

    # If grayscale (1,H,W), squeeze to H,W
    if im1.ndim == 3 and im1.shape[0] == 1:
        im1 = im1.squeeze()
        im2 = im2.squeeze()
        return structural_similarity(im1, im2, data_range=255)

    # Already 2D
    if im1.ndim == 2:
        return structural_similarity(im1, im2, data_range=255)

    raise ValueError(f"Unsupported shape for SSIM: {im1.shape}")


def batch_metrics(sr_batch: torch.Tensor, hr_batch: torch.Tensor) -> Metrics:
    """
    Compute average MSE, PSNR, SSIM across a batch of images.

    Args:
        sr_batch (torch.Tensor): Super-resolved images (B,C,H,W)
        hr_batch (torch.Tensor): Ground-truth images (B,C,H,W)
    """
    mse_scores, psnr_scores, ssim_scores = [], [], []

    for sr_img, hr_img in zip(sr_batch, hr_batch):
        sr_img = to_numpy_uint8(sr_img)
        hr_img = to_numpy_uint8(hr_img)
        mse_scores.append(im_mse(sr_img, hr_img))
        psnr_scores.append(im_psnr(sr_img, hr_img))
        ssim_scores.append(im_ssim(sr_img, hr_img))

    return Metrics(mse=float(np.mean(mse_scores)), psnr=float(np.mean(psnr_scores)), ssim=float(np.mean(ssim_scores)))
