import numpy as np
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)


def im_mse(im1: np.ndarray, im2: np.ndarray) -> float:
    assert im1.shape == im2.shape, "Images must be the same shape"
    return mean_squared_error(im1, im2)


def im_psnr(im1: np.ndarray, im2: np.ndarray) -> float:
    assert im1.shape == im2.shape, "Images must be the same shape"
    return peak_signal_noise_ratio(im1, im2, data_range=1.0)


def im_ssim(im1: np.ndarray, im2: np.ndarray) -> float:
    assert im1.shape == im2.shape, "Images must be the same shape"

    # Case 1: Single image (2D)
    if im1.ndim == 2:
        return structural_similarity(im1, im2, data_range=1.0)

    # Case 2: Single image with channel (1, H, W)
    if im1.ndim == 3 and im1.shape[0] == 1:
        return structural_similarity(im1.squeeze(), im2.squeeze(), data_range=1.0)

    # Case 3: Batch (B, 1, H, W)
    if im1.ndim == 4 and im1.shape[1] == 1:
        scores = []
        for s, h in zip(im1, im2):  # s, h: (1, H, W)
            s = s.squeeze()
            h = h.squeeze()
            scores.append(structural_similarity(s, h, data_range=1.0))
        return float(np.mean(scores))

    raise ValueError(f"Unsupported shape for SSIM: {im1.shape}")
