import os
from dataclasses import dataclass
from typing import Callable, List

import cv2
import numpy as np
import torch

from src.logger import logger
from src.metrics import im_mse, im_psnr, im_ssim


@dataclass
class Metrics:
    mse: float = 0.0
    psnr: float = 0.0
    ssim: float = 0.0

    def calc(self, gt, pred):
        self.mse = im_mse(gt, pred)
        self.psnr = im_psnr(gt, pred)
        self.ssim = im_ssim(gt, pred)
        return self


class SRVisualizer:
    def __init__(
        self,
        config,
        crop_coords: tuple = (1450, 550),
        scale_factor: int = 4,
    ):
        self.config = config
        self.hr_dir = config.data.test.y
        self.lr_dir = config.data.test.x
        self.crop_coords = crop_coords
        self.scale_factor = scale_factor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.config.plotting.dir, exist_ok=True)

    @staticmethod
    def to_tensor(img: np.ndarray) -> torch.Tensor:
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(np.transpose(img, (2, 0, 1))).float()

    @staticmethod
    def prepare_img_output(output: torch.Tensor) -> np.ndarray:
        if output.dtype in [torch.float16, torch.bfloat16]:
            output = output.float()
        output = output.squeeze().cpu().numpy()
        output = (output * 255.0).clip(0, 255).astype(np.uint8)
        output = np.transpose(output, (1, 2, 0))
        return output

    @staticmethod
    def add_text_below(img: np.ndarray, text: str, bold_lines=None, line_colors=None) -> np.ndarray:
        if img is None or img.size == 0:
            raise ValueError("Received empty image patch")

        if bold_lines is None:
            bold_lines = []

        if line_colors is None:
            line_colors = [(255, 255, 255)] * len(text.split("\n"))

        h, w = img.shape[:2]
        line_height = 25
        text_lines = text.split("\n")
        new_img = np.zeros((h + line_height * len(text_lines) + 5, w, 3), dtype=np.uint8)
        new_img[:h, :w] = img
        y0 = h + line_height
        for i, line in enumerate(text_lines):
            color = line_colors[i] if i < len(line_colors) else (255, 255, 255)
            if i in bold_lines:
                offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
                for ox, oy in offsets:
                    cv2.putText(
                        new_img,
                        line,
                        (5 + ox, y0 + i * line_height + oy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
            else:
                cv2.putText(
                    new_img, line, (5, y0 + i * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1, cv2.LINE_AA
                )
        return new_img

    @staticmethod
    def create_grid(patches: List[np.ndarray], ncols=5, spacing=20) -> np.ndarray:
        n = len(patches)
        rows = (n + ncols - 1) // ncols
        while len(patches) < rows * ncols:
            patches.append(np.zeros_like(patches[0]))

        row_imgs = []
        for r in range(rows):
            row = patches[r * ncols : (r + 1) * ncols]
            row_with_spaces = []
            for i, img in enumerate(row):
                row_with_spaces.append(img)
                if i < len(row) - 1:
                    row_with_spaces.append(np.ones((img.shape[0], spacing, 3), dtype=np.uint8) * 255)
            row_imgs.append(np.hstack(row_with_spaces))

        spaced_rows = []
        for i, row in enumerate(row_imgs):
            spaced_rows.append(row)
            if i < len(row_imgs) - 1:
                spaced_rows.append(np.ones((spacing, row.shape[1], 3), dtype=np.uint8) * 255)

        return np.vstack(spaced_rows)

    def load_images(self, idx=108):
        hr_path = os.path.join(self.hr_dir, f"{idx}.png")
        lr_path = os.path.join(self.lr_dir, f"{idx}.png")
        hr_image = cv2.imread(hr_path)
        lr_image = cv2.imread(lr_path)
        return hr_image, lr_image

    def prepare_diff_patches_with_hr(self, hr_patch, lr_patches, metrics):
        """
        Create a list of patches for the diff plot:
        - First patch: HR image (normal)
        - Then diff maps for each model vs HR
        """
        # Compute global max for normalization of diffs
        all_diffs = [np.abs(hr_patch.astype(np.float32) - p.astype(np.float32)) for p in lr_patches.values()]
        global_max_diff = max(d.max() for d in all_diffs)

        diff_patches = []

        for i, (name, patch) in enumerate(lr_patches.items()):
            if name == "HR (Ground Truth)":
                # Keep HR normal
                patch_resized = cv2.resize(
                    patch,
                    (patch.shape[1] * self.scale_factor, patch.shape[0] * self.scale_factor),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                # Diff map
                diff = np.abs(hr_patch.astype(np.float32) - patch.astype(np.float32))
                diff_norm = (diff / global_max_diff * 255.0).clip(0, 255).astype(np.uint8)
                patch_resized = cv2.applyColorMap(diff_norm.max(axis=2), cv2.COLORMAP_JET)
                patch_resized = cv2.resize(
                    patch_resized,
                    (patch_resized.shape[1] * self.scale_factor, patch_resized.shape[0] * self.scale_factor),
                    interpolation=cv2.INTER_NEAREST,
                )

            # Add metrics text below each patch
            mse, psnr, ssim = metrics[name]
            metric_text = (
                f"{name}\nMSE: {mse:.2f} | PSNR: {psnr:.2f} | SSIM: {ssim:.2f}"
                if mse != "-"
                else f"{name}\nMSE: {mse} | PSNR: {psnr} | SSIM: {ssim}"
            )
            line_colors = [(0, 0, 255)] if name == "HR (Ground Truth)" else None
            annotated_patch = self.add_text_below(patch_resized, metric_text, bold_lines=[0], line_colors=line_colors)
            diff_patches.append(annotated_patch)

        return diff_patches

    def run_models(self, models: List[Callable[[torch.Tensor], torch.Tensor]]):
        hr_image, lr_image = self.load_images()
        hr_tensor = self.to_tensor(hr_image)
        hr_np = (hr_tensor.numpy() * 255.0).clip(0, 255).astype(np.uint8)
        hr_np = np.transpose(hr_np, (1, 2, 0))
        gt_patch = hr_np[
            self.crop_coords[1] : self.crop_coords[1] + 128,
            self.crop_coords[0] : self.crop_coords[0] + 128,
        ]
        img_lr = self.to_tensor(lr_image).unsqueeze(0).to(self.device)

        pred_patches = []
        pred_metrics = []

        for model in models:
            with torch.no_grad():
                output = model(img_lr)
            output = self.prepare_img_output(output)
            pred_metrics.append(Metrics().calc(hr_image, output))
            x, y = self.crop_coords
            pred_patches.append(output[y : y + 128, x : x + 128])

        return gt_patch, pred_patches, pred_metrics

    def plot_results(self, gt_patch, pred_patches, pred_metrics, pred_names: List[str], out_prefix="output"):
        lr_patches = {"HR (Ground Truth)": gt_patch}
        metrics = {"HR (Ground Truth)": ("-", "-", "-")}
        for patch, metric, name in zip(pred_patches, pred_metrics, pred_names):
            lr_patches[name] = patch
            metrics[name] = (metric.mse, metric.psnr, metric.ssim)

        annotated_patches = [
            self.add_text_below(
                cv2.resize(patch, (patch.shape[1] * 4, patch.shape[0] * 4), interpolation=cv2.INTER_NEAREST),
                f"{name}\nMSE: {metrics[name][0]:.2f} | PSNR: {metrics[name][1]:.2f} | SSIM: {metrics[name][2]:.2f}"
                if metrics[name][0] != "-"
                else f"{name}\nMSE: - | PSNR: - | SSIM: -",
                bold_lines=[0],
                line_colors=[(0, 0, 255)] if name == "HR (Ground Truth)" else None,
            )
            for name, patch in lr_patches.items()
        ]

        final_img = self.create_grid(annotated_patches, ncols=len(annotated_patches))
        out_path = f"{self.config.plotting.dir}/{out_prefix}.png"
        cv2.imwrite(out_path, final_img)
        logger.info(f"Saved comparison image to {out_path}")

        # diff map version
        diff_patches = self.prepare_diff_patches_with_hr(gt_patch, lr_patches, metrics)
        final_img_diff = self.create_grid(diff_patches, ncols=len(diff_patches))
        out_path_diff = f"{self.config.plotting.dir}/{out_prefix}_diffs.png"
        cv2.imwrite(out_path_diff, final_img_diff)
        logger.info(f"Saved diff comparison image to {out_path_diff}")

        return final_img, final_img_diff
