import os

import cv2
import numpy as np
import torch
from torchsr.models import carn, carn_m, edsr_r16f64, edsr_r32f256, ninasr_b0, rcan

from src.metrics import im_mse, im_psnr, im_ssim

CROP_COORDS = (1450, 550)

TEST_HR_DIR_PATH = "data/test/hr"
TEST_LR_DIR_PATH = "data/test/lr"

MODELS = {
    "carn": carn,
    "carn_finetuned": carn,
    "ninasr_b0": ninasr_b0,
    "edsr_r16f64": edsr_r16f64,
    "edsr_r32f256": edsr_r32f256,
    "carn_m": carn_m,
    "rcan": rcan,
}


def to_tensor(img):
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(img, (2, 0, 1))).float()


def get_model(name, device):
    if name.endswith("finetuned"):
        model = MODELS[name](scale=2, pretrained=False).to(device)
        state_dict = torch.load(f"checkpoints/{name}.pth", map_location=device)
        model.load_state_dict(state_dict)
    else:
        model = MODELS[name](scale=2, pretrained=True).to(device)
    return model


def add_text_below(
    img,
    text,
    bold_lines=None,
    line_colors=None,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    scale=0.75,
    default_color=(255, 255, 255),
    thickness=1,
):
    if img is None or img.size == 0:
        raise ValueError("Received empty image patch")

    if bold_lines is None:
        bold_lines = []

    if line_colors is None:
        line_colors = [default_color] * len(text.split("\n"))

    h, w = img.shape[:2]
    line_height = 25
    text_lines = text.split("\n")
    new_img = np.zeros((h + line_height * len(text_lines) + 5, w, 3), dtype=np.uint8)
    new_img[:h, :w] = img
    y0 = h + line_height
    for i, line in enumerate(text_lines):
        color = line_colors[i] if i < len(line_colors) else default_color
        if i in bold_lines:
            offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
            for ox, oy in offsets:
                cv2.putText(
                    new_img, line, (5 + ox, y0 + i * line_height + oy), font, scale, color, thickness, cv2.LINE_AA
                )
        else:
            cv2.putText(new_img, line, (5, y0 + i * line_height), font, scale, color, thickness, cv2.LINE_AA)
    return new_img


def prepare_diff_patches_with_hr(hr_patch, lr_patches, metrics, scale_factor=4):
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
                patch, (patch.shape[1] * scale_factor, patch.shape[0] * scale_factor), interpolation=cv2.INTER_NEAREST
            )
        else:
            # Diff map
            diff = np.abs(hr_patch.astype(np.float32) - patch.astype(np.float32))
            diff_norm = (diff / global_max_diff * 255.0).clip(0, 255).astype(np.uint8)
            patch_resized = cv2.applyColorMap(diff_norm.max(axis=2), cv2.COLORMAP_JET)
            patch_resized = cv2.resize(
                patch_resized,
                (patch_resized.shape[1] * scale_factor, patch_resized.shape[0] * scale_factor),
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
        annotated_patch = add_text_below(patch_resized, metric_text, bold_lines=[0], line_colors=line_colors)
        diff_patches.append(annotated_patch)

    return diff_patches


def create_grid(patches, cols=4, spacing=20):
    n = len(patches)
    rows = (n + cols - 1) // cols

    while len(patches) < rows * cols:
        patches.append(np.zeros_like(patches[0]))

    row_imgs = []
    for r in range(rows):
        row = patches[r * cols : (r + 1) * cols]
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

    final_img = np.vstack(spaced_rows)
    return final_img


def run_inference_and_plot_separate():
    hr_path = os.path.join(TEST_HR_DIR_PATH, "108.png")
    lr_path = os.path.join(TEST_LR_DIR_PATH, "108.png")
    hr_image = cv2.imread(hr_path)
    lr_image = cv2.imread(lr_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr_patches = {}
    metrics = {}

    hr_tensor = to_tensor(hr_image)
    hr_np = (hr_tensor.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    hr_np = np.transpose(hr_np, (1, 2, 0))
    gt_patch = hr_np[CROP_COORDS[1] : CROP_COORDS[1] + 128, CROP_COORDS[0] : CROP_COORDS[0] + 128]

    lr_patches["HR (Ground Truth)"] = gt_patch
    metrics["HR (Ground Truth)"] = ("-", "-", "-")

    for model_name in MODELS.keys():
        print(f"Running inference for: {model_name}")
        model = get_model(model_name, device)
        model.eval()

        img_lr = to_tensor(lr_image).unsqueeze(0).to(device)
        img_hr = to_tensor(hr_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_lr)

        output = output.squeeze().cpu().numpy()
        output = (output * 255.0).clip(0, 255).astype(np.uint8)
        output = np.transpose(output, (1, 2, 0))

        hr_np_model = img_hr.squeeze().cpu().numpy()
        hr_np_model = (hr_np_model * 255.0).clip(0, 255).astype(np.uint8)
        hr_np_model = np.transpose(hr_np_model, (1, 2, 0))

        mse = im_mse(hr_np_model, output)
        psnr = im_psnr(hr_np_model, output)
        ssim = im_ssim(hr_np_model, output)
        metrics[model_name] = (mse, psnr, ssim)

        x, y = CROP_COORDS
        patch = output[y : y + 128, x : x + 128]
        lr_patches[model_name] = patch

    # Plot normal images with metrics
    annotated_patches = []
    for name, patch in lr_patches.items():
        patch_resized = cv2.resize(patch, (patch.shape[1] * 4, patch.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
        mse, psnr, ssim = metrics[name]
        metric_text = (
            f"{name}\nMSE: {mse:.2f} | PSNR: {psnr:.2f} | SSIM: {ssim:.2f}"
            if mse != "-"
            else f"{name}\nMSE: {mse} | PSNR: {psnr} | SSIM: {ssim}"
        )
        line_colors = [(0, 0, 255)] if name == "HR (Ground Truth)" else None
        annotated_patch = add_text_below(patch_resized, metric_text, bold_lines=[0], line_colors=line_colors)
        annotated_patches.append(annotated_patch)

    final_img = create_grid(annotated_patches)
    cv2.imwrite("comparison_images.png", final_img)
    print("Saved comparison_images.png")

    # Plot diff comparison (first HR, then diff maps)
    diff_patches = prepare_diff_patches_with_hr(gt_patch, lr_patches, metrics)
    final_diff_img = create_grid(diff_patches)
    cv2.imwrite("comparison_diffs.png", final_diff_img)
    print("Saved comparison_diffs.png")


if __name__ == "__main__":
    run_inference_and_plot_separate()
