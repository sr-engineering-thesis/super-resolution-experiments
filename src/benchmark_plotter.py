from dataclasses import dataclass

import cv2
import numpy as np
from matplotlib import pyplot as plt

MODEL_NAMES_MAP = {
    "ninasr_b0": "NinaSR B0",
    "rcan": "RCAN",
    "carn": "CARN",
    "carn_m": "CARN_M",
    "edsr_r16f64": "EDSR_r16f64",
    "edsr_r32f256": "EDSR_r32f256",
    "nearest_torch": "Nearest Neighbor",
    "bilinear_torch": "Bilinear",
    "bicubic_torch": "Bicubic",
    "lanczos": "Lanczos",
    "fsrcnn_big": "FSRCNN-B",
    "fsrcnn_small": "FSRCNN-S",
    "fsrcnn_without": "FSRCNN",
    "ninasr_julia": r"NinaSR_SM",
    "fsrcnn_big_vgg": "Perceptual Loss",
    "fsrcnn_big_fourier": "Fourier Loss",
    "fsrcnn_big_sobel": "Sobel Loss",
    "fsrcnn_big_mse": "MSE",
    "fsrcnn_big_mae": "MAE",
}


@dataclass
class PatchInfo:
    patch_x: int
    patch_y: int
    offset_x: int
    image_idx: int


class BenchmarkPlotter:
    def __init__(self, model_names, pred_metrics, pred_precisions, patch_info: PatchInfo):
        self.model_names = model_names
        self.pred_metrics = pred_metrics
        self.pred_precisions = pred_precisions
        self.header_height = 120
        self.patch_crop_w = 1440
        self.patch_crop_h = 1440

        self.patch_x = patch_info.patch_x
        self.patch_y = patch_info.patch_y
        self.offset_x = patch_info.offset_x
        self.patch_size = 128

        self.header_height = 160
        self.text_px = self.header_height
        self.text_px_small = self.header_height // 2 - 9
        self.scale = 11.25
        self.scale_gt = 5
        self.border = 4
        self.color = (73, 73, 223)

    def latex_rgba(self, text, height=None):
        if height is None:
            height = self.text_px

        fig = plt.figure(figsize=(8, 1), dpi=200)
        plt.text(0.5, 0.5, text, fontsize=50, ha="center", va="center", color="black")
        plt.axis("off")
        fig.patch.set_alpha(0.0)
        fig.canvas.draw()

        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        buf = buf.reshape((h, w, 4))
        img = buf[:, :, [1, 2, 3, 0]]
        plt.close(fig)

        ih, iw, _ = img.shape
        scale = height / ih
        img = cv2.resize(img, (int(iw * scale), height), interpolation=cv2.INTER_LINEAR)
        return img

    def draw_footer(self, final_image, idx):
        line_height = self.text_px
        footer_height = line_height * 2
        footer = np.zeros((footer_height, final_image.shape[1], 4), dtype=np.uint8)

        name_tex = self.latex_rgba(self.model_name_tex(idx), line_height)
        x0 = (footer.shape[1] - name_tex.shape[1]) // 2
        footer[0:line_height, x0 : x0 + name_tex.shape[1]] = name_tex

        metric_tex = self.latex_rgba(self.metric_lines(idx)[0], line_height)
        x1 = (footer.shape[1] - metric_tex.shape[1]) // 2
        footer[line_height : line_height * 2, x1 : x1 + metric_tex.shape[1]] = metric_tex

        return footer

    def overlay_enlarged_patch(self, patch):
        h, w, _ = patch.shape
        cropped = patch[0 : self.patch_crop_h, 0 : self.patch_crop_w, :]

        orig_patch = patch[self.patch_y : self.patch_y + self.patch_size, self.patch_x : self.patch_x + self.patch_size]
        orig_patch = np.ascontiguousarray(orig_patch)

        enlarged = cv2.resize(
            orig_patch,
            (int(self.patch_size * self.scale), int(self.patch_size * self.scale)),
            interpolation=cv2.INTER_NEAREST,
        )

        eh, ew, _ = enlarged.shape
        ph, pw, _ = cropped.shape

        ix = max(0, pw - ew - self.border // 2)
        iy = max(0, ph - eh - self.border // 2)

        cropped[iy : iy + eh, ix : ix + ew] = enlarged

        return cropped

    def render_gt_plain(self, full_img):
        cropped = full_img[0 : self.patch_crop_h, self.offset_x : self.patch_crop_w + self.offset_x, :]
        cropped = np.ascontiguousarray(cropped)

        rx = self.patch_x - self.offset_x
        ry = self.patch_y
        cv2.rectangle(cropped, (rx, ry), (rx + self.patch_size, ry + self.patch_size), self.color, self.border)

        orig_patch = full_img[
            self.patch_y : self.patch_y + self.patch_size, self.patch_x : self.patch_x + self.patch_size
        ]

        enlarged = cv2.resize(
            orig_patch,
            (int(self.patch_size * self.scale_gt), int(self.patch_size * self.scale_gt)),
            interpolation=cv2.INTER_NEAREST,
        )

        eh, ew, _ = enlarged.shape
        ph, pw, _ = cropped.shape

        ix = max(0, pw - ew - self.border // 2)
        iy = max(0, ph - eh - self.border // 2)

        cropped[iy : iy + eh, ix : ix + ew] = enlarged
        cv2.rectangle(cropped, (ix, iy), (ix + ew, iy + eh), self.color, self.border)
        cv2.line(cropped, (rx + self.patch_size, ry + self.patch_size), (ix, iy), self.color, self.border)

        cropped_rgba = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
        cropped_rgba[:, :, 3] = 255

        name_tex = self.latex_rgba(self.model_name_tex(0), height=self.text_px_small)
        metric_tex = self.latex_rgba(r"$\mathbf{MSE\ / PSNR\ / SSIM}$", height=self.text_px_small)

        footer_height = self.text_px_small * 2
        footer = np.zeros((footer_height, cropped_rgba.shape[1], 4), dtype=np.uint8)

        x0 = (footer.shape[1] - name_tex.shape[1]) // 2
        footer[0 : self.text_px_small, x0 : x0 + name_tex.shape[1]] = name_tex

        x1 = (footer.shape[1] - metric_tex.shape[1]) // 2
        footer[self.text_px_small : self.text_px_small * 2, x1 : x1 + metric_tex.shape[1]] = metric_tex

        final = np.vstack([cropped_rgba, footer])
        return final

    def render_gt_scaled(self, full_img):
        full_img = np.ascontiguousarray(full_img.astype(np.uint8))
        cropped = self.overlay_enlarged_patch(full_img)
        cropped_rgba = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
        cropped_rgba[:, :, 3] = 255

        name_tex = self.latex_rgba(self.model_name_tex(0), self.text_px)
        metric_tex = self.latex_rgba(r"$\mathbf{-\ /\ -\ /\ -}$", self.text_px)

        footer = np.zeros((self.text_px * 2, cropped_rgba.shape[1], 4), dtype=np.uint8)
        x0 = (footer.shape[1] - name_tex.shape[1]) // 2
        footer[0 : self.text_px, x0 : x0 + name_tex.shape[1]] = name_tex
        x1 = (footer.shape[1] - metric_tex.shape[1]) // 2
        footer[self.text_px : self.text_px * 2, x1 : x1 + metric_tex.shape[1]] = metric_tex

        final = np.vstack([cropped_rgba, footer])
        return final

    def metric_lines(self, idx):
        if idx == 0:
            return [r"$\mathbf{(MSE\ /\ PSNR\ /\ SSIM)}$"]
        m = self.pred_metrics[idx - 1]
        return [rf"$\mathbf{{{m.mse:.2f}\ /\ {m.psnr:.2f}\ /\ {m.ssim:.2f}}}$"]

    def model_name_tex(self, idx):
        if idx == 0:
            return r"$\mathbf{HR\ Ground\ Truth}$"
        name = MODEL_NAMES_MAP.get(self.model_names[idx - 1], self.model_names[idx - 1])
        name = name.replace("_", r"\_").replace(" ", r"\ ")
        return rf"$\mathbf{{{name}}}$"

    # ---===== DIFF SUPPORT =====---

    def compute_global_diff_maps(self, hr_full, pred_full_list):
        hr_gray = cv2.cvtColor(hr_full, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diffs = []
        for p in pred_full_list:
            g = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY).astype(np.float32)
            diffs.append(np.abs(hr_gray - g))
        global_max = max([d.max() for d in diffs]) if diffs else 1.0
        diff_color = []
        for d in diffs:
            dn = (d / global_max * 255).clip(0, 255).astype(np.uint8)
            cm = cv2.applyColorMap(dn, cv2.COLORMAP_JET)
            diff_color.append(cm)
        return diff_color

    def overlay_enlarged_patch_diff(self, full_img, diff_img):
        h, w, _ = full_img.shape
        cropped = full_img[0 : self.patch_crop_h, self.offset_x : self.patch_crop_w + self.offset_x, :].copy()

        orig_patch_diff = diff_img[
            self.patch_y : self.patch_y + self.patch_size, self.patch_x : self.patch_x + self.patch_size
        ]

        enlarged = cv2.resize(
            orig_patch_diff,
            (int(self.patch_size * self.scale), int(self.patch_size * self.scale)),
            interpolation=cv2.INTER_NEAREST,
        )

        eh, ew, _ = enlarged.shape
        ph, pw, _ = cropped.shape

        ix = max(0, pw - ew - self.border // 2)
        iy = max(0, ph - eh - self.border // 2)

        cropped[iy : iy + eh, ix : ix + ew] = enlarged

        # cv2.rectangle(cropped, (ix, iy), (ix+ew, iy+eh), self.color, self.border)

        # rx = self.patch_x
        # ry = self.patch_y

        # print(rx, ry)

        # cv2.rectangle(cropped, (rx, ry),
        #             (rx+self.patch_size, ry+self.patch_size),
        #             self.color, self.border)

        # cv2.line(cropped,
        #         (rx+self.patch_size, ry+self.patch_size),
        #         (ix, iy), self.color, self.border)

        return cropped

    def render_normal(self, idx, full_img, show_footer=True):
        full_img = np.ascontiguousarray(full_img.astype(np.uint8))
        cropped = self.overlay_enlarged_patch(full_img)
        cropped_rgba = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
        cropped_rgba[:, :, 3] = 255
        img = cropped_rgba
        if show_footer:
            footer = self.draw_footer(img, idx)
            img = np.vstack([img, footer])
        return img

    def render_diff(self, idx, full_img, full_diff, show_footer=True):
        full_img = np.ascontiguousarray(full_img.astype(np.uint8))
        cropped = self.overlay_enlarged_patch_diff(full_img, full_diff)
        cropped_rgba = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
        cropped_rgba[:, :, 3] = 255
        img = cropped_rgba
        if show_footer:
            footer = self.draw_footer(img, idx)
            img = np.vstack([img, footer])
        return img

    def render(self, idx, full_img, full_diff=None, show_footer=True):
        if idx == 0:
            print("Rendering Ground Truth plain...")
            plain = self.render_gt_plain(full_img)
            print("Rendering Ground Truth scaled...")
            scaled = self.render_gt_scaled(full_img)
            return plain, scaled
        else:
            if full_diff is None:
                return self.render_normal(idx, full_img, show_footer)
            return self.render_diff(idx, full_img, full_diff, show_footer)
