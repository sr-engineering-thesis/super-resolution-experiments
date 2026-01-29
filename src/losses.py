import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def rgb_to_luminance(x):
    b, g, r = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def sobel_filter(x):
    x = rgb_to_luminance(x)

    sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=x.dtype, device=x.device).unsqueeze(0)

    sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=x.dtype, device=x.device).unsqueeze(0)

    grad_x = F.conv2d(x, sobel_x, padding=1)
    grad_y = F.conv2d(x, sobel_y, padding=1)
    return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)


class SobelLoss(nn.Module):
    def __init__(self, warmup_batches: int = 10, eps: float = 1e-6, use_scaling: bool = True):
        super().__init__()
        self.pixel = nn.L1Loss()

        self.warmup_batches = warmup_batches
        self.eps = eps
        self.use_scaling = use_scaling

        self.register_buffer("edge_scale", torch.zeros(1))
        self.register_buffer("img_scale", torch.zeros(1))
        self.register_buffer("num_warmup", torch.zeros(1, dtype=torch.long))

        self.scaling_frozen = False

    def forward(self, sr, hr):
        loss_img = self.pixel(sr, hr)
        loss_edge = self.pixel(sobel_filter(sr), sobel_filter(hr))

        if not self.use_scaling:
            return loss_img + loss_edge

        if not self.scaling_frozen:
            self.edge_scale += loss_edge.detach()
            self.img_scale += loss_img.detach()
            self.num_warmup += 1

            if self.num_warmup >= self.warmup_batches:
                self.edge_scale /= self.num_warmup
                self.img_scale /= self.num_warmup
                self.scaling_frozen = True

        if not self.scaling_frozen:
            return loss_img + loss_edge

        loss_img = loss_img / (self.img_scale + self.eps)
        loss_edge = loss_edge / (self.edge_scale + self.eps)
        print(f"Edge Loss: {loss_edge.item():.4f}, Img Loss: {loss_img.item():.4f}")
        return loss_img + loss_edge


class FourierLoss(nn.Module):
    def __init__(self, warmup_batches: int = 10, eps: float = 1e-6, use_scaling: bool = True):
        super().__init__()

        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

        self.use_scaling = use_scaling
        self.warmup_batches = warmup_batches
        self.eps = eps

        self.register_buffer("freq_scale", torch.zeros(1))
        self.register_buffer("img_scale", torch.zeros(1))
        self.register_buffer("num_warmup", torch.zeros(1, dtype=torch.long))

        self.scaling_frozen = False

    def forward(self, sr, hr):
        sr_fft = torch.fft.fftshift(torch.fft.fft2(sr))
        hr_fft = torch.fft.fftshift(torch.fft.fft2(hr))

        freq_loss = self.mse(torch.abs(sr_fft), torch.abs(hr_fft))
        img_loss = self.mae(sr, hr)

        if not self.use_scaling:
            return freq_loss + img_loss

        if not self.scaling_frozen:
            self.freq_scale += freq_loss.detach()
            self.img_scale += img_loss.detach()
            self.num_warmup += 1

            if self.num_warmup >= self.warmup_batches:
                self.freq_scale /= self.num_warmup
                self.img_scale /= self.num_warmup
                self.scaling_frozen = True

        if not self.scaling_frozen:
            return freq_loss + img_loss

        freq_loss = freq_loss / (self.freq_scale + self.eps)
        img_loss = img_loss / (self.img_scale + self.eps)

        print(f"Freq Loss: {freq_loss.item():.4f}, Img Loss: {img_loss.item():.4f}")
        return freq_loss + img_loss


class VGGPerceptualLoss(nn.Module):
    def __init__(self, warmup_batches: int = 10, eps: float = 1e-6, use_scaling: bool = True, device="cuda"):
        super().__init__()

        self.warmup_batches = warmup_batches
        self.eps = eps
        self.use_scaling = use_scaling

        self.register_buffer("perc_scale", torch.zeros(1))
        self.register_buffer("img_scale", torch.zeros(1))
        self.register_buffer("num_warmup", torch.zeros(1, dtype=torch.long))
        self.scaling_frozen = False

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        self.layer_ids = {3, 8, 13, 22, 31}
        self.pixel = nn.L1Loss()
        self.vgg = vgg.to(device).eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, sr, hr):
        loss_perc = 0.0
        loss_pixel = self.pixel(sr, hr)

        sr = normalize_vgg(sr)
        hr = normalize_vgg(hr)

        for i, layer in enumerate(self.vgg.children()):
            sr = layer(sr)
            hr = layer(hr)
            if i in self.layer_ids:
                loss_perc += F.l1_loss(sr, hr)

        if not self.use_scaling:
            return loss_perc + loss_pixel

        if not self.scaling_frozen:
            self.perc_scale += loss_perc.detach()
            self.img_scale += loss_pixel.detach()
            self.num_warmup += 1

            if self.num_warmup >= self.warmup_batches:
                self.perc_scale /= self.num_warmup
                self.img_scale /= self.num_warmup
                self.scaling_frozen = True

        if not self.scaling_frozen:
            return loss_pixel + loss_perc

        loss_perc = loss_perc / (self.perc_scale + self.eps)
        loss_pixel = loss_pixel / (self.img_scale + self.eps)
        print(f"Perc Loss: {loss_perc.item():.4f}, Img Loss: {loss_pixel.item():.4f}")
        return loss_pixel + loss_perc


def normalize_vgg(x):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def get_loss_function(loss_name: str, use_scaling: bool = True) -> nn.Module:
    if loss_name == "mae":
        return nn.L1Loss()
    elif loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "sobel":
        return SobelLoss(use_scaling=use_scaling)
    elif loss_name == "fourier":
        return FourierLoss(use_scaling=use_scaling)
    elif loss_name == "vgg_perceptual":
        return VGGPerceptualLoss(use_scaling=use_scaling)
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")
