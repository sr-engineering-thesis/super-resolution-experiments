from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchsr.models import carn, carn_m, edsr_r16f64, edsr_r32f256, ninasr_b0, rcan

from experiments.ninasr_julia import NinaSR
from models.fsrcnn.fsrcnn import FSRCNN_BIG_PRETRAIN, FSRCNN_SMALL_PRETRAIN, FSRCNN_WITHOUT_PRETRAIN


@dataclass
class ModelConfig:
    """Configuration for a super-resolution model."""

    name: str
    scale: int = 4
    precision: Optional[torch.dtype] = torch.float32
    checkpoint: Optional[str] = None

    def __repr__(self):
        return f"Model: {self.name} | Scale: {self.scale} | Precision: {self.precision} | Checkpoint: {self.checkpoint}"


class SRModelWrapper(ABC):
    """Abstract base class for all SR model wrappers."""

    def __init__(self, config: ModelConfig, device: torch.device):
        self.config = config
        self.device = device
        self.model = self._build_model()

    @abstractmethod
    def _build_model(self):
        """Build and return the model instance."""
        pass

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference on input tensor.

        Args:
            x: Input tensor of shape [B, C, H, W] in range [0, 1]

        Returns:
            Output tensor of shape [B, C, H*scale, W*scale] in range [0, 1]
        """
        pass

    def eval(self):
        if hasattr(self.model, "eval"):
            self.model.eval()
        return self


class TorchInterpolationWrapper(SRModelWrapper):
    MODES = {"bilinear_torch": "bilinear", "bicubic_torch": "bicubic", "nearest_torch": "nearest"}

    def _build_model(self):
        mode_name = self.config.name.lower()
        if mode_name not in self.MODES:
            raise ValueError(f"Unknown interpolation mode: {mode_name}")
        return self.MODES[mode_name]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.config.scale

        with torch.no_grad():
            with torch.autocast(
                device_type="cuda", dtype=self.config.precision, enabled=self.config.precision != torch.float32
            ):
                out = F.interpolate(x, scale_factor=scale, mode=self.MODES[self.config.name.lower()])
        return out


class CV2InterpolationWrapper(SRModelWrapper):
    MODES = {
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }

    def _build_model(self):
        mode_name = self.config.name.lower()
        if mode_name not in self.MODES:
            raise ValueError(f"Unknown interpolation mode: {mode_name}")
        return self.MODES[mode_name]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] in [0, 1]
        x_np = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
        h, w = x_np.shape[:2]
        new_size = (int(w * self.config.scale), int(h * self.config.scale))
        out_np = cv2.resize(x_np, new_size, interpolation=self.model)
        out_np = np.transpose(out_np, (2, 0, 1))[None, ...]
        return torch.from_numpy(out_np).to(x.device)


class TorchSRWrapper(SRModelWrapper):
    MODEL_CLASSES = {
        "carn": carn,
        "ninasr_b0": ninasr_b0,
        "ninasr_julia": NinaSR,
        "edsr_r32f256": edsr_r32f256,
        "carn_m": carn_m,
        "edsr_r16f64": edsr_r16f64,
        "rcan": rcan,
        "carn_finetuned": carn,
    }

    def _build_model(self):
        model_cls = self.MODEL_CLASSES.get(self.config.name.lower())
        if model_cls is None:
            raise ValueError(f"Unknown torchsr model: {self.config.name}")
        if self.config.name.lower() == "ninasr_julia":
            model = model_cls(scale=self.config.scale, n_resblocks=8, n_feats=16).to(self.device)
        else:
            model = model_cls(scale=self.config.scale, pretrained=True).to(self.device)

        if self.config.checkpoint:
            state_dict = torch.load(self.config.checkpoint, map_location=self.device)
            model.load_state_dict(state_dict)

        return model

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda", dtype=self.config.precision, enabled=self.config.precision != torch.float32
            ):
                return self.model(x)


class CustomTorchWrapper(SRModelWrapper):
    MODEL_CLASSES = {
        "fsrcnn_big": FSRCNN_BIG_PRETRAIN,
        "fsrcnn_big_vgg": FSRCNN_BIG_PRETRAIN,
        "fsrcnn_big_fourier": FSRCNN_BIG_PRETRAIN,
        "fsrcnn_big_sobel": FSRCNN_BIG_PRETRAIN,
        "fsrcnn_big_mse": FSRCNN_BIG_PRETRAIN,
        "fsrcnn_big_mae": FSRCNN_BIG_PRETRAIN,
        "fsrcnn_small": FSRCNN_SMALL_PRETRAIN,
        "fsrcnn_without": FSRCNN_WITHOUT_PRETRAIN,
    }

    def _build_model(self):
        model_cls = self.MODEL_CLASSES.get(self.config.name.lower())
        if model_cls is None:
            raise ValueError(f"Unknown custom model: {self.config.name}")

        model = model_cls(scale=self.config.scale).to(self.device)

        if self.config.checkpoint:
            state_dict = torch.load(self.config.checkpoint, map_location=self.device)
            model.load_state_dict(state_dict)

        return model

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda", dtype=self.config.precision, enabled=self.config.precision != torch.float32
            ):
                return self.model(x)


class ModelFactory:
    WRAPPER_MAP = {
        "bilinear": CV2InterpolationWrapper,
        "bicubic": CV2InterpolationWrapper,
        "lanczos": CV2InterpolationWrapper,
        "carn": TorchSRWrapper,
        "ninasr_b0": TorchSRWrapper,
        "fsrcnn_big": CustomTorchWrapper,
        "ninasr_julia": TorchSRWrapper,
        "edsr_r32f256": TorchSRWrapper,
        "carn_m": TorchSRWrapper,
        "edsr_r16f64": TorchSRWrapper,
        "rcan": TorchSRWrapper,
        "carn_finetuned": TorchSRWrapper,
        "fsrcnn_small": CustomTorchWrapper,
        "fsrcnn_without": CustomTorchWrapper,
        "bilinear_torch": TorchInterpolationWrapper,
        "bicubic_torch": TorchInterpolationWrapper,
        "nearest_torch": TorchInterpolationWrapper,
        "fsrcnn_big_vgg": CustomTorchWrapper,
        "fsrcnn_big_fourier": CustomTorchWrapper,
        "fsrcnn_big_sobel": CustomTorchWrapper,
        "fsrcnn_big_mse": CustomTorchWrapper,
        "fsrcnn_big_mae": CustomTorchWrapper,
    }

    @staticmethod
    def create(config: ModelConfig, device: torch.device) -> SRModelWrapper:
        """Create a model wrapper from config."""
        wrapper_cls = ModelFactory.WRAPPER_MAP.get(config.name.lower())
        if wrapper_cls is None:
            raise ValueError(f"Unknown model: {config.name}. Available: {list(ModelFactory.WRAPPER_MAP.keys())}")

        return wrapper_cls(config, device).eval()
