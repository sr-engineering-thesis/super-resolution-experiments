import torch
from torch import nn


class FSRCNN(nn.Module):
    def __init__(self, upscale_factor: int) -> None:
        super(FSRCNN, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(1, 56, (5, 5), (1, 1), (2, 2)), nn.PReLU(56)
        )
        self.shrink = nn.Sequential(
            nn.Conv2d(56, 12, (1, 1), (1, 1), (0, 0)), nn.PReLU(12)
        )
        self.map = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
        )
        self.expand = nn.Sequential(
            nn.Conv2d(12, 56, (1, 1), (1, 1), (0, 0)), nn.PReLU(56)
        )
        self.deconv = nn.ConvTranspose2d(
            56,
            1,
            (9, 9),
            (upscale_factor, upscale_factor),
            (4, 4),
            (upscale_factor - 1, upscale_factor - 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # helper method
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.feature_extraction(x)
        out = self.shrink(out)
        out = self.map(out)
        out = self.expand(out)
        out = self.deconv(out)
        return out
