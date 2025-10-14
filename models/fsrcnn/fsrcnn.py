import math

import torch
import torch.nn as nn


# With bigger pretraining weights
# Weights: checkpoints/FSRCNN-x2.pt
# Best model: fsrcnn_finetuned_0.0036803.pth
class FSRCNN_BIG_PRETRAIN(nn.Module):
    def __init__(self, scale: int) -> None:
        super(FSRCNN_BIG_PRETRAIN, self).__init__()

        if scale not in [2, 3, 4]:
            raise ValueError("must be 2, 3 or 4")

        d = 56
        s = 12

        self.feature_extract = nn.Conv2d(in_channels=3, out_channels=d, kernel_size=5, padding=2)
        nn.init.kaiming_normal_(self.feature_extract.weight)
        nn.init.zeros_(self.feature_extract.bias)

        self.activation_1 = nn.PReLU(num_parameters=d)

        self.shrink = nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1)
        nn.init.kaiming_normal_(self.shrink.weight)
        nn.init.zeros_(self.shrink.bias)

        self.activation_2 = nn.PReLU(num_parameters=s)

        # m = 4
        self.map_1 = nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.map_1.weight)
        nn.init.zeros_(self.map_1.bias)

        self.map_2 = nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.map_2.weight)
        nn.init.zeros_(self.map_2.bias)

        self.map_3 = nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.map_3.weight)
        nn.init.zeros_(self.map_3.bias)

        self.map_4 = nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.map_4.weight)
        nn.init.zeros_(self.map_4.bias)

        self.activation_3 = nn.PReLU(num_parameters=s)

        self.expand = nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1)
        nn.init.kaiming_normal_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

        self.activation_4 = nn.PReLU(num_parameters=d)

        self.deconv = nn.ConvTranspose2d(
            in_channels=d, out_channels=3, kernel_size=9, stride=scale, padding=4, output_padding=scale - 1
        )
        nn.init.normal_(self.deconv.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.deconv.bias)

    def forward(self, X_in):
        X = self.feature_extract(X_in)
        X = self.activation_1(X)

        X = self.shrink(X)
        X = self.activation_2(X)

        X = self.map_1(X)
        X = self.map_2(X)
        X = self.map_3(X)
        X = self.map_4(X)
        X = self.activation_3(X)

        X = self.expand(X)
        X = self.activation_4(X)

        X = self.deconv(X)
        X_out = torch.clip(X, 0.0, 1.0)

        return X_out


# With smaller pretraining weights
# Weights: checkpoints/fsrcnn_x2.pt
# Best model: fsrcnn_finetuned_0.0051799.pth
class FSRCNN_SMALL_PRETRAIN(nn.Module):
    def __init__(self, scale, num_channels=3, d=56, s=12, m=4):
        super(FSRCNN_SMALL_PRETRAIN, self).__init__()
        self.first_part = nn.Sequential(nn.Conv2d(num_channels, d, kernel_size=5, padding=5 // 2), nn.PReLU(d))
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3 // 2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(
            d, num_channels, kernel_size=9, stride=scale, padding=9 // 2, output_padding=scale - 1
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(
                    m.weight.data, mean=0.0, std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel()))
                )
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(
                    m.weight.data, mean=0.0, std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel()))
                )
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x


# Without any pretraining, learning solely on our Quake II training data
# Weights: fsrcnn_finetuned.pth
class FSRCNN_WITHOUT_PRETRAIN(nn.Module):
    def __init__(self, scale: int) -> None:
        super(FSRCNN_WITHOUT_PRETRAIN, self).__init__()
        self.feature_extraction = nn.Sequential(nn.Conv2d(3, 56, (5, 5), (1, 1), (2, 2)), nn.PReLU(56))
        self.shrink = nn.Sequential(nn.Conv2d(56, 12, (1, 1), (1, 1), (0, 0)), nn.PReLU(12))
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
        self.expand = nn.Sequential(nn.Conv2d(12, 56, (1, 1), (1, 1), (0, 0)), nn.PReLU(56))
        self.deconv = nn.ConvTranspose2d(
            56,
            3,
            (9, 9),
            (scale, scale),
            (4, 4),
            (scale - 1, scale - 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.feature_extraction(x)
        out = self.shrink(out)
        out = self.map(out)
        out = self.expand(out)
        out = self.deconv(out)
        return out
