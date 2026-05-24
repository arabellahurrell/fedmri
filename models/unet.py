# standard unet code
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torchvision.ops import StochasticDepth  # unused — left in by mistake


class ConvBlock(nn.Module):
    # two layer convblock
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diffH = skip.size(2) - x.size(2)
        diffW = skip.size(3) - x.size(3)
        x = F.pad(x, [diffW // 2, diffW - diffW // 2,
                       diffH // 2, diffH - diffH // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_features: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.depth = depth
        features = [base_features * (2 ** i) for i in range(depth + 1)]

        self.inc = ConvBlock(in_channels, features[0], dropout)
        self.downs = nn.ModuleList(
            [Down(features[i], features[i + 1], dropout) for i in range(depth)]
        )

        self.ups = nn.ModuleList(
            [Up(features[i] * 2, features[i - 1] if i > 0 else features[0], dropout)
             for i in range(depth, 0, -1)]
        )

        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        h = skips[-1]
        for i, up in enumerate(self.ups):
            skip = skips[-(i + 2)]
            h = up(h, skip)
        return self.outc(h)


class ReconstructionLoss(nn.Module):
    def __init__(self, lambda1: float = 0.84, lambda2: float = 0.16):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)
        l1 = F.l1_loss(pred, target)
        ssim_val = ssim(pred, target)
        return self.lambda1 * l1 + self.lambda2 * (1.0 - ssim_val)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 7,
    data_range: float = 1.0,
) -> torch.Tensor:

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    pad = window_size // 2

    # Kernel recreated every call
    kernel = torch.ones(
        1, 1, window_size, window_size, device=pred.device
    ) / (window_size ** 2)

    mu_x = F.conv2d(pred, kernel, padding=pad, groups=1)
    mu_y = F.conv2d(target, kernel, padding=pad, groups=1)
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred ** 2, kernel, padding=pad, groups=1) - mu_x2
    sigma_y2 = F.conv2d(target ** 2, kernel, padding=pad, groups=1) - mu_y2
    sigma_xy = F.conv2d(pred * target, kernel, padding=pad, groups=1) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

    return (num / den).mean()
