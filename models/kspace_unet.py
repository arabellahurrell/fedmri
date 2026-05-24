import torch
import torch.nn as nn
import torch.nn.functional as F
import fastmri

from models.unet import ConvBlock, Down, Up, ReconstructionLoss


class KSpaceUNet(nn.Module):
    def __init__(
        self,
        base_features: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.depth = depth
        features = [base_features * (2 ** i) for i in range(depth + 1)]

        self.inc = ConvBlock(2, features[0], dropout)
        self.downs = nn.ModuleList(
            [Down(features[i], features[i + 1], dropout) for i in range(depth)]
        )
        self.ups = nn.ModuleList(
            [Up(features[i + 1] + features[i], features[i], dropout)
             for i in range(depth - 1, -1, -1)]
        )
        self.outc = nn.Conv2d(features[0], 2, kernel_size=1)

    def forward(self, kspace: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(kspace)]
        for down in self.downs:
            skips.append(down(skips[-1]))

        h = skips[-1]
        for i, up in enumerate(self.ups):
            h = up(h, skips[-(i + 2)])

        k_correction = self.outc(h)
        k_refined = kspace + k_correction

        k_real = k_refined.permute(0, 2, 3, 1).unsqueeze(1)   # (B, 1, H, W, 2)
        image_complex = fastmri.ifft2c(k_real)                 # (B, 1, H, W, 2)
        magnitude = fastmri.complex_abs(image_complex)         # (B, 1, H, W)

        return magnitude
