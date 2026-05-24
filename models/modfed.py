import torch
import torch.nn as nn
import torch.nn.functional as F
import fastmri
from typing import Optional


class KSpaceCNN(nn.Module):
    def __init__(self, channels: int = 64, num_layers: int = 5, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(2, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.InstanceNorm2d(channels),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(dropout),
            ]
        layers.append(nn.Conv2d(channels, 2, 3, padding=1, bias=False))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")

    def forward(self, kspace: torch.Tensor) -> torch.Tensor:
        return kspace + self.net(kspace)


class DataConsistency(nn.Module):
    def __init__(self, soft: bool = False):
        super().__init__()
        self.soft = soft
        if soft:
            self.lam = nn.Parameter(torch.ones(1))

    def forward(
        self,
        k_predicted: torch.Tensor,  # (B, 2, H, W)
        k_measured: torch.Tensor,   # (B, 2, H, W)
        mask: torch.Tensor,         # (B, 1, 1, W) or (B, 1, H, W) bool
    ) -> torch.Tensor:
        m = mask.float()
        if m.shape[1] == 1:
            m = m.expand_as(k_predicted)

        if self.soft:
            lam = torch.sigmoid(self.lam)
            return m * (lam * k_measured + (1 - lam) * k_predicted) + (1 - m) * k_predicted
        else:
            return m * k_measured + (1 - m) * k_predicted


class ImageRefineCNN(nn.Module):
    def __init__(self, channels: int = 32, num_layers: int = 3):
        super().__init__()
        layers = [
            nn.Conv2d(1, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.InstanceNorm2d(channels),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(channels, 1, 3, padding=1, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ModFedCascade(nn.Module):
    def __init__(self, kspace_channels=64, kspace_layers=5, image_channels=32,
                 soft_dc=True, dropout=0.0):
        super().__init__()
        self.kspace_cnn = KSpaceCNN(kspace_channels, kspace_layers, dropout)
        self.dc = DataConsistency(soft=soft_dc)
        self.image_refine = ImageRefineCNN(image_channels)

    def forward(self, kspace, kspace_measured, mask):
        k_refined = self.kspace_cnn(kspace)
        k_dc = self.dc(k_refined, kspace_measured, mask)
        k_dc_real = k_dc.permute(0, 2, 3, 1).contiguous().unsqueeze(1)
        image_complex = fastmri.ifft2c(k_dc_real)
        magnitude = fastmri.complex_abs(image_complex)
        magnitude = self.image_refine(magnitude)
        phase = torch.atan2(image_complex[..., 1], image_complex[..., 0])
        refined_real = magnitude * torch.cos(phase)
        refined_imag = magnitude * torch.sin(phase)
        refined_2ch = torch.stack([refined_real, refined_imag], dim=-1)
        k_next = fastmri.fft2c(refined_2ch)
        k_next_2ch = k_next.squeeze(1).permute(0, 3, 1, 2).contiguous()
        return magnitude, k_next_2ch


class ModFed(nn.Module):
    def __init__(self, num_cascades=6, kspace_ch=64, kspace_layers=5,
                 image_ch=32, soft_dc=True, dropout=0.0):
        super().__init__()
        self.cascades = nn.ModuleList([
            ModFedCascade(kspace_ch, kspace_layers, image_ch, soft_dc, dropout)
            for _ in range(num_cascades)
        ])

    def forward(self, kspace, mask):
        k_current = kspace
        image_out = None
        for cascade in self.cascades:
            image_out, k_current = cascade(k_current, kspace, mask)
        return image_out


from models.unet import ReconstructionLoss, ssim
