import torch
import torch.nn as nn
import torch.nn.functional as F
import fastmri
from typing import Optional

class _IFFT2c(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return fastmri.ifft2c(x)
    @staticmethod
    def backward(ctx, grad_output):
        return fastmri.fft2c(grad_output)


class _FFT2c(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return fastmri.fft2c(x)
    @staticmethod
    def backward(ctx, grad_output):
        return fastmri.ifft2c(grad_output)


def ifft2c_dp(x):
    return _IFFT2c.apply(x)

def fft2c_dp(x):
    return _FFT2c.apply(x)

class KSpaceCNN(nn.Module):
    def __init__(self, channels: int = 64, num_layers: int = 5, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(2, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2),
        ]
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.InstanceNorm2d(channels),
                nn.LeakyReLU(0.2),
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
            self.lam.requires_grad = False

    def forward(
        self,
        k_predicted: torch.Tensor,
        k_measured: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        m = mask.float()
        while m.dim() > k_predicted.dim():
            m = m.squeeze(2)
        # Crop H and W if mask was built for full-res but k is center-cropped
        for spatial_dim in [-2, -1]:
            size = m.shape[spatial_dim]
            target = k_predicted.shape[spatial_dim]
            if size > 1 and size != target:
                start = (size - target) // 2
                m = m.narrow(m.dim() + spatial_dim, start, target)

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
            nn.ReLU(),
        ]
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.InstanceNorm2d(channels),
                nn.ReLU(),
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
        if kspace.shape[0] == 0:
            k_refined = self.kspace_cnn(kspace)                 # was missing
            _ = self.dc(k_refined, kspace_measured, mask)       # no-grad params, fine
            magnitude = torch.zeros((0, 1, kspace.shape[2], kspace.shape[3]),
                                    device=kspace.device, dtype=kspace.dtype)
            magnitude = self.image_refine(magnitude)
            k_next_2ch = torch.zeros((0, 2, kspace.shape[2], kspace.shape[3]),
                                     device=kspace.device, dtype=kspace.dtype)
            return magnitude, k_next_2ch
        k_refined = self.kspace_cnn(kspace)
        k_dc = self.dc(k_refined, kspace_measured, mask)
        k_dc_real = k_dc.permute(0, 2, 3, 1).contiguous().unsqueeze(1)
        image_complex = ifft2c_dp(k_dc_real)          # was fastmri.ifft2c(...)
        magnitude = fastmri.complex_abs(image_complex)
        magnitude = self.image_refine(magnitude)
        phase = torch.atan2(image_complex[..., 1], image_complex[..., 0])
        refined_real = magnitude * torch.cos(phase)
        refined_imag = magnitude * torch.sin(phase)
        refined_2ch = torch.stack([refined_real, refined_imag], dim=-1)
        k_next = fft2c_dp(refined_2ch)                 # was fastmri.fft2c(...)
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
