import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import pandas as pd
import matplotlib.pyplot as plt

try:
    import lpips as lpips_lib
    _lpips_fn = lpips_lib.LPIPS(net="vgg", verbose=False)
    _lpips_fn.eval()
    _LPIPS_AVAILABLE = True
except ImportError:
    _lpips_fn = None
    _LPIPS_AVAILABLE = False

    def _lpips_batch(pred_np: np.ndarray, target_np: np.ndarray, device: torch.device) -> float:
        if not _LPIPS_AVAILABLE:
            return float("nan")

        def to_rgb(arr):
            t = torch.from_numpy(arr).float()
            t = t.unsqueeze(1).expand(-1, 3, -1, -1)
            return t * 2.0 - 1.0

        p = to_rgb(pred_np).to(device)
        t = to_rgb(target_np).to(device)
        fn = _lpips_fn.to(device)
        with torch.no_grad():
            scores = fn(p, t)
        return float(scores.mean().item())

def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> Dict[str, float]:
    pred_np = pred.detach().cpu().float().numpy()
    target_np = target.detach().cpu().float().numpy()

    if pred_np.ndim == 2:
        pred_np = pred_np[np.newaxis]
        target_np = target_np[np.newaxis]

    ssim_scores, psnr_scores, nmse_scores = [], [], []
    pred_norm_batch, target_norm_batch = [], []

    for p, t in zip(pred_np, target_np):
        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t_n = (t - t_min) / (t_max - t_min)
            p_n = np.clip((p - t_min) / (t_max - t_min), 0, 1)
        else:
            t_n = t
            p_n = p

        ssim_scores.append(structural_similarity(t_n, p_n, data_range=1.0))
        psnr_scores.append(peak_signal_noise_ratio(t_n, p_n, data_range=1.0))
        nmse_scores.append(float(np.sum((t_n - p_n) ** 2) / (np.sum(t_n ** 2) + 1e-8)))
        pred_norm_batch.append(p_n)
        target_norm_batch.append(t_n)

    device = pred.device if isinstance(pred, torch.Tensor) else torch.device("cpu")
    lpips_score = _lpips_batch(
        np.stack(pred_norm_batch),
        np.stack(target_norm_batch),
        device,
    )

    return {
        "ssim": float(np.mean(ssim_scores)),
        "psnr": float(np.mean(psnr_scores)),
        "nmse": float(np.mean(nmse_scores)),
        "lpips": lpips_score,
    }


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    domain: str,
    device: torch.device,
) -> Dict[str, float]:
    from models.unet import ReconstructionLoss
    loss_fn = ReconstructionLoss()

    model.eval()
    all_ssim, all_psnr, all_nmse, all_lpips, all_loss = [], [], [], [], []

    uses_mask = domain != "image" and not model.__class__.__name__ == "KSpaceUNet"
    for batch in loader:
        if domain == "image":
            x = batch["image_input"].to(device)
            y = batch["image_target"].to(device)
            pred = model(x).squeeze(1)
        elif uses_mask:
            k = batch["kspace"].to(device)
            mask = batch["mask"].to(device)
            y = batch["image_target"].to(device)
            pred = model(k, mask).squeeze(1)
        else:
            k = batch["kspace"].to(device)
            y = batch["image_target"].to(device)
            pred = model(k).squeeze(1)

        all_loss.append(loss_fn(pred, y).item())
        m = compute_metrics(pred, y)
        all_ssim.append(m["ssim"])
        all_psnr.append(m["psnr"])
        all_nmse.append(m["nmse"])
        all_lpips.append(m["lpips"])

    return {
        "loss": float(np.mean(all_loss)),
        "ssim": float(np.mean(all_ssim)),
        "psnr": float(np.mean(all_psnr)),
        "nmse": float(np.mean(all_nmse)),
        "lpips": float(np.nanmean(all_lpips)),
    }


class ResultsTracker:
    def __init__(self, save_dir: str = "results"):
        import os
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.records: List[dict] = []

    def log(self, **kwargs) -> None:
        self.records.append(kwargs)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def save_csv(self, filename: str = "results.csv") -> None:
        df = self.to_dataframe()
        path = f"{self.save_dir}/{filename}"
        df.to_csv(path, index=False)
        print(f"Results saved to {path}")

    def plot_privacy_utility_frontier(self, save=True, filename="privacy_utility_frontier.png"):
        df = self.to_dataframe()
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, grp in df.groupby(["model", "domain"]):
            grp_sorted = grp.sort_values("epsilon")
            ax.plot(grp_sorted["epsilon"], grp_sorted["ssim"],
                    marker="o", label=f"{label[0]} ({label[1]})")
        ax.set_xlabel("Privacy budget ε")
        ax.set_ylabel("SSIM")
        ax.set_title("Privacy–Utility Pareto Frontier")
        ax.set_xscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save:
            fig.savefig(f"{self.save_dir}/{filename}", dpi=150, bbox_inches="tight")
        return fig

    def plot_training_curves(self, metric="ssim", save=True, filename="training_curves.png"):
        df = self.to_dataframe()
        x_col = "round" if "round" in df.columns else "epoch" if "epoch" in df.columns else None
        if x_col is None:
            raise ValueError("Records must contain 'round' or 'epoch' column.")

        fig, ax = plt.subplots(figsize=(8, 5))
        for label, grp in df.groupby(["model", "domain"]):
            grp_sorted = grp.sort_values("round")
            ax.plot(grp_sorted["round"], grp_sorted[metric],
                    marker=".", label=f"{label[0]} ({label[1]})")
        ax.set_xlabel("Round / Epoch")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Training — {metric.upper()}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save:
            fig.savefig(f"{self.save_dir}/{filename}", dpi=150, bbox_inches="tight")
        return fig
