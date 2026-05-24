import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import pandas as pd
import matplotlib.pyplot as plt


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

    return {
        "ssim": float(np.mean(ssim_scores)),
        "psnr": float(np.mean(psnr_scores)),
        "nmse": float(np.mean(nmse_scores)),
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
    all_ssim, all_psnr, all_nmse, all_loss = [], [], [], []

    for batch in loader:
        if domain == "image":
            x = batch["image_input"].to(device)
            y = batch["image_target"].to(device)
            pred = model(x).squeeze(1)
        else:
            k = batch["kspace"].to(device)
            mask = batch["mask"].to(device)
            y = batch["image_target"].to(device)
            pred = model(k, mask).squeeze(1)

        all_loss.append(loss_fn(pred, y).item())
        m = compute_metrics(pred, y)
        all_ssim.append(m["ssim"])
        all_psnr.append(m["psnr"])
        all_nmse.append(m["nmse"])

    return {
        "loss": float(np.mean(all_loss)),
        "ssim": float(np.mean(all_ssim)),
        "psnr": float(np.mean(all_psnr)),
        "nmse": float(np.mean(all_nmse)),
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
        if "round" not in df.columns:
            raise ValueError("Records must contain 'round' column.")

        fig, ax = plt.subplots(figsize=(8, 5))
        for label, grp in df.groupby(["model", "domain"]):
            grp_sorted = grp.sort_values("round")
            ax.plot(grp_sorted["round"], grp_sorted[metric],
                    marker=".", label=f"{label[0]} ({label[1]})")
        ax.set_xlabel("Round")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Training — {metric.upper()}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save:
            fig.savefig(f"{self.save_dir}/{filename}", dpi=150, bbox_inches="tight")
        return fig
