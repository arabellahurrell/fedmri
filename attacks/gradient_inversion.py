import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from collections import OrderedDict

def weight_deltas_to_gradients(
    model: nn.Module,
    weight_delta: list,
    lr: float = 1e-3,
) -> list:
    return [delta / lr for delta in weight_delta]


def set_gradients(model: nn.Module, gradients: list) -> None:
    for param, grad in zip(model.parameters(), gradients):
        param.grad = torch.tensor(grad, dtype=param.dtype, device=param.device)

class GradientInversionAttack:
    def __init__(
        self,
        model: nn.Module,
        domain: str = "image",
        device: Optional[torch.device] = None,
        attack_cfg: Optional[dict] = None,
    ):
        self.model = copy.deepcopy(model)
        self.domain = domain
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        # Default breaching config overrides
        self.attack_cfg = {
            "attack": "fishing_for_user_data",   # or "invertgradients"
            "optim": {
                "optimizer": "adam",
                "lr": 0.1,
                "max_iterations": 3000,
                "restarts": 4,
            },
            "regularization": {
                "total_variation": {"scale": 0.01},
            },
        }
        if attack_cfg:
            self.attack_cfg.update(attack_cfg)

    def run(
        self,
        weight_delta: list,
        lr: float = 1e-3,
        ground_truth_batch: Optional[dict] = None,
        batch_size: int = 1,
    ) -> Tuple[torch.Tensor, dict]:
        try:
            import breaching
        except ImportError:
            raise ImportError(
                "breaching is not installed. Run: bash scripts/setup_breaching.sh"
            )

        gradients = weight_deltas_to_gradients(self.model, weight_delta, lr)
        grad_tensors = [
            torch.tensor(g, dtype=p.dtype, device=self.device)
            for g, p in zip(gradients, self.model.parameters())
        ]

        # Set up breaching config
        cfg = breaching.get_config(overrides=[])
        # Apply our overrides manually
        for key, val in self.attack_cfg.get("optim", {}).items():
            setattr(cfg.attack.optim, key, val)

        attacker = breaching.attacks.prepare_attack(
            self.model, cfg.attack, self.device
        )

        server_payload = [{
            "gradients": grad_tensors,
            "buffers": None,
            "metadata": {
                "num_data_points": batch_size,
                "labels": None,
                "local_hyperparams": None,
            },
        }]

        shared_data = {
            "gradients": grad_tensors,
            "buffers": None,
            "metadata": {"num_data_points": batch_size, "labels": None},
        }

        # Run attack
        reconstructed, atk_stats = attacker.reconstruct(
            server_payload, shared_data, {}, dryrun=False
        )

        metrics = {}
        if ground_truth_batch is not None:
            gt = self._extract_input(ground_truth_batch).to(self.device)
            metrics = self._compute_recon_metrics(reconstructed, gt)

        return reconstructed, metrics

    def _extract_input(self, batch: dict) -> torch.Tensor:
        if self.domain == "image":
            return batch["image_input"]
        else:
            return batch["kspace"]

    def _compute_recon_metrics(
        self, recon: torch.Tensor, target: torch.Tensor
    ) -> dict:
        from evaluation.metrics import compute_metrics
        # Use magnitude for k-space
        if self.domain == "kspace":
            recon = torch.sqrt(recon[:, 0] ** 2 + recon[:, 1] ** 2).unsqueeze(1)
            target = torch.sqrt(target[:, 0] ** 2 + target[:, 1] ** 2).unsqueeze(1)
        m = compute_metrics(recon.squeeze(1), target.squeeze(1))
        return m

class TVGradientInversion:
    def __init__(
        self,
        model: nn.Module,
        domain: str = "image",
        device: Optional[torch.device] = None,
        lr: float = 0.1,
        num_iters: int = 2000,
        tv_weight: float = 1e-4,
        restarts: int = 3,
    ):
        self.model = copy.deepcopy(model).eval()
        self.domain = domain
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.lr = lr
        self.num_iters = num_iters
        self.tv_weight = tv_weight
        self.restarts = restarts

    def run(
        self,
        weight_delta: list,
        input_shape: Tuple,
        lr_client: float = 1e-3,
        ground_truth: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        true_grads = [
            torch.tensor(g / lr_client, device=self.device, dtype=torch.float32)
            for g in weight_delta
        ]

        best_recon = None
        best_loss = float("inf")
        from tqdm import tqdm

        for restart in range(self.restarts):
            dummy = torch.randn(input_shape, device=self.device, requires_grad=True)
            optim = torch.optim.Adam([dummy], lr=self.lr)
            from models.unet import ReconstructionLoss
            loss_fn = ReconstructionLoss()

            pbar = tqdm(range(self.num_iters),
                        desc=f"  Restart {restart+1}/{self.restarts}",
                        leave=False)
            
            for it in pbar:
                optim.zero_grad()

                if self.domain == "image":
                    dummy_out = self.model(dummy)
                else:
                    B, _, H, W = input_shape
                    mask = torch.ones(B, 1, 1, W, device=self.device, dtype=torch.bool)
                    dummy_out = self.model(dummy, mask)

                
                dummy_target = torch.zeros_like(dummy_out)
                dummy_loss = loss_fn(dummy_out.squeeze(1), dummy_target.squeeze(1))
                dummy_grads = torch.autograd.grad(
                    dummy_loss, self.model.parameters(), create_graph=True, allow_unused=True
                )
                dummy_grads = [g if g is not None else torch.zeros_like(p) 
                            for g, p in zip(dummy_grads, self.model.parameters())]

                grad_loss = sum(
                    ((dg - tg) ** 2).sum()
                    for dg, tg in zip(dummy_grads, true_grads)
                )

                tv = (
                    ((dummy[:, :, 1:] - dummy[:, :, :-1]) ** 2).sum()
                    + ((dummy[:, :, :, 1:] - dummy[:, :, :, :-1]) ** 2).sum()
                )

                total = grad_loss + self.tv_weight * tv
                total.backward()
                optim.step()
                pbar.set_postfix({"loss": f"{total.item():.4f}"})

                with torch.no_grad():
                    dummy.clamp_(-1.5, 1.5)

            loss_val = total.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_recon = dummy.detach().clone()

        metrics = {}
        if ground_truth is not None:
            from evaluation.metrics import compute_metrics
            gt = ground_truth.to(self.device)
            if self.domain == "kspace":
                best_recon_mag = torch.sqrt(best_recon[:, 0] ** 2 + best_recon[:, 1] ** 2)
                gt_mag = torch.sqrt(gt[:, 0] ** 2 + gt[:, 1] ** 2)
                metrics = compute_metrics(best_recon_mag, gt_mag)
            else:
                metrics = compute_metrics(
                    best_recon[:, 0],
                    gt[:, 0] if gt.shape[1] > 1 else gt.squeeze(1),
                )

        return best_recon, metrics
