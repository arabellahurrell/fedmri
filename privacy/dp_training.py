import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict
import numpy as np

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator


def make_dp_compatible(model: nn.Module) -> nn.Module:
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        model = ModuleValidator.fix(model)
        errors = ModuleValidator.validate(model, strict=True)
        if errors:
            raise ValueError(f"Model still has DP-incompatible layers: {errors}")
    return model


class DPTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        target_epsilon: float = 5.0,
        target_delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: Optional[float] = None,
        lr: float = 1e-3,
        epochs: int = 10,
        device: Optional[torch.device] = None,
        max_physical_batch_size: int = 8,
    ):
        self.device = device or torch.device("cpu")
        self.model = make_dp_compatible(model).to(self.device)
        self.train_loader = train_loader
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.lr = lr
        self.epochs = epochs
        self.max_physical_batch_size = max_physical_batch_size
        self._dp_model = None
        self._dp_optimizer = None
        self._privacy_engine = None
        self._is_setup = False

    def setup(self) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        privacy_engine = PrivacyEngine()
        if self.noise_multiplier is not None:
            dp_model, dp_optimizer, dp_loader = privacy_engine.make_private(
                module=self.model, optimizer=optimizer, data_loader=self.train_loader,
                noise_multiplier=self.noise_multiplier, max_grad_norm=self.max_grad_norm,
            )
        else:
            dp_model, dp_optimizer, dp_loader = privacy_engine.make_private_with_epsilon(
                module=self.model, optimizer=optimizer, data_loader=self.train_loader,
                epochs=self.epochs, target_epsilon=self.target_epsilon,
                target_delta=self.target_delta, max_grad_norm=self.max_grad_norm,
            )
        self._dp_model = dp_model
        self._dp_optimizer = dp_optimizer
        self._dp_loader = dp_loader
        self._privacy_engine = privacy_engine
        self._is_setup = True

    def train_epoch(self, domain: str) -> float:
        if not self._is_setup:
            raise RuntimeError("Call setup() first.")
        from models.unet import ReconstructionLoss
        loss_fn = ReconstructionLoss()
        self._dp_model.train()
        total_loss = 0.0
        n_batches = 0
        with BatchMemoryManager(
            data_loader=self._dp_loader,
            max_physical_batch_size=self.max_physical_batch_size,
            optimizer=self._dp_optimizer,
        ) as memory_safe_loader:
            for batch in memory_safe_loader:
                self._dp_optimizer.zero_grad()
                if domain == "image":
                    x = batch["image_input"].to(self.device)
                    y = batch["image_target"].to(self.device)
                    pred = self._dp_model(x).squeeze(1)
                else:
                    k = batch["kspace"].to(self.device)
                    mask = batch["mask"].to(self.device)
                    y = batch["image_target"].to(self.device)
                    pred = self._dp_model(k, mask).squeeze(1)
                loss = loss_fn(pred, y)
                loss.backward()
                self._dp_optimizer.step()
                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(n_batches, 1)

    def get_epsilon(self) -> float:
        if self._privacy_engine is None:
            return 0.0
        return self._privacy_engine.get_epsilon(self.target_delta)

    def get_model(self) -> nn.Module:
        if self._dp_model is not None:
            return self._dp_model._module
        return self.model


class DPFedMRIClient:

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        domain: str,
        local_epochs: int = 2,
        lr: float = 1e-3,
        target_epsilon: float = 5.0,
        target_delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.client_id = client_id
        self.val_loader = val_loader
        self.domain = domain
        self.local_epochs = local_epochs
        self.device = device or torch.device("cpu")
        self.dp_trainer = DPTrainer(
            model=model, train_loader=train_loader,
            target_epsilon=target_epsilon, target_delta=target_delta,
            max_grad_norm=max_grad_norm, lr=lr,
            epochs=local_epochs,
            device=self.device,
        )

    def set_total_epochs(self, num_fl_rounds: int) -> None:
        self.dp_trainer.epochs = num_fl_rounds * self.local_epochs

    def setup(self) -> None:
        self.dp_trainer.setup()

    def fit(self, parameters, config):
        from federated.fl_simulation import set_parameters, get_parameters
        set_parameters(self.dp_trainer.get_model(), parameters)

        for _ in range(self.local_epochs):
            self.dp_trainer.train_epoch(self.domain)

        eps = self.dp_trainer.get_epsilon()
        print(f"  Client {self.client_id} — round ε (not cumulative!): {eps:.3f}")

        updated_params = get_parameters(self.dp_trainer.get_model())
        n = len(self.dp_trainer._dp_loader.dataset)
        return updated_params, n, {"epsilon": float(eps)}

    def get_epsilon(self) -> float:
        return self.dp_trainer.get_epsilon()


EPSILON_GRID = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, float("inf")]


def epsilon_to_noise_multiplier(epsilon, delta, epochs, dataset_size, batch_size, max_grad_norm=1.0):
    if epsilon == float("inf"):
        return 0.0
    from opacus.accountants.utils import get_noise_multiplier
    return get_noise_multiplier(
        target_epsilon=epsilon, target_delta=delta,
        sample_rate=batch_size / dataset_size, epochs=epochs,
    )
