# using flwr
import copy
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import (
    FitRes, Parameters, Scalar, NDArrays,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

from models.unet import UNet, ReconstructionLoss
from models.modfed import ModFed
from models.kspace_unet import KSpaceUNet
from evaluation.metrics import compute_metrics, evaluate_model
from privacy.dp_training import DPTrainer, make_dp_compatible

MODEL_DOMAINS = {
    "unet":   "image",
    "modfed": "kspace",
    "kspace_unet": "kspace",
}

def _import_dp():
    try:
        from privacy.dp_training import DPTrainer, make_dp_compatible
    except ImportError as e:
        raise ImportError(
            "Could not import DPTrainer"
        ) from e
    return DPTrainer, make_dp_compatible


def make_model(model_type: str, model_kwargs: Optional[dict] = None) -> nn.Module:
    kwargs = model_kwargs or {}
    if model_type == "unet":
        return UNet(in_channels=2, out_channels=1, **kwargs)
    if model_type == "modfed":
        return ModFed(**kwargs)
    if model_type == "kspace_unet":
        return KSpaceUNet(**kwargs)
    raise ValueError(f"Unknown model_type: {model_type!r}")


def get_parameters(model: nn.Module) -> NDArrays:
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: NDArrays) -> None:
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)}
    )
    model.load_state_dict(state_dict, strict=True)


class Adversary:
    def __init__(self, target_client_id: str, attack_module=None):
        self.target_client_id = target_client_id
        self.attack_module = attack_module
        self.intercepted_updates: List[dict] = []

    def intercept(self, client_id, parameters_before, parameters_after, batch_sample=None):
        if client_id != self.target_client_id:
            return
        deltas = [(b - a).copy() for b, a in zip(parameters_before, parameters_after)]
        entry = {"client_id": client_id, "weight_delta": deltas, "batch_sample": batch_sample}
        self.intercepted_updates.append(entry)
        if self.attack_module is not None:
            self.attack_module.attack(entry)

    @property
    def last_update(self):
        return self.intercepted_updates[-1] if self.intercepted_updates else None


class FedMRIClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        domain: str,
        local_epochs: int = 2,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
        adversary: Optional[Adversary] = None,
    ):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.domain = domain
        self.local_epochs = local_epochs
        self.lr = lr
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adversary = adversary
        self.loss_fn = ReconstructionLoss()
        self.model.to(self.device)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        params_before = get_parameters(self.model)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        train_loss = self._train(optimizer)
        params_after = get_parameters(self.model)
        if self.adversary is not None:
            batch_sample = next(iter(self.train_loader))
            self.adversary.intercept(self.client_id, params_before, params_after, batch_sample)
        return params_after, len(self.train_loader.dataset), {"train_loss": float(train_loss)}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, metrics = self._evaluate()
        return float(loss), len(self.val_loader.dataset), metrics

    def _forward(self, batch):
        if self.domain == "image":
            x = batch["image_input"].to(self.device)
            y = batch["image_target"].to(self.device)
            pred = self.model(x)
        elif self.domain == "kspace":
            k = batch["kspace"].to(self.device)
            y = batch["image_target"].to(self.device)
            if hasattr(self.model, "forward") and "mask" in batch:
                try:
                    mask = batch["mask"].to(self.device)
                    pred = self.model(k, mask)
                except TypeError:
                    pred = self.model(k)
            else:
                pred = self.model(k)
        else:
            raise ValueError(f"Unknown domain: {self.domain!r}")
        return pred.squeeze(1), y

    def _train(self, optimizer):
        self.model.train()
        total_loss = 0.0
        for _ in range(self.local_epochs):
            for batch in self.train_loader:
                optimizer.zero_grad()
                pred, y = self._forward(batch)
                loss = self.loss_fn(pred, y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
        return total_loss / (self.local_epochs * max(len(self.train_loader), 1))

    def _evaluate(self):
        self.model.eval()
        total_loss, all_ssim, all_psnr = 0.0, [], []
        with torch.no_grad():
            for batch in self.val_loader:
                pred, y = self._forward(batch)
                total_loss += self.loss_fn(pred, y).item()
                m = compute_metrics(pred, y)
                all_ssim.append(m["ssim"])
                all_psnr.append(m["psnr"])
        n = max(len(self.val_loader), 1)
        return total_loss / n, {"ssim": float(np.mean(all_ssim)), "psnr": float(np.mean(all_psnr))}

class DPFlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        domain: str,
        num_rounds: int,
        dp_config: dict,
        local_epochs: int = 2,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
    ):
        DPTrainer, _ = _import_dp()
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.domain = domain
        self.local_epochs = local_epochs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = ReconstructionLoss()
        # epochs = full horizon so Opacus calibrates sigma for the entire run
        self.dp_trainer = DPTrainer(
            model=model,
            train_loader=train_loader,
            target_epsilon=dp_config["target_epsilon"],
            target_delta=dp_config["target_delta"],
            max_grad_norm=dp_config["max_grad_norm"],
            lr=lr,
            epochs=num_rounds * local_epochs,
            device=self.device,
        )
        self._is_setup = False

    def get_parameters(self, config):
        return get_parameters(self.dp_trainer.get_model())

    def fit(self, parameters, config):
        set_parameters(self.dp_trainer.get_model(), parameters)
        if not self._is_setup:
            self.dp_trainer.setup()
            self._is_setup = True
            # reload incoming weights into the freshly Opacus-wrapped module
            set_parameters(self.dp_trainer.get_model(), parameters)
        losses = [self.dp_trainer.train_epoch(self.domain) for _ in range(self.local_epochs)]
        eps = self.dp_trainer.get_epsilon()  # per-round, NOT cumulative
        params_after = get_parameters(self.dp_trainer.get_model())
        n = len(self.train_loader.dataset)
        return params_after, n, {
            "train_loss": float(np.mean(losses)),
            "train_epsilon_round": float(eps),
        }

    def evaluate(self, parameters, config):
        model = self.dp_trainer.get_model()
        set_parameters(model, parameters)
        model.eval()
        total_loss, all_ssim, all_psnr = 0.0, [], []
        with torch.no_grad():
            for batch in self.val_loader:
                pred, y = self._eval_forward(model, batch)
                total_loss += self.loss_fn(pred, y).item()
                m = compute_metrics(pred, y)
                all_ssim.append(m["ssim"])
                all_psnr.append(m["psnr"])
        n = max(len(self.val_loader), 1)
        return (total_loss / n, len(self.val_loader.dataset),
                {"ssim": float(np.mean(all_ssim)), "psnr": float(np.mean(all_psnr))})

    def _eval_forward(self, model, batch):
        if self.domain == "image":
            x = batch["image_input"].to(self.device)
            y = batch["image_target"].to(self.device)
            pred = model(x)
        else:
            k = batch["kspace"].to(self.device)
            y = batch["image_target"].to(self.device)
            try:
                mask = batch["mask"].to(self.device)
                pred = model(k, mask)
            except (TypeError, KeyError):
                pred = model(k)
        return pred.squeeze(1), y

class FedAvgWithLogging(FedAvg):
    def __init__(self, *args, checkpoint_dir=None, model_type=None, model_kwargs=None,
                 dp=False, target_epsilon="inf", eval_loader=None, eval_domain=None,
                 eval_device=None, metrics_csv=None, round_offset=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_dir = checkpoint_dir
        self.model_type = model_type
        self.model_kwargs = model_kwargs or {}
        self.dp = dp
        self.target_epsilon = target_epsilon          # <-- fixes the AttributeError crash
        self.eval_loader = eval_loader
        self.eval_domain = eval_domain
        self.eval_device = eval_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics_csv = metrics_csv
        self.round_offset = round_offset               # for resume numbering
        self._latest_params = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated_params, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        round_train_eps = None
        if results:
            avg_loss = np.mean([r.metrics.get("train_loss", 0.0) for _, r in results])
            print(f"[Round {server_round}] avg train loss: {avg_loss:.4f}")
            eps_vals = [r.metrics["train_epsilon_round"] for _, r in results
                        if "train_epsilon_round" in r.metrics]
            round_train_eps = float(np.mean(eps_vals)) if eps_vals else None

        if aggregated_params is not None:
            self._latest_params = aggregated_params
            actual_round = server_round + self.round_offset

            ndarrays = parameters_to_ndarrays(aggregated_params)
            model = make_model(self.model_type, self.model_kwargs)
            if self.dp:
                _, make_dp_compatible = _import_dp()
                model = make_dp_compatible(model)
            set_parameters(model, ndarrays)

            if self.checkpoint_dir and self.model_type:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                ckpt_path = os.path.join(
                    self.checkpoint_dir,
                    f"{self.model_type}_scanner_round{actual_round:02d}_eps{self.target_epsilon}.pt"
                )
                torch.save({"model_type": self.model_type, "round": actual_round,
                            "target_epsilon": self.target_epsilon,
                            "model_state_dict": model.state_dict()}, ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")

            if self.eval_loader is not None and self.metrics_csv is not None:
                model.to(self.eval_device)
                m = evaluate_model(model, self.eval_loader, self.eval_domain,
                                   self.eval_device, compute_lpips=False)  # skip per-round LPIPS
                row = {"model": self.model_type, "target_epsilon": self.target_epsilon,
                       "round": actual_round, "round_train_epsilon": round_train_eps,
                       "loss": m["loss"], "ssim": m["ssim"], "psnr": m["psnr"], "nmse": m["nmse"]}
                os.makedirs(os.path.dirname(self.metrics_csv), exist_ok=True)
                import pandas as pd
                pd.DataFrame([row]).to_csv(self.metrics_csv, mode="a", index=False,
                                           header=not os.path.exists(self.metrics_csv))
                print(f"  [eval r{actual_round:02d} eps{self.target_epsilon}] "
                      f"SSIM={m['ssim']:.4f} PSNR={m['psnr']:.2f} NMSE={m['nmse']:.6f}")

        return aggregated_params, aggregated_metrics

def run_simulation(
    model_type: str,
    train_loaders: Dict[str, DataLoader],
    val_loader: DataLoader,
    num_rounds: int = 20,
    local_epochs: int = 2,
    lr: float = 1e-3,
    adversary: Optional[Adversary] = None,
    device: Optional[torch.device] = None,
    model_kwargs: Optional[dict] = None,
    checkpoint_dir: Optional[str] = None,
    resume_round: Optional[int] = None,
    dp_config: Optional[dict] = None,
    eval_loader = None,
    metrics_csv = None,
) -> Tuple[nn.Module, object]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain = MODEL_DOMAINS[model_type]
    model_kwargs = model_kwargs or {}
    client_ids = list(train_loaders.keys())
    num_clients = len(train_loaders)
    global_model = make_model(model_type, model_kwargs)
    if dp_config is not None:
        _, make_dp_compatible = _import_dp()
        global_model = make_dp_compatible(global_model)
    # if resume_round is not None and checkpoint_dir is not None:
    #     ckpt_path = os.path.join(
    #         checkpoint_dir,
    #         f"{model_type}_scanner_round{resume_round:02d}.pt"
    #     )
    #     if os.path.exists(ckpt_path):
    #         ckpt = torch.load(ckpt_path, map_location="cpu")
    #         global_model.load_state_dict(ckpt["model_state_dict"])
    #         print(f"Resumed from round {resume_round}: {ckpt_path}")
    #     else:
    #         print(f"WARNING: checkpoint not found at {ckpt_path}, starting from scratch")
    tag = dp_config["target_epsilon"] if dp_config else "inf"

    # auto-resume: find the latest per-round checkpoint for this (model, eps)
    start_round = 0
    if checkpoint_dir and os.path.isdir(checkpoint_dir):
        import glob, re as _re
        found = []
        for p in glob.glob(os.path.join(checkpoint_dir, f"{model_type}_scanner_round*_eps{tag}.pt")):
            mt = _re.search(r"round(\d+)_eps", os.path.basename(p))
            if mt:
                found.append((int(mt.group(1)), p))
        if found:
            start_round, latest = max(found, key=lambda t: t[0])
            global_model.load_state_dict(torch.load(latest, map_location="cpu")["model_state_dict"])
            print(f"Resuming {model_type} eps{tag} from round {start_round}")

    remaining = num_rounds - start_round
    if remaining <= 0:
        print(f"{model_type} eps{tag}: already complete ({start_round}/{num_rounds}).")
        return global_model, None

    def client_fn(cid: str) -> fl.client.NumPyClient:
        idx = client_ids[int(cid)]
        if dp_config is not None:
            return DPFlowerClient(
                client_id=idx,
                model=make_model(model_type, model_kwargs),
                train_loader=train_loaders[idx],
                val_loader=val_loader,
                domain=domain,
                num_rounds=num_rounds,
                dp_config=dp_config,
                local_epochs=local_epochs,
                lr=lr,
                device=device,
            )
        return FedMRIClient(
            client_id=idx,
            model=make_model(model_type, model_kwargs),
            train_loader=train_loaders[idx],
            val_loader=val_loader,
            domain=domain,
            local_epochs=local_epochs,
            lr=lr,
            device=device,
            adversary=adversary if (adversary and adversary.target_client_id == cid) else None,
        )

    strategy = FedAvgWithLogging(
        min_fit_clients=num_clients, fraction_evaluate=0.0, min_evaluate_clients=0,
        min_available_clients=num_clients,
        initial_parameters=ndarrays_to_parameters(get_parameters(global_model)),
        checkpoint_dir=checkpoint_dir, model_type=model_type, model_kwargs=model_kwargs,
        dp=(dp_config is not None), target_epsilon=tag,
        eval_loader=eval_loader, eval_domain=domain, eval_device=device,
        metrics_csv=metrics_csv, round_offset=start_round,
    )
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=remaining),
        strategy=strategy,
        client_resources={"num_gpus": 1.0, "num_cpus": 1},
    )

    if strategy._latest_params is not None:
        final_ndarrays = parameters_to_ndarrays(strategy._latest_params)
        set_parameters(global_model, final_ndarrays)
        print("Final weights applied to global model.")

    return global_model, history
