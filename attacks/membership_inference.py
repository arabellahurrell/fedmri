import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Optional, Tuple, Dict
from tqdm import tqdm

@torch.no_grad()
def compute_per_sample_loss(
    model: nn.Module,
    loader: DataLoader,
    domain: str,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
) -> np.ndarray:
    if loss_fn is None:
        from models.unet import ReconstructionLoss
        loss_fn = ReconstructionLoss()

    model.eval()
    all_losses = []

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

        # Compute per-sample (not reduced)
        for i in range(pred.shape[0]):
            loss_i = loss_fn(pred[i:i+1], y[i:i+1])
            all_losses.append(loss_i.item())

    return np.array(all_losses)


class LossThresholdMIA:
    def __init__(self, threshold: Optional[float] = None):
        self.threshold = threshold

    def calibrate(
        self,
        member_losses: np.ndarray,
        nonmember_losses: np.ndarray,
    ) -> float:
        all_losses = np.concatenate([member_losses, nonmember_losses])
        labels = np.concatenate([
            np.ones(len(member_losses)),
            np.zeros(len(nonmember_losses))
        ])

        best_acc = 0.0
        best_t = np.median(all_losses)

        for t in np.percentile(all_losses, np.arange(5, 96, 1)):
            preds = (all_losses < t).astype(int)
            acc = accuracy_score(labels, preds)
            if acc > best_acc:
                best_acc = acc
                best_t = t

        self.threshold = best_t
        return best_t

    def predict(self, losses: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise ValueError("Call calibrate() first or set threshold manually.")
        return (losses < self.threshold).astype(int)

    def _metrics(self, losses: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        preds = self.predict(losses)
        auc = roc_auc_score(labels, -losses)
        acc = accuracy_score(labels, preds)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        tpr = tp / max(np.sum(labels == 1), 1)
        fpr = fp / max(np.sum(labels == 0), 1)
        advantage = abs(tpr - fpr)   # privacy advantage

        return {
            "auc": float(auc),
            "accuracy": float(acc),
            "tpr": float(tpr),
            "fpr": float(fpr),
            "advantage": float(advantage),
        }
    
    def evaluate(
        self,
        model: nn.Module,
        member_loader: DataLoader,
        nonmember_loader: DataLoader,
        domain: str,
        device: torch.device,
        calibration_split: float = 0.5,
        seed: int = 42,
    ) -> Dict[str, float]:
        member_losses = compute_per_sample_loss(model, member_loader, domain, device)
        nonmember_losses = compute_per_sample_loss(model, nonmember_loader, domain, device)

        rng = np.random.RandomState(seed)

        def split(arr: np.ndarray):
            idx = rng.permutation(len(arr))
            k = int(round(len(arr) * calibration_split))
            k = min(max(k, 1), len(arr) - 1) if len(arr) >= 2 else len(arr)
            return arr[idx[:k]], arr[idx[k:]]

        mem_cal, mem_eval = split(member_losses)
        non_cal, non_eval = split(nonmember_losses)

        if len(mem_eval) == 0 or len(non_eval) == 0:
            mem_cal, mem_eval = member_losses, member_losses
            non_cal, non_eval = nonmember_losses, nonmember_losses

        self.calibrate(mem_cal, non_cal)

        eval_losses = np.concatenate([mem_eval, non_eval])
        eval_labels = np.concatenate([
            np.ones(len(mem_eval)),
            np.zeros(len(non_eval)),
        ])

        out = self._metrics(eval_losses, eval_labels)
        out.update({
            "threshold": float(self.threshold),
            "n_calibration": int(len(mem_cal) + len(non_cal)),
            "n_eval": int(len(eval_labels)),
            "member_loss_mean": float(member_losses.mean()),
            "nonmember_loss_mean": float(nonmember_losses.mean()),
        })
        return out


class ShadowModelMIA:
    def __init__(
        self,
        model_fn,           # callable: () -> nn.Module
        num_shadow: int = 4,
        shadow_epochs: int = 5,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
    ):
        self.model_fn = model_fn
        self.num_shadow = num_shadow
        self.shadow_epochs = shadow_epochs
        self.lr = lr
        self.device = device or torch.device("cpu")
        self.meta_clf = LogisticRegression(max_iter=1000)
        self.shadow_models = []

    def _train_shadow(self, train_loader: DataLoader, domain: str) -> nn.Module:
        from models.unet import ReconstructionLoss
        model = self.model_fn().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = ReconstructionLoss()

        model.train()
        for _ in range(self.shadow_epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                if domain == "image":
                    x = batch["image_input"].to(self.device)
                    y = batch["image_target"].to(self.device)
                    loss = loss_fn(model(x).squeeze(1), y)
                else:
                    k = batch["kspace"].to(self.device)
                    mask = batch["mask"].to(self.device)
                    y = batch["image_target"].to(self.device)
                    loss = loss_fn(model(k, mask).squeeze(1), y)
                loss.backward()
                optimizer.step()
        return model

    def fit(
        self,
        shadow_train_loaders: list,     # one DataLoader per shadow model (member set)
        shadow_out_loaders: list,       # one DataLoader per shadow model (non-member set)
        domain: str,
    ) -> None:
        features = []
        labels = []

        for i, (tr_loader, out_loader) in enumerate(
            zip(shadow_train_loaders, shadow_out_loaders)
        ):
            print(f"  Training shadow model {i+1}/{self.num_shadow}...")
            shadow = self._train_shadow(tr_loader, domain)
            self.shadow_models.append(shadow)

            member_losses = compute_per_sample_loss(shadow, tr_loader, domain, self.device)
            nonmember_losses = compute_per_sample_loss(shadow, out_loader, domain, self.device)

            features.append(member_losses.reshape(-1, 1))
            labels.append(np.ones(len(member_losses)))
            features.append(nonmember_losses.reshape(-1, 1))
            labels.append(np.zeros(len(nonmember_losses)))

        X = np.vstack(features)
        y = np.concatenate(labels)
        self.meta_clf.fit(X, y)

    def evaluate(
        self,
        target_model: nn.Module,
        member_loader: DataLoader,
        nonmember_loader: DataLoader,
        domain: str,
    ) -> Dict[str, float]:
        member_losses = compute_per_sample_loss(
            target_model, member_loader, domain, self.device
        )
        nonmember_losses = compute_per_sample_loss(
            target_model, nonmember_loader, domain, self.device
        )

        all_losses = np.concatenate([member_losses, nonmember_losses]).reshape(-1, 1)
        true_labels = np.concatenate([
            np.ones(len(member_losses)),
            np.zeros(len(nonmember_losses)),
        ])

        preds = self.meta_clf.predict(all_losses)
        scores = self.meta_clf.predict_proba(all_losses)[:, 1]

        return {
            "auc": float(roc_auc_score(true_labels, scores)),
            "accuracy": float(accuracy_score(true_labels, preds)),
            "advantage": float(abs(
                np.mean(preds[true_labels == 1]) - np.mean(preds[true_labels == 0])
            )),
        }
