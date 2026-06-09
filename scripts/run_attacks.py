import argparse
import json
import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data.fastmri_dataset import FastMRISliceDataset
from torch.utils.data import DataLoader, Subset
from models.unet import UNet, ReconstructionLoss
from models.kspace_unet import KSpaceUNet
from models.modfed import ModFed
from attacks.gradient_inversion import TVGradientInversion, GradientInversionAttack
from attacks.membership_inference import LossThresholdMIA
from evaluation.metrics import compute_metrics, ResultsTracker
from federated.fl_simulation import MODEL_DOMAINS


MODEL_CONSTRUCTORS = {
    "unet":        lambda: UNet(in_channels=2, out_channels=1, base_features=32, depth=4),
    "kspace_unet": lambda: KSpaceUNet(base_features=32, depth=4),
    "modfed":      lambda: ModFed(num_cascades=6, kspace_ch=64, kspace_layers=5),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--data_root",    default="data/fastmri")
    p.add_argument("--acceleration", type=int, default=4)
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--num_workers",  type=int, default=0)
    p.add_argument("--results_dir",  default="results/attacks")
    p.add_argument("--use_breaching", action="store_true")
    p.add_argument("--gi_batches",   type=int, default=10)
    p.add_argument("--gi_iters",     type=int, default=2000)
    p.add_argument("--gi_restarts",  type=int, default=3)
    p.add_argument("--mia_samples",  type=int, default=500)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--attacks", nargs="+", choices=["gia", "mia"],
               default=["gia", "mia"])
    return p.parse_args()


def simulate_client_gradients(model, batch, domain, device):
    import copy
    model_copy = copy.deepcopy(model)
    # model_copy.load_state_dict(model.state_dict())
    model_copy.to(device).train()

    loss_fn = ReconstructionLoss()
    model_copy.zero_grad(set_to_none=True)

    if domain == "image":
        pred = model_copy(batch["image_input"].to(device)).squeeze(1)
        y = batch["image_target"].to(device)
    else:
        k = batch["kspace"].to(device)
        y = batch["image_target"].to(device)
        try:
            mask = batch["mask"].to(device)
            pred = model_copy(k, mask).squeeze(1)
        except TypeError:
            pred = model_copy(k).squeeze(1)

    loss_fn(pred, y).backward()
    grads = []
    for p in model_copy.parameters():
        if p.grad is None:
            grads.append(np.zeros(tuple(p.shape), dtype=np.float32))
        else:
            grads.append(p.grad.detach().cpu().numpy())
    return grads

def recon_to_display(rc, domain):
    import fastmri
    r = rc if rc.dim() == 4 else rc.unsqueeze(0)        # (1, C, H, W)
    if r.shape[1] >= 2:
        if domain == "kspace":
            k = r[:, :2].permute(0, 2, 3, 1).contiguous().unsqueeze(1)  # (1,1,H,W,2)
            mag = fastmri.complex_abs(fastmri.ifft2c(k)).squeeze(1)     # (1,H,W)
        else:
            mag = torch.sqrt(r[:, 0] ** 2 + r[:, 1] ** 2)              # (1,H,W)
        return mag[0].numpy()
    return r.squeeze().numpy()

def run_gia(model, model_type, domain, train_ds, args, device, results_dir):
    loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))

    if args.use_breaching:
        attacker = GradientInversionAttack(model, domain=domain, device=device)
    else:
        attacker = TVGradientInversion(
            model=model, domain=domain, device=device,
            num_iters=args.gi_iters, restarts=args.gi_restarts,
        )
    progress_path = os.path.join(results_dir, f"gia_progress_{model_type}_eps{args.target_epsilon}.jsonl")
    gallery_dir = os.path.join(results_dir, f"gia_gallery_cache_{model_type}_eps{args.target_epsilon}")
    os.makedirs(gallery_dir, exist_ok=True)
    done = {}
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done[int(rec["batch_idx"])] = rec
        if done:
            print(f"  [resume] {model_type}: {len(done)} batches already done, skipping them")

    all_ssim, all_psnr, all_lpips, all_loss = [], [], [], []
    for idx in sorted(done):
        rec = done[idx]
        all_ssim.append(rec.get("ssim", float("nan")))
        all_psnr.append(rec.get("psnr", float("nan")))
        all_lpips.append(rec.get("lpips", float("nan")))
        all_loss.append(rec.get("final_loss", float("nan")))

    from tqdm import tqdm
    pbar = (tqdm(total=args.gi_batches, desc=f"GIA [{model_type}]", initial=len(done)) if args.use_breaching else None)

    for i, batch in enumerate(loader):
        if i >= args.gi_batches:
            break
        if i in done:
            continue
        grads = simulate_client_gradients(model, batch, domain, device)
        input_key = "image_input" if domain == "image" else "kspace"
        if args.use_breaching:
            recon, m = attacker.run(
                true_gradients=grads,
                ground_truth_batch=batch,
                batch_size=1,
            )
            pbar.update(1)
        else:
            print(f"\nbatch {i+1}/{args.gi_batches}")
            recon, m = attacker.run(
                true_gradients=grads,
                input_shape=tuple(batch[input_key].shape),
                ground_truth=batch[input_key],
                mask=batch.get("mask"),
            )
        if m:
            all_ssim.append(m.get("ssim", float("nan")))
            all_psnr.append(m.get("psnr", float("nan")))
            all_lpips.append(m.get("lpips", float("nan")))
            all_loss.append(m.get("final_loss", float("nan")))
        if i < 3:
            np.savez(
                os.path.join(gallery_dir, f"batch_{i:03d}.npz"),
                gt=batch["image_target"][0].cpu().numpy(),
                recon=recon[0].detach().cpu().numpy(),
                ssim=np.float32(m.get("ssim", float("nan")) if m else float("nan")),
            )
        with open(progress_path, "a") as f:
            f.write(json.dumps({"batch_idx": i, **(m or {})}) + "\n")
            f.flush()
            os.fsync(f.fileno())

    if pbar is not None:
        pbar.close()

    gallery_files = sorted(
        p for p in os.listdir(gallery_dir)
        if p.startswith("batch_") and p.endswith(".npz")
    )[:3]
    if gallery_files:
        fig, axes = plt.subplots(len(gallery_files), 2, figsize=(6, 3 * len(gallery_files)))
        if len(gallery_files) == 1:
            axes = [axes]
        for row, fname in enumerate(gallery_files):
            data = np.load(os.path.join(gallery_dir, fname))
            gt = data["gt"]
            rc = torch.from_numpy(data["recon"])
            ssim_val = float(data["ssim"])
            r_img = recon_to_display(rc, domain)
            axes[row][0].imshow(gt, cmap="gray"); axes[row][0].set_title("GT"); axes[row][0].axis("off")
            ssim_s = f"{ssim_val:.3f}" if not np.isnan(ssim_val) else "N/A"
            axes[row][1].imshow(r_img, cmap="gray"); axes[row][1].set_title(f"GIA SSIM={ssim_s}"); axes[row][1].axis("off")
        plt.suptitle(f"{model_type} ({domain}) — GIA Gallery")
        plt.tight_layout()
        fig.savefig(f"{results_dir}/gia_gallery_{model_type}_eps{args.target_epsilon}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    nanmean = lambda x: float(np.nanmean(x)) if x else float("nan")
    nanstd  = lambda x: float(np.nanstd(x))  if x else float("nan")
    return {
        "ssim_mean": nanmean(all_ssim), "ssim_std": nanstd(all_ssim),
        "psnr_mean": nanmean(all_psnr), "psnr_std": nanstd(all_psnr),
        "lpips_mean": nanmean(all_lpips), "lpips_std": nanstd(all_lpips),
        "final_loss_mean": nanmean(all_loss), "loss_std": nanstd(all_loss),
        "n_batches": len(all_ssim),
    }


def run_mia(model, domain, train_ds, val_ds, args, device):
    n = min(args.mia_samples, len(train_ds), len(val_ds))
    member_loader    = DataLoader(Subset(train_ds, list(range(n))), batch_size=1)
    nonmember_loader = DataLoader(Subset(val_ds,   list(range(n))), batch_size=1)
    return LossThresholdMIA().evaluate(model, member_loader, nonmember_loader, domain, device, seed=args.seed)


def plot_comparison(all_results, results_dir):
    models   = [r["model_type"] for r in all_results]
    labels   = {"unet": "UNet\n(image)", "kspace_unet": "KSpaceUNet\n(k-space)", "modfed": "ModFed\n(k-space)"}
    x_labels = [labels.get(m, m) for m in models]
    colors   = ["#4C72B0", "#DD8452", "#55A868"][:len(models)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ssims = [r["gia"]["ssim_mean"] for r in all_results]
    axes[0].bar(x_labels, ssims, color=colors); axes[0].set_title("GIA — SSIM ↓"); axes[0].grid(axis="y", alpha=0.3)
    lpips_ = [r["gia"]["lpips_mean"] for r in all_results]
    axes[1].bar(x_labels, lpips_, color=colors); axes[1].set_title("GIA — LPIPS ↑"); axes[1].grid(axis="y", alpha=0.3)
    aucs = [r["mia"].get("auc", 0) for r in all_results]
    axes[2].bar(x_labels, aucs, color=colors)
    axes[2].axhline(0.5, color="red", linestyle="--")
    axes[2].set_title("MIA — AUC ↓"); axes[2].set_ylim(0, 1); axes[2].grid(axis="y", alpha=0.3)

    plt.suptitle("Privacy Attack Comparison"); plt.tight_layout()
    fig.savefig(f"{results_dir}/attack_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else (
        "mps"  if torch.backends.mps.is_available() else "cpu")
    )

    tracker = ResultsTracker(save_dir=args.results_dir)
    all_results = []

    for ckpt_path in args.checkpoints:
        print(f"\n{'='*65}\n  {os.path.basename(ckpt_path)}\n{'='*65}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model_type = ckpt["model_type"]
        _stem = os.path.splitext(os.path.basename(ckpt_path))[0]
        _m = re.search(r"eps([0-9p]+)", _stem)
        args.target_epsilon = _m.group(1) if _m else "NA"
        model = MODEL_CONSTRUCTORS[model_type]()
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval().to(device)
        domain = ckpt.get("domain", MODEL_DOMAINS[model_type])

        train_ds = FastMRISliceDataset(root=args.data_root, domain=domain, split="train",
                                       acceleration=args.acceleration, seed=args.seed, cache_dir=args.data_root,)
        val_ds = None
        if "mia" in args.attacks:
            val_ds = FastMRISliceDataset(root=args.data_root, domain=domain, split="val",
                                         acceleration=args.acceleration, seed=args.seed, cache_dir=args.data_root,)
        entry = {"model_type": model_type, "domain": domain, "checkpoint": os.path.basename(ckpt_path)}
        gia_metrics = mia_metrics = None
        # val_ds   = FastMRISliceDataset(root=args.data_root, domain=domain, split="val",
        #                                acceleration=args.acceleration, seed=args.seed, cache_dir=args.data_root,)

        # gia_metrics = run_gia(model, model_type, domain, train_ds, args, device, args.results_dir)
        # mia_metrics = run_mia(model, domain, train_ds, val_ds, args, device)

        if "gia" in args.attacks:
            gia_metrics = run_gia(model, model_type, domain, train_ds, args, device, args.results_dir)
            entry.update({f"gia_{k}": v for k, v in gia_metrics.items()})
        if "mia" in args.attacks:
            mia_metrics = run_mia(model, domain, train_ds, val_ds, args, device)
            entry.update({f"mia_{k}": v for k, v in mia_metrics.items()})
        tracker.log(**entry)
        all_results.append({"model_type": model_type, "gia": gia_metrics, "mia": mia_metrics})
        tracker.save_csv(f"attack_results_{'_'.join(args.attacks)}.csv")

        # entry = {"model_type": model_type, "domain": domain, "checkpoint": os.path.basename(ckpt_path)}
        # entry.update({f"gia_{k}": v for k, v in gia_metrics.items()})
        # entry.update({f"mia_{k}": v for k, v in mia_metrics.items()})
        # tracker.log(**entry)
        # all_results.append({"model_type": model_type, "gia": gia_metrics, "mia": mia_metrics})
        # tracker.save_csv("attack_results.csv")

    tracker.save_csv(f"attack_results_{'_'.join(args.attacks)}.csv")
    # if len(all_results) > 1:
    #     plot_comparison(all_results, args.results_dir)
    if len(all_results) > 1 and all(r["gia"] and r["mia"] for r in all_results):
        plot_comparison(all_results, args.results_dir)

    print("\nAttack benchmarking complete.")

if __name__ == "__main__":
    main()
