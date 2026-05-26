import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np

from data.fastmri_dataset import get_client_dataloaders
from federated.fl_simulation import run_simulation, Adversary, MODEL_DOMAINS
from evaluation.metrics import evaluate_model, ResultsTracker


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",
                   choices=["unet", "modfed"],
                   default="unet")
    p.add_argument("--partition",  choices=["scanner", "acquisition", "iid"], default="scanner")
    p.add_argument("--data_root",  default="data/fastmri")
    p.add_argument("--num_rounds", type=int, default=20)
    p.add_argument("--local_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--acceleration", type=int, default=4)
    p.add_argument("--save_dir",   default="checkpoints/federated")
    p.add_argument("--results_dir", default="results/federated")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else (
        "mps"  if torch.backends.mps.is_available() else "cpu")
    )
    pin_memory = device.type == "cuda"
    domain = MODEL_DOMAINS[args.model]
    os.makedirs(args.save_dir, exist_ok=True)

    train_loaders, val_loader = get_client_dataloaders(
        root=args.data_root,
        domain=domain,
        acceleration=args.acceleration,
        batch_size=args.batch_size,
        partition=args.partition,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        seed=args.seed,
        cache_dir=args.data_root,
    )

    adversary = Adversary(target_client_id="0")

    global_model, history = run_simulation(
        model_type=args.model,
        train_loaders=train_loaders,
        val_loader=val_loader,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        adversary=adversary,
        device=device,
    )

    ckpt_path = f"{args.save_dir}/{args.model}_{args.partition}.pt"
    torch.save({
        "model_type": args.model,
        "domain": domain,
        "partition": args.partition,
        "model_state_dict": global_model.state_dict(),
        "args": vars(args),
    }, ckpt_path)
    print(f"Checkpoint: {ckpt_path}")

    global_model = global_model.to(device)
    val_metrics = evaluate_model(global_model, val_loader, domain, device)

    print(f"Final — SSIM: {val_metrics['ssim']:.4f} | PSNR: {val_metrics['psnr']:.2f} dB")
    
    tracker = ResultsTracker(save_dir=args.results_dir)
    tracker.log(
        model=args.model,
        domain=domain,
        partition=args.partition,
        num_rounds=args.num_rounds,
        **{f"val_{k}": v for k, v in val_metrics.items()},
    )
    tracker.save_csv(f"federated_{args.model}_{args.partition}.csv")


if __name__ == "__main__":
    main()
