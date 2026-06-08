import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import json
import math

from data.fastmri_dataset import get_client_dataloaders
from federated.fl_simulation import run_simulation, Adversary, MODEL_DOMAINS
from evaluation.metrics import evaluate_model, ResultsTracker

try:
    from privacy.dp_training import EPSILON_GRID
except Exception:
    EPSILON_GRID = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, float("inf")]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", nargs="+",
                   choices=["unet", "modfed", "kspace_unet"],
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
    p.add_argument("--resume_round", type=int, default=None)
    p.add_argument("--dp", action="store_true",
                   help="enable DP-SGD training (Opacus) for the clients")
    p.add_argument("--epsilon_sweep", action="store_true")
    p.add_argument("--epsilon", type=float, default=5.0)
    p.add_argument("--epsilons", nargs="+", type=float, default=None)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    return p.parse_args()

def eps_tag(eps: float) -> str:
    return "inf" if math.isinf(eps) else f"{eps:g}"


def make_dp_config(eps: float, args) -> dict:
    if math.isinf(eps):
        return None
    return {
        "target_epsilon": eps,
        "target_delta": args.delta,
        "max_grad_norm": args.max_grad_norm,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.epsilon_sweep or args.epsilons is not None:
        args.dp = True

    device = torch.device(
        "cuda" if torch.cuda.is_available() else (
        "mps"  if torch.backends.mps.is_available() else "cpu")
    )
    pin_memory = device.type == "cuda"
    # domain = MODEL_DOMAINS[args.model]
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    if args.epsilons is not None:
        epsilons = list(args.epsilons)
    elif args.epsilon_sweep:
        epsilons = list(EPSILON_GRID)
    elif args.dp:
        epsilons = [args.epsilon]
    else:
        epsilons = [float("inf")]
    
    n_points = len(args.model) * len(epsilons)
    print(f"DP sweep: {len(args.model)} model(s) x {len(epsilons)} epsilon(s) "
          f"= {n_points} federated runs of {args.num_rounds} rounds each.")
    print(f"  models   : {args.model}")
    print(f"  epsilons : {[eps_tag(e) for e in epsilons]}")
    print(f"  partition: {args.partition}\n")

    progress_path = os.path.join(args.results_dir, f"dp_sweep_progress_{args.partition}.jsonl")
    tracker = ResultsTracker(save_dir=args.results_dir)

    for model_type in args.model:
        domain = MODEL_DOMAINS[model_type]

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

        from torch.utils.data import DataLoader, Subset
        from data.fastmri_dataset import FastMRISliceDataset
        eval_ds = FastMRISliceDataset(root=args.data_root, domain=domain, split="val",
                                    acceleration=args.acceleration, seed=args.seed,
                                    cache_dir=args.data_root)
        # subset keeps per-round eval fast; final full-set eval still happens below
        sub_idx = np.random.RandomState(args.seed).choice(len(eval_ds),
                    size=min(100, len(eval_ds)), replace=False)
        eval_loader = DataLoader(Subset(eval_ds, sub_idx), batch_size=1, shuffle=False,
                                num_workers=args.num_workers)
        full_eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
        metrics_csv = os.path.join(args.results_dir, f"per_round_metrics_{args.partition}.csv")

        for eps in epsilons:
            tag = eps_tag(eps)
            ckpt_path = f"{args.save_dir}/{model_type}_{args.partition}_eps{tag}.pt"
            print(f"\n{'='*65}\n  {model_type} | {args.partition} | eps={tag}\n{'='*65}")

            # ----- resume: a finished point already has its checkpoint -----
            if os.path.exists(ckpt_path):
                print(f"  [skip] checkpoint exists -> evaluating from disk")
                ckpt = torch.load(ckpt_path, map_location=device)
                model = _load_model(model_type, ckpt, device)
            else:
                dp_config = make_dp_config(eps, args)
                adversary = Adversary(target_client_id="0")
                global_model, history = run_simulation(
                    model_type=model_type,
                    train_loaders=train_loaders,
                    val_loader=val_loader,
                    num_rounds=args.num_rounds,
                    local_epochs=args.local_epochs,
                    lr=args.lr,
                    adversary=adversary,
                    device=device,
                    checkpoint_dir=args.save_dir,
                    resume_round=args.resume_round,
                    dp_config=dp_config,
                    eval_loader=eval_loader,
                    metrics_csv=metrics_csv,
                )
                model = global_model.to(device)
                achieved_eps = None
                if isinstance(history, dict):
                    achieved_eps = history.get("epsilon") or history.get("final_epsilon")
                torch.save({
                    "model_type": model_type,
                    "domain": domain,
                    "partition": args.partition,
                    "target_epsilon": tag,
                    "achieved_epsilon": achieved_eps,
                    "dp": dp_config is not None,
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                }, ckpt_path)
                print(f"  checkpoint: {ckpt_path}")
            val_metrics = evaluate_model(model, full_eval_loader, domain, device)
            print(f"  eps={tag} -> SSIM {val_metrics['ssim']:.4f} | "
                  f"PSNR {val_metrics['psnr']:.2f} dB")

            row = {
                "model": model_type,
                "domain": domain,
                "partition": args.partition,
                "dp": (not math.isinf(eps)),
                "target_epsilon": tag,
                "num_rounds": args.num_rounds,
                "local_epochs": args.local_epochs,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            with open(progress_path, "a") as f:
                f.write(json.dumps(row) + "\n")
                f.flush()
                os.fsync(f.fileno())
            tracker.log(**row)
            tracker.save_csv(f"dp_sweep_{args.partition}.csv")

    print("\nDP sweep complete.")

def _load_model(model_type, ckpt, device):
    from federated.fl_simulation import make_model
    model = make_model(model_type)
    if ckpt.get("dp"):
        from federated.fl_simulation import _import_dp
        _, make_dp_compatible = _import_dp()
        model = make_dp_compatible(model)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval().to(device)

if __name__ == "__main__":
    main()
