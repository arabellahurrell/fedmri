import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset

from data.fastmri_dataset import FastMRISliceDataset, partition_by_scanner, pad_to_max
from evaluation.metrics import evaluate_model
from models.unet import UNet
from models.kspace_unet import KSpaceUNet
from models.modfed import ModFed
from federated.fl_simulation import MODEL_DOMAINS

MODEL_CONSTRUCTORS = {
    "unet":        lambda: UNet(in_channels=2, out_channels=1, base_features=32, depth=4),
    "kspace_unet": lambda: KSpaceUNet(base_features=32, depth=4),
    "modfed":      lambda: ModFed(num_cascades=6, kspace_ch=64, kspace_layers=5),
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--data_root", default="data/fastmri")
    p.add_argument("--acceleration", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--no_lpips", action="store_true")
    p.add_argument("--out", default="results/per_client_eval.csv")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    rows, val_cache = [], {}

    for ckpt_path in args.checkpoints:
        ckpt = torch.load(ckpt_path, map_location=device)
        model_type = ckpt["model_type"]
        domain = ckpt.get("domain", MODEL_DOMAINS[model_type])
        model = MODEL_CONSTRUCTORS[model_type]()
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval().to(device)

        if domain not in val_cache:
            val_ds = FastMRISliceDataset(root=args.data_root, domain=domain, split="val",
                                         acceleration=args.acceleration, seed=args.seed,
                                         cache_dir=args.data_root)
            val_cache[domain] = (val_ds, partition_by_scanner(val_ds))
        val_ds, groups = val_cache[domain]

        print(f"\n=== {os.path.basename(ckpt_path)}  ({model_type}, {domain}) ===")
        for client in sorted(groups):
            idxs = groups[client]
            loader = DataLoader(Subset(val_ds, idxs), batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                collate_fn=pad_to_max)
            m = evaluate_model(model, loader, domain, device, compute_lpips=not args.no_lpips)
            print(f"  {client:12s} n={len(idxs):5d}  SSIM={m['ssim']:.4f}  "
                  f"PSNR={m['psnr']:.2f}  NMSE={m['nmse']:.4f}")
            rows.append({"checkpoint": os.path.basename(ckpt_path), "model": model_type,
                         "domain": domain, "client": client, "n_slices": len(idxs), **m})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nSaved {args.out}")

if __name__ == "__main__":
    main()