import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.kspace_unet import KSpaceUNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from data.fastmri_dataset import FastMRISliceDataset
from models.unet import UNet, ReconstructionLoss
from models.modfed import ModFed
from evaluation.metrics import evaluate_model, ResultsTracker

def pad_to_max(batch):
    """Pad spatial dims to the largest H×W in the batch, then stack."""
    keys = batch[0].keys()
    result = {}
    for key in keys:
        vals = [item[key] for item in batch]
        if not isinstance(vals[0], torch.Tensor):
            result[key] = vals
            continue
        if vals[0].dim() >= 2:
            max_h = max(v.shape[-2] for v in vals)
            max_w = max(v.shape[-1] for v in vals)
            padded = []
            for v in vals:
                dh = max_h - v.shape[-2]
                dw = max_w - v.shape[-1]
                v = F.pad(v, (0, dw, 0, dh))
                padded.append(v)
            result[key] = torch.stack(padded, dim=0)
        else:
            result[key] = torch.stack(vals, dim=0)
    return result

def parse_args():
    parser = argparse.ArgumentParser(description="Baseline MRI reconstruction training")
    parser.add_argument("--model",       choices=["unet", "modfed", "kspace_unet"], default="unet")
    parser.add_argument("--domain",      choices=["image", "kspace"], default="image")
    parser.add_argument("--data_root",   default="data/fastmri")
    parser.add_argument("--epochs",      type=int, default=20)
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--acceleration",type=int, default=4)
    parser.add_argument("--save_dir",    default="checkpoints/baseline")
    parser.add_argument("--results_dir", default="results/baseline")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed",        type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    ))
    print(f"Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    train_ds = FastMRISliceDataset(
        root=args.data_root,
        domain=args.domain,
        split="train",
        acceleration=args.acceleration,
        seed=args.seed,,
        cache_dir=args.data_root,
    )
    val_ds = FastMRISliceDataset(
        root=args.data_root,
        domain=args.domain,
        split="val",
        acceleration=args.acceleration,
        seed=args.seed,
        cache_dir=args.data_root,
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory, collate_fn=pad_to_max,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory, collate_fn=pad_to_max,
    )
    print(f"Train slices: {len(train_ds)} | Val slices: {len(val_ds)}")

    if args.model == "unet":
        model = UNet(in_channels=2, out_channels=1, base_features=32, depth=4)
    elif args.model == "kspace_unet":
        model = KSpaceUNet(base_features=32, depth=4)
    else:
        model = ModFed(num_cascades=6, kspace_ch=64, kspace_layers=5)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = ReconstructionLoss()

    best_ssim = 0.0

    resume_path = f"{args.save_dir}/{args.model}_{args.domain}_latest.pt"
    start_epoch = 1
    if os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_ssim = ckpt["best_ssim"]
        start_epoch = ckpt["epoch"] + 1
        print(f"  Resumed from epoch {ckpt['epoch']} (best SSIM={best_ssim:.4f})")

    tracker = ResultsTracker(save_dir=args.results_dir)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            optimizer.zero_grad()

            if args.domain == "image":
                x = batch["image_input"].to(device)
                y = batch["image_target"].to(device)
                pred = model(x).squeeze(1)
            else:
                k = batch["kspace"].to(device)
                mask = batch["mask"].to(device)
                y = batch["image_target"].to(device)
                pred = model(k, mask).squeeze(1)

            loss = loss_fn(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        val_metrics = evaluate_model(model, val_loader, args.domain, device)
        print(
            f"Epoch {epoch:3d} | train_loss: {train_loss:.4f} | "
            f"val_ssim: {val_metrics['ssim']:.4f} | val_psnr: {val_metrics['psnr']:.2f} dB"
        )

        tracker.log(
            epoch=epoch,
            model=args.model,
            domain=args.domain,
            train_loss=train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        )

        if val_metrics["ssim"] > best_ssim:
            best_ssim = val_metrics["ssim"]
            ckpt_path = f"{args.save_dir}/{args.model}_{args.domain}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args),
            }, ckpt_path)
            print(f"  Saved best checkpoint (SSIM={best_ssim:.4f})")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_ssim": best_ssim,
            "val_metrics": val_metrics,
            "args": vars(args),
        }, f"{args.save_dir}/{args.model}_{args.domain}_latest.pt")

    tracker.save_csv(f"baseline_{args.model}_{args.domain}.csv")
    tracker.plot_training_curves(metric="val_ssim",
                                  filename=f"baseline_{args.model}_{args.domain}_ssim.png")
    print(f"\nDone. Best SSIM: {best_ssim:.4f}")
    


if __name__ == "__main__":
    main()
