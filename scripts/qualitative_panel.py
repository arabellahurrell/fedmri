import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.fastmri_dataset import FastMRISliceDataset, partition_by_scanner
from models.unet import UNet
from models.kspace_unet import KSpaceUNet
from models.modfed import ModFed
from federated.fl_simulation import MODEL_DOMAINS
from evaluation.metrics import compute_metrics

CTORS = {"unet": lambda: UNet(in_channels=2, out_channels=1, base_features=32, depth=4),
         "kspace_unet": lambda: KSpaceUNet(base_features=32, depth=4),
         "modfed": lambda: ModFed(num_cascades=6, kspace_ch=64, kspace_layers=5)}
NICE = {"unet": "Image U-Net", "kspace_unet": "k-space U-Net", "modfed": "Cascade"}

def load(path, device):
    ck = torch.load(path, map_location=device); low = path.lower()
    if isinstance(ck, dict) and "model_type" in ck: mt = ck["model_type"]
    elif "kspace_unet" in low: mt = "kspace_unet"
    elif "modfed" in low: mt = "modfed"
    elif "unet" in low: mt = "unet"
    else: raise ValueError(path)
    dom = (ck.get("domain") if isinstance(ck, dict) else None) or MODEL_DOMAINS[mt]
    st = ck.get("model_state_dict") or ck.get("state_dict") or ck if isinstance(ck, dict) else ck
    m = CTORS[mt](); m.load_state_dict(st); m.eval().to(device)
    return m, mt, dom

@torch.no_grad()
def recon(m, dom, img_s, ks_s, device):
    if dom == "image":
        return m(img_s["image_input"].unsqueeze(0).to(device)).squeeze().cpu().numpy()
    k = ks_s["kspace"].unsqueeze(0).to(device)
    try:    out = m(k, ks_s["mask"].unsqueeze(0).to(device))
    except (TypeError, KeyError): out = m(k)
    return out.squeeze().cpu().numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--data_root", default="data/fastmri")
    ap.add_argument("--acceleration", type=int, default=4)
    ap.add_argument("--indices", type=int, nargs="*", default=None)
    ap.add_argument("--out", default="figures/qualitative_panel.png")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Initializing fastMRI datasets (this reads files from disk)...")
    img_ds = FastMRISliceDataset(root=a.data_root, domain="image", split="val",
                                 acceleration=a.acceleration, seed=a.seed, cache_dir=a.data_root)
    ks_ds  = FastMRISliceDataset(root=a.data_root, domain="kspace", split="val",
                                 acceleration=a.acceleration, seed=a.seed, cache_dir=a.data_root)

    if a.indices:
        idxs = a.indices
    else:
        print("Scanning metadata to group slices by scanner (1.5T vs 3T)...")
        g = partition_by_scanner(img_ds)
        A, B = sorted(g["hospital_A"]), sorted(g["hospital_B"])
        idxs = [A[len(A)//2], B[len(B)//2]]        # 1.5T and 3T example

    print(f"Selected evaluation slice indices: {idxs}")
    models = {}
    for p in a.checkpoints:
        print(f"Loading weights from checkpoint: {p}")
        m, mt, dom = load(p, dev); models[mt] = (m, dom)
    order = [k for k in ["unet", "kspace_unet", "modfed"] if k in models]


    print("Running forward passes and generating panel figure...")

    panels = []
    for i in idxs:
        img_s, ks_s = img_ds[i], ks_ds[i]
        gt = img_s["image_target"].squeeze().cpu().numpy()
        zf = torch.sqrt(img_s["image_input"][0]**2 + img_s["image_input"][1]**2).cpu().numpy()
        scale = gt.max() + 1e-8
        rec, ssim = {}, {}
        for mt in order:
            m, dom = models[mt]
            r = recon(m, dom, img_s, ks_s, dev)
            ssim[mt] = compute_metrics(torch.from_numpy(r)[None], torch.from_numpy(gt)[None])["ssim"]
            rec[mt] = r / scale
        err = {mt: np.abs(rec[mt] - gt / scale) for mt in order}
        panels.append({"coh": img_s.get("client", ""), "gt": gt / scale,
                       "zf": zf / scale, "rec": rec, "err": err, "ssim": ssim})

    err_vmax = np.percentile(
        np.concatenate([e.ravel() for p in panels for e in p["err"].values()]), 99)

    ncols, nrows = 2 + len(order), 2 * len(panels)
    fig, ax = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.7 * nrows),
                           constrained_layout=True)

    im_err = None
    for s, p in enumerate(panels):
        ir, er = 2 * s, 2 * s + 1                          # image row, error row
        ax[ir][0].imshow(p["gt"], cmap="gray", vmin=0, vmax=1)
        ax[ir][0].set_title(f"Ground truth\n{p['coh']}", fontsize=9)
        ax[ir][1].imshow(p["zf"], cmap="gray", vmin=0, vmax=1)
        ax[ir][1].set_title("Zero-filled", fontsize=9)
        ax[er][0].text(0.5, 0.5, "Absolute\nerror", ha="center", va="center",
                       transform=ax[er][0].transAxes, fontsize=9)
        for c, mt in enumerate(order):
            ax[ir][2 + c].imshow(p["rec"][mt], cmap="gray", vmin=0, vmax=1)
            ax[ir][2 + c].set_title(f"{NICE[mt]}\nSSIM {p['ssim'][mt]:.3f}", fontsize=9)
            im_err = ax[er][2 + c].imshow(p["err"][mt], cmap="magma", vmin=0, vmax=err_vmax)
        for c in range(ncols):
            ax[ir][c].axis("off"); ax[er][c].axis("off")

    fig.colorbar(im_err, ax=ax, fraction=0.02, pad=0.01, label="Absolute error")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200)
    print("Saved", a.out)

    
if __name__ == "__main__":
    main()