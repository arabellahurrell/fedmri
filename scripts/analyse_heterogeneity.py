import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from collections import defaultdict
import pandas as pd

from data.fastmri_dataset import FastMRISliceDataset, partition_by_scanner

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   default="data/fastmri")
    p.add_argument("--split",       default="val", choices=["train", "val"])
    p.add_argument("--results_dir", default="results/heterogeneity")
    p.add_argument("--n_hist_bins", type=int, default=50)
    return p.parse_args()


def build_client_stats(ds: FastMRISliceDataset, groups: dict) -> pd.DataFrame:
    rows = []
    for client, idxs in sorted(groups.items()):
        volumes = set(ds.samples[i][0].stem for i in idxs)
        # Gather acquisition types across volumes
        acqs = defaultdict(int)
        for stem in volumes:
            acqs[ds.volume_meta[stem]["acquisition"]] += 1

        meta_sample = ds.volume_meta[next(iter(volumes))]
        rows.append({
            "client":        client,
            "scanner":       meta_sample["scanner_model"],
            "field_T":       meta_sample["field_strength"],
            "n_slices":      len(idxs),
            "n_volumes":     len(volumes),
            "CORPD_FBK":     acqs.get("CORPD_FBK", 0),
            "CORPDFS_FBK":   acqs.get("CORPDFS_FBK", 0),
        })
    return pd.DataFrame(rows).set_index("client")


def collect_intensity_histograms(
    ds: FastMRISliceDataset,
    groups: dict,
    n_bins: int = 50,
    max_slices_per_client: int = 200,
) -> dict:
    histograms = {}
    bin_edges = None

    for client, idxs in sorted(groups.items()):

        rng = np.random.default_rng(42)
        sample_idxs = rng.choice(idxs, size=min(max_slices_per_client, len(idxs)),
                                  replace=False)
        values = []
        for i in sample_idxs:
            fpath, slice_idx = ds.samples[i]
            with h5py.File(fpath, "r") as f:
                rss = f["reconstruction_rss"][slice_idx]
            values.append(rss.flatten())

        vals = np.concatenate(values)
        cap = np.percentile(vals, 99)
        vals = vals[vals <= cap]

        if bin_edges is None:
            global_cap = cap 
            all_vals_tmp = vals
        else:
            all_vals_tmp = np.concatenate([all_vals_tmp, vals])

        histograms[client] = vals

    all_vals = np.concatenate(list(histograms.values()))
    bin_edges = np.linspace(all_vals.min(), np.percentile(all_vals, 99), n_bins + 1)

    result = {}
    for client, vals in histograms.items():
        counts, _ = np.histogram(vals, bins=bin_edges, density=True)
        result[client] = {"counts": counts, "edges": bin_edges, "raw": vals}

    return result


def compute_pairwise_emd(histograms: dict) -> pd.DataFrame:
    clients = sorted(histograms.keys())
    n = len(clients)
    emd_matrix = np.zeros((n, n))

    normed = {}
    for c in clients:
        raw = histograms[c]["raw"].astype(np.float64)
        p99 = np.percentile(raw, 99) or 1.0
        normed[c] = raw / p99

    for i, ci in enumerate(clients):
        for j, cj in enumerate(clients):
            if i == j:
                continue
            rng = np.random.default_rng(42)
            n_sub = min(5000, len(normed[ci]), len(normed[cj]))
            u = rng.choice(normed[ci], n_sub, replace=False)
            v = rng.choice(normed[cj], n_sub, replace=False)
            emd_matrix[i, j] = wasserstein_distance(u, v)

    return pd.DataFrame(emd_matrix, index=clients, columns=clients)

def plot_acquisition_breakdown(stats: pd.DataFrame, results_dir: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(stats))
    w = 0.35
    b1 = ax.bar(x - w/2, stats["CORPD_FBK"],    w, label="CORPD_FBK",    color="#4C72B0")
    b2 = ax.bar(x + w/2, stats["CORPDFS_FBK"],  w, label="CORPDFS_FBK",  color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{idx}\n({stats.loc[idx,'scanner']})" for idx in stats.index],
        fontsize=9,
    )
    ax.set_ylabel("Number of volumes")
    ax.set_title("Acquisition Protocol Distribution per FL Client (Hospital)")
    ax.legend()
    ax.bar_label(b1, padding=2)
    ax.bar_label(b2, padding=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = f"{results_dir}/acquisition_breakdown.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_intensity_histograms(histograms: dict, results_dir: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for (client, h), color in zip(sorted(histograms.items()), colors):
        centers = 0.5 * (h["edges"][:-1] + h["edges"][1:])
        ax.plot(centers, h["counts"], label=client, color=color, linewidth=1.8)
    ax.set_xlabel("RSS magnitude (normalised pixel intensity)")
    ax.set_ylabel("Density")
    ax.set_title("Intensity Distribution per FL Client — Evidence of Non-IID Data")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = f"{results_dir}/intensity_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_emd_heatmap(emd_df: pd.DataFrame, results_dir: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(emd_df.values, cmap="YlOrRd")
    ax.set_xticks(range(len(emd_df.columns)))
    ax.set_yticks(range(len(emd_df.index)))
    ax.set_xticklabels(emd_df.columns, rotation=30, ha="right")
    ax.set_yticklabels(emd_df.index)
    plt.colorbar(im, ax=ax, label="Wasserstein-1 distance")
    for i in range(len(emd_df)):
        for j in range(len(emd_df.columns)):
            ax.text(j, i, f"{emd_df.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if emd_df.values[i, j] > emd_df.values.max() * 0.6 else "black")
    ax.set_title("Pairwise EMD Between Client Distributions\n(higher = more heterogeneous)")
    plt.tight_layout()
    out = f"{results_dir}/emd_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Loading {args.split} dataset from {args.data_root} ...")
    ds = FastMRISliceDataset(
        root=args.data_root, domain="image", split=args.split
    )
    groups = partition_by_scanner(ds)

    stats = build_client_stats(ds, groups)
    print("\nClient distribution table:")
    print(stats.to_string())
    stats.to_csv(f"{args.results_dir}/client_stats.csv")
    print(f"\n  Saved: {args.results_dir}/client_stats.csv")

    plot_acquisition_breakdown(stats, args.results_dir)

    print("\nCollecting intensity histograms (this reads h5 files) ...")
    histograms = collect_intensity_histograms(ds, groups, n_bins=args.n_hist_bins)
    plot_intensity_histograms(histograms, args.results_dir)

    emd_df = compute_pairwise_emd(histograms)
    print("\nPairwise Earth Mover's Distance (normalised intensity):")
    print(emd_df.round(6).to_string())
    emd_df.to_csv(f"{args.results_dir}/emd_matrix.csv")
    plot_emd_heatmap(emd_df, args.results_dir)

    print("\nHeterogeneity analysis complete.")
    print(f"Outputs in: {args.results_dir}/")


if __name__ == "__main__":
    main()
