# fedmri — Privacy in Federated MRI Reconstruction

An attack-based study of how the **reconstruction modelling paradigm** affects the
privacy–utility trade-off in federated, differentially private MRI reconstruction.

This repository accompanies an MEng individual project (Imperial College London,
Department of Computing). It compares three reconstruction models — a data-driven
image-domain U-Net, a data-driven k-space U-Net, and a physics-informed unrolled
k-space cascade — across centralised and federated training, a differential-privacy
(DP) sweep, and two privacy attacks, over a scanner-partitioned, non-IID federation
of the single-coil fastMRI knee dataset.

---

## Repository structure

```
fedmri/
├── data/
│   └── fastmri_dataset.py       # FastMRISliceDataset, partition_by_scanner (scanner cohorts)
├── models/
│   ├── unet.py                  # image-domain U-Net + ReconstructionLoss (L1 + SSIM)
│   ├── kspace_unet.py           # residual k-space U-Net (same backbone, frequency domain)
│   └── modfed.py                # unrolled data-consistency cascade (ModFed core)
├── federated/
│   ├── fl_simulation.py         # Flower-based FL simulation; MODEL_DOMAINS
├── privacy/
│   └── dp_training.py           # DP-SGD via Opacus; empty-batch / Poisson handling
├── attacks/
│   ├── gradient_inversion.py    # TVGradientInversion (+ optional breaching backend)
│   └── membership_inference.py  # LossThresholdMIA (+ shadow-model variant)
├── evaluation/
│   └── metrics.py               # SSIM / PSNR / NMSE / LPIPS, ResultsTracker
├── scripts/                     # per-cohort eval, qualitative panels, SLURM jobs
├── train_baseline.py            # centralised training entry point
├── run_federated.py             # federated (and DP) training entry point
├── run_attacks.py               # gradient-inversion + membership-inference entry point
├── analyse_heterogeneity.py     # Earth Mover's distances + intensity distributions
└── requirements.txt
```


---

## Installation

Requires Python 3.10+ and a CUDA-capable GPU for training and attacks.

```bash
git clone https://github.com/arabellahurrell/fedmri.git
cd fedmri
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Core dependencies: PyTorch, [Flower](https://flower.ai/) (federated simulation),
[Opacus](https://opacus.ai/) (differential privacy), `fastmri`, `scikit-image`, and
`lpips`.

---

## Data

This project uses the **single-coil knee** subset of the
[fastMRI](https://fastmri.med.nyu.edu/) dataset (Zbontar et al., 2018). The dataset is
**not redistributed here**; obtain it by registering and accepting the data-use
agreement on the fastMRI website.

Place the raw HDF5 files so the loader can find the training and validation splits, e.g.:

```
data/fastmri/
├── singlecoil_train/
└── singlecoil_val/
```

and pass `--data_root data/fastmri` to the scripts (this is the default). The federated
partition is derived from the ISMRMRD acquisition headers already present in the
release: each volume is assigned to one of four Siemens scanner cohorts
(Aera 1.5 T, Skyra 3 T, Biograph mMR 3 T, Prisma fit 3 T).

Acceleration is simulated retrospectively with a 1D Cartesian mask: a fully sampled
central band (8% of lines) plus random lines to reach **4× acceleration**, with a fresh
mask drawn per slice.

---

## Reproducing the experiments

Model identifiers
are `unet`, `kspace_unet`, and `modfed`.

### 1. Heterogeneity analysis (Contribution 4)

```bash
python analyse_heterogeneity.py --data_root data/fastmri
```

Computes pairwise Earth Mover's (1-Wasserstein) distances between the four cohorts'
intensity distributions and plots the per-client intensity histograms, for both splits.

### 2. Centralised training

```bash
python train_baseline.py --model unet        --data_root data/fastmri
python train_baseline.py --model kspace_unet  --data_root data/fastmri
python train_baseline.py --model modfed       --data_root data/fastmri
```

### 3. Federated training (non-private)

```bash
python run_federated.py --model modfed --rounds 20 --local_epochs 2 --data_root data/fastmri
```

Four clients, full participation, 20 communication rounds × 2 local epochs.

### 4. Differentially private sweep (Contribution 2)

```bash
# Image U-Net: eps in {1, 5, 10, 20, 50, inf}
# k-space models: eps in {1, 10, 50, inf}
python run_federated.py --model kspace_unet --dp --epsilon 1  --rounds 10 --data_root data/fastmri
```

DP-SGD via Opacus with per-example clipping `C = 1.0` and `δ = 1e-5`; budgets are
accounted over a client's full participation. See the note on Poisson sampling below.

### 5. Privacy attacks (Contribution 1)

```bash
python run_attacks.py \
    --checkpoints checkpoints/federated/modfed_scanner_round20.pt \
    --attacks gia mia \
    --gi_batches 10 --gi_iters 2000 --gi_restarts 3 \
    --mia_samples 2000 \
    --results_dir results/attacks
```

`run_attacks.py` mounts gradient inversion (single-sample, single-step gradients,
total-variation prior, multiple restarts) and a loss-threshold membership-inference
attack. Recovered inputs are scored in a **common image space** so image- and
k-space-domain models are comparable.

### 6. Per-cohort evaluation and qualitative panels

```bash
python scripts/per_client_eval.py   --checkpoints <ckpt...> --data_root data/fastmri
python scripts/qualitative_panel.py --checkpoints <ckpt...> --out figures/panel.png
```

`qualitative_panel.py` produces reconstruction + absolute-error-map panels per scanner
cohort.

---

## The differentially private cascade (systems contribution)

Training an unrolled, physics-constrained cascade under DP-SG required resolving three obstacles, implemented in `models/modfed.py` and
`federated/dp_training.py`:

- **A differentiable Fourier adjoint.** The cascade applies forward/inverse Fourier
  transforms repeatedly; their analytic adjoints are defined as explicit autograd
  functions so gradients propagate correctly under both Opacus's per-example
  differentiation and the second-order graph the inversion attack requires.
- **Empty / variable-size batches.** Poisson subsampling produces occasionally-empty
  batches that the cascade's Fourier and magnitude/phase operations are undefined on;
  guards return correctly-shaped empty tensors and the loader skips empty batches.
- **DP-compatible normalisation.** Every model uses instance normalisation (not batch
  normalisation), so a clean per-example gradient exists.

---

## Notes and caveats

- **Poisson sampling / ε comparability.** Poisson subsampling is disabled for the
  k-space models for a stable forward pass, but the accountant still credits the
  amplification, so the reported ε for those models is **optimistic**. Within-model
  trends and the k-space-U-Net-vs-cascade comparison (identical sampling) are fair;
  image-vs-k-space comparisons at matched ε are indicative only; ε = ∞ comparisons are
  clean.
- **DP rounds.** Non-private federated runs use 20 rounds; DP runs use 10. The cascade
  trains unstably under DP, so its finite-budget checkpoints are snapshots of an
  oscillating trajectory.
- **Reproducibility.** Fixed seed (`42`), per-round checkpointing of the global model,
  and incremental metric logging (interrupted runs resume).

---

## Citation

If you use this code, please cite the accompanying report:

```
A. Hurrell. Does the Reconstruction Paradigm Shape Privacy? An Attack-Based Study of
Federated, Differentially Private MRI Reconstruction. MEng Individual Project,
Imperial College London, Department of Computing, 2026.
```

---

## Acknowledgements

- **fastMRI** — Zbontar et al., *fastMRI: An Open Dataset and Benchmarks for Accelerated
  MRI* (2018).
- **ModFed** — Wu et al., *Model-based federated learning for accurate MR image
  reconstruction from undersampled k-space data* (2023); the data-consistency cascade
  core is adapted from this work.
- Built with [Flower](https://flower.ai/), [Opacus](https://opacus.ai/), and the
  [fastMRI](https://github.com/facebookresearch/fastMRI) library.

### Use of generative AI

Generative AI tools were used during development: Claude (Anthropic) assisted with
debugging, figure-plotting code, locating library documentation, and sense-checking the
logic behind the code. All AI-assisted output was reviewed, tested, and verified by the
author, and no AI-generated content is presented as independent intellectual
contribution.

