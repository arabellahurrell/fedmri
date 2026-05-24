import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Literal, Optional, Callable, List, Tuple
import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import RandomMaskFunc


    # Single-coil knee FastMRI dataset.

    # Returns a dict with:
    #   'image_input'  (domain='image') : (2, H, W) zero-filled reconstruction
    #   'image_target' (domain='image') : (H, W) RSS ground truth
    #   'kspace'       (domain='kspace'): (2, H, W) masked k-space
    #   'mask'                          : (1, 1, W) undersampling mask
    #   'fname', 'slice_idx'


def build_mask_func(acceleration: int = 4, center_fractions: float = 0.08) -> RandomMaskFunc:
    return RandomMaskFunc(center_fractions=[center_fractions], accelerations=[acceleration])


class FastMRISliceDataset(Dataset):

    def __init__(
        self,
        root: str,
        domain: Literal["kspace", "image"] = "image",
        split: Literal["train", "val", "test"] = "train",
        acceleration: int = 4,
        center_fractions: float = 0.08,
        max_slices_per_volume: Optional[int] = None,
        transform: Optional[Callable] = None,
        seed: int = 42,
    ):
        self.root = Path(root)
        self.domain = domain
        self.split = split
        self.acceleration = acceleration
        self.mask_func = build_mask_func(acceleration, center_fractions)
        self.transform = transform
        self.rng = np.random.RandomState(seed)

        split_dir = self.root / f"knee_singlecoil_{split}"
        if not split_dir.exists():
            raise FileNotFoundError(
                f"FastMRI split directory not found: {split_dir}\n"
                "Please download from https://fastmri.med.nyu.edu/ and extract to data/fastmri/"
            )

        self.samples: List[Tuple[Path, int]] = []
        for h5_path in sorted(split_dir.glob("*.h5")):
            with h5py.File(h5_path, "r") as f:
                num_slices = f["kspace"].shape[0]
            n = num_slices if max_slices_per_volume is None else min(num_slices, max_slices_per_volume)
            self.samples.extend((h5_path, i) for i in range(n))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        fpath, slice_idx = self.samples[idx]

        with h5py.File(fpath, "r") as f:
            kspace_np = f["kspace"][slice_idx]

        kspace = torch.from_numpy(kspace_np.astype(np.complex64))

        # Simple global normalisation by max magnitude
        norm = kspace.abs().max()
        if norm > 0:
            kspace = kspace / norm

        kspace_t, mask, _ = T.apply_mask(
            kspace.unsqueeze(0).unsqueeze(0),
            self.mask_func,
            seed=None,
        )
        kspace_masked = kspace_t.squeeze(0).squeeze(0)

        # Ground truth from fully sampled k-space
        image_gt = fastmri.ifft2c(torch.view_as_real(kspace).unsqueeze(0))
        image_gt_rss = fastmri.rss(fastmri.complex_abs(image_gt), dim=0)

        if self.domain == "image":
            zf_complex = fastmri.ifft2c(torch.view_as_real(kspace_masked).unsqueeze(0))
            image_input = zf_complex.squeeze(0).permute(2, 0, 1)
            sample = {
                "image_input": image_input,
                "image_target": image_gt_rss,
                "mask": mask,
                "fname": str(fpath.name),
                "slice_idx": slice_idx,
            }
        else:
            ks_masked_2ch = torch.view_as_real(kspace_masked).permute(2, 0, 1)
            ks_full_2ch   = torch.view_as_real(kspace).permute(2, 0, 1)
            sample = {
                "kspace": ks_masked_2ch,
                "kspace_target": ks_full_2ch,
                "image_target": image_gt_rss,
                "mask": mask,
                "fname": str(fpath.name),
                "slice_idx": slice_idx,
            }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def get_client_dataloaders(
    root: str,
    domain: str = "image",
    acceleration: int = 4,
    batch_size: int = 4,
    num_clients: int = 4,
    num_workers: int = 4,
    pin_memory: bool = False,
    seed: int = 42,
) -> Tuple[List[DataLoader], DataLoader]:
    # IID split
    from torch.utils.data import Subset
    import numpy as np

    train_ds = FastMRISliceDataset(root=root, domain=domain, split="train",
                                   acceleration=acceleration, seed=seed)
    val_ds   = FastMRISliceDataset(root=root, domain=domain, split="val",
                                   acceleration=acceleration, seed=seed)

    rng = np.random.RandomState(seed)
    indices = np.arange(len(train_ds))
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)

    train_loaders = [
        DataLoader(Subset(train_ds, s.tolist()), batch_size=batch_size,
                   shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        for s in splits
    ]
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    return train_loaders, val_loader
