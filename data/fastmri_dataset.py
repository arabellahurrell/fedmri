import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from typing import Dict, Literal, Optional, Callable, List, Tuple
import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import RandomMaskFunc, EquispacedMaskFractionFunc
import xml.etree.ElementTree as ET
import json

    # Single-coil knee FastMRI dataset.

    # Returns a dict with:
    #   'image_input'  (domain='image') : (2, H, W) zero-filled reconstruction
    #   'image_target' (domain='image') : (H, W) RSS ground truth
    #   'kspace'       (domain='kspace'): (2, H, W) masked k-space
    #   'mask'                          : (1, 1, W) undersampling mask
    #   'fname', 'slice_idx'
#     Four scanner groups found in FastMRI knee:
#   hospital_A — Siemens Aera
#   hospital_B — Siemens Skyra
#   hospital_C — Siemens Biograph_mMR
#   hospital_D — Siemens Prisma_fit

def center_crop(tensor: torch.Tensor, crop_h: int, crop_w: int) -> torch.Tensor:
    #center crop the last two spatial dims of a tensor.
    h, w = tensor.shape[-2], tensor.shape[-1]
    top  = max((h - crop_h) // 2, 0)
    left = max((w - crop_w) // 2, 0)
    return tensor[..., top:top + min(crop_h, h), left:left + min(crop_w, w)]




def build_mask_func(acceleration: int = 4, center_fractions: float = 0.08, mask_type: Literal["random", "equispaced"] = "random",) -> RandomMaskFunc:
    cf = [center_fractions]
    acc = [acceleration]
    if mask_type == "random":
        return RandomMaskFunc(center_fractions=[center_fractions], accelerations=[acceleration])
    return EquispacedMaskFractionFunc(center_fractions=cf, accelerations=acc)

def extract_volume_metadata(h5_path: Path) -> dict:
    #gets the metadata from the ismrmrd header of the h5 file, which contains information about the acquisition, scanner model and field strength. It returns a dict with these values.
    with h5py.File(h5_path, "r") as f:
        acquisition = str(f.attrs.get("acquisition", "unknown"))
        hdr_bytes = f["ismrmrd_header"][()]

    hdr_xml = hdr_bytes.decode() if isinstance(hdr_bytes, bytes) else hdr_bytes
    root = ET.fromstring(hdr_xml)

    def find_text(tag):
        el = root.find(f".//{{{root.tag.split('}')[0].lstrip('{')}}}{tag}")
        if el is None:
            el = root.find(f".//{tag}")
        return el.text.strip() if el is not None and el.text else "unknown"

    return {
        "acquisition":    acquisition,
        "scanner_model":  find_text("systemModel"),
        "field_strength": find_text("systemFieldStrength_T"),
    }


SCANNER_TO_CLIENT = {
    "Aera":         "hospital_A",
    "Skyra":        "hospital_B",
    "Biograph_mMR": "hospital_C",
    "Prisma_fit":   "hospital_D",
}

def load_or_build_slice_cache(
    split_dir: Path,
    cache_path: Optional[Path] = None,
) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, dict]]:
    if cache_path is not None and cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
        return data["slices"], data["norm_scales"], data.get("meta", {})

    print(f"  Building volume cache for {split_dir.name} — this is a one-time scan...")
    cache: Dict[str, int] = {}
    norm_cache: Dict[str, float] = {}
    meta_cache: Dict[str, dict] = {}
    files = sorted(split_dir.glob("*.h5"))
    for i, h5_path in enumerate(files):
        if i % 50 == 0:
            print(f"    {i}/{len(files)} volumes scanned...", flush=True)
        try:
            with h5py.File(h5_path, "r") as f:
                cache[h5_path.stem] = int(f["kspace"].shape[0])
                rss = f["reconstruction_rss"][()]
                norm_cache[h5_path.stem] = float(np.percentile(np.abs(rss), 95)) or 1.0
            meta = extract_volume_metadata(h5_path)
            meta_cache[h5_path.stem] = meta
        except Exception as e:
            print(f"    Warning: could not read {h5_path.name}: {e}")
            meta_cache[h5_path.stem] = {"acquisition": "unknown", "scanner_model": "unknown", "field_strength": "unknown"}

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"slices": cache, "norm_scales": norm_cache, "meta": meta_cache}, f)
        print(f"  Slice cache saved to {cache_path}")

    return cache, norm_cache, meta_cache


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
        crop_size: Optional[Tuple[int, int]] = None,
        seed: int = 42,
        cache_dir: Optional[str] = None, 
    ):
        self.root = Path(root)
        self.domain = domain
        self.split = split
        self.acceleration = acceleration
        self.mask_func = build_mask_func(acceleration, center_fractions)
        self.transform = transform
        self.rng = np.random.RandomState(seed)
        self.crop_size = crop_size

        split_dir = self.root / f"knee_singlecoil_{split}"
        if not split_dir.exists():
            raise FileNotFoundError(
                f"FastMRI split directory not found: {split_dir}\n"
                "Please download from https://fastmri.med.nyu.edu/ and extract to data/fastmri/"
            )
        
        cache_path = None
        if cache_dir is not None:
            cache_path = Path(cache_dir) / f"volume_cache_{split}.json"
        slice_cache, norm_cache, meta_cache = load_or_build_slice_cache(split_dir, cache_path)


        self.samples: List[Tuple[Path, int]] = []
        self.volume_meta: Dict[str, dict] = {}

        for h5_path in sorted(split_dir.glob("*.h5")):
            stem = h5_path.stem
            if stem in slice_cache:
                num_slices = slice_cache[stem]
            else:
                try:
                    with h5py.File(h5_path, "r") as f:
                        num_slices = int(f["kspace"].shape[0])
                except Exception as e:
                    print(f"  Skipping {h5_path.name}: {e}")
                    continue
            # try:
            #     with h5py.File(h5_path, "r") as f:
            #         # num_slices = f["kspace"].shape[0]
            #         rss = f["reconstruction_rss"][()]
            #     norm_scale = float(np.percentile(np.abs(rss), 95)) or 1.0
            # except Exception as e:
            #     norm_scale = 1.0
            norm_scale = norm_cache.get(stem, 1.0)
            
            meta = meta_cache.get(stem, {"acquisition": "unknown", "scanner_model": "unknown", "field_strength": "unknown"})
            meta["client"] = SCANNER_TO_CLIENT.get(meta["scanner_model"], "hospital_unknown")
            meta["norm_scale"] = norm_scale
            self.volume_meta[stem] = meta
            n = num_slices if max_slices_per_volume is None else min(num_slices, max_slices_per_volume)
            self.samples.extend((h5_path, i) for i in range(n))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        fpath, slice_idx = self.samples[idx]
        meta = self.volume_meta[fpath.stem]
        norm_scale = meta["norm_scale"]

        with h5py.File(fpath, "r") as f:
            kspace_np = f["kspace"][slice_idx]

        kspace = torch.from_numpy(kspace_np.astype(np.complex64))

        kspace_t, mask, _ = T.apply_mask(
            kspace.unsqueeze(0).unsqueeze(0),
            self.mask_func,
            seed=None,
        )
        kspace_masked = kspace_t.squeeze(0).squeeze(0)

        kspace = kspace / norm_scale
        kspace_masked = kspace_masked / norm_scale

        # Ground truth from fully sampled k-space
        image_gt = fastmri.ifft2c(torch.view_as_real(kspace).contiguous().unsqueeze(0))
        image_gt_rss = fastmri.rss(fastmri.complex_abs(image_gt), dim=0)

        if self.crop_size is not None:
            ch, cw = self.crop_size
            image_gt_rss = center_crop(image_gt_rss, ch, cw)

        if self.domain == "image":
            zf_complex = fastmri.ifft2c(torch.view_as_real(kspace_masked).contiguous().unsqueeze(0))
            image_input = zf_complex.squeeze(0).permute(2, 0, 1)
            sample = {
                "image_input": image_input.contiguous().clone(),
                "image_target": image_gt_rss.contiguous().clone(),
                "mask": mask.contiguous().clone(),
                "fname": str(fpath.name),
                "slice_idx": slice_idx,
                "acquisition": meta["acquisition"],
                "scanner_model": meta["scanner_model"],
                "client": meta["client"],
                "norm_scale": norm_scale,
            }
        else:
            ks_masked_2ch = torch.view_as_real(kspace_masked).permute(2, 0, 1)
            ks_full_2ch   = torch.view_as_real(kspace).permute(2, 0, 1)
            
            if self.crop_size is not None:
                ch, cw = self.crop_size
                ks_masked_2ch = center_crop(ks_masked_2ch, ch, cw)
                ks_full_2ch   = center_crop(ks_full_2ch, ch, cw)
            
            sample = {
                "kspace": ks_masked_2ch.contiguous().clone(),
                "kspace_target": ks_full_2ch.contiguous().clone(),
                "image_target": image_gt_rss.contiguous().clone(),
                "mask": mask.contiguous().clone(),
                "fname": str(fpath.name),
                "slice_idx": slice_idx,
                "acquisition": meta["acquisition"],
                "scanner_model": meta["scanner_model"],
                "client": meta["client"],
                "norm_scale": norm_scale,
            }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

def partition_by_scanner(
    dataset: FastMRISliceDataset,
) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for i, (fpath, _) in enumerate(dataset.samples):
        label = dataset.volume_meta[fpath.stem]["client"]
        groups.setdefault(label, []).append(i)
    return groups


def partition_by_acquisition(
    dataset: FastMRISliceDataset,
) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for i, (fpath, _) in enumerate(dataset.samples):
        label = dataset.volume_meta[fpath.stem]["acquisition"]
        groups.setdefault(label, []).append(i)
    return groups


def partition_iid(
    dataset: FastMRISliceDataset,
    num_clients: int,
    seed: int = 42,
) -> Dict[str, List[int]]:
    rng = np.random.RandomState(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return {f"client_{i}": s.tolist() for i, s in enumerate(splits)}


def get_client_dataloaders(
    root: str,
    domain: Literal["kspace", "image"] = "image",
    acceleration: int = 4,
    batch_size: int = 4,
    partition: Literal["scanner", "acquisition", "iid"] = "scanner",
    num_clients_iid: int = 4,
    num_workers: int = 4,
    pin_memory: bool = False,
    seed: int = 42,
) -> Tuple[Dict[str, DataLoader], DataLoader]:
    # IID split

    train_ds = FastMRISliceDataset(root=root, domain=domain, split="train",
                                   acceleration=acceleration, seed=seed)
    val_ds   = FastMRISliceDataset(root=root, domain=domain, split="val",
                                   acceleration=acceleration, seed=seed)

    if partition == "scanner":
        groups = partition_by_scanner(train_ds)
    elif partition == "acquisition":
        groups = partition_by_acquisition(train_ds)
    else:
        groups = partition_iid(train_ds, num_clients_iid, seed=seed)

    train_loaders = {
        label: DataLoader(
            Subset(train_ds, idxs),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for label, idxs in groups.items()
    }

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loaders, val_loader
