"""A-22 data pipeline: full 768x768 views from the preprocessed dataset (src/preprocess.py output).

Samples are returned as uint8 tensors; conversion to float and normalization happen on the GPU (gpu_inputs).
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LINE_KEYS = ("strap_cl_3px", "strap_cl_1px", "cord_cl_1px")  # centerline maps stored in each npz


def load_stats(root):
    return json.loads((Path(root) / "stats.json").read_text())


def read_index(root, split):
    index = pd.read_csv(Path(root) / "index.csv")
    return index[index["split"] == split].sort_values("sample_id").reset_index(drop=True)


def normalization(data_cfg):
    """(mean, std) per RGB channel for the configured normalization."""
    if data_cfg["normalization"] == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    stats = load_stats(data_cfg["root"])
    return tuple(stats["image_mean_rgb"]), tuple(stats["image_std_rgb"])


def subset_ids(index, size, seed):
    """Fixed validation subset, stratified over defect states so baselines and every defect are included."""
    rng = np.random.default_rng(seed)
    groups = [list(rng.permutation(g.index.to_numpy())) for _, g in index.groupby("defect")]
    size, picked = min(size, len(index)), []
    while len(picked) < size:
        for g in groups:  # round-robin over states
            if g and len(picked) < size:
                picked.append(g.pop(0))
    return sorted(picked)


def degrade(image, data_cfg, rng):
    """With probability blur_noise_p, apply one slight degradation: Gaussian blur or Gaussian noise."""
    if rng.random() >= data_cfg["blur_noise_p"]:
        return image
    if rng.random() < 0.5:
        return cv2.GaussianBlur(image, (0, 0), rng.uniform(*data_cfg["blur_sigma"]))
    noise = rng.normal(0.0, rng.uniform(*data_cfg["noise_std"]), image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


class A22Dataset(Dataset):
    """One full view per item: image (3,H,W), mask (C,H,W) in model.classes order, and the centerline maps
    (1,H,W) each. All uint8 (masks/lines 0/1). Augmentation (train only) touches the image only."""

    def __init__(self, cfg, split, train=False, subset=None):
        self.root = Path(cfg["data"]["root"])
        self.data_cfg = cfg["data"]
        self.index = read_index(self.root, split)
        if subset is not None:
            self.index = self.index.loc[subset_ids(self.index, subset, cfg["seed"])].reset_index(drop=True)
        channels = load_stats(self.root)["channels"]
        self.channel_idx = [channels.index(c) for c in cfg["model"]["classes"]]
        self.train = train

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        row = self.index.iloc[idx]
        with np.load(self.root / row["npz_path"]) as z:
            image = z["image"]
            mask = z["mask"][self.channel_idx]
            lines = {k: z[k] for k in LINE_KEYS}
        if self.train:
            rng = np.random.default_rng(torch.randint(0, 2**31, ()).item())  # follows the seeded torch RNG
            image = degrade(image, self.data_cfg, rng)
        item = {"image": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
                "mask": torch.from_numpy(np.ascontiguousarray(mask)), "sample_id": row["sample_id"]}
        item.update({k: torch.from_numpy(v)[None] for k, v in lines.items()})
        return item


def gpu_inputs(batch, device, mean, std, with_targets=True):
    """Move a uint8 batch to the GPU, then convert: normalized float image (channels_last) and float targets."""
    image = batch["image"].to(device, non_blocking=True).float().div_(255)
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)
    image = ((image - m) / s).contiguous(memory_format=torch.channels_last)
    if not with_targets:
        return image, None
    targets = {k: batch[k].to(device, non_blocking=True).float() for k in ("mask", *LINE_KEYS)}
    return image, targets


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
