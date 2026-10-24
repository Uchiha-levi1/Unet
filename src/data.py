"""Massachusetts Roads data pipeline: centerline cache, random training crops, full-image eval set."""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import tifffile
import torch
from skimage.morphology import skeletonize
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# The Kaggle mirror's metadata.csv names the columns tiff_image_path / tif_label_path
# (paths relative to the data root); the png_* columns point to files that are not shipped.
IMAGE_COL, MASK_COL = "tiff_image_path", "tif_label_path"


def read_metadata(root, split):
    meta = pd.read_csv(Path(root) / "metadata.csv")
    return meta[meta["split"] == split].sort_values("image_id").reset_index(drop=True)


def read_image(path):
    return tifffile.imread(path)  # (H, W, 3) uint8


def read_mask(path):
    mask = tifffile.imread(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 127  # (H, W) bool


def centerline_from_mask(mask, width):
    skel = skeletonize(mask)
    kernel = np.ones((width, width), np.uint8)
    return cv2.dilate(skel.astype(np.uint8), kernel) > 0


def _centerline_job(args):
    mask_path, out_path, width = args
    mask = read_mask(mask_path)
    cl = centerline_from_mask(mask, width)
    cv2.imwrite(str(out_path), cl.astype(np.uint8) * 255)
    return int(cl.sum()), cl.size


def centerline_path(cache_dir, image_id):
    return Path(cache_dir) / "centerline" / f"{image_id}.png"


def build_cache(data_cfg, workers=8):
    """Precompute centerline targets for all splits and centerline pos/neg counts over the training set.

    Writes <cache_dir>/centerline/<id>.png and <cache_dir>/stats.json. Skips work already done.
    """
    root, cache_dir = Path(data_cfg["root"]), Path(data_cfg["cache_dir"])
    (cache_dir / "centerline").mkdir(parents=True, exist_ok=True)
    stats_path = cache_dir / "stats.json"
    meta = pd.read_csv(root / "metadata.csv")
    todo = [(root / r[MASK_COL], centerline_path(cache_dir, r.image_id), data_cfg["centerline_width"])
            for _, r in meta.iterrows() if not centerline_path(cache_dir, r.image_id).exists()]
    if todo:
        print(f"Building centerline cache for {len(todo)} images ...")
        with Pool(workers) as pool:
            pool.map(_centerline_job, todo, chunksize=8)
    if not stats_path.exists():
        pos = total = 0
        for image_id in meta.loc[meta["split"] == "train", "image_id"]:
            cl = cv2.imread(str(centerline_path(cache_dir, image_id)), cv2.IMREAD_GRAYSCALE) > 127
            pos += int(cl.sum())
            total += cl.size
        stats = {"centerline_pos": pos, "centerline_neg": total - pos,
                 "centerline_neg_over_pos": (total - pos) / max(pos, 1)}
        stats_path.write_text(json.dumps(stats, indent=2))
    return json.loads(stats_path.read_text())


def to_tensor(image, mask, centerline):
    image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
    mask = torch.from_numpy(np.ascontiguousarray(mask))[None].float()
    centerline = torch.from_numpy(np.ascontiguousarray(centerline))[None].float()
    return image, mask, centerline


def train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.0, hue=0.02, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def eval_transform():
    return A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])


class RoadsTrainDataset(Dataset):
    """Random crops from training images. An index only selects the source image; the crop is random."""

    def __init__(self, data_cfg, split="train", transform=None, max_tries=20):
        self.root, self.cache_dir = Path(data_cfg["root"]), Path(data_cfg["cache_dir"])
        self.meta = read_metadata(self.root, split)
        self.crop = data_cfg["crop_size"]
        self.max_white = data_cfg["max_white_frac"]
        self.length = data_cfg.get("samples_per_epoch") or len(self.meta)
        self.transform = transform if transform is not None else train_transform()
        self.max_tries = max_tries

    def __len__(self):
        return self.length

    def random_crop(self, image, rng):
        h, w = image.shape[:2]
        for _ in range(self.max_tries):
            y, x = rng.integers(0, h - self.crop + 1), rng.integers(0, w - self.crop + 1)
            white = (image[y:y + self.crop, x:x + self.crop] == 255).all(-1).mean()
            if white <= self.max_white:
                break
        # If every try is mostly white (a few images are almost entirely no-data), keep the last one.
        return y, x

    def load(self, idx, rng):
        row = self.meta.iloc[idx % len(self.meta)]
        image = read_image(self.root / row[IMAGE_COL])
        mask = read_mask(self.root / row[MASK_COL])
        cl = cv2.imread(str(centerline_path(self.cache_dir, row.image_id)), cv2.IMREAD_GRAYSCALE) > 127
        y, x = self.random_crop(image, rng)
        sl = (slice(y, y + self.crop), slice(x, x + self.crop))
        return image[sl], mask[sl].astype(np.uint8), cl[sl].astype(np.uint8)

    def __getitem__(self, idx):
        rng = np.random.default_rng(torch.randint(0, 2**31, ()).item())  # follows the seeded torch RNG
        image, mask, cl = self.load(idx, rng)
        out = self.transform(image=image, masks=[mask, cl])
        return to_tensor(out["image"], *out["masks"])


class RoadsEvalDataset(Dataset):
    """Full-resolution images for sliding-window validation / test."""

    def __init__(self, data_cfg, split):
        self.root = Path(data_cfg["root"])
        self.meta = read_metadata(self.root, split)
        self.transform = eval_transform()

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        raw = read_image(self.root / row[IMAGE_COL])
        mask = read_mask(self.root / row[MASK_COL])
        image = self.transform(image=raw)["image"]
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        return {"image": image, "mask": torch.from_numpy(mask), "raw": torch.from_numpy(raw),
                "image_id": row.image_id}


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)


if __name__ == "__main__":
    from src.config import load_config
    ap = argparse.ArgumentParser(description="Precompute centerline targets and training-set stats.")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    print(build_cache(load_config(args.config)["data"]))
