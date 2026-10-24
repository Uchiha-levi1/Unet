"""Overfit sanity check: 4 fixed training crops, no augmentation, ~200 steps; the loss should approach zero.

python -m tests.overfit --config configs/default.yaml [--steps 200] [--plot runs/overfit_loss.png]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write to file, no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import load_config
from src.data import RoadsTrainDataset, build_cache, eval_transform, to_tensor
from src.losses import SegLoss
from src.model import build_model
from src.train import build_optimizer, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--plot", default="runs/overfit_loss.png", help="where to save the loss curve")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    stats = build_cache(cfg["data"])
    pos_weight = min(stats["centerline_neg_over_pos"], cfg["loss"]["pos_weight_cap"])
    ds = RoadsTrainDataset(cfg["data"], transform=eval_transform())
    rng = np.random.default_rng(cfg["seed"])
    crops = []
    for idx in rng.permutation(len(ds.meta)):  # first 4 images whose crop contains some road
        image, mask, cl = ds.load(int(idx), rng)
        if mask.mean() > 0.02:
            crops.append(to_tensor(ds.transform(image=image)["image"], mask, cl))
        if len(crops) == 4:
            break
    image, mask, cl = (torch.stack(t).cuda() for t in zip(*crops))
    image = image.contiguous(memory_format=torch.channels_last)

    model = build_model(cfg["model"]).cuda().to(memory_format=torch.channels_last)
    model.freeze_encoder_bn = True  # batch of 4 < 8
    criterion = SegLoss(cfg["loss"], pos_weight).cuda()
    optimizer = build_optimizer(model, cfg["train"])
    model.train()
    total_losses = []
    for step in range(1, args.steps + 1):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["train"]["amp"]):
            out = model(image)
        loss, terms = criterion(out, mask, cl)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_losses.append(terms["total"])
        if step == 1 or step % 20 == 0:
            print(f"step {step:4d} " + " ".join(f"{k} {v:.4f}" for k, v in terms.items()), flush=True)
    print("final total loss:", round(terms["total"], 4))

    # Log y-axis: the loss spans ~2 orders of magnitude, so the tail would be flat on a linear scale.
    plot_path = Path(args.plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(total_losses) + 1), total_losses)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("total loss (log scale)")
    ax.set_title(f"Overfit test: 4 crops, {args.steps} steps (final {total_losses[-1]:.4f})")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    print("loss curve saved to", plot_path)


if __name__ == "__main__":
    main()
