"""Overfit sanity check: 4 fixed full training views (768x768), no augmentation, ~200 steps; the loss should
approach zero. Needs a GPU with >= ~12 GB (run on Turing: jobs/overfit_turing.sbatch).

python -m tests.overfit --config configs/turing.yaml [--data-root DIR] [--steps 200] [--plot runs/overfit_loss.png]

Also prints how large each loss group is relative to the total (to check that focal is not negligible).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write to file, no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import default_collate

from src.config import load_config
from src.data import A22Dataset, gpu_inputs, load_stats, normalization
from src.losses import SegLoss
from src.model import build_model
from src.train import build_optimizer, set_seed


def group_shares(terms, lam):
    """Weighted contribution of each loss group to the total."""
    total = float(terms["total"])
    dice = sum(lam[k.split("/")[1]] * float(v) for k, v in terms.items() if k.startswith("dice/"))
    focal = sum(lam[k.split("/")[1]] * float(v) for k, v in terms.items() if k.startswith("focal/"))
    return {"dice": dice / total, "focal": focal / total, "rest (clDice, BCE)": 1 - (dice + focal) / total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/turing.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--plot", default="runs/overfit_loss.png", help="where to save the loss curve")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    set_seed(cfg["seed"])
    mcfg, lcfg = cfg["model"], cfg["loss"]

    ds = A22Dataset(cfg, "train", train=False)  # train=False: no augmentation
    idx = np.random.default_rng(cfg["seed"]).choice(len(ds), 4, replace=False)
    device = torch.device("cuda")
    mean, std = normalization(cfg["data"])
    image, targets = gpu_inputs(default_collate([ds[int(i)] for i in idx]), device, mean, std)

    stats = load_stats(cfg["data"]["root"])
    pos_weight = min(stats["centerline_neg_over_pos"], lcfg["pos_weight_cap"])
    model = build_model(mcfg).to(device).to(memory_format=torch.channels_last)
    model.freeze_encoder_bn = True  # batch of 4 < 8
    side = mcfg["deep_supervision_classes"] if mcfg["deep_supervision"] else ()
    criterion = SegLoss(lcfg, mcfg["classes"], pos_weight, side).to(device)
    optimizer = build_optimizer(model, cfg["train"])
    model.train()
    totals = []
    for step in range(1, args.steps + 1):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["train"]["amp"]):
            out = model(image)
        loss, terms = criterion(out, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        totals.append(float(terms["total"]))
        if step == 1 or step % 20 == 0:
            shares = group_shares(terms, lcfg["class_weights"])
            print(f"step {step:4d} total {totals[-1]:.4f} | strap dice {float(terms['dice/strap']):.4f} "
                  f"cldice {float(terms.get('cldice/strap', 0)):.4f} cl_bce {float(terms['centerline_bce']):.4f} | "
                  + " ".join(f"{k} {v:.1%}" for k, v in shares.items()), flush=True)
    print("final total loss:", round(totals[-1], 4))
    print("final terms:", {k: round(float(v), 5) for k, v in terms.items()})

    # Log y-axis: the loss spans ~2 orders of magnitude, so the tail would be flat on a linear scale.
    plot_path = Path(args.plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(totals) + 1), totals)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("total loss (log scale)")
    ax.set_title(f"Overfit test: 4 full views, {args.steps} steps (final {totals[-1]:.4f})")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    print("loss curve saved to", plot_path)


if __name__ == "__main__":
    main()
