"""Pixel-level class x class confusion matrix (11 x 11 incl. background) of a checkpoint on one split.

The model is multi-label (channels overlap: strap includes strap hardware and continues under cord/tape), so each
pixel gets one label:
  GT:   the first present channel in PRIORITY order (thin / small classes first), else background
  pred: the channel with the highest probability among those > threshold, else background

CLI: python -m src.confusion --config configs/v1_finetune.yaml --checkpoint runs/v1_finetune/best.pt --split test
     [--data-root Data_v1] [--out runs/v1_finetune/confusion_test.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data import A22Dataset, gpu_inputs, normalization
from src.model import build_model

PRIORITY = ["hardware", "cord_or_tie", "tape", "data_tag", "pack_band", "strap", "skid_board", "honeycomb",
            "parachute_bag", "cargo_cover"]


@torch.no_grad()
def confusion(model, dataset, cfg, amp=True):
    classes = cfg["model"]["classes"]
    labels = ["background"] + PRIORITY
    order = torch.tensor([classes.index(c) for c in PRIORITY])  # channel index for each priority label
    n = len(labels)
    device = next(model.parameters()).device
    mean, std = normalization(cfg["data"])
    loader = DataLoader(dataset, batch_size=cfg["eval"]["batch_size"], num_workers=cfg["data"]["num_workers"],
                        pin_memory=True)
    cm = torch.zeros(n * n, dtype=torch.int64, device=device)
    thr = cfg["eval"]["threshold"]
    model.eval()
    for batch in loader:
        image, _ = gpu_inputs(batch, device, mean, std, with_targets=False)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(image)
        prob = torch.sigmoid(out["mask"].float())[:, order.to(device)]          # (B, 10, H, W) in PRIORITY order
        gt = batch["mask"].to(device)[:, order.to(device)].bool()
        # GT: first present channel in priority order (+1 for background at index 0).
        gt_lab = torch.where(gt.any(1), gt.float().argmax(1) + 1, torch.zeros_like(gt[:, 0], dtype=torch.long))
        # Pred: most probable channel above threshold.
        p = torch.where(prob > thr, prob, torch.zeros_like(prob))
        pred_lab = torch.where(p.amax(1) > 0, p.argmax(1) + 1, torch.zeros_like(gt_lab))
        cm += torch.bincount((gt_lab * n + pred_lab).flatten(), minlength=n * n)
    return labels, cm.view(n, n).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default=None, help="default: <checkpoint dir>/confusion_<split>.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    cfg["model"]["pretrained"] = False
    model = build_model(cfg["model"]).cuda().to(memory_format=torch.channels_last)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    labels, cm = confusion(model, A22Dataset(cfg, args.split), cfg, amp=cfg["train"]["amp"])
    out = Path(args.out or Path(args.checkpoint).parent / f"confusion_{args.split}.json")
    out.write_text(json.dumps({"checkpoint": args.checkpoint, "epoch": state.get("epoch"), "split": args.split,
                               "labels": labels, "rows": "ground truth", "cols": "prediction",
                               "matrix": cm.tolist()}, indent=1))
    rec = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    print("row-normalized (recall), %:")
    print(" " * 14 + " ".join(f"{l[:6]:>6s}" for l in labels))
    for l, r in zip(labels, rec):
        print(f"{l[:13]:13s} " + " ".join(f"{100 * v:6.1f}" for v in r))


if __name__ == "__main__":
    main()
