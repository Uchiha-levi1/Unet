"""Sliding-window inference, evaluation and overlay rendering.

CLI: python -m src.infer --config configs/default.yaml --checkpoint runs/default/best.pt --split test
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import load_config
from src.data import RoadsEvalDataset
from src.metrics import average, image_metrics
from src.model import build_model


def center_weight(tile, device):
    """2D Gaussian (sigma = tile/8, as in nnU-Net) so tile centers dominate the blend over tile borders."""
    r = torch.arange(tile, device=device, dtype=torch.float32) - (tile - 1) / 2
    g = torch.exp(-(r ** 2) / (2 * (tile / 8) ** 2))
    w = g[:, None] * g[None, :]
    return (w / w.max()).clamp_min(1e-3)  # floor keeps image-border pixels (single coverage) well defined


def tile_starts(length, tile, stride):
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)  # last tile flush with the border
    return starts


@torch.no_grad()
def sliding_window_predict(model, image, tile=512, overlap=0.25, tile_batch=8, amp=True):
    """image: (1, 3, H, W) normalized tensor on the model's device.
    Returns sigmoid probabilities {'mask': (C, H, W), 'centerline': (1, H, W)}."""
    _, _, h, w = image.shape
    # Reflect-pad so the padded size is a multiple of 32 and at least one tile.
    ph = max(tile, -(-h // 32) * 32) - h
    pw = max(tile, -(-w // 32) * 32) - w
    x = F.pad(image, (0, pw, 0, ph), mode="reflect")
    H, W = x.shape[-2:]
    stride = int(tile * (1 - overlap))
    weight = center_weight(tile, x.device)
    coords = [(y, x0) for y in tile_starts(H, tile, stride) for x0 in tile_starts(W, tile, stride)]

    acc, norm = {}, torch.zeros(1, H, W, device=x.device)
    for b in range(0, len(coords), tile_batch):
        chunk = coords[b:b + tile_batch]
        batch = torch.cat([x[..., y:y + tile, x0:x0 + tile] for y, x0 in chunk])
        batch = batch.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and x.is_cuda):
            out = model(batch)
        for k in ("mask", "centerline"):
            prob = torch.sigmoid(out[k].float())
            if k not in acc:
                acc[k] = torch.zeros(prob.shape[1], H, W, device=x.device)
            for p, (y, x0) in zip(prob, chunk):
                acc[k][:, y:y + tile, x0:x0 + tile] += p * weight
        for y, x0 in chunk:
            norm[:, y:y + tile, x0:x0 + tile] += weight
    return {k: (v / norm)[:, :h, :w] for k, v in acc.items()}


def overlay(raw, gt, pred, centerline):
    """One row: image | ground truth | prediction (green = TP, red = FP, blue = FN) | centerline prob."""
    gt3 = np.repeat(gt[..., None] * 255, 3, -1).astype(np.uint8)
    err = raw.copy() // 2
    err[pred & gt] = (0, 255, 0)
    err[pred & ~gt] = (255, 0, 0)
    err[~pred & gt] = (0, 0, 255)
    cl = cv2.applyColorMap((centerline * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    row = np.concatenate([raw, gt3, err, cl], axis=1)
    return cv2.resize(row, (row.shape[1] // 2, row.shape[0] // 2), interpolation=cv2.INTER_AREA)


def evaluate(model, dataset, eval_cfg, amp=True, vis_dir=None, num_vis=0, save_pred_dir=None):
    """Per-image metrics on the first mask channel (strap), averaged over images."""
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=1, num_workers=2)
    results = []
    for i, item in enumerate(loader):
        probs = sliding_window_predict(model, item["image"].to(device), eval_cfg["tile"],
                                       eval_cfg["overlap"], eval_cfg["tile_batch"], amp)
        prob = probs["mask"][0].cpu().numpy()
        pred = prob > eval_cfg["threshold"]
        gt = item["mask"][0].numpy()
        m = image_metrics(pred, gt)
        m["image_id"] = item["image_id"][0]
        results.append(m)
        if vis_dir is not None and i < num_vis:
            Path(vis_dir).mkdir(parents=True, exist_ok=True)
            vis = overlay(item["raw"][0].numpy(), gt, pred, probs["centerline"][0].cpu().numpy())
            cv2.imwrite(str(Path(vis_dir) / f"{m['image_id']}.jpg"), vis[..., ::-1])
        if save_pred_dir is not None:
            Path(save_pred_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(save_pred_dir) / f"{m['image_id']}.png"), (prob * 255).astype(np.uint8))
    summary = average([{k: v for k, v in r.items() if k != "image_id"} for r in results])
    return summary, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", default=None, help="output dir (default: <checkpoint dir>/eval_<split>)")
    ap.add_argument("--save-preds", action="store_true", help="also save probability maps as PNG")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["model"]["pretrained"] = False
    model = build_model(cfg["model"]).cuda().to(memory_format=torch.channels_last)
    state = torch.load(args.checkpoint, map_location="cuda")
    model.load_state_dict(state["model"])
    out = Path(args.out or Path(args.checkpoint).parent / f"eval_{args.split}")
    out.mkdir(parents=True, exist_ok=True)

    summary, per_image = evaluate(model, RoadsEvalDataset(cfg["data"], args.split), cfg["eval"],
                                  amp=cfg["train"]["amp"], vis_dir=out / "vis", num_vis=cfg["train"]["num_vis"],
                                  save_pred_dir=out / "pred" if args.save_preds else None)
    (out / "summary.json").write_text(json.dumps({"checkpoint": str(args.checkpoint), "epoch": state.get("epoch"),
                                                  "split": args.split, **summary}, indent=2))
    (out / "per_image.json").write_text(json.dumps(per_image, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
