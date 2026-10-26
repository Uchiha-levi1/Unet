"""Full-image (768x768) inference, evaluation and overlay rendering.

CLI: python -m src.infer --config configs/turing.yaml --checkpoint runs/turing/best.pt --split test
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data import A22Dataset, gpu_inputs, normalization
from src.metrics import average, image_metrics
from src.model import build_model


def overlay(raw, gt, pred, centerline):
    """One row: image | strap GT | strap prediction (green = TP, red = FP, blue = FN) | centerline prob."""
    gt3 = np.repeat(gt[..., None] * 255, 3, -1).astype(np.uint8)
    err = raw.copy() // 2
    err[pred & gt] = (0, 255, 0)
    err[pred & ~gt] = (255, 0, 0)
    err[~pred & gt] = (0, 0, 255)
    cl = cv2.applyColorMap((centerline * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    row = np.concatenate([raw, gt3, err, cl], axis=1)
    return cv2.resize(row, (row.shape[1] // 2, row.shape[0] // 2), interpolation=cv2.INTER_AREA)


@torch.no_grad()
def evaluate(model, dataset, cfg, amp=True, vis_dir=None, num_vis=0, save_pred_dir=None):
    """Per-image metrics for every mask channel (+ clDice / connectivity / centerline coverage), averaged."""
    model.eval()
    device = next(model.parameters()).device
    classes = cfg["model"]["classes"]
    skel_keys = {c: v["skeleton"] for c, v in cfg["loss"]["cldice"].items()}  # GT skeletons for hard clDice
    strap = classes.index("strap")
    mean, std = normalization(cfg["data"])
    loader = DataLoader(dataset, batch_size=cfg["eval"]["batch_size"], num_workers=cfg["data"]["num_workers"],
                        pin_memory=True)
    results, seen = [], 0
    for batch in loader:
        image, _ = gpu_inputs(batch, device, mean, std, with_targets=False)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and image.is_cuda):
            out = model(image)
        prob = torch.sigmoid(out["mask"].float()).cpu().numpy()
        cl_prob = torch.sigmoid(out["centerline"].float())[:, 0].cpu().numpy()
        pred, gt = prob > cfg["eval"]["threshold"], batch["mask"].numpy().astype(bool)
        skels = {c: batch[k][:, 0].numpy().astype(bool) for c, k in skel_keys.items()}
        for i, sid in enumerate(batch["sample_id"]):
            m = image_metrics(pred[i], gt[i], classes, {c: s[i] for c, s in skels.items()})
            m["sample_id"] = sid
            results.append(m)
            if vis_dir is not None and seen < num_vis:
                Path(vis_dir).mkdir(parents=True, exist_ok=True)
                raw = batch["image"][i].permute(1, 2, 0).numpy()
                vis = overlay(raw, gt[i, strap], pred[i, strap], cl_prob[i])
                cv2.imwrite(str(Path(vis_dir) / f"{sid}.jpg"), vis[..., ::-1])
            if save_pred_dir is not None:
                d = Path(save_pred_dir) / sid
                d.mkdir(parents=True, exist_ok=True)
                for c, p in zip(classes, prob[i]):
                    cv2.imwrite(str(d / f"{c}.png"), (p * 255).astype(np.uint8))
                cv2.imwrite(str(d / "centerline.png"), (cl_prob[i] * 255).astype(np.uint8))
            seen += 1
    summary = average([{k: v for k, v in r.items() if k != "sample_id"} for r in results])
    return summary, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--data-root", default=None, help="override data.root")
    ap.add_argument("--out", default=None, help="output dir (default: <checkpoint dir>/eval_<split>)")
    ap.add_argument("--save-preds", action="store_true", help="also save probability maps as PNG")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    cfg["model"]["pretrained"] = False
    model = build_model(cfg["model"]).cuda().to(memory_format=torch.channels_last)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)  # own files (snapshots hold RNG)
    model.load_state_dict(state["model"])
    out = Path(args.out or Path(args.checkpoint).parent / f"eval_{args.split}")
    out.mkdir(parents=True, exist_ok=True)

    summary, per_image = evaluate(model, A22Dataset(cfg, args.split), cfg, amp=cfg["train"]["amp"],
                                  vis_dir=out / "vis", num_vis=cfg["train"]["num_vis"],
                                  save_pred_dir=out / "pred" if args.save_preds else None)
    (out / "summary.json").write_text(json.dumps({"checkpoint": str(args.checkpoint), "epoch": state.get("epoch"),
                                                  "split": args.split, **summary}, indent=2))
    (out / "per_image.json").write_text(json.dumps(per_image, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
