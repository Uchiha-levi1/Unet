"""Run a checkpoint on plain images (no ground truth), e.g. real photos already cropped/resized to 768x768.

Writes per image: <out>/<name>_overlay.png (image | predicted classes painted, strap centerline white) and
<out>/<name>/<class>.png probability maps (0-255), plus centerline.png.

CLI: python -m src.predict_images --config configs/v1_finetune.yaml --checkpoint runs/v1_finetune/best.pt
                                  --images runs/a22_real/input_768 --out runs/a22_real/pred_r1
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src import preprocess as P
from src.config import load_config
from src.data import normalization
from src.model import build_model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", required=True, type=Path, help="folder of RGB images (sides divisible by 32)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["model"]["pretrained"] = False
    classes = cfg["model"]["classes"]
    assert classes == P.CHANNELS, "overlay colors assume the preprocessing channel order"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]).to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    mean, std = (torch.tensor(v).view(1, 3, 1, 1) for v in normalization(cfg["data"]))
    thr = cfg["eval"]["threshold"]
    args.out.mkdir(parents=True, exist_ok=True)
    for path in sorted(p for p in args.images.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")):
        rgb = cv2.imread(str(path), cv2.IMREAD_COLOR)[..., ::-1].copy()
        assert rgb.shape[0] % 32 == 0 and rgb.shape[1] % 32 == 0, path
        x = ((torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255 - mean) / std).to(device)
        out = model(x)
        prob = torch.sigmoid(out["mask"])[0].cpu().numpy()
        cl = torch.sigmoid(out["centerline"])[0, 0].cpu().numpy()
        mask = (prob > thr).astype(np.uint8)
        vis = P.overlay(rgb, mask, (cl > thr).astype(np.uint8), np.zeros_like(mask[0]))
        cv2.imwrite(str(args.out / f"{path.stem}_overlay.png"), vis[..., ::-1])
        d = args.out / path.stem
        d.mkdir(exist_ok=True)
        for c, p in zip(classes, prob):
            cv2.imwrite(str(d / f"{c}.png"), (p * 255).astype(np.uint8))
        cv2.imwrite(str(d / "centerline.png"), (cl * 255).astype(np.uint8))
        print(path.name, " ".join(f"{c}={100 * m.mean():.1f}%" for c, m in zip(classes, mask)), flush=True)


if __name__ == "__main__":
    main()
