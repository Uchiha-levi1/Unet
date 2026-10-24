"""Train UNet 3+ on Massachusetts Roads.

CLI: python -m src.train --config configs/default.yaml
"""
import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.config import load_config
from src.data import RoadsEvalDataset, RoadsTrainDataset, build_cache, seed_worker
from src.infer import evaluate
from src.losses import SegLoss
from src.model import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, train_cfg):
    enc = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc}
    rest = [p for p in model.parameters() if id(p) not in enc_ids]  # decoder + heads
    return torch.optim.AdamW([{"params": enc, "lr": train_cfg["lr_encoder"]},
                              {"params": rest, "lr": train_cfg["lr_decoder"]}],
                             weight_decay=train_cfg["weight_decay"])


def warmup_cosine(optimizer, warmup_steps, total_steps):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def set_encoder_trainable(model, trainable):
    for p in model.encoder.parameters():
        p.requires_grad_(trainable)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, amp):
    model.train()  # also re-freezes encoder BN stats if enabled (see UNet3Plus.train)
    sums, n = {}, 0
    for image, mask, centerline in loader:
        image, mask, centerline = (t.cuda(non_blocking=True) for t in (image, mask, centerline))
        image = image.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(image)
        loss, terms = criterion(out, mask, centerline)  # losses run in float32
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        for k, v in terms.items():
            sums[k] = sums.get(k, 0.0) + v
        n += 1
    return {k: v / n for k, v in sums.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    dcfg, mcfg, lcfg, tcfg = cfg["data"], cfg["model"], cfg["loss"], cfg["train"]

    set_seed(cfg["seed"])
    out_dir = Path(tcfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    stats = build_cache(dcfg)
    pos_weight = min(stats["centerline_neg_over_pos"], lcfg["pos_weight_cap"])
    print(f"centerline neg/pos = {stats['centerline_neg_over_pos']:.1f} -> pos_weight = {pos_weight:.1f}")

    train_set = RoadsTrainDataset(dcfg, "train")
    val_set = RoadsEvalDataset(dcfg, "val")
    loader = DataLoader(train_set, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True,
                        num_workers=dcfg["num_workers"], pin_memory=True, persistent_workers=True,
                        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(cfg["seed"]))

    model = build_model(mcfg).cuda().to(memory_format=torch.channels_last)  # ~30% faster convs under AMP
    model.freeze_encoder_bn = tcfg["batch_size"] < 8
    criterion = SegLoss(lcfg, pos_weight).cuda()
    optimizer = build_optimizer(model, tcfg)
    steps_per_epoch = len(loader)
    scheduler = warmup_cosine(optimizer, tcfg["warmup_epochs"] * steps_per_epoch, tcfg["epochs"] * steps_per_epoch)

    writer = SummaryWriter(out_dir / "tb")
    csv_path, csv_writer, csv_file = out_dir / "log.csv", None, None
    best = -1.0
    for epoch in range(1, tcfg["epochs"] + 1):
        set_encoder_trainable(model, epoch > tcfg["freeze_encoder_epochs"])
        t0 = time.time()
        train_terms = train_one_epoch(model, loader, criterion, optimizer, scheduler, tcfg["amp"])
        t_train = time.time() - t0
        val, _ = evaluate(model, val_set, cfg["eval"], amp=tcfg["amp"],
                          vis_dir=out_dir / "vis" / f"epoch_{epoch:03d}", num_vis=tcfg["num_vis"])

        row = {"epoch": epoch, "lr_encoder": optimizer.param_groups[0]["lr"],
               "lr_decoder": optimizer.param_groups[1]["lr"], "time_train_s": round(t_train, 1),
               **{f"train/{k}": v for k, v in train_terms.items()}, **{f"val/{k}": v for k, v in val.items()}}
        for k, v in row.items():
            if k != "epoch":
                writer.add_scalar(k, v, epoch)
        if csv_writer is None:
            csv_file = open(csv_path, "w", newline="")
            csv_writer = csv.DictWriter(csv_file, fieldnames=list(row))
            csv_writer.writeheader()
        csv_writer.writerow(row)
        csv_file.flush()

        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "val": val}
        torch.save(state, out_dir / "last.pt")
        if val["dice"] > best:
            best = val["dice"]
            torch.save(state, out_dir / "best.pt")
        print(f"epoch {epoch:3d} | loss {train_terms['total']:.4f} (dice {train_terms['dice']:.3f} "
              f"focal {train_terms['focal']:.4f} cldice {train_terms['cldice']:.3f} "
              f"cl_bce {train_terms['centerline_bce']:.3f}) | val dice {val['dice']:.4f} iou {val['iou']:.4f} "
              f"cldice {val['cldice']:.4f} cc {val['cc_pred']:.0f}/{val['cc_gt']:.0f} | "
              f"{t_train:.0f}s{' *' if best == val['dice'] else ''}", flush=True)
    writer.close()
    csv_file.close()


if __name__ == "__main__":
    main()
