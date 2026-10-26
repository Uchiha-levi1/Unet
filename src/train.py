"""Train UNet 3+ on the preprocessed A-22 dataset (full 768x768 views).

CLI:        python -m src.train --config configs/turing.yaml [--data-root /tmp/<job>/Data]
Multi-GPU:  torchrun --nproc_per_node=N -m src.train --config ...
A run resumes automatically from the newest epoch_XXX.pt snapshot in train.out_dir.
"""
import argparse
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from src.config import load_config
from src.data import A22Dataset, gpu_inputs, load_stats, normalization, seed_worker
from src.infer import evaluate
from src.losses import SegLoss
from src.model import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
    """(rank, world_size, device). Single process unless launched by torchrun."""
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, torch.device("cuda")
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank)


def build_optimizer(model, train_cfg):
    """AdamW: encoder and decoder(+heads) groups, each split into decayed weights and non-decayed
    BatchNorm/bias parameters (ndim <= 1). Every group carries an "encoder" flag for the schedule."""
    enc_ids = {id(p) for p in model.encoder.parameters()}
    groups = {}
    for p in model.parameters():
        groups.setdefault((id(p) in enc_ids, p.ndim > 1), []).append(p)
    lr = {True: train_cfg["lr_encoder"], False: train_cfg["lr_decoder"]}
    return torch.optim.AdamW([{"params": ps, "lr": lr[enc], "encoder": enc,
                               "weight_decay": train_cfg["weight_decay"] if decay else 0.0}
                              for (enc, decay), ps in sorted(groups.items(), reverse=True)])


def lr_factor(step, start, warmup, total, min_ratio):
    """0 before `start` (frozen), linear warmup over `warmup` steps, then cosine down to min_ratio at `total`."""
    if step < start:
        return 0.0
    s = step - start
    if s < warmup:
        return (s + 1) / warmup
    t = min((s - warmup) / max(1, total - start - warmup), 1.0)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))


def build_scheduler(optimizer, train_cfg, steps_per_epoch):
    """Per-group schedule, stepped per optimizer step: the decoder warms up from step 0; the encoder stays at 0
    while frozen and warms up right after it is unfrozen. Both decay to min_lr_ratio at the last step."""
    total = train_cfg["epochs"] * steps_per_epoch
    warmup = train_cfg["warmup_epochs"] * steps_per_epoch
    enc_start = train_cfg["freeze_encoder_epochs"] * steps_per_epoch
    fns = [lambda s, start=(enc_start if g["encoder"] else 0):
           lr_factor(s, start, warmup, total, train_cfg["min_lr_ratio"]) for g in optimizer.param_groups]
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fns)


def set_encoder_trainable(model, trainable):
    for p in model.encoder.parameters():
        p.requires_grad_(trainable)


def latest_snapshot(out_dir):
    snaps = sorted(Path(out_dir).glob("epoch_*.pt"))
    return snaps[-1] if snaps else None


def save_snapshot(path, model, optimizer, scheduler, epoch, best):
    """Full resume point. Written to a temp file first so a job killed mid-save leaves no broken snapshot."""
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "epoch": epoch, "best": best,
             "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                     "cuda": torch.cuda.get_rng_state_all()}}
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def restore_rng(rng):
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if len(rng["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(rng["cuda"])


def train_one_epoch(model, loader, criterion, optimizer, scheduler, train_cfg, device, mean, std):
    model.train()  # also re-freezes encoder BN stats if enabled (see UNet3Plus.train)
    accum, clip = train_cfg["grad_accum"], train_cfg["grad_clip"]
    n_steps = len(loader) // accum  # an incomplete last accumulation group is dropped
    params = list(model.parameters())
    sums, grad_norms = {}, []
    batches = iter(loader)
    for _ in range(n_steps):
        for micro in range(accum):
            image, targets = gpu_inputs(next(batches), device, mean, std)
            # DDP: skip the gradient all-reduce on all but the last micro-batch of an accumulation group.
            sync = nullcontext() if micro == accum - 1 or not isinstance(model, DDP) else model.no_sync()
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=train_cfg["amp"]):
                    out = model(image)
                loss, terms = criterion(out, targets)  # losses run in float32
                (loss / accum).backward()
            for k, v in terms.items():  # kept on the GPU: no per-step synchronization
                sums[k] = sums.get(k, 0.0) + v / accum
        grad_norms.append(torch.nn.utils.clip_grad_norm_([p for p in params if p.grad is not None], clip))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    norms = torch.stack(grad_norms)
    out = {k: float(v) / n_steps for k, v in sums.items()}
    out["grad_norm"] = float(norms.mean())          # before clipping
    out["grad_clipped_frac"] = float((norms > clip).float().mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default=None, help="override data.root (e.g. data staged on node-local /tmp)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    dcfg, mcfg, lcfg, tcfg = cfg["data"], cfg["model"], cfg["loss"], cfg["train"]

    rank, world, device = setup_distributed()
    main_proc = rank == 0
    set_seed(cfg["seed"] + rank)
    torch.set_float32_matmul_precision("high")  # TF32 for the float32 parts (losses, optimizer)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    out_dir = Path(tcfg["out_dir"])
    if main_proc:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    stats = load_stats(dcfg["root"])
    pos_weight = min(stats["centerline_neg_over_pos"], lcfg["pos_weight_cap"])
    mean, std = normalization(dcfg)

    train_set = A22Dataset(cfg, "train", train=True)
    sampler = DistributedSampler(train_set, shuffle=True, drop_last=True, seed=cfg["seed"]) if world > 1 else None
    generator = torch.Generator()
    workers = dcfg["num_workers"]
    loader = DataLoader(train_set, batch_size=tcfg["batch_size"], shuffle=sampler is None, sampler=sampler,
                        drop_last=True, num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
                        prefetch_factor=2 if workers > 0 else None, worker_init_fn=seed_worker, generator=generator)
    val_set = A22Dataset(cfg, "validation",
                         subset=tcfg["val_subset_size"] if tcfg["val_mode"] == "subset" else None)

    net = build_model(mcfg).to(device).to(memory_format=torch.channels_last)  # ~30% faster convs under AMP
    net.freeze_encoder_bn = tcfg["freeze_encoder_bn"]
    side = mcfg["deep_supervision_classes"] if mcfg["deep_supervision"] else ()
    criterion = SegLoss(lcfg, mcfg["classes"], pos_weight, side).to(device)
    optimizer = build_optimizer(net, tcfg)
    steps_per_epoch = len(loader) // tcfg["grad_accum"]
    scheduler = build_scheduler(optimizer, tcfg, steps_per_epoch)
    enc_group, dec_group = (next(i for i, g in enumerate(optimizer.param_groups) if g["encoder"] == e)
                            for e in (True, False))

    start_epoch, best = 1, -1.0
    snap = latest_snapshot(out_dir)
    if snap is not None:
        state = torch.load(snap, map_location="cpu", weights_only=False)  # own file: includes RNG states
        net.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"])
        start_epoch, best = state["epoch"] + 1, state["best"]
        if main_proc:
            print(f"resumed from {snap} (epoch {state['epoch']}, best strap dice {best:.4f})", flush=True)
    if main_proc:
        print(f"{len(train_set)} train views, {steps_per_epoch} optimizer steps/epoch, "
              f"{len(val_set)} validation views ({tcfg['val_mode']}), pos_weight {pos_weight:.1f}", flush=True)

    writer = SummaryWriter(out_dir / "tb") if main_proc else None
    log_path = out_dir / "log.csv"
    rows = []
    if main_proc and snap is not None and log_path.exists():
        rows = [r for r in pd.read_csv(log_path).to_dict("records") if r["epoch"] < start_epoch]
    model, encoder_trainable, first_val = net, None, True
    for epoch in range(start_epoch, tcfg["epochs"] + 1):
        trainable = epoch > tcfg["freeze_encoder_epochs"]
        if trainable != encoder_trainable:
            set_encoder_trainable(net, trainable)
            # DDP registers the trainable parameters when it is built, so re-wrap when that set changes.
            model = DDP(net, device_ids=[device.index]) if world > 1 else net
            encoder_trainable = trainable
        if sampler is not None:
            sampler.set_epoch(epoch)
        else:
            generator.manual_seed(cfg["seed"] + epoch)  # shuffle order depends only on the epoch (resume-safe)

        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()
        train_terms = train_one_epoch(model, loader, criterion, optimizer, scheduler, tcfg, device, mean, std)
        row = {"epoch": epoch, "lr_encoder": optimizer.param_groups[enc_group]["lr"],
               "lr_decoder": optimizer.param_groups[dec_group]["lr"], "time_train_s": round(time.time() - t0, 1),
               "gpu_mem_peak_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
               **{f"train/{k}": v for k, v in train_terms.items()}}

        validated = False
        if main_proc and (epoch % tcfg["val_every"] == 0 or epoch == tcfg["epochs"]):
            t1 = time.time()
            val, _ = evaluate(net, val_set, cfg, amp=tcfg["amp"],
                              vis_dir=out_dir / "vis" / f"epoch_{epoch:03d}", num_vis=tcfg["num_vis"])
            row["time_val_s"] = round(time.time() - t1, 1)
            if first_val:
                print(f"first validation ({tcfg['val_mode']}, {len(val_set)} images) took {row['time_val_s']:.0f}s",
                      flush=True)
                first_val = False
            row.update({f"val/{k}": v for k, v in val.items()})
            validated = val["dice/strap"] > best
            if validated:
                best = val["dice/strap"]
                torch.save({"model": net.state_dict(), "epoch": epoch, "val": val}, out_dir / "best.pt")
        if world > 1:
            dist.barrier()  # other ranks wait while rank 0 validates

        if main_proc:
            for k, v in row.items():
                if k != "epoch":
                    writer.add_scalar(k, v, epoch)
            writer.flush()
            rows.append(row)
            pd.DataFrame(rows).to_csv(log_path, index=False)
            msg = (f"epoch {epoch:3d} | loss {row['train/total']:.4f} (strap dice {row['train/dice/strap']:.3f}"
                   f" cldice {row.get('train/cldice/strap', 0):.3f} cl_bce {row['train/centerline_bce']:.3f})"
                   f" | grad {row['train/grad_norm']:.2f} clip {row['train/grad_clipped_frac']:.0%}"
                   f" | {row['time_train_s']:.0f}s {row['gpu_mem_peak_gb']:.1f}GB")
            if "val/dice/strap" in row:
                msg += (f" | val strap dice {row['val/dice/strap']:.4f} cldice {row['val/cldice/strap']:.4f}"
                        f" cov {row['val/strap_cl_coverage']:.3f} cc {row['val/cc_pred']:.0f}/{row['val/cc_gt']:.0f}"
                        f" cord dice {row['val/dice/cord_or_tie']:.3f} hw dice {row['val/dice/hardware']:.3f}"
                        f" ({row['time_val_s']:.0f}s){' *' if validated else ''}")
            print(msg, flush=True)
            if epoch % tcfg["snapshot_every"] == 0 or epoch == tcfg["epochs"]:
                save_snapshot(out_dir / f"epoch_{epoch:03d}.pt", net, optimizer, scheduler, epoch, best)

    if main_proc:
        writer.close()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
