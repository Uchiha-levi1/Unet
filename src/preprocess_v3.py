"""Preprocess single_a22_physical_v13 (single A-22, single label per pixel, MS COCO backgrounds) into Data_v3/.

Per sample (<split>/<family>/<view>/<state>/), written to <out>/npz/<sample_id>.npz:
  image          (768,768,3) uint8, rgb.png as is (background kept)
  mask           (8,768,768) uint8 0/1, one-hot of class_map in CHANNELS order (data_tag dropped: its pixels are
                 all-zero), attachment_hardware = convex hull per buckle object, overriding what lies inside it
  data_tag       (768,768) uint8 0/1, kept separately (not a channel) in case it is needed as an ignore region
  webbing_cl_1px, webbing_cl_3px, cord_cl_1px, cord_cl_3px   (768,768) uint8 0/1 visible centerlines
Centerline per registry component_id (a strap and its selvedge objects together): the component's pixels,
closed by CLOSE_PX but only into webbing/cord/tape/hardware pixels (bridges short gaps under crossings),
skeletonized, spurs shorter than SPUR_PX removed.

A sample is used only if all of SAMPLE_FILES exist (missing samples are skipped, families are not).
Per family a worker writes npz files and then <out>/meta/<family>.json (marks the family done; resumable).
--finalize builds index.csv, skipped.csv and stats.json (train split) from the meta files.

CLI:
  python -m src.preprocess_v3 --src SRC --out Data_v3 [--shard i --nshards N] [--only f000000 ...] [--png]
  python -m src.preprocess_v3 --src SRC --out Data_v3 --finalize
"""
import argparse
import csv
import json
import os
import shutil
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from src import preprocess as P  # COLORS (overlay)

# Source class_map values.
SRC_CLASSES = {1: "cargo_cover", 2: "webbing", 3: "cord_or_tie", 4: "attachment_hardware", 5: "tape",
               6: "skid_board", 7: "honeycomb", 8: "parachute_bag", 9: "data_tag"}
WEBBING, CORD, HARDWARE, TAPE, DATA_TAG = 2, 3, 4, 5, 9
CHANNELS = ["webbing", "cargo_cover", "cord_or_tie", "attachment_hardware", "tape", "skid_board", "honeycomb",
            "parachute_bag"]
LINE_KEYS = ["webbing_cl_1px", "webbing_cl_3px", "cord_cl_1px", "cord_cl_3px"]
SAMPLE_FILES = ["rgb.png", "class_map.png", "object_map.png", "annotations.json", "background.json"]
SPLITS = ["train", "validation", "test"]
SIZE = 768
CLOSE_PX = 7                          # gap-bridging closing kernel (square), px
SPUR_PX = {WEBBING: 8, CORD: 5}       # skeleton end branches shorter than this are removed
POS_WEIGHT_CAP = 30.0

# Overlay colors: reuse the earlier palette (webbing = old strap, attachment_hardware = old hardware).
COLORS = {**{k: P.COLORS[k] for k in CHANNELS if k in P.COLORS},
          "webbing": P.COLORS["strap"], "attachment_hardware": P.COLORS["hardware"]}
BLOBS = ["cargo_cover", "skid_board", "honeycomb", "parachute_bag"]
THIN = ["tape", "cord_or_tie", "webbing", "attachment_hardware"]

_W = {}  # per-worker settings


def load_components(src):
    """{class_id: {component_id: np.array(object_ids)}} for webbing and cord, from registry.json."""
    reg = json.loads((src / "registry.json").read_text())
    assert {int(k): v for k, v in reg["classes"].items() if k != "0"} == SRC_CLASSES, reg["classes"]
    comps = {WEBBING: {}, CORD: {}}
    for o in reg["objects"]:
        if o["class_id"] in comps:
            comps[o["class_id"]].setdefault(o["component_id"], []).append(o["object_id"])
    return {c: {k: np.array(v) for k, v in d.items()} for c, d in comps.items()}


def _init_worker(src, out, png):
    cv2.setNumThreads(1)
    _W.update(src=src, out=out, png=png, comps=load_components(src))


# --- masks -----------------------------------------------------------------------------------------

def hardware_hulls(cm, om):
    """Union of convex hulls of every attachment_hardware object (buckles are open frames)."""
    out = np.zeros(cm.shape, np.uint8)
    for i in np.unique(om[cm == HARDWARE]):
        ys, xs = np.nonzero((om == i) & (cm == HARDWARE))
        pts = np.c_[xs, ys].astype(np.int32)
        cv2.fillPoly(out, [cv2.convexHull(pts)], 1) if len(pts) >= 3 else out.__setitem__((ys, xs), 1)
    return out.astype(bool)


def build_mask(cm, hw):
    mask = np.stack([cm == next(k for k, v in SRC_CLASSES.items() if v == c) for c in CHANNELS])
    mask[:, hw] = False
    mask[CHANNELS.index("attachment_hardware")] = hw
    return mask.astype(np.uint8)


# --- centerlines -----------------------------------------------------------------------------------

NB = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


def prune_spurs(skel, min_len):
    """Remove branches that run from an endpoint to a junction and are shorter than min_len px."""
    skel = skel.copy()
    nb = ndi.convolve(skel.astype(np.uint8), NB, mode="constant") * skel
    junction = skel & (nb >= 3)
    if not junction.any():
        return skel
    seg, n = ndi.label(skel & ~junction, structure=np.ones((3, 3)))
    if n == 0:
        return skel
    ends = skel & (nb == 1)
    near_j = ndi.binary_dilation(junction, np.ones((3, 3)))
    size = ndi.sum(np.ones_like(seg), seg, np.arange(1, n + 1))
    has_end = ndi.maximum(ends, seg, np.arange(1, n + 1)) > 0
    touches_j = ndi.maximum(near_j, seg, np.arange(1, n + 1)) > 0
    drop = np.nonzero(has_end & touches_j & (size < min_len))[0] + 1
    skel[np.isin(seg, drop)] = False
    return skeletonize(skel)  # re-thin junction pixels left behind


def centerline(cm, om, allowed, comps, cls):
    """1 px visible centerline of every component of one class (webbing or cord)."""
    out = np.zeros(cm.shape, bool)
    present = np.unique(om[cm == cls])
    k = np.ones((CLOSE_PX, CLOSE_PX), np.uint8)
    pad = CLOSE_PX + SPUR_PX[cls]
    for ids in comps[cls].values():
        ids = ids[np.isin(ids, present)]
        if not len(ids):
            continue
        m = np.isin(om, ids) & (cm == cls)
        ys, xs = np.nonzero(m)
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, SIZE)
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, SIZE)
        sub = m[y0:y1, x0:x1]
        closed = cv2.morphologyEx(sub.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
        sub = sub | (closed & allowed[y0:y1, x0:x1])
        out[y0:y1, x0:x1] |= prune_spurs(skeletonize(sub), SPUR_PX[cls])
    return out.astype(np.uint8)


def overlay(rgb, mask, web_cl, cord_cl):
    """RGB | RGB with channels painted (blobs blended, thin solid), webbing centerline white, cord cyan."""
    vis = rgb.astype(np.float32)
    for c in BLOBS:
        m = mask[CHANNELS.index(c)].astype(bool)
        vis[m] = 0.55 * vis[m] + 0.45 * np.array(COLORS[c])
    vis = vis.astype(np.uint8)
    for c in THIN:
        vis[mask[CHANNELS.index(c)].astype(bool)] = COLORS[c]
    vis[cord_cl.astype(bool)] = (0, 255, 255)
    vis[web_cl.astype(bool)] = (255, 255, 255)
    return np.concatenate([rgb, vis], axis=1)


# --- per-family job --------------------------------------------------------------------------------

def sample_dirs(fam_dir):
    return sorted(d for v in fam_dir.iterdir() if v.is_dir() for d in v.iterdir() if d.is_dir())


def process_family(job):
    """All complete samples of one family. Skipped if its meta file exists. Returns (family, n_written, s)."""
    split, fam = job
    src, out, png, comps = _W["src"], _W["out"], _W["png"], _W["comps"]
    meta_path = out / "meta" / f"{fam}.json"
    if meta_path.exists():
        return fam, 0, 0.0
    t0 = time.time()
    meta = {"split": split, "samples": {}, "skipped": {}}
    for d in sample_dirs(src / split / fam):
        view, state = d.parent.name, d.name
        sid = f"{fam}__{view}__{state}"
        missing = [f for f in SAMPLE_FILES if not (d / f).exists()]
        if missing:
            meta["skipped"][sid] = missing
            continue
        rd = lambda f: cv2.imread(str(d / f), cv2.IMREAD_UNCHANGED)
        rgb, cm, om = rd("rgb.png"), rd("class_map.png"), rd("object_map.png")
        if rgb is None or cm is None or om is None:  # unreadable / truncated file
            meta["skipped"][sid] = ["unreadable"]
            continue
        rgb = rgb[..., ::-1]
        ann = json.loads((d / "annotations.json").read_text())
        bg = json.loads((d / "background.json").read_text())
        assert rgb.shape == (SIZE, SIZE, 3) and cm.shape == om.shape == (SIZE, SIZE), sid

        hw = hardware_hulls(cm, om)
        mask = build_mask(cm, hw)
        allowed = np.isin(cm, (WEBBING, CORD, TAPE)) | hw
        wcl = centerline(cm, om, allowed, comps, WEBBING)
        ccl = centerline(cm, om, allowed, comps, CORD)
        k3 = np.ones((3, 3), np.uint8)
        lines = {"webbing_cl_1px": wcl, "webbing_cl_3px": cv2.dilate(wcl, k3),
                 "cord_cl_1px": ccl, "cord_cl_3px": cv2.dilate(ccl, k3)}
        data_tag = (cm == DATA_TAG).astype(np.uint8)

        if png:
            pd_ = out / "png" / sid
            pd_.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(d / "rgb.png", pd_ / "rgb.png")
            for c, m in zip(CHANNELS, mask):
                cv2.imwrite(str(pd_ / f"{c}.png"), m * 255)
            for c, m in lines.items():
                cv2.imwrite(str(pd_ / f"{c}.png"), m * 255)
            cv2.imwrite(str(pd_ / "data_tag.png"), data_tag * 255)
            cv2.imwrite(str(pd_ / "overlay.png"), overlay(rgb, mask, wcl, ccl)[..., ::-1])

        npz, tmp = out / "npz" / f"{sid}.npz", out / "npz" / f"{sid}.npz.tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, image=np.ascontiguousarray(rgb), mask=mask, data_tag=data_tag, **lines)
        os.replace(tmp, npz)
        meta["samples"][sid] = {"view": view, "state": state, "observable": bool(ann["observable"]),
                                "coco_id": bg["coco_id"]}
    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta))
    os.replace(tmp, meta_path)  # written last: marks the family done
    return fam, len(meta["samples"]), time.time() - t0


def families(src):
    return sorted((s, f.name) for s in SPLITS for f in (src / s).iterdir() if f.is_dir())


def run(args):
    jobs = families(args.src)
    if args.only:
        unknown = set(args.only) - {f for _, f in jobs}
        assert not unknown, f"unknown families: {sorted(unknown)}"
        jobs = [j for j in jobs if j[1] in set(args.only)]
    else:
        jobs = jobs[args.shard::args.nshards]
    print(f"shard {args.shard}/{args.nshards}: {len(jobs)} families, {args.workers} workers", flush=True)
    t0, n, busy = time.time(), 0, 0.0
    with Pool(args.workers, initializer=_init_worker, initargs=(args.src, args.out, args.png)) as pool:
        for k, (fam, m, sec) in enumerate(pool.imap_unordered(process_family, jobs), 1):
            n, busy = n + m, busy + sec
            if k % 20 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} families, {n} samples written, {time.time() - t0:.0f}s wall"
                      + (f", {busy / n:.2f}s per sample per core" if n else ""), flush=True)


# --- finalize --------------------------------------------------------------------------------------

def _stats_job(path):
    with np.load(path) as z:
        img = z["image"].reshape(-1, 3).astype(np.float64) / 255
        return (img.sum(0), (img ** 2).sum(0), z["mask"].reshape(len(CHANNELS), -1).sum(1),
                int(z["data_tag"].sum()), np.array([int(z[k].sum()) for k in LINE_KEYS]))


def compute_stats(out, rows, workers):
    paths = [out / r["npz_path"] for r in rows if r["split"] == "train"]
    if not paths:
        return None
    with Pool(workers) as pool:
        parts = pool.map(_stats_job, paths, chunksize=8)
    s, s2, mpix, tag, lines = (sum(p[i] for p in parts) for i in range(5))
    n = len(paths) * SIZE * SIZE
    mean = s / n
    frac = {k: v / n for k, v in zip(LINE_KEYS, lines)}
    nop = {k: (n - lines[LINE_KEYS.index(k)]) / max(lines[LINE_KEYS.index(k)], 1)
           for k in ("webbing_cl_3px", "cord_cl_3px")}
    return {"n_train_views": len(paths), "channels": CHANNELS, "line_keys": LINE_KEYS,
            "image_mean_rgb": mean.round(6).tolist(), "image_std_rgb": np.sqrt(s2 / n - mean ** 2).round(6).tolist(),
            "mask_pixel_fraction": {k: float(v / n) for k, v in zip(CHANNELS, mpix)},
            "data_tag_pixel_fraction": tag / n,
            "line_fraction": {k: float(v) for k, v in frac.items()},
            "centerline_neg_over_pos": {k: float(v) for k, v in nop.items()},  # from the 3 px targets
            "pos_weight_cap": POS_WEIGHT_CAP,
            "pos_weight": {k: float(min(v, POS_WEIGHT_CAP)) for k, v in nop.items()}}


def finalize(args):
    out = args.out
    jobs = families(args.src)
    rows, skipped, not_done = [], [], 0
    for split, fam in jobs:
        mp = out / "meta" / f"{fam}.json"
        if not mp.exists():
            not_done += 1
            continue
        meta = json.loads(mp.read_text())
        for sid, m in meta["samples"].items():
            rows.append({"sample_id": sid, "family_id": fam, "view": m["view"], "defect": m["state"],
                         "split": split, "defect_visible": int(m["observable"]), "coco_id": m["coco_id"],
                         "npz_path": f"npz/{sid}.npz"})
        skipped += [(sid, split, ";".join(v)) for sid, v in meta["skipped"].items()]
    rows.sort(key=lambda r: r["sample_id"])
    with open(out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "skipped.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "split", "missing"])
        w.writerows(sorted(skipped))
    print(f"{len(rows)} samples from {len(jobs) - not_done} families; {len(skipped)} samples skipped; "
          f"{not_done} families not done")
    for s in SPLITS:
        r = [x for x in rows if x["split"] == s]
        print(f"  {s}: {len({x['family_id'] for x in r})} families, {len(r)} samples, "
              f"per state {dict(sorted(Counter(x['defect'] for x in r).items()))}, "
              f"skipped {sum(1 for x in skipped if x[1] == s)}")
    stats = compute_stats(out, rows, args.workers)
    if stats is not None:
        (out / "stats.json").write_text(json.dumps(stats, indent=2))
        print("stats:", json.dumps({k: stats[k] for k in ("n_train_views", "mask_pixel_fraction", "line_fraction",
                                                           "pos_weight")}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, type=Path, help="dataset folder (read only)")
    ap.add_argument("--out", default="Data_v3", type=Path)
    ap.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--only", nargs="*", help="process only these families (test run)")
    ap.add_argument("--png", action="store_true", help="also write PNGs + overlay per sample (test run)")
    ap.add_argument("--finalize", action="store_true", help="build index.csv, skipped.csv, stats.json")
    args = ap.parse_args()
    args.src, args.out = args.src.resolve(), args.out.resolve()
    assert 0 <= args.shard < args.nshards
    for sub in ("npz", "meta"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    finalize(args) if args.finalize else run(args)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
