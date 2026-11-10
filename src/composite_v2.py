"""Build Data_v2: Data_v1 with every image recomposited onto a random photo from bg/ (see exploration/run2/).

Per view (seeded by sample_id): a photo from the view's split pool (bg/catalog.csv), a random square crop
(side in [max(0.6*short, min(short, 768)), short], random position), horizontal flip with p=0.5, resized to 768,
then image = a*foreground + (1-a)*background with the straight-alpha foreground.png of the foregrounds dataset.
No shadow, no exposure change. Masks and centerlines are copied unchanged from the Data_v1 npz (identical GT).

Photos are cached once as bg/cache/<background_id>.npy (RGB uint8, short side capped at 2048 px; a 60% crop
of 2048 px is still > 768, so output quality is unchanged) and memory-mapped by all workers.

Per inspection a worker writes <out>/npz/<sample_id>.npz for its 8 views, then <out>/meta/<inspection_id>.json
(per-view background choice + QA); the meta file marks the inspection done (resumable).
--finalize writes index.csv (Data_v1 columns, background_id replaced, + bg_crop_box, bg_flip), copies qa.csv and
defect_states.json, and recomputes stats.json (train split).

CLI:
  python -m src.composite_v2 --fg FOREGROUNDS --v1 Data_v1 --bg bg --out Data_v2 [--shard i --nshards N]
                             [--only ID ...] [--png] [--workers N]
  python -m src.composite_v2 --fg FOREGROUNDS --v1 Data_v1 --bg bg --out Data_v2 --finalize [--workers N]
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from src import preprocess as P  # SIZE, overlay, compute_stats, sample_id

CACHE_MAX_SHORT = 2048
MIN_CROP_FRAC = 0.6
SEED = 0

_W = {}  # per-worker settings


# --- background photos -------------------------------------------------------------------------------

def read_catalog(bg):
    with open(bg / "catalog.csv") as f:
        return list(csv.DictReader(f))


def prepare_cache(bg, catalog):
    """bg/cache/<background_id>.npy: RGB uint8, EXIF orientation applied, short side capped at CACHE_MAX_SHORT."""
    cache = bg / "cache"
    cache.mkdir(exist_ok=True)
    for row in catalog:
        dst = cache / f"{row['background_id']}.npy"
        if dst.exists():
            continue
        im = ImageOps.exif_transpose(Image.open(bg / "ready" / row["file"])).convert("RGB")
        s = CACHE_MAX_SHORT / min(im.size)
        if s < 1:
            im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.LANCZOS)
        arr = np.asarray(im)
        assert min(arr.shape[:2]) >= P.SIZE, row["file"]
        tmp = dst.with_name(dst.stem + f".{os.getpid()}.tmp.npy")
        np.save(tmp, arr)
        os.replace(tmp, dst)  # atomic: concurrent array tasks never see a partial file


def view_rng(sid):
    return np.random.default_rng([SEED, int(hashlib.md5(sid.encode()).hexdigest()[:15], 16)])


def background_for(sid, pool, cache):
    """Random photo / crop / flip for one view -> (RGB uint8 (768,768,3), background_id, crop box, flip).
    Crop box is normalized (x0, y0, x1, y1) like the crop_box of the composites' catalog."""
    rng = view_rng(sid)
    bid = pool[rng.integers(len(pool))]
    photo = np.load(cache / f"{bid}.npy", mmap_mode="r")
    h, w = photo.shape[:2]
    short = min(h, w)
    lo = max(MIN_CROP_FRAC * short, min(short, P.SIZE))
    s = int(rng.uniform(lo, short + 1)) if lo < short else short
    s = min(s, short)
    x0, y0 = int(rng.integers(0, w - s + 1)), int(rng.integers(0, h - s + 1))
    flip = bool(rng.random() < 0.5)
    crop = np.ascontiguousarray(photo[y0:y0 + s, x0:x0 + s])
    if flip:
        crop = crop[:, ::-1]
    interp = cv2.INTER_AREA if s > P.SIZE else cv2.INTER_LANCZOS4
    bgimg = cv2.resize(np.ascontiguousarray(crop), (P.SIZE, P.SIZE), interpolation=interp) if s != P.SIZE else crop
    box = (round(x0 / w, 5), round(y0 / h, 5), round((x0 + s) / w, 5), round((y0 + s) / h, 5))
    return bgimg, bid, box, flip


# --- per-inspection job ------------------------------------------------------------------------------

def _init_worker(fg, v1, bg, out, png, pools):
    cv2.setNumThreads(1)
    _W.update(fg=fg, v1=v1, cache=bg / "cache", out=out, png=png, pools=pools)


def process_inspection(insp):
    """All 8 views of one inspection. Skipped if its meta file exists. Returns (inspection_id, n_views, seconds)."""
    fg, v1, out, cache = _W["fg"], _W["v1"], _W["out"], _W["cache"]
    iid = insp["inspection_id"]
    meta_path = out / "meta" / f"{iid}.json"
    if meta_path.exists():
        return iid, 0, 0.0
    t0 = time.time()
    pool = _W["pools"][insp["split"]]
    meta = {}
    for v in insp["views"]:
        sid = P.sample_id(iid, v["view_id"])
        rgba = cv2.imread(str(fg / "inspections" / iid / v["view_id"] / "foreground.png"), cv2.IMREAD_UNCHANGED)
        assert rgba.shape == (P.SIZE, P.SIZE, 4), sid
        fgrgb, a = rgba[..., 2::-1].astype(np.float32), rgba[..., 3:].astype(np.float32) / 255
        bgimg, bid, box, flip = background_for(sid, pool, cache)
        image = np.clip(a * fgrgb + (1 - a) * bgimg.astype(np.float32) + 0.5, 0, 255).astype(np.uint8)

        with np.load(v1 / "npz" / f"{sid}.npz") as z:
            arrays = {k: z[k] for k in z.files}
        # QA: fully opaque pixels must match the Data_v1 image exactly (same foreground render -> right pairing).
        opaque = rgba[..., 3] == 255
        qa_diff = int(np.abs(arrays["image"][opaque].astype(np.int16) - image[opaque].astype(np.int16)).max())
        arrays["image"] = image

        if _W["png"]:
            d = out / "png" / sid
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / "rgb.png"), image[..., ::-1])
            cv2.imwrite(str(d / "background.png"), bgimg[..., ::-1])
            cv2.imwrite(str(d / "overlay.png"), P.overlay(image, arrays["mask"], arrays["strap_cl_1px"],
                                                          arrays["cord_cl_1px"])[..., ::-1])

        npz, tmp = out / "npz" / f"{sid}.npz", out / "npz" / f"{sid}.npz.tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, npz)
        meta[sid] = {"background_id": bid, "bg_crop_box": box, "bg_flip": int(flip), "qa_opaque_max_diff": qa_diff}
    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta))
    os.replace(tmp, meta_path)  # written last: marks the inspection done
    return iid, len(meta), time.time() - t0


def run(args, inspections, pools):
    if args.only:
        missing = set(args.only) - {i["inspection_id"] for i in inspections}
        assert not missing, f"unknown inspections: {sorted(missing)}"
        todo = [i for i in inspections if i["inspection_id"] in set(args.only)]
    else:
        todo = sorted(inspections, key=lambda i: i["inspection_id"])[args.shard::args.nshards]
    print(f"shard {args.shard}/{args.nshards}: {len(todo)} inspections, {args.workers} workers", flush=True)
    t0, n_views, busy = time.time(), 0, 0.0
    init = (args.fg, args.v1, args.bg, args.out, args.png, pools)
    with Pool(args.workers, initializer=_init_worker, initargs=init) as pool:
        for k, (iid, n, sec) in enumerate(pool.imap_unordered(process_inspection, todo), 1):
            n_views, busy = n_views + n, busy + sec
            if k % 50 == 0 or k == len(todo):
                print(f"{k}/{len(todo)} inspections, {n_views} views written, {time.time() - t0:.0f}s wall"
                      + (f", {busy / n_views:.2f}s per view per core" if n_views else ""), flush=True)


# --- index and statistics ----------------------------------------------------------------------------

def finalize(args, catalog):
    out = args.out
    with open(args.v1 / "index.csv") as f:
        v1_rows = list(csv.DictReader(f))
    photo_split = {r["background_id"]: r["split"] for r in catalog}
    metas, rows, n_missing = {}, [], 0
    for p in sorted((out / "meta").glob("*.json")):
        metas.update(json.loads(p.read_text()))
    use, bad_qa = defaultdict(Counter), []
    for r in v1_rows:
        m = metas.get(r["sample_id"])
        if m is None:
            n_missing += 1
            continue
        assert photo_split[m["background_id"]] == r["split"], r["sample_id"]  # no photo crosses splits
        rows.append({**r, "background_id": m["background_id"], "bg_crop_box": " ".join(map(str, m["bg_crop_box"])),
                     "bg_flip": m["bg_flip"]})
        use[r["split"]][m["background_id"]] += 1
        if m["qa_opaque_max_diff"] != 0:
            bad_qa.append((r["sample_id"], m["qa_opaque_max_diff"]))
    print(f"{len(rows)} views, {n_missing} views not done")
    print(f"QA (opaque pixels identical to Data_v1): {len(rows) - len(bad_qa)}/{len(rows)} ok"
          + (f", e.g. {bad_qa[:5]}" if bad_qa else ""))
    for s, c in sorted(use.items()):
        print(f"{s}: {len(c)} photos, views per photo min {min(c.values())} max {max(c.values())}")

    with open(out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for name in ("qa.csv", "defect_states.json"):
        shutil.copyfile(args.v1 / name, out / name)
    shutil.copyfile(args.bg / "catalog.csv", out / "bg_catalog.csv")
    stats = P.compute_stats(out, rows, args.workers)
    if stats is not None:
        (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"stats: {stats['n_train_views'] if stats else 0} train views")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fg", required=True, type=Path, help="foregrounds dataset (read only)")
    ap.add_argument("--v1", default="Data_v1", type=Path, help="Data_v1 (masks, index)")
    ap.add_argument("--bg", default="bg", type=Path, help="folder with catalog.csv and ready/")
    ap.add_argument("--out", default="Data_v2", type=Path)
    ap.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--only", nargs="*", help="process only these inspection folders (test run)")
    ap.add_argument("--png", action="store_true", help="also write rgb/background/overlay PNGs per view (test run)")
    ap.add_argument("--finalize", action="store_true", help="build index.csv, stats.json; copy qa.csv etc.")
    args = ap.parse_args()
    for k in ("fg", "v1", "bg", "out"):
        setattr(args, k, getattr(args, k).resolve())
    assert 0 <= args.shard < args.nshards
    for sub in ("npz", "meta"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    catalog = read_catalog(args.bg)
    t0 = time.time()
    if args.finalize:
        finalize(args, catalog)
    else:
        prepare_cache(args.bg, catalog)
        pools = defaultdict(list)
        for r in catalog:
            pools[r["split"]].append(r["background_id"])
        inspections = json.loads((args.fg / "manifest.json").read_text())["inspections"]
        run(args, inspections, dict(pools))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
