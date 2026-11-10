"""Preprocess a composites variant (e.g. variant_001) into Data_v1/, same npz/index/stats format as Data/.

Reuses the target builders of src/preprocess.py unchanged. Differences: npz only (PNGs only with --png),
sharded for SLURM job arrays, index.csv drops png_dir and adds background_id.

Per inspection, a worker writes <out>/npz/<sample_id>.npz for its 8 views and then <out>/meta/<inspection_id>.json
(per-view centerline coverage + background id). The meta file marks the inspection done (resumable).
--finalize then builds index.csv, qa.csv, defect_states.json and stats.json (train split) from all meta files.

CLI:
  python -m src.preprocess_v1 --src SRC --out Data_v1 [--shard i --nshards N] [--only ID ...] [--png] [--workers N]
  python -m src.preprocess_v1 --src SRC --out Data_v1 --finalize [--workers N]
"""
import argparse
import csv
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from src import preprocess as P  # build_mask, amodal_strap_centerline, overlay, compute_stats, constants

_W = {}  # per-worker settings


def _init_worker(src, out, png):
    cv2.setNumThreads(1)
    P._G.update(P.load_registry(src))  # P.build_mask reads the registry sets from P._G
    _W.update(src=src, out=out, png=png)


def process_inspection(insp):
    """All 8 views of one inspection. Skipped if its meta file exists. Returns (inspection_id, n_views, seconds)."""
    src, out, png = _W["src"], _W["out"], _W["png"]
    iid = insp["inspection_id"]
    meta_path = out / "meta" / f"{iid}.json"
    if meta_path.exists():
        return iid, 0, 0.0
    t0 = time.time()
    insp_dir = src / "inspections" / iid
    assembly = json.loads((insp_dir / "assembly.json").read_text())
    pack_band = set(P._G["pack_band"].tolist())
    meta = {}
    for v in insp["views"]:
        sid = P.sample_id(iid, v["view_id"])
        d = insp_dir / v["view_id"]
        rd = lambda f: cv2.imread(str(d / f), cv2.IMREAD_UNCHANGED)
        rgb = rd("rgb.png")[..., ::-1]
        cm, im = rd("class_map.png"), rd("instance_map.png")
        cord_cl = (rd("cord_centerline_mask.png") > 127).astype(np.uint8)
        ann = json.loads((d / "annotations.json").read_text())
        assert rgb.shape == (P.SIZE, P.SIZE, 3) and cm.shape == im.shape == (P.SIZE, P.SIZE), sid

        mask = P.build_mask(cm, im)
        cl1 = P.amodal_strap_centerline(assembly, ann["camera"], cm, im, pack_band)
        cl3 = cv2.dilate(cl1, np.ones((3, 3), np.uint8))

        # QA: the amodal centerline must cover the provided (visible) strap centerline within +-2 px.
        provided = (rd("centerline_mask.png") > 127) & ~np.isin(rd("centerline_instance_map.png"), P._G["pack_band"])
        near = cv2.dilate(cl1, np.ones((5, 5), np.uint8)).astype(bool)
        coverage = float((provided & near).sum() / provided.sum()) if provided.any() else 1.0

        if png:
            png_dir = out / "png" / sid
            png_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(d / "rgb.png", png_dir / "rgb.png")
            for k, m in zip(P.CHANNELS, mask):
                cv2.imwrite(str(png_dir / f"{k}.png"), m * 255)
            for k, m in (("strap_cl_1px", cl1), ("strap_cl_3px", cl3), ("cord_cl_1px", cord_cl)):
                cv2.imwrite(str(png_dir / f"{k}.png"), m * 255)
            cv2.imwrite(str(png_dir / "overlay.png"), P.overlay(rgb, mask, cl1, cord_cl)[..., ::-1])

        npz, tmp = out / "npz" / f"{sid}.npz", out / "npz" / f"{sid}.npz.tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, image=np.ascontiguousarray(rgb), mask=mask, strap_cl_1px=cl1,
                                strap_cl_3px=cl3, cord_cl_1px=cord_cl)
        os.replace(tmp, npz)
        meta[sid] = {"centerline_coverage": coverage,
                     "background_id": ann["composition"]["background"]["id"]}
    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta))
    os.replace(tmp, meta_path)  # written last: marks the inspection done
    return iid, len(meta), time.time() - t0


def run(args, inspections):
    if args.only:
        missing = set(args.only) - {i["inspection_id"] for i in inspections}
        assert not missing, f"unknown inspections: {sorted(missing)}"
        todo = [i for i in inspections if i["inspection_id"] in set(args.only)]
    else:
        todo = sorted(inspections, key=lambda i: i["inspection_id"])[args.shard::args.nshards]
    print(f"shard {args.shard}/{args.nshards}: {len(todo)} inspections, {args.workers} workers", flush=True)
    t0, n_views, busy = time.time(), 0, 0.0
    with Pool(args.workers, initializer=_init_worker, initargs=(args.src, args.out, args.png)) as pool:
        for k, (iid, n, sec) in enumerate(pool.imap_unordered(process_inspection, todo), 1):
            n_views, busy = n_views + n, busy + sec
            if k % 20 == 0 or k == len(todo):
                print(f"{k}/{len(todo)} inspections, {n_views} views written, {time.time() - t0:.0f}s wall"
                      + (f", {busy / n_views:.1f}s per view per core" if n_views else ""), flush=True)


def finalize(args, inspections):
    out = args.out
    rows, qa, bg = [], [], defaultdict(Counter)
    n_missing = 0
    for insp in inspections:
        mp = out / "meta" / f"{insp['inspection_id']}.json"
        if not mp.exists():
            n_missing += 1
            continue
        meta = json.loads(mp.read_text())
        defect = insp["defect_ids"][0] if insp["defect_ids"] else "baseline"
        for v in insp["views"]:
            sid = P.sample_id(insp["inspection_id"], v["view_id"])
            visible = ""
            if defect != "baseline":
                state = next(c["label_state"] for c in v["checks"] if c["check_id"] == defect)
                visible = int(state == "fault_present")
            rows.append({"sample_id": sid, "family_id": insp["family_id"], "defect": defect,
                         "view": v["view_id"], "split": insp["split"], "defect_visible": visible,
                         "npz_path": f"npz/{sid}.npz", "background_id": meta[sid]["background_id"]})
            qa.append((sid, meta[sid]["centerline_coverage"]))
            bg[insp["split"]][meta[sid]["background_id"]] += 1
    rows.sort(key=lambda r: r["sample_id"])
    qa.sort()
    print(f"{len(rows)} views from {len(inspections) - n_missing} inspections; {n_missing} inspections not done")

    with open(out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "qa.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "centerline_coverage"])
        w.writerows((s, f"{c:.4f}") for s, c in qa)
    states = {(i["defect_ids"][0] if i["defect_ids"] else "baseline"): i["state"] for i in inspections}
    (out / "defect_states.json").write_text(json.dumps(dict(sorted(states.items())), indent=2))

    low = [(s, c) for s, c in qa if c < 0.99]
    print(f"centerline coverage: min {min(c for _, c in qa):.4f}, {len(low)} views below 0.99"
          + (f" (e.g. {low[:5]})" if low else ""))
    print("background_id per split:", {s: dict(c) for s, c in sorted(bg.items())})

    stats = P.compute_stats(out, rows, args.workers)
    if stats is not None:
        (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"stats: {stats['n_train_views'] if stats else 0} train views")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, type=Path, help="variant folder (read only)")
    ap.add_argument("--out", default="Data_v1", type=Path, help="output folder")
    ap.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--only", nargs="*", help="process only these inspection folders (test run)")
    ap.add_argument("--png", action="store_true", help="also write PNGs + overlay per view (test run)")
    ap.add_argument("--finalize", action="store_true", help="build index.csv, qa.csv, defect_states.json, stats.json")
    args = ap.parse_args()
    args.src, args.out = args.src.resolve(), args.out.resolve()
    assert 0 <= args.shard < args.nshards
    for sub in ("npz", "meta"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    inspections = json.loads((args.src / "manifest.json").read_text())["inspections"]
    t0 = time.time()
    finalize(args, inspections) if args.finalize else run(args, inspections)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
