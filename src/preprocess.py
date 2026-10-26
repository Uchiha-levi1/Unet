"""Preprocess the Blender A-22 dataset (dataset_main/) into training targets. See exploration/decisions.md.

Reads dataset_main/ without modifying it and writes, per view:
  <out>/npz/<sample_id>.npz   image (768,768,3) uint8, mask (10,768,768) uint8 0/1,
                              strap_cl_1px, strap_cl_3px, cord_cl_1px (768,768) uint8 0/1
  <out>/png/<sample_id>/      rgb.png, one 0/255 PNG per mask channel and centerline, overlay.png
plus <out>/index.csv, <out>/defect_states.json, <out>/stats.json (train split) and <out>/qa.csv.

CLI: python -m src.preprocess --src dataset_main --out Data [--workers N] [--only family_000066_E13 ...]
"""
import argparse
import csv
import json
import os
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

CHANNELS = ["strap", "cargo_cover", "cord_or_tie", "hardware", "tape", "skid_board", "honeycomb",
            "parachute_bag", "data_tag", "pack_band"]
# Single-class channels: channel name -> class_map value (strap, hardware, pack_band are built separately).
CLASS_CHANNELS = {"cargo_cover": 1, "cord_or_tie": 3, "tape": 5, "skid_board": 6, "honeycomb": 7,
                  "parachute_bag": 8, "data_tag": 9}
WEBBING, CORD, HARDWARE, TAPE = 2, 3, 4, 5
SIZE = 768
UNDER_CLOSE_PX = 5     # closing that extends the strap under crossing cords / tape
MAX_BEHIND_M = 0.10    # a hidden strap point is kept only this close behind a crossing webbing/cord
MAX_GAP_PX = 6         # hidden centerline runs up to this long (between visible runs) are bridged
OCCLUDER_MATCH_PX = 8  # max image distance to the occluder's own projected centerline
POS_WEIGHT_CAP = 30.0

# Overlay colors (RGB) and paint order: blobs first (blended), thin structures last (solid).
COLORS = {"cargo_cover": (128, 128, 128), "skid_board": (245, 130, 48), "honeycomb": (145, 30, 180),
          "parachute_bag": (0, 128, 128), "data_tag": (240, 50, 230), "tape": (0, 130, 200),
          "pack_band": (250, 190, 212), "cord_or_tie": (60, 180, 75), "strap": (230, 25, 75),
          "hardware": (255, 225, 25)}
BLOBS = ["cargo_cover", "skid_board", "honeycomb", "parachute_bag", "data_tag"]
THIN = ["tape", "pack_band", "cord_or_tie", "strap", "hardware"]

_G = {}  # per-worker globals (registry lookups), set by _init_worker


def load_registry(src):
    """Instance-ID sets derived from the global component registry."""
    reg = json.loads((src / "component_registry.json").read_text())
    name = {}
    for e in reg:
        name.setdefault(e["instance_id"], (e["component_name"], e["class_id"]))
    # LVCP webbing = pack bands (carrying handles, retaining bands); LVCP hardware sits on the pack, not on a strap.
    pack_band = {i for i, (n, c) in name.items() if c == WEBBING and n.startswith("LVCP_")}
    strap_hw = {i for i, (n, c) in name.items() if c == HARDWARE and not n.startswith("LVCP_")}
    assert pack_band == {113, 114, 115, 116, 125, 126}, sorted(pack_band)
    return {"pack_band": np.array(sorted(pack_band)), "strap_hw": strap_hw}


def _init_worker(src, out):
    _G.update(load_registry(src), src=src, out=out)


# --- amodal strap centerline ---------------------------------------------------------------------

def project_dense(points_w, K, w2c, step=0.5):
    """Project a 3D polyline into the image and resample it every `step` px. Returns (uv (N,2), z (N,))."""
    P = np.c_[points_w, np.ones(len(points_w))] @ w2c.T
    uvw = P[:, :3] @ K.T
    uv, z = uvw[:, :2] / uvw[:, 2:3], P[:, 2]
    t = np.r_[0, np.cumsum(np.linalg.norm(np.diff(uv, axis=0), axis=1))]
    s = np.arange(0, t[-1] + 1e-9, step)
    return np.c_[np.interp(s, t, uv[:, 0]), np.interp(s, t, uv[:, 1])], np.interp(s, t, z)


def visible_or_crossed(uv, z, inst, cm, im, occluders):
    """Keep a centerline sample if its strap is visible within +-2 px, it lies under hardware, or it lies
    under webbing/cord that is less than MAX_BEHIND_M in front of it (rejects straps on the far side of the
    package that merely project onto a front strap)."""
    xy = np.round(uv).astype(int)
    inb = (xy[:, 0] >= 0) & (xy[:, 0] < SIZE) & (xy[:, 1] >= 0) & (xy[:, 1] < SIZE)
    xs, ys = np.clip(xy[:, 0], 0, SIZE - 1), np.clip(xy[:, 1], 0, SIZE - 1)
    same = np.zeros(len(xy), bool)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            same |= im[np.clip(ys + dy, 0, SIZE - 1), np.clip(xs + dx, 0, SIZE - 1)] == inst
    cls, occ = cm[ys, xs], im[ys, xs]
    keep = inb & (same | (cls == HARDWARE))
    cand = inb & ~keep & np.isin(cls, (WEBBING, CORD))
    for o in np.unique(occ[cand]):
        if o not in occluders:
            continue
        tree, oz = occluders[o]
        idx = np.nonzero(cand & (occ == o))[0]
        dist, k = tree.query(uv[idx])
        keep[idx] = (dist < OCCLUDER_MATCH_PX) & (z[idx] - oz[k] < MAX_BEHIND_M)
    # Bridge short hidden runs that have kept samples on both sides.
    lab, n = ndi.label(~keep)
    for run in ndi.find_objects(lab):
        a, b = run[0].start, run[0].stop
        if a > 0 and b < len(keep) and (b - a) * 0.5 <= MAX_GAP_PX:
            keep[a:b] = True
    return keep


def amodal_strap_centerline(assembly, camera, cm, im, pack_band):
    K, w2c = np.array(camera["K"]), np.array(camera["world_to_camera_cv"])
    removed = set(assembly["removed_object_ids"])
    curves = [c for c in assembly["centerlines_world"] if c["object_id"] not in removed]
    proj = [(c, *project_dense(np.array(c["points_world_m"]), K, w2c)) for c in curves]
    # Every webbing/cord centerline can act as an occluder: index its projected samples with their depth.
    occ_uv, occ_z = {}, {}
    for c, uv, z in proj:
        occ_uv.setdefault(c["instance_id"], []).append(uv)
        occ_z.setdefault(c["instance_id"], []).append(z)
    occluders = {i: (cKDTree(np.vstack(occ_uv[i])), np.concatenate(occ_z[i])) for i in occ_uv}

    cl = np.zeros((SIZE, SIZE), np.uint8)
    for c, uv, z in proj:
        if c["class_id"] != WEBBING or c["instance_id"] in pack_band:
            continue
        keep = visible_or_crossed(uv, z, c["instance_id"], cm, im, occluders)
        pts = np.round(uv).astype(np.int32)
        for j in np.nonzero(keep[:-1] & keep[1:])[0]:
            cv2.line(cl, tuple(map(int, pts[j])), tuple(map(int, pts[j + 1])), 1, 1)
    return cl


# --- masks -----------------------------------------------------------------------------------------

def fill_instances(im, ids):
    """Union of per-instance hole-filled masks (fill holes only; open shapes such as hooks stay thin)."""
    out = np.zeros(im.shape, bool)
    lab = np.where(np.isin(im, list(ids)), im, 0)
    for i, sl in enumerate(ndi.find_objects(lab), start=0):
        if sl is None:
            continue
        sl = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in sl)
        out[sl] |= ndi.binary_fill_holes(lab[sl] == i + 1)
    return out


def build_mask(cm, im):
    pack = np.isin(im, _G["pack_band"])
    hw_all = fill_instances(im, set(np.unique(im[cm == HARDWARE]).tolist()))
    hw_strap = fill_instances(im, _G["strap_hw"] & set(np.unique(im[cm == HARDWARE]).tolist()))
    strap = ((cm == WEBBING) & ~pack) | hw_strap
    # The strap continues under thin cords / tape lying across it: pixels a small closing fills.
    closed = cv2.morphologyEx(strap.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((UNDER_CLOSE_PX, UNDER_CLOSE_PX), np.uint8)).astype(bool)
    strap |= closed & np.isin(cm, (CORD, TAPE))
    chans = {"strap": strap, "hardware": hw_all, "pack_band": (cm == WEBBING) & pack}
    chans.update({k: cm == v for k, v in CLASS_CHANNELS.items()})
    return np.stack([chans[k] for k in CHANNELS]).astype(np.uint8)


def overlay(rgb, mask, cl1, cord_cl):
    """RGB | RGB with channels painted (blobs blended, thin solid), strap centerline white, cord centerline cyan."""
    vis = rgb.astype(np.float32)
    for k in BLOBS:
        m = mask[CHANNELS.index(k)].astype(bool)
        vis[m] = 0.55 * vis[m] + 0.45 * np.array(COLORS[k])
    vis = vis.astype(np.uint8)
    for k in THIN:
        vis[mask[CHANNELS.index(k)].astype(bool)] = COLORS[k]
    vis[cord_cl.astype(bool)] = (0, 255, 255)
    vis[cl1.astype(bool)] = (255, 255, 255)
    return np.concatenate([rgb, vis], axis=1)


# --- per-inspection job ----------------------------------------------------------------------------

def sample_id(inspection_id, view_id):
    return f"{inspection_id}__{view_id}"


def process_inspection(inspection):
    """Process all views of one inspection folder; skips views whose npz already exists (resumable).
    Returns [(sample_id, centerline coverage or None if skipped)]."""
    src, out = _G["src"], _G["out"]
    insp_dir = src / "inspections" / inspection["inspection_id"]
    todo = [v for v in inspection["views"]
            if not (out / "npz" / f"{sample_id(inspection['inspection_id'], v['view_id'])}.npz").exists()]
    if not todo:
        return []
    assembly = json.loads((insp_dir / "assembly.json").read_text())
    results = []
    for v in todo:
        sid = sample_id(inspection["inspection_id"], v["view_id"])
        d = insp_dir / v["view_id"]
        rd = lambda f: cv2.imread(str(d / f), cv2.IMREAD_UNCHANGED)
        rgb = rd("rgb.png")[..., ::-1]
        cm, im = rd("class_map.png"), rd("instance_map.png")
        cord_cl = (rd("cord_centerline_mask.png") > 127).astype(np.uint8)
        camera = json.loads((d / "annotations.json").read_text())["camera"]
        assert rgb.shape == (SIZE, SIZE, 3) and cm.shape == im.shape == (SIZE, SIZE), sid

        mask = build_mask(cm, im)
        cl1 = amodal_strap_centerline(assembly, camera, cm, im, set(_G["pack_band"].tolist()))
        cl3 = cv2.dilate(cl1, np.ones((3, 3), np.uint8))

        # QA: the amodal centerline must cover the provided (visible) strap centerline within +-2 px.
        provided = (rd("centerline_mask.png") > 127) & ~np.isin(rd("centerline_instance_map.png"), _G["pack_band"])
        near = cv2.dilate(cl1, np.ones((5, 5), np.uint8)).astype(bool)
        coverage = float((provided & near).sum() / provided.sum()) if provided.any() else 1.0

        png_dir = out / "png" / sid
        png_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(d / "rgb.png", png_dir / "rgb.png")
        for k, m in zip(CHANNELS, mask):
            cv2.imwrite(str(png_dir / f"{k}.png"), m * 255)
        for k, m in (("strap_cl_1px", cl1), ("strap_cl_3px", cl3), ("cord_cl_1px", cord_cl)):
            cv2.imwrite(str(png_dir / f"{k}.png"), m * 255)
        cv2.imwrite(str(png_dir / "overlay.png"), overlay(rgb, mask, cl1, cord_cl)[..., ::-1])

        # npz last and atomically: its existence marks the view as done.
        npz, tmp = out / "npz" / f"{sid}.npz", out / "npz" / f"{sid}.npz.tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, image=np.ascontiguousarray(rgb), mask=mask, strap_cl_1px=cl1,
                                strap_cl_3px=cl3, cord_cl_1px=cord_cl)
        os.replace(tmp, npz)
        results.append((sid, coverage))
    return results


# --- index and statistics --------------------------------------------------------------------------

def index_rows(inspections, out):
    rows = []
    for insp in inspections:
        defect = insp["defect_ids"][0] if insp["defect_ids"] else "baseline"
        for v in insp["views"]:
            sid = sample_id(insp["inspection_id"], v["view_id"])
            if not (out / "npz" / f"{sid}.npz").exists():
                continue
            visible = ""
            if defect != "baseline":
                state = next(c["label_state"] for c in v["checks"] if c["check_id"] == defect)
                visible = int(state == "fault_present")
            rows.append({"sample_id": sid, "family_id": insp["family_id"], "defect": defect,
                         "view": v["view_id"], "split": insp["split"], "defect_visible": visible,
                         "npz_path": f"npz/{sid}.npz", "png_dir": f"png/{sid}"})
    return rows


def _stats_job(path):
    z = np.load(path)
    img = z["image"].reshape(-1, 3).astype(np.float64) / 255
    return (img.sum(0), (img ** 2).sum(0), z["mask"].reshape(len(CHANNELS), -1).sum(1),
            int(z["strap_cl_1px"].sum()), int(z["strap_cl_3px"].sum()))


def compute_stats(out, rows, workers):
    paths = [out / r["npz_path"] for r in rows if r["split"] == "train"]
    if not paths:
        return None
    with Pool(workers) as pool:
        parts = pool.map(_stats_job, paths, chunksize=4)
    s, s2, mpix, c1, c3 = (sum(p[i] for p in parts) for i in range(5))
    n = len(paths) * SIZE * SIZE
    mean = s / n
    neg_over_pos = (n - c3) / max(c3, 1)
    return {"n_train_views": len(paths), "channels": CHANNELS,
            "image_mean_rgb": mean.round(6).tolist(),
            "image_std_rgb": np.sqrt(s2 / n - mean ** 2).round(6).tolist(),
            "mask_pixel_fraction": {k: float(v / n) for k, v in zip(CHANNELS, mpix)},
            "strap_cl_1px_fraction": c1 / n, "strap_cl_3px_fraction": c3 / n,
            "centerline_neg_over_pos": neg_over_pos,  # from the 3 px target the BCE trains on
            "pos_weight_cap": POS_WEIGHT_CAP, "pos_weight": min(neg_over_pos, POS_WEIGHT_CAP)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default="dataset_main", type=Path, help="dataset_main folder (read only)")
    ap.add_argument("--out", default="Data", type=Path, help="output folder")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--only", nargs="*", help="process only these inspection folders (e.g. for a test run)")
    args = ap.parse_args()
    src, out = args.src.resolve(), args.out.resolve()
    (out / "npz").mkdir(parents=True, exist_ok=True)
    (out / "png").mkdir(exist_ok=True)

    inspections = json.loads((src / "manifest.json").read_text())["inspections"]
    if args.only:
        missing = set(args.only) - {i["inspection_id"] for i in inspections}
        assert not missing, f"unknown inspections: {sorted(missing)}"
        inspections = [i for i in inspections if i["inspection_id"] in set(args.only)]
    states = {(i["defect_ids"][0] if i["defect_ids"] else "baseline"): i["state"]
              for i in json.loads((src / "manifest.json").read_text())["inspections"]}

    t0 = time.time()
    done = []
    with Pool(args.workers, initializer=_init_worker, initargs=(src, out)) as pool:
        for k, res in enumerate(pool.imap_unordered(process_inspection, inspections), 1):
            done += res
            if k % 10 == 0 or k == len(inspections):
                print(f"{k}/{len(inspections)} inspections, {len(done)} views written, "
                      f"{time.time() - t0:.0f}s", flush=True)

    # QA log for views written in this run (appended, so resumed runs keep earlier entries).
    qa = out / "qa.csv"
    new = not qa.exists()
    with open(qa, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["sample_id", "centerline_coverage"])
        w.writerows(sorted((s, f"{c:.4f}") for s, c in done))
    low = [(s, c) for s, c in done if c < 0.99]
    if done:
        print(f"centerline coverage: min {min(c for _, c in done):.4f}, {len(low)} views below 0.99"
              + (f" (e.g. {low[:5]})" if low else ""))

    rows = index_rows(inspections, out)
    with open(out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "defect_states.json").write_text(json.dumps(dict(sorted(states.items())), indent=2))
    stats = compute_stats(out, rows, args.workers)
    if stats is not None:
        (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"index: {len(rows)} views | stats: {stats['n_train_views'] if stats else 0} train views | "
          f"{time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
