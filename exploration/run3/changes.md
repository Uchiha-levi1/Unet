# Run 3: changes from run 2

Run 2 records are in `../run2/`. Everything in run 2 still applies, except what is listed here.

## Background photos (`bg/`, git-ignored)
- **`bg/raw/`:** the 31 photos as provided, untouched, plus `papa_original.jpg`, `osan.jpg` and `mcentire_original.jpg` (the three run 2 backgrounds, downloaded from DVIDS; SHA-256 matches `composition_config.json`).
  - `bg/raw/fullres/`: full-resolution Wikimedia Commons originals of 9 photos that were provided as 960 px thumbnails. They replace those thumbnails.
- **`bg/ready/`:** 34 photos named `bg_NNN_<name>.jpg`.
  - 4 photos still under 768 px on the short side were upscaled to 768 (Lanczos, aspect kept): `c17_kloader_cds` ×1.20, `defender25_dulmen` ×1.07, `warehouse_md17` ×1.07, `ww2_war_workers` ×1.005.
  - All others were copied unchanged (1366–8312 px wide).
- **`bg/catalog.csv`:** file, `background_id`, split, location group, size, original size, upscale factor, source, original file name.
- **Split:** 24 train / 5 validation / 5 test, by seeded random draw (seed 0) over location groups. `papa` stays in train, `osan` in validation and `mcentire` in test. The two `abandoned_terminal` photos are one location group, so they share a split (train).
  - validation: `osan`, `asc_poland_site`, `freight_bicycle`, `motby_warehouses`, `parcel_locker`
  - test: `mcentire`, `abandoned_military_warehouses`, `brdy_missile_store`, `c17_open_ramp`, `torun_attic`
- All photos were reviewed by the user; none was filtered for real straps or bundles.
- **`bg/cache/`:** each photo as RGB uint8 `.npy`, EXIF orientation applied, short side capped at 2048 px (a 60% crop of 2048 px is still more than 768, so output quality is unchanged). 448 MB, memory-mapped by the workers.

## Data_v2 (`src/composite_v2.py`, git-ignored output)
- **Source:** `/scratch/mhameed/shared_dataset/foregrounds/`: the same 5004 inspections × 8 views as `variant_001`. Each view's `foreground.png` (RGBA, straight alpha) is the package with no background, and the GT is identical to `variant_001`.
- **Per view, seeded by `sample_id`:**
  - a random photo from the view's split pool
  - a random square crop, side in [max(0.6 × short side, min(short side, 768)), short side], at a random position; the same rule for all 8 views (no special floor crop for views 05/06)
  - horizontal flip with p = 0.5, then resized to 768 (area interpolation when downscaling, Lanczos otherwise)
  - `image = α·foreground + (1 − α)·background` in 8-bit sRGB; **no shadow**, no exposure change
- **Masks and centerlines:** copied unchanged from the `Data_v1` npz files.
- **Output `Data_v2/`:**
  - `npz/`: same keys as `Data_v1`, only `image` differs
  - `index.csv`: `Data_v1` columns, with `background_id` = photo id, plus `bg_crop_box` (normalized x0 y0 x1 y1) and `bg_flip`
  - `qa.csv` and `defect_states.json`: copied from `Data_v1`
  - `bg_catalog.csv`: copy of `bg/catalog.csv`
  - `stats.json`: recomputed
- **Jobs:** `jobs/composite_v2_test.sbatch` (2 inspections + PNGs), `jobs/composite_v2.sbatch` (4-task array, 16 CPUs each), `jobs/composite_v2_finalize.sbatch`, `jobs/composite_v2_submit.sh`.

## Results (2026-11-10; test job 2494957, array 2494959, finalize 2494960)
- **Time:** about 0.3 s per view per core; the array tasks ran about 2–2.5 min, finalize 3.5 min.
- **Output:** 40,032 npz files, none missing, no leftover `.tmp` files, 46 GB.
- **QA:** opaque pixels identical to the `Data_v1` image in 40,032/40,032 views, so the foreground and labels are paired correctly.
- **Photo use:** no photo appears in two splits. Views per photo: train 1086–1256 (24 photos), validation 1155–1282 (5), test 1174–1220 (5).
- **`stats.json` (train split):** image mean RGB (0.416, 0.410, 0.378) and std (0.232, 0.231, 0.233); in `Data_v1` they were (0.338, 0.331, 0.315) and (0.160, 0.159, 0.153). Mask and centerline fractions are unchanged, and `pos_weight` stays 30.
- **Visible limitations:** with no shadow the package looks pasted, slightly floating. Some elevated views (05/06) show sky or walls behind a package seen from above.
