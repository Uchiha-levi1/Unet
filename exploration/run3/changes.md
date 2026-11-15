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

## Training r2 (job 2494969)
- **Same as r1** (`configs/v1_finetune.yaml`, run 2), except for the data: `configs/v2_finetune.yaml` sets `data.root: Data_v2` and `train.out_dir: runs/v2_finetune`.
  - initialized from `runs/turing/best.pt` (run 1, epoch 42), weights only
  - learning rate 3e-5 / 3e-4, no encoder freeze, warmup 0.3 epoch, 15 epochs, cord λ 1.5, clDice 2
  - full validation (6000 views, the 5 validation photos) every 3 epochs; snapshots every 3 epochs in `runs/v2_finetune/snapshots/`
  - ImageNet normalization (kept, although `Data_v2`'s own mean and std differ more from ImageNet's)
- No smoke test: only the config and data differ from r1, whose code path passed its smoke test.
- `CONFIG=configs/v2_finetune.yaml DATA=Data_v2 INIT=runs/turing/best.pt sbatch -t 12:00:00 jobs/train_turing.sbatch`
- It runs alongside r1 (job 2494741) on a second L40S.
- **r1 so far (`Data_v1`, one train photo):** validation strap Dice peaked at epoch 3 (0.848, `tprec` 0.635), then 0.830 / 0.834 at epochs 6 / 9, while train loss kept falling. It is fitting the train photo.
- **After both runs:** a 2×2 evaluation of the r1 and r2 `best.pt` on the `Data_v1` and `Data_v2` validation splits.
- **First r2 attempt (job 2494969) failed** after 7 min: out of host memory (48 GB). `/tmp` on the nodes is likely RAM-backed, so the 46 GB of staged `Data_v2` counts against the job's memory; r1's 38 GB had just fit. **Resubmitted as job 2497658** with `--mem=96G`, otherwise unchanged:
  `CONFIG=configs/v2_finetune.yaml DATA=Data_v2 INIT=runs/turing/best.pt sbatch -t 12:00:00 --mem=96G jobs/train_turing.sbatch`
- **r1 finished** (job 2494741, 5 h 28 min). Validation strap Dice by epoch: 3 = 0.848 (best), 6 = 0.830, 9 = 0.834, 12 = 0.839, 15 = 0.834. Final cord Dice 0.763, hardware Dice 0.895. r1 test-set evaluation (`Data_v1` test, `mcentire`): job 2497657.
- **r2 finished** (job 2497658, 5 h 32 min). Validation (6000 views, the 5 validation photos):

| epoch | train loss | strap Dice | strap `tprec` | components (pred; GT 9.3) | cord Dice | hardware Dice | tape Dice | cargo_cover Dice |
|---|---|---|---|---|---|---|---|---|
| 3 | 0.415 | 0.973 | 0.985 | 4.8 | 0.922 | 0.956 | 0.892 | 0.987 |
| 6 | 0.356 | 0.975 | 0.986 | 4.7 | 0.926 | 0.960 | 0.896 | 0.988 |
| 9 | 0.324 | 0.975 | 0.985 | 4.9 | 0.928 | 0.961 | 0.899 | 0.988 |
| 12 | 0.305 | **0.976** | 0.986 | 4.9 | 0.929 | 0.962 | 0.900 | 0.989 |
| 15 | 0.298 | 0.976 | 0.985 | 4.9 | 0.930 | 0.962 | 0.900 | 0.989 |

  `best.pt` is from epoch 12. Compared with r1 (best strap Dice 0.848, `tprec` 0.635, 160 components), the 34 background photos removed most of the background false positives on unseen photos.

## r1 test results and confusion matrix
- r1 `best.pt` (epoch 3) on the `Data_v1` test split (6000 views, `mcentire`), job 2497657: strap Dice 0.936, cord 0.890, hardware 0.931, tape 0.731, cargo_cover 0.881, components 76 / 9.5. Full table: `results_r1.md`.
- **New `src/confusion.py` + `jobs/confusion_turing.sbatch`:** an 11×11 pixel class × class confusion matrix (rows = GT, columns = prediction).
  - GT label: the first present channel in the order hardware > cord_or_tie > tape > data_tag > pack_band > strap > skid_board > honeycomb > parachute_bag > cargo_cover; otherwise background.
  - Predicted label: the most likely channel above 0.5; otherwise background.
  - Output: `<checkpoint dir>/confusion_<split>.json`.
- r1 test confusion (job 2497920): overall pixel accuracy 95.8%. Largest confusion: 4.4% of background pixels predicted as cargo_cover, which is 18.9% of all predicted cargo_cover pixels. Written up in `results_r1.md` (repo root).

## Real photos (`A22/`, git-ignored)
- 16 phone photos (4000×3000, EXIF orientation 6, so 3000×4000 portrait once upright), plus 2 videos and 3 `.blend` files (not used).
- **Preprocessing:** EXIF orientation applied, center square crop (3000×3000, 500 px cut from top and bottom), resized to 768 (Lanczos). Saved to `runs/a22_real/input_768/` and approved by the user.
- **New `src/predict_images.py`:** runs a checkpoint on plain images (no GT) and writes `<name>_overlay.png` (image | predictions painted in the preprocessing colors, strap centerline in white) plus per-class probability PNGs.
- **r1 `best.pt` → `runs/a22_real/pred_r1/`:** strap is partly found but broken; the centerline fires almost everywhere (fabric folds, rug lines, ceiling net); cargo_cover covers 42–53% of every image, including walls and floor.
- **r2 `best.pt` → `runs/a22_real/pred_r2/`:** background false positives are almost gone, but strap recall is low (0.6–2.2% of pixels; mostly fragments of the lateral straps), and cargo_cover is 8–19%, probably under-predicted. The remaining problem is the gap in appearance between rendered and real webbing (color, texture, stitching), not the background.
- **Training facts (checked):** BatchNorm is in the encoder and in every decoder block, and trains normally in r1 and r2 (batch 16). Inputs are RGB (3 channels), ImageNet-normalized; no grayscale conversion or random-grayscale augmentation.
