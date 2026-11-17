# Run 4: changes from run 3

Run 3 records are in `../run3/`. Everything in run 3 still applies, except what is listed here.

## Source data: `single_a22_physical_v13`
- `/scratch/mhameed/shared_dataset/single_a22_physical_v13/` (read only).
- One A-22 package per image (previous data: two packages).
- EEVEE renders, 768×768, orthographic camera.
- **Backgrounds:** MS COCO test2017 photos, "center fit", one photo per family and view, the same across that view's states. Kept as is (no recompositing).
- **Labels:** `class_map.png` (uint8) is a single label per pixel (visible surface only); previous data was multi-label.
  - Classes: 0 background, 1 cargo_cover, 2 webbing, 3 cord_or_tie, 4 attachment_hardware, 5 tape, 6 skid_board, 7 honeycomb, 8 parachute_bag, 9 data_tag. There is no `pack_band`; pack webbing is plain `webbing`.
  - `object_map.png` / `instance_map.png` (uint16, identical) and `registry.json` (object → class, `component_id`).
- **No centerline ground truth** is provided (and no 3D curves).
- **States per view:** intact, E01_missing_pack, E03_missing_upper_lateral, E04_missing_support_pair, E05_missing_front_skid_ties, E13_missing_data_tag. The geometry is identical across states, so the 6 states of a view are near-duplicates.
- Other files per sample: `foreground.png` (RGBA), `depth.exr`, `defect_expected_region.png`, `annotations.json` (camera, `removed_object_ids`, `observable`), `background.json` (COCO id).
- **Incomplete:** `validation.json` reports `partial: true`, 43,318 of 48,000 samples. 903 of 1000 families exist (700 train / 150 validation / 53 test); 26 test samples inside those families were never rendered. Decision: use every existing sample; skip only missing samples, never whole families.

## Data_v3 (`src/preprocess_v3.py`, git-ignored output)
- **Mask:** 8 channels, one-hot of `class_map`: `webbing, cargo_cover, cord_or_tie, attachment_hardware, tape, skid_board, honeycomb, parachute_bag`.
  - `data_tag` is not a channel. Its pixels are stored separately (`data_tag` key) and are **ignored** in every loss term, metric and the confusion matrix (not counted as background).
  - **attachment_hardware = convex hull per buckle object.** The source labels buckles as thin open frames; on 40 random test views (372 buckles) the earlier per-instance hole-fill changed only 1% of them, the hull fills them (14,705 → 34,298 px). The hull overrides whatever lies inside it (single label kept).
- **Centerlines** (`webbing_cl_1px/3px`, `cord_cl_1px/3px`), visible only:
  1. per registry `component_id` (a strap and its selvedge objects together), the component's pixels
  2. closed with a 7 px kernel, but only into webbing ∪ cord ∪ tape ∪ hardware pixels (bridges gaps under crossings)
  3. skeletonized; end branches shorter than 8 px (webbing) / 5 px (cord) removed
  4. 3 px target = 1 px dilated with a 3×3 kernel
- A sample is used only if `rgb.png, class_map.png, object_map.png, annotations.json, background.json` all exist; others go to `skipped.csv`.
- **Output:** `npz/<family>__<view>__<state>.npz` (image, mask, data_tag, 4 centerline maps), `meta/<family>.json`, `index.csv` (sample_id, family_id, view, defect = state, split, defect_visible = observable, coco_id, npz_path), `skipped.csv`, `stats.json`.
- **Jobs:** `jobs/preprocess_v3_test.sbatch` (2 families + PNGs), `jobs/preprocess_v3.sbatch` (4-task array, 16 CPUs, 32 GB), `jobs/preprocess_v3_finalize.sbatch`, `jobs/preprocess_v3_submit.sh`.

## Data_v3 results (2026-11-16; test job 2527878, array 2527920, finalize 2527921)
- About 0.4 s per sample per core; the array ran about 5 min, finalize 80 s.
- 43,318 samples, 0 skipped: train 700 families / 33,600, validation 150 / 7,200, test 53 / 2,518. Train and validation have 1/6 of each state.
- About 1.1 MB per sample, about 49 GB in total.
- `stats.json` (train) pixel fractions: webbing 7.0%, cargo_cover 21.4%, cord 2.0%, attachment_hardware 0.12%, tape 0.03%, skid_board 1.3%, honeycomb 3.1%, parachute_bag 6.9%. Centerline 3 px: webbing 3.3%, cord 2.0%; `pos_weight` 29.7 (webbing), 30 (cord, cap).
- Visual check (overlays): centerlines run through the middle of straps and through buckles / crossings; a few 1–3 px stubs remain at cord knots.

## Training r3 (planned): `configs/v3_finetune.yaml`
- Base `v2_finetune.yaml` (r2 optimization unchanged): learning rate 3e-5 / 3e-4, no encoder freeze, warmup 0.3 epoch, 15 epochs, full validation (7,200) every 3 epochs, snapshots every 3 epochs.
- Initialized from r2 `runs/v2_finetune/best.pt` with **head remapping by class name**: webbing ← strap, attachment_hardware ← hardware, other classes ← same name; webbing centerline ← strap centerline; the cord centerline channel starts fresh. All other weights load exactly.
- **9 → 8 sigmoid channels** (no background channel, no softmax).
- **Two centerline head channels** (webbing, cord), weighted BCE on the 3 px targets with their own `pos_weight`.
- **λ:** webbing 1, cord 1.5, attachment_hardware 1, tape 0.5, cargo_cover / parachute_bag / honeycomb / skid_board 0.25 (r1/r2 values).
- **clDice:** webbing 2 (5 iterations), cord 2 (3 iterations), using the generated 1 px skeletons.
- **New consistency loss** (weight 0.5): per centerline channel, Σ σ(cl)·(1 − max σ(mask) over webbing, cord, tape, attachment_hardware) / Σ σ(cl), i.e. the share of predicted centerline lying outside the predicted support. Not detached, so mask and centerline are learnt together.
- `best.pt` is chosen by validation webbing Dice.
- Memory: `--mem=64G` (49 GB staged in RAM-backed `/tmp` + about 10 GB for training).
- Submit: `CONFIG=configs/v3_finetune.yaml DATA=Data_v3 INIT=runs/v2_finetune/best.pt sbatch -t 12:00:00 --mem=64G jobs/train_turing.sbatch`
- Smoke test: `configs/v3_finetune_smoke.yaml` (job 2528504).

## Code changes (config-driven; r1/r2 configs behave as before)
- `losses.py`: optional ignore map (Dice, focal, clDice, BCE, consistency); several centerline channels with per-channel `pos_weight`; consistency term; clDice entries for classes the model does not have are skipped.
- `model.py`: `num_centerlines`; `model.centerlines` (BCE target keys, default `[strap_cl_3px]`) and `model.primary_class` (default `strap`).
- `data.py`: `data.line_keys` and `data.ignore_key`.
- `metrics.py` / `infer.py`: primary class instead of hard-coded strap (`<primary>_cl_coverage`), ignore map, per-centerline probability PNGs.
- `train.py`: `train.init_head_map` for head remapping; best by primary-class Dice; console line shows the consistency term.
- `confusion.py`: on single-label data the config's class order is used; ignored pixels are left out.
- `predict_images.py`: v3 overlay (webbing centerline white, cord cyan).
- `tests/test_v3.py`: 5 new tests (two heads, perfect vs noisy loss, ignore region, consistency, head remapping). All 21 tests pass.
- `CLAUDE.md`: "explain" = brief; long explanations only for "explain in detail".
