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
- Smoke test: `configs/v3_finetune_smoke.yaml` (2 × 50 steps, 100-view validation). First submitted as job 2528504 (L40S only, cancelled while queued), rerun as **job 2528710**: passed in 6 min 55 s on an L40S.
  - staging 50 GB took 255 s; r2 weights loaded with the head remapping; `pos_weight` [29.7, 30]
  - epoch 1 → 2: loss 3.71 → 1.28, consistency 0.615 → 0.208; gradient clipped 94% of steps in epoch 1 (fresh cord centerline head), 24% in epoch 2
  - validation after 100 steps: webbing Dice 0.904, `tprec` 0.910, cord Dice 0.844, hardware Dice 0.939
  - peak GPU memory 37.5 GB (fits L40S / A100-80G / H100 / H200, not A30)
- **Connected components (`cc`) are not used for v3 comparisons.** Webbing is labeled visible-only, so every crossing cord or tape splits a strap: GT has about 60 components per image (old amodal strap: about 9). Still logged.

## GPUs
- The env's PyTorch 2.14 + CUDA 12.6 supports up to sm_90; bf16 and about 38 GB of GPU memory are needed.
  - Compatible: L40S (48 GB), A100-80G, H100, H200.
  - Not compatible: RTX PRO 6000 Blackwell (sm_120, needs a CUDA 12.8+ PyTorch), V100 (no bf16), A30 (24 GB).
- All partitions share the same nodes; account `nitin` has QOS `short`, `quick` and `long`.
- Jobs now take the first free compatible GPU: `--gres=gpu:1 --constraint="L40S|A100-80G|H100|H200"`.

## Full runs (2026-11-17)
- **r3** (fine-tune from r2), **job 2529511**:
  `CONFIG=configs/v3_finetune.yaml DATA=Data_v3 INIT=runs/v2_finetune/best.pt sbatch -t 12:00:00 --mem=64G --gres=gpu:1 --constraint="L40S|A100-80G|H100|H200" jobs/train_turing.sbatch`
- **fromscratch** (`configs/fromscratch_a22.yaml`), **job 2529512**: the same data, classes, losses and validation as r3, but no `INIT`.
  - encoder: timm ImageNet `resnet34.a1_in1k` (cached in `hf_cache`, checked); decoder and heads random
  - run 1's from-scratch optimization: learning rate 1e-4 / 1e-3, encoder frozen in epoch 1, warmup 1 epoch, 30 epochs; validation and snapshots every 3 epochs; seed 42 (same shuffle order as r3)
  - no smoke test of this config, and no consistency ramp (weight 0.5 from step 0), by decision
  - `CONFIG=configs/fromscratch_a22.yaml DATA=Data_v3 sbatch -t 24:00:00 --mem=64G --gres=gpu:1 --constraint="L40S|A100-80G|H100|H200" jobs/train_turing.sbatch`
- **Comparison:** equal compute (fromscratch at epoch 15 vs r3 at epoch 15) and best vs best; validation, test, and the real photos in `A22/`.

## Code changes (config-driven; r1/r2 configs behave as before)
- `losses.py`: optional ignore map (Dice, focal, clDice, BCE, consistency); several centerline channels with per-channel `pos_weight`; consistency term; clDice entries for classes the model does not have are skipped.
- `model.py`: `num_centerlines`; `model.centerlines` (BCE target keys, default `[strap_cl_3px]`) and `model.primary_class` (default `strap`).
- `data.py`: `data.line_keys` and `data.ignore_key`.
- `metrics.py` / `infer.py`: primary class instead of hard-coded strap (`<primary>_cl_coverage`), ignore map, per-centerline probability PNGs.
- `train.py`: `train.init_head_map` for head remapping; best by primary-class Dice; console line shows the consistency term.
- `confusion.py`: on single-label data the config's class order is used; ignored pixels are left out.
- `predict_images.py`: v3 overlay (webbing centerline white, cord cyan).
- `tests/test_v3.py`: 5 new tests (two heads, perfect vs noisy loss, ignore region, consistency, head remapping). All 21 tests pass.
- `configs/fromscratch_a22.yaml`: from-scratch run (see Full runs).
- `CLAUDE.md`: "explain" = brief; long explanations only for "explain in detail".

## Results (full runs finished 2026-11-18)
- **r3** (`runs/v3_finetune`, job 2529511): 15 epochs, about 18.5 min/epoch, peak GPU 37.5 GB. **fromscratch** (`runs/fromscratch_a22`, job 2529512): 30 epochs, about 23.7 min/epoch, peak 34.4 GB.
- Validation Dice (7,200 views, threshold 0.5):

| class | r3 ep 15 | fromscratch ep 15 | fromscratch ep 30 |
|---|---|---|---|
| webbing | 0.9647 | 0.9655 | 0.9673 |
| cargo_cover | 0.9924 | 0.9932 | 0.9935 |
| cord_or_tie | 0.9144 | 0.9110 | 0.9149 |
| attachment_hardware | 0.9837 | 0.9831 | 0.9832 |
| tape | 0.9719 | 0.9729 | 0.9728 |
| skid_board | 0.9934 | 0.9940 | 0.9943 |
| honeycomb | 0.9962 | 0.9964 | 0.9965 |
| parachute_bag | 0.9809 | 0.9641 | 0.9734 |

- clDice (webbing / cord): r3 0.979 / 0.938, fromscratch 0.979 / 0.938. Train total loss: r3 0.428 (ep 15), fromscratch 0.407 (ep 15) and 0.376 (ep 30). Consistency term about 0.02 in both.
- `best.pt` (by webbing Dice): r3 epoch 15, fromscratch epoch 30.
- Reading: at equal compute the two runs are within about 0.003 Dice on every class except parachute_bag (noisy: fromscratch dips at epochs 12 and 15). fromscratch is ahead on webbing, cargo_cover, tape, skid_board, honeycomb; r3 on cord_or_tie and attachment_hardware. The r2 initialization does not clearly beat the ImageNet encoder on this data. Test split and real photos not yet compared.
- Topology precision < sensitivity on validation (webbing 0.968 / 0.989; cord 0.916 / 0.960): the models over-predict slightly, they are not conservative there.
- Plots: `runs/val_dice_r3_vs_fromscratch.png` (validation Dice per class vs epoch), `runs/train_loss_r3_vs_fromscratch.png`. No validation loss is logged, only validation metrics every 3 epochs; a true validation loss would need a separate eval job.
- `runs/<run>/vis/epoch_XXX/` holds the first 4 validation images (family f000700, view v00, states E01/E03/E04/E05, near-duplicates), webbing only: image | GT | error map (green TP, red FP, blue FN) | centerline probability.

## Real photos (2026-11-18/19)
- Two new webcam photos `A22/WIN_20261117_14_41_05_Pro.jpg.jpeg`, `..._14_42_38_Pro.jpg.jpeg` (2160×3840, portrait) added to `runs/a22_real/input_768/` (now 18 images).
- **Decision: pad to square, no crop.** Resized to 432×768 (Lanczos), centered in 768×768 with **black** side padding (168 px each side). A center crop would cut 22% off the top and bottom (parachute bag, honeycomb, ties). Reflect padding was tried first and rejected: it mirrored the package.
- Inference of r3 and fromscratch only (r1/r2 not run on the new photos): `jobs/predict_real_v3.sbatch` → `runs/a22_real/pred_r3/`, `pred_fromscratch/`. Job 2540054 (`quick`, 1 GPU, 2 CPUs, 8 GB, no GPU-type constraint), 32 s. An earlier run (job 2538567) used the bad reflect-padded inputs and was overwritten.
- Console output (fraction of the 768×768 image, black padding included): attachment_hardware and tape are 0.0% on every image, both models. Webbing is lower on the photos (r3 2.4% / 4.0%, fromscratch 0.7% / 1.4% on the new ones) than the 7% in training. Probably a domain gap (render vs photo); probability maps not yet checked (open: threshold 0.3 on photos, color/texture augmentation).

## Cluster notes
- `quick` partition (12 h, same nodes as `short`) is available to account `nitin`; used for the inference job.
- A30 (24 GB) estimate: r3 about 10-13 h, fromscratch about 20-27 h with batch 8 × grad_accum 2 (not run). Inference fits an A30 (fp32, one image at a time).
- Multi-GPU: `train.py` supports DDP on one node (`torchrun --standalone`, chosen by `jobs/train_turing.sbatch` when more than one GPU). Staging is once per node (49 GB shared by all ranks). `batch_size` is per GPU (4 GPUs × 16 = 64; use 4 per GPU to keep 16), no SyncBN, validation runs on rank 0 only. Not tested; multi-node not supported.
- `jobs/predict_real_v3.sbatch` has no GPU-type constraint; on an RTX PRO 6000 Blackwell node (sm_120) it would likely fail.

## Status note (2026-11-20)
- `runs/v3_finetune/best.pt` (r3, epoch 15) is the model trained sequentially over the datasets (run 1 `Data` → r2 `Data_v2` → r3 `Data_v3`; r1 `Data_v1` is a side branch, r2 started from run 1), and it has **the best results on real data so far** (user's visual review of the real-photo overlays; no metric exists). It is not on `main` (106 MB, over GitHub's 100 MB file limit); it is on the local branch `checkpoint-r3` and at `runs/v3_finetune/best.pt`. Its `config.yaml` and `log.csv` are committed.
- Full hand-over document for a new session: `ref_nov20.md` (repo root). Environment: `environment.yml`.
