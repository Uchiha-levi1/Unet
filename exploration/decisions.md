# Decisions log — A-22 strap segmentation (UNet 3+)

**Finalised: 2026-10-24 21:11 EDT.** This file is also the Phase 2 preprocessing plan. Details of the dataset are in `dataset_report.md`.

## Paths
- **Source:** `dataset_main/`, read-only. Not used: `depth.exr`, `archive (1).zip`.
- **Output:** `Data/` at the repo root.
  - Local: `/home/uchiha_levi/AR/Unet/Data`
  - Turing: `/home/vchezhian/scratch/Unet/Data` (copied from local; the local copy is kept)
- `Data/` is added to `.gitignore`. Paths inside the index are relative to `Data/`.

## Preprocessing (`preprocess.py`)
- **Views:** all 4800 views (100 packages × 6 states × 8 views).
- **Splits:** the existing `splits.json` (by package, 70/15/15; split names `train` / `validation` / `test`). No hold-outs.
- **Mask:** `(10, 768, 768)` uint8, 0/1, multi-label. Channels can overlap.

| # | Channel | Built from |
|---|---|---|
| 0 | `strap` | webbing (class 2) minus pack-band instances, ∪ hole-filled **strap** hardware (center joins, end adapters, D-rings), ∪ cord or tape pixels filled by a 5 px closing of the strap mask (the strap continues under cord and tape) |
| 1 | `cargo_cover` | class 1 |
| 2 | `cord_or_tie` | class 3, visible only |
| 3 | `hardware` | class 4, all hardware (including pack snap hooks and O-ring), holes filled per instance (`binary_fill_holes`). Hooks and rings stay thin |
| 4 | `tape` | class 5 |
| 5 | `skid_board` | class 6 |
| 6 | `honeycomb` | class 7 |
| 7 | `parachute_bag` | class 8 |
| 8 | `data_tag` | class 9 |
| 9 | `pack_band` | webbing instances 113–116 (LVCP carrying handles) and 125–126 (LVCP retaining bands) |

- **Strap centerline (amodal):**
  1. Project `assembly.json → centerlines_world` (webbing, excluding pack bands and removed objects) with the view's `K` and `world_to_camera_cv`, densified to 0.5 px steps.
  2. Keep a point if one of these holds:
     - the strap itself is visible within ±2 px
     - it lies under hardware
     - it lies under webbing or cord **and is less than 10 cm behind it**, using the occluder's own 3D centerline depth. This rejects straps on the far side of the package.
  3. Bridge hidden gaps of 6 px or less.
  4. Store a **1 px** version (the GT skeleton for clDice) and a **3 px** version (3×3 dilation; the target for the head).
  5. Check: the result must cover 100% of the provided `centerline_mask` (±2 px).
- **Cord centerline:** the provided `cord_centerline_mask`, 1 px, visible only (the GT skeleton for cord clDice).
- **Outputs per view:**
  - `Data/npz/<sample_id>.npz` with keys `image` (768,768,3 uint8), `mask`, `strap_cl_1px`, `strap_cl_3px`, `cord_cl_1px`. Compressed.
  - `Data/png/<sample_id>/`: copy of `rgb.png`, one 0/255 PNG per channel, the centerline PNGs, `overlay.png`.
- **`Data/index.csv`:** columns `sample_id`, `family_id`, `defect` (baseline / E01 / E03 / E04 / E05 / E13), `view`, `split`, `defect_visible` (1 / 0 / empty for baseline), `npz_path`, `png_dir`.
- **`Data/defect_states.json`:** defect code → state name.
- **`Data/stats.json`** (train split):
  - image mean and std per RGB channel
  - pixel fraction per mask channel
  - strap centerline fraction at 1 px and 3 px; `pos_weight` = neg/pos of the **3 px** version, capped at 30
- **How it runs:** paths come from command-line arguments, it uses multiprocessing, and it is deterministic. It is resumable: views whose outputs are complete get skipped.
- **Rollout:**
  1. Run on the two sampled inspections first (`family_000066_E13` and `family_000072_E13`, all 8 views each).
  2. Show the overlays.
  3. Full run **only after approval**.

## Model and training (Phase 3: edit the existing `src/` and `configs/`)
- **Input:** full 768×768 images. No cropping, tiling or sliding-window blending.
- **Heads:**
  - mask: 10 sigmoid channels
  - strap centerline: 1 channel, weighted BCE on the 3 px target
- **Not now:** crossing head, cord centerline head, instance outputs.
- **Losses:**
  - strap: Dice 1, focal 1, clDice 2. clDice uses the 1 px amodal strap centerline as the GT skeleton.
  - cord: Dice 1, focal 1, clDice 2. clDice uses the 1 px visible cord centerline as the GT skeleton.
  - other classes: Dice + focal, weights to be set in Phase 3. No clDice for blob classes.
- **clDice iterations per class:** strap 5 (6 considered), cord 3.
- **Metrics:** Dice and IoU per class, clDice, connected-component count (pred vs. GT), and strap centerline coverage (share of the amodal GT centerline covered by the predicted strap mask). The best checkpoint is chosen by strap Dice.
- **Validation:** `full` (all validation images, the default) or `subset` (fixed, seeded 40 images mixing baselines and defects). Print the time of the first validation run.
- **Training data:** all 6 states with fewer epochs.
- **Tests:** unit tests and the overfit test run at 768.
- **Still open (Phase 3):**
  - the remaining training parameters
  - `configs/local.yaml` and `configs/turing.yaml`
  - `jobs/train_turing.sbatch`
  - DDP and automatic resume

## Phase 3 training parameters (agreed one by one)
1. **Encoder ResNet-34** (ResNet-50 is a possible later ablation; it needs about 5.5 GB per image and does not fit batch 16 on an L40S).
   - Turing: 1 GPU × batch 16, no accumulation, BatchNorm trains.
   - Local: batch 2 × 8 accumulation (effective 16), encoder BatchNorm frozen.
   - Local and Turing runs need not match; local is for debugging.
2. **Optimizer:** AdamW, learning rate 1e-4 (encoder) / 1e-3 (decoder + heads). Cosine schedule down to 1% of the base rate, stepped per optimizer step. Weight decay 1e-2, not applied to BatchNorm and bias parameters. Gradient-norm clipping at 1.0, applied after accumulation and before the optimizer step.
   - **Warmup and encoder freeze, per parameter group:**
     - decoder + heads: linear warmup in epoch 1, then cosine
     - encoder: frozen in epoch 1, linear warmup in epoch 2, then cosine
   - **Encoder BatchNorm while the encoder is frozen:**
     - Turing: statistics keep updating (train mode, AdaBN-style)
     - local: frozen (eval mode) for the whole run
3. **Length:** Turing 50 epochs (10 500 steps, about 3 h); local 2 epochs (smoke test). Epoch counts are set in epochs; the schedule steps per optimizer step.
   - Validation every 2 epochs.
   - Checkpoints: a full snapshot (model, optimizer, scheduler, RNG) every 5 epochs, `epoch_XXX.pt`. Resume uses the newest snapshot, so up to 5 epochs can be lost. Plus `best.pt` by validation strap Dice.
4. **Losses:**
   - centerline: BCE with λ = 1 and `pos_weight` = min(neg/pos of the 3 px target, 30) = 30
   - focal: averaged over all pixels (no foreground normalization), γ = 2, α off
   - Dice : focal = 1 : 1 in every class; the class λ multiplies both.
   - **Per-class λ:**
     - strap 1 (+ clDice 2)
     - cord 1 (+ clDice 2)
     - hardware 1
     - pack_band, tape, data_tag 0.5
     - cargo_cover, parachute_bag, honeycomb, skid_board 0.25
   - check the per-term loss magnitudes during the overfit test; if focal is negligible, raise one global focal scale
   - No early stopping.
   - **Automatic resume:** `last.pt` holds model, optimizer, scheduler, step, epoch, best score and RNG states. A run restarts from it automatically.

5. **Deep supervision: off.** It is a config flag. When switched on, side heads on D2–D5 predict **only the large classes** (cargo_cover, parachute_bag, honeycomb, skid_board), trained with Dice + focal and the same per-class λ, never clDice. It costs about +0.6 GB per image and about −30% speed; a bigger GPU can be requested if needed.

6. **Augmentation:** only a slight Gaussian blur (σ 0.3–0.7 px) or Gaussian noise (std 2–5 of 255), one of the two, total **p = 0.10**, applied to the image only. Nothing else: no flips, rotation, scale, color jitter or grayscale, since the generator already renders that variety. Color jitter is to be revisited at the sim-to-real stage.

7. **Input normalization: ImageNet mean/std.** Config key `normalization: imagenet | dataset`; the dataset values come from `stats.json`.

8. **Mixed precision:** bf16 autocast on both machines, plus TF32 for the fp32 parts. Losses are computed in fp32. No fp16 fallback for now.

9. **Data loading:**
   - **Turing:** `-c 6` CPUs (4 dataloader workers + 2 for the main process); data staged to `/tmp/$SLURM_JOB_ID`. Local also uses 4 workers.
   - `pin_memory`, `persistent_workers`, `prefetch_factor` 2. Training: shuffle + drop_last. No RAM cache.
   - Batches go to the GPU as uint8 and are converted to float and normalized there.
   - The validation loader uses batch 16.
   - `DistributedSampler` only if multi-GPU is used.

10. **Logging,** per epoch only, to TensorBoard + `log.csv`:
    - every loss term per class, centerline BCE, total
    - learning rate per group, gradient norm before clipping, epoch time, GPU memory peak
    - validation metrics: Dice and IoU per class, clDice (strap, cord), connected-component count, strap centerline coverage
    - 4 fixed validation overlays per validation
    - a console line per epoch
    - Checkpoints are saved from CPU memory; resume loads with `map_location="cpu"`.

## Revisit after the first Turing run
- **Per-class λ:** raise it for classes whose validation Dice lags; lower it for classes that are fine but noisy.
- **Focal scale:** if focal is under about 1% of the total loss, raise one global focal scale.
- **Focal α:** 0.6–0.75 for hardware, tape and data_tag if their recall is low.
- **ResNet-50:** as an ablation, if strap Dice levels off.
- **Gradient clipping:** the local check showed gradient norms of about 15–18 at the start, so 100% of steps were clipped at 1.0. If `train/grad_clipped_frac` stays near 100% after the first epochs, raise the cap (e.g. 5–10).

## Turing HPC (added 2026-10-24)
- **Job:** partition `short` (24 h limit), `--account=nitin`, `--gres=gpu:L40S:1` (48 GB), 6 CPUs, 32 GB RAM. Single GPU, to keep wait times short.
  - Fallbacks: `gpu:A100:1` (80 GB nodes gpu-4-xx) or H100.
  - Avoid `rtx_pro_6000_b`: it is Blackwell and needs CUDA 12.8 or newer.
- **Environment:** a miniconda env or venv in scratch with pip. torch wheel built for cu126 (no CUDA module needed); the rest from `requirements.txt`.
- **Pretrained weights:** fetched once on the login node into `HF_HOME` in scratch. Jobs run with `HF_HUB_OFFLINE=1`.
- **Data:** the job copies `Data/npz`, `index.csv` and `stats.json` to `/tmp/$SLURM_JOB_ID` (node-local, wiped at the end of the job). Checkpoints and logs go to `/home/vchezhian/scratch/Unet/runs/`.
- **Memory measured at 768, bf16, 10 channels + clDice:** about 2.3–2.5 GB per image. Batch 16 ≈ 37–40 GB, which fits an L40S.
- **Local GPU:** RTX 5070 Laptop, 8 GB, so at most batch 3.
- **Data loading:** one npz loads in 17 ms (1 core).

## Dropped
Crossing head and crossing records, physical-strap merge, instance maps and polylines, amodal cord centerline, `defect_expected_region`, depth, camera data, lighting and `background_mode` columns.
