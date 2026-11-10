# Run 2: changes from run 1

Run 1 records are in `../run1/`. Everything in run 1 still applies, except what is listed here.

## Dataset
- **Source:** `/scratch/mhameed/shared_dataset/composites/variant_001`, read-only. The layout, label map, registry and GT encodings are the same as in `dataset_main/`, and the pack-band instance IDs are still {113–116, 125, 126}.
- **Size:** 834 packages × 6 states × 8 views = **40,032 views** (5004 inspections). The split is still by package: 28,032 train / 6000 validation / 6000 test.
- **Renderer:** EEVEE with Cryptomatte labels (run 1 used Cycles).
- **Backgrounds:** real photos composited after rendering. `background_mode` is always `transparent`. Each split has its own photo, so the test background is never seen in training:

| split | views 01–04, 07–08 (elevation 0–25°) | views 05–06 (elevation 25–65°) |
|---|---|---|
| train | `papa_scene` | `papa_ground` |
| validation | `osan_scene` | `osan_ground` |
| test | `mcentire_scene` | `mcentire_ground` |

  The `*_scene` backgrounds use the whole photo. The `*_ground` backgrounds use only the bottom 40% of it, as a backdrop for the elevated views.
- **New per-view files:** `foreground.png` (RGBA render), `package_mask.png` (equal to `class_map > 0`) and `complete.json` (checksums). New top-level items: `previews/`, `index.html`, `contact_sheet.jpg`. None of these are used.

## Preprocessing
- **Script:** `src/preprocess_v1.py`, a new file. It reuses the target builders of `src/preprocess.py` unchanged, so the targets are identical to run 1.
- **Output:** written to `Data_v1/`, which is in `.gitignore`. `Data/` is kept.
- **npz only.** PNGs are written only with `--png`, and were used for the test run alone (`Data_v1/png/`, kept).
- **Resume:** `Data_v1/meta/<inspection_id>.json` holds the per-view centerline coverage and `background_id`. It is written last and marks the inspection done.
- **`index.csv`:** the `png_dir` column is dropped and a `background_id` column is added.
- **Turing jobs (CPU only, no GPU):**
  - `jobs/preprocess_test.sbatch`: the test run on `family_000001_baseline` and `family_000001_E03`, with PNGs
  - `jobs/preprocess_v1.sbatch`: a 4-task array (`--array=0-3`), each task 16 CPUs, 32 GB, 1 h, running `--shard i --nshards 4`
  - `jobs/preprocess_v1_finalize.sbatch`: builds `index.csv`, `qa.csv`, `defect_states.json` and `stats.json`
  - `jobs/preprocess_v1_submit.sh`: submits the array, then the finalize step to run once the array succeeds

## Results (2026-11-09; jobs 2493679 array, 2493680 finalize)
- **Time:** about 0.35 s per view per core. The array tasks ran 3–4 minutes and the finalize step 3 minutes.
- **Output:** 40,032 npz files, none missing, no leftover `.tmp` files, 38 GB.
- **Centerline coverage:** 1.0 on every view.
- **`stats.json` (train split, 28,032 views):**
  - image mean RGB (0.338, 0.331, 0.315), std (0.160, 0.159, 0.153)
  - strap pixel fraction 4.8%
  - strap centerline fraction 0.71% at 1 px, 2.17% at 3 px
  - neg/pos 45, so `pos_weight` = 30 (the cap)
- **Not yet verified (data):** split counts and family leakage in `index.csv`, defect-visible counts against `validation_report.json`, and loading of the validation and test npz files. These should run as an sbatch job, not on the login node.

## Zero-shot check: run 1 `best.pt` on `Data_v1` validation (job 2494198)
| metric | run 1 on old validation | on `Data_v1` validation (6000 views) |
|---|---|---|
| strap Dice / topology precision / sensitivity | 0.966 / 0.980 / 0.982 | 0.746 / 0.523 / 0.957 |
| connected components, predicted / GT | 5.3 / 10.3 | 246 / 9.3 |
| cord Dice | 0.852 | 0.507 |

- **Cause:** false positives on the photo backgrounds (beams, ducts, pillars). Recall on the package stays high.
- **By view:** views 05/06 (`*_ground` crop) score strap Dice 0.869; the other views (`*_scene`) score 0.705. There is no difference between defect states.
- Results are in `runs/turing/eval_v1_validation/`.

## Training (run 2): fine-tune only
- **Initialization:** `runs/turing/best.pt` (run 1, epoch 42), weights only. The optimizer and schedule start fresh. When a snapshot exists, the run resumes from it and `--init-checkpoint` is ignored.
- **Config:** `configs/v1_finetune.yaml` (inherits `turing.yaml`). Differences from run 1:
  - 15 epochs (1752 steps/epoch, 26k steps in total)
  - learning rate 3e-5 (encoder) / 3e-4 (decoder + heads), cosine down to 1%
  - no encoder freeze; warmup 0.3 epoch (about 525 steps)
  - cord_or_tie λ 1.0 → 1.5 (Dice + focal); clDice weights unchanged (strap 2, cord 2)
  - validation on the full validation split (6000 views) every 3 epochs; `best.pt` is chosen by validation strap Dice
  - snapshots every 3 epochs, all kept, in `runs/v1_finetune/snapshots/`
  - 8 dataloader workers
  - unchanged: ImageNet normalization, blur/noise augmentation (p 0.10), `pos_weight` 30, bf16, batch 16 on 1× L40S, BatchNorm trains
- **Job:** `jobs/train_turing.sbatch` now uses `-c 10` and 48 GB, plus the `DATA` and `INIT` environment variables. It stages the data to `/tmp` if there is room for the data plus 10 GB, otherwise it reads from `/scratch`, and it logs which one it used. The time limit is 12 h.
  - `CONFIG=configs/v1_finetune.yaml DATA=Data_v1 INIT=runs/turing/best.pt sbatch -t 12:00:00 jobs/train_turing.sbatch`
- **Estimate:** about 20 min per epoch plus about 6 min per validation, about 5.5–6 h in total.
- **Smoke test first:** `configs/v1_finetune_smoke.yaml`, 2 epochs × 50 steps, a 100-view validation and a snapshot every epoch, output to `runs/v1_finetune_smoke/`.
- **Code changes:**
  - `src/train.py`: `--init-checkpoint`, `train.snapshot_subdir`, `train.max_steps_per_epoch`, fractional `warmup_epochs` rounded to steps, per-epoch `train/data_wait_s` (time spent waiting on the loader), and strap/cord topology precision (`tprec`, which tracks false positives) in the console line
  - `tests/`: 3 new tests
- **No test-set evaluation** for this run.

## Other changes
- **`jobs/eval_turing.sbatch`** (new): evaluates a checkpoint on one split; `CKPT`, `SPLIT`, `DATA` and `OUT` are set as environment variables. Used for the zero-shot check.
  - `CKPT=runs/turing/best.pt SPLIT=validation DATA=Data_v1 OUT=runs/turing/eval_v1_validation sbatch jobs/eval_turing.sbatch`
- **Records:** the run 1 files (`dataset_report.md`, `decisions.md`, visualizations) were moved from `exploration/` to `exploration/run1/`. Run 2 notes are in `exploration/run2/`.
