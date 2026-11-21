# Reference, 2026-11-20: UNet 3+ for A-22 cargo packages (state of the project)

Written so that someone else, or a new Claude session, can understand the whole project and continue from here.
Details per step are in `exploration/run1..run4/` and `results_r1.md`; this file is the map.

## 1. Where things stand
- **Goal:** segment the parts of a rigged A-22 cargo bag (webbing straps, cords, hardware, tape, cargo cover, ...) and the strap and cord **centerlines**, from one RGB image (768×768), as the base for later defect detection. Training data is synthetic (Blender renders on photo backgrounds); the target is real photos.
- **Current best model:** `runs/v3_finetune/best.pt` (r3, epoch 15, 106 MB). It is **not on `main`**, because it exceeds GitHub's 100 MB file limit. Its `config.yaml` and `log.csv` are committed. The checkpoint is on the local branch `checkpoint-r3` (commit `7a660a6`) and at `/scratch/vchezhian/Unet/runs/v3_finetune/best.pt` on Turing; share it as a release asset or a file link.
  - It is the model that was **trained sequentially over the datasets**: run 1 (`Data`) → r2 (`Data_v2`) → r3 (`Data_v3`). Note: r1 (`Data_v1`) was a separate fine-tune of run 1 and is not part of this chain, because r2 started from run 1's weights, not r1's.
  - Per the user's review of the real-photo overlays, it has **the best results on real data so far**. This is a visual judgement; no metric on real photos exists (there is no ground truth for them).
- **Comparison model:** `fromscratch_a22` (same data, classes and losses as r3, but started from the ImageNet encoder). Its validation numbers are about equal to r3's.
- **Open problem:** real photos. Hardware and tape are predicted as 0.0% on every real photo and webbing coverage is low; most likely the render-to-photo appearance gap (see section 8).
- **Assistant preferences** (`CLAUDE.md` and the user's global rules): be brief, no filler; long explanations only for "explain in detail" ("explain" = briefly); ask before making changes or running anything; suggest options first when asked to implement; ask instead of assuming; do not modify existing code without approval; flag anything that looks wrong; offer improvements when there is scope for a large quality or run-time gain. Never modify the source datasets in `/scratch/mhameed/...`. No heavy compute on the login node (use `sbatch` or `srun`). State the requested resources before submitting a job. Commit to `main` only when asked; push only when asked.

## 2. Naming of runs
The notes use two numberings. `exploration/run<N>/` is a documentation round; `r1`, `r2`, `r3` are the fine-tune trainings.

| Doc round | Training name | Data | Output dir | Job |
|---|---|---|---|---|
| run 1 | "run 1" / turing | `Data` (`dataset_main`, 100 packages) | `runs/turing` | first Turing run |
| run 2 | **r1** | `Data_v1` (`variant_001` composites) | `runs/v1_finetune` | 2494741 |
| run 3 | **r2** | `Data_v2` (34 background photos) | `runs/v2_finetune` | 2497658 |
| run 4 | **r3** and **fromscratch** | `Data_v3` (`single_a22_physical_v13`) | `runs/v3_finetune`, `runs/fromscratch_a22` | 2529511, 2529512 |

## 3. The previous runs in brief
- **Run 1 (`runs/turing`).** Source `dataset_main/`: 100 double-package families × 6 defect states × 8 views = 4,800 Cycles renders; split by package 70/15/15. Multi-label masks (10 channels, overlaps allowed) and an **amodal** strap centerline (projected from the 3D centerlines, kept where the strap is visible or hidden by a nearby occluder). 50 epochs, learning rate 1e-4 / 1e-3, encoder frozen in epoch 1; best epoch 42: validation strap Dice 0.966, cord 0.852, hardware 0.935, tape 0.804, strap clDice 0.981. Plain-colored renders only, with no photo backgrounds. Decisions: `exploration/run1/decisions.md`.
- **r1 (`Data_v1`).** 40,032 EEVEE views (834 packages), backgrounds composited by the dataset's authors: one photo per split (train `papa`, validation `osan`, test `mcentire`). Fine-tuned from run 1, 15 epochs, learning rate 3e-5 / 3e-4, cord λ 1.5. Zero-shot run 1 on it: strap Dice 0.746 (false positives on the photo backgrounds). After fine-tuning: best at epoch 3, validation strap Dice 0.848 (topology precision 0.635), test 0.936; it overfit the single train photo. Results and confusion matrix: `results_r1.md`.
- **r2 (`Data_v2`).** Same foregrounds, recomposited by `src/composite_v2.py` onto **34 photos** (`bg/`, 24 train / 5 validation / 5 test by location group; random crop ≥60% of the short side, flip, no shadow). Same recipe as r1, initialized from run 1 (not r1). Best epoch 12: validation strap Dice 0.976, cord 0.929, hardware 0.962, tape 0.900; background false positives mostly gone. On real photos strap recall was still low (0.6–2.2% of pixels): the remaining gap is the look of real webbing, not the background. Notes: `exploration/run3/changes.md`.
- **Real photos (`A22/`).** 16 phone photos (4000×3000) were center-cropped to a 768 square and used as a qualitative check for every model; r1 predicted cargo_cover over walls and floor, r2 almost no strap.

## 4. Data_v3 (the current dataset) and how it was built
- **Source:** `/scratch/mhameed/shared_dataset/single_a22_physical_v13` (read-only, owned by another user). One A-22 per image (previous data: two), EEVEE, 768×768, orthographic camera, **MS COCO test2017 backgrounds** (one photo per family and view, same across the 6 states). Layout `<split>/f######/v0X/<state>/` with `rgb.png`, `class_map.png` (**one label per pixel**, visible surface only), `object_map.png` (= `instance_map.png`, uint16), `foreground.png`, `depth.exr`, `annotations.json`, `background.json`, `complete.json`; `registry.json` maps objects to class and `component_id`.
- **Classes in the source:** 0 background, 1 cargo_cover, 2 webbing, 3 cord_or_tie, 4 attachment_hardware, 5 tape, 6 skid_board, 7 honeycomb, 8 parachute_bag, 9 data_tag. No pack_band. No centerline ground truth is provided.
- **States per view:** intact, E01_missing_pack, E03_missing_upper_lateral, E04_missing_support_pair, E05_missing_front_skid_ties, E13_missing_data_tag (geometry identical across states, so near-duplicates).
- **Partial dataset:** 903 of 1000 families, 43,318 of 48,000 samples (the 26 missing test samples have no folders).
- **Output `Data_v3/`** (git-ignored, about 49 GB, regenerate with the jobs below): 43,318 samples, 0 skipped. train 700 families / 33,600 samples, validation 150 / 7,200, test 53 / 2,518.

### Preprocessing: `src/preprocess_v3.py` (with the why)
| Step | What | Why |
|---|---|---|
| Mask | 8 sigmoid channels from `class_map`: `webbing, cargo_cover, cord_or_tie, attachment_hardware, tape, skid_board, honeycomb, parachute_bag` | keeps the multi-label sigmoid head of earlier runs (head remapping works), no background channel |
| data_tag | not a channel; its pixels are saved separately (`data_tag` key) and **ignored** in every loss term, metric and confusion matrix (not counted as background) | the model should not predict it, but those pixels must not be taught as background |
| Hardware | **convex hull per buckle object**, overriding what lies inside | the source labels buckles as thin open frames; on 40 test views (372 buckles) per-instance hole-filling changed 1%, the hull filled them (14,705 → 34,298 px) |
| Centerlines | `webbing_cl_1px/3px`, `cord_cl_1px/3px`: per registry `component_id`, close with a 7 px kernel only into webbing ∪ cord ∪ tape ∪ hardware pixels, skeletonize, remove end spurs shorter than 8 px (webbing) / 5 px (cord); the 3 px target is the 1 px dilated with a 3×3 kernel | no GT provided; this bridges gaps under crossings without joining separate straps; visible centerlines only |
| Backgrounds | the COCO backgrounds are kept (no recompositing) | decided by the user: they are what the dataset provides |
| Missing samples | a sample is used only if `rgb.png, class_map.png, object_map.png, annotations.json, background.json` exist; missing samples are skipped, never the whole family | decided by the user; the count is based on the finished data |
| Output | `npz/<family>__<view>__<state>.npz` (image, mask, data_tag, 4 line maps), `meta/<family>.json`, `index.csv`, `skipped.csv`, `stats.json` (mean/std, class fractions, `pos_weight`) | |
- Fractions (train): webbing 7.0%, cargo_cover 21.4%, cord 2.0%, hardware 0.12%, tape 0.03%, skid_board 1.3%, honeycomb 3.1%, parachute_bag 6.9%; `pos_weight` 29.7 (webbing) and 30 (cord, capped).
- **Run it:** `sbatch jobs/preprocess_v3_test.sbatch` (2 families, with PNG overlays: look at them first), then `bash jobs/preprocess_v3_submit.sh` (4-task array, 16 CPUs / 32 GB each, then the finalize job, 32 CPUs / 64 GB). About 0.4 s per sample per core; 5 min for the array. It resumes (finished families are skipped).

## 5. Classes, model and losses
- **Model (`src/model.py`):** UNet 3+ (full-scale skip connections, 64 channels per branch, fused width 320), timm `resnet34` encoder (`resnet34.a1_in1k`), BatchNorm everywhere. Deep supervision is off. Heads on the full-resolution D1 output: a `mask_head` with 8 channels (sigmoid, multi-label) and a `centerline_head` with 2 channels (`webbing_cl_3px`, `cord_cl_3px`). Input is full 768×768, ImageNet-normalized; no tiling.
- **Per-class loss:** λ × (soft Dice + focal), focal γ = 2, no α. λ: webbing 1, cord_or_tie 1.5, attachment_hardware 1, tape 0.5, cargo_cover / parachute_bag / honeycomb / skid_board 0.25.
- **clDice** (topology): webbing 2 (5 iterations, `webbing_cl_1px`), cord_or_tie 2 (3 iterations, `cord_cl_1px`).
- **Centerline BCE:** weight 1, `pos_weight` = neg/pos of the 3 px target, capped at 30 (a thin line is a tiny fraction of pixels).
- **Consistency loss (new in r3, weight 0.5):** per centerline channel, Σσ(cl)·(1 − max σ(webbing, cord_or_tie, tape, attachment_hardware)) / Σσ(cl): the share of the predicted centerline lying outside the predicted support. It is not detached, so mask and centerline are learned together (answer to "can the centerline be tied back to the strap or cord it belongs to?").
- **Ignore map:** data_tag pixels are masked out of Dice, focal, clDice, BCE and the consistency term.
- **Metrics (`src/metrics.py`):** per-class Dice and IoU, clDice with topology precision (`tprec`) and sensitivity (`tsens`), centerline coverage. Connected-component counts are still logged but **not used** for v3: webbing is labeled visible-only, so every crossing cord or tape splits a strap (about 60 components per GT image against about 9 for the old amodal strap).
- **Why this design:** thin structures need Dice and clDice rather than plain BCE; focal handles the class imbalance; the λ values come from the original plan (`exploration/run1/decisions.md`) and were kept so runs stay comparable.

## 6. Training setup (`jobs/train_turing.sbatch`, `src/train.py`)
- AdamW, weight decay 1e-2 (not on BN/bias), cosine to 1%, gradient clipping 1.0, bf16 autocast (losses in fp32), batch 16 on one GPU, seed 42, blur/noise augmentation with p = 0.10 (nothing else: the generator already supplies the variety). Full validation every 3 epochs; `best.pt` by validation webbing Dice; snapshots every 3 epochs. Resubmitting the same command resumes from the newest snapshot.
- **r3 (`configs/v3_finetune.yaml`):** fine-tune from r2 `runs/v2_finetune/best.pt` with **head remapping by class name** (`train.init_head_map`): webbing ← strap, attachment_hardware ← hardware, the other classes ← the same name, webbing centerline ← strap centerline, the cord centerline head starts fresh; everything else loads exactly. Learning rate 3e-5 / 3e-4, no encoder freeze, warmup 0.3 epoch, 15 epochs.
- **fromscratch (`configs/fromscratch_a22.yaml`):** no `INIT`; the encoder starts from the timm ImageNet weights, the decoder and heads are random. Learning rate 1e-4 / 1e-3, encoder frozen in epoch 1, warmup 1 epoch, 30 epochs. Made to compare the value of the r2 initialization. No smoke test and no consistency ramp, by decision.
- **Memory:** the data (49 GB) is staged to the node's `/tmp`, which is RAM-backed, so the job needs `--mem=64G`. An earlier r2 attempt died at 48 GB. GPU peak is 37.5 GB, so the GPU must have at least 48 GB.
- **GPUs:** this PyTorch (2.14.1+cu126, up to sm_90) works on L40S, A100-80G, H100, H200. It does not work on RTX PRO 6000 Blackwell (sm_120), V100 or A30 (24 GB). Use `--gres=gpu:1 --constraint="L40S|A100-80G|H100|H200"`.
- **Multi-GPU:** `train.py` supports DDP on one node (`torchrun`, chosen automatically when more than one GPU is visible). `batch_size` is per GPU, there is no SyncBN; untested.

```
# r3
CONFIG=configs/v3_finetune.yaml DATA=Data_v3 INIT=runs/v2_finetune/best.pt sbatch -t 12:00:00 --mem=64G \
  --gres=gpu:1 --constraint="L40S|A100-80G|H100|H200" jobs/train_turing.sbatch
# fromscratch
CONFIG=configs/fromscratch_a22.yaml DATA=Data_v3 sbatch -t 24:00:00 --mem=64G \
  --gres=gpu:1 --constraint="L40S|A100-80G|H100|H200" jobs/train_turing.sbatch
# smoke test of r3 first (2 x 50 steps): CONFIG=configs/v3_finetune_smoke.yaml (same flags, INIT as r3)
```

## 7. Results for both versions of r3 (validation, 7,200 views)
| class | r3 ep 15 (best) | fromscratch ep 15 | fromscratch ep 30 (best) |
|---|---|---|---|
| webbing | 0.9647 | 0.9655 | 0.9673 |
| cargo_cover | 0.9924 | 0.9932 | 0.9935 |
| cord_or_tie | 0.9144 | 0.9110 | 0.9149 |
| attachment_hardware | 0.9837 | 0.9831 | 0.9832 |
| tape | 0.9719 | 0.9729 | 0.9728 |
| skid_board | 0.9934 | 0.9940 | 0.9943 |
| honeycomb | 0.9962 | 0.9964 | 0.9965 |
| parachute_bag | 0.9809 | 0.9641 | 0.9734 |
- clDice webbing / cord: 0.979 / 0.938 for both. Train total loss: r3 0.428, fromscratch 0.407 (epoch 15) and 0.376 (epoch 30). r3 ran about 18.5 min/epoch, fromscratch about 23.7 min.
- At equal compute the runs are within about 0.003 Dice except on parachute_bag (noisy). The r2 initialization does not clearly beat the ImageNet encoder on this data. On validation the models over-predict slightly (topology precision < sensitivity), so they are not conservative there.
- **Not done:** test-split evaluation and confusion matrices for r3 / fromscratch (`jobs/eval_turing.sbatch`, `jobs/confusion_turing.sbatch`). Plots: `runs/val_dice_r3_vs_fromscratch.png`, `runs/train_loss_r3_vs_fromscratch.png` (git-ignored; no validation loss is logged, only metrics every 3 epochs).

## 8. Real photos
- `A22/` (git-ignored): 16 phone photos (4000×3000, EXIF orientation 6) and 2 webcam photos (`WIN_20261117_*`, 2160×3840, portrait), plus videos and `.blend` files (not used).
- **Processed inputs `runs/a22_real/input_768/` (18 PNGs, committed):** phone photos: EXIF orientation, center square crop (500 px cut from top and bottom), Lanczos to 768. Webcam photos: **padded, not cropped** (resize to 432×768, **black** side padding of 168 px), because a crop would cut off the parachute bag and the honeycomb with the ties. (Reflect padding was tried and rejected: it mirrored the package.)
- **Inference:** `sbatch jobs/predict_real_v3.sbatch` (1 GPU, 8 CPUs, 16 GB, 20 min; needs a GPU the PyTorch supports; the job has no GPU constraint, so on a Blackwell node it would fail). It runs `src/predict_images.py` with r3 and fromscratch on the folder and writes `<name>_overlay.png` (webbing centerline white, cord centerline cyan) plus per-class probability PNGs to `runs/a22_real/pred_r3/` and `pred_fromscratch/`.
- **Observation:** attachment_hardware and tape come out at 0.0% on every photo, and webbing is 1–5% (7% in training; the image fractions include the black padding for the webcam photos). Probably domain gap; the probability maps have not been inspected. Not tried yet: threshold 0.3 on photos, color/brightness/texture augmentation.

## 9. Environment and how to reproduce
- `environment.yml` (conda + pip; Python 3.11, torch 2.14.1+cu126, torchvision 0.29.1+cu126, timm 1.0.30, numpy 2.4.6, scipy 1.17.1, scikit-image 0.26.0, opencv-python-headless 5.0.0.93, pandas 3.1.0, matplotlib 3.11.2, tensorboard 2.21.0, pytest). Create it with `conda env create -f environment.yml` or `-p $REPO/env`; `requirements.txt` is the short pip list.
- **Turing specifics:** `module load miniconda3`; the sbatch files use `REPO=/home/vchezhian/scratch/Unet`, `--account=nitin`, partition `short` (also `quick` 12 h, `long` 7 days) and `HF_HOME=/home/vchezhian/scratch/hf_cache HF_HUB_OFFLINE=1`. **Change these for another user.** Download the encoder once on the login node: `timm.create_model("resnet34.a1_in1k", pretrained=True)` with `HF_HOME` set; jobs run offline.
- **Tests:** `pytest` (21 tests; run through `srun`, not on the login node).
- **To use the committed model:** `python -m src.predict_images --config runs/v3_finetune/config.yaml --checkpoint runs/v3_finetune/best.pt --images <folder of 768-multiple-of-32 images> --out <dir>`.
- **Not in git:** `Data*/` (regenerable), `bg/`, `A22/`, `runs/` except the files listed in section 10 (so also the r3 `best.pt`), `env/`, and the r2 weights `runs/v2_finetune/best.pt` (needed as `INIT` to repeat r3 exactly; the fromscratch config needs none).

## 10. File map
- `src/`: `model.py`, `losses.py`, `data.py`, `train.py`, `metrics.py`, `infer.py` (evaluation), `confusion.py`, `predict_images.py`, `preprocess.py` (run 1), `preprocess_v1.py`, `composite_v2.py`, `preprocess_v3.py` (current).
- `configs/`: `default.yaml` → `turing.yaml` → `v1_finetune.yaml` → `v2_finetune.yaml` → `v3_finetune.yaml` (each uses `base:`) → `v3_finetune_smoke.yaml`; `fromscratch_a22.yaml` inherits `v3_finetune.yaml`. The resolved config of every run is saved as `runs/<run>/config.yaml`.
- `jobs/`: `train_turing.sbatch`, `eval_turing.sbatch`, `confusion_turing.sbatch`, `predict_real_v3.sbatch`, `preprocess_v3*.sbatch` and `preprocess_v3_submit.sh`, plus the older `preprocess_v1*`, `composite_v2*` and `overfit_turing.sbatch`.
- `exploration/run1..run4/`: decisions and changes per round (`run4/changes.md` is the most detailed record of Data_v3 and r3). `results_r1.md`: r1 results.
- Committed binaries: `runs/a22_real/input_768/`. Committed with the checkpoint: `runs/v3_finetune/{config.yaml,log.csv}`. `best.pt` is not on `main` (size limit), see section 1.

## 11. Suggestions for the next session
1. Evaluate r3 and fromscratch on the Data_v3 test split and build confusion matrices (cheap: 1 GPU, about 1 h).
2. Inspect the real-photo probability maps; try threshold 0.3 on photos.
3. Color, brightness and texture augmentation to close the render-to-photo gap (the main open problem).
4. A zero-shot r2 baseline on Data_v3, to measure what fine-tuning added.
5. Add a vis panel for cord and hardware, and different families in the validation overlays (the current four are near-duplicates).
