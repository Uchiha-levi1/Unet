# Task: UNet 3+ for thin-structure segmentation on Massachusetts Roads

I'm developing a segmentation network for webbing straps on cargo. Before my own
synthetic data is ready, I want to validate the network and losses on the
Massachusetts Roads dataset (Kaggle mirror by balraj98). Build a clean, configurable
PyTorch project that trains UNet 3+ on it with the losses below.

## 0. Inspect the data first
The dataset is at `./data/massachusetts-roads/` (change if needed). Expected layout:
`tiff/{train,train_labels,val,val_labels,test,test_labels}`, `metadata.csv`
(image_id, split, sat_image_path, map_image_path), `label_class_dict.csv`.
Before writing training code, verify the layout, split counts (~1108/14/49), image and
mask shapes (1500×1500), mask values (0/255; take one channel if 3), and the fraction
of road pixels. Report what you find and adapt the loader if anything differs.

## 1. Data pipeline
- Binary strap mask: `mask > 127`.
- Centerline target: skeletonize the mask (skimage), then dilate to ~3 px wide.
  Precompute and cache it to disk.
- Training: random 512×512 crops. Skip crops where >50% of image pixels are pure
  white (no-data regions in this dataset).
- Augmentation (albumentations): flips, rot90, small brightness/contrast/hue jitter.
  Apply identical geometric transforms to image, mask and centerline.
- Normalize with ImageNet mean/std.
- Validation/test: full 1500×1500 images via sliding-window inference
  (512 tiles, 25% overlap, center-weighted blending, pad to a multiple of 32).

## 2. Model: UNet 3+
- Encoder: `timm.create_model('resnet34', pretrained=True, features_only=True)`,
  five feature maps E1..E5 (E5 acts as D5).
- Decoder levels D4..D1 with **full-scale skip connections** (Huang et al., 2020):
  each decoder level takes every finer encoder level (max-pooled down), the encoder
  level at the same scale, and every deeper decoder level (bilinear upsampled).
  Project each input to 64 channels with 3×3 conv-BN-ReLU, concatenate (5×64 = 320),
  then fuse with 3×3 conv-BN-ReLU (320 channels).
- Output heads on D1, at full input resolution, raw logits:
  - `strap`: 1 channel
  - `centerline`: 1 channel
  Write the head so the number of class channels is configurable (I will later add a
  `seal` channel, multi-label with sigmoid).
- **Deep supervision: optional, OFF by default** (config flag). If on, add a 3×3 conv
  side head on D2–D5 producing a strap-mask logit, bilinearly upsampled to full
  resolution and supervised against the full-resolution mask with Dice + focal only.
- **No classification-guided module.**

## 3. Losses (each term weighted, weights in config, default 1.0)
- Strap mask:
  - Soft **Dice** loss.
  - **Focal** loss (sigmoid, γ = 2, α configurable).
  - **Soft-clDice** (Shit et al., CVPR 2021): implement the soft skeleton with
    iterative min/max-pooling (configurable iterations, default 5). Apply ONLY to the
    final full-resolution output, never to deep-supervision side outputs.
- Centerline:
  - **Weighted BCE** (`BCEWithLogitsLoss` with `pos_weight`). Compute
    pos_weight = (#neg / #pos) over the training set and cap it (default 30).
- Total = λ_dice·Dice + λ_focal·Focal + λ_cldice·clDice + λ_cl·BCE
  (+ side-output losses if deep supervision is on).
- Log each loss term separately.

## 4. Training
- AdamW with two parameter groups: encoder lr 1e-4, decoder and heads lr 1e-3,
  cosine schedule with warmup.
- Optional encoder freeze for the first N epochs (config, default 0).
- If batch size < 8, keep encoder BatchNorm layers in eval mode (frozen stats);
  re-apply after every `model.train()`.
- Mixed precision (AMP). Gradient checkpointing as an optional flag.
- Save the best checkpoint by validation Dice, plus the last checkpoint.
- Fixed random seeds.

## 5. Evaluation (per image, averaged)
- Dice and IoU on the strap mask (threshold 0.5).
- clDice (hard version, on thresholded masks).
- Connectivity: number of connected components in the prediction vs. ground truth
  (a proxy for broken segments).
- Save a few visual overlays (image / ground truth / prediction / centerline) each
  validation epoch.

## 6. Project structure and usability
- `configs/default.yaml`, plus `configs/no_cldice.yaml` for an ablation
  (λ_cldice = 0).
- `src/data.py`, `src/model.py`, `src/losses.py`, `src/metrics.py`, `src/train.py`,
  `src/infer.py`.
- CLI: `python -m src.train --config configs/default.yaml`.
- Logging to TensorBoard and a CSV.
- `requirements.txt`.

## 7. Checks before full training
- Unit tests: model output shapes for 512×512 input, with and without deep
  supervision; each loss returns a finite scalar and is ~0 for a perfect prediction.
- Overfit test: train on 4 crops for ~200 steps; the loss must go close to zero.
- Then run the full training and the no-clDice ablation, and give me a short table
  comparing Dice, clDice and connectivity.

Keep the code readable and commented where the logic isn't obvious (full-scale skips,
soft skeleton). Don't add features I didn't ask for; ask me if something is ambiguous.
