# Results: r1 (job 2494741)

## Starting conditions
- **Data:** `Data_v1` (`composites/variant_001`): 28,032 train / 6000 validation / 6000 test views at 768×768, split by package. Each split has one background photo: train `papa`, validation `osan`, test `mcentire`.
- **Initialization:** run 1 `runs/turing/best.pt` (epoch 42, trained on `Data_v1`'s predecessor `dataset_main`), weights only. The optimizer and schedule start fresh.
- **Model:** UNet 3+ with a ResNet-34 encoder. 10 sigmoid mask channels (strap, cargo_cover, cord_or_tie, hardware, tape, skid_board, honeycomb, parachute_bag, data_tag, pack_band) plus 1 strap centerline channel. Deep supervision off.
- **Loss:**
  - per class, λ × (Dice + focal) with focal γ 2 and no α. λ: strap 1, **cord_or_tie 1.5**, hardware 1, tape / data_tag / pack_band 0.5, cargo_cover / parachute_bag / honeycomb / skid_board 0.25
  - clDice: strap 2 (5 iterations), cord 2 (3 iterations)
  - centerline: weighted BCE, λ 1, `pos_weight` 30
- **Optimization:**
  - AdamW, learning rate 3e-5 (encoder) / 3e-4 (decoder + heads), weight decay 1e-2, gradient clipping 1.0
  - no encoder freeze, warmup 0.3 epoch, cosine down to 1%
  - 15 epochs × 1752 steps, batch 16 on 1× L40S, bf16, BatchNorm trains
- **Input:** ImageNet normalization. Augmentation: blur or noise, p = 0.10.
- **Validation:** full validation split every 3 epochs; `best.pt` is chosen by validation strap Dice.

## Results

### Validation (6000 views, `osan`)
| epoch | train loss | strap Dice | strap topology precision | cord Dice |
|---|---|---|---|---|
| 3 | 0.379 | **0.848** | 0.635 | 0.747 |
| 6 | 0.319 | 0.830 | 0.597 | 0.744 |
| 9 | 0.285 | 0.834 | 0.609 | 0.764 |
| 12 | 0.265 | 0.839 | 0.613 | 0.762 |
| 15 | 0.258 | 0.834 | 0.606 | 0.763 |

The best checkpoint is from epoch 3.

### Per-class Dice, `best.pt` (epoch 3)
| class | validation (`osan`) | test (`mcentire`) |
|---|---|---|
| strap | 0.848 | 0.936 |
| cord_or_tie | 0.747 | 0.890 |
| hardware | 0.901 | 0.931 |
| tape | 0.642 | 0.731 |
| pack_band | 0.754 | 0.778 |
| data_tag | 0.850 | 0.965 |
| cargo_cover | 0.398 | 0.881 |
| parachute_bag | 0.652 | 0.814 |
| honeycomb | 0.896 | 0.991 |
| skid_board | 0.461 | 0.988 |

| metric | validation | test |
|---|---|---|
| strap clDice (topology precision / sensitivity) | 0.758 (0.635 / 0.984) | 0.905 (0.845 / 0.980) |
| cord clDice (topology precision / sensitivity) | 0.667 (0.536 / 0.949) | 0.882 (0.833 / 0.941) |
| strap centerline coverage | 0.984 | 0.980 |
| strap connected components, predicted / GT | 160 / 9.3 | 76 / 9.5 |

### Confusion matrix, test (6000 views, `best.pt` epoch 3)
Pixel counts over 3.54 billion pixels. Rows are ground truth and columns are predictions; each row is normalized to 100% (recall).

How each pixel gets one label:
- **GT:** the first present channel in the order hardware > cord_or_tie > tape > data_tag > pack_band > strap > skid_board > honeycomb > parachute_bag > cargo_cover; otherwise background.
- **Prediction:** the most likely channel above 0.5; otherwise background.

| GT \ pred | backgr. | hardware | cord | tape | data_tag | pack_band | strap | skid | honeyc. | para. bag | cover |
|---|---|---|---|---|---|---|---|---|---|---|---|
| background | **95.0** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.5 | 0.0 | 0.0 | 0.1 | 4.4 |
| hardware | 1.7 | **91.3** | 0.2 | 0.1 | 0.0 | 0.0 | 6.1 | 0.0 | 0.0 | 0.1 | 0.6 |
| cord_or_tie | 2.4 | 0.0 | **89.5** | 0.2 | 0.0 | 0.2 | 4.0 | 0.1 | 0.7 | 0.2 | 2.5 |
| tape | 2.1 | 0.5 | 1.5 | **88.8** | 0.0 | 0.0 | 6.2 | 0.0 | 0.0 | 0.1 | 0.7 |
| data_tag | 0.2 | 0.0 | 0.0 | 0.0 | **99.1** | 0.0 | 0.6 | 0.0 | 0.0 | 0.0 | 0.2 |
| pack_band | 2.1 | 0.0 | 0.2 | 0.0 | 0.0 | **96.1** | 0.0 | 0.0 | 0.0 | 1.4 | 0.2 |
| strap | 0.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **98.2** | 0.0 | 0.0 | 0.0 | 1.4 |
| skid_board | 0.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **99.1** | 0.2 | 0.0 | 0.0 |
| honeycomb | 0.4 | 0.0 | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | **99.1** | 0.0 | 0.2 |
| parachute_bag | 1.1 | 0.0 | 0.1 | 0.0 | 0.0 | 0.1 | 0.1 | 0.0 | 0.0 | **98.5** | 0.1 |
| cargo_cover | 0.5 | 0.0 | 0.2 | 0.0 | 0.0 | 0.0 | 0.7 | 0.0 | 0.0 | 0.0 | **98.6** |

**Per-class precision and recall from this matrix (%):**

| | backgr. | hardware | cord | tape | data_tag | pack_band | strap | skid | honeyc. | para. bag | cover |
|---|---|---|---|---|---|---|---|---|---|---|---|
| recall | 95.0 | 91.3 | 89.5 | 88.8 | 99.1 | 96.1 | 98.2 | 99.1 | 99.1 | 98.5 | 98.6 |
| precision | 99.8 | 97.3 | 87.9 | 74.8 | 99.1 | 96.7 | 90.0 | 98.7 | 99.5 | 97.1 | 80.7 |
| predicted pixels that are GT background | — | 1.8 | 6.0 | 19.5 | 0.0 | 0.3 | 7.0 | 1.0 | 0.1 | 2.7 | 18.9 |

Overall pixel accuracy: 95.8%.

Raw counts: `runs/v1_finetune/confusion_test.json`. Per-image metrics: `runs/v1_finetune/eval_test/`.
