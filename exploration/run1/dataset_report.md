# dataset_main — Phase 1 report

Sampled: `family_000066_E13/view_01` (validation split) and `family_000072_E13/view_05` (train split), picked with `random.Random(0)`. Both happened to be E13 (missing data tag), and E13 is not observable in either view. So I made one extra read-only check: `defect_expected_region.png` in `family_000072_E04/view_05` and `family_000072_E01/view_05`, to see how the defect mask is encoded. I looked at nothing else in the dataset. Dataset-wide numbers come from the top-level JSON files.

Visualizations: `exploration/family_000066_E13__view_01.png` and `exploration/family_000072_E13__view_05.png` (full view), plus `*__zoom.png` (192 px crop at 3× zoom).

## 0. What the existing code expects (src/, configs/)
- **Index:** `<root>/metadata.csv` with columns `image_id`, `split` ∈ {`train`,`val`,`test`}, `tiff_image_path`, `tif_label_path` (relative to root). Read with `tifffile`.
- **Mask:** a single 0/255 file, binarized with `> 127`. There is one target channel (`strap`) even though `model.classes` can be longer: the loader always returns a `(1,H,W)` mask.
- **Centerline:** `skeletonize(mask)` dilated to 3 px, cached as 0/255 PNG at `<cache_dir>/centerline/<image_id>.png`. Stats go in `<cache_dir>/stats.json` (`centerline_neg_over_pos` → `pos_weight`, capped at 30).
- **Train / eval:** training uses random 512 crops (skipping crops where >50% of pixels are white) with flips, rot90 and small color jitter, plus ImageNet normalization. Validation and test use sliding-window inference (512 tiles, 25% overlap).
- **Model:** UNet 3+ with a ResNet-34 encoder. Output is a dict: `mask` = `len(classes)` logits, `centerline` = 1 logit, and optional `side` outputs (deep supervision, off by default).
- **Losses:** Dice, focal and clDice take `out["mask"]` against the mask. Weighted BCE takes `out["centerline"]` against the centerline. All loss weights are 1.0, focal α = 0.25, γ = 2, clDice uses 5 iterations.
- **Training settings:** batch size 8, 60 epochs, lr 1e-4 (encoder) and 1e-3 (decoder), warmup + cosine, bf16. There is no DDP, no resume and no gradient accumulation.

## 1. Folder layout
```
dataset_main/
  classes.json               label map (below)
  component_registry.json    440 objects -> class_id, instance_id (global, same for every folder)
  manifest.json              600 inspections: family_id, split, state, defect_ids, per-view checks
  splits.json                train/validation/test lists of inspection folders
  generation_config.json     generator settings (view layout, ranges); mostly irrelevant
  validation_report.json     generator self-check: passed; summary counts
  checksums.json, run_summary.json, scene_check_report.json   integrity only; irrelevant
  inspections/<family_XXXXXX>_<state>/
    assembly.json            per-inspection: split, state, removed objects, defect targets, scene params
    view_01 .. view_08/      14 files each (below)
```
- **Terminology.** Each folder in `inspections/` is `family_<id>_<state>`. The dataset's `family_id` is the physical package (100 of them). Each package is rendered in 6 **states**: `baseline` plus `E01`, `E03`, `E04`, `E05`, `E13`. That gives 600 folders.
- The states of one package share the same camera, background and clutter. They are near-identical images that differ only by the removed component.

## 2. Images
`rgb.png`: 768×768, 3 channels, 8-bit PNG. It is antialiased and denoised. Brightness varies with the background mode: the mean RGB is about (111, 129, 113) for a "similar" background and about (187, 187, 170) for a "contrast" background. There are no white no-data regions.

## 3. Ground truth (per view)
Label boundaries are hard, non-antialiased object-index labels from Cycles.

| file | encoding | content |
|---|---|---|
| `class_map.png` | uint8, 0–9 | semantic class per pixel (label map below) |
| `object_map.png` | uint16, 0–440 | object ID (registry `object_id`) |
| `instance_map.png` | uint16, 0–178 | instance ID (registry `instance_id`; stitches and bar tacks are merged into their strap) |
| `strap_mask.png` | 0/255 | exactly `class_map ∈ {2 webbing, 4 attachment_hardware}` (verified) |
| `hardware_mask.png` | 0/255 | exactly class 4 |
| `cord_mask.png` | 0/255 | exactly class 3 |
| `centerline_mask.png` | 0/255 | **1 px wide**, webbing only. It lies 100% inside `strap_mask` and never on hardware |
| `centerline_instance_map.png` | uint16 | instance ID along the centerline. It is equal to `instance_map` at every centerline pixel |
| `cord_centerline_mask.png` | 0/255 | 1 px cord centerline. It lies inside `cord_mask` |
| `defect_expected_region.png` | 0/255 | pixels where the *removed* component would be visible in the baseline. It is empty when the defect is not observable |
| `depth.exr` | float EXR | camera-axis depth in metres, 0 = invalid. The installed OpenCV cannot read it and there is no OpenEXR package. Not needed for this task |
| `annotations.json` | — | camera K and extrinsics, per-view defect checks, per-object visibility, background and clutter |

The sampled views agree with themselves and with each other: `class_map` and `instance_map` both equal the registry lookup of `object_map`.

**Label map** (`classes.json`): 0 background (floor, backdrop, clutter), 1 cargo_cover, 2 webbing, 3 cord_or_tie, 4 attachment_hardware, 5 tape, 6 skid_board, 7 honeycomb, 8 parachute_bag, 9 data_tag. `strap_mask_class_ids = [2, 4]`.

**Instances**
- Instance IDs are global, so the same physical part has the same ID in every view and every package.
- There are 55 webbing instances: lateral wraps Upper/Middle/Lower, support webs, end supports, top webs, and LVCP pack bands. There are 33 hardware instances: center joins, end adapters, D-rings, snap hooks and an O-ring.
- `classes.json` notes: "instance = modeled component segment, not a physical strap inventory". One physical strap can therefore be split into several instances.

**Seal / buckle**
- There is no `seal` class.
- Buckles and adapters are `attachment_hardware`. They have their own class and instances, and they are included in `strap_mask`.
- They render as thin rectangular outlines and cover about 0.1–0.2% of the image.

**Occlusion and crossings**
- Each pixel carries the label of the visible surface only (z-buffer).
- At a crossing, the strap on top owns the pixels and the strap underneath is cut. Lateral wraps break into up to 38 connected components per view.
- The centerline is depth-filtered with a 15 mm tolerance, so it also breaks wherever its strap is hidden.

**Centerlines**
- They are provided, projected from the procedural ribbon geometry, and are not skeletonized.
- 3D polylines per object are also in `assembly.json` → `centerlines_world` (55 webbing + 76 cord).
- In each view, 1–2 visible webbing instances have no centerline pixels (tiny or edge-on).

## 4. JSON schema (relevant fields only)
- `manifest.json` → `inspections[]`: `inspection_id`, `family_id`, `split`, `state`, `defect_ids`, `views[].checks[]` (`check_id`, `scene_fault_present`, `label_state`).
- `assembly.json`: `split`, `state`, `defect_ids`, `removed_object_ids`, and `inspection_targets.<E>` (`removed`, `evidence` object IDs, `description`). `parameters` holds:
  - `background_mode`: seen values `similar` and `contrast`. The config has 3 background weights (0.4/0.4/0.2), so a third mode probably exists but was not seen.
  - lighting, all continuous: `light_rotation`, `light_multiplier`, `light_size`, `light_color`, `fill_multiplier`, `world_strength`.
  - colors, `wrinkle_strength`, `yaw_degrees`.
- `annotations.json`: `camera.K` (3×3), `world_to_camera_cv` (4×4), `camera_to_world_blender`, `environment.background_mode`, `checks[]`, and `object_visibility[]` (`visible_pixels`, `visible_centerline_pixels` per object).
- Irrelevant: checksums, render device and time, code hashes, clutter box details, `generation_config` ranges.

## 5. Defects
| state | defect | removed |
|---|---|---|
| baseline | none (defect-free, 100 inspections) | — |
| E01 missing_pack_assembly | pack + lines, hooks, tape, ties | 52 objects |
| E03 missing_upper_lateral | upper lateral strap | 68 objects |
| E04 missing_end_support | one end support web + stitches | 2 objects |
| E05 missing_skid_tie_group | one skid tie group | 12 objects |
| E13 missing_data_tag | data tag + text | 5 objects |

- **All defects are missing components.** There is no damage, misrouting or twisting.
- **How defects are labeled:**
  - per inspection: `state` and `defect_ids`
  - per view: `label_state` ∈ {`fault_present`, `not_observable`, `checked_no_fault_observed`}
  - per pixel: `defect_expected_region`
- **Observable fault views (of 800 each):** E01 800, E03 800, E04 301, E05 324, E13 300.
- **The GT reflects the defective scene:** removed parts are absent from every mask. For example, E03 has no upper-lateral strap pixels.

## 6. Size and splits
- **Size:** 100 packages × 6 states × 8 views = **4800 views**, 9.0 GB. Per view, `depth.exr` is about 0.8–1.1 MB, `rgb.png` about 0.6 MB, and the masks are small.
- **Splits already exist** (`splits.json`, `manifest.split`): split **by family_id**, so all 6 states of a package stay together, in a 70/15/15 ratio of packages:
  - train: 70 packages = 3360 images
  - validation: 15 packages = 720 images
  - test: 15 packages = 720 images
  - verified: no package appears in more than one split
- **Class balance (two views):**
  - strap_mask is 3.4–5.0% of pixels, of which hardware is about 0.1–0.2%.
  - The centerline is 0.57–0.66% of pixels. The generator reports a median of 3644 px. That gives neg/pos ≈ 150, so `pos_weight` hits the cap of 30.
  - Foreground (the package) is about 20–23% of the image.

## 7. Issues / surprises
1. **"Family" means two things.** A folder is one state of one package, and the 6 folders of a package are near-duplicates. A split by folder would leak, so the split must use `family_id`. The provided split already does this.
2. **`strap_mask` includes hardware, but the centerline does not.** As a result the centerline has gaps at every buckle and adapter.
3. **The provided centerline is 1 px wide, while the code trains on a 3 px dilated skeleton.**
4. **Occlusion fragments strap GT heavily** (up to 38 components for one strap). The code's connected-component count metric will be large on GT too, and it measures occlusion, not breaks.
5. **There is no seal class.** `attachment_hardware` is the closest equivalent.
6. **Webbing includes more than the main straps:** bar tacks and sewn edges (merged into their strap instances), and LVCP pack bands on the parachute bag. Tape (class 5) and cords (class 3) are not in the strap mask.
7. **Code and data formats don't match:**
   - The code reads TIFF and `metadata.csv`; the data is PNG with its own manifest.
   - The split is named `validation` here, not `val`.
   - The white-crop filter does nothing on this data.
   - The loader returns only 1 target channel.
8. **Things I noticed outside the sample:**
   - `depth.exr` can't be read with the current environment.
   - `archive (1).zip` (6.2 GB) sits in the project root. It is probably the dataset archive; I didn't touch it.
   - Nothing was missing, mis-sized or unexplained in the sampled views.
