# Task: Understand my package dataset, preprocess it, and prepare UNet 3+ training

## Background
I'm building a pipeline that segments the webbing straps on a rigged cargo package
(an A-22 cargo bag), separates them into individual straps, and later detects defects.
The data is rendered in Blender. The model and losses are already specified in
`unet3plus_task.md` (UNet 3+, ResNet-34 encoder, strap + centerline outputs,
Dice + focal + clDice + weighted BCE). This task adapts that work to my own dataset.

The dataset is in the folder `dataset_main/`.

What I already know about it:
- Inside `dataset_main/inspections/`, each subfolder is one instance of the package,
  called a **family**.
- Families differ in the defects present, lighting conditions, background, etc.
- Each family folder contains **8 views** from different camera positions, plus their
  ground truths.
- JSON files hold the details (cameras, components, defects, labels, ...). Some are
  verbose and redundant.

Work in **three phases**. Stop at the end of each phase and wait for me.

---

## Before Phase 1: Read the existing code

Start by reading the `src/` and `configs/` folders of my existing UNet 3+ project
(built from `unet3plus_task.md`). Focus on:
- the data loader: expected file layout, file formats, mask encoding (0/1 vs 0/255),
  how centerline targets are stored, and the index/metadata file it reads
- the configs: paths, input size, normalization, loss weights, training settings
- the model's output heads (number and order of channels) and what the losses take
  as input

Summarize what the code expects in a few lines. Use it so that Phase 2 writes data
in the format the loader already reads, and Phase 3 becomes small edits rather than a
rewrite.

Don't let the code limit the dataset: if the dataset contains something the code
doesn't handle (e.g. instance IDs, a seal class), report it in Phase 1 rather than
ignoring it. Don't change any code yet.

---

## Phase 1: Explore and report (read-only)

### Rules for this phase
- **Do not modify, move, or delete anything** in `dataset_main/`.
- **Do not read the whole dataset.** Sample sparingly:
  1. Read the top-level JSON files (outside the families), if any.
  2. Pick **two random families**. In each, inspect **one view**: its image(s), every
     ground-truth file, and every JSON file that belongs to it or to the family.
  3. Treat these two samples as representative of the whole dataset. Only look further
     if something is clearly inconsistent, and tell me if you do.
- For large or verbose JSON files: print the key structure (keys, types, list lengths,
  one example entry), not the full content. Skim; don't dump.
- **Keep only what matters for the task.** Don't carry irrelevant details (redundant
  metadata, render settings that don't affect labels, etc.) into your notes or
  decisions. If a field looks irrelevant, mention it in one line and move on.

### What to find out
- **Folder layout:** naming conventions for families, views and files; file formats.
- **Images:** resolution (expected 768×768), channels, bit depth, format.
- **Ground truth:** what each GT file contains and how it's encoded, for example:
  - instance ID maps, semantic masks, depth, normals, centerlines or keypoints
  - how pixel values map to objects (strap IDs, seal/buckle IDs, cover, skid board,
    cords, background); look for a label map in the JSON
  - whether straps have unique instance IDs that are consistent across the 8 views
  - whether seals/buckles/adapters are separate from strap bodies
  - how occlusion and crossings are handled
  - whether centerlines or strap tangents are provided, or would need to be derived
- **JSON contents:** camera intrinsics/extrinsics, scene/component lists, defect
  descriptions (type, location, severity), lighting/background parameters.
- **Defects:** which defect types exist, how they are labeled (per family, per view,
  per strap, per pixel?), and whether defect-free families exist.
- **Dataset size:** number of families, total images, approximate disk size.
- **Splits:** whether train/val/test splits already exist.
- **Anything surprising:** missing files, mismatched sizes, empty masks, inconsistent
  naming, unexplained label values.

Make small visualizations for the two sampled views (image, GT overlays, instance
IDs in distinct colors) and save them to an `exploration/` folder outside
`dataset_main/`.

### Deliver at the end of Phase 1
1. A concise report: `exploration/dataset_report.md`, with the folder structure, file
   formats, GT encoding, label map, JSON schema summary (relevant fields only),
   defect labeling, dataset size, and issues found.
2. **A list of numbered questions for me** about anything unclear or anything that
   needs a decision (e.g. which label values count as "strap", how to treat
   buckles, which defects matter, how to split the data).

**Then stop and wait for my answers.**

---

## Phase 2: Preprocessing script

After I answer the Phase 1 questions:

1. **Propose a preprocessing plan first and wait for my approval.** Ask me before
   making any decision, especially on:
   - the label mapping (which objects become the `strap` channel and which, if any,
     become `seal`)
   - how centerline targets are produced (from provided geometry, or by
     skeletonizing the strap mask)
   - how to split the data (**split by family, never by view**, so views of the same
     package never appear in both train and test; also decide whether specific
     lighting/backgrounds/defects are held out)
   - output format and storage (e.g. PNG/NPZ per view plus an index CSV)
   - anything that would drop or alter data
2. Then write `preprocess.py`:
   - reads `dataset_main/` without modifying it, writes to a separate output folder
   - produces, per view: the RGB image, the binary strap mask, the centerline
     target, and (if approved) the seal mask and the strap instance-ID map (kept for
     later instance evaluation)
   - writes an index file (CSV) with family, view, split, file paths, and the
     relevant metadata (defect types, lighting, background) so I can filter later
   - computes and saves dataset statistics: strap pixel fraction, centerline pixel
     fraction (for the BCE `pos_weight`), and per-channel image mean/std
   - is deterministic (fixed seed for splits), resumable, and parallelized
     (multiprocessing) since the cluster has many cores
   - takes paths as command-line arguments so it runs both locally and on the cluster
3. Test it on the two sampled families first, show me overlays of the outputs, and
   only then run it on everything.

**Stop and wait for my approval before running on the full dataset.**

---

## Phase 3: Training setup and parameter discussion

### Fixed decisions (do not change these)
1. **Images are 768×768. Load and train on the entire image. No cropping, no tiling,
   no sliding-window blending** at training or inference. 768 is divisible by 32,
   so UNet 3+ needs no padding (E5 is 24×24).
2. **Two run targets from the same code:**
   - my **local computer** (smaller GPU: small batch, possibly gradient accumulation)
   - **Turing HPC at WPI** (SLURM scheduler; I can request up to **100 GB of GPU
     memory**)
   Use one codebase with config files for each target (e.g. `configs/local.yaml`,
   `configs/turing.yaml`) and a SLURM job script (`jobs/train_turing.sbatch`).
   Ask me for the details you need about Turing rather than guessing: partition
   names, GPU types, module system / conda setup, storage paths (home vs. scratch),
   job time limits, and whether multi-GPU jobs are allowed.
   Make the code support single-GPU and multi-GPU (DDP via `torchrun`), checkpoint
   regularly, and resume automatically, since cluster jobs have time limits.

### Changes to `unet3plus_task.md` that follow from this
- Replace random 512 crops with full 768×768 images.
- Remove the sliding-window inference and blending.
- Keep everything else: the model, the output heads, the losses, the metrics, the
  ablation config, the unit tests, and the overfit test.
- Modify the existing code in `src/` and `configs/` (read before Phase 1) rather
  than writing a new project.

### Parameter discussion
Before launching any long training run, **discuss each training parameter with me,
one by one**, keeping in mind that the main runs will be on the cluster with more
compute. For each parameter, give the value you propose, why, the alternatives, and
how it differs between the local and Turing configs. Cover at least:
- batch size per GPU and total, gradient accumulation, number of GPUs
- whether BatchNorm statistics stay frozen (they can be trained if batch ≥ 8)
- learning rates (encoder vs. decoder), warmup, schedule, weight decay
- number of epochs / iterations, early stopping
- encoder freeze epochs at the start
- loss weights (λ for Dice, focal, clDice, BCE), focal γ and α, clDice iterations,
  BCE `pos_weight` (from the Phase 2 statistics)
- deep supervision on or off
- augmentation choices (flips, rotation, color jitter, random grayscale, blur/noise),
  and which must be avoided (e.g. anything that breaks the label geometry)
- input normalization (ImageNet stats vs. dataset stats)
- mixed precision type (fp16 vs. bf16, depending on the GPU)
- data loading (num_workers, pin_memory, caching the dataset in RAM, local scratch)
- validation frequency, checkpoint frequency, logging
- time and memory estimates for one epoch, locally and on Turing

Give me a run-time and memory estimate for the 768×768 setup before I submit jobs.

---

## General rules
- Ask before any change or decision that I haven't already approved.
- Never modify `dataset_main/`.
- Keep notes and reports short and relevant; don't repeat verbose JSON content.
- Don't add features I didn't ask for.
