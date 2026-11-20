#!/bin/bash
# Inference of r3 and fromscratch on the real photos in runs/a22_real/input_768 (no GPU-heavy work, ~18 images).
#SBATCH -J pred_real_v3
#SBATCH -p quick
#SBATCH --account=nitin
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o %x_%j.out
set -euo pipefail
cd /home/vchezhian/scratch/Unet
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate "$PWD/env"
export HF_HOME=/home/vchezhian/scratch/hf_cache HF_HUB_OFFLINE=1
IN=runs/a22_real/input_768
python -m src.predict_images --config runs/v3_finetune/config.yaml --checkpoint runs/v3_finetune/best.pt --images $IN --out runs/a22_real/pred_r3
python -m src.predict_images --config runs/fromscratch_a22/config.yaml --checkpoint runs/fromscratch_a22/best.pt --images $IN --out runs/a22_real/pred_fromscratch
