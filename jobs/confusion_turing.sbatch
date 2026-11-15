#!/bin/bash
# Pixel class x class confusion matrix of a checkpoint on one split. Submit from the repo root, e.g.:
#   CONFIG=configs/v1_finetune.yaml CKPT=runs/v1_finetune/best.pt SPLIT=test DATA=Data_v1 sbatch jobs/confusion_turing.sbatch
#SBATCH -J unet3p_confusion
#SBATCH -p short
#SBATCH --account=nitin
#SBATCH --gres=gpu:L40S:1
#SBATCH -c 10
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o %x_%j.out

set -euo pipefail
REPO=/home/vchezhian/scratch/Unet
CONFIG=${CONFIG:-configs/v1_finetune.yaml}
CKPT=${CKPT:-runs/v1_finetune/best.pt}
SPLIT=${SPLIT:-test}
DATA=${DATA:-Data_v1}
cd "$REPO"

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate "$REPO/env"
export HF_HOME=/home/vchezhian/scratch/hf_cache HF_HUB_OFFLINE=1

python -m src.confusion --config "$CONFIG" --checkpoint "$CKPT" --split "$SPLIT" --data-root "$DATA"
