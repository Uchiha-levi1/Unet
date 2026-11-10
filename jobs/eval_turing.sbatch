#!/bin/bash
# Evaluate a checkpoint on one split. Submit from the repo root, e.g.:
#   CKPT=runs/turing/best.pt SPLIT=test DATA=Data OUT=runs/turing/eval_test sbatch jobs/eval_turing.sbatch
#SBATCH -J unet3p_eval
#SBATCH -p short
#SBATCH --account=nitin
#SBATCH --gres=gpu:L40S:1
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o %x_%j.out

set -euo pipefail
REPO=/home/vchezhian/scratch/Unet
CONFIG=${CONFIG:-configs/turing.yaml}
CKPT=${CKPT:-runs/turing/best.pt}
SPLIT=${SPLIT:-test}
DATA=${DATA:-Data}
OUT=${OUT:-$(dirname "$CKPT")/eval_${SPLIT}}
cd "$REPO"

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate "$REPO/env"
export HF_HOME=/home/vchezhian/scratch/hf_cache HF_HUB_OFFLINE=1

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m src.infer --config "$CONFIG" --checkpoint "$CKPT" --split "$SPLIT" --data-root "$DATA" --out "$OUT"
